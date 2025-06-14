"""
WebSocket Manager for real-time updates

Manages WebSocket connections and broadcasts processing events to all connected clients.
"""

from typing import Set, Dict, Any, List
from fastapi import WebSocket
import json
import asyncio
import logging
from datetime import datetime
from middleware.transform import backend_to_frontend

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send initial connection confirmation
        event = self._create_event("connection", {
            "status": "connected"
        })
        await self.send_to_client(websocket, event)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_client(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            if websocket not in self.active_connections:
                logger.warning("Attempted to send to disconnected client")
                return
                
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending to client: {type(e).__name__}: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
            
        # Create tasks for all sends
        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self.send_to_client(connection, data))
        
        # Execute all sends concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _create_event(self, event_type: str, data: Any) -> Dict[str, Any]:
        """Create a standardized event with camelCase data"""
        return {
            "type": event_type,  # Keep snake_case for event types
            "data": backend_to_frontend(data) if isinstance(data, dict) else data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def disconnect_all(self):
        """Disconnect all clients (for shutdown)"""
        for connection in self.active_connections.copy():
            try:
                await connection.close()
            except Exception:
                pass
        self.active_connections.clear()
    
    # Processing event methods
    async def notify_processing_started(self, photo_id: str, recipe_id: str = None, filename: str = None):
        """Notify clients that processing has started for a photo"""
        event = self._create_event("processing_started", {
            "photo_id": photo_id,
            "filename": filename or f"photo_{photo_id}",
            "recipe_id": recipe_id
        })
        await self.broadcast(event)
    
    async def notify_processing_completed(self, photo_id: str, success: bool = True, 
                                        processed_path: str = None, filename: str = None):
        """Notify clients that processing has completed"""
        event = self._create_event("processing_completed", {
            "photo_id": photo_id,
            "filename": filename or f"photo_{photo_id}",
            "success": success,
            "processed_path": processed_path
        })
        await self.broadcast(event)
    
    async def notify_processing_failed(self, photo_id: str, error: str, filename: str = None):
        """Notify clients that processing has failed"""
        event = self._create_event("processing_failed", {
            "photo_id": photo_id,
            "filename": filename or f"photo_{photo_id}",
            "error": error
        })
        await self.broadcast(event)
    
    async def notify_queue_updated(self, queue_stats: Dict[str, int]):
        """Notify clients of queue status changes"""
        event = self._create_event("queue_updated", {
            "pending": queue_stats.get("pending", 0),
            "processing": queue_stats.get("processing", 0),
            "completed": queue_stats.get("completed", 0)
        })
        await self.broadcast(event)
    
    async def notify_stats_updated(self, stats: Dict[str, Any]):
        """Notify clients of statistics updates"""
        event = self._create_event("stats_updated", stats)
        await self.broadcast(event)
    
    async def notify_photo_uploaded(self, photo_id: str, filename: str):
        """Notify clients that a photo has been uploaded"""
        event = self._create_event("photo_uploaded", {
            "photo_id": photo_id,
            "filename": filename
        })
        await self.broadcast(event)
    
    async def notify_recipe_updated(self, recipe: Dict[str, Any], action: str):
        """Notify clients that a recipe has been updated"""
        event = self._create_event("recipe_updated", {
            "recipe": recipe,
            "action": action
        })
        await self.broadcast(event)
    
    # AI Processing specific events
    async def notify_culling_started(self, photo_ids: List[str]):
        """Notify clients that culling has started"""
        event = self._create_event("culling_started", {
            "photo_ids": photo_ids,
            "count": len(photo_ids)
        })
        await self.broadcast(event)
    
    async def notify_culling_completed(self, kept_photos: List[str], culled_photos: List[str]):
        """Notify clients that culling has completed"""
        event = self._create_event("culling_completed", {
            "kept_photos": kept_photos,
            "culled_photos": culled_photos,
            "kept_count": len(kept_photos),
            "culled_count": len(culled_photos)
        })
        await self.broadcast(event)
    
    async def notify_burst_grouping_completed(self, groups: List[Dict[str, Any]]):
        """Notify clients that burst grouping has completed"""
        event = self._create_event("burst_grouping_completed", {
            "groups": groups,
            "group_count": len(groups)
        })
        await self.broadcast(event)
    
    async def notify_scene_analysis_started(self, photo_id: str):
        """Notify clients that scene analysis has started"""
        event = self._create_event("scene_analysis_started", {
            "photo_id": photo_id
        })
        await self.broadcast(event)
    
    async def notify_scene_analysis_completed(self, photo_id: str, analysis: Dict[str, Any]):
        """Notify clients that scene analysis has completed"""
        event = self._create_event("scene_analysis_completed", {
            "photo_id": photo_id,
            "analysis": analysis
        })
        await self.broadcast(event)
    
    async def notify_raw_processing_started(self, photo_id: str, parameters: Dict[str, Any]):
        """Notify clients that RAW processing has started"""
        event = self._create_event("raw_processing_started", {
            "photo_id": photo_id,
            "parameters": parameters
        })
        await self.broadcast(event)
    
    async def notify_upload_progress(self, session_id: str, progress: Dict[str, Any]):
        """Notify clients of upload progress"""
        event = self._create_event("upload_progress", {
            "session_id": session_id,
            "progress": progress
        })
        await self.broadcast(event)
    
    async def notify_batch_completed(self, session_id: str, summary: Dict[str, Any]):
        """Notify clients that a batch upload/processing has completed"""
        event = self._create_event("batch_completed", {
            "session_id": session_id,
            "summary": summary
        })
        await self.broadcast(event)
    
    async def notify_processing_stage(self, photo_id: str, stage: str, progress: float = None):
        """Notify clients of processing stage changes"""
        data = {
            "photo_id": photo_id,
            "stage": stage
        }
        
        if progress is not None:
            data["progress"] = progress
        
        event = self._create_event("processing_stage", data)
        await self.broadcast(event)
    
    # System status events
    async def notify_system_status(self, status: str, message: str):
        """Notify clients of system status changes"""
        event = self._create_event("system_status", {
            "status": status,
            "message": message
        })
        await self.broadcast(event)
    
    async def notify_error(self, error_type: str, message: str, photo_id: str = None):
        """Notify clients of errors"""
        data = {
            "error_type": error_type,
            "message": message
        }
        
        if photo_id:
            data["photo_id"] = photo_id
        
        event = self._create_event("error", data)
        await self.broadcast(event)
    
    # AI Analysis specific events
    async def notify_nima_analysis_started(self, photo_id: str):
        """Notify clients when NIMA analysis starts"""
        event = self._create_event("nima_analysis_started", {
            "photo_id": photo_id,
            "message": "Starting NIMA aesthetic analysis..."
        })
        await self.broadcast(event)
    
    async def notify_nima_analysis_completed(self, photo_id: str, aesthetic_score: float, quality_level: str, confidence: float):
        """Notify clients when NIMA analysis completes"""
        event = self._create_event("nima_analysis_completed", {
            "photo_id": photo_id,
            "aesthetic_score": aesthetic_score,
            "quality_level": quality_level,
            "confidence": confidence,
            "message": f"NIMA analysis completed: {quality_level} quality (score: {aesthetic_score:.2f})"
        })
        await self.broadcast(event)
    
    async def notify_photo_status_changed(self, photo_id: str, status: str, message: str = None):
        """Notify clients when photo status changes"""
        event = self._create_event("photo_status_changed", {
            "photo_id": photo_id,
            "status": status,
            "message": message
        })
        await self.broadcast(event)
    
    async def notify_ai_analysis_progress(self, photo_id: str, analysis_type: str, progress: float, stage: str = None):
        """Notify clients of AI analysis progress"""
        data = {
            "photo_id": photo_id,
            "analysis_type": analysis_type,
            "progress": progress
        }
        
        if stage:
            data["stage"] = stage
        
        event = self._create_event("ai_analysis_progress", data)
        await self.broadcast(event)