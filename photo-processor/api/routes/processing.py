"""
Processing queue and control routes
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Dict, Any

from models.processing import (
    QueueStatus, ProcessingStatus, ProcessingApproval,
    BatchOperation, ProcessingSettings, QueueItem
)
from services.processing_service_v2 import processing_service
from services.websocket_manager import WebSocketManager
from middleware.transform import transform_queue_status, backend_to_frontend, frontend_to_backend

router = APIRouter()
# processing_service is imported as singleton from processing_service_v2

# WebSocket manager will be injected
ws_manager = None

def set_websocket_manager(manager):
    """Set the WebSocket manager instance"""
    global ws_manager
    ws_manager = manager

@router.get("/queue")
async def get_queue_status():
    """Get current processing queue status"""
    try:
        status = await processing_service.get_queue_status()
        # Transform to frontend format
        return transform_queue_status(status.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_processing_status():
    """Get current processing status and statistics"""
    try:
        status = await processing_service.get_processing_status()
        # Transform to frontend format
        return backend_to_frontend(status.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/pause")
async def pause_processing():
    """Pause all processing"""
    success = await processing_service.pause_processing()
    if success:
        if ws_manager:
            await ws_manager.broadcast({
                "type": "processing_paused",
                "message": "Processing has been paused"
            })
        return {"status": "paused", "message": "Processing paused successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to pause processing")

@router.put("/resume")
async def resume_processing():
    """Resume processing"""
    success = await processing_service.resume_processing()
    if success:
        if ws_manager:
            await ws_manager.broadcast({
                "type": "processing_resumed",
                "message": "Processing has been resumed"
            })
        return {"status": "resumed", "message": "Processing resumed successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to resume processing")

@router.post("/approve/{photo_id}")
async def approve_processing(
    photo_id: str,
    adjustments: Dict[str, Any] = Body(default={})
):
    """Approve processing for a photo with optional adjustments"""
    try:
        result = await processing_service.approve_processing(
            photo_id=photo_id,
            adjustments=adjustments
        )
        
        # Notify clients
        if ws_manager:
            await ws_manager.notify_processing_started(photo_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reject/{photo_id}")
async def reject_processing(
    photo_id: str,
    reason: Optional[str] = Query(None, description="Rejection reason")
):
    """Reject processing for a photo"""
    try:
        result = await processing_service.reject_processing(
            photo_id=photo_id,
            reason=reason
        )
        
        # Notify clients
        if ws_manager:
            await ws_manager.broadcast({
                "type": "processing_rejected",
                "photoId": photo_id,
                "reason": reason
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_process(operation: BatchOperation):
    """Queue multiple photos for batch processing"""
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    
    try:
        # Enhanced logging for batch operations
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING REQUEST")
        logger.info("=" * 60)
        logger.info(f"Photo IDs: {operation.photo_ids}")
        logger.info(f"Recipe ID: {operation.recipe_id}")
        logger.info(f"Priority: {operation.priority}")
        logger.info(f"Skip AI: {operation.skip_ai}")
        logger.info(f"Total photos: {len(operation.photo_ids)}")
        
        # Validate recipe exists if specified
        if operation.recipe_id:
            from services.recipe_service_v2 import recipe_service
            try:
                recipe_exists = await recipe_service.get_recipe(operation.recipe_id)
                if recipe_exists:
                    logger.info(f"Recipe '{recipe_exists.get('name', 'Unknown')}' found and validated")
                else:
                    logger.warning(f"Recipe {operation.recipe_id} not found!")
            except Exception as e:
                logger.error(f"Failed to validate recipe: {e}")
        
        result = await processing_service.batch_process(operation)
        
        logger.info("-" * 60)
        logger.info("BATCH PROCESSING RESULT")
        logger.info("-" * 60)
        logger.info(f"Queued: {result.get('queued', 0)} photos")
        logger.info(f"Skipped: {result.get('skipped', 0)} photos")
        if result.get('errors'):
            logger.warning(f"Errors encountered: {result['errors']}")
        if result.get('skipped_photos'):
            logger.info(f"Skipped photo IDs: {result['skipped_photos']}")
        logger.info("=" * 60)
        
        # Notify clients of queue update
        if ws_manager:
            queue_stats = await processing_service.get_queue_stats()
            await ws_manager.notify_queue_updated(queue_stats)
        
        return backend_to_frontend(result)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("BATCH PROCESSING FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Operation: {operation.dict()}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        logger.error("=" * 60)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/reorder")
async def reorder_queue(
    photo_id: str = Query(..., description="Photo to move"),
    new_position: int = Query(..., ge=1, description="New position in queue")
):
    """Reorder items in the processing queue"""
    try:
        success = await processing_service.reorder_queue(photo_id, new_position)
        if not success:
            raise HTTPException(status_code=404, detail="Photo not found in queue")
        
        # Notify clients
        if ws_manager:
            queue_stats = await processing_service.get_queue_stats()
            await ws_manager.notify_queue_updated(queue_stats)
        
        return {"message": "Queue reordered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/queue/{photo_id}")
async def remove_from_queue(photo_id: str):
    """Remove a photo from the processing queue"""
    try:
        success = await processing_service.remove_from_queue(photo_id)
        if not success:
            raise HTTPException(status_code=404, detail="Photo not found in queue")
        
        # Notify clients
        if ws_manager:
            queue_stats = await processing_service.get_queue_stats()
            await ws_manager.notify_queue_updated(queue_stats)
        
        return {"message": "Photo removed from queue"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/settings", response_model=ProcessingSettings)
async def get_processing_settings():
    """Get current processing settings"""
    settings = await processing_service.get_settings()
    return settings

@router.put("/settings")
async def update_processing_settings(settings: ProcessingSettings):
    """Update processing settings"""
    try:
        updated = await processing_service.update_settings(settings)
        
        # Notify clients
        if ws_manager:
            await ws_manager.broadcast({
                "type": "settings_updated",
                "settings": updated.dict()
            })
        
        return updated
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))