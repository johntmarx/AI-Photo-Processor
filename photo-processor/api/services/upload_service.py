"""
Enhanced upload service for handling large batch uploads
Supports chunked uploads, progress tracking, and concurrent processing.
"""

import asyncio
import aiofiles
import hashlib
from typing import Dict, Any, List, Optional, AsyncGenerator
from pathlib import Path
from datetime import datetime
import uuid
import json
from fastapi import UploadFile, HTTPException
import sys
import logging

# Add parent directory to path to import AI components
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.photo_service_sqlite import sqlite_photo_service as photo_service
from services.processing_service_v2 import processing_service
from hash_tracker import HashTracker

logger = logging.getLogger(__name__)


class UploadSession:
    """Manages an upload session for batch uploads"""
    
    def __init__(self, session_id: str, expected_files: int):
        self.session_id = session_id
        self.expected_files = expected_files
        self.uploaded_files = []
        self.failed_files = []
        self.created_at = datetime.now()
        self.status = "uploading"
        self.total_size = 0
        self.uploaded_size = 0
    
    def add_file(self, file_info: Dict[str, Any]):
        """Add a successfully uploaded file"""
        self.uploaded_files.append(file_info)
        self.uploaded_size += file_info.get('size', 0)
        
        if len(self.uploaded_files) >= self.expected_files:
            self.status = "completed"
    
    def add_failure(self, filename: str, error: str):
        """Add a failed upload"""
        self.failed_files.append({
            'filename': filename,
            'error': error,
            'timestamp': datetime.now()
        })
    
    def get_progress(self) -> Dict[str, Any]:
        """Get upload progress"""
        # Convert datetime objects to ISO format strings for JSON serialization
        uploaded_files_serialized = []
        for file_info in self.uploaded_files:
            file_copy = file_info.copy()
            if 'uploaded_at' in file_copy and hasattr(file_copy['uploaded_at'], 'isoformat'):
                file_copy['uploaded_at'] = file_copy['uploaded_at'].isoformat()
            uploaded_files_serialized.append(file_copy)
        
        failed_files_serialized = []
        for file_info in self.failed_files:
            file_copy = file_info.copy()
            if 'timestamp' in file_copy and hasattr(file_copy['timestamp'], 'isoformat'):
                file_copy['timestamp'] = file_copy['timestamp'].isoformat()
            failed_files_serialized.append(file_copy)
        
        return {
            'session_id': self.session_id,
            'status': self.status,
            'uploaded_count': len(self.uploaded_files),
            'failed_count': len(self.failed_files),
            'expected_count': self.expected_files,
            'total_size': self.total_size,
            'uploaded_size': self.uploaded_size,
            'progress_percent': (len(self.uploaded_files) / self.expected_files * 100) if self.expected_files > 0 else 0,
            'created_at': self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else self.created_at,
            'uploaded_files': uploaded_files_serialized,
            'failed_files': failed_files_serialized
        }


class EnhancedUploadService:
    """Enhanced upload service for large batch processing"""
    
    def __init__(self):
        self.upload_sessions: Dict[str, UploadSession] = {}
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks
        self.max_concurrent_uploads = 5
        self.hash_tracker = HashTracker()
        self.ws_manager = None  # Will be injected
        
        # Paths
        self.temp_path = Path("/app/data/temp")
        self.inbox_path = Path("/app/data/inbox")
        
        # Ensure directories exist
        for path in [self.temp_path, self.inbox_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Allowed file types for batch processing
        self.allowed_extensions = {
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp',
            '.nef', '.cr2', '.cr3', '.arw', '.dng', '.orf', 
            '.rw2', '.raf', '.3fr', '.fff', '.dcr', '.kdc',
            '.srf', '.sr2', '.erf', '.mef', '.mos', '.nrw',
            '.pef', '.raw', '.rwl', '.rw2', '.x3f'
        }
    
    def set_websocket_manager(self, ws_manager):
        """Set the WebSocket manager for broadcasting events"""
        self.ws_manager = ws_manager
    
    async def create_upload_session(
        self,
        expected_files: int,
        total_size: Optional[int] = None,
        recipe_id: Optional[str] = None,
        auto_process: bool = True
    ) -> Dict[str, Any]:
        """Create a new upload session for batch uploads"""
        session_id = str(uuid.uuid4())
        session = UploadSession(session_id, expected_files)
        
        if total_size:
            session.total_size = total_size
        
        self.upload_sessions[session_id] = session
        
        return {
            'session_id': session_id,
            'upload_url': f'/api/upload/session/{session_id}/file',
            'expected_files': expected_files,
            'max_file_size': 500 * 1024 * 1024,  # 500MB per file
            'allowed_extensions': list(self.allowed_extensions),
            'auto_process': auto_process,
            'recipe_id': recipe_id
        }
    
    async def upload_file_to_session(
        self,
        session_id: str,
        file: UploadFile,
        auto_process: bool = True,
        recipe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a file to an existing session"""
        session = self.upload_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if session.status == "completed":
            raise HTTPException(status_code=400, detail="Upload session already completed")
        
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            error_msg = f"File type {file_ext} not supported"
            session.add_failure(file.filename, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        try:
            # Stream file to temporary location
            temp_file_path = self.temp_path / f"{session_id}_{uuid.uuid4()}_{file.filename}"
            file_size = 0
            file_hash = hashlib.sha256()
            
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                while chunk := await file.read(self.chunk_size):
                    await temp_file.write(chunk)
                    file_size += len(chunk)
                    file_hash.update(chunk)
            
            # Calculate final hash
            final_hash = file_hash.hexdigest()
            
            # Check for duplicates using the temporary file path
            if self.hash_tracker.is_already_processed(str(temp_file_path)):
                # Remove temp file
                temp_file_path.unlink()
                
                session.add_failure(file.filename, "Duplicate file already processed")
                return {
                    'filename': file.filename,
                    'status': 'duplicate',
                    'message': 'File already processed',
                    'session_progress': session.get_progress()
                }
            
            # Move to inbox with unique name
            photo_id = str(uuid.uuid4())
            final_path = self.inbox_path / f"{photo_id}_{file.filename}"
            temp_file_path.rename(final_path)
            
            # Add to photo service with initial status
            photo_data = {
                'id': photo_id,
                'filename': file.filename,
                'status': 'processing' if auto_process else 'completed',
                'created_at': datetime.now(),
                'original_path': str(final_path),
                'file_hash': final_hash,
                'file_size': file_size,
                'session_id': session_id,
                'metadata': {
                    'size': file_size,
                    'content_type': file.content_type
                },
                'ai_analysis': {
                    'status': 'pending' if auto_process else 'not_available',
                    'queued_at': datetime.now().isoformat() if auto_process else None
                }
            }
            
            if recipe_id:
                photo_data['recipe_id'] = recipe_id
            
            await photo_service.db.add_photo(photo_data)
            
            # Queue for NIMA analysis if auto_process is enabled
            if auto_process:
                # Import here to avoid circular imports
                from tasks.ai_tasks import analyze_photo_nima
                
                # Queue NIMA analysis as background task
                task = analyze_photo_nima.delay(
                    photo_id=photo_id,
                    photo_path=str(final_path),
                    include_technical=True  # Include both aesthetic and technical analysis
                )
                
                logger.info(f"✓ File uploaded: {photo_id}")
                
                # Update photo with task ID for tracking
                await photo_service.db.update_photo_metadata(photo_id, {
                    'nima_task_id': task.id,
                    'processing_started_at': datetime.now().isoformat()
                })
            
            # Update session
            file_info = {
                'photo_id': photo_id,
                'filename': file.filename,
                'size': file_size,
                'hash': final_hash,
                'uploaded_at': datetime.now()
            }
            
            session.add_file(file_info)
            
            # Notify clients of upload
            if self.ws_manager:
                await self.ws_manager.notify_photo_uploaded(photo_id, file.filename)
                await self.ws_manager.notify_upload_progress(session_id, session.get_progress())
            
            return {
                'photo_id': photo_id,
                'filename': file.filename,
                'size': file_size,
                'status': photo_data['status'],
                'session_progress': session.get_progress()
            }
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file_path.exists():
                temp_file_path.unlink()
            
            session.add_failure(file.filename, str(e))
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    async def upload_multiple_files(
        self,
        files: List[UploadFile],
        auto_process: bool = True,
        recipe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload multiple files concurrently"""
        # Create session
        session_info = await self.create_upload_session(
            expected_files=len(files),
            recipe_id=recipe_id,
            auto_process=auto_process
        )
        
        session_id = session_info['session_id']
        
        # Upload files concurrently with semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_uploads)
        
        async def upload_single_file(file: UploadFile) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.upload_file_to_session(
                        session_id=session_id,
                        file=file,
                        auto_process=auto_process,
                        recipe_id=recipe_id
                    )
                except Exception as e:
                    return {
                        'filename': file.filename,
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Process all files
        upload_tasks = [upload_single_file(file) for file in files]
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Get final session state
        session = self.upload_sessions[session_id]
        
        # Notify clients of batch completion
        if self.ws_manager:
            await self.ws_manager.notify_batch_completed(session_id, session.get_progress())
        
        return {
            'session_id': session_id,
            'upload_results': results,
            'session_summary': session.get_progress(),
            'successful_uploads': len(session.uploaded_files),
            'failed_uploads': len(session.failed_files)
        }
    
    async def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get progress of an upload session"""
        session = self.upload_sessions.get(session_id)
        if not session:
            return None
        
        return session.get_progress()
    
    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """Mark session as completed and start processing if needed"""
        session = self.upload_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        session.status = "completed"
        
        # Start batch processing for all uploaded files
        if session.uploaded_files:
            photo_ids = [file_info['photo_id'] for file_info in session.uploaded_files]
            
            # Queue all photos for processing
            for file_info in session.uploaded_files:
                await processing_service.queue_photo_processing(
                    photo_id=file_info['photo_id'],
                    photo_path=Path(f"/app/data/inbox/{file_info['photo_id']}_{file_info['filename']}"),
                    recipe_id=None,  # Use individual recipe if set
                    priority='normal'
                )
        
        return {
            'session_id': session_id,
            'status': 'completed',
            'summary': session.get_progress(),
            'processing_queued': len(session.uploaded_files)
        }
    
    async def cancel_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an upload session and clean up"""
        session = self.upload_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        # Clean up uploaded files if needed
        files_removed = 0
        for file_info in session.uploaded_files:
            photo_id = file_info['photo_id']
            file_path = Path(f"/app/data/inbox/{photo_id}_{file_info['filename']}")
            
            if file_path.exists():
                file_path.unlink()
                files_removed += 1
            
            # Remove from photo service
            await photo_service.db.delete_photo(photo_id)
        
        # Remove session
        del self.upload_sessions[session_id]
        
        return {
            'session_id': session_id,
            'status': 'cancelled',
            'files_removed': files_removed
        }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old upload sessions"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.upload_sessions.items():
            age_hours = (current_time - session.created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)
        
        # Clean up old sessions
        for session_id in sessions_to_remove:
            try:
                await self.cancel_session(session_id)
            except Exception:
                # Session might already be cleaned up
                pass
        
        return {
            'cleaned_sessions': len(sessions_to_remove),
            'active_sessions': len(self.upload_sessions)
        }
    
    async def get_upload_stats(self) -> Dict[str, Any]:
        """Get overall upload statistics"""
        active_sessions = len(self.upload_sessions)
        total_files = sum(len(s.uploaded_files) for s in self.upload_sessions.values())
        total_failures = sum(len(s.failed_files) for s in self.upload_sessions.values())
        
        return {
            'active_sessions': active_sessions,
            'total_uploaded_files': total_files,
            'total_failed_files': total_failures,
            'supported_extensions': list(self.allowed_extensions),
            'max_concurrent_uploads': self.max_concurrent_uploads,
            'chunk_size_mb': self.chunk_size / (1024 * 1024)
        }


# Singleton instance
upload_service = EnhancedUploadService()