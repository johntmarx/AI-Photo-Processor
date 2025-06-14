"""
SQLite-based Photo Service

Enhanced photo service using SQLite database for scalable storage.
Replaces JSON-based storage with proper relational database.
"""

import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import json
import uuid
import logging
import traceback
from fastapi import UploadFile
import aiofiles
import sys

# Add parent directory to path to import AI components
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.photo import Photo, PhotoDetail, PhotoList, PhotoComparison
from .sqlite_database import SQLiteDatabase
from recipe_storage import RecipeStorage
from hash_tracker import HashTracker

# Configure logger
logger = logging.getLogger(__name__)

class SQLitePhotoService:
    """Enhanced photo service with SQLite database backend"""
    
    def __init__(self):
        logger.info("Initializing SQLitePhotoService")
        
        # Paths
        self.data_path = Path("/app/data")
        self.inbox_path = self.data_path / "inbox"
        self.originals_path = self.data_path / "originals"
        self.processed_path = self.data_path / "processed"
        self.thumbnails_path = self.data_path / "thumbnails"
        self.web_path = self.data_path / "web"
        self.recipes_path = self.data_path / "recipes"
        
        # Ensure directories exist
        for path in [self.inbox_path, self.originals_path, self.processed_path,
                     self.thumbnails_path, self.web_path, self.recipes_path, self.data_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        
        # Initialize services
        self.db = SQLiteDatabase(self.data_path)
        self.recipe_storage = RecipeStorage(str(self.recipes_path))
        self.hash_tracker = HashTracker()
        
        logger.info("SQLitePhotoService initialized successfully")
        
        self.processing_active = True
    
    async def list_photos(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc",
        search: Optional[str] = None,
        min_aesthetic_score: Optional[float] = None,
        max_aesthetic_score: Optional[float] = None,
        min_technical_score: Optional[float] = None,
        max_technical_score: Optional[float] = None
    ) -> PhotoList:
        """List photos with pagination"""
        logger.info(f"Listing photos - page: {page}, size: {page_size}, status: {status}, sort: {sort_by} {order}")
        
        try:
            # Build filters
            filters = {}
            if status:
                filters['status'] = status
            if search:
                filters['search'] = search
            if min_aesthetic_score is not None:
                filters['min_aesthetic_score'] = min_aesthetic_score
            if max_aesthetic_score is not None:
                filters['max_aesthetic_score'] = max_aesthetic_score
            if min_technical_score is not None:
                filters['min_technical_score'] = min_technical_score
            if max_technical_score is not None:
                filters['max_technical_score'] = max_technical_score
                
            photos_data, total = await self.db.list_photos(
                filters=filters,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                order=order
            )
            
            logger.debug(f"Found {total} total photos, returning {len(photos_data)} for page {page}")
            
            # Convert to Photo models
            photo_models = []
            for photo_data in photos_data:
                try:
                    # Ensure datetime fields are properly handled
                    created_at = photo_data.get('created_at')
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    elif created_at is None:
                        created_at = datetime.now()
                    
                    processed_at = photo_data.get('processed_at')
                    if isinstance(processed_at, str):
                        processed_at = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                    
                    # Ensure file_size is valid
                    file_size = photo_data.get('file_size', 0)
                    if file_size is None or not isinstance(file_size, (int, float)):
                        file_size = 0
                    
                    photo_models.append(Photo(
                        id=photo_data['id'],
                        filename=photo_data['filename'],
                        status=photo_data['status'],
                        created_at=created_at,
                        processed_at=processed_at,
                        original_path=photo_data.get('original_path'),
                        processed_path=photo_data.get('processed_path'),
                        thumbnail_path=photo_data.get('thumbnail_path'),
                        web_path=photo_data.get('web_path'),
                        file_size=int(file_size)
                    ))
                except Exception as e:
                    logger.error(f"Error converting photo {photo_data.get('id')} to model: {e}")
                    logger.error(f"Photo data: {photo_data}")
                    raise
            
            result = PhotoList(
                photos=photo_models,
                total=total,
                page=page,
                page_size=page_size,
                has_next=(page * page_size) < total,
                has_prev=page > 1
            )
            
            logger.info(f"Successfully listed {len(photo_models)} photos")
            return result
            
        except Exception as e:
            logger.error(f"Failed to list photos: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def get_photo(self, photo_id: str) -> Optional[PhotoDetail]:
        """Get photo details by ID"""
        photo_data = await self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        # Get file sizes if not already calculated
        original_size = photo_data.get('file_size', 0)
        processed_size = 0
        
        if photo_data.get('processed_path'):
            processed_path = Path(photo_data['processed_path'])
            if processed_path.exists():
                processed_size = processed_path.stat().st_size
        
        # Handle datetime fields
        created_at = photo_data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        processed_at = photo_data.get('processed_at')
        if isinstance(processed_at, str):
            processed_at = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
        
        return PhotoDetail(
            id=photo_data['id'],
            filename=photo_data['filename'],
            status=photo_data['status'],
            created_at=created_at,
            processed_at=processed_at,
            original_path=photo_data.get('original_path'),
            processed_path=photo_data.get('processed_path'),
            thumbnail_path=photo_data.get('thumbnail_path'),
            web_path=photo_data.get('web_path'),
            file_size=original_size,
            recipe_id=photo_data.get('recipe_id'),
            ai_analysis=photo_data.get('ai_analysis', {}),
            processing_time=photo_data.get('processing_time', 0),
            error_message=photo_data.get('error_message')
        )
    
    async def get_comparison(self, photo_id: str) -> Optional[PhotoComparison]:
        """Get comparison data for a photo"""
        photo_data = await self.db.get_photo(photo_id)
        if not photo_data or photo_data['status'] != 'completed':
            return None
        
        return PhotoComparison(
            photo_id=photo_id,
            original_url=f"/api/photos/{photo_id}/original",
            processed_url=f"/api/photos/{photo_id}/processed",
            metadata={
                'original_size': photo_data.get('file_size', 0),
                'processed_size': photo_data.get('processed_size', 0),
                'compression_ratio': photo_data.get('compression_ratio', 1.0)
            }
        )
    
    async def save_upload(
        self,
        file: UploadFile,
        auto_process: bool = True,
        recipe_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save uploaded file and queue for processing"""
        logger.info(f"Saving upload: {file.filename}, auto_process: {auto_process}, recipe_id: {recipe_id}")
        
        try:
            # Read file content
            content = await file.read()
            file_hash = hashlib.sha256(content).hexdigest()
            logger.debug(f"File hash: {file_hash}, size: {len(content)} bytes")
            
            # Check if already processed
            temp_file = self.inbox_path / f"temp_{file_hash}.tmp"
            try:
                async with aiofiles.open(temp_file, 'wb') as f:
                    await f.write(content)
                
                if self.hash_tracker.is_already_processed(str(temp_file)):
                    logger.info(f"Duplicate file detected: {file.filename} (hash: {file_hash})")
                    return {
                        "photo_id": file_hash,
                        "filename": file.filename,
                        "status": "duplicate",
                        "message": "Photo already processed"
                    }
            finally:
                if temp_file.exists():
                    temp_file.unlink()
            
            # Generate unique ID
            photo_id = str(uuid.uuid4())
            logger.info(f"Generated photo ID: {photo_id}")
            
            # Save to inbox
            file_path = self.inbox_path / f"{photo_id}_{file.filename}"
            logger.debug(f"Saving file to: {file_path}")
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            logger.info(f"File saved successfully: {file_path}")
            
            # Add to database with AI analysis placeholder
            photo_data = {
                'id': photo_id,
                'filename': file.filename,
                'status': 'processing' if auto_process else 'completed',
                'created_at': datetime.now(),
                'original_path': str(file_path),
                'file_hash': file_hash,
                'file_size': len(content),
                'metadata': {
                    'size': len(content),
                    'content_type': file.content_type
                },
                'ai_analysis': {
                    'status': 'pending' if auto_process else 'not_available',
                    'queued_at': datetime.now().isoformat() if auto_process else None
                }
            }
            
            if recipe_id:
                photo_data['recipe_id'] = recipe_id
            
            await self.db.add_photo(photo_data)
            
            # Queue for NIMA analysis if auto_process is enabled
            if auto_process:
                # Import here to avoid circular imports
                from tasks.ai_tasks import analyze_photo_nima
                
                # Queue NIMA analysis as background task
                task = analyze_photo_nima.delay(
                    photo_id=photo_id,
                    photo_path=str(file_path),
                    include_technical=True
                )
                
                logger.info(f"Queued NIMA analysis for photo {photo_id}, task_id: {task.id}")
                
                # Update photo with task ID for tracking
                await self.db.update_photo_metadata(photo_id, {
                    'nima_task_id': task.id,
                    'processing_started_at': datetime.now().isoformat()
                })
            
            result = {
                "photo_id": photo_id,
                "filename": file.filename,
                "status": photo_data['status'],
                "message": "Photo uploaded successfully"
            }
            
            logger.info(f"Upload complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            logger.error(f"File: {file.filename}, Content-Type: {file.content_type}")
            logger.error(traceback.format_exc())
            raise
    
    async def delete_photo(
        self,
        photo_id: str,
        delete_original: bool = False
    ) -> bool:
        """Delete a photo"""
        photo_data = await self.db.get_photo(photo_id)
        if not photo_data:
            return False
        
        # Delete files
        files_to_delete = []
        
        if photo_data.get('processed_path'):
            files_to_delete.append(Path(photo_data['processed_path']))
        
        if photo_data.get('thumbnail_path'):
            files_to_delete.append(Path(photo_data['thumbnail_path']))
        
        if photo_data.get('web_path'):
            files_to_delete.append(Path(photo_data['web_path']))
        
        if delete_original and photo_data.get('original_path'):
            files_to_delete.append(Path(photo_data['original_path']))
        
        # Delete files
        for file_path in files_to_delete:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path}")
        
        # Remove from database (this also removes AI analysis via CASCADE)
        return await self.db.delete_photo(photo_id)
    
    async def reprocess_photo(
        self,
        photo_id: str,
        recipe_id: Optional[str] = None,
        priority: str = "normal"
    ) -> Optional[Dict[str, Any]]:
        """Queue a photo for reprocessing"""
        photo_data = await self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        # Update status
        await self.db.update_photo_status(photo_id, 'processing', 'Queued for reprocessing')
        
        # Re-queue NIMA analysis
        from tasks.ai_tasks import analyze_photo_nima
        
        original_path = photo_data.get('original_path', '')
        if not Path(original_path).exists():
            logger.error(f"Original file not found for reprocessing: {original_path}")
            await self.db.update_photo_status(photo_id, 'failed', 'Original file not found')
            return None
        
        task = analyze_photo_nima.delay(
            photo_id=photo_id,
            photo_path=original_path,
            include_technical=True
        )
        
        # Update AI analysis record
        await self.db.update_photo_ai_analysis(photo_id, {
            'status': 'pending',
            'queued_at': datetime.now().isoformat(),
            'task_id': task.id,
            'error': None  # Clear previous errors
        })
        
        logger.info(f"Photo {photo_id} queued for reprocessing, task_id: {task.id}")
        
        return {
            "photo_id": photo_id,
            "status": "processing",
            "message": "Photo queued for reprocessing",
            "task_id": task.id
        }
    
    async def get_ai_analysis(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """Get AI analysis results"""
        photo_data = await self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        ai_analysis = photo_data.get('ai_analysis', {})
        
        # Return analysis or placeholder
        if ai_analysis.get('status') == 'not_available':
            return {
                'status': 'not_available',
                'message': 'AI analysis not available for this photo'
            }
        
        return ai_analysis
    
    # Additional methods for SQLite-specific features
    async def update_photo_status(self, photo_id: str, status: str, processing_message: str = None):
        """Update photo status"""
        await self.db.update_photo_status(photo_id, status, processing_message)
    
    async def update_photo_ai_analysis(self, photo_id: str, ai_analysis: Dict[str, Any]):
        """Update AI analysis for a photo"""
        await self.db.update_photo_ai_analysis(photo_id, ai_analysis)
    
    async def update_photo_metadata(self, photo_id: str, metadata: Dict[str, Any]):
        """Update photo metadata"""
        await self.db.update_photo_metadata(photo_id, metadata)
    
    async def get_photos_by_ai_status(self, status: str) -> List[Dict[str, Any]]:
        """Get photos by AI analysis status"""
        return await self.db.get_photos_by_ai_status(status)
    
    async def get_photos_for_reanalysis(self, force: bool = False) -> List[Dict[str, Any]]:
        """Get photos that need reanalysis"""
        if force:
            # Get all photos for forced reanalysis
            photos_data, _ = await self.db.list_photos(page_size=1000)
            return photos_data
        else:
            # Get photos with failed or pending AI analysis
            failed_photos = await self.db.get_photos_by_ai_status('failed')
            pending_photos = await self.db.get_photos_by_ai_status('pending')
            return failed_photos + pending_photos
    
    # Upload session methods (for batch uploads)
    async def create_upload_session(self, session_data: Dict[str, Any]) -> str:
        """Create upload session"""
        return await self.db.create_upload_session(session_data)
    
    async def get_upload_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get upload session"""
        return await self.db.get_upload_session(session_id)
    
    async def update_upload_session(self, session_id: str, updates: Dict[str, Any]):
        """Update upload session"""
        await self.db.update_upload_session(session_id, updates)
    
    async def add_session_file(self, session_id: str, file_data: Dict[str, Any]):
        """Add file to upload session"""
        await self.db.add_session_file(session_id, file_data)
    
    # Statistics and maintenance
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return await self.db.get_statistics()
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data"""
        return await self.db.cleanup_old_data(days_old)
    
    # Legacy methods for compatibility
    def pause_processing(self):
        """Pause processing queue"""
        self.processing_active = False
    
    def resume_processing(self):
        """Resume processing queue"""
        self.processing_active = True
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        photos_data, _ = await self.db.list_photos(page_size=1000)
        
        status_counts = {}
        for photo in photos_data:
            status = photo.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'active': self.processing_active,
            'total_photos': len(photos_data),
            'status_counts': status_counts,
            'processing': status_counts.get('processing', 0),
            'pending': status_counts.get('pending', 0),
            'completed': status_counts.get('completed', 0),
            'failed': status_counts.get('failed', 0)
        }


# Create singleton instance
sqlite_photo_service = SQLitePhotoService()