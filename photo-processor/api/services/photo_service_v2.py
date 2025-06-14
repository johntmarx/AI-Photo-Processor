"""
Enhanced Photo Service with AI Integration
Connects the API layer to the actual photo processing components.
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
# Temporarily disable AI imports until module structure is fixed
# from ai_components.orchestrator import PhotoProcessingOrchestrator, ProcessingConfig
# from ai_components.services import (
#     SceneAnalysisService,
#     CullingService,
#     RotationDetectionService
# )
from recipe_storage import RecipeStorage
from hash_tracker import HashTracker

# Configure logger
logger = logging.getLogger(__name__)

# Simple file-based database for demo
class PhotoDatabase:
    """Simple JSON-based photo database"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_file = db_path / "photos.json"
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database file exists"""
        if not self.db_file.exists():
            self.save({"photos": {}})
    
    def load(self) -> Dict[str, Any]:
        """Load database"""
        try:
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded {len(data.get('photos', {}))} photos from database")
                return data
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def save(self, data: Dict[str, Any]):
        """Save database"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                logger.debug(f"Saved {len(data.get('photos', {}))} photos to database")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def add_photo(self, photo_data: Dict[str, Any]) -> str:
        """Add photo to database"""
        photo_id = photo_data.get('id', str(uuid.uuid4()))
        logger.info(f"Adding photo {photo_id} to database: {photo_data.get('filename')}")
        
        db = self.load()
        db['photos'][photo_id] = photo_data
        self.save(db)
        
        logger.info(f"Successfully added photo {photo_id}")
        return photo_id
    
    def get_photo(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """Get photo by ID"""
        db = self.load()
        return db['photos'].get(photo_id)
    
    def update_photo(self, photo_id: str, updates: Dict[str, Any]):
        """Update photo data"""
        db = self.load()
        if photo_id in db['photos']:
            db['photos'][photo_id].update(updates)
            self.save(db)
    
    def delete_photo(self, photo_id: str) -> bool:
        """Delete photo from database"""
        db = self.load()
        if photo_id in db['photos']:
            del db['photos'][photo_id]
            self.save(db)
            return True
        return False
    
    def list_photos(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all photos with optional filters"""
        db = self.load()
        photos = list(db['photos'].values())
        
        # Apply filters
        if filters:
            if 'status' in filters:
                photos = [p for p in photos if p.get('status') == filters['status']]
        
        return photos


class EnhancedPhotoService:
    """Enhanced photo service with AI integration"""
    
    def __init__(self):
        logger.info("Initializing EnhancedPhotoService")
        
        # Paths
        self.data_path = Path("/app/data")
        self.inbox_path = self.data_path / "inbox"
        self.originals_path = self.data_path / "originals"
        self.processed_path = self.data_path / "processed"
        self.recipes_path = self.data_path / "recipes"
        
        # Ensure directories exist
        for path in [self.inbox_path, self.originals_path, 
                     self.processed_path, self.recipes_path, self.data_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
        
        # Initialize services
        self.db = PhotoDatabase(self.data_path)
        self.recipe_storage = RecipeStorage(str(self.recipes_path))
        self.hash_tracker = HashTracker()
        
        logger.info("EnhancedPhotoService initialized successfully")
        
        # Initialize AI services
        # self._init_ai_services()  # Temporarily disabled
        
        # Processing is now handled by processing_service_v2
        self.processing_active = True
    
    # def _init_ai_services(self):
    #     """Initialize AI services"""
    #     device = "cuda"  # or "cpu" based on availability
    #     
    #     self.scene_service = SceneAnalysisService(device=device)
    #     self.culling_service = CullingService(device=device)
    #     self.rotation_service = RotationDetectionService()
    #     
    #     # Create orchestrator config
    #     self.processing_config = ProcessingConfig(
    #         style_preset="natural",
    #         cull_aggressively=True,
    #         quality_threshold=5.0,
    #         enable_rotation_correction=True
    #     )
    #     
    #     # Initialize orchestrator
    #     self.orchestrator = PhotoProcessingOrchestrator(
    #         config=self.processing_config,
    #         output_dir=self.processed_path
    #     )
    
    async def list_photos(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> PhotoList:
        """List photos with pagination"""
        logger.info(f"Listing photos - page: {page}, size: {page_size}, status: {status}, sort: {sort_by} {order}")
        
        try:
            # Get all photos
            filters = {'status': status} if status else {}
            all_photos = self.db.list_photos(filters)
            logger.debug(f"Found {len(all_photos)} total photos with filters: {filters}")
            
            # Sort
            reverse = (order == "desc")
            all_photos.sort(key=lambda p: p.get(sort_by, ''), reverse=reverse)
            
            # Paginate
            start = (page - 1) * page_size
            end = start + page_size
            photos = all_photos[start:end]
            logger.debug(f"Returning {len(photos)} photos for page {page}")
            
            # Convert to Photo models
            photo_models = []
            for p in photos:
                try:
                    # Ensure datetime fields are datetime objects or convert from string
                    created_at = p.get('created_at')
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    elif created_at is None:
                        created_at = datetime.now()
                    
                    processed_at = p.get('processed_at')
                    if isinstance(processed_at, str):
                        processed_at = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                    
                    # Ensure file_size is a valid number
                    file_size = p.get('file_size', 0)
                    if file_size is None or not isinstance(file_size, (int, float)):
                        file_size = 0
                    
                    photo_models.append(Photo(
                        id=p['id'],
                        filename=p['filename'],
                        status=p['status'],
                        created_at=created_at,
                        processed_at=processed_at,
                        original_path=p.get('original_path'),
                        processed_path=p.get('processed_path'),
                        thumbnail_path=p.get('thumbnail_path'),
                        web_path=p.get('web_path'),
                        file_size=int(file_size)
                    ))
                except Exception as e:
                    logger.error(f"Error converting photo {p.get('id')} to model: {e}")
                    logger.error(f"Photo data: {p}")
                    raise
            
            result = PhotoList(
                photos=photo_models,
                total=len(all_photos),
                page=page,
                page_size=page_size,
                has_next=end < len(all_photos),
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
        photo_data = self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        # Get file sizes
        original_size = 0
        processed_size = 0
        
        if photo_data.get('original_path'):
            original_path = Path(photo_data['original_path'])
            if original_path.exists():
                original_size = original_path.stat().st_size
        
        if photo_data.get('processed_path'):
            processed_path = Path(photo_data['processed_path'])
            if processed_path.exists():
                processed_size = processed_path.stat().st_size
        
        return PhotoDetail(
            id=photo_data['id'],
            filename=photo_data['filename'],
            status=photo_data['status'],
            created_at=photo_data.get('created_at'),
            processed_at=photo_data.get('processed_at'),
            original_path=photo_data.get('original_path'),
            processed_path=photo_data.get('processed_path'),
            thumbnail_path=photo_data.get('thumbnail_path'),
            web_path=photo_data.get('web_path'),
            file_size=photo_data.get('file_size', 0),
            recipe_id=photo_data.get('recipe_used'),
            ai_analysis=photo_data.get('ai_analysis', {}),
            processing_time=photo_data.get('processing_time', 0),
            error_message=photo_data.get('error_message')
        )
    
    async def get_comparison(self, photo_id: str) -> Optional[PhotoComparison]:
        """Get comparison data for a photo"""
        photo_data = self.db.get_photo(photo_id)
        if not photo_data or photo_data['status'] != 'completed':
            return None
        
        return PhotoComparison(
            photo_id=photo_id,
            original_url=f"/api/photos/{photo_id}/original",
            processed_url=f"/api/photos/{photo_id}/processed",
            metadata={
                'original_size': photo_data.get('original_size', 0),
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
            # Create temporary file to check hash
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
                # Clean up temp file
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
            
            # Add to database
            photo_data = {
                'id': photo_id,
                'filename': file.filename,
                'status': 'pending' if auto_process else 'completed',
                'created_at': datetime.now(),
                'original_path': str(file_path),
                'file_hash': file_hash,
                'file_size': len(content),
                'metadata': {
                    'size': len(content),
                    'content_type': file.content_type
                }
            }
        
            if recipe_id:
                photo_data['recipe_id'] = recipe_id
            
            self.db.add_photo(photo_data)
            
            # Queue for processing if requested
            if auto_process:
                logger.info(f"Queuing photo {photo_id} for processing")
                # Use the processing service to queue the photo
                from .processing_service_v2 import processing_service
                await processing_service.queue_photo_processing(
                    photo_id=photo_id,
                    photo_path=file_path,
                    recipe_id=recipe_id,
                    priority='normal'
                )
                logger.info(f"Photo {photo_id} queued successfully")
            
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
        photo_data = self.db.get_photo(photo_id)
        if not photo_data:
            return False
        
        # Delete files
        if photo_data.get('processed_path'):
            processed_path = Path(photo_data['processed_path'])
            if processed_path.exists():
                processed_path.unlink()
        
        if delete_original and photo_data.get('original_path'):
            original_path = Path(photo_data['original_path'])
            if original_path.exists():
                original_path.unlink()
        
        # Remove from database
        return self.db.delete_photo(photo_id)
    
    async def reprocess_photo(
        self,
        photo_id: str,
        recipe_id: Optional[str] = None,
        priority: str = "normal"
    ) -> Optional[Dict[str, Any]]:
        """Queue a photo for reprocessing"""
        photo_data = self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        # Update status
        self.db.update_photo(photo_id, {'status': 'pending'})
        
        # Add to queue using processing service
        from .processing_service_v2 import processing_service
        original_path = Path(photo_data.get('original_path', ''))
        await processing_service.queue_photo_processing(
            photo_id=photo_id,
            photo_path=original_path,
            recipe_id=recipe_id,
            priority=priority
        )
        
        return {
            "photo_id": photo_id,
            "status": "pending",
            "message": "Photo queued for reprocessing"
        }
    
    async def get_ai_analysis(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """Get AI analysis results"""
        photo_data = self.db.get_photo(photo_id)
        if not photo_data:
            return None
        
        # If analysis not cached, return placeholder
        # AI analysis is disabled until scene_service is reimplemented
        if not photo_data.get('ai_analysis'):
            return {
                'status': 'not_available',
                'message': 'AI analysis is temporarily disabled'
            }
        
        return photo_data.get('ai_analysis')
    
    
    
    def pause_processing(self):
        """Pause processing queue"""
        self.processing_active = False
    
    def resume_processing(self):
        """Resume processing queue"""
        self.processing_active = True
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        queue_size = self.processing_queue.qsize() if self.processing_queue else 0
        return {
            'active': self.processing_active,
            'queue_size': queue_size,
            'processing': sum(1 for p in self.db.list_photos() 
                            if p.get('status') == 'processing'),
            'pending': sum(1 for p in self.db.list_photos() 
                         if p.get('status') == 'pending')
        }


# Singleton instance
photo_service = EnhancedPhotoService()