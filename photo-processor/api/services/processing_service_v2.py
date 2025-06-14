"""
Enhanced Processing service with AI orchestration
Connects the API layer to the actual photo processing components.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
import logging
import traceback

# Add parent directory to path to import AI components
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.processing import (
    QueueStatus, ProcessingStatus, ProcessingSettings,
    BatchOperation, QueueItem
)
# from ai_components.orchestrator import PhotoProcessingOrchestrator, ProcessingConfig  # Temporarily disabled
from recipe_storage import RecipeStorage

# Configure logger
logger = logging.getLogger(__name__)


class EnhancedProcessingService:
    """Enhanced processing service with AI orchestration"""
    
    def __init__(self):
        logger.info("Initializing EnhancedProcessingService")
        self.is_paused = False
        self.settings = ProcessingSettings()
        self.ws_manager = None  # Will be injected
        
        # Initialize AI orchestrator - DISABLED FOR API STARTUP
        # self.config = ProcessingConfig(
        #     style_preset="natural",
        #     cull_aggressively=True,
        #     quality_threshold=5.0,
        #     enable_rotation_correction=True
        # )
        
        # Simple config replacement
        self.config = {
            'style_preset': 'natural',
            'cull_aggressively': True,
            'quality_threshold': 5.0,
            'enable_rotation_correction': True
        }
        
        self.output_dir = Path("/app/data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.thumbnail_dir = Path("/app/data/thumbnails")
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        self.web_dir = Path("/app/data/web")
        self.web_dir.mkdir(parents=True, exist_ok=True)
        
        # self.orchestrator = PhotoProcessingOrchestrator(
        #     config=self.config,
        #     output_dir=self.output_dir
        # )
        
        # Simple orchestrator replacement - processing disabled
        self.orchestrator = None
        
        # Initialize recipe storage
        self.recipe_storage = RecipeStorage(str(Path("/app/data/recipes")))
        
        # Processing queues and stats
        self.processing_queue = []
        self.processing_items = {}
        self.completed_items = []
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'processing_times': [],
            'errors_today': 0
        }
    
    async def get_queue_status(self) -> QueueStatus:
        """Get current queue status"""
        # Convert internal items to QueueItem models
        pending = []
        for i, item in enumerate(self.processing_queue):
            # Extract filename from photo_path
            photo_path = Path(item.get('photo_path', ''))
            filename = photo_path.name if photo_path else f"photo_{item['photo_id']}"
            
            pending.append(QueueItem(
                photo_id=item['photo_id'],
                filename=filename,
                position=i + 1,
                recipe_id=item.get('recipe_id'),
                priority=item.get('priority', 'normal'),
                created_at=item.get('created_at', datetime.now()),
                estimated_time=item.get('estimated_time', 30)
            ))
        
        processing = []
        for photo_id, data in self.processing_items.items():
            processing.append(QueueItem(
                photo_id=photo_id,
                filename=data.get('filename', f"photo_{photo_id}"),
                position=1,  # Currently processing
                recipe_id=data.get('recipe_id'),
                priority=data.get('priority', 'normal'),
                created_at=data.get('started_at', datetime.now()),
                estimated_time=data.get('estimated_time', 30)
            ))
        
        completed = []
        for i, item in enumerate(self.completed_items[-20:]):  # Last 20
            completed.append(QueueItem(
                photo_id=item['photo_id'],
                filename=item.get('filename', f"photo_{item['photo_id']}"),
                position=0,  # Completed items don't have a position
                recipe_id=item.get('recipe_id'),
                priority=item.get('priority', 'normal'),
                created_at=item.get('completed_at', datetime.now()),
                estimated_time=item.get('actual_time', 30)
            ))
        
        return QueueStatus(
            pending=pending,
            processing=processing,
            completed=completed,
            is_paused=self.is_paused,
            stats=await self.get_processing_status()
        )
    
    async def get_processing_status(self) -> ProcessingStatus:
        """Get processing statistics"""
        current_photo = None
        if self.processing_items:
            photo_id = list(self.processing_items.keys())[0]
            current_photo = {
                'id': photo_id,
                'filename': self.processing_items[photo_id].get('filename', 'Unknown')
            }
        
        # Calculate processing rate
        processing_rate = 0.0
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            processing_rate = 3600 / avg_time if avg_time > 0 else 0
        
        average_time = (
            sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            if self.stats['processing_times'] else 0.0
        )
        
        return ProcessingStatus(
            is_paused=self.is_paused,
            current_photo=current_photo,
            queue_length=len(self.processing_queue),
            processing_rate=processing_rate,
            average_time=average_time,
            errors_today=self.stats['errors_today']
        )
    
    def set_websocket_manager(self, ws_manager):
        """Set the WebSocket manager for broadcasting events"""
        self.ws_manager = ws_manager
    
    async def generate_thumbnail(self, image_path: Path, thumbnail_path: Path, size: tuple = (400, 400)) -> bool:
        """Generate a thumbnail for an image"""
        try:
            from PIL import Image
            import io
            
            # Check if this is a RAW file
            raw_extensions = {'.arw', '.nef', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.raf'}
            if image_path.suffix.lower() in raw_extensions:
                logger.info(f"Detected RAW file: {image_path.suffix}")
                try:
                    import rawpy
                    import numpy as np
                    
                    # Process RAW file with rawpy
                    with rawpy.imread(str(image_path)) as raw:
                        # Process with default settings for thumbnail
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            use_auto_wb=False,
                            no_auto_bright=False,
                            output_bps=8
                        )
                    
                    # Convert to PIL Image
                    img = Image.fromarray(rgb)
                    
                except ImportError:
                    logger.error("rawpy not installed - cannot process RAW files")
                    return False
                except Exception as e:
                    logger.error(f"Failed to process RAW file: {e}")
                    return False
            else:
                # Open regular image
                with Image.open(image_path) as img:
                    # Convert RGBA to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
            
            # Calculate thumbnail size maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save thumbnail with optimization
            img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            logger.info(f"Thumbnail generated: {thumbnail_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def generate_web_version(self, image_path: Path, web_path: Path, max_size: int = 1920) -> bool:
        """Generate a web-optimized version of an image"""
        try:
            from PIL import Image
            import io
            
            logger.info(f"Generating web version for {image_path} -> {web_path}")
            
            # Check if this is a RAW file
            raw_extensions = {'.arw', '.nef', '.cr2', '.cr3', '.dng', '.orf', '.rw2', '.raf'}
            if image_path.suffix.lower() in raw_extensions:
                logger.info(f"Detected RAW file for web version: {image_path.suffix}")
                try:
                    import rawpy
                    import numpy as np
                    
                    # Process RAW file with rawpy
                    with rawpy.imread(str(image_path)) as raw:
                        # Process with better settings for web display
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            use_auto_wb=False,
                            no_auto_bright=False,
                            output_bps=8,
                            bright=1.2  # Slightly brighten for web display
                        )
                    
                    # Convert to PIL Image
                    img = Image.fromarray(rgb)
                    
                except ImportError:
                    logger.error("rawpy not installed - cannot process RAW files")
                    return False
                except Exception as e:
                    logger.error(f"Failed to process RAW file: {e}")
                    return False
            else:
                # Open regular image
                with Image.open(image_path) as img:
                    # Convert RGBA to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
            
            # Calculate new size if image is larger than max_size
            width, height = img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Save web version with good quality
            img.save(web_path, 'JPEG', quality=90, optimize=True)
            
            logger.info(f"Web version generated successfully: {web_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate web version: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def pause_processing(self) -> bool:
        """Pause processing"""
        self.is_paused = True
        if self.ws_manager:
            await self.ws_manager.notify_system_status("paused", "Processing has been paused")
        return True
    
    async def resume_processing(self) -> bool:
        """Resume processing"""
        self.is_paused = False
        if self.ws_manager:
            await self.ws_manager.notify_system_status("resumed", "Processing has been resumed")
        return True
    
    async def queue_photo_processing(
        self,
        photo_id: str,
        photo_path: Path,
        recipe_id: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Queue a photo for processing"""
        try:
            queue_item = {
                'photo_id': photo_id,
                'photo_path': str(photo_path),
                'filename': photo_path.name if photo_path else f"photo_{photo_id}",
                'recipe_id': recipe_id,
                'priority': priority,
                'created_at': datetime.now(),
                'estimated_time': 30
            }
            
            # Add to queue based on priority
            if priority == "high":
                self.processing_queue.insert(0, queue_item)
            else:
                self.processing_queue.append(queue_item)
            
            # Notify clients of queue update
            if self.ws_manager:
                queue_stats = await self.get_queue_stats()
                await self.ws_manager.notify_queue_updated(queue_stats)
            return {
                "photo_id": photo_id,
                "status": "queued",
                "message": f"Photo queued for processing with priority: {priority}"
            }
            
        except Exception as e:
            logger.error(f"Failed to queue photo {photo_id}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def process_next_item(self) -> Optional[Dict[str, Any]]:
        """Process the next item in queue"""
        if self.is_paused:
            logger.debug("Processing is paused, skipping")
            return None
            
        if not self.processing_queue:
            return None
        
        # Get next item
        item = self.processing_queue.pop(0)
        photo_id = item['photo_id']
        photo_path = Path(item['photo_path'])
        recipe_id = item.get('recipe_id')
        
        logger.info(f"→ Processing started: {photo_id}")
        
        # Mark as processing
        self.processing_items[photo_id] = {
            'started_at': datetime.now(),
            'filename': photo_path.name,
            'recipe_id': recipe_id
        }
        
        # Notify clients that processing started
        if self.ws_manager:
            filename = photo_path.name if photo_path else f"photo_{photo_id}"
            await self.ws_manager.notify_processing_started(photo_id, recipe_id, filename)
            await self.ws_manager.notify_processing_stage(photo_id, "starting")
        
        try:
            start_time = datetime.now()
            
            # Get recipe if specified
            recipe = None
            recipe_data = None
            if recipe_id:
                try:
                    # Try to load recipe from JSON file first (service format)
                    from .recipe_adapter import RecipeAdapter
                    recipe_path = Path(f"/app/data/recipes/{recipe_id}.json")
                    recipe_data = RecipeAdapter.load_service_recipe(recipe_path)
                    
                    if recipe_data:
                        logger.info(f"Loaded service recipe {recipe_id}: {recipe_data.get('name', 'Unknown')}")
                        # Convert to ProcessingRecipe format if needed
                        recipe = RecipeAdapter.service_to_processing_recipe(recipe_data)
                    else:
                        # Fallback to recipe storage
                        recipe = self.recipe_storage.load_recipe(recipe_id)
                        if recipe:
                            logger.info(f"Loaded recipe {recipe_id} from storage")
                        else:
                            logger.warning(f"Recipe {recipe_id} not found")
                except Exception as e:
                    logger.error(f"Failed to load recipe {recipe_id}: {e}")
                    logger.error(traceback.format_exc())
                
                if self.ws_manager:
                    await self.ws_manager.notify_processing_stage(photo_id, "applying_recipe")
            
            # Notify of AI processing stages
            if self.ws_manager:
                await self.ws_manager.notify_processing_stage(photo_id, "ai_analysis")
            
            # Apply recipe operations if we have a recipe
            if recipe_data:
                # Use recipe processor to apply operations
                from .recipe_processor import recipe_processor
                
                processed_filename = f"{photo_id}_processed.jpg"
                processed_path = self.output_dir / processed_filename
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Applying recipe '{recipe_data.get('name', 'Unknown')}' to photo")
                result = await recipe_processor.process_image_with_recipe(
                    image_path=photo_path,
                    recipe=recipe_data,
                    output_path=processed_path
                )
                
                if not result['success']:
                    logger.error(f"Recipe processing failed: {result.get('error', 'Unknown error')}")
                    # Fall back to simple copy
                    import shutil
                    shutil.copy2(photo_path, processed_path)
                else:
                    logger.info(f"Recipe applied successfully: {result['operations_applied']} operations")
            else:
                # No recipe - just copy file to processed directory
                processed_path = self.output_dir / photo_path.name
                import shutil
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(photo_path, processed_path)
            
            # Generate thumbnail
            thumbnail_path = self.thumbnail_dir / f"{photo_id}_thumb.jpg"
            thumbnail_generated = await self.generate_thumbnail(processed_path, thumbnail_path)
            
            if not thumbnail_generated:
                logger.warning(f"Failed to generate thumbnail for {photo_id}, using processed image as fallback")
                thumbnail_path = None
            
            # Generate web version
            web_path = self.web_dir / f"{photo_id}_web.jpg"
            web_generated = await self.generate_web_version(processed_path, web_path)
            
            if not web_generated:
                logger.warning(f"Failed to generate web version for {photo_id}, using processed image as fallback")
                web_path = None
            
            # OneAlign Analysis is done during upload, not during processing
            # Just notify the stage change
            if self.ws_manager:
                await self.ws_manager.notify_processing_stage(photo_id, "ai_analysis")
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            result = {
                'photos_processed': 1,
                'photos_culled': 0,
                'processing_time': processing_time,
                'processed_photos': [{
                    'processed_path': str(processed_path)
                }]
            }
            
            # Update stats
            self.stats['processing_times'].append(processing_time)
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
            
            self.stats['total_processed'] += 1
            
            # Move to completed
            completed_item = {
                'photo_id': photo_id,
                'filename': photo_path.name if photo_path else f"photo_{photo_id}",
                'recipe_id': recipe_id,
                'completed_at': end_time,
                'actual_time': processing_time,
                'result': {
                    'photos_processed': result['photos_processed'],
                    'photos_culled': result['photos_culled'],
                    'processing_time': result['processing_time']
                }
            }
            
            self.completed_items.append(completed_item)
            if len(self.completed_items) > 1000:
                self.completed_items = self.completed_items[-1000:]
            
            # Update photo status in database
            from services.photo_service_sqlite import sqlite_photo_service as photo_service
            
            # Processing completed successfully
            final_status = 'completed'
            
            update_data = {
                'status': final_status,
                'processed_at': end_time,
                'processed_path': str(processed_path),
                'processing_time': processing_time,
                'recipe_id': recipe_id
            }
            
            if thumbnail_path:
                update_data['thumbnail_path'] = str(thumbnail_path)
            
            if web_path:
                update_data['web_path'] = str(web_path)
            
            # OneAlign analysis is done separately during upload
            # Just update the processing status
            await photo_service.db.update_photo(photo_id, update_data)
            
            # Remove from processing
            del self.processing_items[photo_id]
            
            # Notify clients of completion
            if self.ws_manager:
                processed_path = None
                if result['photos_processed'] > 0:
                    processed_path = str(result['processed_photos'][0]['processed_path'])
                filename = photo_path.name if photo_path else f"photo_{photo_id}"
                
                # No extra data needed - AI analysis is done separately
                
                await self.ws_manager.notify_processing_completed(
                    photo_id, 
                    success=True, 
                    processed_path=processed_path,
                    filename=filename
                )
                # Update queue stats
                queue_stats = await self.get_queue_stats()
                await self.ws_manager.notify_queue_updated(queue_stats)
            
            return {
                "photo_id": photo_id,
                "status": "completed",
                "processing_time": processing_time,
                "result": completed_item['result']
            }
            
        except Exception as e:
            logger.error(f"Error processing photo {photo_id}: {e}")
            logger.error(f"Photo path: {photo_path}")
            logger.error(traceback.format_exc())
            
            # Handle error
            self.stats['total_errors'] += 1
            self.stats['errors_today'] += 1
            
            # Remove from processing
            if photo_id in self.processing_items:
                del self.processing_items[photo_id]
            
            # Create user-friendly error message
            error_msg = str(e)
            if "'RecipeStorage' object has no attribute 'get_recipe'" in error_msg:
                error_msg = "Recipe system error - please try again"
            elif "recipe_id" in error_msg.lower():
                error_msg = "Invalid or missing recipe"
            elif "file not found" in error_msg.lower():
                error_msg = "Photo file not found"
            elif "permission" in error_msg.lower():
                error_msg = "File access permission denied"
            else:
                # Log the full error for debugging
                logger.error(f"Full error details: {error_msg}")
                error_msg = "Processing failed - check logs for details"
            
            # Update photo status to error
            try:
                from services.photo_service_sqlite import sqlite_photo_service as photo_service
                await photo_service.db.update_photo(photo_id, {
                    'status': 'failed',
                    'error_message': error_msg,
                    'processed_at': datetime.now()
                })
            except Exception as db_error:
                logger.error(f"Failed to update photo status to error: {db_error}")
            
            # Notify clients of error
            if self.ws_manager:
                filename = photo_path.name if photo_path else f"photo_{photo_id}"
                await self.ws_manager.notify_processing_failed(photo_id, str(e), filename)
                await self.ws_manager.notify_error("processing_error", str(e), photo_id)
                # Update queue stats
                queue_stats = await self.get_queue_stats()
                await self.ws_manager.notify_queue_updated(queue_stats)
            
            return {
                "photo_id": photo_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def approve_processing(
        self,
        photo_id: str,
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Approve processing with adjustments"""
        # Apply adjustments to processing config for this photo - SIMPLIFIED
        # adjusted_config = self.config.copy()
        # if 'quality_threshold' in adjustments:
        #     adjusted_config.quality_threshold = adjustments['quality_threshold']
        # if 'cull_aggressively' in adjustments:
        #     adjusted_config.cull_aggressively = adjustments['cull_aggressively']
        
        # Simple config adjustment
        adjusted_config = self.config.copy()
        if 'quality_threshold' in adjustments:
            adjusted_config['quality_threshold'] = adjustments['quality_threshold']
        if 'cull_aggressively' in adjustments:
            adjusted_config['cull_aggressively'] = adjustments['cull_aggressively']
        
        return {
            "photo_id": photo_id,
            "status": "approved",
            "message": "Processing approved with adjustments"
        }
    
    async def reject_processing(
        self,
        photo_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reject processing"""
        # Remove from queue if present
        self.processing_queue = [
            item for item in self.processing_queue 
            if item['photo_id'] != photo_id
        ]
        
        return {
            "photo_id": photo_id,
            "status": "rejected",
            "reason": reason or "Processing rejected by user"
        }
    
    async def batch_process(self, operation: BatchOperation) -> Dict[str, Any]:
        """Process multiple photos"""
        queued_count = 0
        skipped_photos = []
        errors = []
        
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + " BATCH PROCESSING SERVICE ".center(58) + "║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║ Photos to process: {str(len(operation.photo_ids)).ljust(38)} ║")
        logger.info(f"║ Recipe ID: {str(operation.recipe_id or 'None').ljust(46)} ║")
        logger.info(f"║ Priority: {operation.priority.ljust(47)} ║")
        logger.info("╚" + "═" * 58 + "╝")
        
        # Import photo service to get photo details
        from services.photo_service_sqlite import sqlite_photo_service as photo_service
        
        for i, photo_id in enumerate(operation.photo_ids, 1):
            try:
                logger.info(f"\n[{i}/{len(operation.photo_ids)}] Processing photo: {photo_id}")
                
                # Get photo details from database
                photo_detail = await photo_service.get_photo(photo_id)
                
                if not photo_detail:
                    error_msg = f"Photo {photo_id} not found in database"
                    logger.warning(f"  ⚠️  {error_msg}")
                    skipped_photos.append(photo_id)
                    errors.append(error_msg)
                    continue
                
                # PhotoDetail has original_path as an attribute
                if not photo_detail.original_path:
                    error_msg = f"Photo {photo_id} has no original_path"
                    logger.warning(f"  ⚠️  {error_msg}")
                    skipped_photos.append(photo_id)
                    errors.append(error_msg)
                    continue
                
                photo_path = Path(photo_detail.original_path)
                logger.info(f"  📁 Path: {photo_path}")
                logger.info(f"  📄 Filename: {photo_detail.filename}")
                logger.info(f"  📊 Status: {photo_detail.status}")
                
                if not photo_path.exists():
                    error_msg = f"Photo file not found: {photo_path}"
                    logger.warning(f"  ❌ {error_msg}")
                    skipped_photos.append(photo_id)
                    errors.append(error_msg)
                    continue
                
                # Queue the photo for processing
                logger.info(f"  🔄 Queuing with recipe: {operation.recipe_id or 'default'}")
                await self.queue_photo_processing(
                    photo_id=photo_id,
                    photo_path=photo_path,
                    recipe_id=operation.recipe_id,
                    priority=operation.priority
                )
                queued_count += 1
                logger.info(f"  ✅ Successfully queued!")
                
            except Exception as e:
                error_msg = f"Error processing photo {photo_id}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                skipped_photos.append(photo_id)
                errors.append(error_msg)
        
        result = {
            "queued": queued_count,
            "skipped": len(operation.photo_ids) - queued_count,
            "message": f"Queued {queued_count} photos for processing",
            "skipped_photos": skipped_photos,
            "errors": errors
        }
        
        logger.info("\n" + "─" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("─" * 60)
        logger.info(f"✅ Successfully queued: {queued_count} photos")
        if skipped_photos:
            logger.info(f"⚠️  Skipped: {len(skipped_photos)} photos")
            for photo_id in skipped_photos[:5]:  # Show first 5
                logger.info(f"   - {photo_id}")
            if len(skipped_photos) > 5:
                logger.info(f"   ... and {len(skipped_photos) - 5} more")
        if errors:
            logger.warning(f"❌ Errors encountered: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                logger.warning(f"   - {error}")
            if len(errors) > 3:
                logger.warning(f"   ... and {len(errors) - 3} more errors")
        logger.info("─" * 60 + "\n")
        
        return result
    
    async def reorder_queue(self, photo_id: str, new_position: int) -> bool:
        """Reorder queue items"""
        # Find item in queue
        item_index = None
        for i, item in enumerate(self.processing_queue):
            if item['photo_id'] == photo_id:
                item_index = i
                break
        
        if item_index is None:
            return False
        
        # Move item to new position
        item = self.processing_queue.pop(item_index)
        new_position = max(0, min(new_position, len(self.processing_queue)))
        self.processing_queue.insert(new_position, item)
        
        return True
    
    async def remove_from_queue(self, photo_id: str) -> bool:
        """Remove item from queue"""
        original_length = len(self.processing_queue)
        self.processing_queue = [
            item for item in self.processing_queue 
            if item['photo_id'] != photo_id
        ]
        return len(self.processing_queue) < original_length
    
    async def get_settings(self) -> ProcessingSettings:
        """Get processing settings"""
        return self.settings
    
    async def update_settings(self, settings: ProcessingSettings) -> ProcessingSettings:
        """Update processing settings"""
        self.settings = settings
        
        # Update orchestrator config - SIMPLIFIED
        # self.config.style_preset = getattr(settings, 'default_style', 'natural')
        # self.config.quality_threshold = getattr(settings, 'quality_threshold', 5.0)
        # self.config.cull_aggressively = getattr(settings, 'aggressive_culling', True)
        
        # Simple config updates
        self.config['style_preset'] = getattr(settings, 'default_style', 'natural')
        self.config['quality_threshold'] = getattr(settings, 'quality_threshold', 5.0)
        self.config['cull_aggressively'] = getattr(settings, 'aggressive_culling', True)
        
        return self.settings
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            "pending": len(self.processing_queue),
            "processing": len(self.processing_items),
            "completed": len(self.completed_items)
        }
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        return {
            "total_processed": self.stats['total_processed'],
            "total_errors": self.stats['total_errors'],
            "errors_today": self.stats['errors_today'],
            "average_processing_time": (
                sum(self.stats['processing_times']) / len(self.stats['processing_times'])
                if self.stats['processing_times'] else 0
            ),
            "queue_stats": await self.get_queue_stats()
        }


# Singleton instance
processing_service = EnhancedProcessingService()