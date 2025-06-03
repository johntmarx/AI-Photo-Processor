"""
Main photo processing service with file watching
"""
import os
import time
import logging
import signal
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import magic
from typing import Set

from image_processor_v2 import ImageProcessor
from ai_analyzer import AIAnalyzer
from immich_client import ImmichClient
from hash_tracker import HashTracker
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG') == 'true' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/photo_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class PhotoFileHandler(FileSystemEventHandler):
    """Handle new photo files"""
    
    def __init__(self, processor_service):
        self.processor_service = processor_service
        self.processing_files: Set[str] = set()
        
        # Supported RAW and image formats
        self.supported_extensions = {
            '.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2', 
            '.pef', '.srw', '.x3f', '.fff', '.3fr', '.mrw', '.raw',
            '.jpg', '.jpeg', '.png', '.tiff', '.tif'
        }
    
    def on_created(self, event):
        if not event.is_directory:
            self._handle_file(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory:
            self._handle_file(event.dest_path)
    
    def _handle_file(self, file_path: str):
        """Handle a new file"""
        try:
            # Check if file extension is supported
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                return
            
            # Avoid processing the same file multiple times
            if file_path in self.processing_files:
                return
            
            # Check if file has already been processed using hash
            if self.processor_service.hash_tracker.is_already_processed(file_path):
                logger.info(f"Skipping already processed file: {os.path.basename(file_path)}")
                return
                
            # Wait for file to be fully written
            self._wait_for_file_ready(file_path)
            
            # Add to processing set
            self.processing_files.add(file_path)
            
            logger.info(f"New photo detected: {file_path}")
            
            # Process the file
            success = self.processor_service.process_photo(file_path)
            
            if success:
                logger.info(f"Successfully processed: {file_path}")
                # Optionally delete original file after successful processing
                # os.remove(file_path)
            else:
                logger.error(f"Failed to process: {file_path}")
            
        except Exception as e:
            logger.error(f"Error handling file {file_path}: {e}")
        finally:
            # Remove from processing set
            self.processing_files.discard(file_path)
    
    def _wait_for_file_ready(self, file_path: str, timeout: int = 30):
        """Wait for file to be fully written"""
        start_time = time.time()
        last_size = -1
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    # File size hasn't changed, likely finished writing
                    time.sleep(1)  # Extra safety margin
                    return
                last_size = current_size
                time.sleep(2)
            except OSError:
                # File might not exist yet or still being written
                time.sleep(2)

class PhotoProcessorService:
    """Main photo processing service"""
    
    def __init__(self):
        # Configuration from environment
        self.watch_folder = os.getenv('WATCH_FOLDER', '/app/inbox')
        self.output_folder = os.getenv('OUTPUT_FOLDER', '/app/processed')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.immich_api_url = os.getenv('IMMICH_API_URL', 'http://immich_server:2283')
        self.immich_api_key = os.getenv('IMMICH_API_KEY')
        
        # Create directories with proper permissions
        os.makedirs(self.watch_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create processed originals folder for moving completed files
        self.processed_originals_folder = os.path.join(self.output_folder, 'originals')
        os.makedirs(self.processed_originals_folder, exist_ok=True)
        
        # Set proper permissions on directories (755)
        self._set_permissions(self.watch_folder, 0o755)
        self._set_permissions(self.output_folder, 0o755)
        self._set_permissions(self.processed_originals_folder, 0o755)
        
        # Initialize services
        self.image_processor = ImageProcessor()
        self.ai_analyzer = AIAnalyzer(self.ollama_host)
        self.immich_client = ImmichClient(self.immich_api_url, self.immich_api_key)
        self.hash_tracker = HashTracker()
        
        # Album for processed photos
        self.album_id = None
        
        logger.info("PhotoProcessorService initialized")
    
    def _set_permissions(self, path: str, mode: int = 0o644):
        """Set permissions to allow user access"""
        try:
            os.chmod(path, mode)
            # Try to set ownership to UID/GID 1000 (user john)
            # This will only work if running as root or the same user
            try:
                os.chown(path, 1000, 1000)
            except PermissionError:
                # If we can't change ownership, at least log it
                logger.debug(f"Could not change ownership of {path} (running as non-root)")
        except Exception as e:
            logger.warning(f"Could not set permissions on {path}: {e}")
    
    def startup_checks(self) -> bool:
        """Perform startup health checks"""
        logger.info("Performing startup checks...")
        
        # Check Ollama connection and model
        if not self.ai_analyzer.test_connection():
            logger.error("Failed to connect to Ollama")
            return False
            
        if not self.ai_analyzer.ensure_model_available():
            logger.error("Failed to ensure AI model availability")
            return False
        
        # Check Immich connection
        if not self.immich_client.test_connection():
            logger.error("Failed to connect to Immich")
            return False
        
        # Create album for processed photos
        self.album_id = self.immich_client.get_or_create_album("AI Processed Photos")
        if not self.album_id:
            logger.warning("Failed to create/find album, photos will be uploaded without album")
        
        logger.info("All startup checks passed!")
        return True
    
    def process_photo(self, file_path: str) -> bool:
        """Process a single photo through the complete pipeline"""
        start_time = time.time()
        original_filename = os.path.basename(file_path)
        
        try:
            logger.info(f"Starting processing pipeline for: {original_filename}")
            
            # Step 1: Load and convert image
            if file_path.lower().endswith(('.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf')):
                # RAW file processing
                rgb_image, metadata = self.image_processor.convert_raw_to_rgb(file_path)
                logger.info(f"Converted RAW file: {metadata.get('width')}x{metadata.get('height')}")
            else:
                # Regular image file
                from PIL import Image
                import numpy as np
                pil_image = Image.open(file_path)
                rgb_image = np.array(pil_image)
                metadata = {'width': pil_image.width, 'height': pil_image.height}
            
            # Step 2: Check for blur
            is_blurry, blur_score = self.image_processor.detect_blur(rgb_image)
            if is_blurry:
                logger.warning(f"Image detected as blurry (score: {blur_score:.2f}), but continuing with processing")
            
            # Step 3: Prepare image for AI analysis
            ai_image = self.image_processor.resize_for_ai_analysis(rgb_image)
            temp_ai_path = self.image_processor.save_temp_image_for_ai(ai_image)
            
            try:
                # Step 4: AI Analysis
                analysis = self.ai_analyzer.analyze_photo(temp_ai_path, original_filename)
                if not analysis:
                    logger.error("AI analysis failed")
                    return False
                
                # Step 5: Apply processing based on AI recommendations
                
                # Log the AI's analysis
                logger.info(f"Subject box: x={analysis.primary_subject_box.x:.1f}%, y={analysis.primary_subject_box.y:.1f}%, "
                           f"w={analysis.primary_subject_box.width:.1f}%, h={analysis.primary_subject_box.height:.1f}%")
                logger.info(f"Recommended crop: x={analysis.recommended_crop.crop_box.x:.1f}%, y={analysis.recommended_crop.crop_box.y:.1f}%, "
                           f"w={analysis.recommended_crop.crop_box.width:.1f}%, h={analysis.recommended_crop.crop_box.height:.1f}%")
                logger.info(f"Rotation: {analysis.recommended_crop.rotation_degrees:.1f} degrees")
                
                # Apply processing
                working_image = rgb_image
                
                # Step 4a: Apply rotation if needed
                rotation_needed = analysis.recommended_crop.rotation_degrees
                if abs(rotation_needed) > 0.1:
                    logger.info(f"Applying rotation: {rotation_needed:.1f} degrees")
                    working_image = self.image_processor.apply_rotation(working_image, rotation_needed)
                
                # Step 4b: Always apply the recommended crop
                logger.info("Applying recommended crop")
                cropped_image = self.image_processor.apply_smart_crop(working_image, analysis.recommended_crop.crop_box)
                crop_height, crop_width = cropped_image.shape[:2]
                logger.info(f"Applied crop: {crop_width}x{crop_height} pixels (aspect: {analysis.recommended_crop.aspect_ratio})")
                working_image = cropped_image
                
                # Step 4c: Apply color enhancements
                processed_image = self.image_processor.enhance_image(working_image, analysis.color_analysis)
                logger.info("Applied color enhancements")
                
                # Step 5: Save processed image
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = original_filename.rsplit('.', 1)[0]
                output_filename = f"processed_{base_name}_{timestamp}.jpg"
                output_path = os.path.join(self.output_folder, output_filename)
                
                self.image_processor.save_high_quality_jpeg(processed_image, output_path)
                # Set permissions on the processed file
                self._set_permissions(output_path, 0o644)
                
                # Step 6: Upload to Immich
                asset_id = self.immich_client.upload_photo(
                    output_path, 
                    output_filename, 
                    analysis, 
                    self.album_id
                )
                
                if asset_id and asset_id != "duplicate":
                    # Step 7: Move original file to processed folder and mark as processed
                    try:
                        moved_path = os.path.join(self.processed_originals_folder, original_filename)
                        shutil.move(file_path, moved_path)
                        # Set permissions on the moved file
                        self._set_permissions(moved_path, 0o644)
                        logger.info(f"Moved original file to: {moved_path}")
                        
                        # Mark as processed in hash tracker
                        self.hash_tracker.mark_as_processed(moved_path)
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Successfully completed processing pipeline for {original_filename} in {processing_time:.2f}s")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to move original file: {e}")
                        # Still mark as processed even if move fails
                        self.hash_tracker.mark_as_processed(file_path)
                        return True
                elif asset_id == "duplicate":
                    # Handle duplicate - still mark as processed but don't move file
                    logger.info(f"Duplicate detected for {original_filename}, marking as processed")
                    self.hash_tracker.mark_as_processed(file_path)
                    processing_time = time.time() - start_time
                    logger.info(f"Completed processing (duplicate) for {original_filename} in {processing_time:.2f}s")
                    return True
                else:
                    logger.error("Failed to upload to Immich")
                    return False
                
            finally:
                # Cleanup temp files
                try:
                    os.remove(temp_ai_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}", exc_info=True)
            return False
    
    def start_watching(self):
        """Start file watching service"""
        logger.info(f"Starting file watcher on: {self.watch_folder}")
        
        # Create event handler and observer
        event_handler = PhotoFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.watch_folder, recursive=True)
        
        # Start observer
        observer.start()
        logger.info("File watcher started successfully")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down file watcher...")
            observer.stop()
        
        observer.join()
    
    def process_all_files_in_folder(self):
        """Process all existing files in the watch folder"""
        logger.info("Processing all existing files in watch folder...")
        
        # Get all supported files in the watch folder
        all_files = []
        for file_path in Path(self.watch_folder).rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                if file_ext in {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2', 
                              '.pef', '.srw', '.x3f', '.fff', '.3fr', '.mrw', '.raw',
                              '.jpg', '.jpeg', '.png', '.tiff', '.tif'}:
                    all_files.append(str(file_path))
        
        total_files = len(all_files)
        logger.info(f"Found {total_files} image files to process")
        
        if total_files == 0:
            logger.info("No image files found to process")
            return
        
        # Process each file
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(all_files, 1):
            filename = os.path.basename(file_path)
            logger.info(f"Processing [{i}/{total_files}]: {filename}")
            
            try:
                # Check if already processed
                if self.hash_tracker.is_already_processed(file_path):
                    logger.info(f"Skipping already processed file: {filename}")
                    skipped_count += 1
                    continue
                
                # Process the file
                success = self.process_photo(file_path)
                
                if success:
                    processed_count += 1
                    logger.info(f"✓ Successfully processed: {filename}")
                else:
                    failed_count += 1
                    logger.error(f"✗ Failed to process: {filename}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"✗ Error processing {filename}: {e}")
        
        # Print summary
        logger.info(f"""
=== BULK PROCESSING SUMMARY ===
Total files found: {total_files}
Successfully processed: {processed_count}
Already processed (skipped): {skipped_count}
Failed: {failed_count}
===============================""")
        
        # Print hash tracker stats
        stats = self.hash_tracker.get_stats()
        logger.info(f"Hash tracker: {stats['total_processed']} total files processed historically")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting AI Photo Processor Service")
    
    # Initialize service
    service = PhotoProcessorService()
    
    # Perform startup checks
    if not service.startup_checks():
        logger.error("Startup checks failed, exiting")
        sys.exit(1)
    
    # Process ALL existing files in the watch folder using bulk processing
    service.process_all_files_in_folder()
    
    # Start file watching for new files
    service.start_watching()

if __name__ == "__main__":
    main()