"""
Main photo processing service with file watching and AI-powered image enhancement.

This service monitors a designated folder for new photos (RAW and standard formats),
processes them using AI analysis for intelligent cropping and enhancement, then
uploads the results to an Immich photo management instance.

Key features:
- Real-time file monitoring using watchdog
- AI-powered image analysis via Ollama/Gemma
- RAW file conversion and processing
- Smart cropping and color enhancement
- Automatic upload to Immich with metadata
- Duplicate detection via SHA256 hashing
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

# Configure logging with both console and file output for debugging and monitoring
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
    """
    Handle new photo files detected by the file system watcher.
    
    This handler processes files when they are created or moved into the watched
    directory. It includes duplicate detection, file validation, and manages
    the processing pipeline for each detected photo.
    """
    
    def __init__(self, processor_service):
        self.processor_service = processor_service
        self.processing_files: Set[str] = set()
        
        # Supported RAW and image formats - covers most camera manufacturers
        # RAW: Canon (CR2/CR3), Nikon (NEF), Sony (ARW), Adobe (DNG), etc.
        # Standard: JPEG, PNG, TIFF for already-processed images
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
        """
        Handle a new file detected in the watch folder.
        
        This method performs several checks before processing:
        1. Validates file extension is supported
        2. Prevents concurrent processing of the same file
        3. Checks hash tracker to avoid reprocessing
        4. Waits for file write completion
        5. Initiates the processing pipeline
        
        Args:
            file_path: Full path to the detected file
        """
        try:
            # Check if file extension is supported (case-insensitive)
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                return
            
            # Avoid processing the same file multiple times concurrently
            if file_path in self.processing_files:
                return
            
            # Check if file has already been processed using SHA256 hash tracking
            if self.processor_service.hash_tracker.is_already_processed(file_path):
                logger.info(f"Skipping already processed file: {os.path.basename(file_path)}")
                return
                
            # Wait for file to be fully written (important for large RAW files)
            self._wait_for_file_ready(file_path)
            
            # Add to processing set
            self.processing_files.add(file_path)
            
            logger.info(f"New photo detected: {file_path}")
            
            # Process the file through the complete AI-enhanced pipeline
            success = self.processor_service.process_photo(file_path)
            
            if success:
                logger.info(f"Successfully processed: {file_path}")
                # Original files are moved to processed folder instead of deleted
                # This preserves originals while preventing reprocessing
            else:
                logger.error(f"Failed to process: {file_path}")
            
        except Exception as e:
            logger.error(f"Error handling file {file_path}: {e}")
        finally:
            # Remove from processing set
            self.processing_files.discard(file_path)
    
    def _wait_for_file_ready(self, file_path: str, timeout: int = 30):
        """
        Wait for file to be fully written to disk.
        
        This method monitors file size changes to detect when a file has finished
        being written. This is crucial for large RAW files that may take several
        seconds to copy or download.
        
        Args:
            file_path: Path to the file to monitor
            timeout: Maximum seconds to wait before proceeding
        """
        start_time = time.time()
        last_size = -1
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    # File size stable and non-empty, likely finished writing
                    time.sleep(1)  # Extra safety margin
                    return
                last_size = current_size
                time.sleep(2)
            except OSError:
                # File might not exist yet or still being written
                time.sleep(2)

class PhotoProcessorService:
    """
    Main photo processing service orchestrating all components.
    
    This service manages:
    - Service initialization and configuration
    - Health checks for all dependencies (Ollama, Immich)
    - File watching and processing pipeline
    - Integration between AI analysis, image processing, and upload
    - File permissions and organization
    """
    
    def __init__(self):
        # Configuration from environment variables for Docker deployment flexibility
        self.watch_folder = os.getenv('WATCH_FOLDER', '/app/inbox')
        self.output_folder = os.getenv('OUTPUT_FOLDER', '/app/processed')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.immich_api_url = os.getenv('IMMICH_API_URL', 'http://immich_server:2283')
        self.immich_api_key = os.getenv('IMMICH_API_KEY')
        
        # Create required directories with proper permissions for file access
        os.makedirs(self.watch_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create processed originals folder for archiving completed files
        # This preserves originals while preventing reprocessing
        self.processed_originals_folder = os.path.join(self.output_folder, 'originals')
        os.makedirs(self.processed_originals_folder, exist_ok=True)
        
        # Set proper permissions on directories (755 = rwxr-xr-x)
        # Ensures the service and users can access processed files
        self._set_permissions(self.watch_folder, 0o755)
        self._set_permissions(self.output_folder, 0o755)
        self._set_permissions(self.processed_originals_folder, 0o755)
        
        # Initialize core services with dependency injection pattern
        self.image_processor = ImageProcessor()
        self.ai_analyzer = AIAnalyzer(self.ollama_host)
        self.immich_client = ImmichClient(self.immich_api_url, self.immich_api_key)
        self.hash_tracker = HashTracker()
        
        # Album ID for organizing processed photos in Immich
        self.album_id = None
        
        logger.info("PhotoProcessorService initialized")
    
    def _set_permissions(self, path: str, mode: int = 0o644):
        """
        Set file/directory permissions for proper access control.
        
        Attempts to set both permissions (mode) and ownership (UID/GID 1000).
        Ownership change requires root privileges, so it fails gracefully
        when running as non-root user.
        
        Args:
            path: File or directory path
            mode: Unix permission mode (default 0o644 = rw-r--r--)
        """
        try:
            os.chmod(path, mode)
            # Try to set ownership to UID/GID 1000 (typical first user)
            # This ensures files are accessible outside the container
            try:
                os.chown(path, 1000, 1000)
            except PermissionError:
                # Non-root containers can't change ownership, which is expected
                logger.debug(f"Could not change ownership of {path} (running as non-root)")
        except Exception as e:
            logger.warning(f"Could not set permissions on {path}: {e}")
    
    def startup_checks(self) -> bool:
        """
        Perform comprehensive startup health checks.
        
        Validates all external dependencies are available and properly configured:
        1. Ollama AI service connectivity
        2. Required AI model availability
        3. Immich API connectivity and authentication
        4. Album creation for photo organization
        
        Returns:
            bool: True if all checks pass, False otherwise
        """
        logger.info("Performing startup checks...")
        
        # Check Ollama AI service is reachable
        if not self.ai_analyzer.test_connection():
            logger.error("Failed to connect to Ollama")
            return False
            
        if not self.ai_analyzer.ensure_model_available():
            logger.error("Failed to ensure AI model availability")
            return False
        
        # Check Immich API connectivity and authentication
        if not self.immich_client.test_connection():
            logger.error("Failed to connect to Immich")
            return False
        
        # Create or find album for organizing processed photos in Immich
        self.album_id = self.immich_client.get_or_create_album("AI Processed Photos")
        if not self.album_id:
            logger.warning("Failed to create/find album, photos will be uploaded without album")
        
        logger.info("All startup checks passed!")
        return True
    
    def process_photo(self, file_path: str) -> bool:
        """
        Process a single photo through the complete AI-enhanced pipeline.
        
        Pipeline stages:
        1. Load and convert image (RAW conversion if needed)
        2. Blur detection for quality assessment
        3. AI analysis for subject detection and composition
        4. Apply intelligent processing (rotation, crop, enhancement)
        5. Save high-quality processed image
        6. Upload to Immich with metadata
        7. Archive original file
        
        Args:
            file_path: Path to the photo file to process
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        start_time = time.time()
        original_filename = os.path.basename(file_path)
        
        try:
            logger.info(f"Starting processing pipeline for: {original_filename}")
            
            # Step 1: Load and convert image based on file type
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
            
            # Step 2: Check for blur (warning only, doesn't stop processing)
            is_blurry, blur_score = self.image_processor.detect_blur(rgb_image)
            if is_blurry:
                logger.warning(f"Image detected as blurry (score: {blur_score:.2f}), but continuing with processing")
            
            # Step 3: Prepare smaller version for efficient AI analysis
            ai_image = self.image_processor.resize_for_ai_analysis(rgb_image)
            temp_ai_path = self.image_processor.save_temp_image_for_ai(ai_image)
            
            try:
                # Step 4: AI Analysis using Ollama/Gemma for intelligent processing
                analysis = self.ai_analyzer.analyze_photo(temp_ai_path, original_filename)
                if not analysis:
                    logger.error("AI analysis failed")
                    return False
                
                # Step 5: Apply AI-recommended processing transformations
                
                # Log the AI's analysis for debugging and monitoring
                logger.info(f"Subject box: x={analysis.primary_subject_box.x:.1f}%, y={analysis.primary_subject_box.y:.1f}%, "
                           f"w={analysis.primary_subject_box.width:.1f}%, h={analysis.primary_subject_box.height:.1f}%")
                logger.info(f"Recommended crop: x={analysis.recommended_crop.crop_box.x:.1f}%, y={analysis.recommended_crop.crop_box.y:.1f}%, "
                           f"w={analysis.recommended_crop.crop_box.width:.1f}%, h={analysis.recommended_crop.crop_box.height:.1f}%")
                logger.info(f"Rotation: {analysis.recommended_crop.rotation_degrees:.1f} degrees")
                
                # Apply processing pipeline with AI recommendations
                working_image = rgb_image
                
                # Step 5a: Apply rotation to correct tilted horizons or orientation
                rotation_needed = analysis.recommended_crop.rotation_degrees
                if abs(rotation_needed) > 0.1:
                    logger.info(f"Applying rotation: {rotation_needed:.1f} degrees")
                    working_image = self.image_processor.apply_rotation(working_image, rotation_needed)
                
                # Step 5b: Apply intelligent crop based on subject and composition
                logger.info("Applying recommended crop")
                cropped_image = self.image_processor.apply_smart_crop(working_image, analysis.recommended_crop.crop_box)
                crop_height, crop_width = cropped_image.shape[:2]
                logger.info(f"Applied crop: {crop_width}x{crop_height} pixels (aspect: {analysis.recommended_crop.aspect_ratio})")
                working_image = cropped_image
                
                # Step 5c: Apply color enhancements (brightness, contrast, CLAHE)
                processed_image = self.image_processor.enhance_image(working_image, analysis.color_analysis)
                logger.info("Applied color enhancements")
                
                # Step 6: Save processed image with timestamp to avoid overwrites
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = original_filename.rsplit('.', 1)[0]
                output_filename = f"processed_{base_name}_{timestamp}.jpg"
                output_path = os.path.join(self.output_folder, output_filename)
                
                self.image_processor.save_high_quality_jpeg(processed_image, output_path)
                # Set permissions to ensure file is accessible outside container
                self._set_permissions(output_path, 0o644)
                
                # Step 7: Upload to Immich with AI-generated metadata and tags
                asset_id = self.immich_client.upload_photo(
                    output_path, 
                    output_filename, 
                    analysis, 
                    self.album_id
                )
                
                if asset_id and asset_id != "duplicate":
                    # Step 8: Archive original file to prevent reprocessing
                    try:
                        moved_path = os.path.join(self.processed_originals_folder, original_filename)
                        shutil.move(file_path, moved_path)
                        # Ensure archived file has proper permissions
                        self._set_permissions(moved_path, 0o644)
                        logger.info(f"Moved original file to: {moved_path}")
                        
                        # Record file hash to prevent future reprocessing
                        self.hash_tracker.mark_as_processed(moved_path)
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Successfully completed processing pipeline for {original_filename} in {processing_time:.2f}s")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to move original file: {e}")
                        # Still mark as processed to prevent retry loops
                        self.hash_tracker.mark_as_processed(file_path)
                        return True
                elif asset_id == "duplicate":
                    # Handle Immich duplicate detection gracefully
                    logger.info(f"Duplicate detected for {original_filename}, marking as processed")
                    self.hash_tracker.mark_as_processed(file_path)
                    processing_time = time.time() - start_time
                    logger.info(f"Completed processing (duplicate) for {original_filename} in {processing_time:.2f}s")
                    return True
                else:
                    logger.error("Failed to upload to Immich")
                    return False
                
            finally:
                # Cleanup temporary files used for AI analysis
                try:
                    os.remove(temp_ai_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}", exc_info=True)
            return False
    
    def start_watching(self):
        """
        Start the file system watcher for real-time photo detection.
        
        Uses the watchdog library to monitor the configured folder for new
        photos. Runs indefinitely until interrupted by shutdown signal.
        """
        logger.info(f"Starting file watcher on: {self.watch_folder}")
        
        # Create watchdog components for file system monitoring
        event_handler = PhotoFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.watch_folder, recursive=True)
        
        # Start the file system observer thread
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
        """
        Process all existing files in the watch folder on startup.
        
        This method handles bulk processing of files that were already present
        in the watch folder when the service starts. It processes them sequentially,
        tracks progress, and provides a summary of results.
        
        Useful for:
        - Initial bulk import of existing photos
        - Reprocessing after service restart
        - Catching up on files added while service was down
        """
        logger.info("Processing all existing files in watch folder...")
        
        # Recursively find all supported image files in watch folder
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
                # Skip files that have already been processed (via hash check)
                if self.hash_tracker.is_already_processed(file_path):
                    logger.info(f"Skipping already processed file: {filename}")
                    skipped_count += 1
                    continue
                
                # Process the file through the full pipeline
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
        
        # Print comprehensive processing summary for monitoring
        logger.info(f"""
=== BULK PROCESSING SUMMARY ===
Total files found: {total_files}
Successfully processed: {processed_count}
Already processed (skipped): {skipped_count}
Failed: {failed_count}
===============================""")
        
        # Print historical processing statistics from hash tracker
        stats = self.hash_tracker.get_stats()
        logger.info(f"Hash tracker: {stats['total_processed']} total files processed historically")

def signal_handler(signum, frame):
    """
    Handle shutdown signals for graceful service termination.
    
    Responds to SIGINT (Ctrl+C) and SIGTERM (Docker stop) to ensure
    clean shutdown of file watchers and processing threads.
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """
    Main entry point for the photo processing service.
    
    Initialization sequence:
    1. Register signal handlers for graceful shutdown
    2. Initialize the photo processor service
    3. Run startup health checks
    4. Process any existing files in the watch folder
    5. Start real-time file watching
    """
    # Register signal handlers for graceful shutdown on SIGINT/SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting AI Photo Processor Service")
    
    # Initialize service with all components
    service = PhotoProcessorService()
    
    # Perform startup checks to ensure all dependencies are available
    if not service.startup_checks():
        logger.error("Startup checks failed, exiting")
        sys.exit(1)
    
    # Process ALL existing files in the watch folder on startup
    # This handles files that were added while the service was down
    service.process_all_files_in_folder()
    
    # Start real-time file watching for new photos (runs indefinitely)
    service.start_watching()

if __name__ == "__main__":
    main()