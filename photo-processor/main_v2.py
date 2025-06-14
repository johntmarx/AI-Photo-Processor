"""
Enhanced Photo Processor with Original Preservation

This is the main entry point for the photo processing service that preserves
original files while creating optimized versions for Immich.
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import hashlib
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_analyzer import AIAnalyzer
from image_processor_v2 import ImageProcessor
from immich_client_v2 import EnhancedImmichClient, DualUploadResult
from hash_tracker import HashTracker
from recipe_storage import RecipeStorage, ProcessingRecipe, ProcessingOperation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/photo_processor.log')
    ]
)
logger = logging.getLogger(__name__)


class EnhancedPhotoProcessor:
    """Enhanced photo processor that preserves originals"""
    
    def __init__(self):
        # Initialize components
        self.ai_analyzer = AIAnalyzer()
        self.image_processor = ImageProcessor()
        self.immich_client = EnhancedImmichClient(
            base_url=os.getenv('IMMICH_API_URL', 'http://immich-server:2283'),
            api_key=os.getenv('IMMICH_API_KEY', '')
        )
        self.hash_tracker = HashTracker('/app/data/hash_tracker.db')
        self.recipe_storage = RecipeStorage(Path('/app/data/recipes'))
        
        # Configure paths
        self.inbox_folder = Path(os.getenv('INBOX_FOLDER', '/app/inbox'))
        self.originals_folder = Path('/app/data/originals')
        self.processed_folder = Path('/app/data/processed')
        self.working_folder = Path('/app/data/working')
        
        # Create necessary directories
        for folder in [self.originals_folder, self.processed_folder, 
                      self.working_folder, '/app/logs']:
            Path(folder).mkdir(parents=True, exist_ok=True)
            
        # Processing settings
        self.preserve_originals = True  # Always true in v2
        self.dual_upload = True  # Upload both versions
        self.enable_ai_processing = os.getenv('ENABLE_AI_PROCESSING', 'true').lower() == 'true'
        
        logger.info("Enhanced Photo Processor initialized with original preservation")
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def store_original(self, file_path: Path, file_hash: str) -> Path:
        """Store original file in permanent storage (copy, don't move!)"""
        # Organize by date
        file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
        date_path = self.originals_folder / f"{file_date.year:04d}" / f"{file_date.month:02d}"
        date_path.mkdir(parents=True, exist_ok=True)
        
        # Store with hash prefix to avoid collisions
        original_filename = f"{file_hash[:8]}_{file_path.name}"
        stored_path = date_path / original_filename
        
        # Copy (not move!) the original file
        shutil.copy2(file_path, stored_path)
        
        logger.info(f"Stored original file: {stored_path}")
        return stored_path
    
    def create_processing_recipe(self, 
                               file_path: Path,
                               file_hash: str,
                               ai_results: Optional[Dict[str, Any]] = None) -> ProcessingRecipe:
        """Create a processing recipe from AI analysis"""
        recipe = ProcessingRecipe(
            original_hash=file_hash,
            original_filename=file_path.name
        )
        
        if ai_results:
            # Store AI metadata
            recipe.ai_metadata = {
                'analysis_version': '1.0',
                'model': 'gemma3:4b',
                'timestamp': datetime.now().isoformat(),
                'results': ai_results
            }
            
            # Convert AI suggestions to operations
            # Rotation
            if ai_results.get('rotation_needed'):
                rotation_angle = ai_results.get('rotation_angle', 0)
                if abs(rotation_angle) > 0.5:  # Only rotate if significant
                    recipe.add_operation(
                        'rotate',
                        {'angle': rotation_angle},
                        source='ai'
                    )
            
            # Crop
            if ai_results.get('suggested_crop'):
                crop = ai_results['suggested_crop']
                recipe.add_operation(
                    'crop',
                    {
                        'x1': crop.get('x1', 0),
                        'y1': crop.get('y1', 0),
                        'x2': crop.get('x2', 1),
                        'y2': crop.get('y2', 1)
                    },
                    source='ai'
                )
            
            # Color adjustments
            adjustments = ai_results.get('color_adjustments', {})
            if any(abs(v) > 0.05 for v in adjustments.values()):  # Only if significant
                recipe.add_operation(
                    'enhance',
                    adjustments,
                    source='ai'
                )
        
        return recipe
    
    def apply_recipe(self, 
                    image_path: Path,
                    recipe: ProcessingRecipe,
                    output_path: Path) -> bool:
        """Apply a processing recipe to an image"""
        try:
            # Load image using cv2
            import cv2
            import numpy as np
            
            # Check if it's a RAW file
            if str(image_path).lower().endswith(('.nef', '.cr2', '.arw', '.dng', '.orf', '.rw2')):
                # Convert RAW to RGB
                image, _ = self.image_processor.convert_raw_to_rgb(str(image_path))
            else:
                # Load regular image
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply each operation in sequence
            for operation in recipe.operations:
                if operation.type == 'rotate':
                    angle = operation.parameters.get('angle', 0)
                    # Simple rotation
                    if angle != 0:
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        image = cv2.warpAffine(image, M, (w, h))
                    
                elif operation.type == 'crop':
                    p = operation.parameters
                    height, width = image.shape[:2]
                    x1 = int(p.get('x1', 0) * width)
                    y1 = int(p.get('y1', 0) * height)
                    x2 = int(p.get('x2', 1) * width)
                    y2 = int(p.get('y2', 1) * height)
                    image = image[y1:y2, x1:x2]
                    
                elif operation.type == 'enhance':
                    # Use the image processor's enhance method
                    from schemas import ColorAnalysis
                    color_analysis = ColorAnalysis(
                        dominant_colors=["blue", "white"],
                        exposure_assessment="properly_exposed",
                        white_balance_assessment="neutral",
                        contrast_level="normal",
                        brightness_adjustment_needed=int(operation.parameters.get('brightness', 0) * 100),
                        contrast_adjustment_needed=int(operation.parameters.get('contrast', 0) * 100)
                    )
                    image = self.image_processor.enhance_image(image, color_analysis)
            
            # Save processed image
            self.image_processor.save_high_quality_jpeg(
                image,
                str(output_path),
                quality=95
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply recipe: {e}")
            return False
    
    def process_single_file(self, file_path: Path) -> Optional[DualUploadResult]:
        """Process a single file with original preservation"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Calculate hash and check if already processed
            file_hash = self.calculate_file_hash(file_path)
            
            if self.hash_tracker.is_already_processed(file_path):
                logger.info(f"File already processed: {file_path}")
                # Remove the file from inbox since it's a duplicate
                file_path.unlink()
                return None
            
            # Step 2: Store original file (COPY, not move!)
            original_stored_path = self.store_original(file_path, file_hash)
            
            # Step 3: Create working copy for processing
            working_path = self.working_folder / f"{file_hash}_working{file_path.suffix}"
            shutil.copy2(original_stored_path, working_path)
            
            # Step 4: Create initial recipe
            recipe = self.create_processing_recipe(file_path, file_hash)
            
            # Step 5: Run AI analysis if enabled
            if self.enable_ai_processing:
                try:
                    # Convert RAW to RGB if needed
                    if self.image_processor.is_raw_file(str(working_path)):
                        rgb_path = working_path.with_suffix('.tiff')
                        if self.image_processor.convert_raw_to_rgb(
                            str(working_path), str(rgb_path)
                        ):
                            working_path = rgb_path
                    
                    # Run AI analysis
                    ai_results = self.ai_analyzer.analyze_photo(str(working_path))
                    
                    # Update recipe with AI results
                    recipe = self.create_processing_recipe(
                        file_path, file_hash, ai_results
                    )
                    
                except Exception as e:
                    logger.error(f"AI analysis failed: {e}")
                    # Continue with basic processing
            
            # Step 6: Apply processing recipe
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_filename = f"{file_path.stem}_processed_{timestamp}.jpg"
            processed_path = self.processed_folder / processed_filename
            
            if recipe.operations:
                # Apply recipe
                success = self.apply_recipe(
                    working_path,
                    recipe,
                    processed_path
                )
                if not success:
                    logger.error("Failed to apply recipe")
                    processed_path = working_path  # Use original as fallback
            else:
                # No operations, just convert to JPEG
                import cv2
                if str(working_path).lower().endswith(('.nef', '.cr2', '.arw', '.dng', '.orf', '.rw2')):
                    # Convert RAW to RGB
                    image, _ = self.image_processor.convert_raw_to_rgb(str(working_path))
                else:
                    # Load regular image
                    image = cv2.imread(str(working_path))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if image is not None:
                    self.image_processor.save_high_quality_jpeg(
                        image,
                        str(processed_path),
                        quality=95
                    )
            
            # Step 7: Save recipe
            self.recipe_storage.save_recipe(recipe)
            
            # Step 8: Upload BOTH files to Immich
            upload_result = self.immich_client.upload_photo_pair(
                original_path=original_stored_path,
                processed_path=processed_path,
                recipe=recipe,
                original_album="Original Files",
                processed_album="Processed Photos"
            )
            
            # Step 9: Track successful processing
            if upload_result.original.success:
                self.hash_tracker.mark_as_processed(str(file_path))
            
            # Step 10: Clean up
            # Remove working files
            working_path.unlink(missing_ok=True)
            if working_path.suffix != file_path.suffix:
                working_path.with_suffix(file_path.suffix).unlink(missing_ok=True)
            
            # Remove original from inbox (it's safely stored now)
            file_path.unlink()
            
            logger.info(f"Successfully processed {file_path.name}")
            return upload_result
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
            return None
    
    def scan_inbox(self):
        """Scan inbox for new files to process"""
        if not self.inbox_folder.exists():
            logger.warning(f"Inbox folder does not exist: {self.inbox_folder}")
            return
        
        # Get list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', 
                          '.nef', '.cr2', '.arw', '.dng', '.orf', '.rw2'}
        
        files_to_process = [
            f for f in self.inbox_folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if files_to_process:
            logger.info(f"Found {len(files_to_process)} files to process")
            
            for file_path in files_to_process:
                try:
                    self.process_single_file(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    
                # Small delay between files
                time.sleep(1)
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting Enhanced Photo Processor...")
        logger.info(f"Inbox folder: {self.inbox_folder}")
        logger.info(f"Originals folder: {self.originals_folder}")
        logger.info(f"Original preservation: ENABLED")
        logger.info(f"Dual upload: ENABLED")
        
        while True:
            try:
                self.scan_inbox()
                
                # Wait before next scan
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(30)  # Wait longer on error


if __name__ == "__main__":
    processor = EnhancedPhotoProcessor()
    processor.run()