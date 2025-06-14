#!/usr/bin/env python3
"""
Test script to verify the enhancement functionality works for both RAW and JPEG files
"""

import os
import sys
import logging
from pathlib import Path

# Add the API directory to the path
sys.path.insert(0, '/app/api')

from services.intelligent_enhancer_v2 import ModernIntelligentEnhancer, EnhancementMode, EnhancementSettings
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_test_images(directory="/app/data/inbox"):
    """Find JPEG and RAW files for testing"""
    jpeg_files = []
    raw_files = []
    
    # Common image extensions
    jpeg_extensions = {'.jpg', '.jpeg', '.png'}
    raw_extensions = {'.arw', '.cr2', '.nef', '.dng', '.raf', '.orf', '.rw2', '.cr3'}
    
    try:
        for file in Path(directory).iterdir():
            if file.is_file():
                ext = file.suffix.lower()
                if ext in jpeg_extensions:
                    jpeg_files.append(str(file))
                elif ext in raw_extensions:
                    raw_files.append(str(file))
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
    
    return jpeg_files, raw_files

def test_enhancement(image_path):
    """Test enhancement on a single image"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing enhancement on: {image_path}")
    logger.info(f"File size: {os.path.getsize(image_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Initialize enhancer
        enhancer = ModernIntelligentEnhancer()
        
        # Test with default intelligent settings
        logger.info("Testing with default intelligent enhancement (strength=1.0)...")
        settings = enhancer.create_default_intelligent_settings(strength=1.0)
        
        original, enhanced = enhancer.get_enhancement_preview(
            image_path,
            EnhancementMode.INTELLIGENT,
            settings
        )
        
        logger.info(f"Original size: {original.size}")
        logger.info(f"Enhanced size: {enhanced.size}")
        
        # Save the enhanced preview
        output_path = f"/app/data/temp/test_enhanced_{Path(image_path).stem}.jpg"
        enhanced.save(output_path, 'JPEG', quality=95)
        logger.info(f"Saved enhanced preview to: {output_path}")
        
        # Test with custom settings (individual controls)
        logger.info("\nTesting with custom settings...")
        custom_settings = EnhancementSettings(
            white_balance=True,
            white_balance_strength=0.8,
            exposure=True,
            exposure_strength=1.0,
            contrast=True,
            contrast_strength=0.7,
            vibrance=True,
            vibrance_strength=1.2,
            shadow_highlight=True,
            shadow_highlight_strength=0.9,
            overall_strength=0.9
        )
        
        original2, enhanced2 = enhancer.get_enhancement_preview(
            image_path,
            EnhancementMode.CUSTOM,
            custom_settings
        )
        
        output_path2 = f"/app/data/temp/test_enhanced_custom_{Path(image_path).stem}.jpg"
        enhanced2.save(output_path2, 'JPEG', quality=95)
        logger.info(f"Saved custom enhanced preview to: {output_path2}")
        
        logger.info("✅ Enhancement test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("Starting enhancement functionality test...")
    
    # Create temp directory if it doesn't exist
    os.makedirs("/app/data/temp", exist_ok=True)
    
    # Find test images
    jpeg_files, raw_files = find_test_images()
    
    logger.info(f"\nFound {len(jpeg_files)} JPEG files and {len(raw_files)} RAW files")
    
    if not jpeg_files and not raw_files:
        logger.error("No test images found in /app/data/inbox")
        return
    
    # Test JPEG files
    if jpeg_files:
        logger.info("\n" + "="*60)
        logger.info("TESTING JPEG FILES")
        logger.info("="*60)
        for jpeg_file in jpeg_files[:2]:  # Test first 2 JPEG files
            test_enhancement(jpeg_file)
    
    # Test RAW files
    if raw_files:
        logger.info("\n" + "="*60)
        logger.info("TESTING RAW FILES")
        logger.info("="*60)
        for raw_file in raw_files[:2]:  # Test first 2 RAW files
            test_enhancement(raw_file)
    
    logger.info("\n" + "="*60)
    logger.info("Enhancement functionality test completed!")
    logger.info("Check /app/data/temp/ for output files")

if __name__ == "__main__":
    main()