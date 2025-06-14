#!/usr/bin/env python3
"""
Debug test to find what's taking so long
"""

import os
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, '/app/api')

from services.intelligent_enhancer_v2 import ModernIntelligentEnhancer, EnhancementMode, EnhancementSettings
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_raw_processing():
    """Test RAW file processing speed"""
    # Find an ARW file
    arw_file = None
    for f in Path("/app/data/inbox").glob("*.ARW"):
        arw_file = str(f)
        break
    
    if not arw_file:
        logger.error("No ARW file found")
        return
    
    logger.info(f"Testing with RAW file: {arw_file}")
    
    # Test 1: Just opening the RAW file
    start = time.time()
    logger.info("Step 1: Opening RAW file with rawpy...")
    
    import rawpy
    with rawpy.imread(arw_file) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8
        )
        img = Image.fromarray(rgb)
    
    elapsed = time.time() - start
    logger.info(f"RAW file opened and converted in {elapsed:.2f} seconds")
    logger.info(f"Image size: {img.size}")
    
    # Test 2: Enhancement with minimal settings
    start = time.time()
    logger.info("\nStep 2: Testing minimal enhancement...")
    
    enhancer = ModernIntelligentEnhancer()
    
    # Disable most features
    minimal_settings = EnhancementSettings(
        white_balance=True,
        white_balance_strength=1.0,
        exposure=False,
        contrast=False,
        vibrance=False,
        detail_recovery=False,
        sharpening=False,
        noise_reduction=False,
        overall_strength=1.0
    )
    
    enhanced = enhancer.enhance_image(img, EnhancementMode.CUSTOM, minimal_settings)
    elapsed = time.time() - start
    logger.info(f"Minimal enhancement took {elapsed:.2f} seconds")
    
    # Test 3: Add features one by one
    features = [
        ('exposure', 'exposure'),
        ('contrast', 'contrast'), 
        ('vibrance', 'vibrance'),
        ('detail_recovery', 'detail_recovery'),
        ('sharpening', 'sharpening'),
        ('noise_reduction', 'noise_reduction')
    ]
    
    for feature_name, attr_name in features:
        start = time.time()
        logger.info(f"\nStep 3: Testing with {feature_name} enabled...")
        
        setattr(minimal_settings, attr_name, True)
        enhanced = enhancer.enhance_image(img, EnhancementMode.CUSTOM, minimal_settings)
        
        elapsed = time.time() - start
        logger.info(f"Enhancement with {feature_name} took {elapsed:.2f} seconds")
        
        # Reset for next test
        setattr(minimal_settings, attr_name, False)

if __name__ == "__main__":
    test_raw_processing()