#!/usr/bin/env python3
"""
Test the simplified core enhancement features
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, '/app/api')

from services.intelligent_enhancer_v2 import ModernIntelligentEnhancer, EnhancementMode, EnhancementSettings
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_core_enhancement():
    """Test core enhancement features only"""
    # Find a test image
    test_image = None
    for ext in ['.jpg', '.jpeg', '.png', '.arw', '.cr2']:
        for f in Path("/app/data/inbox").glob(f"*{ext}"):
            test_image = str(f)
            break
        if test_image:
            break
    
    if not test_image:
        logger.error("No test image found")
        return
    
    logger.info(f"Testing with image: {test_image}")
    
    # Initialize enhancer
    enhancer = ModernIntelligentEnhancer()
    
    # Test each feature individually
    features = [
        ("White Balance Only", EnhancementSettings(
            white_balance=True, white_balance_strength=1.0,
            exposure=False, contrast=False, vibrance=False, shadow_highlight=False
        )),
        ("Exposure Only", EnhancementSettings(
            white_balance=False, exposure=True, exposure_strength=1.0,
            contrast=False, vibrance=False, shadow_highlight=False
        )),
        ("Contrast Only", EnhancementSettings(
            white_balance=False, exposure=False, 
            contrast=True, contrast_strength=1.0,
            vibrance=False, shadow_highlight=False
        )),
        ("Vibrance Only", EnhancementSettings(
            white_balance=False, exposure=False, contrast=False,
            vibrance=True, vibrance_strength=1.0,
            shadow_highlight=False
        )),
        ("Shadow/Highlight Only", EnhancementSettings(
            white_balance=False, exposure=False, contrast=False, vibrance=False,
            shadow_highlight=True, shadow_highlight_strength=1.0
        )),
        ("All Features Combined", enhancer.create_default_intelligent_settings(strength=1.0))
    ]
    
    # Create output directory
    output_dir = Path("/app/data/temp/core_enhancement_test")
    output_dir.mkdir(exist_ok=True)
    
    # Test each feature
    for feature_name, settings in features:
        logger.info(f"\nTesting: {feature_name}")
        
        try:
            original, enhanced = enhancer.get_enhancement_preview(
                test_image,
                EnhancementMode.CUSTOM,
                settings
            )
            
            # Save the result
            output_path = output_dir / f"{Path(test_image).stem}_{feature_name.lower().replace(' ', '_')}.jpg"
            enhanced.save(output_path, 'JPEG', quality=95)
            logger.info(f"✅ Saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
    
    # Save original for comparison
    original_path = output_dir / f"{Path(test_image).stem}_original.jpg"
    original.save(original_path, 'JPEG', quality=95)
    logger.info(f"\nOriginal saved to: {original_path}")
    
    logger.info(f"\n✨ All tests completed! Check {output_dir} for results.")

if __name__ == "__main__":
    test_core_enhancement()