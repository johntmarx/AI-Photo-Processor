#!/usr/bin/env python3
"""
Simple test script to verify rotation analysis works
"""

import sys
import os
sys.path.append('/home/john/immich/photo-processor')

import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

def rotate_and_crop_image(image: Image.Image, angle: float) -> Image.Image:
    """Simple rotation with crop"""
    if abs(angle) < 0.01:
        return image
    
    # Rotate with high quality resampling - handle PIL version differences
    try:
        # Try modern PIL approach first
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        # Fallback for older PIL versions
        resample = Image.BICUBIC
    
    rotated = image.rotate(
        angle,
        resample=resample,
        expand=True,
        fillcolor=(0, 0, 0)
    )
    
    # Simple center crop to remove black borders
    original_width, original_height = image.size
    rotated_width, rotated_height = rotated.size
    
    # Calculate crop area (simple approach)
    crop_width = min(original_width, rotated_width)
    crop_height = min(original_height, rotated_height)
    
    left = (rotated_width - crop_width) // 2
    top = (rotated_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return rotated.crop((left, top, right, bottom))

def test_rotation_basic():
    """Test basic rotation without OneAlign"""
    print("Testing basic rotation functionality...")
    
    # Find a test image - try different paths
    possible_paths = [
        "/home/john/immich/photo-processor/data/inbox/0012a6ed-5675-4674-841c-ac603012e195_IMG_2561.jpeg",
        "/app/data/inbox/0012a6ed-5675-4674-841c-ac603012e195_IMG_2561.jpeg"
    ]
    
    test_image_path = None
    for path in possible_paths:
        if Path(path).exists():
            test_image_path = path
            break
    
    if not test_image_path:
        print(f"ERROR: Test image not found in any location")
        # List available images in both possible directories
        for inbox_path in ["/home/john/immich/photo-processor/data/inbox", "/app/data/inbox"]:
            inbox_dir = Path(inbox_path)
        if inbox_dir.exists():
            images = list(inbox_dir.glob("*.jpeg"))[:3]
            print(f"Available images: {[img.name for img in images]}")
            if images:
                test_image_path = str(images[0])
                print(f"Using: {test_image_path}")
            else:
                print("No JPEG images found in inbox")
                return False
        else:
            print("Inbox directory not found")
            return False
    
    try:
        # Load image
        print(f"Loading image: {test_image_path}")
        source_image = Image.open(test_image_path)
        if source_image.mode not in ('RGB', 'RGBA'):
            source_image = source_image.convert('RGB')
        
        print(f"Original image size: {source_image.size}")
        
        # Test rotation angles
        angles = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="rotation_test_"))
        print(f"Temp directory: {temp_dir}")
        
        try:
            for angle in angles:
                print(f"Testing rotation: {angle}°")
                
                # Rotate image
                rotated = rotate_and_crop_image(source_image, angle)
                print(f"  Rotated size: {rotated.size}")
                
                # Save rotated image
                output_path = temp_dir / f"rotated_{angle:+05.1f}deg.jpg"
                rotated.save(output_path, format='JPEG', quality=90)
                print(f"  Saved: {output_path}")
            
            print(f"\n✓ Basic rotation test PASSED")
            print(f"Generated {len(angles)} rotated images in {temp_dir}")
            return True
            
        finally:
            # Clean up
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory")
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_dir}: {e}")
        
    except Exception as e:
        print(f"ERROR: Basic rotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onealign_import():
    """Test if OneAlign can be imported"""
    print("\nTesting OneAlign import...")
    
    try:
        # Add paths
        sys.path.append('/home/john/immich/photo-processor/ai-components')
        
        from ai_components.onealign.onealign_model import OneAlign
        print("✓ OneAlign imported successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: OneAlign import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple Rotation Analysis Test ===\n")
    
    # Test 1: Basic rotation
    basic_ok = test_rotation_basic()
    
    # Test 2: OneAlign import
    import_ok = test_onealign_import()
    
    print(f"\n=== Results ===")
    print(f"Basic rotation: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"OneAlign import: {'✓ PASS' if import_ok else '✗ FAIL'}")
    
    if basic_ok and import_ok:
        print("\n✓ Core functionality works - can proceed with integration")
    else:
        print("\n✗ Core issues need to be fixed first")