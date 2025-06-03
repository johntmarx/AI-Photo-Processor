#!/usr/bin/env python3
"""
Test the improved AI photo processor with better cropping and color correction
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor_v2 import ImageProcessor
from ai_analyzer import AIAnalyzer
from immich_client import ImmichClient
import numpy as np
from PIL import Image

def compare_processing():
    """Compare original vs enhanced processing"""
    print("=== Testing Improved Photo Processing ===\n")
    
    # Initialize components
    processor = ImageProcessor()
    ai_analyzer = AIAnalyzer(os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
    
    # Find a test file
    test_files = list(Path("/app/inbox").glob("*.ARW"))
    if not test_files:
        print("No RAW files found")
        return
    
    test_file = str(test_files[0])
    filename = os.path.basename(test_file)
    print(f"Testing with: {filename}")
    
    # Step 1: Convert RAW
    print("\n1. Converting RAW file...")
    start = time.time()
    rgb_image, metadata = processor.convert_raw_to_rgb(test_file)
    print(f"   ✓ Converted in {time.time() - start:.2f}s: {metadata['width']}x{metadata['height']}")
    
    # Step 2: AI Analysis
    print("\n2. Running AI analysis with improved prompting...")
    ai_image = processor.resize_for_ai_analysis(rgb_image)
    temp_path = processor.save_temp_image_for_ai(ai_image)
    
    start = time.time()
    analysis = ai_analyzer.analyze_photo(temp_path, filename)
    print(f"   ✓ Analysis completed in {time.time() - start:.2f}s")
    
    if not analysis:
        print("   ✗ AI analysis failed")
        return
    
    # Print analysis results
    print(f"\n3. AI Analysis Results:")
    print(f"   - Quality: {analysis.quality}")
    print(f"   - Subject: {analysis.primary_subject}")
    print(f"   - Subject Box: x={analysis.primary_subject_box.x:.1f}%, y={analysis.primary_subject_box.y:.1f}%, "
          f"w={analysis.primary_subject_box.width:.1f}%, h={analysis.primary_subject_box.height:.1f}%")
    print(f"   - Recommended Crop: x={analysis.recommended_crop.crop_box.x:.1f}%, y={analysis.recommended_crop.crop_box.y:.1f}%, "
          f"w={analysis.recommended_crop.crop_box.width:.1f}%, h={analysis.recommended_crop.crop_box.height:.1f}%")
    print(f"   - Crop Aspect: {analysis.recommended_crop.aspect_ratio}")
    print(f"   - Orientation: {analysis.orientation}")
    print(f"   - White Balance: {analysis.color_analysis.white_balance_assessment}")
    print(f"   - Exposure: {analysis.color_analysis.exposure_assessment}")
    print(f"   - Brightness Adj: {analysis.color_analysis.brightness_adjustment_needed}")
    print(f"   - Contrast Adj: {analysis.color_analysis.contrast_adjustment_needed}")
    print(f"   - Processing: {analysis.processing_recommendation}")
    
    # Step 4: Process with improvements
    print(f"\n4. Processing with enhanced algorithms...")
    
    # Save original for comparison
    original_8bit = (rgb_image / 256).astype(np.uint8) if rgb_image.dtype == np.uint16 else rgb_image
    Image.fromarray(original_8bit).save(f"/app/processed/original_{filename.replace('.ARW', '.jpg')}", quality=95)
    
    # Apply cropping if recommended
    if analysis.processing_recommendation in ['crop_and_enhance', 'crop_only']:
        print("   - Applying tight crop...")
        cropped = processor.apply_smart_crop(rgb_image, analysis.recommended_crop.crop_box)
        # Calculate actual crop size
        crop_height, crop_width = cropped.shape[:2]
        print(f"   ✓ Cropped to {crop_width}x{crop_height} ({analysis.recommended_crop.aspect_ratio})")
    else:
        cropped = rgb_image
    
    # Apply enhancement
    print("   - Applying advanced color correction...")
    start = time.time()
    enhanced = processor.enhance_image(cropped, analysis.color_analysis)
    print(f"   ✓ Enhanced in {time.time() - start:.2f}s")
    
    # Save processed image
    output_path = f"/app/processed/enhanced_{filename.replace('.ARW', '.jpg')}"
    processor.save_high_quality_jpeg(enhanced, output_path)
    
    # Calculate file sizes
    original_size = os.path.getsize(f"/app/processed/original_{filename.replace('.ARW', '.jpg')}")
    enhanced_size = os.path.getsize(output_path)
    
    print(f"\n5. Results:")
    print(f"   - Original size: {original_size / 1024 / 1024:.1f} MB")
    print(f"   - Enhanced size: {enhanced_size / 1024 / 1024:.1f} MB")
    print(f"   - Files saved to /app/processed/")
    
    # Cleanup
    try:
        os.remove(temp_path)
    except:
        pass
    
    print("\n" + "="*50)
    print("Test complete! Check the processed folder for results.")

if __name__ == "__main__":
    compare_processing()