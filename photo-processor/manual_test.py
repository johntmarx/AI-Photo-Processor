#!/usr/bin/env python3
"""
Manual test of the complete pipeline
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from ai_analyzer import AIAnalyzer
from immich_client import ImmichClient

def manual_pipeline_test():
    """Test the complete pipeline manually"""
    print("=== Manual Pipeline Test ===\n")
    
    # Initialize components
    processor = ImageProcessor()
    ai_analyzer = AIAnalyzer(os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
    immich_client = ImmichClient(
        os.getenv('IMMICH_API_URL', 'http://immich_server:2283'),
        os.getenv('IMMICH_API_KEY')
    )
    
    # Find a test file
    inbox_files = list(Path("/app/inbox").glob("*.ARW"))
    if not inbox_files:
        print("No RAW files found in inbox")
        return False
    
    test_file = str(inbox_files[0])
    filename = os.path.basename(test_file)
    print(f"Processing: {filename}")
    
    # Step 1: Convert RAW
    print("\n1. Converting RAW to RGB...")
    rgb_image, metadata = processor.convert_raw_to_rgb(test_file)
    print(f"   ✓ Converted: {metadata['width']}x{metadata['height']}")
    
    # Step 2: Prepare for AI
    print("\n2. Preparing for AI analysis...")
    ai_image = processor.resize_for_ai_analysis(rgb_image)
    temp_path = processor.save_temp_image_for_ai(ai_image)
    print(f"   ✓ Saved temp image: {temp_path}")
    
    # Step 3: AI Analysis
    print("\n3. Running AI analysis...")
    analysis = ai_analyzer.analyze_photo(temp_path, filename)
    
    if not analysis:
        print("   ✗ AI analysis failed")
        return False
    
    print(f"   ✓ Quality: {analysis.quality}")
    print(f"   ✓ Subject: {analysis.primary_subject}")
    print(f"   ✓ Processing: {analysis.processing_recommendation}")
    
    # Step 4: Process image
    print("\n4. Processing image...")
    if analysis.processing_recommendation == 'enhance_only':
        processed_image = processor.enhance_image(rgb_image, analysis.color_analysis)
        print("   ✓ Applied enhancement")
    elif analysis.processing_recommendation == 'crop_and_enhance':
        cropped = processor.apply_smart_crop(rgb_image, analysis.recommended_crop.crop_box)
        processed_image = processor.enhance_image(cropped, analysis.color_analysis)
        print("   ✓ Applied crop and enhancement")
    else:
        processed_image = rgb_image
        print("   ✓ No processing needed")
    
    # Step 5: Save processed image
    output_path = f"/app/processed/test_{filename.replace('.ARW', '.jpg')}"
    processor.save_high_quality_jpeg(processed_image, output_path)
    print(f"\n5. Saved processed image: {output_path}")
    
    # Step 6: Upload to Immich
    print("\n6. Uploading to Immich...")
    album_id = immich_client.get_or_create_album("AI Processed Photos")
    print(f"   Album ID: {album_id}")
    
    try:
        asset_id = immich_client.upload_photo(
            output_path,
            os.path.basename(output_path),
            analysis,
            album_id
        )
        
        if asset_id:
            print(f"   ✓ Upload successful! Asset ID: {asset_id}")
            return True
        else:
            print("   ✗ Upload failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Upload error: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass

if __name__ == "__main__":
    success = manual_pipeline_test()
    print("\n" + "="*50)
    print(f"Pipeline test: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)