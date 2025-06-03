#!/usr/bin/env python3
"""
Test script to verify each component of the photo processing pipeline
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor
from ai_analyzer import AIAnalyzer
from immich_client import ImmichClient
from schemas import BoundingBox, ColorAnalysis

def test_1_environment():
    """Test 1: Verify environment variables and connections"""
    print("=== TEST 1: Environment Check ===")
    
    ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
    immich_api_url = os.getenv('IMMICH_API_URL', 'http://immich_server:2283')
    immich_api_key = os.getenv('IMMICH_API_KEY')
    
    print(f"Ollama Host: {ollama_host}")
    print(f"Immich API URL: {immich_api_url}")
    print(f"Immich API Key: {'Set' if immich_api_key else 'NOT SET'}")
    
    return immich_api_key is not None

def test_2_ai_connection():
    """Test 2: Test AI analyzer connection"""
    print("\n=== TEST 2: AI Analyzer Connection ===")
    
    ai = AIAnalyzer(os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
    connected = ai.test_connection()
    print(f"Ollama Connection: {'SUCCESS' if connected else 'FAILED'}")
    
    if connected:
        model_available = ai.ensure_model_available()
        print(f"Model Available: {'YES' if model_available else 'NO'}")
        return model_available
    
    return False

def test_3_image_processing():
    """Test 3: Test image processing on a test file"""
    print("\n=== TEST 3: Image Processing ===")
    
    test_file = "/app/inbox/DSC09497.ARW"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        # List available files
        inbox_files = list(Path("/app/inbox").glob("*.ARW"))
        if inbox_files:
            test_file = str(inbox_files[0])
            print(f"Using alternative test file: {test_file}")
        else:
            print("No ARW files found in inbox")
            return False
    
    processor = ImageProcessor()
    
    try:
        # Test RAW conversion
        print(f"Converting RAW file: {test_file}")
        rgb_image, metadata = processor.convert_raw_to_rgb(test_file)
        print(f"Converted successfully: {metadata['width']}x{metadata['height']}")
        
        # Test resize for AI
        ai_image = processor.resize_for_ai_analysis(rgb_image)
        print(f"Resized for AI: {ai_image.shape}")
        
        # Test enhancement
        color_analysis = ColorAnalysis(
            dominant_colors=["blue", "white"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="normal",
            brightness_adjustment_needed=5,
            contrast_adjustment_needed=10
        )
        enhanced = processor.enhance_image(rgb_image, color_analysis)
        print("Enhancement applied successfully")
        
        return True
        
    except Exception as e:
        print(f"Image processing failed: {e}")
        return False

def test_4_immich_connection():
    """Test 4: Test Immich API connection"""
    print("\n=== TEST 4: Immich Connection ===")
    
    client = ImmichClient(
        os.getenv('IMMICH_API_URL', 'http://immich_server:2283'),
        os.getenv('IMMICH_API_KEY')
    )
    
    connected = client.test_connection()
    print(f"Immich Connection: {'SUCCESS' if connected else 'FAILED'}")
    
    if connected:
        # Test album creation/retrieval
        album_id = client.get_or_create_album("AI Processed Photos - Test")
        print(f"Album ID: {album_id if album_id else 'FAILED TO CREATE'}")
        return album_id is not None
    
    return False

def test_5_upload_format():
    """Test 5: Test the upload format requirements"""
    print("\n=== TEST 5: Upload Format Test ===")
    
    # Create test data matching API requirements
    current_time = datetime.now().isoformat()
    
    test_data = {
        'deviceAssetId': 'test-asset-123',
        'deviceId': 'ai-photo-processor',
        'fileCreatedAt': current_time,
        'fileModifiedAt': current_time,
        'isFavorite': False  # Boolean, not string!
    }
    
    print("Test upload data:")
    print(json.dumps(test_data, indent=2))
    
    # Verify date format
    try:
        datetime.fromisoformat(test_data['fileCreatedAt'])
        print("✓ Date format is valid ISO format")
    except:
        print("✗ Date format is invalid")
        return False
    
    # Verify boolean type
    if isinstance(test_data['isFavorite'], bool):
        print("✓ isFavorite is boolean")
    else:
        print("✗ isFavorite should be boolean, not string")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Photo Processor Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        test_1_environment,
        test_2_ai_connection,
        test_3_image_processing,
        test_4_immich_connection,
        test_5_upload_format
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if not all(results):
        print("\nFailed tests need to be fixed before the pipeline can work properly.")
    else:
        print("\nAll tests passed! Pipeline should work correctly.")

if __name__ == "__main__":
    main()