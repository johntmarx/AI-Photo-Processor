#!/usr/bin/env python3
"""
Test the AI analyzer directly to see raw responses
"""
import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_analyzer import AIAnalyzer
from image_processor import ImageProcessor
from pathlib import Path

# Set up debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_ai_response():
    """Test AI analyzer and print raw response"""
    print("=== Testing AI Analyzer Raw Response ===\n")
    
    # Initialize
    ai = AIAnalyzer(os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
    processor = ImageProcessor()
    
    # Find test file
    test_files = list(Path("/app/inbox").glob("*.ARW"))
    if not test_files:
        print("No test files found")
        return
    
    test_file = str(test_files[0])
    filename = os.path.basename(test_file)
    print(f"Testing with: {filename}")
    
    # Prepare image
    print("\nPreparing image for AI...")
    rgb_image, metadata = processor.convert_raw_to_rgb(test_file)
    ai_image = processor.resize_for_ai_analysis(rgb_image)
    temp_path = processor.save_temp_image_for_ai(ai_image)
    
    # Get AI analysis
    print("\nCalling AI analyzer...")
    analysis = ai.analyze_photo(temp_path, filename)
    
    if analysis:
        print("\n=== AI Analysis Results ===")
        print(f"Quality: {analysis.quality}")
        print(f"Subject: {analysis.primary_subject}")
        print(f"Subject Box: {analysis.primary_subject_box}")
        print(f"Crop Box: {analysis.recommended_crop.crop_box}")
        print(f"Processing: {analysis.processing_recommendation}")
        print(f"Orientation: {analysis.orientation}")
        
        # Print full JSON
        print("\n=== Full Analysis JSON ===")
        print(json.dumps(analysis.model_dump(), indent=2))
    else:
        print("Analysis failed!")
    
    # Cleanup
    try:
        os.remove(temp_path)
    except:
        pass

if __name__ == "__main__":
    test_ai_response()