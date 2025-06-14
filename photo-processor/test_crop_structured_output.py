#!/usr/bin/env python3
"""
Test script to verify the crop VLM structured output is working correctly
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ai_components.qwen25vl.qwen25vl_ollama import Qwen25VLOllama
from PIL import Image
import json


async def test_crop_suggestions():
    """Test the suggest_crops method with structured output"""
    
    # Initialize the model
    print("Initializing Qwen2.5-VL model...")
    qwen = Qwen25VLOllama(
        ollama_host="http://ollama:11434",
        model="qwen2.5vl:7b"
    )
    
    # Test with a sample image
    test_image_path = "/home/john/immich/photo-processor/test-data/test_image.jpg"
    
    if not Path(test_image_path).exists():
        print(f"Test image not found at {test_image_path}")
        # Create a simple test image
        print("Creating a test image...")
        test_image = Image.new('RGB', (800, 600), color='blue')
        test_image_path = "/tmp/test_crop_image.jpg"
        test_image.save(test_image_path)
    
    print(f"\nTesting crop suggestions for: {test_image_path}")
    print("-" * 60)
    
    try:
        # Get crop suggestions
        crop_analysis = await qwen.suggest_crops(image=test_image_path)
        
        print(f"Original aspect ratio: {crop_analysis.original_aspect:.2f}")
        print(f"Number of suggestions: {len(crop_analysis.crop_suggestions)}")
        print(f"Primary subject preserved: {crop_analysis.primary_subject_preserved}")
        print(f"Notes: {crop_analysis.notes}")
        print("\nCrop Suggestions:")
        print("-" * 60)
        
        for i, crop in enumerate(crop_analysis.crop_suggestions):
            print(f"\n{i+1}. {crop.name}")
            print(f"   Description: {crop.description}")
            print(f"   Aspect Ratio: {crop.aspect_ratio}")
            print(f"   Emphasis: {crop.emphasis}")
            print(f"   Impact Score: {crop.impact_score}")
            
        # Verify the main crop has concise instructions
        main_crop = crop_analysis.crop_suggestions[0]
        print("\n" + "="*60)
        print("MAIN CROP ANALYSIS:")
        print("="*60)
        print(f"Photo Description (emphasis): {main_crop.emphasis}")
        print(f"Crop Instructions: {main_crop.description}")
        
        # Check if the output is concise
        emphasis_words = len(main_crop.emphasis.split())
        description_sentences = main_crop.description.count('.') or 1
        
        print(f"\nConciseness Check:")
        print(f"- Photo description word count: {emphasis_words}")
        print(f"- Crop instructions sentence count: {description_sentences}")
        
        if emphasis_words > 20:
            print("⚠️  WARNING: Photo description is too long!")
        else:
            print("✓ Photo description is concise")
            
        if description_sentences > 2:
            print("⚠️  WARNING: Crop instructions have too many sentences!")
        else:
            print("✓ Crop instructions are concise")
            
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_crop_suggestions())