#!/usr/bin/env python3
"""
Test script to verify the model-agnostic crop suggestion works with different models
"""

import asyncio
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ai_components.ollama_vlm import OllamaVLM
from ai_components.shared.ollama_vlm_base import OllamaVLMBase
from PIL import Image
import json


async def test_crop_with_model(model_name: str, test_image_path: str):
    """Test crop suggestions with a specific model"""
    
    print(f"\n{'='*60}")
    print(f"Testing with model: {model_name}")
    print('='*60)
    
    try:
        # Initialize the model
        vlm = OllamaVLM(
            ollama_host="http://ollama:11434",
            model=model_name
        )
        
        # Get crop suggestions
        crop_analysis = await vlm.suggest_crops(image=test_image_path)
        
        print(f"✓ Model: {model_name}")
        print(f"✓ Original aspect ratio: {crop_analysis.original_aspect:.2f}")
        print(f"✓ Number of suggestions: {len(crop_analysis.crop_suggestions)}")
        print(f"✓ Notes: {crop_analysis.notes}")
        
        # Check the main crop suggestion
        if crop_analysis.crop_suggestions:
            main_crop = crop_analysis.crop_suggestions[0]
            print(f"\nMain Crop Suggestion:")
            print(f"  Photo Description: {main_crop.emphasis}")
            print(f"  Instructions: {main_crop.description}")
            print(f"  Aspect Ratio: {main_crop.aspect_ratio}")
            
            # Verify conciseness
            emphasis_words = len(main_crop.emphasis.split())
            description_sentences = main_crop.description.count('.') or 1
            
            print(f"\nConciseness Check:")
            print(f"  - Description words: {emphasis_words} {'✓' if emphasis_words <= 20 else '✗'}")
            print(f"  - Instruction sentences: {description_sentences} {'✓' if description_sentences <= 2 else '✗'}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error with {model_name}: {type(e).__name__}: {e}")
        return False


async def test_all_models():
    """Test crop suggestions with different models"""
    
    # Test image setup
    test_image_path = "/home/john/immich/photo-processor/test-data/test_image.jpg"
    
    if not Path(test_image_path).exists():
        print(f"Test image not found at {test_image_path}")
        # Create a simple test image
        print("Creating a test image...")
        test_image = Image.new('RGB', (800, 600), color='blue')
        # Add some variation
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([200, 150, 600, 450], fill='red')
        draw.ellipse([350, 250, 450, 350], fill='yellow')
        test_image_path = "/tmp/test_model_agnostic.jpg"
        test_image.save(test_image_path)
    
    # List of models to test
    models_to_test = [
        "qwen2.5vl:7b",      # Qwen 2.5 Vision Language
        "gemma3:12b",        # Gemma 3 (if supports vision)
        "llava:latest",      # LLaVA
        # Add more models as needed
    ]
    
    # Test with environment variable
    env_model = os.getenv("OLLAMA_MODEL") or os.getenv("VLM_MODEL")
    if env_model and env_model not in models_to_test:
        models_to_test.insert(0, env_model)
    
    print("Testing Model-Agnostic Crop Suggestions")
    print(f"Test image: {test_image_path}")
    
    results = []
    for model in models_to_test:
        success = await test_crop_with_model(model, test_image_path)
        results.append((model, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for model, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{model:20} {status}")
    
    # Test direct OllamaVLMBase usage
    print(f"\n{'='*60}")
    print("Testing Direct OllamaVLMBase Usage")
    print('='*60)
    
    base_vlm = OllamaVLMBase(model=models_to_test[0])
    crop_result = await base_vlm.suggest_crops(test_image_path)
    print(f"✓ Direct base class usage works")
    print(f"✓ Got {len(crop_result.crop_suggestions)} suggestions")


if __name__ == "__main__":
    asyncio.run(test_all_models())