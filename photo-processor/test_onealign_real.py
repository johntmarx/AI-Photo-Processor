#!/usr/bin/env python3
"""
Test OneAlign with a real image to see actual output
"""

import sys
import os

# Set up Python path
sys.path.append('/home/john/immich/photo-processor')
sys.path.append('/home/john/immich/photo-processor/ai_components')

# Set cache directory
os.environ['HF_HOME'] = '/app/model_cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'

try:
    # Import required libraries
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    from transformers import AutoModelForCausalLM
    print("Transformers imported successfully")
    
    from PIL import Image
    print("PIL imported successfully")
    
    # Test with a real image
    test_image_path = "/app/data/processed/0b1902cb-e715-4454-9766-edfdb6fd0242_IMG_2610.jpeg"
    
    if os.path.exists(test_image_path):
        print(f"\nTesting with real image: {test_image_path}")
        
        # Load model
        print("\nLoading OneAlign model from Hugging Face...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            torch_dtype = torch.float16
            device_map = 'auto'
        else:
            torch_dtype = torch.float32
            device_map = 'cpu'
            
        model = AutoModelForCausalLM.from_pretrained(
            "q-future/one-align",
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir="/app/model_cache"
        )
        
        print("Model loaded successfully!")
        
        # Load image
        image = Image.open(test_image_path)
        print(f"Image loaded: {image.size} {image.mode}")
        
        # Test quality assessment
        print("\n=== Testing Quality Assessment ===")
        quality_score = model.score([image], task_="quality", input_="image")
        print(f"Quality score: {quality_score}")
        print(f"Type: {type(quality_score)}")
        
        # Test aesthetics assessment
        print("\n=== Testing Aesthetics Assessment ===")
        aesthetics_score = model.score([image], task_="aesthetics", input_="image")
        print(f"Aesthetics score: {aesthetics_score}")
        print(f"Type: {type(aesthetics_score)}")
        
        # Try different variations to understand the output
        print("\n=== Testing with different parameters ===")
        
        # Try single image without list
        try:
            single_score = model.score(image, task_="quality", input_="image")
            print(f"Single image score: {single_score}")
        except Exception as e:
            print(f"Single image failed: {e}")
        
        # Check if it returns multiple values
        print("\n=== Checking return structure ===")
        result = model.score([image], task_="quality", input_="image")
        if isinstance(result, (list, tuple)):
            print(f"Result is a {type(result).__name__} with {len(result)} items")
            for i, item in enumerate(result):
                print(f"  Item {i}: {item} (type: {type(item)})")
        else:
            print(f"Result is a single value: {result} (type: {type(result)})")
            
    else:
        print(f"Test image not found: {test_image_path}")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure to run this inside the Docker container with:")
    print("docker exec -it photo_processor_celery_ai_worker python /home/john/immich/photo-processor/test_onealign_real.py")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()