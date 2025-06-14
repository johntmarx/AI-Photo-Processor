#!/usr/bin/env python3
"""Test photo upload and NIMA analysis"""

import requests
import time
import sys

def upload_photo(file_path):
    """Upload a photo and monitor NIMA analysis"""
    
    # Upload the photo
    url = "http://localhost:8000/api/upload/single"
    
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.split('/')[-1], f, 'image/jpeg')}
        data = {'auto_process': 'true'}
        
        print(f"Uploading {file_path}...")
        response = requests.post(url, files=files, data=data)
        
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return None
        
    result = response.json()
    photo_id = result.get('photo_id')
    print(f"Photo uploaded successfully: {photo_id}")
    
    # Wait a bit for processing
    print("Waiting for NIMA analysis...")
    time.sleep(5)
    
    # Check AI analysis
    ai_url = f"http://localhost:8000/api/photos/{photo_id}/ai-analysis"
    ai_response = requests.get(ai_url)
    
    if ai_response.status_code == 200:
        ai_data = ai_response.json()
        if ai_data.get('status') == 'completed':
            print(f"\nNIMA Analysis Results:")
            print(f"- Aesthetic Score: {ai_data.get('aesthetic_score', 'N/A'):.2f}/10")
            tech_score = ai_data.get('technical_score')
            if tech_score is not None:
                print(f"- Technical Score: {tech_score:.2f}/10")
            else:
                print(f"- Technical Score: N/A")
            print(f"- Quality Level: {ai_data.get('quality_level', 'N/A')}")
            print(f"- Confidence: {ai_data.get('confidence', 'N/A'):.3f}")
        else:
            print(f"AI analysis status: {ai_data.get('status')}")
    else:
        print(f"Failed to get AI analysis: {ai_response.text}")
    
    return photo_id

if __name__ == "__main__":
    # Use a test image
    test_image = "/home/john/immich/photo-processor/test-data/test_image.jpg"
    
    # Create a test image if it doesn't exist
    import os
    if not os.path.exists(test_image):
        print("Creating test image...")
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_image, 'JPEG')
    
    # Upload and test
    upload_photo(test_image)