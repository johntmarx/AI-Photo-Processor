#!/usr/bin/env python3
"""
Test the complete frontend workflow using API endpoints
Simulates what the React frontend would do
"""

import requests
import json
import time
import sys
from datetime import datetime


BASE_URL = "http://localhost:8000"


def test_frontend_workflow():
    """Test the complete workflow as the frontend would use it"""
    print("\n" + "="*60)
    print("TESTING FRONTEND WORKFLOW")
    print("="*60)
    
    # 1. Get list of photos
    print("\n1. Getting photo list...")
    response = requests.get(f"{BASE_URL}/api/photos")
    if response.status_code == 200:
        photos = response.json()
        print(f"   ✓ Found {photos['total']} photos")
        
        # Find unprocessed photos
        pending_photos = [p for p in photos['photos'] if p['status'] == 'pending']
        if pending_photos:
            print(f"   Found {len(pending_photos)} pending photos")
        else:
            print("   No pending photos found")
            # Get any photo for testing
            if photos['photos']:
                test_photo = photos['photos'][0]
                print(f"   Using photo: {test_photo['filename']} (status: {test_photo['status']})")
    else:
        print(f"   ❌ Failed to get photos: {response.status_code}")
        return False
    
    # 2. Get available recipes
    print("\n2. Getting recipe list...")
    response = requests.get(f"{BASE_URL}/api/recipes")
    if response.status_code == 200:
        recipes = response.json()
        print(f"   ✓ Found {recipes['total']} recipes")
        
        # Find portrait recipe
        portrait_recipe = None
        for recipe in recipes['recipes']:
            print(f"   - {recipe['name']} (ID: {recipe['id'][:8]}...)")
            if 'portrait' in recipe['name'].lower():
                portrait_recipe = recipe
                
        if portrait_recipe:
            print(f"   ✓ Using recipe: {portrait_recipe['name']}")
        else:
            print("   ⚠️  No portrait recipe found")
            if recipes['recipes']:
                portrait_recipe = recipes['recipes'][0]
                print(f"   Using first recipe: {portrait_recipe['name']}")
    else:
        print(f"   ❌ Failed to get recipes: {response.status_code}")
        return False
    
    # 3. Check processing status
    print("\n3. Checking processing status...")
    response = requests.get(f"{BASE_URL}/api/processing/status")
    if response.status_code == 200:
        status = response.json()
        print(f"   ✓ Processing status:")
        print(f"     - Is paused: {status['is_paused']}")
        print(f"     - Queue length: {status['queue_length']}")
        print(f"     - Processing rate: {status['processing_rate']:.2f} photos/hour")
        print(f"     - Errors today: {status['errors_today']}")
    else:
        print(f"   ❌ Failed to get status: {response.status_code}")
    
    # 4. Check queue
    print("\n4. Checking processing queue...")
    response = requests.get(f"{BASE_URL}/api/processing/queue")
    if response.status_code == 200:
        queue = response.json()
        print(f"   ✓ Queue status:")
        print(f"     - Pending: {len(queue['pending'])}")
        print(f"     - Processing: {len(queue['processing'])}")
        print(f"     - Completed: {len(queue['completed'])}")
    else:
        print(f"   ❌ Failed to get queue: {response.status_code}")
    
    # 5. Test WebSocket connection (simulate)
    print("\n5. WebSocket connection test...")
    print("   ℹ️  WebSocket connection would be established here")
    print("   ℹ️  Frontend would receive real-time updates")
    
    # 6. Upload a new photo (simulate)
    print("\n6. Photo upload simulation...")
    print("   ℹ️  Frontend would use FormData to upload files")
    print("   ℹ️  POST /api/upload with multipart/form-data")
    
    # 7. Batch process photos
    if photos['photos']:
        print("\n7. Testing batch processing...")
        # Take up to 3 photos for testing
        test_photo_ids = [p['id'] for p in photos['photos'][:3]]
        
        batch_data = {
            "photo_ids": test_photo_ids,
            "recipe_id": portrait_recipe['id'],
            "priority": "normal",
            "skip_ai": False
        }
        
        print(f"   Sending batch request for {len(test_photo_ids)} photos...")
        response = requests.post(
            f"{BASE_URL}/api/processing/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Batch processing result:")
            print(f"     - Queued: {result['queued']}")
            print(f"     - Skipped: {result['skipped']}")
            if result.get('errors'):
                print(f"     - Errors: {result['errors']}")
        else:
            print(f"   ❌ Batch processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    # 8. Check image serving
    if photos['photos']:
        print("\n8. Testing image serving...")
        test_photo = photos['photos'][0]
        
        # Test thumbnail
        if test_photo.get('thumbnail_path'):
            url = f"{BASE_URL}/images/thumbnails/{test_photo['id']}_thumb.jpg"
            response = requests.head(url)
            if response.status_code == 200:
                print(f"   ✓ Thumbnail accessible: {url}")
            else:
                print(f"   ❌ Thumbnail not accessible: {response.status_code}")
        
        # Test web version
        if test_photo.get('web_path'):
            url = f"{BASE_URL}/images/web/{test_photo['id']}_web.jpg"
            response = requests.head(url)
            if response.status_code == 200:
                print(f"   ✓ Web version accessible: {url}")
            else:
                print(f"   ❌ Web version not accessible: {response.status_code}")
    
    # 9. Test recipe operations
    print("\n9. Testing recipe operations...")
    
    # Create a recipe
    new_recipe = {
        "name": f"Frontend Test Recipe {datetime.now().strftime('%H:%M:%S')}",
        "description": "Created by frontend workflow test",
        "operations": [
            {
                "operation": "enhance",
                "parameters": {"strength": 0.5},
                "enabled": True
            }
        ],
        "style_preset": "natural",
        "is_default": False
    }
    
    response = requests.post(
        f"{BASE_URL}/api/recipes",
        json=new_recipe,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        created_recipe = response.json()
        print(f"   ✓ Created recipe: {created_recipe['name']}")
        
        # Update the recipe
        update_data = {
            "name": created_recipe['name'] + " (Updated)",
            "operations": [
                {
                    "operation": "enhance",
                    "parameters": {"strength": 0.7},
                    "enabled": True
                },
                {
                    "operation": "denoise",
                    "parameters": {"strength": 0.3},
                    "enabled": True
                }
            ]
        }
        
        response = requests.put(
            f"{BASE_URL}/api/recipes/{created_recipe['id']}",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"   ✓ Updated recipe successfully")
        else:
            print(f"   ❌ Failed to update recipe: {response.status_code}")
    else:
        print(f"   ❌ Failed to create recipe: {response.status_code}")
    
    print("\n" + "="*60)
    print("FRONTEND WORKFLOW TEST COMPLETED")
    print("="*60)
    
    return True


def test_api_health():
    """Test basic API health"""
    print("\nTesting API health...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False


def main():
    """Run all frontend workflow tests"""
    print("Starting Frontend Workflow Tests...")
    print(f"API URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API health first
    if not test_api_health():
        print("\nAPI is not accessible. Make sure the containers are running.")
        return False
    
    # Run workflow tests
    success = test_frontend_workflow()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Frontend Workflow: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)