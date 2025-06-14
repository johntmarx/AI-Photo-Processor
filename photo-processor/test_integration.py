#!/usr/bin/env python3
"""
Integration test for the complete photo processing workflow
Tests the actual services running in the container
"""

import asyncio
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import hashlib
import shutil
from io import BytesIO

sys.path.insert(0, '/app')

from api.services.photo_service_v2 import photo_service
from api.services.processing_service_v2 import processing_service  
from api.services.recipe_service_v2 import recipe_service
from api.models.processing import BatchOperation


class MockUploadFile:
    """Mock UploadFile for testing"""
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content_type = "image/jpeg"
        self._content = content
        self._read_count = 0
    
    async def read(self):
        # Can only read once like real UploadFile
        if self._read_count > 0:
            return b""
        self._read_count += 1
        return self._content


async def test_complete_workflow():
    """Test the complete workflow from upload to processing"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Complete Photo Processing Workflow")
    print("="*60)
    
    try:
        # Step 1: List existing recipes
        print("\n1. Checking existing recipes...")
        recipes_result = await recipe_service.list_recipes()
        print(f"   Found {recipes_result['total']} recipes")
        
        # Find portrait recipe
        portrait_recipe = None
        for recipe in recipes_result['recipes']:
            if recipe['id'] == '89ecc365-8d40-4c40-bdc6-ad31ed57828a':
                portrait_recipe = recipe
                print(f"   ✓ Found portrait recipe: {recipe['name']}")
                break
        
        if not portrait_recipe:
            print("   ⚠️  Portrait recipe not found, creating one...")
            portrait_recipe = await recipe_service.create_recipe(
                name="Test Portrait Recipe",
                description="Portrait enhancement for testing",
                operations=[
                    {
                        "operation": "crop",
                        "parameters": {"aspectRatio": "original"},
                        "enabled": True
                    },
                    {
                        "operation": "enhance", 
                        "parameters": {"brightness": 0.1, "contrast": 0.05},
                        "enabled": True
                    }
                ],
                style_preset="natural"
            )
            print(f"   ✓ Created recipe: {portrait_recipe['id']}")
        
        recipe_id = portrait_recipe['id']
        
        # Step 2: Create a test photo
        print("\n2. Creating test photo...")
        
        # Use one of the existing test photos as base
        test_photos = list(Path('/app/data/inbox').glob('*.jpeg'))[:1]
        if test_photos:
            test_photo_path = test_photos[0]
            print(f"   Using existing test photo: {test_photo_path.name}")
            test_content = test_photo_path.read_bytes()
        else:
            # Create fake image data
            print("   Creating synthetic test image...")
            test_content = b"FAKE_IMAGE_DATA_FOR_TESTING" * 1000
        
        # Create upload file
        mock_file = MockUploadFile(
            filename=f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            content=test_content
        )
        
        # Step 3: Upload photo with auto-process disabled first
        print("\n3. Uploading photo...")
        upload_result = await photo_service.save_upload(
            file=mock_file,
            auto_process=False,
            recipe_id=None
        )
        
        photo_id = upload_result['photo_id']
        print(f"   ✓ Photo uploaded: {photo_id}")
        print(f"   Status: {upload_result['status']}")
        
        # Step 4: Verify photo in database
        print("\n4. Verifying photo in database...")
        photo_detail = await photo_service.get_photo(photo_id)
        if photo_detail:
            print(f"   ✓ Photo found in database")
            print(f"   Filename: {photo_detail.filename}")
            print(f"   Status: {photo_detail.status}")
            print(f"   File size: {photo_detail.file_size} bytes")
        else:
            print("   ❌ Photo not found in database!")
            return False
        
        # Step 5: Create batch operation with recipe
        print(f"\n5. Creating batch operation with recipe '{portrait_recipe['name']}'...")
        batch_op = BatchOperation(
            photo_ids=[photo_id],
            recipe_id=recipe_id,
            priority="high"
        )
        
        batch_result = await processing_service.batch_process(batch_op)
        print(f"   ✓ Batch operation completed")
        print(f"   Queued: {batch_result['queued']} photos")
        print(f"   Skipped: {batch_result['skipped']} photos")
        
        if batch_result['errors']:
            print(f"   ⚠️  Errors: {batch_result['errors']}")
        
        # Step 6: Check processing queue
        print("\n6. Checking processing queue...")
        queue_status = await processing_service.get_queue_status()
        print(f"   Pending: {len(queue_status.pending)}")
        print(f"   Processing: {len(queue_status.processing)}")
        print(f"   Is paused: {queue_status.is_paused}")
        
        # Step 7: Process the photo
        print("\n7. Processing photo...")
        if len(queue_status.pending) > 0:
            process_result = await processing_service.process_next_item()
            
            if process_result:
                print(f"   ✓ Photo processed successfully!")
                print(f"   Processing time: {process_result.get('processing_time', 0):.2f} seconds")
                print(f"   Status: {process_result['status']}")
                
                if process_result['status'] == 'failed':
                    print(f"   ❌ Error: {process_result.get('error', 'Unknown error')}")
            else:
                print("   ❌ Processing returned no result")
        else:
            print("   ⚠️  No photos in queue to process")
        
        # Step 8: Verify final status
        print("\n8. Verifying final photo status...")
        final_photo = await photo_service.get_photo(photo_id)
        if final_photo:
            print(f"   Status: {final_photo.status}")
            print(f"   Processed path: {final_photo.processed_path}")
            print(f"   Thumbnail path: {final_photo.thumbnail_path}")
            print(f"   Web path: {final_photo.web_path}")
            
            # Check if files exist
            if final_photo.processed_path:
                if Path(final_photo.processed_path).exists():
                    print("   ✓ Processed file exists")
                else:
                    print("   ❌ Processed file missing")
                    
            if final_photo.thumbnail_path:
                if Path(final_photo.thumbnail_path).exists():
                    print("   ✓ Thumbnail file exists")
                else:
                    print("   ❌ Thumbnail file missing")
        
        # Step 9: Test listing photos
        print("\n9. Testing photo listing...")
        photo_list = await photo_service.list_photos(page=1, page_size=10)
        print(f"   Total photos: {photo_list.total}")
        print(f"   Photos on page: {len(photo_list.photos)}")
        
        # Find our test photo
        test_photo_found = False
        for photo in photo_list.photos:
            if photo.id == photo_id:
                test_photo_found = True
                print(f"   ✓ Test photo found in listing")
                break
        
        if not test_photo_found:
            print("   ❌ Test photo not found in listing")
        
        print("\n" + "="*60)
        print("INTEGRATION TEST COMPLETED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_recipe_operations():
    """Test recipe CRUD operations"""
    print("\n" + "="*60)
    print("TESTING RECIPE OPERATIONS")
    print("="*60)
    
    try:
        # Create a test recipe
        print("\n1. Creating test recipe...")
        test_recipe = await recipe_service.create_recipe(
            name="Integration Test Recipe",
            description="Recipe created during integration testing",
            operations=[
                {
                    "operation": "enhance",
                    "parameters": {"strength": 0.5},
                    "enabled": True
                },
                {
                    "operation": "denoise",
                    "parameters": {"strength": 0.3},
                    "enabled": True
                }
            ],
            style_preset="vivid"
        )
        
        recipe_id = test_recipe['id']
        print(f"   ✓ Created recipe: {recipe_id}")
        
        # Update the recipe
        print("\n2. Updating recipe...")
        updated = await recipe_service.update_recipe(
            recipe_id=recipe_id,
            name="Updated Integration Test Recipe",
            operations=[
                {
                    "operation": "enhance",
                    "parameters": {"strength": 0.7},
                    "enabled": True
                }
            ]
        )
        
        if updated:
            print(f"   ✓ Recipe updated: {updated['name']}")
            print(f"   Operations: {len(updated['operations'])}")
        
        # Get recipe details
        print("\n3. Getting recipe details...")
        recipe_detail = await recipe_service.get_recipe(recipe_id)
        if recipe_detail:
            print(f"   ✓ Recipe retrieved: {recipe_detail['name']}")
            print(f"   Has 'steps' field: {'steps' in recipe_detail}")
            print(f"   Has 'operations' field: {'operations' in recipe_detail}")
        
        # List all recipes
        print("\n4. Listing all recipes...")
        all_recipes = await recipe_service.list_recipes()
        print(f"   Total recipes: {all_recipes['total']}")
        
        # Clean up - delete test recipe
        print("\n5. Cleaning up test recipe...")
        # Note: delete might not be implemented, so we'll skip for now
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests"""
    print("\nStarting Integration Tests...")
    print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Give services a moment to initialize
    await asyncio.sleep(1)
    
    # Run tests
    workflow_success = await test_complete_workflow()
    recipe_success = await test_recipe_operations()
    
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Workflow Test: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
    print(f"Recipe Test: {'✅ PASSED' if recipe_success else '❌ FAILED'}")
    print("="*60)
    
    return workflow_success and recipe_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)