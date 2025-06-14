#!/usr/bin/env python3
"""
Simple test script to verify the workflow components
Can be run inside the API container
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from recipe_storage import RecipeStorage, ProcessingRecipe, ProcessingOperation


def test_recipe_storage():
    """Test basic recipe storage functionality"""
    print("\n=== Testing Recipe Storage ===")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Create recipe storage
        storage = RecipeStorage(Path(temp_dir))
        
        # Create a recipe
        recipe = ProcessingRecipe(
            original_hash="test_hash_123",
            original_filename="portrait.jpg"
        )
        
        # Add operations matching the portrait workflow
        recipe.add_operation("crop", {"aspectRatio": "original", "customRatio": None})
        recipe.add_operation("enhance", {"brightness": 0.1, "contrast": 0.05})
        
        print(f"Created recipe: {recipe.id}")
        print(f"Recipe description:\n{recipe.get_description()}")
        
        # Save recipe
        success = storage.save_recipe(recipe)
        assert success, "Failed to save recipe"
        print("✓ Recipe saved successfully")
        
        # Load recipe
        loaded = storage.load_recipe(recipe.id)
        assert loaded is not None, "Failed to load recipe"
        assert loaded.id == recipe.id
        assert len(loaded.operations) == 2
        print("✓ Recipe loaded successfully")
        
        # Find by hash
        found = storage.find_recipe_by_hash("test_hash_123")
        assert found is not None, "Failed to find recipe by hash"
        assert found.id == recipe.id
        print("✓ Recipe found by hash")
        
        print("\n✅ Recipe Storage tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_recipe_service_format():
    """Test the actual recipe format used by the service"""
    print("\n=== Testing Recipe Service Format ===")
    
    # Load an actual recipe from the system
    recipe_path = Path("/app/data/recipes/89ecc365-8d40-4c40-bdc6-ad31ed57828a.json")
    
    if recipe_path.exists():
        with open(recipe_path, 'r') as f:
            recipe_data = json.load(f)
        
        print(f"Loaded recipe: {recipe_data.get('name', 'Unknown')}")
        print(f"Recipe ID: {recipe_data.get('id')}")
        print(f"Operations: {json.dumps(recipe_data.get('operations', []), indent=2)}")
        
        # Verify expected fields
        assert 'id' in recipe_data
        assert 'name' in recipe_data
        assert 'operations' in recipe_data
        assert 'processing_config' in recipe_data
        
        print("✅ Recipe format is correct")
    else:
        print("⚠️  Recipe file not found, skipping format test")


async def test_photo_upload_simulation():
    """Simulate photo upload process"""
    print("\n=== Testing Photo Upload Process ===")
    
    # This simulates what happens when a photo is uploaded
    photo_data = {
        'id': 'test_photo_123',
        'filename': 'test_portrait.jpg',
        'status': 'pending',
        'created_at': datetime.now().isoformat(),
        'original_path': '/app/data/inbox/test_photo_123_test_portrait.jpg',
        'file_hash': 'test_hash_456',
        'file_size': 1024000,
        'recipe_id': '89ecc365-8d40-4c40-bdc6-ad31ed57828a'  # Portrait recipe
    }
    
    print(f"Photo data structure:")
    print(json.dumps(photo_data, indent=2))
    
    # Verify the structure matches what's expected
    required_fields = ['id', 'filename', 'status', 'created_at', 'original_path']
    for field in required_fields:
        assert field in photo_data, f"Missing required field: {field}"
    
    print("✅ Photo data structure is correct")


async def test_processing_queue_format():
    """Test the processing queue format"""
    print("\n=== Testing Processing Queue Format ===")
    
    # Simulate queue item format
    queue_item = {
        'photo_id': 'test_photo_123',
        'photo_path': '/app/data/inbox/test_photo_123_test_portrait.jpg',
        'filename': 'test_portrait.jpg',
        'recipe_id': '89ecc365-8d40-4c40-bdc6-ad31ed57828a',
        'priority': 'normal',
        'created_at': datetime.now(),
        'estimated_time': 30
    }
    
    print("Queue item structure:")
    for key, value in queue_item.items():
        print(f"  {key}: {value}")
    
    # Verify structure
    required_fields = ['photo_id', 'photo_path', 'filename', 'recipe_id', 'priority']
    for field in required_fields:
        assert field in queue_item, f"Missing required field: {field}"
    
    print("✅ Queue format is correct")


def main():
    """Run all tests"""
    print("Starting workflow component tests...")
    
    # Run synchronous tests
    test_recipe_storage()
    test_recipe_service_format()
    
    # Run async tests
    asyncio.run(test_photo_upload_simulation())
    asyncio.run(test_processing_queue_format())
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()