"""
Unit tests for Recipe Storage System
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from recipe_storage import ProcessingOperation, ProcessingRecipe, RecipeStorage


class TestProcessingOperation:
    """Test ProcessingOperation class"""
    
    def test_create_operation(self):
        """Test creating a processing operation"""
        op = ProcessingOperation(
            type='crop',
            parameters={'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9},
            source='user'
        )
        
        assert op.type == 'crop'
        assert op.parameters['x1'] == 0.1
        assert op.source == 'user'
        assert isinstance(op.timestamp, datetime)
    
    def test_operation_to_dict(self):
        """Test converting operation to dictionary"""
        op = ProcessingOperation(
            type='rotate',
            parameters={'angle': 90},
            source='ai'
        )
        
        data = op.to_dict()
        assert data['type'] == 'rotate'
        assert data['parameters']['angle'] == 90
        assert data['source'] == 'ai'
        assert 'timestamp' in data
    
    def test_operation_from_dict(self):
        """Test creating operation from dictionary"""
        data = {
            'type': 'enhance',
            'parameters': {'brightness': 0.1, 'contrast': 0.2},
            'timestamp': datetime.now().isoformat(),
            'source': 'preset'
        }
        
        op = ProcessingOperation.from_dict(data)
        assert op.type == 'enhance'
        assert op.parameters['brightness'] == 0.1
        assert op.parameters['contrast'] == 0.2
        assert op.source == 'preset'


class TestProcessingRecipe:
    """Test ProcessingRecipe class"""
    
    def test_create_recipe(self):
        """Test creating a processing recipe"""
        recipe = ProcessingRecipe(
            original_hash='abc123',
            original_filename='test.jpg'
        )
        
        assert recipe.original_hash == 'abc123'
        assert recipe.original_filename == 'test.jpg'
        assert recipe.version == 1
        assert len(recipe.operations) == 0
        assert isinstance(recipe.id, str)
        assert len(recipe.id) > 0
    
    def test_add_operation(self):
        """Test adding operations to recipe"""
        recipe = ProcessingRecipe()
        
        recipe.add_operation('crop', {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        recipe.add_operation('rotate', {'angle': 45}, source='user')
        
        assert len(recipe.operations) == 2
        assert recipe.operations[0].type == 'crop'
        assert recipe.operations[1].type == 'rotate'
        assert recipe.operations[1].source == 'user'
    
    def test_recipe_to_json(self):
        """Test converting recipe to JSON"""
        recipe = ProcessingRecipe(
            original_hash='hash123',
            original_filename='photo.jpg'
        )
        recipe.add_operation('enhance', {'brightness': 0.5})
        recipe.ai_metadata = {'model': 'test', 'score': 0.9}
        
        json_str = recipe.to_json()
        data = json.loads(json_str)
        
        assert data['original_hash'] == 'hash123'
        assert data['original_filename'] == 'photo.jpg'
        assert len(data['operations']) == 1
        assert data['operations'][0]['type'] == 'enhance'
        assert data['ai_metadata']['model'] == 'test'
    
    def test_recipe_from_json(self):
        """Test creating recipe from JSON"""
        recipe1 = ProcessingRecipe(
            original_hash='xyz789',
            original_filename='image.raw'
        )
        recipe1.add_operation('crop', {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9})
        recipe1.add_operation('rotate', {'angle': -5})
        recipe1.ai_metadata = {'detected_objects': ['cat', 'dog']}
        
        json_str = recipe1.to_json()
        recipe2 = ProcessingRecipe.from_json(json_str)
        
        assert recipe2.id == recipe1.id
        assert recipe2.original_hash == 'xyz789'
        assert len(recipe2.operations) == 2
        assert recipe2.operations[0].type == 'crop'
        assert recipe2.operations[1].parameters['angle'] == -5
        assert 'cat' in recipe2.ai_metadata['detected_objects']
    
    def test_get_description(self):
        """Test generating human-readable description"""
        recipe = ProcessingRecipe(
            original_filename='sunset.jpg'
        )
        recipe.add_operation('crop', {'x1': 0.1, 'y1': 0.2, 'x2': 0.9, 'y2': 0.8})
        recipe.add_operation('rotate', {'angle': 15})
        recipe.add_operation('enhance', {'brightness': 0.1, 'contrast': -0.2})
        
        description = recipe.get_description()
        
        assert 'sunset.jpg' in description
        assert 'Crop to' in description
        assert 'Rotate 15°' in description
        assert 'brightness=+0.10' in description
        assert 'contrast=-0.20' in description


class TestRecipeStorage:
    """Test RecipeStorage class"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_init_storage(self, temp_storage):
        """Test initializing recipe storage"""
        storage = RecipeStorage(temp_storage)
        
        assert storage.storage_path.exists()
        assert storage.index_file.exists()
        assert isinstance(storage.index, dict)
    
    def test_save_and_load_recipe(self, temp_storage):
        """Test saving and loading a recipe"""
        storage = RecipeStorage(temp_storage)
        
        # Create recipe
        recipe = ProcessingRecipe(
            original_hash='test_hash',
            original_filename='test.jpg'
        )
        recipe.add_operation('crop', {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        
        # Save recipe
        success = storage.save_recipe(recipe)
        assert success
        
        # Load recipe
        loaded = storage.load_recipe(recipe.id)
        assert loaded is not None
        assert loaded.id == recipe.id
        assert loaded.original_hash == 'test_hash'
        assert len(loaded.operations) == 1
        assert loaded.operations[0].type == 'crop'
    
    def test_find_recipe_by_hash(self, temp_storage):
        """Test finding recipe by file hash"""
        storage = RecipeStorage(temp_storage)
        
        # Save multiple recipes for same file
        recipe1 = ProcessingRecipe(
            original_hash='same_hash',
            original_filename='photo.jpg',
            version=1
        )
        recipe1.add_operation('crop', {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9})
        
        # Simulate later version
        recipe2 = ProcessingRecipe(
            original_hash='same_hash',
            original_filename='photo.jpg',
            version=2
        )
        recipe2.add_operation('crop', {'x1': 0.2, 'y1': 0.2, 'x2': 0.8, 'y2': 0.8})
        
        storage.save_recipe(recipe1)
        storage.save_recipe(recipe2)
        
        # Should find most recent version
        found = storage.find_recipe_by_hash('same_hash')
        assert found is not None
        assert found.version == 2
        assert found.operations[0].parameters['x1'] == 0.2
    
    def test_list_recipes(self, temp_storage):
        """Test listing recipes"""
        storage = RecipeStorage(temp_storage)
        
        # Save multiple recipes
        recipes = []
        for i in range(5):
            recipe = ProcessingRecipe(
                original_hash=f'hash_{i}',
                original_filename=f'photo_{i}.jpg'
            )
            storage.save_recipe(recipe)
            recipes.append(recipe)
        
        # List recipes
        listed = storage.list_recipes(limit=3)
        assert len(listed) == 3
        
        # Should be sorted by creation date (most recent first)
        assert all('original_hash' in r for r in listed)
    
    def test_index_persistence(self, temp_storage):
        """Test that index persists across storage instances"""
        # First instance
        storage1 = RecipeStorage(temp_storage)
        recipe = ProcessingRecipe(
            original_hash='persist_test',
            original_filename='persist.jpg'
        )
        storage1.save_recipe(recipe)
        
        # Second instance should load existing index
        storage2 = RecipeStorage(temp_storage)
        assert recipe.id in storage2.index
        assert storage2.index[recipe.id]['original_hash'] == 'persist_test'
        
        # Should be able to load the recipe
        loaded = storage2.load_recipe(recipe.id)
        assert loaded is not None
        assert loaded.original_hash == 'persist_test'
    
    def test_complex_recipe_roundtrip(self, temp_storage):
        """Test saving and loading a complex recipe"""
        storage = RecipeStorage(temp_storage)
        
        # Create complex recipe
        recipe = ProcessingRecipe(
            original_hash='complex_hash',
            original_filename='complex.raw'
        )
        
        # Add multiple operations
        recipe.add_operation('crop', {
            'x1': 0.15, 'y1': 0.25, 'x2': 0.85, 'y2': 0.75,
            'aspect_ratio': '16:9'
        })
        recipe.add_operation('rotate', {'angle': -3.5, 'auto_crop': True})
        recipe.add_operation('enhance', {
            'brightness': 0.1,
            'contrast': 0.15,
            'saturation': -0.05,
            'highlights': -0.2,
            'shadows': 0.1
        })
        
        # Add metadata
        recipe.ai_metadata = {
            'model': 'gemma3:4b',
            'analysis': {
                'subjects': ['landscape', 'sunset'],
                'quality_score': 8.5,
                'suggested_tags': ['nature', 'golden hour']
            }
        }
        
        recipe.user_overrides = {
            'crop_approved': True,
            'custom_note': 'Beautiful sunset shot'
        }
        
        # Save and reload
        storage.save_recipe(recipe)
        loaded = storage.load_recipe(recipe.id)
        
        # Verify everything was preserved
        assert loaded.original_hash == 'complex_hash'
        assert len(loaded.operations) == 3
        assert loaded.operations[0].parameters['aspect_ratio'] == '16:9'
        assert loaded.operations[1].parameters['angle'] == -3.5
        assert loaded.operations[2].parameters['brightness'] == 0.1
        assert loaded.ai_metadata['analysis']['quality_score'] == 8.5
        assert loaded.user_overrides['custom_note'] == 'Beautiful sunset shot'