"""
Comprehensive test suite for photo processing workflow
Tests the complete pipeline from upload to processing with recipes
"""

import asyncio
import pytest
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import sys
import hashlib
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from recipe_storage import RecipeStorage, ProcessingRecipe, ProcessingOperation
from api.services.photo_service_v2 import EnhancedPhotoService
from api.services.processing_service_v2 import EnhancedProcessingService
from api.services.recipe_service_v2 import EnhancedRecipeService
from api.models.photo import Photo, PhotoDetail
from api.models.processing import BatchOperation


class TestRecipeStorage:
    """Test recipe storage functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = RecipeStorage(Path(self.temp_dir))
        
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_and_save_recipe(self):
        """Test creating and saving a recipe"""
        recipe = ProcessingRecipe(
            original_hash="test_hash_123",
            original_filename="test_photo.jpg"
        )
        
        # Add operations
        recipe.add_operation("crop", {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9})
        recipe.add_operation("enhance", {"brightness": 0.1, "contrast": 0.05})
        recipe.add_operation("rotate", {"angle": 90})
        
        # Save recipe
        assert self.storage.save_recipe(recipe)
        
        # Verify file was created
        recipe_file = Path(self.temp_dir) / f"{recipe.id}.json"
        assert recipe_file.exists()
        
        # Verify index was updated
        assert recipe.id in self.storage.index
        assert self.storage.index[recipe.id]['original_hash'] == "test_hash_123"
        assert self.storage.index[recipe.id]['operations_count'] == 3
    
    def test_load_recipe(self):
        """Test loading a saved recipe"""
        # Create and save recipe
        original_recipe = ProcessingRecipe(
            original_hash="test_hash_456",
            original_filename="portrait.jpg"
        )
        original_recipe.add_operation("enhance", {"brightness": 0.2})
        self.storage.save_recipe(original_recipe)
        
        # Load recipe
        loaded_recipe = self.storage.load_recipe(original_recipe.id)
        
        assert loaded_recipe is not None
        assert loaded_recipe.id == original_recipe.id
        assert loaded_recipe.original_hash == "test_hash_456"
        assert loaded_recipe.original_filename == "portrait.jpg"
        assert len(loaded_recipe.operations) == 1
        assert loaded_recipe.operations[0].type == "enhance"
        assert loaded_recipe.operations[0].parameters["brightness"] == 0.2
    
    def test_find_recipe_by_hash(self):
        """Test finding recipes by file hash"""
        hash_value = "duplicate_hash_789"
        
        # Create multiple recipes for same hash
        recipe1 = ProcessingRecipe(original_hash=hash_value, original_filename="photo1.jpg")
        recipe1.add_operation("enhance", {"brightness": 0.1})
        self.storage.save_recipe(recipe1)
        
        # Create newer recipe
        recipe2 = ProcessingRecipe(original_hash=hash_value, original_filename="photo1.jpg")
        recipe2.add_operation("enhance", {"brightness": 0.2})
        recipe2.version = 2
        self.storage.save_recipe(recipe2)
        
        # Find recipe should return the newest one
        found_recipe = self.storage.find_recipe_by_hash(hash_value)
        assert found_recipe is not None
        assert found_recipe.version == 2
        assert found_recipe.operations[0].parameters["brightness"] == 0.2
    
    def test_recipe_description(self):
        """Test recipe description generation"""
        recipe = ProcessingRecipe(
            original_hash="desc_hash",
            original_filename="landscape.jpg"
        )
        
        recipe.add_operation("crop", {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9})
        recipe.add_operation("rotate", {"angle": -5})
        recipe.add_operation("enhance", {"brightness": 0.1, "contrast": 0.2, "saturation": -0.1})
        
        description = recipe.get_description()
        
        assert "Processing Recipe v1" in description
        assert "landscape.jpg" in description
        assert "Crop to (0.1, 0.1, 0.9, 0.9)" in description
        assert "Rotate -5°" in description
        assert "brightness=+0.10" in description
        assert "contrast=+0.20" in description
        assert "saturation=-0.10" in description


class TestPhotoService:
    """Test photo service functionality"""
    
    @pytest.fixture
    def photo_service(self, tmp_path):
        """Create photo service with temporary paths"""
        with patch('api.services.photo_service_v2.Path') as mock_path:
            # Mock all the paths to use tmp_path
            mock_path.return_value = tmp_path
            service = EnhancedPhotoService()
            service.data_path = tmp_path
            service.inbox_path = tmp_path / "inbox"
            service.originals_path = tmp_path / "originals"
            service.processed_path = tmp_path / "processed"
            service.recipes_path = tmp_path / "recipes"
            
            # Create directories
            for path in [service.inbox_path, service.originals_path, 
                        service.processed_path, service.recipes_path]:
                path.mkdir(parents=True, exist_ok=True)
                
            # Create photos.json
            (tmp_path / "photos.json").write_text('{"photos": {}}')
            
            return service
    
    @pytest.mark.asyncio
    async def test_save_upload(self, photo_service):
        """Test saving uploaded photo"""
        # Create mock upload file
        mock_file = AsyncMock()
        mock_file.filename = "test_portrait.jpg"
        mock_file.content_type = "image/jpeg"
        
        test_content = b"fake image content"
        mock_file.read.return_value = test_content
        
        # Save upload without auto-processing
        result = await photo_service.save_upload(
            file=mock_file,
            auto_process=False,
            recipe_id=None
        )
        
        assert result["status"] == "completed"
        assert result["filename"] == "test_portrait.jpg"
        assert "photo_id" in result
        
        # Verify file was saved
        saved_files = list(photo_service.inbox_path.glob("*test_portrait.jpg"))
        assert len(saved_files) == 1
        
        # Verify database entry
        photo_data = photo_service.db.get_photo(result["photo_id"])
        assert photo_data is not None
        assert photo_data["filename"] == "test_portrait.jpg"
        assert photo_data["status"] == "completed"
        assert photo_data["file_size"] == len(test_content)
    
    @pytest.mark.asyncio  
    async def test_list_photos_pagination(self, photo_service):
        """Test listing photos with pagination"""
        # Add multiple test photos to database
        for i in range(25):
            photo_data = {
                'id': f'test_photo_{i}',
                'filename': f'photo_{i}.jpg',
                'status': 'completed' if i % 2 == 0 else 'pending',
                'created_at': datetime.now(),
                'file_size': 1000 + i * 100
            }
            photo_service.db.add_photo(photo_data)
        
        # Test first page
        result = await photo_service.list_photos(page=1, page_size=10)
        assert len(result.photos) == 10
        assert result.total == 25
        assert result.has_next == True
        assert result.has_prev == False
        
        # Test second page
        result = await photo_service.list_photos(page=2, page_size=10)
        assert len(result.photos) == 10
        assert result.has_next == True
        assert result.has_prev == True
        
        # Test filtering by status
        result = await photo_service.list_photos(status='completed')
        completed_photos = [p for p in result.photos if p.status == 'completed']
        assert len(completed_photos) == len(result.photos)
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, photo_service):
        """Test duplicate photo detection"""
        # Create mock file
        mock_file = AsyncMock()
        mock_file.filename = "duplicate.jpg"
        mock_file.content_type = "image/jpeg"
        
        test_content = b"duplicate image content"
        mock_file.read.return_value = test_content
        
        # First upload
        result1 = await photo_service.save_upload(mock_file, auto_process=False)
        assert result1["status"] == "completed"
        
        # Reset mock for second read
        mock_file.read.return_value = test_content
        
        # Second upload of same file
        result2 = await photo_service.save_upload(mock_file, auto_process=False)
        assert result2["status"] == "duplicate"
        assert "already processed" in result2["message"]


class TestProcessingService:
    """Test processing service functionality"""
    
    @pytest.fixture
    def processing_service(self, tmp_path):
        """Create processing service with temporary paths"""
        with patch('api.services.processing_service_v2.Path') as mock_path:
            mock_path.return_value = tmp_path
            service = EnhancedProcessingService()
            service.output_dir = tmp_path / "processed"
            service.thumbnail_dir = tmp_path / "thumbnails"
            service.web_dir = tmp_path / "web"
            
            # Create directories
            for dir_path in [service.output_dir, service.thumbnail_dir, service.web_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                
            return service
    
    @pytest.mark.asyncio
    async def test_queue_photo_processing(self, processing_service):
        """Test queuing photos for processing"""
        photo_id = "test_123"
        photo_path = Path("/tmp/test_photo.jpg")
        
        result = await processing_service.queue_photo_processing(
            photo_id=photo_id,
            photo_path=photo_path,
            recipe_id="portrait",
            priority="high"
        )
        
        assert result["status"] == "queued"
        assert result["photo_id"] == photo_id
        assert len(processing_service.processing_queue) == 1
        
        # Check high priority puts it at front
        queue_item = processing_service.processing_queue[0]
        assert queue_item["photo_id"] == photo_id
        assert queue_item["priority"] == "high"
        assert queue_item["recipe_id"] == "portrait"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processing_service, photo_service):
        """Test batch processing multiple photos"""
        # Create test photos in database
        photo_ids = []
        for i in range(3):
            photo_data = {
                'id': f'batch_photo_{i}',
                'filename': f'batch_{i}.jpg',
                'status': 'pending',
                'created_at': datetime.now(),
                'original_path': f'/tmp/batch_{i}.jpg',
                'file_size': 1000
            }
            photo_service.db.add_photo(photo_data)
            photo_ids.append(photo_data['id'])
            
            # Create dummy file
            Path(photo_data['original_path']).touch()
        
        # Create batch operation
        batch_op = BatchOperation(
            photo_ids=photo_ids,
            recipe_id="portrait",
            priority="normal"
        )
        
        # Mock the photo service in processing service
        with patch('api.services.processing_service_v2.photo_service', photo_service):
            result = await processing_service.batch_process(batch_op)
        
        assert result["queued"] == 3
        assert result["skipped"] == 0
        assert len(processing_service.processing_queue) == 3
    
    @pytest.mark.asyncio
    async def test_processing_with_recipe(self, processing_service, tmp_path):
        """Test processing a photo with a recipe"""
        # Create test photo
        test_photo = tmp_path / "test_input.jpg"
        test_photo.write_bytes(b"fake image data")
        
        # Create recipe storage with portrait recipe
        recipe_storage = RecipeStorage(tmp_path / "recipes")
        recipe = ProcessingRecipe(
            id="portrait",
            original_hash="test_hash",
            original_filename="portrait.jpg"
        )
        recipe.add_operation("enhance", {"brightness": 0.1})
        recipe_storage.save_recipe(recipe)
        
        # Mock recipe storage in processing service
        processing_service.recipe_storage = recipe_storage
        
        # Queue photo
        await processing_service.queue_photo_processing(
            photo_id="process_test",
            photo_path=test_photo,
            recipe_id="portrait",
            priority="normal"
        )
        
        # Mock photo service update
        with patch.object(processing_service, 'ws_manager', None):
            with patch('api.services.processing_service_v2.photo_service') as mock_photo_service:
                mock_db = Mock()
                mock_db.update_photo = Mock()
                mock_photo_service.db = mock_db
                
                # Process the photo
                result = await processing_service.process_next_item()
        
        assert result is not None
        assert result["status"] == "completed"
        assert result["photo_id"] == "process_test"
        assert "processing_time" in result
        
        # Verify output files were created
        processed_file = processing_service.output_dir / "test_input.jpg"
        assert processed_file.exists()


class TestRecipeService:
    """Test recipe service functionality"""
    
    @pytest.fixture
    def recipe_service(self, tmp_path):
        """Create recipe service with temporary path"""
        with patch('api.services.recipe_service_v2.Path') as mock_path:
            mock_path.return_value = tmp_path
            service = EnhancedRecipeService()
            service.recipes_path = tmp_path / "recipes"
            service.recipes_path.mkdir(parents=True, exist_ok=True)
            
            # Create initial index
            index_file = service.recipes_path / "recipe_index.json"
            index_file.write_text('{}')
            
            return service
    
    @pytest.mark.asyncio
    async def test_create_recipe(self, recipe_service):
        """Test creating a new recipe"""
        operations = [
            {"operation": "crop", "parameters": {"aspectRatio": "16:9"}, "enabled": True},
            {"operation": "enhance", "parameters": {"strength": 0.5}, "enabled": True}
        ]
        
        recipe = await recipe_service.create_recipe(
            name="Test Portrait Recipe",
            description="A test recipe for portraits",
            operations=operations,
            style_preset="natural"
        )
        
        assert recipe["name"] == "Test Portrait Recipe"
        assert recipe["description"] == "A test recipe for portraits"
        assert len(recipe["operations"]) == 2
        assert recipe["style_preset"] == "natural"
        assert recipe["is_builtin"] == False
        
        # Verify file was created
        recipe_file = recipe_service.recipes_path / f"{recipe['id']}.json"
        assert recipe_file.exists()
    
    @pytest.mark.asyncio
    async def test_update_recipe(self, recipe_service):
        """Test updating an existing recipe"""
        # Create initial recipe
        recipe = await recipe_service.create_recipe(
            name="Original Name",
            description="Original description",
            operations=[],
            style_preset="natural"
        )
        
        recipe_id = recipe["id"]
        
        # Update recipe
        updated = await recipe_service.update_recipe(
            recipe_id=recipe_id,
            name="Updated Name",
            description="Updated description",
            operations=[{"operation": "enhance", "parameters": {}, "enabled": True}]
        )
        
        assert updated["name"] == "Updated Name"
        assert updated["description"] == "Updated description"
        assert len(updated["operations"]) == 1
        
        # Verify persistence
        loaded = await recipe_service.get_recipe(recipe_id)
        assert loaded["name"] == "Updated Name"
    
    @pytest.mark.asyncio
    async def test_list_recipes_with_correct_format(self, recipe_service):
        """Test listing recipes returns correct format for frontend"""
        # Create test recipes
        for i in range(3):
            await recipe_service.create_recipe(
                name=f"Recipe {i}",
                description=f"Description {i}",
                operations=[{"operation": "enhance", "parameters": {"strength": i * 0.1}}],
                style_preset="natural"
            )
        
        # List recipes
        result = await recipe_service.list_recipes(page=1, page_size=10)
        
        assert "recipes" in result
        assert "total" in result
        assert result["total"] == 3
        
        # Check recipe format
        for recipe in result["recipes"]:
            assert "id" in recipe
            assert "name" in recipe
            assert "steps" in recipe  # Frontend expects 'steps'
            assert "operations" in recipe  # Backwards compatibility
            assert "is_preset" in recipe  # Frontend expects this


class TestEndToEndWorkflow:
    """Test complete workflow from upload to processing"""
    
    @pytest.fixture
    async def services(self, tmp_path):
        """Set up all services for integration testing"""
        # Mock all Path references to use tmp_path
        with patch('api.services.photo_service_v2.Path') as photo_path_mock, \
             patch('api.services.processing_service_v2.Path') as proc_path_mock, \
             patch('api.services.recipe_service_v2.Path') as recipe_path_mock:
            
            # Configure path mocks
            for mock in [photo_path_mock, proc_path_mock, recipe_path_mock]:
                mock.return_value = tmp_path
            
            # Create services
            photo_service = EnhancedPhotoService()
            processing_service = EnhancedProcessingService() 
            recipe_service = EnhancedRecipeService()
            
            # Set up paths
            for service in [photo_service, processing_service]:
                service.data_path = tmp_path
                service.inbox_path = tmp_path / "inbox"
                service.originals_path = tmp_path / "originals"
                service.processed_path = tmp_path / "processed"
                service.recipes_path = tmp_path / "recipes"
                
            processing_service.output_dir = tmp_path / "processed"
            processing_service.thumbnail_dir = tmp_path / "thumbnails"
            processing_service.web_dir = tmp_path / "web"
            recipe_service.recipes_path = tmp_path / "recipes"
            
            # Create all directories
            for path in [tmp_path / "inbox", tmp_path / "originals", 
                        tmp_path / "processed", tmp_path / "recipes",
                        tmp_path / "thumbnails", tmp_path / "web"]:
                path.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            (tmp_path / "photos.json").write_text('{"photos": {}}')
            (tmp_path / "recipes" / "recipe_index.json").write_text('{}')
            
            # Connect services
            processing_service.set_websocket_manager(None)  # No websocket in tests
            
            return {
                'photo': photo_service,
                'processing': processing_service,
                'recipe': recipe_service
            }
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, services, tmp_path):
        """Test complete workflow: upload -> recipe -> process -> verify"""
        photo_service = services['photo']
        processing_service = services['processing']
        recipe_service = services['recipe']
        
        # Step 1: Create a portrait recipe
        recipe_result = await recipe_service.create_recipe(
            name="Portrait Enhancement",
            description="Enhance portrait photos",
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
            style_preset="natural",
            processing_config={
                "qualityThreshold": 80,
                "export": {
                    "format": "jpeg",
                    "quality": 90
                }
            }
        )
        
        recipe_id = recipe_result["id"]
        
        # Step 2: Upload a photo
        mock_file = AsyncMock()
        mock_file.filename = "portrait_test.jpg"
        mock_file.content_type = "image/jpeg"
        test_content = b"test portrait image data"
        mock_file.read.return_value = test_content
        
        # Mock the processing service queue method
        with patch.object(processing_service, 'queue_photo_processing') as mock_queue:
            mock_queue.return_value = {"status": "queued", "photo_id": "test_123"}
            
            upload_result = await photo_service.save_upload(
                file=mock_file,
                auto_process=True,
                recipe_id=recipe_id
            )
        
        assert upload_result["status"] == "pending"  # Auto-process sets to pending
        photo_id = upload_result["photo_id"]
        
        # Step 3: Verify photo is in database
        photo = await photo_service.get_photo(photo_id)
        assert photo is not None
        assert photo.filename == "portrait_test.jpg"
        assert photo.status == "pending"
        
        # Step 4: Create batch operation with recipe
        batch_op = BatchOperation(
            photo_ids=[photo_id],
            recipe_id=recipe_id,
            priority="normal"
        )
        
        # Mock photo service in processing service
        with patch('api.services.processing_service_v2.photo_service', photo_service):
            batch_result = await processing_service.batch_process(batch_op)
        
        assert batch_result["queued"] == 1
        assert batch_result["skipped"] == 0
        
        # Step 5: Process the photo
        # First create the actual file so processing works
        photo_data = photo_service.db.get_photo(photo_id)
        actual_path = Path(photo_data['original_path'])
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        actual_path.write_bytes(test_content)
        
        with patch('api.services.processing_service_v2.photo_service', photo_service):
            process_result = await processing_service.process_next_item()
        
        assert process_result is not None
        assert process_result["status"] == "completed"
        
        # Step 6: Verify outputs
        assert (processing_service.output_dir / actual_path.name).exists()
        
        # Verify photo status was updated
        assert photo_id in photo_service.db.load()["photos"]
        final_photo = photo_service.db.get_photo(photo_id)
        assert final_photo["status"] == "completed"
        assert "processed_path" in final_photo
        assert "processing_time" in final_photo
    
    @pytest.mark.asyncio
    async def test_recipe_application_to_multiple_photos(self, services):
        """Test applying a recipe to multiple photos"""
        photo_service = services['photo']
        processing_service = services['processing']
        recipe_service = services['recipe']
        
        # Create recipe
        recipe = await recipe_service.create_recipe(
            name="Batch Recipe",
            description="For batch processing",
            operations=[{"operation": "enhance", "parameters": {"strength": 0.5}}]
        )
        recipe_id = recipe["id"]
        
        # Upload multiple photos
        photo_ids = []
        for i in range(3):
            mock_file = AsyncMock()
            mock_file.filename = f"batch_photo_{i}.jpg"
            mock_file.content_type = "image/jpeg"
            mock_file.read.return_value = f"photo {i} data".encode()
            
            with patch.object(processing_service, 'queue_photo_processing'):
                result = await photo_service.save_upload(
                    file=mock_file,
                    auto_process=False
                )
            photo_ids.append(result["photo_id"])
        
        # Apply recipe to all photos
        batch_op = BatchOperation(
            photo_ids=photo_ids,
            recipe_id=recipe_id,
            priority="high"
        )
        
        with patch('api.services.processing_service_v2.photo_service', photo_service):
            result = await processing_service.batch_process(batch_op)
        
        assert result["queued"] == 3
        assert len(processing_service.processing_queue) == 3
        
        # Verify all are high priority
        for item in processing_service.processing_queue:
            assert item["priority"] == "high"
            assert item["recipe_id"] == recipe_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])