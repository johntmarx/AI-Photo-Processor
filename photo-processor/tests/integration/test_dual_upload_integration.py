"""
Integration tests for the dual upload system
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time
import json
from unittest.mock import patch, Mock

from main_v2 import EnhancedPhotoProcessor
from recipe_storage import ProcessingRecipe, RecipeStorage
from immich_client_v2 import EnhancedImmichClient, UploadResult, DualUploadResult
from hash_tracker import HashTracker


class TestDualUploadIntegration:
    """Integration tests for the complete dual upload workflow"""
    
    @pytest.fixture
    def test_environment(self):
        """Set up complete test environment"""
        base_dir = tempfile.mkdtemp()
        
        env = {
            'base_dir': Path(base_dir),
            'inbox': Path(base_dir) / 'inbox',
            'data': Path(base_dir) / 'data',
            'originals': Path(base_dir) / 'data' / 'originals',
            'processed': Path(base_dir) / 'data' / 'processed',
            'working': Path(base_dir) / 'data' / 'working',
            'recipes': Path(base_dir) / 'data' / 'recipes'
        }
        
        # Create all directories
        for path in env.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
        
        yield env
        
        # Cleanup
        shutil.rmtree(base_dir)
    
    @pytest.fixture
    def test_image_file(self, test_environment):
        """Create a test image file"""
        image_path = test_environment['inbox'] / 'test_photo.jpg'
        
        # Create a simple JPEG-like file
        jpeg_header = b'\xff\xd8\xff\xe0'  # JPEG SOI and APP0 marker
        image_data = jpeg_header + b'fake image data' * 1000
        
        with open(image_path, 'wb') as f:
            f.write(image_data)
            
        return image_path
    
    @pytest.fixture
    def configured_processor(self, test_environment, monkeypatch):
        """Create a fully configured processor for integration testing"""
        # Set environment variables
        monkeypatch.setenv('INBOX_FOLDER', str(test_environment['inbox']))
        monkeypatch.setenv('IMMICH_API_URL', 'http://test-immich:2283')
        monkeypatch.setenv('IMMICH_API_KEY', 'test-integration-key')
        monkeypatch.setenv('ENABLE_AI_PROCESSING', 'false')  # Disable AI for integration tests
        
        # Create real instances of components
        processor = EnhancedPhotoProcessor()
        
        # Override paths
        processor.inbox_folder = test_environment['inbox']
        processor.originals_folder = test_environment['originals']
        processor.processed_folder = test_environment['processed']
        processor.working_folder = test_environment['working']
        
        # Use real recipe storage and hash tracker
        processor.recipe_storage = RecipeStorage(test_environment['recipes'])
        processor.hash_tracker = HashTracker(
            str(test_environment['data'] / 'hash_tracker.db')
        )
        
        return processor
    
    def test_complete_workflow_no_ai(self, configured_processor, test_image_file):
        """Test complete workflow without AI processing"""
        processor = configured_processor
        processor.enable_ai_processing = False
        
        # Mock only the Immich client uploads
        with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
            # Configure mock to return successful upload
            mock_upload.return_value = DualUploadResult(
                original=UploadResult('orig-001', test_image_file.name, True),
                processed=UploadResult('proc-001', 'processed.jpg', True),
                linked=True,
                recipe_stored=True
            )
            
            # Process the file
            result = processor.process_single_file(test_image_file)
        
        # Verify results
        assert result is not None
        assert result.original.success is True
        assert result.processed.success is True
        
        # Verify original was stored
        originals = list(processor.originals_folder.rglob('*.jpg'))
        assert len(originals) == 1
        stored_original = originals[0]
        assert test_image_file.name in stored_original.name
        
        # Verify processed file was created
        processed_files = list(processor.processed_folder.glob('*.jpg'))
        assert len(processed_files) == 1
        
        # Verify recipe was saved
        recipes = processor.recipe_storage.list_recipes()
        assert len(recipes) == 1
        
        # Verify hash was tracked
        file_hash = processor.calculate_file_hash(stored_original)
        assert processor.hash_tracker.is_processed(file_hash)
        
        # Verify original was removed from inbox
        assert not test_image_file.exists()
    
    def test_workflow_with_recipe_operations(self, configured_processor, test_image_file):
        """Test workflow with processing operations in recipe"""
        processor = configured_processor
        
        # Mock AI analyzer to return processing suggestions
        with patch.object(processor.ai_analyzer, 'analyze_photo') as mock_ai:
            mock_ai.return_value = {
                'rotation_needed': True,
                'rotation_angle': 90,
                'suggested_crop': {
                    'x1': 0.1, 'y1': 0.1,
                    'x2': 0.9, 'y2': 0.9
                },
                'color_adjustments': {
                    'brightness': 0.1,
                    'contrast': 0.2
                }
            }
            
            processor.enable_ai_processing = True
            
            # Mock Immich uploads
            with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
                mock_upload.return_value = DualUploadResult(
                    original=UploadResult('orig-002', test_image_file.name, True),
                    processed=UploadResult('proc-002', 'processed.jpg', True),
                    linked=True,
                    recipe_stored=True
                )
                
                result = processor.process_single_file(test_image_file)
        
        # Load the saved recipe
        recipes = processor.recipe_storage.list_recipes()
        recipe_id = recipes[0]['id']
        recipe = processor.recipe_storage.load_recipe(recipe_id)
        
        # Verify recipe contains operations
        assert len(recipe.operations) == 3  # rotate, crop, enhance
        assert recipe.operations[0].type == 'rotate'
        assert recipe.operations[0].parameters['angle'] == 90
        assert recipe.operations[1].type == 'crop'
        assert recipe.operations[2].type == 'enhance'
        
        # Verify AI metadata was stored
        assert 'results' in recipe.ai_metadata
        assert recipe.ai_metadata['results']['rotation_angle'] == 90
    
    def test_duplicate_file_handling(self, configured_processor, test_image_file):
        """Test handling of duplicate files"""
        processor = configured_processor
        
        # First, process the file once
        with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
            mock_upload.return_value = DualUploadResult(
                original=UploadResult('orig-003', test_image_file.name, True),
                processed=UploadResult('proc-003', 'processed.jpg', True),
                linked=True,
                recipe_stored=True
            )
            
            result1 = processor.process_single_file(test_image_file)
        
        assert result1 is not None
        
        # Create another file with same content
        duplicate_file = test_image_file.parent / 'duplicate.jpg'
        shutil.copy2(test_image_file, duplicate_file)
        
        # Try to process the duplicate
        result2 = processor.process_single_file(duplicate_file)
        
        # Should detect as duplicate and skip
        assert result2 is None
        assert not duplicate_file.exists()  # Should be deleted
    
    def test_upload_failure_handling(self, configured_processor, test_image_file):
        """Test handling of upload failures"""
        processor = configured_processor
        
        # Mock upload failure
        with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
            mock_upload.return_value = DualUploadResult(
                original=UploadResult('', test_image_file.name, False, 'Network error'),
                processed=UploadResult('', 'processed.jpg', False, 'Original failed'),
                linked=False,
                recipe_stored=False
            )
            
            result = processor.process_single_file(test_image_file)
        
        assert result is not None
        assert result.original.success is False
        
        # Original should still be stored locally
        originals = list(processor.originals_folder.rglob('*.jpg'))
        assert len(originals) == 1
        
        # Hash should NOT be marked as processed (since upload failed)
        file_hash = processor.calculate_file_hash(originals[0])
        assert not processor.hash_tracker.is_processed(file_hash)
    
    def test_recipe_persistence_and_reload(self, configured_processor, test_image_file):
        """Test that recipes can be saved and reloaded correctly"""
        processor = configured_processor
        
        # Process with some operations
        with patch.object(processor.ai_analyzer, 'analyze_photo') as mock_ai:
            mock_ai.return_value = {
                'rotation_needed': True,
                'rotation_angle': 45,
                'quality_score': 8.5,
                'detected_subjects': ['landscape', 'sunset']
            }
            
            processor.enable_ai_processing = True
            
            with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
                mock_upload.return_value = DualUploadResult(
                    original=UploadResult('orig-004', test_image_file.name, True),
                    processed=UploadResult('proc-004', 'processed.jpg', True),
                    linked=True,
                    recipe_stored=True
                )
                
                processor.process_single_file(test_image_file)
        
        # Get the file hash
        originals = list(processor.originals_folder.rglob('*.jpg'))
        file_hash = processor.calculate_file_hash(originals[0])
        
        # Find recipe by hash
        found_recipe = processor.recipe_storage.find_recipe_by_hash(file_hash)
        
        assert found_recipe is not None
        assert found_recipe.original_hash == file_hash
        assert len(found_recipe.operations) > 0
        assert found_recipe.ai_metadata['results']['quality_score'] == 8.5
    
    def test_concurrent_file_processing(self, configured_processor, test_environment):
        """Test processing multiple files concurrently"""
        processor = configured_processor
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            file_path = test_environment['inbox'] / f'photo_{i}.jpg'
            with open(file_path, 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0' + f'image {i}'.encode() * 100)
            test_files.append(file_path)
        
        # Mock uploads with different IDs for each file
        upload_results = []
        for i in range(5):
            upload_results.append(DualUploadResult(
                original=UploadResult(f'orig-{i:03d}', f'photo_{i}.jpg', True),
                processed=UploadResult(f'proc-{i:03d}', f'processed_{i}.jpg', True),
                linked=True,
                recipe_stored=True
            ))
        
        with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
            mock_upload.side_effect = upload_results
            
            # Process all files
            processor.scan_inbox()
        
        # Verify all files were processed
        originals = list(processor.originals_folder.rglob('*.jpg'))
        assert len(originals) == 5
        
        processed_files = list(processor.processed_folder.glob('*.jpg'))
        assert len(processed_files) == 5
        
        # Verify all recipes were saved
        recipes = processor.recipe_storage.list_recipes()
        assert len(recipes) == 5
        
        # Verify inbox is empty
        remaining_files = list(test_environment['inbox'].glob('*.jpg'))
        assert len(remaining_files) == 0
    
    def test_raw_file_processing(self, configured_processor, test_environment):
        """Test processing RAW files with conversion"""
        # Create a fake RAW file
        raw_file = test_environment['inbox'] / 'test_photo.nef'
        with open(raw_file, 'wb') as f:
            f.write(b'FAKE_NEF_DATA' * 100)
        
        processor = configured_processor
        
        # Mock RAW conversion
        with patch.object(processor.image_processor, 'is_raw_file', return_value=True), \
             patch.object(processor.image_processor, 'convert_raw_to_rgb', return_value=True):
            
            with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
                mock_upload.return_value = DualUploadResult(
                    original=UploadResult('orig-raw', 'test_photo.nef', True),
                    processed=UploadResult('proc-raw', 'processed.jpg', True),
                    linked=True,
                    recipe_stored=True
                )
                
                result = processor.process_single_file(raw_file)
        
        assert result is not None
        assert result.original.success is True
        
        # Verify RAW conversion was called
        processor.image_processor.convert_raw_to_rgb.assert_called_once()
    
    def test_original_preservation_guarantee(self, configured_processor, test_image_file):
        """Test that originals are NEVER lost, even on errors"""
        processor = configured_processor
        
        # Calculate hash before processing
        original_hash = processor.calculate_file_hash(test_image_file)
        original_size = test_image_file.stat().st_size
        
        # Mock various failures
        with patch.object(processor.immich_client, 'upload_photo_pair') as mock_upload:
            # Simulate upload failure
            mock_upload.side_effect = Exception("Catastrophic failure!")
            
            # Process should handle the error
            result = processor.process_single_file(test_image_file)
        
        # Result should be None due to error
        assert result is None
        
        # But original should still be safely stored
        originals = list(processor.originals_folder.rglob('*.jpg'))
        assert len(originals) >= 1  # At least one original
        
        # Verify original content is intact
        stored_original = None
        for orig in originals:
            if processor.calculate_file_hash(orig) == original_hash:
                stored_original = orig
                break
        
        assert stored_original is not None
        assert stored_original.stat().st_size == original_size