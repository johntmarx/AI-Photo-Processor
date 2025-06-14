"""
Unit tests for Enhanced Photo Processor
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil
from datetime import datetime

from main_v2 import EnhancedPhotoProcessor
from recipe_storage import ProcessingRecipe
from immich_client_v2 import UploadResult, DualUploadResult


class TestEnhancedPhotoProcessor:
    """Test EnhancedPhotoProcessor class"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        base_dir = tempfile.mkdtemp()
        dirs = {
            'inbox': Path(base_dir) / 'inbox',
            'originals': Path(base_dir) / 'originals',
            'processed': Path(base_dir) / 'processed',
            'working': Path(base_dir) / 'working',
            'data': Path(base_dir) / 'data'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        yield dirs
        shutil.rmtree(base_dir)
    
    @pytest.fixture
    def processor(self, temp_dirs, monkeypatch):
        """Create processor instance with mocked components"""
        # Set environment variables
        monkeypatch.setenv('INBOX_FOLDER', str(temp_dirs['inbox']))
        monkeypatch.setenv('IMMICH_API_URL', 'http://test-immich:2283')
        monkeypatch.setenv('IMMICH_API_KEY', 'test-key')
        
        # Mock components
        with patch('main_v2.AIAnalyzer'), \
             patch('main_v2.ImageProcessor'), \
             patch('main_v2.EnhancedImmichClient'), \
             patch('main_v2.HashTracker'), \
             patch('main_v2.RecipeStorage'):
            
            # Create processor with mocked paths
            processor = EnhancedPhotoProcessor()
            processor.inbox_folder = temp_dirs['inbox']
            processor.originals_folder = temp_dirs['originals']
            processor.processed_folder = temp_dirs['processed']
            processor.working_folder = temp_dirs['working']
            
            return processor
    
    @pytest.fixture
    def test_image(self, temp_dirs):
        """Create a test image file"""
        image_path = temp_dirs['inbox'] / 'test_photo.jpg'
        with open(image_path, 'wb') as f:
            f.write(b'fake image data')
        return image_path
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.preserve_originals is True
        assert processor.dual_upload is True
        assert processor.enable_ai_processing is True
        assert processor.inbox_folder.exists()
        assert processor.originals_folder.exists()
    
    def test_calculate_file_hash(self, processor, test_image):
        """Test file hash calculation"""
        hash1 = processor.calculate_file_hash(test_image)
        hash2 = processor.calculate_file_hash(test_image)
        
        assert hash1 == hash2  # Same file should give same hash
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
    
    def test_store_original(self, processor, test_image):
        """Test storing original file"""
        file_hash = processor.calculate_file_hash(test_image)
        
        stored_path = processor.store_original(test_image, file_hash)
        
        # Check file was copied (not moved)
        assert test_image.exists()  # Original still exists
        assert stored_path.exists()  # Copy exists
        
        # Check organization by date
        assert stored_path.parent.parent.name.isdigit()  # Year
        assert stored_path.parent.name.isdigit()  # Month
        
        # Check filename includes hash prefix
        assert file_hash[:8] in stored_path.name
        
        # Verify content is identical
        with open(test_image, 'rb') as f1, open(stored_path, 'rb') as f2:
            assert f1.read() == f2.read()
    
    def test_create_processing_recipe_basic(self, processor, test_image):
        """Test creating basic processing recipe"""
        file_hash = 'test_hash_123'
        
        recipe = processor.create_processing_recipe(
            test_image,
            file_hash
        )
        
        assert recipe.original_hash == file_hash
        assert recipe.original_filename == test_image.name
        assert len(recipe.operations) == 0  # No AI results, no operations
    
    def test_create_processing_recipe_with_ai(self, processor, test_image):
        """Test creating recipe with AI results"""
        file_hash = 'test_hash_456'
        ai_results = {
            'rotation_needed': True,
            'rotation_angle': 15.5,
            'suggested_crop': {
                'x1': 0.1, 'y1': 0.2,
                'x2': 0.9, 'y2': 0.8
            },
            'color_adjustments': {
                'brightness': 0.1,
                'contrast': 0.15,
                'saturation': -0.05
            }
        }
        
        recipe = processor.create_processing_recipe(
            test_image,
            file_hash,
            ai_results
        )
        
        assert len(recipe.operations) == 3  # rotate, crop, enhance
        assert recipe.operations[0].type == 'rotate'
        assert recipe.operations[0].parameters['angle'] == 15.5
        assert recipe.operations[1].type == 'crop'
        assert recipe.operations[2].type == 'enhance'
        assert recipe.ai_metadata['results'] == ai_results
    
    def test_apply_recipe(self, processor, test_image):
        """Test applying a processing recipe"""
        # Create test image file
        test_image.write_bytes(b'\x89PNG\r\n\x1a\n' + b'fake png data')
        
        # Mock cv2 module before it's imported
        import sys
        from unittest.mock import MagicMock
        import numpy as np
        
        # Create mock cv2 module
        mock_cv2 = MagicMock()
        mock_image = np.zeros((1000, 1500, 3), dtype=np.uint8)
        
        # Mock cv2 functions
        mock_cv2.imread.return_value = mock_image
        mock_cv2.cvtColor.return_value = mock_image
        mock_cv2.getRotationMatrix2D.return_value = np.eye(2, 3)
        mock_cv2.warpAffine.return_value = mock_image
        mock_cv2.COLOR_BGR2RGB = 4
        
        # Inject mock into sys.modules
        sys.modules['cv2'] = mock_cv2
        
        try:
            # Mock image processor methods
            processor.image_processor.enhance_image.return_value = mock_image
            processor.image_processor.save_high_quality_jpeg.return_value = None
            
            # Create recipe with operations
            recipe = ProcessingRecipe()
            recipe.add_operation('rotate', {'angle': 45})
            recipe.add_operation('crop', {'x1': 0.1, 'y1': 0.1, 'x2': 0.9, 'y2': 0.9})
            recipe.add_operation('enhance', {'brightness': 0.2})
            
            output_path = processor.processed_folder / 'output.jpg'
            
            # Apply recipe
            success = processor.apply_recipe(test_image, recipe, output_path)
            
            assert success is True
            # Verify the image processor methods were called
            processor.image_processor.save_high_quality_jpeg.assert_called_once()
            
        finally:
            # Clean up sys.modules
            if 'cv2' in sys.modules and isinstance(sys.modules['cv2'], MagicMock):
                del sys.modules['cv2']
    
    def test_process_single_file_duplicate(self, processor, test_image):
        """Test processing a duplicate file"""
        # Mock hash tracker to indicate file already processed
        processor.hash_tracker.is_already_processed.return_value = True
        
        result = processor.process_single_file(test_image)
        
        assert result is None
        assert not test_image.exists()  # Should be deleted as duplicate
    
    @patch('main_v2.shutil.copy2')
    def test_process_single_file_success(self, mock_copy, processor, test_image):
        """Test successful file processing"""
        # Setup mocks
        processor.hash_tracker.is_already_processed.return_value = False
        processor.enable_ai_processing = False  # Skip AI for this test
        
        # Mock image processing
        processor.image_processor.convert_raw_to_rgb.return_value = (MagicMock(shape=(100, 100, 3)), {})
        processor.image_processor.save_high_quality_jpeg.return_value = None
        
        # Mock recipe storage
        processor.recipe_storage.save_recipe.return_value = True
        
        # Mock successful upload
        dual_result = DualUploadResult(
            original=UploadResult('orig-123', 'test.jpg', True),
            processed=UploadResult('proc-456', 'test_proc.jpg', True),
            linked=True,
            recipe_stored=True
        )
        processor.immich_client.upload_photo_pair.return_value = dual_result
        
        # Process file
        with patch.object(processor, 'store_original') as mock_store:
            mock_store.return_value = Path('/fake/original.jpg')
            result = processor.process_single_file(test_image)
        
        assert result is not None
        assert result.original.success is True
        assert result.processed.success is True
        
        # Verify hash was tracked
        processor.hash_tracker.mark_as_processed.assert_called_once()
        
        # Verify original was removed from inbox
        assert not test_image.exists()
    
    def test_process_single_file_with_ai(self, processor, test_image, temp_dirs):
        """Test file processing with AI analysis"""
        processor.hash_tracker.is_already_processed.return_value = False
        processor.enable_ai_processing = True
        
        # Mock AI results
        ai_results = {
            'rotation_needed': True,
            'rotation_angle': 5.0,
            'quality_score': 8.5
        }
        processor.ai_analyzer.analyze_photo.return_value = ai_results
        
        # Mock image operations
        processor.image_processor.is_raw_file.return_value = False
        processor.image_processor.convert_raw_to_rgb.return_value = (MagicMock(shape=(100, 100, 3)), {})
        processor.image_processor.enhance_image.return_value = MagicMock(shape=(100, 100, 3))
        processor.image_processor.save_high_quality_jpeg.return_value = None
        
        # Mock successful processing
        with patch.object(processor, 'apply_recipe', return_value=True), \
             patch.object(processor, 'store_original') as mock_store:
            
            # Create a real file to return
            real_original = temp_dirs['originals'] / 'test_original.jpg'
            real_original.parent.mkdir(parents=True, exist_ok=True)
            real_original.write_bytes(b'test image data')
            mock_store.return_value = real_original
            
            # Mock upload
            dual_result = DualUploadResult(
                original=UploadResult('orig-123', 'test.jpg', True),
                processed=UploadResult('proc-456', 'test_proc.jpg', True),
                linked=True,
                recipe_stored=True
            )
            processor.immich_client.upload_photo_pair.return_value = dual_result
            
            result = processor.process_single_file(test_image)
        
        assert result is not None
        processor.ai_analyzer.analyze_photo.assert_called_once()
        
        # Verify recipe was created with AI results
        saved_recipe_call = processor.recipe_storage.save_recipe.call_args[0][0]
        assert saved_recipe_call.ai_metadata['results'] == ai_results
    
    def test_scan_inbox_no_files(self, processor):
        """Test scanning empty inbox"""
        # Inbox is empty
        processor.scan_inbox()
        
        # Should complete without errors
        # No files should be processed
        processor.hash_tracker.is_processed.assert_not_called()
    
    def test_scan_inbox_with_files(self, processor, temp_dirs):
        """Test scanning inbox with multiple files"""
        # Create test files
        files = []
        for ext in ['.jpg', '.png', '.nef']:
            file_path = temp_dirs['inbox'] / f'test{ext}'
            file_path.write_bytes(b'test data')
            files.append(file_path)
        
        # Create non-image file (should be ignored)
        txt_file = temp_dirs['inbox'] / 'readme.txt'
        txt_file.write_text('not an image')
        
        # Mock processing
        with patch.object(processor, 'process_single_file') as mock_process:
            processor.scan_inbox()
        
        # Should process only image files
        assert mock_process.call_count == 3
        
        # Verify correct files were processed
        processed_files = [call[0][0].name for call in mock_process.call_args_list]
        assert 'test.jpg' in processed_files
        assert 'test.png' in processed_files
        assert 'test.nef' in processed_files
        assert 'readme.txt' not in processed_files
    
    def test_raw_file_processing(self, processor, temp_dirs):
        """Test processing RAW files"""
        raw_file = temp_dirs['inbox'] / 'test.nef'
        raw_file.write_bytes(b'fake raw data')
        
        processor.hash_tracker.is_already_processed.return_value = False
        processor.enable_ai_processing = True
        
        # Mock RAW conversion
        processor.image_processor.is_raw_file.return_value = True
        processor.image_processor.convert_raw_to_rgb.return_value = (MagicMock(shape=(100, 100, 3)), {})
        processor.image_processor.save_high_quality_jpeg.return_value = None
        
        with patch.object(processor, 'store_original') as mock_store, \
             patch.object(processor, 'apply_recipe', return_value=True):
            
            # Create a real file to return
            real_original = temp_dirs['originals'] / 'test_original.nef'
            real_original.parent.mkdir(parents=True, exist_ok=True)
            real_original.write_bytes(b'fake raw data')
            mock_store.return_value = real_original
            
            # Mock upload
            dual_result = DualUploadResult(
                original=UploadResult('orig-123', 'test.nef', True),
                processed=UploadResult('proc-456', 'test.jpg', True),
                linked=True,
                recipe_stored=True
            )
            processor.immich_client.upload_photo_pair.return_value = dual_result
            
            result = processor.process_single_file(raw_file)
        
        assert result is not None
        processor.image_processor.convert_raw_to_rgb.assert_called_once()
    
    def test_error_handling_in_process_file(self, processor, test_image):
        """Test error handling during file processing"""
        processor.hash_tracker.is_already_processed.return_value = False
        
        # Mock an error during store_original
        with patch.object(processor, 'store_original') as mock_store:
            mock_store.side_effect = Exception("Storage failed")
            
            result = processor.process_single_file(test_image)
            
            assert result is None  # Should return None on error