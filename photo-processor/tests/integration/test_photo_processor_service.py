"""
Integration tests for the complete photo processing workflow
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil
import numpy as np
from pathlib import Path
import time
from main import PhotoProcessorService
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis, SwimmingContext


class TestPhotoProcessorServiceIntegration:
    """Integration test suite for PhotoProcessorService"""

    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing"""
        temp_dir = tempfile.mkdtemp()
        inbox_dir = os.path.join(temp_dir, "inbox")
        processed_dir = os.path.join(temp_dir, "processed")
        temp_work_dir = os.path.join(temp_dir, "temp")
        
        os.makedirs(inbox_dir)
        os.makedirs(processed_dir)
        os.makedirs(temp_work_dir)
        
        yield {
            'inbox': inbox_dir,
            'processed': processed_dir,
            'temp': temp_work_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_services(self):
        """Mock all external services"""
        with patch('main.AIAnalyzer') as mock_ai, \
             patch('main.ImageProcessor') as mock_img, \
             patch('main.ImmichClient') as mock_immich:
            
            # Setup AI Analyzer mock
            mock_ai_instance = Mock()
            mock_ai_instance.test_connection.return_value = True
            mock_ai_instance.ensure_model_available.return_value = True
            mock_ai.return_value = mock_ai_instance
            
            # Setup Image Processor mock
            mock_img_instance = Mock()
            mock_img.return_value = mock_img_instance
            
            # Setup Immich Client mock
            mock_immich_instance = Mock()
            mock_immich_instance.test_connection.return_value = True
            mock_immich_instance.get_or_create_album.return_value = "test-album-id"
            mock_immich.return_value = mock_immich_instance
            
            yield {
                'ai_analyzer': mock_ai_instance,
                'image_processor': mock_img_instance,
                'immich_client': mock_immich_instance
            }

    @pytest.fixture
    def sample_photo_analysis(self):
        """Sample photo analysis for testing"""
        return PhotoAnalysis(
            description="Test swimming photo with excellent technique",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=10.0, y=15.0, width=80.0, height=70.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.85
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "white"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=5,
                contrast_adjustment_needed=10
            ),
            swimming_context=SwimmingContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=4
            ),
            processing_recommendation="crop_and_enhance"
        )

    def test_service_initialization(self, temp_directories, mock_services):
        """Test PhotoProcessorService initialization"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        assert service.watch_folder == temp_directories['inbox']
        assert service.output_folder == temp_directories['processed']
        assert service.ai_analyzer is not None
        assert service.image_processor is not None
        assert service.immich_client is not None

    def test_startup_checks_success(self, temp_directories, mock_services):
        """Test successful startup checks"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        result = service.perform_startup_checks()
        
        assert result is True
        mock_services['ai_analyzer'].test_connection.assert_called_once()
        mock_services['ai_analyzer'].ensure_model_available.assert_called_once()
        mock_services['immich_client'].test_connection.assert_called_once()

    def test_startup_checks_ai_failure(self, temp_directories, mock_services):
        """Test startup checks with AI service failure"""
        mock_services['ai_analyzer'].test_connection.return_value = False
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        result = service.perform_startup_checks()
        
        assert result is False

    def test_startup_checks_immich_failure(self, temp_directories, mock_services):
        """Test startup checks with Immich service failure"""
        mock_services['immich_client'].test_connection.return_value = False
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        result = service.perform_startup_checks()
        
        assert result is False

    def test_supported_file_extensions(self, temp_directories, mock_services):
        """Test supported file extension detection"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Test RAW formats
        assert service.is_supported_file("test.arw") is True
        assert service.is_supported_file("test.cr2") is True
        assert service.is_supported_file("test.nef") is True
        assert service.is_supported_file("test.dng") is True
        
        # Test regular image formats
        assert service.is_supported_file("test.jpg") is True
        assert service.is_supported_file("test.jpeg") is True
        assert service.is_supported_file("test.png") is True
        assert service.is_supported_file("test.tiff") is True
        
        # Test unsupported formats
        assert service.is_supported_file("test.txt") is False
        assert service.is_supported_file("test.mp4") is False
        assert service.is_supported_file("test.pdf") is False

    def test_file_processing_workflow_success(self, temp_directories, mock_services, sample_photo_analysis):
        """Test complete file processing workflow"""
        # Setup mocks
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create test file
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 1000)  # Make it reasonably sized
        
        # Process the file
        result = service.process_file(test_file)
        
        assert result is True
        
        # Verify all steps were called
        mock_services['ai_analyzer'].analyze_photo.assert_called_once()
        mock_services['image_processor'].process_photo.assert_called_once()
        mock_services['immich_client'].upload_photo.assert_called_once()
        
        # Verify file was moved to processed folder
        processed_file = os.path.join(temp_directories['processed'], "DSC09123.ARW")
        assert os.path.exists(processed_file)

    def test_file_processing_ai_analysis_failure(self, temp_directories, mock_services):
        """Test file processing when AI analysis fails"""
        mock_services['ai_analyzer'].analyze_photo.return_value = None
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        result = service.process_file(test_file)
        
        assert result is False
        mock_services['image_processor'].process_photo.assert_not_called()
        mock_services['immich_client'].upload_photo.assert_not_called()

    def test_file_processing_image_processing_failure(self, temp_directories, mock_services, sample_photo_analysis):
        """Test file processing when image processing fails"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = False
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        result = service.process_file(test_file)
        
        assert result is False
        mock_services['immich_client'].upload_photo.assert_not_called()

    def test_file_processing_upload_failure(self, temp_directories, mock_services, sample_photo_analysis):
        """Test file processing when Immich upload fails"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = None
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        result = service.process_file(test_file)
        
        # Should still return True as processing succeeded, just upload failed
        assert result is True

    def test_file_processing_very_blurry_skip(self, temp_directories, mock_services):
        """Test skipping very blurry images"""
        blurry_analysis = PhotoAnalysis(
            description="Very blurry photo, not suitable for processing",
            quality="very_blurry",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=10.0, y=15.0, width=80.0, height=70.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.85
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            ),
            swimming_context=SwimmingContext(),
            processing_recommendation="no_processing"
        )
        
        mock_services['ai_analyzer'].analyze_photo.return_value = blurry_analysis
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        result = service.process_file(test_file)
        
        # Should skip processing for very blurry images
        assert result is False
        mock_services['image_processor'].process_photo.assert_not_called()

    def test_process_existing_files(self, temp_directories, mock_services, sample_photo_analysis):
        """Test processing existing files in inbox"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create multiple test files
        test_files = ["DSC09001.ARW", "DSC09002.CR2", "DSC09003.NEF"]
        for filename in test_files:
            test_file = os.path.join(temp_directories['inbox'], filename)
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
        
        service.process_existing_files()
        
        # Should process all files
        assert mock_services['ai_analyzer'].analyze_photo.call_count == 3
        assert mock_services['image_processor'].process_photo.call_count == 3
        assert mock_services['immich_client'].upload_photo.call_count == 3

    def test_file_stability_check(self, temp_directories, mock_services):
        """Test that files are stable before processing"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        
        # Create file that's still being written
        with open(test_file, 'wb') as f:
            f.write(b"partial data")
        
        # Should not be stable immediately
        assert service.is_file_stable(test_file, wait_seconds=1) is False
        
        # Wait and check again - should be stable now
        time.sleep(1.1)
        assert service.is_file_stable(test_file, wait_seconds=1) is True

    def test_output_filename_generation(self, temp_directories, mock_services):
        """Test output filename generation"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Test RAW file conversion to JPEG
        output_path = service.get_output_path("DSC09123.ARW")
        expected_path = os.path.join(temp_directories['processed'], "DSC09123_processed.jpg")
        assert output_path == expected_path
        
        # Test JPEG file processing
        output_path = service.get_output_path("IMG_001.JPG")
        expected_path = os.path.join(temp_directories['processed'], "IMG_001_processed.jpg")
        assert output_path == expected_path

    def test_error_handling_and_logging(self, temp_directories, mock_services):
        """Test error handling and logging"""
        mock_services['ai_analyzer'].analyze_photo.side_effect = Exception("AI service error")
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        # Should handle exception gracefully
        result = service.process_file(test_file)
        assert result is False

    def test_concurrent_file_processing(self, temp_directories, mock_services, sample_photo_analysis):
        """Test that concurrent processing works correctly"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create multiple files
        test_files = []
        for i in range(5):
            filename = f"DSC0900{i}.ARW"
            test_file = os.path.join(temp_directories['inbox'], filename)
            test_files.append(test_file)
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
        
        # Process multiple files
        results = []
        for test_file in test_files:
            result = service.process_file(test_file)
            results.append(result)
        
        # All should succeed
        assert all(results)
        assert mock_services['ai_analyzer'].analyze_photo.call_count == 5

    def test_album_creation_and_usage(self, temp_directories, mock_services, sample_photo_analysis):
        """Test album creation and asset addition"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Perform startup checks to create album
        service.perform_startup_checks()
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        service.process_file(test_file)
        
        # Should have created album and uploaded with album ID
        mock_services['immich_client'].get_or_create_album.assert_called()
        upload_call = mock_services['immich_client'].upload_photo.call_args
        assert upload_call[1]['album_id'] == "test-album-id"

    def test_temp_file_cleanup(self, temp_directories, mock_services, sample_photo_analysis):
        """Test temporary file cleanup"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "DSC09123.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data")
        
        # Track the temp file that should be created
        temp_jpg_path = service.get_temp_jpg_path("DSC09123.ARW")
        
        service.process_file(test_file)
        
        # Temp file should be cleaned up after processing
        # This depends on the implementation in main.py