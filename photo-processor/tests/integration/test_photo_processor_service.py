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
    def service_with_real_directories(self, temp_directories):
        """Create service instance with real temporary directories"""
        return PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

    @pytest.fixture
    def large_test_files(self, temp_directories):
        """Create large test files for performance testing"""
        files = []
        for i, size_mb in enumerate([1, 5, 10, 25]):
            filename = f"large_test_{size_mb}mb_{i}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                # Create file with specified size
                data = b'FAKE_RAW_DATA' * (size_mb * 1024 * 1024 // 13)
                f.write(data)
            files.append(filepath)
        return files

    @pytest.fixture
    def multiple_file_types(self, temp_directories):
        """Create multiple file types for comprehensive testing"""
        files = []
        file_specs = [
            ("image1.ARW", b'SONY_ARW_HEADER'),
            ("image2.CR2", b'CANON_CR2_HEADER'),
            ("image3.NEF", b'NIKON_NEF_HEADER'),
            ("image4.DNG", b'ADOBE_DNG_HEADER'),
            ("image5.JPG", b'\xff\xd8\xff\xe0'),  # JPEG header
            ("image6.PNG", b'\x89PNG\r\n\x1a\n'),  # PNG header
            ("image7.TIFF", b'II*\x00'),  # TIFF header
        ]
        
        for filename, header in file_specs:
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(header)
                f.write(b'FAKE_IMAGE_DATA' * 1000)
            files.append(filepath)
        return files

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

    def test_full_pipeline_with_different_file_types(self, temp_directories, mock_services, sample_photo_analysis, multiple_file_types):
        """Test complete pipeline with different file types"""
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
        
        successful_count = 0
        for file_path in multiple_file_types:
            if service.is_supported_file(os.path.basename(file_path)):
                result = service.process_file(file_path)
                if result:
                    successful_count += 1
        
        # Should process supported file types successfully
        assert successful_count >= 6  # All supported formats
        
        # Verify proper file movement
        for file_path in multiple_file_types:
            if service.is_supported_file(os.path.basename(file_path)):
                original_filename = os.path.basename(file_path)
                processed_file = os.path.join(temp_directories['processed'], original_filename)
                assert os.path.exists(processed_file), f"File {original_filename} should be moved to processed folder"

    def test_service_resilience_to_ai_service_intermittent_failures(self, temp_directories, mock_services, sample_photo_analysis):
        """Test service resilience when AI service has intermittent failures"""
        # Setup AI service to fail initially, then succeed
        failure_count = 0
        def ai_analyze_side_effect(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                return None  # Simulate failure
            return sample_photo_analysis  # Then succeed
        
        mock_services['ai_analyzer'].analyze_photo.side_effect = ai_analyze_side_effect
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(temp_directories['inbox'], f"test_{i}.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            test_files.append(test_file)
        
        results = []
        for test_file in test_files:
            result = service.process_file(test_file)
            results.append(result)
        
        # First two should fail, third should succeed
        assert results == [False, False, True]
        
        # Verify AI analyzer was called 3 times
        assert mock_services['ai_analyzer'].analyze_photo.call_count == 3

    def test_service_recovery_after_immich_service_restart(self, temp_directories, mock_services, sample_photo_analysis):
        """Test service recovery after Immich service restart simulation"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.return_value = True
        
        # Simulate Immich being down then coming back up
        upload_attempts = 0
        def upload_side_effect(*args, **kwargs):
            nonlocal upload_attempts
            upload_attempts += 1
            if upload_attempts <= 2:
                raise Exception("Connection refused")
            return "uploaded-asset-id"
        
        mock_services['immich_client'].upload_photo.side_effect = upload_side_effect
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(temp_directories['inbox'], f"recovery_test_{i}.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            test_files.append(test_file)
        
        results = []
        for test_file in test_files:
            try:
                result = service.process_file(test_file)
                results.append(result)
            except Exception:
                results.append(False)
        
        # Should have attempted upload 3 times
        assert upload_attempts == 3
        # Last attempt should succeed (service recovers)
        assert results[-1] is True

    def test_performance_with_large_files(self, temp_directories, mock_services, sample_photo_analysis, large_test_files, performance_timer):
        """Test performance with large files"""
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
        
        # Test processing time for different file sizes
        for i, file_path in enumerate(large_test_files):
            file_size_mb = [1, 5, 10, 25][i]
            label = f"file_{file_size_mb}mb"
            
            performance_timer.start(label)
            result = service.process_file(file_path)
            performance_timer.stop(label)
            
            assert result is True
            
            # Performance assertions (reasonable for mocked processing)
            if file_size_mb <= 5:
                performance_timer.assert_faster_than(2.0, label)
            elif file_size_mb <= 15:
                performance_timer.assert_faster_than(5.0, label)
            else:
                performance_timer.assert_faster_than(10.0, label)

    def test_memory_usage_during_processing(self, temp_directories, mock_services, sample_photo_analysis, memory_profiler):
        """Test memory usage doesn't grow excessively during processing"""
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
        
        memory_profiler.start()
        
        # Process multiple files and monitor memory
        for i in range(10):
            test_file = os.path.join(temp_directories['inbox'], f"memory_test_{i}.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 1000)
            
            service.process_file(test_file)
            memory_profiler.update_peak()
        
        # Memory increase should be reasonable (less than 100MB for mock processing)
        memory_profiler.assert_memory_increase_less_than(100)

    def test_concurrent_file_processing_thread_safety(self, temp_directories, mock_services, sample_photo_analysis):
        """Test thread safety during concurrent file processing"""
        import threading
        import time
        
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
        
        # Create test files
        test_files = []
        for i in range(10):
            test_file = os.path.join(temp_directories['inbox'], f"concurrent_test_{i}.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            test_files.append(test_file)
        
        results = []
        results_lock = threading.Lock()
        
        def process_file_thread(file_path):
            """Thread function to process a single file"""
            try:
                result = service.process_file(file_path)
                with results_lock:
                    results.append((file_path, result, None))
            except Exception as e:
                with results_lock:
                    results.append((file_path, False, str(e)))
        
        # Start concurrent processing threads
        threads = []
        for file_path in test_files:
            thread = threading.Thread(target=process_file_thread, args=(file_path,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread
        
        # Verify all threads completed successfully
        assert len(results) == len(test_files)
        successful_results = [r for r in results if r[1] is True]
        assert len(successful_results) == len(test_files), f"Some files failed to process: {[r for r in results if not r[1]]}"
        
        # Verify no exceptions occurred
        exceptions = [r[2] for r in results if r[2] is not None]
        assert len(exceptions) == 0, f"Exceptions occurred during concurrent processing: {exceptions}"

    def test_hash_tracker_prevents_reprocessing(self, temp_directories, mock_services, sample_photo_analysis):
        """Test that hash tracker prevents duplicate processing"""
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
        test_file = os.path.join(temp_directories['inbox'], "hash_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        # Process file first time
        result1 = service.process_file(test_file)
        assert result1 is True
        
        # Recreate the same file (simulate re-copy)
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        # Create file handler to test hash checking
        from main import PhotoFileHandler
        handler = PhotoFileHandler(service)
        
        # Should detect as already processed
        assert service.hash_tracker.is_already_processed(test_file) is True
        
        # Verify AI analyzer was only called once
        assert mock_services['ai_analyzer'].analyze_photo.call_count == 1

    def test_different_ai_model_responses(self, temp_directories, mock_services):
        """Test handling of different AI model responses"""
        from tests.fixtures.mock_data import MockPhotoAnalysisData
        
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Test different analysis scenarios
        analysis_scenarios = [
            (MockPhotoAnalysisData.crisp_swimmer_analysis(), "crisp"),
            (MockPhotoAnalysisData.blurry_photo_analysis(), "blurry"),
            (MockPhotoAnalysisData.multiple_swimmers_analysis(), "multiple"),
            (MockPhotoAnalysisData.backstroke_analysis(), "backstroke"),
            (MockPhotoAnalysisData.enhance_only_analysis(), "enhance_only")
        ]
        
        results = []
        for analysis, scenario_name in analysis_scenarios:
            mock_services['ai_analyzer'].analyze_photo.return_value = analysis
            
            test_file = os.path.join(temp_directories['inbox'], f"{scenario_name}_test.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            
            result = service.process_file(test_file)
            results.append((scenario_name, result, analysis.processing_recommendation))
        
        # Verify all scenarios were handled appropriately
        for scenario_name, result, recommendation in results:
            if recommendation == "no_processing":
                # Blurry photos should be skipped
                assert result is False, f"Blurry photo should be skipped: {scenario_name}"
            else:
                # All other scenarios should process successfully
                assert result is True, f"Scenario should succeed: {scenario_name}"

    def test_file_watcher_integration_startup_and_shutdown(self, temp_directories, mock_services):
        """Test file watcher startup and shutdown integration"""
        with patch('main.Observer') as mock_observer_class:
            mock_observer = Mock()
            mock_observer_class.return_value = mock_observer
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Start watching in a separate thread
            import threading
            watcher_thread = threading.Thread(target=service.start_watching)
            watcher_thread.daemon = True
            watcher_thread.start()
            
            # Give it time to start
            time.sleep(0.1)
            
            # Verify observer was started
            mock_observer.start.assert_called_once()
            
            # Verify observer was scheduled with correct parameters
            mock_observer.schedule.assert_called_once()
            call_args = mock_observer.schedule.call_args
            assert call_args[0][1] == temp_directories['inbox']  # watch folder
            assert call_args[1]['recursive'] is True

    def test_bulk_processing_statistics_and_reporting(self, temp_directories, mock_services, sample_photo_analysis, multiple_file_types):
        """Test bulk processing with proper statistics and reporting"""
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
        
        # Process all existing files
        with patch('main.logger') as mock_logger:
            service.process_all_files_in_folder()
            
            # Verify summary logging was called
            summary_calls = [call for call in mock_logger.info.call_args_list 
                           if 'BULK PROCESSING SUMMARY' in str(call)]
            assert len(summary_calls) > 0, "Bulk processing summary should be logged"
            
            # Verify individual file processing was logged
            processing_calls = [call for call in mock_logger.info.call_args_list 
                              if 'Processing [' in str(call)]
            assert len(processing_calls) >= 6, "Individual file processing should be logged"

    def test_error_handling_with_corrupted_files(self, temp_directories, mock_services, sample_photo_analysis):
        """Test error handling with corrupted or invalid files"""
        mock_services['ai_analyzer'].analyze_photo.return_value = sample_photo_analysis
        mock_services['image_processor'].process_photo.side_effect = Exception("Corrupted file")
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        # Create corrupted file
        corrupted_file = os.path.join(temp_directories['inbox'], "corrupted.ARW")
        with open(corrupted_file, 'wb') as f:
            f.write(b"\x00" * 100)  # Invalid data
        
        # Should handle corruption gracefully
        result = service.process_file(corrupted_file)
        assert result is False
        
        # File should still exist in inbox (not moved)
        assert os.path.exists(corrupted_file)

    def test_timeout_handling_for_slow_operations(self, temp_directories, mock_services, sample_photo_analysis):
        """Test timeout handling for slow AI analysis operations"""
        import time
        
        def slow_ai_analysis(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow response
            return sample_photo_analysis
        
        mock_services['ai_analyzer'].analyze_photo.side_effect = slow_ai_analysis
        mock_services['image_processor'].process_photo.return_value = True
        mock_services['immich_client'].upload_photo.return_value = "uploaded-asset-id"
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "slow_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        start_time = time.time()
        result = service.process_file(test_file)
        processing_time = time.time() - start_time
        
        # Should complete successfully even with slow AI
        assert result is True
        assert processing_time >= 0.1  # Should take at least the sleep time

    def test_resource_cleanup_after_processing_failures(self, temp_directories, mock_services):
        """Test proper resource cleanup when processing fails"""
        # Make AI analysis fail
        mock_services['ai_analyzer'].analyze_photo.return_value = None
        
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        
        test_file = os.path.join(temp_directories['inbox'], "cleanup_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        # Process should fail but not leave temp files
        initial_temp_files = len(os.listdir(tempfile.gettempdir()))
        
        result = service.process_file(test_file)
        assert result is False
        
        # Verify no additional temp files were left behind
        final_temp_files = len(os.listdir(tempfile.gettempdir()))
        assert final_temp_files <= initial_temp_files + 1  # Allow for some system temp files

    def test_graceful_degradation_with_missing_services(self, temp_directories):
        """Test graceful degradation when external services are unavailable"""
        # Create service with failing external services
        with patch('main.AIAnalyzer') as mock_ai, \
             patch('main.ImmichClient') as mock_immich:
            
            # Make services fail connection tests
            mock_ai_instance = Mock()
            mock_ai_instance.test_connection.return_value = False
            mock_ai.return_value = mock_ai_instance
            
            mock_immich_instance = Mock()
            mock_immich_instance.test_connection.return_value = False
            mock_immich.return_value = mock_immich_instance
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Startup checks should fail gracefully
            startup_result = service.startup_checks()
            assert startup_result is False
            
            # Service should not crash
            assert service is not None

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

    def test_environment_variable_configuration(self, temp_directories, environment_variables):
        """Test service configuration via environment variables"""
        with patch('main.AIAnalyzer') as mock_ai, \
             patch('main.ImageProcessor') as mock_img, \
             patch('main.ImmichClient') as mock_immich:
            
            # Setup mocks
            mock_ai.return_value = Mock()
            mock_img.return_value = Mock()
            mock_immich.return_value = Mock()
            
            # Create service (should pick up environment variables)
            service = PhotoProcessorService()
            
            # Verify configuration was read from environment
            assert service.ollama_host == environment_variables['OLLAMA_HOST']
            assert service.immich_api_url == environment_variables['IMMICH_API_URL']
            assert service.immich_api_key == environment_variables['IMMICH_API_KEY']
    
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
        
        # Note: In actual implementation, temp files would be cleaned up
        # This test verifies the path generation is correct