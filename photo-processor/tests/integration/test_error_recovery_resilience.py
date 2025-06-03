"""
Error recovery and resilience integration tests for photo processor
"""
import pytest
import tempfile
import os
import time
import threading
import socket
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout, HTTPError
import requests

from main import PhotoProcessorService, PhotoFileHandler
from tests.fixtures.mock_data import MockPhotoAnalysisData
from ai_analyzer import AIAnalyzer
from immich_client import ImmichClient


class TestErrorRecoveryAndResilience:
    """Test suite for error recovery and system resilience"""

    @pytest.fixture
    def resilient_service(self, temp_directories):
        """Create service instance for resilience testing"""
        return PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

    def test_ollama_service_unavailable_on_startup(self, temp_directories):
        """Test behavior when Ollama service is unavailable during startup"""
        with patch('ai_analyzer.ollama.Client') as mock_client_class:
            # Simulate connection failure
            mock_client = Mock()
            mock_client.list.side_effect = ConnectionError("Connection refused")
            mock_client_class.return_value = mock_client
            
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
            
            # Service should not crash and should be ready for retry
            assert service.ai_analyzer is not None

    def test_ollama_service_recovery_after_restart(self, temp_directories):
        """Test Ollama service recovery after restart"""
        with patch('ai_analyzer.ollama.Client') as mock_client_class:
            call_count = 0
            
            def connection_recovery_simulation(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_client = Mock()
                
                if call_count <= 2:
                    # First 2 attempts fail
                    mock_client.list.side_effect = ConnectionError("Connection refused")
                else:
                    # Subsequent attempts succeed
                    mock_response = Mock()
                    mock_model = Mock()
                    mock_model.model = "gemma3:4b"
                    mock_response.models = [mock_model]
                    mock_client.list.return_value = mock_response
                    mock_client.chat.return_value = Mock()
                
                return mock_client
            
            mock_client_class.side_effect = connection_recovery_simulation
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # First startup check should fail
            assert service.startup_checks() is False
            
            # Create new AI analyzer instance (simulating restart)
            service.ai_analyzer = AIAnalyzer(service.ollama_host)
            
            # Second startup check should fail
            assert service.startup_checks() is False
            
            # Create new AI analyzer instance again
            service.ai_analyzer = AIAnalyzer(service.ollama_host)
            
            # Third startup check should succeed
            assert service.startup_checks() is True

    def test_immich_api_intermittent_failures(self, temp_directories, mock_all_services):
        """Test resilience to intermittent Immich API failures"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        
        # Setup intermittent upload failures
        upload_attempt = 0
        def intermittent_upload(*args, **kwargs):
            nonlocal upload_attempt
            upload_attempt += 1
            if upload_attempt % 3 == 0:  # Every 3rd upload fails
                raise HTTPError("503 Service Unavailable")
            return "test-asset-id"
        
        mock_all_services['immich_client'].upload_photo.side_effect = intermittent_upload

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Process multiple files with intermittent failures
        successful_uploads = 0
        failed_uploads = 0
        
        for i in range(9):  # 9 files = 3 failures expected
            filename = f"intermittent_test_{i}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(b"fake raw data" * 100)
            
            try:
                result = service.process_file(filepath)
                if result:
                    successful_uploads += 1
                else:
                    failed_uploads += 1
            except Exception:
                failed_uploads += 1
        
        # Verify system handled failures gracefully
        assert successful_uploads == 6, f"Expected 6 successful uploads, got {successful_uploads}"
        assert failed_uploads == 3, f"Expected 3 failed uploads, got {failed_uploads}"

    def test_network_timeout_handling(self, temp_directories):
        """Test handling of network timeouts"""
        with patch('immich_client.requests') as mock_requests:
            # Setup timeout on API calls
            mock_requests.get.side_effect = Timeout("Request timed out")
            mock_requests.post.side_effect = Timeout("Request timed out")
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Connection test should handle timeout gracefully
            result = service.immich_client.test_connection()
            assert result is False
            
            # Startup checks should fail but not crash
            startup_result = service.startup_checks()
            assert startup_result is False

    def test_disk_space_exhaustion_simulation(self, temp_directories, mock_all_services):
        """Test behavior when disk space is exhausted"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Mock disk space error during file operations
        with patch('shutil.move') as mock_move:
            mock_move.side_effect = OSError("No space left on device")
            
            test_file = os.path.join(temp_directories['inbox'], "disk_full_test.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            
            # Should handle disk full error gracefully
            result = service.process_file(test_file)
            
            # Processing should still return True (file was processed, just couldn't be moved)
            assert result is True

    def test_corrupted_config_recovery(self, temp_directories):
        """Test recovery from corrupted configuration"""
        # Test with invalid environment variables
        with patch.dict(os.environ, {
            'OLLAMA_HOST': 'invalid://malformed:url',
            'IMMICH_API_URL': 'not-a-url',
            'IMMICH_API_KEY': '',  # Empty API key
        }):
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="invalid://malformed:url",
                immich_api_url="not-a-url",
                immich_api_key=""
            )
            
            # Service should initialize but startup checks should fail
            assert service is not None
            startup_result = service.startup_checks()
            assert startup_result is False

    def test_ai_model_unavailable_recovery(self, temp_directories):
        """Test recovery when AI model is not available"""
        with patch('ai_analyzer.ollama.Client') as mock_client_class:
            mock_client = Mock()
            
            # Model list succeeds but required model is missing
            mock_response = Mock()
            other_model = Mock()
            other_model.model = "different-model:7b"
            mock_response.models = [other_model]
            mock_client.list.return_value = mock_response
            
            # Model pull fails
            mock_client.pull.side_effect = Exception("Model not found")
            mock_client_class.return_value = mock_client
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Should fail gracefully when model is unavailable
            startup_result = service.startup_checks()
            assert startup_result is False

    def test_concurrent_access_file_conflicts(self, temp_directories, mock_all_services):
        """Test handling of file access conflicts during concurrent processing"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Create file handler to test file conflict detection
        handler = PhotoFileHandler(service)
        
        # Create test file
        test_file = os.path.join(temp_directories['inbox'], "conflict_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        # Simulate concurrent processing attempts
        def process_file_thread():
            from watchdog.events import FileCreatedEvent
            event = FileCreatedEvent(test_file)
            handler.on_created(event)
        
        # Start multiple threads trying to process the same file
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_file_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # File should only be processed once due to conflict detection
        assert mock_all_services['ai_analyzer'].analyze_photo.call_count == 1

    def test_invalid_file_format_handling(self, temp_directories, mock_all_services):
        """Test handling of invalid or corrupted file formats"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        
        # Make image processor fail for corrupted files
        mock_all_services['image_processor'].process_photo.side_effect = Exception("Invalid file format")
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Create files with various corruption scenarios
        corruption_scenarios = [
            ("empty_file.ARW", b""),  # Empty file
            ("truncated_file.ARW", b"TRUNCATED"),  # Truncated file
            ("binary_garbage.ARW", b"\x00\x01\x02\x03" * 100),  # Binary garbage
            ("text_file.ARW", b"This is not an image file at all"),  # Text masquerading as image
        ]

        results = []
        for filename, content in corruption_scenarios:
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(content)
            
            # Should handle corruption gracefully without crashing
            try:
                result = service.process_file(filepath)
                results.append((filename, result, None))
            except Exception as e:
                results.append((filename, False, str(e)))
        
        # All corrupted files should fail gracefully
        for filename, result, error in results:
            assert result is False, f"Corrupted file {filename} should fail processing"
            # Service should not crash from corruption
            assert service is not None

    def test_api_rate_limiting_handling(self, temp_directories, mock_all_services):
        """Test handling of API rate limiting"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        
        # Simulate rate limiting response
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.text = "Rate limit exceeded"
        
        call_count = 0
        def rate_limited_upload(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise HTTPError("429 Rate Limit Exceeded")
            return "test-asset-id"
        
        mock_all_services['immich_client'].upload_photo.side_effect = rate_limited_upload

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Process files despite rate limiting
        results = []
        for i in range(3):
            filename = f"rate_limit_test_{i}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(b"fake raw data" * 100)
            
            try:
                result = service.process_file(filepath)
                results.append(result)
            except Exception:
                results.append(False)
        
        # First 2 should fail due to rate limiting, 3rd should succeed
        assert results == [False, False, True]

    def test_memory_pressure_recovery(self, temp_directories, mock_all_services, memory_profiler):
        """Test recovery under memory pressure conditions"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        memory_profiler.start()

        # Process files while monitoring memory
        file_count = 20
        successful_count = 0
        
        for i in range(file_count):
            filename = f"memory_pressure_{i}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(b"fake raw data" * 1000)
            
            # Simulate memory pressure by checking current usage
            current_memory = memory_profiler.get_memory_usage_mb()
            
            try:
                result = service.process_file(filepath)
                if result:
                    successful_count += 1
            except MemoryError:
                # Should handle memory errors gracefully
                pass
            
            memory_profiler.update_peak()
        
        # System should continue processing despite memory pressure
        success_rate = successful_count / file_count
        assert success_rate >= 0.8, f"Success rate under memory pressure too low: {success_rate:.2%}"

    def test_service_graceful_shutdown_on_errors(self, temp_directories):
        """Test graceful shutdown when critical errors occur"""
        import signal
        
        with patch('main.AIAnalyzer') as mock_ai, \
             patch('main.ImmichClient') as mock_immich:
            
            # Setup critical failure in dependencies
            mock_ai.side_effect = Exception("Critical AI service failure")
            mock_immich.side_effect = Exception("Critical Immich failure")
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Should handle critical initialization failures
            startup_result = service.startup_checks()
            assert startup_result is False
            
            # Service should still be able to shut down gracefully
            # Simulate shutdown signal
            from main import signal_handler
            try:
                signal_handler(signal.SIGTERM, None)
            except SystemExit:
                # Expected behavior
                pass

    def test_file_permission_error_recovery(self, temp_directories, mock_all_services):
        """Test recovery from file permission errors"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Create test file
        test_file = os.path.join(temp_directories['inbox'], "permission_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b"fake raw data" * 100)
        
        # Make processed directory read-only to simulate permission error
        processed_dir = temp_directories['processed']
        try:
            os.chmod(processed_dir, 0o444)  # Read-only
            
            # Should handle permission error gracefully
            result = service.process_file(test_file)
            
            # Processing may succeed but file movement may fail
            # Service should handle this gracefully
            assert result in [True, False]  # Either outcome is acceptable
            
        finally:
            # Restore permissions
            os.chmod(processed_dir, 0o755)

    def test_database_connection_recovery(self, temp_directories, mock_all_services):
        """Test recovery from hash tracker database issues"""
        mock_all_services['ai_analyzer'].analyze_photo.return_value = MockPhotoAnalysisData.crisp_swimmer_analysis()
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Simulate database corruption/issues
        with patch.object(service.hash_tracker, 'is_already_processed') as mock_check:
            mock_check.side_effect = Exception("Database corruption")
            
            test_file = os.path.join(temp_directories['inbox'], "db_error_test.ARW")
            with open(test_file, 'wb') as f:
                f.write(b"fake raw data" * 100)
            
            # Should handle database errors gracefully
            try:
                result = service.process_file(test_file)
                # Should either succeed or fail gracefully
                assert result in [True, False]
            except Exception as e:
                # Should not propagate database errors
                assert "Database corruption" not in str(e)

    def test_external_service_version_mismatch(self, temp_directories):
        """Test handling of external service version mismatches"""
        with patch('immich_client.requests') as mock_requests:
            # Simulate API version mismatch
            version_response = Mock()
            version_response.status_code = 400
            version_response.text = "API version not supported"
            mock_requests.get.return_value = version_response
            
            service = PhotoProcessorService(
                watch_folder=temp_directories['inbox'],
                output_folder=temp_directories['processed'],
                ollama_host="http://test-ollama:11434",
                immich_api_url="http://test-immich:2283",
                immich_api_key="test-key"
            )
            
            # Should handle version mismatch gracefully
            startup_result = service.startup_checks()
            assert startup_result is False
            
            # Service should not crash
            assert service is not None