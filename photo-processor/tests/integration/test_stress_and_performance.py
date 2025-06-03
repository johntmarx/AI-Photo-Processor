"""
Stress testing and performance integration tests for photo processor
"""
import pytest
import tempfile
import os
import time
import threading
import concurrent.futures
import psutil
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np

from main import PhotoProcessorService, PhotoFileHandler
from tests.fixtures.mock_data import MockPhotoAnalysisData, TestFileGenerator


class TestStressAndPerformance:
    """Stress testing and performance test suite"""

    @pytest.fixture
    def performance_service(self, temp_directories, mock_all_services):
        """Create service optimized for performance testing"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )
        return service

    @pytest.mark.slow
    def test_high_volume_file_processing(self, temp_directories, mock_all_services, performance_timer):
        """Test processing large number of files"""
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

        # Create 100 test files
        file_count = 100
        test_files = []
        
        performance_timer.start("file_creation")
        for i in range(file_count):
            filename = f"stress_test_{i:03d}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(b'FAKE_RAW_DATA' * 1000)  # ~13KB per file
            test_files.append(filepath)
        performance_timer.stop("file_creation")

        # Process all files and measure performance
        performance_timer.start("bulk_processing")
        service.process_all_files_in_folder()
        performance_timer.stop("bulk_processing")

        # Verify all files were processed
        assert mock_all_services['ai_analyzer'].analyze_photo.call_count == file_count
        assert mock_all_services['image_processor'].process_photo.call_count == file_count
        assert mock_all_services['immich_client'].upload_photo.call_count == file_count

        # Performance assertions
        bulk_time = performance_timer.get_elapsed("bulk_processing")
        avg_time_per_file = bulk_time / file_count
        
        # Should process at least 10 files per second (with mocks)
        assert avg_time_per_file < 0.1, f"Average processing time {avg_time_per_file:.3f}s per file is too slow"
        
        # Total processing should complete in reasonable time
        performance_timer.assert_faster_than(30.0, "bulk_processing")

    @pytest.mark.slow
    def test_memory_usage_under_high_load(self, temp_directories, mock_all_services, memory_profiler):
        """Test memory usage doesn't grow excessively under high load"""
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
        initial_memory = memory_profiler.get_memory_usage_mb()

        # Process files in batches to monitor memory growth
        batch_size = 20
        total_batches = 5
        
        for batch in range(total_batches):
            # Create batch of files
            for i in range(batch_size):
                filename = f"memory_test_batch{batch}_file{i}.ARW"
                filepath = os.path.join(temp_directories['inbox'], filename)
                with open(filepath, 'wb') as f:
                    f.write(b'FAKE_RAW_DATA' * 2000)  # ~26KB per file
                
                service.process_file(filepath)
            
            memory_profiler.update_peak()
            batch_memory = memory_profiler.get_memory_usage_mb()
            
            # Memory should not grow linearly with file count
            memory_increase = batch_memory - initial_memory
            max_expected_increase = 50 * (batch + 1)  # 50MB per batch max
            
            assert memory_increase < max_expected_increase, \
                f"Memory increased by {memory_increase:.2f}MB after batch {batch}, expected < {max_expected_increase}MB"

        # Final memory usage should be reasonable
        memory_profiler.assert_memory_increase_less_than(200)  # Total increase < 200MB

    @pytest.mark.slow
    def test_concurrent_processing_stress(self, temp_directories, mock_all_services):
        """Test system under high concurrent load"""
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

        # Create test files
        num_threads = 20
        files_per_thread = 5
        test_files = []

        for thread_id in range(num_threads):
            for file_id in range(files_per_thread):
                filename = f"concurrent_t{thread_id}_f{file_id}.ARW"
                filepath = os.path.join(temp_directories['inbox'], filename)
                with open(filepath, 'wb') as f:
                    f.write(b'FAKE_RAW_DATA' * 1000)
                test_files.append(filepath)

        # Process files concurrently
        results = []
        exceptions = []
        start_time = time.time()

        def process_file_batch(file_batch):
            """Process a batch of files in one thread"""
            thread_results = []
            try:
                for filepath in file_batch:
                    result = service.process_file(filepath)
                    thread_results.append((filepath, result))
            except Exception as e:
                exceptions.append(e)
            return thread_results

        # Split files into batches for threads
        file_batches = [test_files[i:i + files_per_thread] for i in range(0, len(test_files), files_per_thread)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_batch = {executor.submit(process_file_batch, batch): batch for batch in file_batches}
            
            for future in concurrent.futures.as_completed(future_to_batch, timeout=60):
                batch_results = future.result()
                results.extend(batch_results)

        processing_time = time.time() - start_time

        # Verify results
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == len(test_files), f"Not all files processed: {len(results)}/{len(test_files)}"
        
        successful_results = [r for r in results if r[1] is True]
        assert len(successful_results) == len(test_files), "Some files failed to process"

        # Performance assertions
        assert processing_time < 30.0, f"Concurrent processing took {processing_time:.2f}s, expected < 30s"
        
        # Should achieve some parallelism benefit
        sequential_estimate = len(test_files) * 0.1  # Estimated 0.1s per file sequentially
        parallel_efficiency = sequential_estimate / processing_time
        assert parallel_efficiency > 2.0, f"Parallel efficiency {parallel_efficiency:.2f}x should be > 2x"

    def test_large_file_handling(self, temp_directories, mock_all_services, performance_timer):
        """Test handling of very large files"""
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

        # Test different large file sizes
        file_sizes_mb = [50, 100, 200]  # Large RAW files
        
        for size_mb in file_sizes_mb:
            filename = f"large_file_{size_mb}mb.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            
            # Create large file
            performance_timer.start(f"create_{size_mb}mb")
            with open(filepath, 'wb') as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                chunk_data = b'X' * chunk_size
                for _ in range(size_mb):
                    f.write(chunk_data)
            performance_timer.stop(f"create_{size_mb}mb")
            
            # Process large file
            performance_timer.start(f"process_{size_mb}mb")
            result = service.process_file(filepath)
            performance_timer.stop(f"process_{size_mb}mb")
            
            assert result is True, f"Failed to process {size_mb}MB file"
            
            # Performance expectations for large files (with mocks, should still be fast)
            if size_mb <= 50:
                performance_timer.assert_faster_than(5.0, f"process_{size_mb}mb")
            elif size_mb <= 100:
                performance_timer.assert_faster_than(10.0, f"process_{size_mb}mb")
            else:
                performance_timer.assert_faster_than(20.0, f"process_{size_mb}mb")
            
            # Cleanup large file
            try:
                os.unlink(filepath)
            except:
                pass

    def test_hash_tracker_performance_with_many_files(self, temp_directories, mock_all_services, stress_test_data):
        """Test hash tracker performance with large number of processed files"""
        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Generate large list of hashes to simulate many processed files
        large_hash_list = stress_test_data.large_hash_list(count=5000)
        
        # Simulate marking many files as processed
        start_time = time.time()
        for i, hash_value in enumerate(large_hash_list):
            # Create fake file path for hash tracking
            fake_filepath = f"/fake/path/file_{i:06d}.jpg"
            service.hash_tracker.processed_hashes.add(hash_value)
        marking_time = time.time() - start_time
        
        # Test lookup performance
        start_time = time.time()
        lookup_count = 1000
        for i in range(lookup_count):
            test_hash = large_hash_list[i % len(large_hash_list)]
            service.hash_tracker.processed_hashes.__contains__(test_hash)
        lookup_time = time.time() - start_time
        
        # Performance assertions
        avg_marking_time = marking_time / len(large_hash_list)
        avg_lookup_time = lookup_time / lookup_count
        
        assert avg_marking_time < 0.001, f"Hash marking too slow: {avg_marking_time:.6f}s per hash"
        assert avg_lookup_time < 0.001, f"Hash lookup too slow: {avg_lookup_time:.6f}s per lookup"

    def test_file_watcher_performance_with_rapid_file_creation(self, temp_directories, mock_all_services):
        """Test file watcher performance under rapid file creation"""
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

        # Create file handler
        handler = PhotoFileHandler(service)
        
        # Simulate rapid file creation events
        file_count = 50
        processed_files = []

        start_time = time.time()
        for i in range(file_count):
            filename = f"rapid_test_{i:03d}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            
            # Create file
            with open(filepath, 'wb') as f:
                f.write(b'FAKE_RAW_DATA' * 100)
            
            # Simulate file creation event
            from watchdog.events import FileCreatedEvent
            event = FileCreatedEvent(filepath)
            handler.on_created(event)
            
            processed_files.append(filepath)
        
        # Wait for processing to complete
        max_wait_time = 30
        wait_start = time.time()
        while (len(handler.processing_files) > 0 and 
               time.time() - wait_start < max_wait_time):
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        # Verify all files were processed
        assert len(handler.processing_files) == 0, "Some files still being processed"
        assert mock_all_services['ai_analyzer'].analyze_photo.call_count == file_count
        
        # Performance assertion
        avg_time_per_file = total_time / file_count
        assert avg_time_per_file < 0.5, f"Average time per file {avg_time_per_file:.3f}s too slow"

    def test_error_recovery_under_stress(self, temp_directories, mock_all_services):
        """Test error recovery behavior under stress conditions"""
        # Setup intermittent failures
        call_count = 0
        def intermittent_ai_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                return None
            return MockPhotoAnalysisData.crisp_swimmer_analysis()
        
        mock_all_services['ai_analyzer'].analyze_photo.side_effect = intermittent_ai_failure
        mock_all_services['image_processor'].process_photo.return_value = True
        mock_all_services['immich_client'].upload_photo.return_value = "test-asset-id"

        service = PhotoProcessorService(
            watch_folder=temp_directories['inbox'],
            output_folder=temp_directories['processed'],
            ollama_host="http://test-ollama:11434",
            immich_api_url="http://test-immich:2283",
            immich_api_key="test-key"
        )

        # Create test files
        file_count = 30
        test_files = []
        for i in range(file_count):
            filename = f"error_recovery_{i:03d}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            with open(filepath, 'wb') as f:
                f.write(b'FAKE_RAW_DATA' * 100)
            test_files.append(filepath)

        # Process files with intermittent failures
        successful_count = 0
        failed_count = 0
        
        for filepath in test_files:
            try:
                result = service.process_file(filepath)
                if result:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        # Verify system handled failures gracefully
        expected_failures = file_count // 3  # Every 3rd should fail
        expected_successes = file_count - expected_failures
        
        assert successful_count >= expected_successes - 2, \
            f"Too few successes: {successful_count}, expected ~{expected_successes}"
        assert failed_count >= expected_failures - 2, \
            f"Too few failures: {failed_count}, expected ~{expected_failures}"
        
        # System should continue processing despite failures
        assert successful_count > 0, "No files processed successfully despite intermittent failures"

    def test_resource_limits_and_cleanup(self, temp_directories, mock_all_services):
        """Test resource usage stays within limits during intensive operations"""
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

        # Monitor system resources
        process = psutil.Process()
        initial_open_files = len(process.open_files())
        initial_memory = process.memory_info().rss
        
        # Process many files to stress resource management
        file_count = 100
        for i in range(file_count):
            filename = f"resource_test_{i:03d}.ARW"
            filepath = os.path.join(temp_directories['inbox'], filename)
            
            with open(filepath, 'wb') as f:
                f.write(b'FAKE_RAW_DATA' * 500)
            
            service.process_file(filepath)
            
            # Check resource usage periodically
            if i % 20 == 0:
                current_open_files = len(process.open_files())
                current_memory = process.memory_info().rss
                
                # File handles should not accumulate
                file_handle_increase = current_open_files - initial_open_files
                assert file_handle_increase < 50, f"Too many open file handles: +{file_handle_increase}"
                
                # Memory should not grow excessively
                memory_increase_mb = (current_memory - initial_memory) / (1024 * 1024)
                assert memory_increase_mb < 500, f"Memory usage too high: +{memory_increase_mb:.2f}MB"

        # Final resource check
        final_open_files = len(process.open_files())
        final_memory = process.memory_info().rss
        
        # Resources should be cleaned up properly
        final_file_increase = final_open_files - initial_open_files
        final_memory_increase = (final_memory - initial_memory) / (1024 * 1024)
        
        assert final_file_increase < 20, f"File handles leaked: +{final_file_increase}"
        assert final_memory_increase < 200, f"Memory leaked: +{final_memory_increase:.2f}MB"

    @pytest.mark.slow
    def test_long_running_stability(self, temp_directories, mock_all_services):
        """Test system stability over extended operation"""
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

        # Simulate long-running operation with periodic file processing
        cycles = 50
        files_per_cycle = 5
        total_processed = 0
        errors = []
        
        start_time = time.time()
        
        for cycle in range(cycles):
            try:
                # Create and process files for this cycle
                for file_idx in range(files_per_cycle):
                    filename = f"stability_c{cycle:03d}_f{file_idx}.ARW"
                    filepath = os.path.join(temp_directories['inbox'], filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(b'FAKE_RAW_DATA' * 200)
                    
                    result = service.process_file(filepath)
                    if result:
                        total_processed += 1
                    
                # Small delay between cycles to simulate real-world timing
                time.sleep(0.01)
                
            except Exception as e:
                errors.append(f"Cycle {cycle}: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Verify system remained stable
        expected_total = cycles * files_per_cycle
        success_rate = total_processed / expected_total
        
        assert len(errors) == 0, f"Errors occurred during long run: {errors}"
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert total_time < 60.0, f"Long-running test took too long: {total_time:.2f}s"
        
        # Verify service is still responsive
        test_file = os.path.join(temp_directories['inbox'], "final_test.ARW")
        with open(test_file, 'wb') as f:
            f.write(b'FAKE_RAW_DATA' * 100)
        
        final_result = service.process_file(test_file)
        assert final_result is True, "Service became unresponsive after long run"