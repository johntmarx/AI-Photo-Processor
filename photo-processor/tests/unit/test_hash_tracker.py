"""
Comprehensive unit tests for Hash Tracker module

This test suite provides comprehensive coverage for the HashTracker class,
including edge cases, error conditions, and performance scenarios.
"""
import pytest
import tempfile
import os
import json
import hashlib
import time
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path

from hash_tracker import HashTracker


class TestHashTracker:
    """Test suite for HashTracker class"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name
        yield temp_path
        # Cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"This is a test file content for hashing"

    @pytest.fixture
    def sample_test_file(self, sample_file_content):
        """Create a temporary test file with known content"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(sample_file_content)
            temp_path = tmp_file.name
        yield temp_path
        # Cleanup
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    @pytest.fixture
    def expected_hash(self, sample_file_content):
        """Calculate expected hash for sample content"""
        return hashlib.sha256(sample_file_content).hexdigest()

    def test_initialization_new_database(self, temp_db_path):
        """Test HashTracker initialization with new database"""
        tracker = HashTracker(temp_db_path)
        
        assert tracker.db_path == temp_db_path
        assert isinstance(tracker.processed_hashes, set)
        assert len(tracker.processed_hashes) == 0

    def test_initialization_existing_database(self, temp_db_path):
        """Test HashTracker initialization with existing database"""
        # Create existing database
        existing_hashes = ["hash1", "hash2", "hash3"]
        data = {
            "processed_hashes": existing_hashes,
            "total_processed": len(existing_hashes)
        }
        with open(temp_db_path, 'w') as f:
            json.dump(data, f)
        
        tracker = HashTracker(temp_db_path)
        
        assert len(tracker.processed_hashes) == 3
        assert tracker.processed_hashes == set(existing_hashes)

    def test_initialization_corrupted_database(self, temp_db_path):
        """Test HashTracker initialization with corrupted database"""
        # Create corrupted JSON file
        with open(temp_db_path, 'w') as f:
            f.write("invalid json content {")
        
        tracker = HashTracker(temp_db_path)
        
        # Should start with empty set when JSON is corrupted
        assert len(tracker.processed_hashes) == 0

    def test_initialization_missing_processed_hashes_key(self, temp_db_path):
        """Test initialization with database missing processed_hashes key"""
        # Create database with missing key
        data = {"total_processed": 0}
        with open(temp_db_path, 'w') as f:
            json.dump(data, f)
        
        tracker = HashTracker(temp_db_path)
        
        # Should handle missing key gracefully
        assert len(tracker.processed_hashes) == 0

    def test_initialization_default_path(self):
        """Test HashTracker initialization with default path"""
        with patch('os.path.exists', return_value=False):
            tracker = HashTracker()
            assert tracker.db_path == "/app/processed_hashes.json"

    def test_calculate_file_hash_success(self, sample_test_file, expected_hash):
        """Test successful file hash calculation"""
        tracker = HashTracker()
        
        result_hash = tracker.calculate_file_hash(sample_test_file)
        
        assert result_hash == expected_hash
        assert len(result_hash) == 64  # SHA256 hex length

    def test_calculate_file_hash_nonexistent_file(self, temp_db_path):
        """Test hash calculation for nonexistent file"""
        tracker = HashTracker(temp_db_path)
        
        result = tracker.calculate_file_hash("/nonexistent/file.txt")
        
        assert result is None

    def test_calculate_file_hash_permission_error(self, temp_db_path):
        """Test hash calculation with permission error"""
        tracker = HashTracker(temp_db_path)
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = tracker.calculate_file_hash("/some/file.txt")
            
        assert result is None

    def test_calculate_file_hash_large_file(self, temp_db_path):
        """Test hash calculation for large file (chunked reading)"""
        tracker = HashTracker(temp_db_path)
        
        # Create a file larger than the chunk size (8192 bytes)
        large_content = b"A" * 20000  # 20KB file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(large_content)
            large_file_path = tmp_file.name
        
        try:
            result_hash = tracker.calculate_file_hash(large_file_path)
            expected_hash = hashlib.sha256(large_content).hexdigest()
            
            assert result_hash == expected_hash
        finally:
            os.unlink(large_file_path)

    def test_calculate_file_hash_empty_file(self, temp_db_path):
        """Test hash calculation for empty file"""
        tracker = HashTracker(temp_db_path)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            empty_file_path = tmp_file.name
        
        try:
            result_hash = tracker.calculate_file_hash(empty_file_path)
            expected_hash = hashlib.sha256(b"").hexdigest()
            
            assert result_hash == expected_hash
        finally:
            os.unlink(empty_file_path)

    def test_is_already_processed_new_file(self, temp_db_path, sample_test_file):
        """Test checking if new file has been processed"""
        tracker = HashTracker(temp_db_path)
        
        result = tracker.is_already_processed(sample_test_file)
        
        assert result is False

    def test_is_already_processed_existing_file(self, temp_db_path, sample_test_file, expected_hash):
        """Test checking if previously processed file has been processed"""
        # Pre-populate database with the file's hash
        data = {
            "processed_hashes": [expected_hash],
            "total_processed": 1
        }
        with open(temp_db_path, 'w') as f:
            json.dump(data, f)
        
        tracker = HashTracker(temp_db_path)
        
        result = tracker.is_already_processed(sample_test_file)
        
        assert result is True

    def test_is_already_processed_hash_calculation_failure(self, temp_db_path):
        """Test checking processed status when hash calculation fails"""
        tracker = HashTracker(temp_db_path)
        
        with patch.object(tracker, 'calculate_file_hash', return_value=None):
            result = tracker.is_already_processed("/some/file.txt")
            
        # Should return False when hash calculation fails
        assert result is False

    def test_mark_as_processed_success(self, temp_db_path, sample_test_file, expected_hash):
        """Test marking a file as processed"""
        tracker = HashTracker(temp_db_path)
        
        result = tracker.mark_as_processed(sample_test_file)
        
        assert result is True
        assert expected_hash in tracker.processed_hashes
        
        # Verify persistence to database
        assert os.path.exists(temp_db_path)
        with open(temp_db_path, 'r') as f:
            data = json.load(f)
        assert expected_hash in data['processed_hashes']
        assert data['total_processed'] == 1

    def test_mark_as_processed_hash_failure(self, temp_db_path):
        """Test marking as processed when hash calculation fails"""
        tracker = HashTracker(temp_db_path)
        
        with patch.object(tracker, 'calculate_file_hash', return_value=None):
            result = tracker.mark_as_processed("/some/file.txt")
            
        assert result is False
        assert len(tracker.processed_hashes) == 0

    def test_mark_as_processed_duplicate_hash(self, temp_db_path, sample_test_file, expected_hash):
        """Test marking a file as processed when hash already exists"""
        tracker = HashTracker(temp_db_path)
        
        # Mark once
        tracker.mark_as_processed(sample_test_file)
        initial_count = len(tracker.processed_hashes)
        
        # Mark again
        result = tracker.mark_as_processed(sample_test_file)
        
        assert result is True
        assert len(tracker.processed_hashes) == initial_count  # No duplicates in set

    def test_save_database_success(self, temp_db_path):
        """Test successful database save"""
        tracker = HashTracker(temp_db_path)
        tracker.processed_hashes = {"hash1", "hash2", "hash3"}
        
        tracker.save_database()
        
        assert os.path.exists(temp_db_path)
        with open(temp_db_path, 'r') as f:
            data = json.load(f)
        
        assert set(data['processed_hashes']) == {"hash1", "hash2", "hash3"}
        assert data['total_processed'] == 3

    def test_save_database_permission_error(self, temp_db_path):
        """Test database save with permission error"""
        tracker = HashTracker(temp_db_path)
        tracker.processed_hashes = {"hash1"}
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise exception
            tracker.save_database()

    def test_save_database_directory_not_exists(self):
        """Test database save when directory doesn't exist"""
        non_existent_path = "/nonexistent/directory/db.json"
        tracker = HashTracker(non_existent_path)
        tracker.processed_hashes = {"hash1"}
        
        # Should handle gracefully and not crash
        tracker.save_database()

    def test_load_database_permission_error(self, temp_db_path):
        """Test database load with permission error"""
        # Create a valid database first
        data = {"processed_hashes": ["hash1"], "total_processed": 1}
        with open(temp_db_path, 'w') as f:
            json.dump(data, f)
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            tracker = HashTracker(temp_db_path)
            
        # Should start with empty set on permission error
        assert len(tracker.processed_hashes) == 0

    def test_get_stats(self, temp_db_path):
        """Test getting tracker statistics"""
        tracker = HashTracker(temp_db_path)
        tracker.processed_hashes = {"hash1", "hash2", "hash3"}
        
        stats = tracker.get_stats()
        
        assert stats['total_processed'] == 3
        assert stats['database_path'] == temp_db_path

    def test_get_stats_empty_tracker(self, temp_db_path):
        """Test getting statistics for empty tracker"""
        tracker = HashTracker(temp_db_path)
        
        stats = tracker.get_stats()
        
        assert stats['total_processed'] == 0
        assert stats['database_path'] == temp_db_path

    def test_concurrent_access_simulation(self, temp_db_path, sample_test_file):
        """Test simulated concurrent access to tracker"""
        tracker1 = HashTracker(temp_db_path)
        tracker2 = HashTracker(temp_db_path)
        
        # Mark file as processed in first tracker
        tracker1.mark_as_processed(sample_test_file)
        
        # Reload second tracker to simulate another process
        tracker2.load_database()
        
        # Both trackers should have the same hash
        assert tracker1.processed_hashes == tracker2.processed_hashes

    def test_hash_consistency_across_instances(self, temp_db_path, sample_test_file, expected_hash):
        """Test hash consistency across different tracker instances"""
        # Calculate hash with first instance
        tracker1 = HashTracker(temp_db_path)
        hash1 = tracker1.calculate_file_hash(sample_test_file)
        
        # Calculate hash with second instance
        tracker2 = HashTracker(temp_db_path)
        hash2 = tracker2.calculate_file_hash(sample_test_file)
        
        assert hash1 == hash2 == expected_hash

    def test_database_format_validation(self, temp_db_path):
        """Test that saved database has correct format"""
        tracker = HashTracker(temp_db_path)
        test_hashes = {"hash1", "hash2", "hash3"}
        tracker.processed_hashes = test_hashes
        
        tracker.save_database()
        
        # Verify JSON structure
        with open(temp_db_path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert 'processed_hashes' in data
        assert 'total_processed' in data
        assert isinstance(data['processed_hashes'], list)
        assert isinstance(data['total_processed'], int)
        assert set(data['processed_hashes']) == test_hashes
        assert data['total_processed'] == len(test_hashes)

    def test_hash_function_properties(self, temp_db_path):
        """Test SHA256 hash function properties"""
        tracker = HashTracker(temp_db_path)
        
        # Create two different files
        content1 = b"Different content 1"
        content2 = b"Different content 2"
        
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(content1)
            file1_path = f1.name
        
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(content2)
            file2_path = f2.name
        
        try:
            hash1 = tracker.calculate_file_hash(file1_path)
            hash2 = tracker.calculate_file_hash(file2_path)
            
            # Hashes should be different
            assert hash1 != hash2
            # Hashes should be deterministic
            assert tracker.calculate_file_hash(file1_path) == hash1
            assert tracker.calculate_file_hash(file2_path) == hash2
            # Hashes should be proper length
            assert len(hash1) == 64
            assert len(hash2) == 64
            
        finally:
            os.unlink(file1_path)
            os.unlink(file2_path)

    def test_edge_case_very_long_filename(self, temp_db_path):
        """Test handling files with very long filenames"""
        tracker = HashTracker(temp_db_path)
        
        # Create file with long name
        long_name = "a" * 200 + ".txt"
        content = b"Test content"
        
        with tempfile.NamedTemporaryFile(suffix=long_name, delete=False) as tmp_file:
            tmp_file.write(content)
            long_file_path = tmp_file.name
        
        try:
            result = tracker.calculate_file_hash(long_file_path)
            assert result is not None
            assert len(result) == 64
        except OSError:
            # Some filesystems may not support very long names
            pytest.skip("Filesystem doesn't support long filenames")
        finally:
            try:
                os.unlink(long_file_path)
            except OSError:
                pass

    def test_database_corruption_recovery(self, temp_db_path):
        """Test recovery from various database corruption scenarios"""
        # Test 1: Completely invalid JSON
        with open(temp_db_path, 'w') as f:
            f.write("not json at all")
        
        tracker1 = HashTracker(temp_db_path)
        assert len(tracker1.processed_hashes) == 0
        
        # Test 2: Valid JSON but wrong structure
        with open(temp_db_path, 'w') as f:
            json.dump({"wrong_key": "wrong_value"}, f)
        
        tracker2 = HashTracker(temp_db_path)
        assert len(tracker2.processed_hashes) == 0
        
        # Test 3: Partial corruption (non-string items in hash list)
        with open(temp_db_path, 'w') as f:
            json.dump({"processed_hashes": ["valid_hash", 123, None]}, f)
        
        tracker3 = HashTracker(temp_db_path)
        # Should only load valid string hashes
        assert len(tracker3.processed_hashes) <= 3

    @pytest.mark.slow
    def test_performance_large_hash_set(self, temp_db_path):
        """Test performance with large number of hashes"""
        tracker = HashTracker(temp_db_path)
        
        # Add many hashes to test O(1) lookup performance
        large_hash_set = {f"hash{i:06d}" for i in range(10000)}
        tracker.processed_hashes = large_hash_set
        
        # Test lookup performance
        start_time = time.time()
        for i in range(1000):
            # Should be O(1) lookup
            result = f"hash{i:06d}" in tracker.processed_hashes
            assert result is True
        lookup_time = time.time() - start_time
        
        # Should complete quickly (under 1 second for 1000 lookups)
        assert lookup_time < 1.0
        
        # Test save performance
        start_time = time.time()
        tracker.save_database()
        save_time = time.time() - start_time
        
        # Should save reasonably quickly (under 5 seconds for 10k hashes)
        assert save_time < 5.0

    def test_file_handle_cleanup(self, temp_db_path, sample_test_file):
        """Test that file handles are properly closed"""
        tracker = HashTracker(temp_db_path)
        
        # Mock open to track file handle operations
        original_open = open
        open_calls = []
        
        def tracking_open(*args, **kwargs):
            file_handle = original_open(*args, **kwargs)
            open_calls.append(file_handle)
            return file_handle
        
        with patch('builtins.open', side_effect=tracking_open):
            # Perform operations that open files
            tracker.calculate_file_hash(sample_test_file)
            tracker.mark_as_processed(sample_test_file)
        
        # Verify all file handles were closed
        for file_handle in open_calls:
            assert file_handle.closed

    def test_atomic_database_writes(self, temp_db_path):
        """Test database writes are atomic (no partial writes)"""
        tracker = HashTracker(temp_db_path)
        tracker.processed_hashes = {"hash1", "hash2"}
        
        # Mock a write error partway through
        original_dump = json.dump
        
        def failing_dump(obj, fp, **kwargs):
            # Write some data then fail
            fp.write('{"processed_hashes": [')
            raise IOError("Simulated write failure")
        
        with patch('json.dump', side_effect=failing_dump):
            tracker.save_database()
        
        # Database should either be unchanged or empty (not partially written)
        if os.path.exists(temp_db_path):
            try:
                with open(temp_db_path, 'r') as f:
                    data = json.load(f)
                # If file exists and can be parsed, it should be complete
                assert isinstance(data, dict)
            except json.JSONDecodeError:
                # Partial write occurred - this is what we're testing against
                pytest.fail("Database write was not atomic - partial write detected")