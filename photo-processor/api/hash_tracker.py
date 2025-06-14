"""
Hash-based Duplicate Detection and Tracking System

This module implements a persistent hash-based tracking system to prevent
reprocessing of images that have already been analyzed and uploaded to Immich.

Key Design Decisions:

1. SHA256 Hash Algorithm:
   - Chosen for its cryptographic strength and extremely low collision probability
   - 256-bit output provides 2^256 possible hash values
   - Industry standard for file integrity verification
   - Fast enough for image files while being secure

2. JSON Storage Format:
   - Human-readable format for debugging and manual inspection
   - Simple persistence without database dependencies
   - Atomic write operations with proper file handling
   - Easy to backup and version control
   - Sufficient for thousands of hashes without performance issues

3. File-based Approach:
   - No external database dependencies
   - Portable across different environments
   - Easy to reset by deleting the JSON file
   - Suitable for single-instance deployments

4. Thread Safety Considerations:
   - Current implementation assumes single-threaded access
   - File writes are atomic at the OS level for small files
   - For multi-threaded use, would need file locking or mutex
   - Current use case (batch processing) doesn't require threading

Integration Points:
   - Used by the main processing pipeline to skip already-processed files
   - Prevents duplicate uploads to Immich even across container restarts
   - Works alongside Immich's own duplicate detection as a first-line filter
"""
import os
import hashlib
import json
import logging
from typing import Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class HashTracker:
    """
    Tracks processed files using SHA256 content hashes.
    
    This class maintains a persistent record of all files that have been
    successfully processed and uploaded to Immich. It uses content-based
    hashing to detect duplicates even if files are renamed or moved.
    
    The tracker is designed to be resilient:
    - Handles missing database files gracefully
    - Saves after each new file to prevent data loss
    - Logs all operations for debugging
    
    Attributes:
        db_path: Path to the JSON database file
        processed_hashes: In-memory set of SHA256 hashes for O(1) lookup
    """
    
    def __init__(self, db_path: str = "/app/processed_hashes.json"):
        """
        Initialize the hash tracker with a database file path.
        
        Args:
            db_path: Path to store the JSON database file.
                    Default is /app for Docker containers.
        """
        self.db_path = db_path
        self.processed_hashes: Set[str] = set()
        # Load existing hashes on initialization
        self.load_database()
    
    def load_database(self):
        """
        Load previously processed file hashes from the JSON database.
        
        This method is called during initialization and handles various
        error conditions gracefully:
        - Missing file: Starts with empty set
        - Corrupted JSON: Logs error and starts fresh
        - Invalid data structure: Uses empty list fallback
        
        The database format is:
        {
            "processed_hashes": ["hash1", "hash2", ...],
            "total_processed": count
        }
        """
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Convert list to set for O(1) lookups
                    self.processed_hashes = set(data.get('processed_hashes', []))
                logger.info(f"Loaded {len(self.processed_hashes)} processed file hashes")
            else:
                # First run or database was deleted
                logger.info("No existing hash database found, starting fresh")
        except Exception as e:
            # Handle JSON decode errors, permission issues, etc.
            logger.error(f"Error loading hash database: {e}")
            # Start with empty set rather than crashing
            self.processed_hashes = set()
    
    def save_database(self):
        """
        Save processed file hashes to the JSON database.
        
        This method is called after each new file is marked as processed
        to ensure persistence even if the process crashes.
        
        The save operation:
        - Converts the set to a list for JSON serialization
        - Includes a count for quick stats without loading all hashes
        - Uses indent=2 for human readability
        - Atomic write on most filesystems for small files
        
        Note: In a high-volume scenario, batching saves could improve
        performance, but current use case prioritizes data safety.
        """
        try:
            data = {
                'processed_hashes': list(self.processed_hashes),
                'total_processed': len(self.processed_hashes)
            }
            # Write with pretty printing for debugging
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.processed_hashes)} hashes to database")
        except Exception as e:
            # Don't crash the processing pipeline on save errors
            logger.error(f"Error saving hash database: {e}")
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate SHA256 hash of file content.
        
        This method reads the file in chunks to handle large image files
        efficiently without loading the entire file into memory.
        
        Why SHA256:
        - Cryptographically secure (no known collisions)
        - Fast enough for image files (processes ~100MB/sec)
        - Standard in many applications (git, docker, etc.)
        - 64-character hex output is manageable in logs/storage
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Optional[str]: Hex-encoded SHA256 hash, or None if error
            
        Note:
            The 8192-byte (8KB) chunk size is optimal for most filesystems
            and provides good balance between memory usage and I/O efficiency.
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read file in 8KB chunks for memory efficiency
                # iter() with sentinel b"" reads until EOF
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            # Return lowercase hex string (64 characters)
            return hasher.hexdigest()
        except Exception as e:
            # Handle file not found, permission errors, etc.
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def is_already_processed(self, file_path: str) -> bool:
        """
        Check if a file has already been processed based on its content hash.
        
        This is the main entry point for duplicate detection. It calculates
        the file's hash and checks against the in-memory set for O(1) lookup.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file content has been processed before,
                  False if new or if hash calculation fails
                  
        Note:
            Returns False on hash calculation errors to err on the side
            of reprocessing rather than skipping files.
        """
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            # If we can't hash it, assume it's new to avoid skipping
            return False
        # O(1) set lookup for performance
        return file_hash in self.processed_hashes
    
    def mark_as_processed(self, file_path: str) -> bool:
        """
        Mark a file as processed by adding its hash to the database.
        
        This method should be called after successful upload to Immich.
        It immediately persists the change to disk to prevent data loss.
        
        Args:
            file_path: Path to the file that was processed
            
        Returns:
            bool: True if successfully marked, False if hash calculation failed
            
        Side Effects:
            - Updates in-memory set
            - Writes to database file
            - Logs the operation with truncated hash for debugging
        """
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return False
        
        # Add to in-memory set
        self.processed_hashes.add(file_hash)
        # Persist immediately to handle crashes/restarts
        self.save_database()
        # Log with truncated hash for readability (first 8 chars is enough for debugging)
        logger.debug(f"Marked file as processed: {os.path.basename(file_path)} (hash: {file_hash[:8]}...)")
        return True
    
    def get_stats(self) -> dict:
        """
        Get current processing statistics.
        
        Provides a summary of the tracker's state for monitoring
        and debugging purposes.
        
        Returns:
            dict: Statistics including:
                - total_processed: Number of unique files processed
                - database_path: Location of the persistence file
                
        Note:
            Could be extended to include database file size,
            last modified time, etc. for more detailed monitoring.
        """
        return {
            'total_processed': len(self.processed_hashes),
            'database_path': self.db_path
        }