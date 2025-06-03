"""
Hash-based tracking system to prevent duplicate image processing
"""
import os
import hashlib
import json
import logging
from typing import Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class HashTracker:
    def __init__(self, db_path: str = "/app/processed_hashes.json"):
        self.db_path = db_path
        self.processed_hashes: Set[str] = set()
        self.load_database()
    
    def load_database(self):
        """Load previously processed file hashes from database"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.processed_hashes = set(data.get('processed_hashes', []))
                logger.info(f"Loaded {len(self.processed_hashes)} processed file hashes")
            else:
                logger.info("No existing hash database found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading hash database: {e}")
            self.processed_hashes = set()
    
    def save_database(self):
        """Save processed file hashes to database"""
        try:
            data = {
                'processed_hashes': list(self.processed_hashes),
                'total_processed': len(self.processed_hashes)
            }
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.processed_hashes)} hashes to database")
        except Exception as e:
            logger.error(f"Error saving hash database: {e}")
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 hash of file content"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def is_already_processed(self, file_path: str) -> bool:
        """Check if file has already been processed based on its hash"""
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return False
        return file_hash in self.processed_hashes
    
    def mark_as_processed(self, file_path: str) -> bool:
        """Mark file as processed by adding its hash to the database"""
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return False
        
        self.processed_hashes.add(file_hash)
        self.save_database()
        logger.debug(f"Marked file as processed: {os.path.basename(file_path)} (hash: {file_hash[:8]}...)")
        return True
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return {
            'total_processed': len(self.processed_hashes),
            'database_path': self.db_path
        }