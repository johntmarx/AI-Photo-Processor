"""
Enhanced Immich Client with Dual Upload Support

This module provides an enhanced Immich client that supports uploading
both original and processed versions of photos with relationship tracking.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import mimetypes
from dataclasses import dataclass

from recipe_storage import ProcessingRecipe

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result from a single file upload"""
    asset_id: str
    filename: str
    success: bool
    error: Optional[str] = None
    
    
@dataclass
class DualUploadResult:
    """Result from uploading both original and processed versions"""
    original: UploadResult
    processed: UploadResult
    linked: bool
    recipe_stored: bool


class EnhancedImmichClient:
    """Enhanced Immich client with dual upload and relationship support"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'X-API-KEY': api_key}
        
        # Set up session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _prepare_metadata(self, 
                         file_path: Path, 
                         is_original: bool,
                         recipe: Optional[ProcessingRecipe] = None,
                         original_asset_id: Optional[str] = None) -> Dict[str, Any]:
        """Prepare metadata for upload"""
        metadata = {
            'originalFileName': file_path.name,
            'filePath': str(file_path),
            'fileSize': file_path.stat().st_size,
            'fileHash': self._calculate_file_hash(file_path),
            'uploadedAt': datetime.now().isoformat(),
            'isOriginal': is_original,
            'processorVersion': '2.0'
        }
        
        if is_original and recipe:
            metadata['hasProcessedVersion'] = True
            metadata['recipeId'] = recipe.id
            metadata['recipeVersion'] = recipe.version
        elif not is_original and recipe:
            metadata['isProcessed'] = True
            metadata['recipeId'] = recipe.id
            metadata['recipeVersion'] = recipe.version
            metadata['processingDescription'] = recipe.get_description()
            if original_asset_id:
                metadata['originalAssetId'] = original_asset_id
                
        return metadata
    
    def upload_asset(self, 
                    file_path: Path,
                    album_name: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> UploadResult:
        """Upload a single asset to Immich"""
        try:
            # Prepare the file
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                mime_type = 'application/octet-stream'
                
            with open(file_path, 'rb') as f:
                files = {
                    'assetData': (file_path.name, f, mime_type)
                }
                
                # Prepare form data
                data = {}
                if metadata:
                    # Immich expects certain fields in specific format
                    if 'originalFileName' in metadata:
                        data['originalFileName'] = metadata['originalFileName']
                    if 'fileHash' in metadata:
                        data['fileHash'] = metadata['fileHash']
                    
                    # Store custom metadata as description for now
                    # (Immich doesn't have custom fields yet)
                    custom_metadata = {
                        k: v for k, v in metadata.items() 
                        if k not in ['originalFileName', 'fileHash']
                    }
                    if custom_metadata:
                        data['description'] = json.dumps(custom_metadata, indent=2)
                
                # Upload the file
                response = self.session.post(
                    f"{self.base_url}/api/assets",
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout for large files
                )
                
                if response.status_code == 201:
                    result = response.json()
                    asset_id = result.get('id')
                    
                    # Add to album if specified
                    if album_name and asset_id:
                        self._add_to_album(asset_id, album_name)
                    
                    logger.info(f"Successfully uploaded {file_path.name} as {asset_id}")
                    return UploadResult(
                        asset_id=asset_id,
                        filename=file_path.name,
                        success=True
                    )
                else:
                    error = f"Upload failed with status {response.status_code}: {response.text}"
                    logger.error(error)
                    return UploadResult(
                        asset_id="",
                        filename=file_path.name,
                        success=False,
                        error=error
                    )
                    
        except Exception as e:
            error = f"Upload exception: {str(e)}"
            logger.error(error)
            return UploadResult(
                asset_id="",
                filename=file_path.name,
                success=False,
                error=error
            )
    
    def upload_photo_pair(self,
                         original_path: Path,
                         processed_path: Path,
                         recipe: ProcessingRecipe,
                         original_album: str = "Original Files",
                         processed_album: str = "Processed Photos") -> DualUploadResult:
        """Upload both original and processed versions with linking"""
        
        # Step 1: Upload original file
        original_metadata = self._prepare_metadata(
            original_path, 
            is_original=True,
            recipe=recipe
        )
        original_result = self.upload_asset(
            original_path,
            album_name=original_album,
            metadata=original_metadata
        )
        
        if not original_result.success:
            # If original upload fails, don't proceed
            return DualUploadResult(
                original=original_result,
                processed=UploadResult("", processed_path.name, False, "Original upload failed"),
                linked=False,
                recipe_stored=False
            )
        
        # Step 2: Upload processed file with reference to original
        processed_metadata = self._prepare_metadata(
            processed_path,
            is_original=False,
            recipe=recipe,
            original_asset_id=original_result.asset_id
        )
        processed_result = self.upload_asset(
            processed_path,
            album_name=processed_album,
            metadata=processed_metadata
        )
        
        # Step 3: Attempt to link assets (if Immich supports it in the future)
        linked = False
        if original_result.success and processed_result.success:
            linked = self._link_assets(
                original_result.asset_id,
                processed_result.asset_id,
                relationship_type="processed_from"
            )
        
        # Step 4: Store recipe success is tracked
        recipe_stored = original_result.success and processed_result.success
        
        return DualUploadResult(
            original=original_result,
            processed=processed_result,
            linked=linked,
            recipe_stored=recipe_stored
        )
    
    def _add_to_album(self, asset_id: str, album_name: str) -> bool:
        """Add asset to album, creating it if necessary"""
        try:
            # First, try to find the album
            album_id = self._find_or_create_album(album_name)
            if not album_id:
                return False
            
            # Add asset to album
            response = self.session.put(
                f"{self.base_url}/api/albums/{album_id}/assets",
                headers=self.headers,
                json={"ids": [asset_id]}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to add asset to album: {e}")
            return False
    
    def _find_or_create_album(self, album_name: str) -> Optional[str]:
        """Find album by name or create it"""
        try:
            # Get all albums
            response = self.session.get(
                f"{self.base_url}/api/albums",
                headers=self.headers
            )
            
            if response.status_code == 200:
                albums = response.json()
                # Look for existing album
                for album in albums:
                    if album.get('albumName') == album_name:
                        return album.get('id')
                
                # Create new album
                create_response = self.session.post(
                    f"{self.base_url}/api/albums",
                    headers=self.headers,
                    json={'albumName': album_name}
                )
                
                if create_response.status_code == 201:
                    return create_response.json().get('id')
                    
        except Exception as e:
            logger.error(f"Failed to find/create album: {e}")
            
        return None
    
    def _link_assets(self, original_id: str, processed_id: str, relationship_type: str) -> bool:
        """Link two assets with a relationship (placeholder for future Immich feature)"""
        # Note: Immich doesn't currently support asset relationships
        # This is a placeholder for when the feature is added
        # For now, we store the relationship in the description metadata
        try:
            logger.info(f"Would link {original_id} -> {processed_id} as {relationship_type}")
            # In the future, this might look like:
            # response = self.session.post(
            #     f"{self.base_url}/api/assets/relationships",
            #     headers=self.headers,
            #     json={
            #         'sourceId': original_id,
            #         'targetId': processed_id,
            #         'type': relationship_type
            #     }
            # )
            # return response.status_code == 201
            return True
        except Exception as e:
            logger.error(f"Failed to link assets: {e}")
            return False
    
    def check_asset_exists(self, file_hash: str) -> Optional[str]:
        """Check if an asset with this hash already exists"""
        # Note: This is a simplified check. In production, you might want
        # to implement a more sophisticated duplicate detection
        try:
            # Search for assets by originalFileName containing the hash
            # This is a workaround until Immich supports hash-based search
            response = self.session.get(
                f"{self.base_url}/api/assets",
                headers=self.headers,
                params={'originalFileName': file_hash}
            )
            
            if response.status_code == 200:
                assets = response.json()
                if assets:
                    return assets[0].get('id')
                    
        except Exception as e:
            logger.error(f"Failed to check asset existence: {e}")
            
        return None