"""
Immich API Client Module

This module provides a Python interface to the Immich photo management platform API.
It handles authentication, album management, and photo uploads with AI-generated metadata.

The client is designed to be resilient with proper error handling and logging,
making it suitable for automated photo processing pipelines.

Key Features:
- Connection testing with the Immich server
- Album creation and management (get-or-create pattern)
- Photo uploads with metadata and AI analysis results
- Automatic duplicate detection based on Immich's response
- Structured error handling and comprehensive logging

Integration Points:
- Uses PhotoAnalysis schema from schemas.py for structured AI metadata
- Integrates with Immich's REST API (see immich-api-documentation.md)
- Designed to work with the photo processing pipeline in main.py
"""
import os
import requests
import logging
from typing import Optional, Dict, Any
from schemas import PhotoAnalysis

logger = logging.getLogger(__name__)

class ImmichClient:
    """
    Client for interacting with the Immich photo management API.
    
    This client provides methods for uploading photos, creating albums,
    and managing assets in an Immich instance. It's designed to be used
    in automated workflows where photos are processed and then uploaded
    with enriched metadata.
    
    Attributes:
        api_url: Base URL of the Immich API (e.g., 'http://localhost:2283')
        api_key: API key for authentication (generated in Immich user settings)
        headers: Default headers including authentication for JSON requests
    """
    
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the Immich API client.
        
        Args:
            api_url: Base URL of the Immich instance (trailing slash removed automatically)
            api_key: API key for authentication
        """
        # Remove trailing slash to ensure consistent URL construction
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        # Default headers for JSON requests - note that file uploads use different headers
        self.headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        Test connection to Immich API using the ping endpoint.
        
        This method is called during initialization to verify that:
        1. The Immich server is reachable
        2. The API key is valid
        3. The API is responding correctly
        
        The ping endpoint is lightweight and doesn't require any permissions
        beyond a valid API key, making it ideal for connection testing.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Use a reasonable timeout to avoid hanging on network issues
            response = requests.get(
                f"{self.api_url}/api/server/ping",
                headers={'x-api-key': self.api_key},
                timeout=10  # 10 second timeout for connection test
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Immich API")
                return True
            else:
                # Log the status code to help diagnose authentication vs server issues
                logger.error(f"Immich API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            # Catch all exceptions including network errors, timeouts, etc.
            logger.error(f"Failed to connect to Immich API: {e}")
            return False
    
    def create_album(self, album_name: str, description: str = "") -> Optional[str]:
        """
        Create a new album in Immich.
        
        Albums in Immich are containers for organizing photos. This method
        creates a new album that can be used to group related photos together.
        
        Args:
            album_name: Name of the album to create
            description: Optional description for the album
            
        Returns:
            Optional[str]: Album ID if successful, None if creation failed
            
        Note:
            This method doesn't check if an album with the same name already exists.
            Use get_or_create_album() for idempotent album creation.
        """
        try:
            # Immich API expects camelCase field names
            data = {
                "albumName": album_name,
                "description": description
            }
            
            response = requests.post(
                f"{self.api_url}/api/albums",
                headers=self.headers,
                json=data,
                timeout=30  # 30 second timeout for album creation
            )
            
            if response.status_code == 201:  # 201 Created is the expected response
                album_data = response.json()
                album_id = album_data.get('id')
                logger.info(f"Created album '{album_name}' with ID: {album_id}")
                return album_id
            else:
                # Log full response for debugging API issues
                logger.error(f"Failed to create album: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            # Handle any exceptions (network, JSON parsing, etc.)
            logger.error(f"Error creating album: {e}")
            return None
    
    def get_or_create_album(self, album_name: str) -> Optional[str]:
        """
        Get existing album by name or create a new one if it doesn't exist.
        
        This method implements an idempotent pattern for album management,
        ensuring that we don't create duplicate albums with the same name.
        It's the preferred method for album operations in automated workflows.
        
        The method works by:
        1. Fetching all albums from the API
        2. Searching for an album with the exact name match
        3. Creating a new album only if no match is found
        
        Args:
            album_name: Name of the album to find or create
            
        Returns:
            Optional[str]: Album ID if successful, None if operation failed
            
        Note:
            This approach fetches all albums each time, which could be inefficient
            with thousands of albums. A future optimization could cache album
            names/IDs or use a search API if Immich provides one.
        """
        try:
            # First, try to find existing album by fetching all albums
            response = requests.get(
                f"{self.api_url}/api/albums",
                headers={'x-api-key': self.api_key},
                timeout=30
            )
            
            if response.status_code == 200:
                albums = response.json()
                # Linear search through albums - case-sensitive exact match
                for album in albums:
                    if album.get('albumName') == album_name:
                        album_id = album.get('id')
                        logger.info(f"Found existing album '{album_name}' with ID: {album_id}")
                        return album_id
            
            # Album doesn't exist, create it with a descriptive default
            return self.create_album(album_name, "Auto-processed photos from AI photo pipeline")
            
        except Exception as e:
            logger.error(f"Error getting/creating album: {e}")
            return None
    
    def upload_photo(self, 
                    image_path: str, 
                    original_filename: str,
                    analysis: PhotoAnalysis,
                    album_id: Optional[str] = None) -> Optional[str]:
        """
        Upload a processed photo to Immich with AI-generated metadata.
        
        This is the main upload method that combines the processed image file
        with AI analysis results to create a rich, searchable photo entry in Immich.
        
        The method handles:
        - File upload with proper multipart/form-data encoding
        - Metadata attachment including AI-generated descriptions
        - Duplicate detection based on server response
        - Optional album assignment after successful upload
        
        Args:
            image_path: Path to the processed image file to upload
            original_filename: Original filename to preserve in metadata
            analysis: PhotoAnalysis object containing AI analysis results
            album_id: Optional album ID to add the photo to after upload
            
        Returns:
            Optional[str]: Asset ID if successful, "duplicate" if already exists,
                          None if upload failed
            
        Note:
            Immich uses multipart/form-data for file uploads, which requires
            different headers than JSON requests. The 'assetData' field name
            is required by the Immich API.
        """
        try:
            # Create a rich description combining all AI analysis insights
            description = self._create_description(analysis, original_filename)
            
            # Open file in binary mode for upload
            with open(image_path, 'rb') as image_file:
                # Immich expects the file in the 'assetData' field
                files = {
                    'assetData': (original_filename, image_file, 'image/jpeg')
                }
                
                # Prepare metadata for the upload
                from datetime import datetime
                current_time = datetime.now().isoformat()
                data = {
                    # Unique device asset ID to prevent duplicates from same source
                    'deviceAssetId': f"ai-processed-{original_filename}",
                    # Identify uploads from this processor
                    'deviceId': 'ai-photo-processor',
                    # Use current time as we don't have original EXIF data
                    'fileCreatedAt': current_time,
                    'fileModifiedAt': current_time,
                    # Note: multipart/form-data requires string values, not boolean
                    'isFavorite': 'false'
                }
                
                # Add AI-generated description to metadata
                if description:
                    data['description'] = description
                
                # For file uploads, we only need the API key header
                # Content-Type is automatically set by requests for multipart/form-data
                upload_headers = {'x-api-key': self.api_key}
                response = requests.post(
                    f"{self.api_url}/api/assets",
                    headers=upload_headers,
                    files=files,
                    data=data,
                    timeout=60  # Longer timeout for file uploads
                )
                
                if response.status_code in [200, 201]:  # Both are valid success codes
                    asset_data = response.json()
                    asset_id = asset_data.get('id')
                    logger.info(f"Successfully uploaded {original_filename} with asset ID: {asset_id}")
                    
                    # Add to album if specified - this is a separate API call
                    if album_id and asset_id:
                        self._add_to_album(album_id, asset_id)
                    
                    return asset_id
                elif response.status_code == 400 and "duplicate" in response.text.lower():
                    # Immich returns 400 with "duplicate" message when file already exists
                    # This is not an error - it's expected behavior for idempotent uploads
                    logger.warning(f"Duplicate photo detected for {original_filename}, skipping upload")
                    return "duplicate"  # Special return value for caller to handle
                else:
                    # Log full response for debugging unexpected errors
                    logger.error(f"Failed to upload photo: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            # Catch all exceptions including file I/O, network errors, etc.
            logger.error(f"Error uploading photo: {e}")
            return None
    
    def _add_to_album(self, album_id: str, asset_id: str) -> bool:
        """
        Add an uploaded asset to an album.
        
        This is a private method called after successful photo upload
        to organize photos into albums. The Immich API requires a
        separate call to add assets to albums after upload.
        
        Args:
            album_id: ID of the album to add the asset to
            asset_id: ID of the asset to add
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            The API expects an array of IDs even for single assets,
            allowing bulk operations in other contexts.
        """
        try:
            # API expects an array even for single asset
            data = {
                "ids": [asset_id]
            }
            
            # Use PUT method to add assets to album
            response = requests.put(
                f"{self.api_url}/api/albums/{album_id}/assets",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Added asset {asset_id} to album {album_id}")
                return True
            else:
                # Non-fatal error - photo is uploaded but not in album
                logger.error(f"Failed to add asset to album: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding asset to album: {e}")
            return False
    
    def _create_description(self, analysis: PhotoAnalysis, original_filename: str) -> str:
        """
        Create a rich, searchable description from AI analysis results.
        
        This method transforms the structured PhotoAnalysis data into a
        human-readable description that will be stored with the photo in Immich.
        The description is designed to be both informative and searchable.
        
        The description includes:
        - Original filename for traceability
        - AI-generated natural language description
        - Primary subject identification
        - Technical details (aspect ratio, rotation)
        - Swimming-specific context when applicable
        - Color/exposure information when relevant
        
        Args:
            analysis: PhotoAnalysis object with AI results
            original_filename: Original filename for reference
            
        Returns:
            str: Formatted description with emojis for visual structure
            
        Note:
            Emojis are used to make the description more scannable in the
            Immich UI while maintaining searchability of the text content.
        """
        try:
            # Build description parts conditionally based on available data
            description_parts = [
                f"📸 Original: {original_filename}",
                f"🤖 AI Analysis: {analysis.description}",
                f"🎯 Subject: {analysis.primary_subject.replace('_', ' ').title()}",
                f"📐 Aspect: {analysis.recommended_crop.aspect_ratio}",
            ]
            
            # Only include rotation if it's significant (more than 0.1 degrees)
            if abs(analysis.recommended_crop.rotation_degrees) > 0.1:
                description_parts.append(f"🔄 Rotation: {analysis.recommended_crop.rotation_degrees:.1f}°")
            
            # Add swimming-specific context when identified
            if analysis.swimming_context.event_type:
                description_parts.append(f"🏊 Event: {analysis.swimming_context.event_type.replace('_', ' ').title()}")
            
            if analysis.swimming_context.pool_type:
                description_parts.append(f"🏊 Pool: {analysis.swimming_context.pool_type.replace('_', ' ').title()}")
            
            # Only mention exposure if it needs correction
            if analysis.color_analysis.exposure_assessment != 'properly_exposed':
                description_parts.append(f"💡 Exposure: {analysis.color_analysis.exposure_assessment.replace('_', ' ').title()}")
            
            # Join with newlines for readability in Immich UI
            return "\n".join(description_parts)
            
        except Exception as e:
            # Fallback to basic description if analysis parsing fails
            logger.error(f"Error creating description: {e}")
            return f"Auto-processed from {original_filename}"