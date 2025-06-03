"""
Immich API client for uploading processed photos
"""
import os
import requests
import logging
from typing import Optional, Dict, Any
from schemas import PhotoAnalysis

logger = logging.getLogger(__name__)

class ImmichClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
    
    def test_connection(self) -> bool:
        """
        Test connection to Immich API
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/server/ping",
                headers={'x-api-key': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Immich API")
                return True
            else:
                logger.error(f"Immich API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Immich API: {e}")
            return False
    
    def create_album(self, album_name: str, description: str = "") -> Optional[str]:
        """
        Create a new album in Immich
        """
        try:
            data = {
                "albumName": album_name,
                "description": description
            }
            
            response = requests.post(
                f"{self.api_url}/api/albums",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 201:
                album_data = response.json()
                album_id = album_data.get('id')
                logger.info(f"Created album '{album_name}' with ID: {album_id}")
                return album_id
            else:
                logger.error(f"Failed to create album: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating album: {e}")
            return None
    
    def get_or_create_album(self, album_name: str) -> Optional[str]:
        """
        Get existing album or create new one
        """
        try:
            # First, try to find existing album
            response = requests.get(
                f"{self.api_url}/api/albums",
                headers={'x-api-key': self.api_key},
                timeout=30
            )
            
            if response.status_code == 200:
                albums = response.json()
                for album in albums:
                    if album.get('albumName') == album_name:
                        album_id = album.get('id')
                        logger.info(f"Found existing album '{album_name}' with ID: {album_id}")
                        return album_id
            
            # Album doesn't exist, create it
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
        Upload processed photo to Immich with metadata
        """
        try:
            # Prepare description with AI analysis
            description = self._create_description(analysis, original_filename)
            
            # Prepare file upload
            with open(image_path, 'rb') as image_file:
                files = {
                    'assetData': (original_filename, image_file, 'image/jpeg')
                }
                
                # Prepare metadata
                from datetime import datetime
                current_time = datetime.now().isoformat()
                data = {
                    'deviceAssetId': f"ai-processed-{original_filename}",
                    'deviceId': 'ai-photo-processor',
                    'fileCreatedAt': current_time,
                    'fileModifiedAt': current_time,
                    'isFavorite': 'false'  # Must be string for multipart/form-data
                }
                
                # Add description if analysis available
                if description:
                    data['description'] = description
                
                # Upload file
                upload_headers = {'x-api-key': self.api_key}
                response = requests.post(
                    f"{self.api_url}/api/assets",
                    headers=upload_headers,
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code in [200, 201]:
                    asset_data = response.json()
                    asset_id = asset_data.get('id')
                    logger.info(f"Successfully uploaded {original_filename} with asset ID: {asset_id}")
                    
                    # Add to album if specified
                    if album_id and asset_id:
                        self._add_to_album(album_id, asset_id)
                    
                    return asset_id
                elif response.status_code == 400 and "duplicate" in response.text.lower():
                    logger.warning(f"Duplicate photo detected for {original_filename}, skipping upload")
                    return "duplicate"  # Return special value to indicate duplicate
                else:
                    logger.error(f"Failed to upload photo: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error uploading photo: {e}")
            return None
    
    def _add_to_album(self, album_id: str, asset_id: str) -> bool:
        """
        Add asset to album
        """
        try:
            data = {
                "ids": [asset_id]
            }
            
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
                logger.error(f"Failed to add asset to album: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding asset to album: {e}")
            return False
    
    def _create_description(self, analysis: PhotoAnalysis, original_filename: str) -> str:
        """
        Create detailed description from AI analysis
        """
        try:
            description_parts = [
                f"📸 Original: {original_filename}",
                f"🤖 AI Analysis: {analysis.description}",
                f"🎯 Subject: {analysis.primary_subject.replace('_', ' ').title()}",
                f"📐 Aspect: {analysis.recommended_crop.aspect_ratio}",
            ]
            
            # Add rotation info if applied
            if abs(analysis.recommended_crop.rotation_degrees) > 0.1:
                description_parts.append(f"🔄 Rotation: {analysis.recommended_crop.rotation_degrees:.1f}°")
            
            # Add swimming context if available
            if analysis.swimming_context.event_type:
                description_parts.append(f"🏊 Event: {analysis.swimming_context.event_type.replace('_', ' ').title()}")
            
            if analysis.swimming_context.pool_type:
                description_parts.append(f"🏊 Pool: {analysis.swimming_context.pool_type.replace('_', ' ').title()}")
            
            # Add color analysis
            if analysis.color_analysis.exposure_assessment != 'properly_exposed':
                description_parts.append(f"💡 Exposure: {analysis.color_analysis.exposure_assessment.replace('_', ' ').title()}")
            
            return "\n".join(description_parts)
            
        except Exception as e:
            logger.error(f"Error creating description: {e}")
            return f"Auto-processed from {original_filename}"