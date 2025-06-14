"""
Unit tests for Enhanced Immich Client
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import os

from immich_client_v2 import EnhancedImmichClient, UploadResult, DualUploadResult
from recipe_storage import ProcessingRecipe


class TestEnhancedImmichClient:
    """Test EnhancedImmichClient class"""
    
    @pytest.fixture
    def client(self):
        """Create test client instance"""
        return EnhancedImmichClient(
            base_url='http://test-immich:2283',
            api_key='test-api-key'
        )
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary test file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(b'test image data')
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink()
    
    @pytest.fixture
    def mock_recipe(self):
        """Create a mock recipe"""
        recipe = ProcessingRecipe(
            id='test-recipe-123',
            original_hash='abc123def456',
            original_filename='test.jpg'
        )
        recipe.add_operation('crop', {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1})
        return recipe
    
    def test_client_initialization(self):
        """Test client initialization"""
        client = EnhancedImmichClient(
            base_url='http://immich:2283/',
            api_key='my-key'
        )
        
        assert client.base_url == 'http://immich:2283'  # Trailing slash removed
        assert client.api_key == 'my-key'
        assert client.headers['X-API-KEY'] == 'my-key'
        assert client.session is not None
    
    def test_calculate_file_hash(self, client, temp_file):
        """Test file hash calculation"""
        # Write known content
        with open(temp_file, 'wb') as f:
            f.write(b'Hello, World!')
        
        hash_value = client._calculate_file_hash(temp_file)
        
        # This is the SHA256 hash of "Hello, World!"
        expected_hash = 'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
        assert hash_value == expected_hash
    
    def test_prepare_metadata_for_original(self, client, temp_file, mock_recipe):
        """Test metadata preparation for original file"""
        metadata = client._prepare_metadata(
            temp_file,
            is_original=True,
            recipe=mock_recipe
        )
        
        assert metadata['originalFileName'] == temp_file.name
        assert metadata['isOriginal'] is True
        assert metadata['hasProcessedVersion'] is True
        assert metadata['recipeId'] == 'test-recipe-123'
        assert metadata['processorVersion'] == '2.0'
        assert 'fileHash' in metadata
        assert 'fileSize' in metadata
    
    def test_prepare_metadata_for_processed(self, client, temp_file, mock_recipe):
        """Test metadata preparation for processed file"""
        metadata = client._prepare_metadata(
            temp_file,
            is_original=False,
            recipe=mock_recipe,
            original_asset_id='original-123'
        )
        
        assert metadata['isOriginal'] is False
        assert metadata['isProcessed'] is True
        assert metadata['recipeId'] == 'test-recipe-123'
        assert metadata['originalAssetId'] == 'original-123'
        assert 'processingDescription' in metadata
    
    @patch('requests.Session.post')
    def test_upload_asset_success(self, mock_post, client, temp_file):
        """Test successful asset upload"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'asset-123'}
        mock_post.return_value = mock_response
        
        result = client.upload_asset(temp_file)
        
        assert result.success is True
        assert result.asset_id == 'asset-123'
        assert result.filename == temp_file.name
        assert result.error is None
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == 'http://test-immich:2283/api/assets'
        assert 'X-API-KEY' in call_args[1]['headers']
    
    @patch('requests.Session.post')
    def test_upload_asset_failure(self, mock_post, client, temp_file):
        """Test failed asset upload"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad request'
        mock_post.return_value = mock_response
        
        result = client.upload_asset(temp_file)
        
        assert result.success is False
        assert result.asset_id == ""
        assert "Upload failed with status 400" in result.error
    
    @patch('requests.Session.post')
    def test_upload_asset_with_album(self, mock_post, client, temp_file):
        """Test asset upload with album"""
        # Mock successful upload
        mock_upload_response = Mock()
        mock_upload_response.status_code = 201
        mock_upload_response.json.return_value = {'id': 'asset-456'}
        
        # Mock album operations
        with patch.object(client, '_add_to_album', return_value=True) as mock_add:
            mock_post.return_value = mock_upload_response
            
            result = client.upload_asset(temp_file, album_name='Test Album')
            
            assert result.success is True
            mock_add.assert_called_once_with('asset-456', 'Test Album')
    
    @patch.object(EnhancedImmichClient, 'upload_asset')
    def test_upload_photo_pair_success(self, mock_upload, client, temp_file, mock_recipe):
        """Test successful photo pair upload"""
        # Create a second temp file for processed version
        with tempfile.NamedTemporaryFile(delete=False, suffix='_processed.jpg') as f:
            f.write(b'processed image data')
            processed_file = Path(f.name)
        
        try:
            # Mock successful uploads
            original_result = UploadResult(
                asset_id='original-123',
                filename='test.jpg',
                success=True
            )
            processed_result = UploadResult(
                asset_id='processed-456',
                filename='test_processed.jpg',
                success=True
            )
            
            mock_upload.side_effect = [original_result, processed_result]
            
            # Test dual upload
            with patch.object(client, '_link_assets', return_value=True):
                result = client.upload_photo_pair(
                    original_path=temp_file,
                    processed_path=processed_file,
                    recipe=mock_recipe
                )
            
            assert result.original.success is True
            assert result.processed.success is True
            assert result.linked is True
            assert result.recipe_stored is True
            
            # Verify both uploads were called
            assert mock_upload.call_count == 2
            
        finally:
            processed_file.unlink()
    
    @patch.object(EnhancedImmichClient, 'upload_asset')
    def test_upload_photo_pair_original_fails(self, mock_upload, client, temp_file, mock_recipe):
        """Test photo pair upload when original fails"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='_processed.jpg') as f:
            processed_file = Path(f.name)
        
        try:
            # Mock failed original upload
            original_result = UploadResult(
                asset_id='',
                filename='test.jpg',
                success=False,
                error='Upload failed'
            )
            
            mock_upload.return_value = original_result
            
            result = client.upload_photo_pair(
                original_path=temp_file,
                processed_path=processed_file,
                recipe=mock_recipe
            )
            
            assert result.original.success is False
            assert result.processed.success is False
            assert result.processed.error == 'Original upload failed'
            assert result.linked is False
            assert result.recipe_stored is False
            
            # Only one upload attempt should be made
            mock_upload.assert_called_once()
            
        finally:
            processed_file.unlink()
    
    @patch('requests.Session.get')
    @patch('requests.Session.post')
    @patch('requests.Session.put')
    def test_add_to_album(self, mock_put, mock_post, mock_get, client):
        """Test adding asset to album"""
        # Mock finding existing album
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {'id': 'album-123', 'albumName': 'Test Album'},
            {'id': 'album-456', 'albumName': 'Other Album'}
        ]
        
        # Mock adding to album
        mock_put.return_value.status_code = 200
        
        result = client._add_to_album('asset-789', 'Test Album')
        
        assert result is True
        mock_put.assert_called_once_with(
            'http://test-immich:2283/api/albums/album-123/assets',
            headers={'X-API-KEY': 'test-api-key'},
            json={'ids': ['asset-789']}
        )
    
    @patch('requests.Session.get')
    @patch('requests.Session.post')
    def test_find_or_create_album_new(self, mock_post, mock_get, client):
        """Test creating new album when it doesn't exist"""
        # Mock no existing albums
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = []
        
        # Mock creating album
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {'id': 'new-album-123'}
        
        album_id = client._find_or_create_album('New Album')
        
        assert album_id == 'new-album-123'
        mock_post.assert_called_once_with(
            'http://test-immich:2283/api/albums',
            headers={'X-API-KEY': 'test-api-key'},
            json={'albumName': 'New Album'}
        )
    
    def test_link_assets(self, client):
        """Test asset linking (placeholder functionality)"""
        # Since this is a placeholder for future Immich functionality,
        # just verify it returns True and logs appropriately
        with patch('logging.Logger.info') as mock_log:
            result = client._link_assets('asset-1', 'asset-2', 'processed_from')
            
            assert result is True
            mock_log.assert_called_once()
            log_message = mock_log.call_args[0][0]
            assert 'asset-1' in log_message
            assert 'asset-2' in log_message
            assert 'processed_from' in log_message
    
    @patch('requests.Session.get')
    def test_check_asset_exists(self, mock_get, client):
        """Test checking if asset exists"""
        # Mock finding existing asset
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {'id': 'existing-asset-123', 'originalFileName': 'test_abc123.jpg'}
        ]
        
        asset_id = client.check_asset_exists('abc123')
        
        assert asset_id == 'existing-asset-123'
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_check_asset_not_exists(self, mock_get, client):
        """Test checking asset that doesn't exist"""
        # Mock empty response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = []
        
        asset_id = client.check_asset_exists('nonexistent')
        
        assert asset_id is None