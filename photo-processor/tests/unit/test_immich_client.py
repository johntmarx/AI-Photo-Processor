"""
Comprehensive unit tests for Immich Client module
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
import requests
from immich_client import ImmichClient
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis


class TestImmichClient:
    """Test suite for ImmichClient class"""

    @pytest.fixture
    def immich_client(self):
        """Create ImmichClient instance for testing"""
        return ImmichClient(
            api_url="http://test-immich:2283",
            api_key="test-api-key"
        )

    @pytest.fixture
    def sample_photo_analysis(self):
        """Sample photo analysis for testing"""
        return PhotoAnalysis(
            description="Test swimming photo showing freestyle stroke technique",
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
                brightness_adjustment_needed=10,
                contrast_adjustment_needed=5
            ),
            swimming_context={
                "event_type": "freestyle",
                "pool_type": "indoor",
                "time_of_event": "mid_race",
                "lane_number": 4
            },
            processing_recommendation="crop_and_enhance"
        )

    def test_initialization(self, immich_client):
        """Test ImmichClient initialization"""
        assert immich_client.api_url == "http://test-immich:2283"
        assert immich_client.api_key == "test-api-key"
        assert immich_client.headers['x-api-key'] == "test-api-key"
        assert immich_client.headers['Content-Type'] == "application/json"

    def test_initialization_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from API URL"""
        client = ImmichClient(
            api_url="http://test-immich:2283/",
            api_key="test-key"
        )
        assert client.api_url == "http://test-immich:2283"

    @patch('immich_client.requests.get')
    def test_test_connection_success(self, mock_get, immich_client):
        """Test successful connection to Immich API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = immich_client.test_connection()

        assert result is True
        mock_get.assert_called_once_with(
            "http://test-immich:2283/api/server/ping",
            headers={'x-api-key': 'test-api-key'},
            timeout=10
        )

    @patch('immich_client.requests.get')
    def test_test_connection_failure(self, mock_get, immich_client):
        """Test failed connection to Immich API"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = immich_client.test_connection()

        assert result is False

    @patch('immich_client.requests.get')
    def test_test_connection_exception(self, mock_get, immich_client):
        """Test connection with exception"""
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        result = immich_client.test_connection()

        assert result is False

    @patch('immich_client.requests.post')
    def test_create_album_success(self, mock_post, immich_client):
        """Test successful album creation"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'test-album-id'}
        mock_post.return_value = mock_response

        result = immich_client.create_album("Test Album", "Test Description")

        assert result == "test-album-id"
        mock_post.assert_called_once_with(
            "http://test-immich:2283/api/albums",
            headers=immich_client.headers,
            json={
                "albumName": "Test Album",
                "description": "Test Description"
            },
            timeout=30
        )

    @patch('immich_client.requests.post')
    def test_create_album_failure(self, mock_post, immich_client):
        """Test failed album creation"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        result = immich_client.create_album("Test Album")

        assert result is None

    @patch('immich_client.requests.post')
    def test_create_album_exception(self, mock_post, immich_client):
        """Test album creation with exception"""
        mock_post.side_effect = requests.RequestException("Network error")

        result = immich_client.create_album("Test Album")

        assert result is None

    @patch('immich_client.requests.get')
    def test_get_or_create_album_existing(self, mock_get, immich_client):
        """Test getting existing album"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'id': 'album-1', 'albumName': 'Different Album'},
            {'id': 'album-2', 'albumName': 'Test Album'},
            {'id': 'album-3', 'albumName': 'Another Album'}
        ]
        mock_get.return_value = mock_response

        result = immich_client.get_or_create_album("Test Album")

        assert result == "album-2"
        mock_get.assert_called_once_with(
            "http://test-immich:2283/api/albums",
            headers={'x-api-key': 'test-api-key'},
            timeout=30
        )

    @patch('immich_client.requests.get')
    @patch.object(ImmichClient, 'create_album')
    def test_get_or_create_album_create_new(self, mock_create, mock_get, immich_client):
        """Test creating new album when it doesn't exist"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'id': 'album-1', 'albumName': 'Different Album'}
        ]
        mock_get.return_value = mock_response
        mock_create.return_value = "new-album-id"

        result = immich_client.get_or_create_album("Test Album")

        assert result == "new-album-id"
        mock_create.assert_called_once_with("Test Album", "Auto-processed photos from AI photo pipeline")

    @patch('immich_client.requests.get')
    def test_get_or_create_album_get_failure(self, mock_get, immich_client):
        """Test get_or_create_album when get request fails"""
        mock_get.side_effect = requests.RequestException("Network error")

        result = immich_client.get_or_create_album("Test Album")

        assert result is None

    def test_create_description(self, immich_client, sample_photo_analysis):
        """Test description creation from photo analysis"""
        description = immich_client._create_description(sample_photo_analysis, "DSC09123.ARW")

        assert "DSC09123.ARW" in description
        assert "swimmer" in description.lower()
        assert "crisp" in description.lower()
        assert "freestyle" in description.lower()
        assert "indoor" in description.lower()
        assert "crop and enhance" in description.lower()

    def test_create_description_minimal_analysis(self, immich_client):
        """Test description creation with minimal analysis data"""
        minimal_analysis = PhotoAnalysis(
            description="Simple photo",
            quality="crisp",
            primary_subject="person",
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
            swimming_context={},
            processing_recommendation="enhance_only"
        )

        description = immich_client._create_description(minimal_analysis, "test.jpg")

        assert "test.jpg" in description
        assert "person" in description.lower()

    def test_create_description_exception(self, immich_client):
        """Test description creation with exception"""
        # Pass None to trigger exception
        description = immich_client._create_description(None, "test.jpg")

        assert description == "Auto-processed from test.jpg"

    @patch('immich_client.requests.put')
    def test_add_to_album_success(self, mock_put, immich_client):
        """Test successful addition of asset to album"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        result = immich_client._add_to_album("album-id", "asset-id")

        assert result is True
        mock_put.assert_called_once_with(
            "http://test-immich:2283/api/albums/album-id/assets",
            headers=immich_client.headers,
            json={"ids": ["asset-id"]},
            timeout=30
        )

    @patch('immich_client.requests.put')
    def test_add_to_album_failure(self, mock_put, immich_client):
        """Test failed addition of asset to album"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_put.return_value = mock_response

        result = immich_client._add_to_album("album-id", "asset-id")

        assert result is False

    @patch('immich_client.requests.put')
    def test_add_to_album_exception(self, mock_put, immich_client):
        """Test asset addition with exception"""
        mock_put.side_effect = requests.RequestException("Network error")

        result = immich_client._add_to_album("album-id", "asset-id")

        assert result is False

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_success(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test successful photo upload"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'asset-id-123'}
        mock_post.return_value = mock_response

        with patch.object(immich_client, '_add_to_album') as mock_add_album:
            mock_add_album.return_value = True

            result = immich_client.upload_photo(
                "/test/image.jpg",
                "DSC09123.ARW",
                sample_photo_analysis,
                album_id="album-id"
            )

            assert result == "asset-id-123"
            mock_post.assert_called_once()
            mock_add_album.assert_called_once_with("album-id", "asset-id-123")

            # Check upload request details
            call_args = mock_post.call_args
            assert "asset/upload" in call_args[0][0]
            assert call_args[1]['headers']['x-api-key'] == 'test-api-key'
            assert 'files' in call_args[1]
            assert 'data' in call_args[1]

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_without_album(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload without album"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'asset-id-123'}
        mock_post.return_value = mock_response

        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )

        assert result == "asset-id-123"

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_failure(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test failed photo upload"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )

        assert result is None

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_upload_photo_file_not_found(self, mock_file, immich_client, sample_photo_analysis):
        """Test photo upload with file not found"""
        result = immich_client.upload_photo(
            "/nonexistent/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )

        assert result is None

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_request_exception(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload with request exception"""
        mock_post.side_effect = requests.RequestException("Network error")

        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )

        assert result is None

    def test_upload_metadata_includes_device_info(self, immich_client, sample_photo_analysis):
        """Test that upload metadata includes correct device information"""
        with patch('immich_client.requests.post') as mock_post, \
             patch('builtins.open', new_callable=mock_open, read_data=b"fake image data"):
            
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {'id': 'asset-id-123'}
            mock_post.return_value = mock_response

            immich_client.upload_photo(
                "/test/image.jpg",
                "DSC09123.ARW",
                sample_photo_analysis
            )

            # Check metadata in upload request
            call_args = mock_post.call_args
            data = call_args[1]['data']
            assert data['deviceAssetId'] == "ai-processed-DSC09123.ARW"
            assert data['deviceId'] == 'ai-photo-processor'
            assert data['isFavorite'] == 'false'

    def test_upload_includes_description(self, immich_client, sample_photo_analysis):
        """Test that upload includes AI-generated description"""
        with patch('immich_client.requests.post') as mock_post, \
             patch('builtins.open', new_callable=mock_open, read_data=b"fake image data"):
            
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {'id': 'asset-id-123'}
            mock_post.return_value = mock_response

            immich_client.upload_photo(
                "/test/image.jpg",
                "DSC09123.ARW",
                sample_photo_analysis
            )

            # Check description is included
            call_args = mock_post.call_args
            data = call_args[1]['data']
            assert 'description' in data
            description = data['description']
            assert "DSC09123.ARW" in description
            assert "swimmer" in description.lower()

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_album_add_failure(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload where asset upload succeeds but album addition fails"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {'id': 'asset-id-123'}
        mock_post.return_value = mock_response

        with patch.object(immich_client, '_add_to_album') as mock_add_album:
            mock_add_album.return_value = False

            result = immich_client.upload_photo(
                "/test/image.jpg",
                "DSC09123.ARW",
                sample_photo_analysis,
                album_id="album-id"
            )

            # Should still return asset ID even if album addition fails
            assert result == "asset-id-123"
            mock_add_album.assert_called_once()

    def test_headers_format(self, immich_client):
        """Test that headers are properly formatted"""
        headers = immich_client.headers
        assert isinstance(headers, dict)
        assert headers.get('x-api-key') == 'test-api-key'
        assert headers.get('Content-Type') == 'application/json'

    def test_api_url_construction(self, immich_client):
        """Test API URL construction for various endpoints"""
        assert immich_client.api_url + "/api/server/ping" == "http://test-immich:2283/api/server/ping"
        assert immich_client.api_url + "/api/albums" == "http://test-immich:2283/api/albums"
        assert immich_client.api_url + "/api/asset/upload" == "http://test-immich:2283/api/asset/upload"

    def test_timeout_configuration(self, immich_client):
        """Test that timeouts are properly configured in requests"""
        with patch('immich_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            immich_client.test_connection()

            # Check timeout is set
            call_args = mock_get.call_args
            assert call_args[1]['timeout'] == 10

    @patch('immich_client.requests.get')
    def test_album_list_parsing(self, mock_get, immich_client):
        """Test parsing of album list response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'id': 'album-1', 'albumName': 'Family Photos', 'description': 'Family album'},
            {'id': 'album-2', 'albumName': 'Vacation', 'description': 'Vacation photos'},
        ]
        mock_get.return_value = mock_response

        result = immich_client.get_or_create_album("Family Photos")

        assert result == "album-1"

    @patch('immich_client.requests.get')
    def test_test_connection_timeout(self, mock_get, immich_client):
        """Test connection test with timeout"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_get.side_effect = MockImmichResponses.timeout_ping()
        
        result = immich_client.test_connection()
        
        assert result is False

    @patch('immich_client.requests.get')
    def test_test_connection_unauthorized(self, mock_get, immich_client):
        """Test connection test with unauthorized response"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_get.return_value = MockImmichResponses.unauthorized_ping()
        
        result = immich_client.test_connection()
        
        assert result is False

    @patch('immich_client.requests.get')
    def test_test_connection_server_error(self, mock_get, immich_client):
        """Test connection test with server error"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_get.return_value = MockImmichResponses.server_error_ping()
        
        result = immich_client.test_connection()
        
        assert result is False

    @patch('immich_client.requests.get')
    def test_get_or_create_album_empty_list(self, mock_get, immich_client):
        """Test album creation when no albums exist"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_get.return_value = MockImmichResponses.empty_albums_list()
        
        with patch.object(immich_client, 'create_album', return_value="new-album-id"):
            result = immich_client.get_or_create_album("New Album")
            
        assert result == "new-album-id"

    @patch('immich_client.requests.get')
    def test_get_or_create_album_malformed_response(self, mock_get, immich_client):
        """Test album handling with malformed response"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_get.return_value = MockImmichResponses.malformed_albums_list()
        
        result = immich_client.get_or_create_album("Test Album")
        
        assert result is None

    @patch('immich_client.requests.post')
    def test_create_album_unauthorized(self, mock_post, immich_client):
        """Test album creation with unauthorized response"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_post.return_value = MockImmichResponses.unauthorized_album_creation()
        
        result = immich_client.create_album("Test Album")
        
        assert result is None

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_duplicate(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload with duplicate detection"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_post.return_value = MockImmichResponses.duplicate_upload()
        
        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )
        
        # Should return the existing asset ID for duplicates
        assert result == "existing-asset-id"

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake image data")
    def test_upload_photo_insufficient_storage(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload with insufficient storage"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_post.return_value = MockImmichResponses.insufficient_storage_upload()
        
        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )
        
        assert result is None

    @patch('immich_client.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b"large file data" * 10000)
    def test_upload_photo_large_file(self, mock_file, mock_post, immich_client, sample_photo_analysis):
        """Test photo upload with large file rejection"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_post.return_value = MockImmichResponses.large_file_upload()
        
        result = immich_client.upload_photo(
            "/test/image.jpg",
            "DSC09123.ARW",
            sample_photo_analysis
        )
        
        assert result is None

    @patch('immich_client.requests.put')
    def test_add_to_album_partial_success(self, mock_put, immich_client):
        """Test album asset addition with partial success"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        mock_put.return_value = MockImmichResponses.partial_album_add()
        
        result = immich_client._add_to_album("album-id", "asset-id")
        
        # Partial success should still return False
        assert result is False

    @pytest.mark.slow
    def test_upload_photo_performance_benchmark(self, immich_client, sample_photo_analysis, performance_timer):
        """Test photo upload performance benchmark"""
        with patch('immich_client.requests.post') as mock_post, \
             patch('builtins.open', new_callable=mock_open, read_data=b"fake image data"):
            
            from tests.fixtures.mock_data import MockImmichResponses
            mock_post.return_value = MockImmichResponses.successful_upload()
            
            performance_timer.start("upload")
            
            for _ in range(5):  # Upload 5 times
                immich_client.upload_photo(
                    "/test/image.jpg",
                    "DSC09123.ARW",
                    sample_photo_analysis
                )
            
            performance_timer.stop("upload")
            
            # Should complete 5 uploads in under 1 second (mocked)
            performance_timer.assert_faster_than(1.0, "upload")

    def test_description_creation_edge_cases(self, immich_client):
        """Test description creation with edge cases"""
        from tests.fixtures.mock_data import MockPhotoAnalysisData
        
        # Test with edge case analysis
        edge_analysis = MockPhotoAnalysisData.edge_case_minimal_analysis()
        description = immich_client._create_description(edge_analysis, "test.jpg")
        
        assert "test.jpg" in description
        # Should handle empty description gracefully
        assert len(description) > len("test.jpg")

    def test_api_url_trailing_slash_handling(self):
        """Test API URL handling with various formats"""
        # Test with trailing slash
        client1 = ImmichClient("http://test:2283/", "key")
        assert client1.api_url == "http://test:2283"
        
        # Test with multiple trailing slashes
        client2 = ImmichClient("http://test:2283///", "key")
        assert client2.api_url == "http://test:2283"
        
        # Test with no trailing slash
        client3 = ImmichClient("http://test:2283", "key")
        assert client3.api_url == "http://test:2283"

    def test_upload_metadata_validation(self, immich_client, sample_photo_analysis):
        """Test upload metadata contains required fields"""
        with patch('immich_client.requests.post') as mock_post, \
             patch('builtins.open', new_callable=mock_open, read_data=b"fake image data"):
            
            from tests.fixtures.mock_data import MockImmichResponses
            mock_post.return_value = MockImmichResponses.successful_upload()
            
            immich_client.upload_photo(
                "/test/image.jpg",
                "DSC09123.ARW",
                sample_photo_analysis
            )
            
            # Verify all required metadata fields are present
            call_args = mock_post.call_args
            data = call_args[1]['data']
            
            required_fields = ['deviceAssetId', 'deviceId', 'isFavorite', 'description']
            for field in required_fields:
                assert field in data, f"Required field '{field}' missing from upload metadata"

    def test_concurrent_requests_handling(self, immich_client, sample_photo_analysis):
        """Test handling of concurrent requests (thread safety simulation)"""
        from tests.fixtures.mock_data import MockImmichResponses
        
        with patch('immich_client.requests.post') as mock_post, \
             patch('builtins.open', new_callable=mock_open, read_data=b"fake image data"):
            
            mock_post.return_value = MockImmichResponses.successful_upload()
            
            # Simulate multiple concurrent uploads
            results = []
            for i in range(10):
                result = immich_client.upload_photo(
                    f"/test/image_{i}.jpg",
                    f"DSC0912{i}.ARW",
                    sample_photo_analysis
                )
                results.append(result)
            
            # All uploads should succeed
            assert all(result == "asset-id-456" for result in results)
            assert mock_post.call_count == 10