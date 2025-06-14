"""
Tests for photo routes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import io

from models.photo import PhotoList, Photo, PhotoDetail, PhotoComparison

class TestPhotoRoutes:
    """Test photo management routes"""
    
    @patch('routes.photos.photo_service')
    def test_list_photos(self, mock_service, test_client: TestClient, sample_photo_data):
        """Test listing photos with pagination"""
        # Setup mock
        mock_service.list_photos.return_value = PhotoList(
            photos=[Photo(**sample_photo_data)],
            total=1,
            page=1,
            page_size=20,
            has_next=False,
            has_prev=False
        )
        
        # Test default parameters
        response = test_client.get("/api/photos/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["photos"]) == 1
        assert data["page"] == 1
        
        # Test with parameters
        response = test_client.get("/api/photos/?page=2&page_size=10&status=completed")
        assert response.status_code == 200
        
        # Verify service was called correctly
        mock_service.list_photos.assert_called_with(
            page=2,
            page_size=10,
            status="completed",
            sort_by="created_at",
            order="desc"
        )
    
    @patch('routes.photos.photo_service')
    def test_get_photo_success(self, mock_service, test_client: TestClient, sample_photo_data):
        """Test getting photo details"""
        # Setup mock
        mock_service.get_photo.return_value = PhotoDetail(
            **sample_photo_data,
            recipe_id="recipe123",
            recipe_name="Test Recipe",
            ai_analysis={"quality_score": 8.5},
            processing_time=1.2
        )
        
        response = test_client.get("/api/photos/abc123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "abc123"
        assert data["recipe_id"] == "recipe123"
        assert data["ai_analysis"]["quality_score"] == 8.5
    
    @patch('routes.photos.photo_service')
    def test_get_photo_not_found(self, mock_service, test_client: TestClient):
        """Test getting non-existent photo"""
        mock_service.get_photo.return_value = None
        
        response = test_client.get("/api/photos/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Photo not found"
    
    @patch('routes.photos.photo_service')
    def test_get_photo_comparison(self, mock_service, test_client: TestClient):
        """Test getting photo comparison data"""
        mock_service.get_comparison.return_value = PhotoComparison(
            photo_id="abc123",
            original={"path": "/original.jpg", "size": 1000000},
            processed={"path": "/processed.jpg", "size": 800000},
            ai_overlays={"boxes": []},
            statistics={"compression_ratio": 0.8}
        )
        
        response = test_client.get("/api/photos/abc123/comparison")
        assert response.status_code == 200
        data = response.json()
        assert data["photo_id"] == "abc123"
        assert "original" in data
        assert "processed" in data
    
    @patch('routes.photos.photo_service')
    def test_upload_photo_success(self, mock_service, test_client: TestClient):
        """Test uploading a photo"""
        from unittest.mock import AsyncMock
        
        # Create async mock that returns the result
        mock_service.save_upload = AsyncMock(return_value={
            "photo_id": "new123",
            "filename": "test.jpg",
            "status": "queued",
            "message": "Photo uploaded successfully"
        })
        
        # Create test file
        file_content = b"fake image data"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        
        response = test_client.post("/api/photos/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["photo_id"] == "new123"
        assert data["status"] == "queued"
    
    def test_upload_photo_invalid_type(self, test_client: TestClient):
        """Test uploading invalid file type"""
        file_content = b"fake text data"
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        
        response = test_client.post("/api/photos/upload", files=files)
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]
    
    @patch('routes.photos.photo_service')
    def test_upload_photo_with_recipe(self, mock_service, test_client: TestClient):
        """Test uploading photo with recipe"""
        from unittest.mock import AsyncMock
        
        # Create async mock that returns the result
        mock_save_upload = AsyncMock(return_value={
            "photo_id": "new123",
            "filename": "test.jpg",
            "status": "queued"
        })
        
        mock_service.save_upload = mock_save_upload
        
        file_content = b"fake image data"
        files = {"file": ("test.jpg", io.BytesIO(file_content), "image/jpeg")}
        
        response = test_client.post(
            "/api/photos/upload?auto_process=false&recipe_id=recipe123",
            files=files
        )
        assert response.status_code == 200
        
        # Verify service was called with recipe
        mock_service.save_upload.assert_called_once()
        call_args = mock_service.save_upload.call_args
        assert call_args.kwargs["auto_process"] is False
        assert call_args.kwargs["recipe_id"] == "recipe123"
    
    @patch('routes.photos.photo_service')
    def test_delete_photo_success(self, mock_service, test_client: TestClient):
        """Test deleting a photo"""
        mock_service.delete_photo.return_value = True
        
        response = test_client.delete("/api/photos/abc123")
        assert response.status_code == 200
        assert response.json()["message"] == "Photo deleted successfully"
        
        # Verify service called correctly
        mock_service.delete_photo.assert_called_with(
            photo_id="abc123",
            delete_original=False
        )
    
    @patch('routes.photos.photo_service')
    def test_delete_photo_with_original(self, mock_service, test_client: TestClient):
        """Test deleting photo including original"""
        mock_service.delete_photo.return_value = True
        
        response = test_client.delete("/api/photos/abc123?delete_original=true")
        assert response.status_code == 200
        
        # Verify original deletion flag
        mock_service.delete_photo.assert_called_with(
            photo_id="abc123",
            delete_original=True
        )
    
    @patch('routes.photos.photo_service')
    def test_delete_photo_not_found(self, mock_service, test_client: TestClient):
        """Test deleting non-existent photo"""
        mock_service.delete_photo.return_value = False
        
        response = test_client.delete("/api/photos/nonexistent")
        assert response.status_code == 404
    
    @patch('routes.photos.photo_service')
    def test_reprocess_photo(self, mock_service, test_client: TestClient):
        """Test reprocessing a photo"""
        mock_service.reprocess_photo.return_value = {
            "photo_id": "abc123",
            "status": "queued",
            "message": "Photo queued for reprocessing"
        }
        
        response = test_client.post(
            "/api/photos/abc123/reprocess?recipe_id=recipe456&priority=high"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        
        # Verify service call
        mock_service.reprocess_photo.assert_called_with(
            photo_id="abc123",
            recipe_id="recipe456",
            priority="high"
        )
    
    @patch('routes.photos.photo_service')
    def test_get_ai_analysis(self, mock_service, test_client: TestClient):
        """Test getting AI analysis results"""
        mock_service.get_ai_analysis.return_value = {
            "quality_score": 8.5,
            "detected_objects": ["person", "pool"],
            "suggested_crop": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}
        }
        
        response = test_client.get("/api/photos/abc123/ai-analysis")
        assert response.status_code == 200
        data = response.json()
        assert data["quality_score"] == 8.5
        assert "person" in data["detected_objects"]