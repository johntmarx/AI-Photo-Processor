"""
Tests for main FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

class TestMainApp:
    """Test main application endpoints"""
    
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns correct information"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Photo Processor API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert "endpoints" in data
        assert all(endpoint in data["endpoints"] for endpoint in [
            "photos", "processing", "recipes", "stats", "websocket"
        ])
    
    def test_health_endpoint(self, test_client: TestClient):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "photo-processor-api"
    
    async def test_root_endpoint_async(self, async_client: AsyncClient):
        """Test root endpoint with async client"""
        response = await async_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Photo Processor API"
    
    def test_cors_headers(self, test_client: TestClient):
        """Test CORS headers are properly set"""
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
    
    def test_invalid_endpoint(self, test_client: TestClient):
        """Test 404 for invalid endpoint"""
        response = test_client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_image_endpoint_invalid_type(self, test_client: TestClient):
        """Test image endpoint with invalid type"""
        response = test_client.get("/images/invalid-type/test.jpg")
        assert response.status_code == 200
        assert response.json() == {"error": "Invalid image type"}
    
    def test_image_endpoint_not_found(self, test_client: TestClient):
        """Test image endpoint when file doesn't exist"""
        response = test_client.get("/images/originals/nonexistent.jpg")
        assert response.status_code == 404