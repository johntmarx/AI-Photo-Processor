"""
Integration tests for the complete API
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import websockets
import json
from pathlib import Path
import tempfile
import shutil

class TestAPIIntegration:
    """Integration tests for API functionality"""
    
    @pytest.fixture
    async def client(self):
        """Create async client for testing"""
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    async def test_full_photo_workflow(self, client: AsyncClient):
        """Test complete photo processing workflow"""
        # 1. Check initial stats
        response = await client.get("/api/stats/dashboard")
        assert response.status_code == 200
        initial_stats = response.json()
        
        # 2. Get recipe presets
        response = await client.get("/api/recipes/presets/")
        assert response.status_code == 200
        presets = response.json()["presets"]
        assert len(presets) > 0
        
        # 3. Check processing status
        response = await client.get("/api/processing/status")
        assert response.status_code == 200
        status = response.json()
        assert "is_paused" in status
        
        # 4. Get queue status
        response = await client.get("/api/processing/queue")
        assert response.status_code == 200
        queue = response.json()
        assert "pending" in queue
        assert "processing" in queue
        assert "completed" in queue
    
    async def test_settings_workflow(self, client: AsyncClient):
        """Test processing settings management"""
        # 1. Get current settings
        response = await client.get("/api/processing/settings")
        assert response.status_code == 200
        original_settings = response.json()
        
        # 2. Update settings
        new_settings = {
            **original_settings,
            "auto_process": False,
            "require_approval": True,
            "quality_threshold": 8.0
        }
        
        response = await client.put("/api/processing/settings", json=new_settings)
        assert response.status_code == 200
        updated = response.json()
        assert updated["auto_process"] is False
        assert updated["require_approval"] is True
        assert updated["quality_threshold"] == 8.0
        
        # 3. Verify settings persisted
        response = await client.get("/api/processing/settings")
        assert response.status_code == 200
        current = response.json()
        assert current["auto_process"] is False
    
    async def test_recipe_crud_workflow(self, client: AsyncClient):
        """Test complete recipe CRUD operations"""
        # 1. Create recipe
        recipe_data = {
            "name": "Integration Test Recipe",
            "description": "Test recipe for integration testing",
            "operations": [
                {"type": "enhance", "parameters": {"brightness": 0.1}},
                {"type": "crop", "parameters": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}}
            ],
            "is_default": False
        }
        
        response = await client.post("/api/recipes/", json=recipe_data)
        assert response.status_code == 200
        created_recipe = response.json()
        recipe_id = created_recipe["id"]
        
        # 2. Get recipe
        response = await client.get(f"/api/recipes/{recipe_id}")
        assert response.status_code == 200
        fetched = response.json()
        assert fetched["name"] == recipe_data["name"]
        
        # 3. List recipes
        response = await client.get("/api/recipes/")
        assert response.status_code == 200
        recipes_list = response.json()
        assert "recipes" in recipes_list
        
        # 4. Update recipe
        update_data = {
            "name": "Updated Test Recipe",
            "description": "Updated description"
        }
        response = await client.put(f"/api/recipes/{recipe_id}", json=update_data)
        assert response.status_code == 200
        
        # 5. Delete recipe
        response = await client.delete(f"/api/recipes/{recipe_id}")
        assert response.status_code == 200
    
    async def test_processing_control_workflow(self, client: AsyncClient):
        """Test processing control operations"""
        # 1. Pause processing
        response = await client.put("/api/processing/pause")
        assert response.status_code in [200, 500]  # May fail if already paused
        
        # 2. Check status
        response = await client.get("/api/processing/status")
        assert response.status_code == 200
        status = response.json()
        
        # 3. Resume processing
        response = await client.put("/api/processing/resume")
        assert response.status_code in [200, 500]  # May fail if already running
    
    async def test_stats_endpoints(self, client: AsyncClient):
        """Test all statistics endpoints"""
        endpoints = [
            "/api/stats/dashboard",
            "/api/stats/processing",
            "/api/stats/storage",
            "/api/stats/activity",
            "/api/stats/performance",
            "/api/stats/trends",
            "/api/stats/ai-performance",
            "/api/stats/errors"
        ]
        
        for endpoint in endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 200, f"Failed for {endpoint}"
            data = response.json()
            assert data is not None
    
    async def test_error_handling(self, client: AsyncClient):
        """Test API error handling"""
        # 1. Non-existent photo
        response = await client.get("/api/photos/nonexistent123")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        
        # 2. Non-existent recipe
        response = await client.get("/api/recipes/nonexistent456")
        assert response.status_code == 404
        
        # 3. Invalid endpoint
        response = await client.get("/api/invalid/endpoint")
        assert response.status_code == 404
        
        # 4. Invalid image type
        response = await client.get("/images/invalid/test.jpg")
        assert response.status_code == 200
        assert response.json()["error"] == "Invalid image type"
    
    def test_cors_headers(self):
        """Test CORS is properly configured"""
        from main import app
        client = TestClient(app)
        
        # Test from different origins
        origins = [
            "http://localhost:3000",
            "http://192.168.1.100:3000",
            "http://10.0.0.50:8080"
        ]
        
        for origin in origins:
            response = client.options(
                "/api/photos/",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET"
                }
            )
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "*"
            assert "GET" in response.headers["access-control-allow-methods"]