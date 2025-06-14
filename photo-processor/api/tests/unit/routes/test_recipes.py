"""
Tests for recipe routes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

class TestRecipeRoutes:
    """Test recipe management routes"""
    
    @patch('routes.recipes.recipe_service')
    def test_list_recipes(self, mock_service, test_client: TestClient, sample_recipe_data):
        """Test listing recipes"""
        mock_service.list_recipes.return_value = {
            "recipes": [sample_recipe_data],
            "total": 1,
            "page": 1,
            "pageSize": 20
        }
        
        response = test_client.get("/api/recipes/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["recipes"]) == 1
        assert data["recipes"][0]["name"] == "Test Recipe"
        
        # Test with pagination
        response = test_client.get("/api/recipes/?page=2&page_size=10")
        assert response.status_code == 200
        
        mock_service.list_recipes.assert_called_with(
            page=2,
            page_size=10,
            sort_by="created_at",
            order="desc"
        )
    
    @patch('routes.recipes.recipe_service')
    def test_get_recipe_success(self, mock_service, test_client: TestClient, sample_recipe_data):
        """Test getting recipe details"""
        mock_service.get_recipe.return_value = sample_recipe_data
        
        response = test_client.get("/api/recipes/recipe123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "recipe123"
        assert data["name"] == "Test Recipe"
        assert len(data["operations"]) == 2
    
    @patch('routes.recipes.recipe_service')
    def test_get_recipe_not_found(self, mock_service, test_client: TestClient):
        """Test getting non-existent recipe"""
        mock_service.get_recipe.return_value = None
        
        response = test_client.get("/api/recipes/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Recipe not found"
    
    @patch('routes.recipes.recipe_service')
    def test_create_recipe(self, mock_service, test_client: TestClient, sample_recipe_data):
        """Test creating a new recipe"""
        mock_service.create_recipe.return_value = sample_recipe_data
        
        recipe_data = {
            "name": "New Recipe",
            "description": "A new test recipe",
            "operations": [
                {"type": "enhance", "parameters": {"brightness": 0.2}}
            ],
            "is_default": False
        }
        
        response = test_client.post("/api/recipes/", json=recipe_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Recipe"  # Returns sample data
        
        # Verify service called correctly
        mock_service.create_recipe.assert_called_with(
            name="New Recipe",
            description="A new test recipe",
            operations=recipe_data["operations"],
            is_default=False
        )
    
    @patch('routes.recipes.recipe_service')
    def test_create_recipe_validation_error(self, mock_service, test_client: TestClient):
        """Test creating recipe with validation error"""
        mock_service.create_recipe.side_effect = ValueError("Invalid operation type")
        
        recipe_data = {
            "name": "Bad Recipe",
            "operations": [{"type": "invalid", "parameters": {}}]
        }
        
        response = test_client.post("/api/recipes/", json=recipe_data)
        assert response.status_code == 400
        assert "Invalid operation type" in response.json()["detail"]
    
    @patch('routes.recipes.recipe_service')
    def test_update_recipe(self, mock_service, test_client: TestClient, sample_recipe_data):
        """Test updating a recipe"""
        updated_data = {**sample_recipe_data, "name": "Updated Recipe"}
        mock_service.update_recipe.return_value = updated_data
        
        update_data = {
            "name": "Updated Recipe",
            "description": "Updated description"
        }
        
        response = test_client.put("/api/recipes/recipe123", json=update_data)
        assert response.status_code == 200
        
        # Verify service call
        mock_service.update_recipe.assert_called_with(
            recipe_id="recipe123",
            name="Updated Recipe",
            description="Updated description",
            operations=None,
            is_default=None
        )
    
    @patch('routes.recipes.recipe_service')
    def test_update_recipe_not_found(self, mock_service, test_client: TestClient):
        """Test updating non-existent recipe"""
        mock_service.update_recipe.return_value = None
        
        response = test_client.put("/api/recipes/nonexistent", json={"name": "Test"})
        assert response.status_code == 404
    
    @patch('routes.recipes.recipe_service')
    def test_delete_recipe_success(self, mock_service, test_client: TestClient):
        """Test deleting a recipe"""
        mock_service.delete_recipe.return_value = True
        
        response = test_client.delete("/api/recipes/recipe123")
        assert response.status_code == 200
        assert "successfully" in response.json()["message"]
    
    @patch('routes.recipes.recipe_service')
    def test_delete_recipe_not_found(self, mock_service, test_client: TestClient):
        """Test deleting non-existent recipe"""
        mock_service.delete_recipe.return_value = False
        
        response = test_client.delete("/api/recipes/nonexistent")
        assert response.status_code == 404
    
    @patch('routes.recipes.recipe_service')
    def test_duplicate_recipe(self, mock_service, test_client: TestClient, sample_recipe_data):
        """Test duplicating a recipe"""
        duplicated = {**sample_recipe_data, "id": "new123", "name": "Copy of Test Recipe"}
        mock_service.duplicate_recipe.return_value = duplicated
        
        response = test_client.post(
            "/api/recipes/recipe123/duplicate",
            json={"new_name": "Copy of Test Recipe"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "new123"
        assert data["name"] == "Copy of Test Recipe"
    
    @patch('routes.recipes.recipe_service')
    def test_apply_recipe_to_photos(self, mock_service, test_client: TestClient):
        """Test applying recipe to multiple photos"""
        mock_service.apply_to_photos.return_value = {
            "recipeId": "recipe123",
            "photosQueued": 3,
            "priority": "high",
            "message": "Recipe queued for 3 photos"
        }
        
        request_data = {
            "photo_ids": ["photo1", "photo2", "photo3"],
            "priority": "high"
        }
        
        response = test_client.post("/api/recipes/recipe123/apply", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["photosQueued"] == 3
        assert data["priority"] == "high"
    
    @patch('routes.recipes.recipe_service')
    def test_preview_recipe(self, mock_service, test_client: TestClient):
        """Test previewing recipe on photo"""
        mock_service.preview_recipe.return_value = {
            "original": "/path/to/original.jpg",
            "preview": "/path/to/preview.jpg",
            "operations": ["enhance", "crop"]
        }
        
        response = test_client.get("/api/recipes/recipe123/preview?photo_id=photo456")
        assert response.status_code == 200
        data = response.json()
        assert "original" in data
        assert "preview" in data
        assert len(data["operations"]) == 2
    
    @patch('routes.recipes.recipe_service')
    def test_preview_recipe_not_found(self, mock_service, test_client: TestClient):
        """Test preview with non-existent recipe/photo"""
        mock_service.preview_recipe.return_value = None
        
        response = test_client.get("/api/recipes/nonexistent/preview?photo_id=photo123")
        assert response.status_code == 404
    
    @patch('routes.recipes.recipe_service')
    def test_get_recipe_presets(self, mock_service, test_client: TestClient):
        """Test getting recipe presets"""
        mock_service.get_presets.return_value = [
            {
                "id": "auto-enhance",
                "name": "Auto Enhance",
                "description": "Automatic enhancement",
                "operations": [{"type": "auto", "parameters": {}}]
            },
            {
                "id": "sports-action",
                "name": "Sports Action",
                "description": "For sports photos",
                "operations": [{"type": "enhance", "parameters": {"contrast": 0.15}}]
            }
        ]
        
        response = test_client.get("/api/recipes/presets/")
        assert response.status_code == 200
        data = response.json()
        assert len(data["presets"]) == 2
        assert data["presets"][0]["id"] == "auto-enhance"
        assert data["presets"][1]["id"] == "sports-action"