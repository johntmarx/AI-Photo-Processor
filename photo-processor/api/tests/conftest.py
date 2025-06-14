"""
Pytest configuration and fixtures for API tests
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from services.websocket_manager import WebSocketManager
from services.photo_service import PhotoService
from services.processing_service import ProcessingService
from services.recipe_service import RecipeService
from services.stats_service import StatsService

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client() -> TestClient:
    """Create test client for sync tests"""
    return TestClient(app)

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary data directory"""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir)
    
    # Create subdirectories
    (data_path / "inbox").mkdir(parents=True)
    (data_path / "originals").mkdir(parents=True)
    (data_path / "processed").mkdir(parents=True)
    (data_path / "recipes").mkdir(parents=True)
    
    yield data_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_photo_service() -> Mock:
    """Mock PhotoService"""
    service = Mock(spec=PhotoService)
    service.list_photos = AsyncMock()
    service.get_photo = AsyncMock()
    service.get_comparison = AsyncMock()
    service.save_upload = AsyncMock()
    service.delete_photo = AsyncMock()
    service.reprocess_photo = AsyncMock()
    service.get_ai_analysis = AsyncMock()
    return service

@pytest.fixture
def mock_processing_service() -> Mock:
    """Mock ProcessingService"""
    service = Mock(spec=ProcessingService)
    service.get_queue_status = AsyncMock()
    service.get_processing_status = AsyncMock()
    service.pause_processing = AsyncMock()
    service.resume_processing = AsyncMock()
    service.approve_processing = AsyncMock()
    service.reject_processing = AsyncMock()
    service.batch_process = AsyncMock()
    service.reorder_queue = AsyncMock()
    service.remove_from_queue = AsyncMock()
    service.get_settings = AsyncMock()
    service.update_settings = AsyncMock()
    service.get_queue_stats = AsyncMock()
    return service

@pytest.fixture
def mock_recipe_service() -> Mock:
    """Mock RecipeService"""
    service = Mock(spec=RecipeService)
    service.list_recipes = AsyncMock()
    service.get_recipe = AsyncMock()
    service.create_recipe = AsyncMock()
    service.update_recipe = AsyncMock()
    service.delete_recipe = AsyncMock()
    service.duplicate_recipe = AsyncMock()
    service.apply_to_photos = AsyncMock()
    service.preview_recipe = AsyncMock()
    service.get_presets = AsyncMock()
    return service

@pytest.fixture
def mock_stats_service() -> Mock:
    """Mock StatsService"""
    service = Mock(spec=StatsService)
    service.get_dashboard_stats = AsyncMock()
    service.get_processing_stats = AsyncMock()
    service.get_storage_stats = AsyncMock()
    service.get_recent_activity = AsyncMock()
    service.get_performance_metrics = AsyncMock()
    service.get_processing_trends = AsyncMock()
    service.get_ai_performance_stats = AsyncMock()
    service.get_error_stats = AsyncMock()
    return service

@pytest.fixture
def mock_websocket_manager() -> Mock:
    """Mock WebSocketManager"""
    manager = Mock(spec=WebSocketManager)
    manager.connect = AsyncMock()
    manager.disconnect = Mock()
    manager.send_to_client = AsyncMock()
    manager.broadcast = AsyncMock()
    manager.disconnect_all = AsyncMock()
    manager.notify_processing_started = AsyncMock()
    manager.notify_processing_completed = AsyncMock()
    manager.notify_processing_failed = AsyncMock()
    manager.notify_queue_updated = AsyncMock()
    manager.notify_stats_updated = AsyncMock()
    return manager

@pytest.fixture
def sample_photo_data():
    """Sample photo data for tests"""
    return {
        "id": "abc123",
        "filename": "test_photo.jpg",
        "original_path": "/app/data/originals/2024/01/abc123_test.jpg",
        "processed_path": "/app/data/processed/2024/01/abc123_processed.jpg",
        "status": "completed",
        "created_at": "2024-01-01T12:00:00",
        "processed_at": "2024-01-01T12:01:00",
        "file_size": 1024000
    }

@pytest.fixture
def sample_recipe_data():
    """Sample recipe data for tests"""
    return {
        "id": "recipe123",
        "name": "Test Recipe",
        "description": "A test recipe",
        "operations": [
            {"type": "enhance", "parameters": {"brightness": 0.1}},
            {"type": "crop", "parameters": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}}
        ],
        "isDefault": False,
        "createdAt": "2024-01-01T12:00:00",
        "updatedAt": "2024-01-01T12:00:00",
        "usageCount": 5
    }

@pytest.fixture
def sample_queue_item():
    """Sample queue item for tests"""
    return {
        "photo_id": "photo123",
        "filename": "test.jpg",
        "position": 1,
        "added_at": "2024-01-01T12:00:00",
        "priority": "normal",
        "recipe_id": None,
        "manual_approval": False
    }

@pytest.fixture
def sample_processing_stats():
    """Sample processing statistics"""
    return {
        "is_paused": False,
        "current_photo": None,
        "queue_length": 5,
        "processing_rate": 2.5,
        "average_time": 3.2,
        "errors_today": 0
    }