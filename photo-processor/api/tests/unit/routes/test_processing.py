"""
Tests for processing routes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock

from models.processing import (
    QueueStatus, ProcessingStatus, ProcessingSettings,
    BatchOperation, QueueItem
)

class TestProcessingRoutes:
    """Test processing control routes"""
    
    @patch('routes.processing.processing_service')
    def test_get_queue_status(self, mock_service, test_client: TestClient, sample_queue_item, sample_processing_stats):
        """Test getting queue status"""
        mock_service.get_queue_status.return_value = QueueStatus(
            pending=[QueueItem(**sample_queue_item)],
            processing=[],
            completed=[],
            is_paused=False,
            stats=ProcessingStatus(**sample_processing_stats)
        )
        
        response = test_client.get("/api/processing/queue")
        assert response.status_code == 200
        data = response.json()
        assert len(data["pending"]) == 1
        assert data["pending"][0]["photo_id"] == "photo123"
        assert data["is_paused"] is False
        assert data["stats"]["queue_length"] == 5
    
    @patch('routes.processing.processing_service')
    def test_get_processing_status(self, mock_service, test_client: TestClient, sample_processing_stats):
        """Test getting processing status"""
        mock_service.get_processing_status.return_value = ProcessingStatus(**sample_processing_stats)
        
        response = test_client.get("/api/processing/status")
        assert response.status_code == 200
        data = response.json()
        assert data["is_paused"] is False
        assert data["queue_length"] == 5
        assert data["processing_rate"] == 2.5
        assert data["average_time"] == 3.2
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_pause_processing(self, mock_service, mock_ws, test_client: TestClient):
        """Test pausing processing"""
        mock_service.pause_processing.return_value = True
        mock_ws.broadcast = AsyncMock()
        
        response = test_client.put("/api/processing/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"
        assert "successfully" in data["message"]
        
        # Verify WebSocket broadcast
        mock_ws.broadcast.assert_called_once()
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_pause_processing_failure(self, mock_service, mock_ws, test_client: TestClient):
        """Test pause processing failure"""
        mock_service.pause_processing.return_value = False
        
        response = test_client.put("/api/processing/pause")
        assert response.status_code == 500
        assert "Failed to pause" in response.json()["detail"]
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_resume_processing(self, mock_service, mock_ws, test_client: TestClient):
        """Test resuming processing"""
        mock_service.resume_processing.return_value = True
        mock_ws.broadcast = AsyncMock()
        
        response = test_client.put("/api/processing/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "resumed"
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_approve_processing(self, mock_service, mock_ws, test_client: TestClient):
        """Test approving processing"""
        mock_service.approve_processing.return_value = {
            "photo_id": "abc123",
            "status": "processing",
            "message": "Processing approved"
        }
        mock_ws.notify_processing_started = AsyncMock()
        
        adjustments = {"brightness": 0.1, "contrast": 0.2}
        response = test_client.post(
            "/api/processing/approve/abc123",
            json=adjustments
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        
        # Verify service called with adjustments
        mock_service.approve_processing.assert_called_with(
            photo_id="abc123",
            adjustments=adjustments
        )
        mock_ws.notify_processing_started.assert_called_with("abc123")
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_reject_processing(self, mock_service, mock_ws, test_client: TestClient):
        """Test rejecting processing"""
        mock_service.reject_processing.return_value = {
            "photo_id": "abc123",
            "status": "rejected",
            "reason": "Poor quality"
        }
        mock_ws.broadcast = AsyncMock()
        
        response = test_client.post(
            "/api/processing/reject/abc123?reason=Poor%20quality"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        
        # Verify broadcast
        mock_ws.broadcast.assert_called_once()
        broadcast_data = mock_ws.broadcast.call_args[0][0]
        assert broadcast_data["type"] == "processing_rejected"
        assert broadcast_data["photoId"] == "abc123"
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_batch_process(self, mock_service, mock_ws, test_client: TestClient):
        """Test batch processing"""
        mock_service.batch_process.return_value = {
            "queued": 3,
            "message": "Queued 3 photos for processing"
        }
        mock_service.get_queue_stats.return_value = {
            "pending": 5,
            "processing": 1,
            "completed": 10
        }
        mock_ws.notify_queue_updated = AsyncMock()
        
        batch_data = {
            "photo_ids": ["photo1", "photo2", "photo3"],
            "recipe_id": "recipe123",
            "priority": "high",
            "skip_ai": False
        }
        
        response = test_client.post("/api/processing/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert data["queued"] == 3
        
        # Verify WebSocket notification
        mock_ws.notify_queue_updated.assert_called_once()
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_reorder_queue(self, mock_service, mock_ws, test_client: TestClient):
        """Test reordering queue"""
        mock_service.reorder_queue.return_value = True
        mock_service.get_queue_stats.return_value = {"pending": 5}
        mock_ws.notify_queue_updated = AsyncMock()
        
        response = test_client.put(
            "/api/processing/reorder?photo_id=photo123&new_position=3"
        )
        assert response.status_code == 200
        assert "successfully" in response.json()["message"]
        
        mock_service.reorder_queue.assert_called_with("photo123", 3)
    
    @patch('routes.processing.processing_service')
    def test_reorder_queue_not_found(self, mock_service, test_client: TestClient):
        """Test reordering non-existent queue item"""
        mock_service.reorder_queue.return_value = False
        
        response = test_client.put(
            "/api/processing/reorder?photo_id=nonexistent&new_position=1"
        )
        assert response.status_code == 404
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_remove_from_queue(self, mock_service, mock_ws, test_client: TestClient):
        """Test removing from queue"""
        mock_service.remove_from_queue.return_value = True
        mock_service.get_queue_stats.return_value = {"pending": 4}
        mock_ws.notify_queue_updated = AsyncMock()
        
        response = test_client.delete("/api/processing/queue/photo123")
        assert response.status_code == 200
        assert "removed" in response.json()["message"]
    
    @patch('routes.processing.processing_service')
    def test_get_processing_settings(self, mock_service, test_client: TestClient):
        """Test getting processing settings"""
        mock_service.get_settings.return_value = ProcessingSettings(
            auto_process=True,
            require_approval=False,
            max_concurrent=2,
            quality_threshold=7.0,
            enable_ai=True
        )
        
        response = test_client.get("/api/processing/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["auto_process"] is True
        assert data["max_concurrent"] == 2
        assert data["quality_threshold"] == 7.0
    
    @patch('routes.processing.processing_service')
    @patch('routes.processing.ws_manager')
    def test_update_processing_settings(self, mock_service, mock_ws, test_client: TestClient):
        """Test updating processing settings"""
        new_settings = ProcessingSettings(
            auto_process=False,
            require_approval=True,
            max_concurrent=4,
            quality_threshold=8.0,
            enable_ai=True
        )
        mock_service.update_settings.return_value = new_settings
        mock_ws.broadcast = AsyncMock()
        
        response = test_client.put(
            "/api/processing/settings",
            json=new_settings.dict()
        )
        assert response.status_code == 200
        data = response.json()
        assert data["auto_process"] is False
        assert data["require_approval"] is True
        
        # Verify broadcast
        mock_ws.broadcast.assert_called_once()