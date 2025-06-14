"""
Tests for WebSocket manager
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from fastapi import WebSocket
from datetime import datetime

from services.websocket_manager import WebSocketManager

class TestWebSocketManager:
    """Test WebSocket manager functionality"""
    
    @pytest.fixture
    def ws_manager(self):
        """Create WebSocket manager instance"""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket"""
        ws = Mock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        return ws
    
    async def test_connect(self, ws_manager, mock_websocket):
        """Test WebSocket connection"""
        await ws_manager.connect(mock_websocket)
        
        # Verify WebSocket accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify connection added
        assert mock_websocket in ws_manager.active_connections
        
        # Verify initial message sent
        mock_websocket.send_json.assert_called_once()
        call_data = mock_websocket.send_json.call_args[0][0]
        assert call_data["type"] == "connection"
        assert call_data["status"] == "connected"
    
    def test_disconnect(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection"""
        # Add connection first
        ws_manager.active_connections.add(mock_websocket)
        
        # Disconnect
        ws_manager.disconnect(mock_websocket)
        
        # Verify removed
        assert mock_websocket not in ws_manager.active_connections
    
    async def test_send_to_client_success(self, ws_manager, mock_websocket):
        """Test sending data to specific client"""
        data = {"type": "test", "message": "Hello"}
        
        await ws_manager.send_to_client(mock_websocket, data)
        
        mock_websocket.send_json.assert_called_once_with(data)
    
    async def test_send_to_client_error(self, ws_manager, mock_websocket):
        """Test handling send errors"""
        # Setup error
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        ws_manager.active_connections.add(mock_websocket)
        
        data = {"type": "test"}
        await ws_manager.send_to_client(mock_websocket, data)
        
        # Verify client disconnected on error
        assert mock_websocket not in ws_manager.active_connections
    
    async def test_broadcast_empty(self, ws_manager):
        """Test broadcast with no connections"""
        # Should not raise error
        await ws_manager.broadcast({"type": "test"})
    
    async def test_broadcast_multiple_clients(self, ws_manager):
        """Test broadcasting to multiple clients"""
        # Create multiple mock clients
        clients = []
        for i in range(3):
            ws = Mock(spec=WebSocket)
            ws.send_json = AsyncMock()
            clients.append(ws)
            ws_manager.active_connections.add(ws)
        
        data = {"type": "broadcast", "message": "Hello all"}
        await ws_manager.broadcast(data)
        
        # Verify all clients received message
        for client in clients:
            client.send_json.assert_called_once_with(data)
    
    async def test_broadcast_with_failed_client(self, ws_manager):
        """Test broadcast continues despite client failures"""
        # Create mix of good and bad clients
        good_client = Mock(spec=WebSocket)
        good_client.send_json = AsyncMock()
        
        bad_client = Mock(spec=WebSocket)
        bad_client.send_json = AsyncMock(side_effect=Exception("Failed"))
        
        ws_manager.active_connections.add(good_client)
        ws_manager.active_connections.add(bad_client)
        
        data = {"type": "test"}
        await ws_manager.broadcast(data)
        
        # Good client should still receive message
        good_client.send_json.assert_called_once_with(data)
        
        # Bad client should be removed
        assert bad_client not in ws_manager.active_connections
        assert good_client in ws_manager.active_connections
    
    async def test_disconnect_all(self, ws_manager):
        """Test disconnecting all clients"""
        # Add multiple clients
        clients = []
        for i in range(3):
            ws = Mock(spec=WebSocket)
            ws.close = AsyncMock()
            clients.append(ws)
            ws_manager.active_connections.add(ws)
        
        await ws_manager.disconnect_all()
        
        # Verify all closed
        for client in clients:
            client.close.assert_called_once()
        
        # Verify connections cleared
        assert len(ws_manager.active_connections) == 0
    
    async def test_notify_processing_started(self, ws_manager):
        """Test processing started notification"""
        ws_manager.broadcast = AsyncMock()
        
        await ws_manager.notify_processing_started("photo123", "recipe456")
        
        ws_manager.broadcast.assert_called_once()
        call_data = ws_manager.broadcast.call_args[0][0]
        assert call_data["type"] == "processing_started"
        assert call_data["photoId"] == "photo123"
        assert call_data["recipeId"] == "recipe456"
        assert "timestamp" in call_data
    
    async def test_notify_processing_completed(self, ws_manager):
        """Test processing completed notification"""
        ws_manager.broadcast = AsyncMock()
        
        await ws_manager.notify_processing_completed(
            "photo123",
            success=True,
            processed_path="/path/to/processed.jpg"
        )
        
        ws_manager.broadcast.assert_called_once()
        call_data = ws_manager.broadcast.call_args[0][0]
        assert call_data["type"] == "processing_completed"
        assert call_data["photoId"] == "photo123"
        assert call_data["success"] is True
        assert call_data["processedPath"] == "/path/to/processed.jpg"
    
    async def test_notify_processing_failed(self, ws_manager):
        """Test processing failed notification"""
        ws_manager.broadcast = AsyncMock()
        
        await ws_manager.notify_processing_failed("photo123", "Out of memory")
        
        ws_manager.broadcast.assert_called_once()
        call_data = ws_manager.broadcast.call_args[0][0]
        assert call_data["type"] == "processing_failed"
        assert call_data["photoId"] == "photo123"
        assert call_data["error"] == "Out of memory"
    
    async def test_notify_queue_updated(self, ws_manager):
        """Test queue update notification"""
        ws_manager.broadcast = AsyncMock()
        
        queue_stats = {
            "pending": 10,
            "processing": 2,
            "completed": 50
        }
        
        await ws_manager.notify_queue_updated(queue_stats)
        
        ws_manager.broadcast.assert_called_once()
        call_data = ws_manager.broadcast.call_args[0][0]
        assert call_data["type"] == "queue_updated"
        assert call_data["pending"] == 10
        assert call_data["processing"] == 2
        assert call_data["completed"] == 50
    
    async def test_notify_stats_updated(self, ws_manager):
        """Test statistics update notification"""
        ws_manager.broadcast = AsyncMock()
        
        stats = {
            "totalPhotos": 100,
            "processedToday": 25,
            "averageTime": 3.5
        }
        
        await ws_manager.notify_stats_updated(stats)
        
        ws_manager.broadcast.assert_called_once()
        call_data = ws_manager.broadcast.call_args[0][0]
        assert call_data["type"] == "stats_updated"
        assert call_data["stats"] == stats
    
    async def test_concurrent_broadcasts(self, ws_manager):
        """Test handling concurrent broadcasts"""
        ws_manager.broadcast = AsyncMock()
        
        # Create multiple broadcast tasks
        tasks = [
            ws_manager.notify_processing_started(f"photo{i}", None)
            for i in range(5)
        ]
        
        # Run concurrently
        await asyncio.gather(*tasks)
        
        # Verify all broadcasts sent
        assert ws_manager.broadcast.call_count == 5