"""
WebSocket integration tests
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

class TestWebSocketIntegration:
    """Test WebSocket functionality"""
    
    def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        from main import app
        client = TestClient(app)
        
        with client.websocket_connect("/ws") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "timestamp" in data
    
    def test_websocket_keep_alive(self):
        """Test WebSocket stays alive"""
        from main import app
        client = TestClient(app)
        
        with client.websocket_connect("/ws") as websocket:
            # Receive initial connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            
            # Send a message
            websocket.send_text("ping")
            
            # Connection should still be alive
            # Send another message
            websocket.send_text("test message")
    
    @patch('main.ws_manager')
    def test_websocket_broadcast_integration(self, mock_ws_manager):
        """Test WebSocket receives broadcasts"""
        from main import app
        client = TestClient(app)
        
        # Setup mock to track connections
        connections = set()
        
        async def mock_connect(ws):
            connections.add(ws)
            await ws.accept()
            await ws.send_json({
                "type": "connection",
                "status": "connected"
            })
        
        def mock_disconnect(ws):
            connections.discard(ws)
        
        mock_ws_manager.connect = mock_connect
        mock_ws_manager.disconnect = mock_disconnect
        
        with client.websocket_connect("/ws") as websocket:
            # Should receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            
            # Verify connection tracked
            assert mock_ws_manager.connect.called
    
    def test_multiple_websocket_clients(self):
        """Test multiple concurrent WebSocket connections"""
        from main import app
        client = TestClient(app)
        
        # Connect multiple clients
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                # Both should receive connection messages
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()
                
                assert data1["type"] == "connection"
                assert data2["type"] == "connection"
                
                # Both connections should work independently
                ws1.send_text("client1 message")
                ws2.send_text("client2 message")
    
    def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        from main import app
        client = TestClient(app)
        
        with pytest.raises(Exception):
            # Try to receive without connecting
            with client.websocket_connect("/invalid-ws-endpoint") as websocket:
                websocket.receive_json()