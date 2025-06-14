"""
Tests for statistics routes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

class TestStatsRoutes:
    """Test statistics and dashboard routes"""
    
    @patch('routes.stats.stats_service')
    def test_get_dashboard_stats(self, mock_service, test_client: TestClient):
        """Test getting dashboard statistics"""
        mock_service.get_dashboard_stats.return_value = {
            "totalPhotos": 100,
            "processedToday": 25,
            "inQueue": 5,
            "failedToday": 2,
            "averageProcessingTime": 3.5,
            "storageUsed": {
                "originals": {"bytes": 1000000, "formatted": "1.00 MB"},
                "processed": {"bytes": 800000, "formatted": "800.00 KB"}
            },
            "recentActivity": [],
            "systemStatus": {
                "processing": "active",
                "aiModels": "loaded",
                "storage": "healthy"
            }
        }
        
        response = test_client.get("/api/stats/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert data["totalPhotos"] == 100
        assert data["processedToday"] == 25
        assert data["systemStatus"]["processing"] == "active"
    
    @patch('routes.stats.stats_service')
    def test_get_processing_stats(self, mock_service, test_client: TestClient):
        """Test getting processing statistics"""
        mock_service.get_processing_stats.return_value = {
            "period": "today",
            "processed": 50,
            "failed": 3,
            "averageTime": 2.8,
            "successRate": 94.3,
            "topRecipes": ["auto-enhance", "sports-action"],
            "hourlyDistribution": []
        }
        
        response = test_client.get("/api/stats/processing")
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "today"
        assert data["processed"] == 50
        assert data["successRate"] == 94.3
        
        # Test with different period
        response = test_client.get("/api/stats/processing?period=week")
        assert response.status_code == 200
        mock_service.get_processing_stats.assert_called_with("week")
    
    @patch('routes.stats.stats_service')
    def test_get_storage_stats(self, mock_service, test_client: TestClient):
        """Test getting storage statistics"""
        mock_service.get_storage_stats.return_value = {
            "originals": {"bytes": 5000000000, "formatted": "5.00 GB"},
            "processed": {"bytes": 3000000000, "formatted": "3.00 GB"},
            "total": {"bytes": 8000000000, "formatted": "8.00 GB"},
            "disk": {
                "total": "100.00 GB",
                "used": "40.00 GB",
                "free": "60.00 GB",
                "percentUsed": 40.0
            }
        }
        
        response = test_client.get("/api/stats/storage")
        assert response.status_code == 200
        data = response.json()
        assert data["total"]["formatted"] == "8.00 GB"
        assert data["disk"]["percentUsed"] == 40.0
    
    @patch('routes.stats.stats_service')
    def test_get_recent_activity(self, mock_service, test_client: TestClient):
        """Test getting recent activity"""
        mock_service.get_recent_activity.return_value = [
            {
                "timestamp": "2024-01-01T12:00:00",
                "type": "photo_processed",
                "message": "Processed photo123.jpg",
                "details": {"photo_id": "abc123"}
            },
            {
                "timestamp": "2024-01-01T11:58:00",
                "type": "recipe_applied",
                "message": "Applied 'Auto Enhance' recipe",
                "details": {"recipe_id": "recipe123"}
            }
        ]
        
        response = test_client.get("/api/stats/activity")
        assert response.status_code == 200
        data = response.json()
        assert len(data["activities"]) == 2
        assert data["activities"][0]["type"] == "photo_processed"
        
        # Test with limit
        response = test_client.get("/api/stats/activity?limit=10")
        assert response.status_code == 200
        mock_service.get_recent_activity.assert_called_with(10)
    
    @patch('routes.stats.stats_service')
    def test_get_performance_metrics(self, mock_service, test_client: TestClient):
        """Test getting performance metrics"""
        mock_service.get_performance_metrics.return_value = {
            "cpu": {"usage": 45.2, "cores": 8},
            "memory": {"used": 2048, "total": 8192, "percent": 25.0},
            "gpu": {"available": True, "usage": 30.5},
            "processingSpeed": {
                "current": 2.5,
                "average": 3.0,
                "peak": 5.2
            }
        }
        
        response = test_client.get("/api/stats/performance")
        assert response.status_code == 200
        data = response.json()
        assert data["cpu"]["usage"] == 45.2
        assert data["memory"]["percent"] == 25.0
        assert data["gpu"]["available"] is True
    
    @patch('routes.stats.stats_service')
    def test_get_processing_trends(self, mock_service, test_client: TestClient):
        """Test getting processing trends"""
        mock_service.get_processing_trends.return_value = {
            "days": 7,
            "trends": [
                {"date": "2024-01-01", "processed": 50, "failed": 2, "averageTime": 3.0},
                {"date": "2024-01-02", "processed": 45, "failed": 1, "averageTime": 2.8}
            ],
            "summary": {
                "totalProcessed": 350,
                "totalFailed": 15,
                "averageDaily": 50.0
            }
        }
        
        response = test_client.get("/api/stats/trends")
        assert response.status_code == 200
        data = response.json()
        assert data["days"] == 7
        assert len(data["trends"]) == 2
        assert data["summary"]["totalProcessed"] == 350
        
        # Test with custom days
        response = test_client.get("/api/stats/trends?days=30")
        assert response.status_code == 200
        mock_service.get_processing_trends.assert_called_with(30)
    
    @patch('routes.stats.stats_service')
    def test_get_ai_performance_stats(self, mock_service, test_client: TestClient):
        """Test getting AI performance statistics"""
        mock_service.get_ai_performance_stats.return_value = {
            "models": {
                "objectDetection": {
                    "averageTime": 0.5,
                    "successRate": 98.5,
                    "totalRuns": 1000
                },
                "qualityAssessment": {
                    "averageTime": 0.3,
                    "successRate": 99.2,
                    "totalRuns": 1000
                }
            },
            "overall": {
                "averageTime": 0.8,
                "successRate": 98.8
            }
        }
        
        response = test_client.get("/api/stats/ai-performance")
        assert response.status_code == 200
        data = response.json()
        assert data["models"]["objectDetection"]["successRate"] == 98.5
        assert data["overall"]["averageTime"] == 0.8
    
    @patch('routes.stats.stats_service')
    def test_get_error_stats(self, mock_service, test_client: TestClient):
        """Test getting error statistics"""
        mock_service.get_error_stats.return_value = {
            "period": "today",
            "totalErrors": 5,
            "errorsByType": {
                "processing_failed": 3,
                "upload_failed": 2
            },
            "commonIssues": [
                {"type": "out_of_memory", "count": 2},
                {"type": "invalid_format", "count": 3}
            ],
            "errorRate": 2.5
        }
        
        response = test_client.get("/api/stats/errors")
        assert response.status_code == 200
        data = response.json()
        assert data["totalErrors"] == 5
        assert data["errorsByType"]["processing_failed"] == 3
        assert len(data["commonIssues"]) == 2
        
        # Test with period
        response = test_client.get("/api/stats/errors?period=week")
        assert response.status_code == 200
        mock_service.get_error_stats.assert_called_with("week")
    
    @patch('routes.stats.stats_service')
    def test_stats_service_error(self, mock_service, test_client: TestClient):
        """Test handling service errors"""
        mock_service.get_dashboard_stats.side_effect = Exception("Database error")
        
        response = test_client.get("/api/stats/dashboard")
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]