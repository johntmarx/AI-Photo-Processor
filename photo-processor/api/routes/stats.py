"""
Statistics and dashboard routes
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from services.stats_service import StatsService
from tasks.ai_tasks import cleanup_gpu_memory_task

router = APIRouter()
stats_service = StatsService()

@router.get("/dashboard")
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        stats = await stats_service.get_dashboard_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing")
async def get_processing_stats(
    period: str = Query("today", description="Time period: today, week, month, all")
):
    """Get processing statistics for a time period"""
    try:
        stats = await stats_service.get_processing_stats(period)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage")
async def get_storage_stats():
    """Get storage usage statistics"""
    try:
        stats = await stats_service.get_storage_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity")
async def get_recent_activity(
    limit: int = Query(20, ge=1, le=100, description="Number of activities to return")
):
    """Get recent system activity"""
    try:
        activities = await stats_service.get_recent_activity(limit)
        return {"activities": activities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics():
    """Get system performance metrics"""
    try:
        metrics = await stats_service.get_performance_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_processing_trends(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
):
    """Get processing trends over time"""
    try:
        trends = await stats_service.get_processing_trends(days)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-performance")
async def get_ai_performance_stats():
    """Get AI model performance statistics"""
    try:
        stats = await stats_service.get_ai_performance_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/errors")
async def get_error_stats(
    period: str = Query("today", description="Time period: today, week, month")
):
    """Get error statistics and common issues"""
    try:
        errors = await stats_service.get_error_stats(period)
        return errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gpu/cleanup")
async def cleanup_gpu_memory():
    """
    Clean up GPU memory by unloading all AI models.
    This is useful to free GPU memory for Ollama or other processes.
    
    Returns:
        Dict with cleanup status and memory info
    """
    try:
        # Queue the cleanup task
        task = cleanup_gpu_memory_task.delay()
        
        # Wait for result with timeout
        result = task.get(timeout=30)
        
        return {
            "status": "success",
            "task_id": task.id,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"GPU cleanup failed: {str(e)}"
        )

@router.get("/gpu/status")
async def get_gpu_status():
    """Get current GPU memory status"""
    try:
        import torch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_total_gb": round(memory_total, 2),
                "memory_free_gb": round(memory_total - memory_reserved, 2)
            }
        else:
            return {
                "gpu_available": False,
                "message": "No GPU available"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))