"""
Enhanced upload routes for large batch processing
Supports chunked uploads, progress tracking, and session management.
"""

from fastapi import APIRouter, HTTPException, Query, Body, UploadFile, File, Form
from typing import Optional, List, Dict, Any
from pathlib import Path

from services.upload_service import upload_service

router = APIRouter()

@router.post("/session")
async def create_upload_session(
    expected_files: int = Body(..., description="Number of files to upload"),
    total_size: Optional[int] = Body(None, description="Total size in bytes"),
    recipe_id: Optional[str] = Body(None, description="Recipe to apply"),
    auto_process: bool = Body(True, description="Auto-process uploads")
):
    """Create a new upload session for batch uploads"""
    try:
        if expected_files <= 0:
            raise HTTPException(status_code=400, detail="Expected files must be greater than 0")
        
        if expected_files > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 files per session")
        
        session_info = await upload_service.create_upload_session(
            expected_files=expected_files,
            total_size=total_size,
            recipe_id=recipe_id,
            auto_process=auto_process
        )
        
        return session_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/file")
async def upload_file_to_session(
    session_id: str,
    file: UploadFile = File(...),
    auto_process: bool = Form(True),
    recipe_id: Optional[str] = Form(None)
):
    """Upload a file to an existing session"""
    try:
        # Validate file size (500MB max)
        if file.size and file.size > 500 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 500MB)")
        
        result = await upload_service.upload_file_to_session(
            session_id=session_id,
            file=file,
            auto_process=auto_process,
            recipe_id=recipe_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    auto_process: bool = Form(True),
    recipe_id: Optional[str] = Form(None)
):
    """Upload multiple files in a single request"""
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files per batch request")
        
        # Validate total size
        total_size = sum(file.size or 0 for file in files)
        if total_size > 10 * 1024 * 1024 * 1024:  # 10GB
            raise HTTPException(status_code=400, detail="Total batch size too large (max 10GB)")
        
        result = await upload_service.upload_multiple_files(
            files=files,
            auto_process=auto_process,
            recipe_id=recipe_id
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/progress")
async def get_session_progress(session_id: str):
    """Get progress of an upload session"""
    progress = await upload_service.get_session_progress(session_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    return progress

@router.post("/session/{session_id}/complete")
async def complete_session(session_id: str):
    """Mark session as completed and start processing"""
    try:
        result = await upload_service.complete_session(session_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/session/{session_id}")
async def cancel_session(session_id: str):
    """Cancel an upload session and clean up files"""
    try:
        result = await upload_service.cancel_session(session_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_upload_stats():
    """Get overall upload statistics"""
    try:
        stats = await upload_service.get_upload_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_old_sessions(
    max_age_hours: int = Query(24, description="Maximum age in hours")
):
    """Clean up old upload sessions"""
    try:
        if max_age_hours < 1:
            raise HTTPException(status_code=400, detail="Max age must be at least 1 hour")
        
        result = await upload_service.cleanup_old_sessions(max_age_hours)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    stats = await upload_service.get_upload_stats()
    return {
        "supported_extensions": stats["supported_extensions"],
        "max_file_size_mb": 500,
        "max_batch_size_gb": 10,
        "max_files_per_batch": 50,
        "max_files_per_session": 1000
    }

@router.post("/single")
async def upload_single_file(
    file: UploadFile = File(...),
    recipe_id: Optional[str] = Form(None),
    auto_process: bool = Form(True)
):
    """Upload a single file (simplified endpoint for frontend)"""
    try:
        # Create a session for this single file
        session_info = await upload_service.create_upload_session(
            expected_files=1,
            total_size=None,
            recipe_id=recipe_id,
            auto_process=auto_process
        )
        
        # Upload the file to the session
        result = await upload_service.upload_file_to_session(
            session_id=session_info["session_id"],
            file=file,
            auto_process=auto_process,
            recipe_id=recipe_id
        )
        
        # Complete the session
        completion_result = await upload_service.complete_session(
            session_id=session_info["session_id"]
        )
        
        return {
            "status": "uploaded",
            "photo_id": result["photo_id"],
            "filename": result["filename"],
            "session_id": session_info["session_id"],
            "processing_queued": completion_result.get("processing_queued", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))