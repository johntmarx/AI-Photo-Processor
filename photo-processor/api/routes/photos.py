"""
Photo management routes
"""

from fastapi import APIRouter, Query, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
from pathlib import Path
import hashlib
from datetime import datetime

from models.photo import Photo, PhotoDetail, PhotoList, PhotoComparison
from services.photo_service_sqlite import sqlite_photo_service as photo_service
from middleware.transform import backend_to_frontend, transform_photo, transform_pagination_response

router = APIRouter()
# photo_service is imported as singleton from photo_service_v2

@router.get("")
async def list_photos(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort field"),
    order: str = Query("desc", description="Sort order (asc/desc)"),
    search: Optional[str] = Query(None, description="Search query"),
    sort: Optional[str] = Query(None, description="Sort by field_order format"),
    min_aesthetic_score: Optional[float] = Query(None, ge=0, le=10, description="Minimum aesthetic score"),
    max_aesthetic_score: Optional[float] = Query(None, ge=0, le=10, description="Maximum aesthetic score"),
    min_technical_score: Optional[float] = Query(None, ge=0, le=10, description="Minimum technical score"),
    max_technical_score: Optional[float] = Query(None, ge=0, le=10, description="Maximum technical score")
):
    """List photos with pagination and filtering"""
    try:
        # Handle sort parameter format (e.g., "aesthetic_desc")
        if sort:
            parts = sort.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['asc', 'desc']:
                sort_by = parts[0]
                order = parts[1]
        
        photos = await photo_service.list_photos(
            page=page,
            page_size=page_size,
            status=status,
            sort_by=sort_by,
            order=order,
            search=search,
            min_aesthetic_score=min_aesthetic_score,
            max_aesthetic_score=max_aesthetic_score,
            min_technical_score=min_technical_score,
            max_technical_score=max_technical_score
        )
        # Transform pagination response with photo-specific transformations
        response = photos.dict()
        
        # Transform each photo to ensure proper field mapping
        transformed_photos = []
        for photo in response['photos']:
            # Ensure datetime fields are properly serialized
            if isinstance(photo.get('created_at'), datetime):
                photo['created_at'] = photo['created_at'].isoformat()
            if isinstance(photo.get('processed_at'), datetime):
                photo['processed_at'] = photo['processed_at'].isoformat()
            
            # Transform the photo data
            transformed_photo = transform_photo(photo)
            transformed_photos.append(transformed_photo)
        
        response['photos'] = transformed_photos
        return backend_to_frontend(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{photo_id}/thumbnail")
async def get_photo_thumbnail(photo_id: str):
    """Get thumbnail image for a photo"""
    photo = await photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Check if photo is still processing
    if photo.status in ['pending', 'queued', 'processing']:
        return JSONResponse(
            status_code=202,  # Accepted - processing
            content={"status": "processing", "message": "Thumbnail not ready", "retry_after": 1}
        )
    
    # Try to find thumbnail file
    data_path = Path("/app/data")
    thumbnail_path = data_path / "thumbnails" / f"{photo_id}_thumb.jpg"
    
    if thumbnail_path.exists():
        return FileResponse(thumbnail_path, media_type="image/jpeg")
    
    # Fallback to web version
    web_path = data_path / "web" / f"{photo_id}_web.jpg"
    if web_path.exists():
        return FileResponse(web_path, media_type="image/jpeg")
    
    # Fallback to processed
    processed_path = data_path / "processed" / f"{photo_id}_{photo.filename}"
    if processed_path.exists():
        return FileResponse(processed_path)
    
    # If photo is completed but no thumbnail found, it's still processing or failed
    if photo.status == 'completed':
        # Try one more time with the path from database
        if photo.thumbnail_path and Path(photo.thumbnail_path).exists():
            return FileResponse(photo.thumbnail_path, media_type="image/jpeg")
        # Thumbnail generation must have failed
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")
    else:
        # Still processing
        return JSONResponse(
            status_code=202,
            content={"status": "processing", "message": "Thumbnail not ready", "retry_after": 1}
        )

@router.get("/{photo_id}/preview")
async def get_photo_preview(photo_id: str):
    """Get preview image for a photo"""
    photo = await photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Try to find web-optimized version first
    data_path = Path("/app/data")
    web_path = data_path / "web" / f"{photo_id}_web.jpg"
    
    if web_path.exists():
        return FileResponse(web_path, media_type="image/jpeg")
    
    # Fallback to processed
    processed_path = data_path / "processed" / f"{photo_id}_{photo.filename}"
    if processed_path.exists():
        return FileResponse(processed_path)
    
    # Fallback to thumbnail
    thumbnail_path = data_path / "thumbnails" / f"{photo_id}_thumb.jpg"
    if thumbnail_path.exists():
        return FileResponse(thumbnail_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Preview not found")

@router.get("/{photo_id}/download")
async def download_photo(photo_id: str):
    """Download original or best quality version of photo"""
    photo = await photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Try original first
    data_path = Path("/app/data")
    original_path = data_path / "originals" / f"{photo_id}_{photo.filename}"
    
    if original_path.exists():
        return FileResponse(
            original_path, 
            filename=photo.filename,
            media_type="application/octet-stream"
        )
    
    # Fallback to processed
    processed_path = data_path / "processed" / f"{photo_id}_{photo.filename}"
    if processed_path.exists():
        return FileResponse(
            processed_path,
            filename=photo.filename,
            media_type="application/octet-stream"
        )
    
    raise HTTPException(status_code=404, detail="Photo file not found")

@router.get("/{photo_id}/comparison", response_model=PhotoComparison)
async def get_photo_comparison(photo_id: str):
    """Get comparison data for original vs processed photo"""
    comparison = await photo_service.get_comparison(photo_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Photo not found")
    return comparison

@router.get("/{photo_id}/ai-analysis")
async def get_ai_analysis(photo_id: str):
    """Get AI analysis results for a photo"""
    analysis = await photo_service.get_ai_analysis(photo_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis

@router.get("/{photo_id}")
async def get_photo(photo_id: str):
    """Get detailed information about a specific photo"""
    photo = await photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    # Transform photo detail
    return transform_photo(photo.dict())

@router.post("/upload")
async def upload_photo(
    file: UploadFile = File(...),
    auto_process: bool = Query(True, description="Automatically process the photo"),
    recipe_id: Optional[str] = Query(None, description="Recipe to apply")
):
    """Upload a new photo for processing"""
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', 
                         '.nef', '.cr2', '.arw', '.dng', '.orf', '.rw2'}
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not supported"
        )
    
    try:
        # Save uploaded file
        result = await photo_service.save_upload(
            file=file,
            auto_process=auto_process,
            recipe_id=recipe_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{photo_id}")
async def delete_photo(photo_id: str, delete_original: bool = Query(False)):
    """Delete a photo (processed version only by default)"""
    success = await photo_service.delete_photo(
        photo_id=photo_id,
        delete_original=delete_original
    )
    if not success:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    return {"message": "Photo deleted successfully", "photo_id": photo_id}

@router.post("/{photo_id}/reprocess")
async def reprocess_photo(
    photo_id: str,
    recipe_id: Optional[str] = Query(None, description="Recipe to apply"),
    priority: str = Query("normal", description="Processing priority")
):
    """Reprocess a photo with optional recipe"""
    result = await photo_service.reprocess_photo(
        photo_id=photo_id,
        recipe_id=recipe_id,
        priority=priority
    )
    if not result:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    return result

