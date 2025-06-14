"""
Static file serving routes
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Base data directory
DATA_PATH = Path("/app/data")

@router.get("/{file_type}/{file_path:path}")
async def serve_file(file_type: str, file_path: str):
    """Serve static files from data directories"""
    
    # Validate file type
    allowed_types = ["processed", "thumbnails", "web", "inbox", "originals", "temp"]
    if file_type not in allowed_types:
        raise HTTPException(status_code=404, detail="Invalid file type")
    
    # Construct full path
    full_path = DATA_PATH / file_type / file_path
    
    # Security check - ensure path doesn't escape data directory
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(DATA_PATH)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=404, detail="Invalid path")
    
    # Check if file exists
    if not full_path.exists() or not full_path.is_file():
        logger.warning(f"File not found: {full_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type
    content_type = "application/octet-stream"
    if full_path.suffix.lower() in ['.jpg', '.jpeg']:
        content_type = "image/jpeg"
    elif full_path.suffix.lower() == '.png':
        content_type = "image/png"
    elif full_path.suffix.lower() in ['.tif', '.tiff']:
        content_type = "image/tiff"
    
    logger.info(f"Serving file: {full_path} as {content_type}")
    return FileResponse(full_path, media_type=content_type)