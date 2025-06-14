"""
Batch Processing Routes

Powerful batch processing with quality filtering, model selection,
and tar.gz download of processed images.
"""

from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
import json
import logging
import asyncio
from datetime import datetime
import tarfile
import tempfile
import shutil
from PIL import Image

from tasks.ai_tasks import analyze_rotation_cv, analyze_rotation_onealign, analyze_crop_vlm_round1, analyze_crop_vlm_round2
from services.photo_service_sqlite import sqlite_photo_service as photo_service
from services.intelligent_enhancer import IntelligentEnhancer
from services.websocket_manager import WebSocketManager
from ai_components.shared.image_utils import load_image

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for batch processing sessions
batch_sessions = {}

# Initialize services
intelligent_enhancer = IntelligentEnhancer()
ws_manager = None

def set_websocket_manager(manager: WebSocketManager):
    """Inject websocket manager from main.py"""
    global ws_manager
    ws_manager = manager


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""
    # Photo selection criteria
    min_aesthetic_score: Optional[float] = None
    min_technical_score: Optional[float] = None
    max_photos: Optional[int] = None  # Limit number of photos
    
    # Rotation settings
    rotation_enabled: bool = True
    rotation_method: str = "cv"  # "cv" or "vlm"
    rotation_vlm_model: Optional[str] = "qwen2.5-vl:7b"
    
    # Crop settings
    crop_enabled: bool = True
    crop_method: str = "cv"  # "cv" or "vlm"
    crop_vlm_model: Optional[str] = "qwen2.5-vl:7b"
    crop_aspect_ratio: str = "original"  # "original", "16:9", "4:3", etc.
    
    # Enhancement settings
    enhance_enabled: bool = True
    enhance_strength: float = 1.0
    
    # Output settings
    output_format: str = "jpeg"  # "jpeg" or "png"
    jpeg_quality: int = 95


@router.post("/batch/start")
async def start_batch_processing(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start a batch processing job with the specified criteria"""
    
    # Create session
    session_id = str(uuid.uuid4())
    session = {
        'id': session_id,
        'status': 'initializing',
        'created_at': datetime.now().isoformat(),
        'request': request.dict(),
        'photos': [],
        'processed_count': 0,
        'total_count': 0,
        'current_stage': None,
        'output_path': None,
        'error': None
    }
    
    batch_sessions[session_id] = session
    
    # Start processing in background
    background_tasks.add_task(process_batch, session_id)
    
    logger.info(f"Started batch processing session {session_id}")
    
    return {
        'session_id': session_id,
        'status': 'started',
        'message': 'Batch processing started. Use /batch/{session_id}/status to check progress.'
    }


async def process_batch(session_id: str):
    """Main batch processing function"""
    session = batch_sessions.get(session_id)
    if not session:
        return
    
    try:
        request = BatchProcessingRequest(**session['request'])
        
        # Stage 1: Find photos matching criteria
        session['status'] = 'finding_photos'
        session['current_stage'] = 'Finding photos'
        await notify_progress(session_id)
        
        photos = await find_matching_photos(
            min_aesthetic_score=request.min_aesthetic_score,
            min_technical_score=request.min_technical_score,
            max_photos=request.max_photos
        )
        
        if not photos:
            raise Exception("No photos found matching the criteria")
        
        session['photos'] = photos
        session['total_count'] = len(photos)
        logger.info(f"Found {len(photos)} photos for batch processing")
        
        # Create temp directory for processed images
        temp_dir = Path(f"/app/data/temp/batch_{session_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each photo through all stages
        for idx, photo in enumerate(photos):
            try:
                session['current_photo'] = photo['id']
                session['current_photo_index'] = idx + 1
                
                # Load original image
                photo_path = Path(photo['original_path'])
                if not photo_path.exists():
                    logger.warning(f"Photo file not found: {photo_path}")
                    continue
                
                current_image = load_image(str(photo_path))
                current_path = photo_path
                
                # Stage 2: Rotation
                if request.rotation_enabled:
                    session['current_stage'] = f'Rotating photo {idx + 1}/{len(photos)}'
                    await notify_progress(session_id)
                    
                    current_image, rotation_angle = await apply_rotation(
                        current_image,
                        current_path,
                        request.rotation_method,
                        request.rotation_vlm_model
                    )
                
                # Stage 3: Cropping
                if request.crop_enabled:
                    session['current_stage'] = f'Cropping photo {idx + 1}/{len(photos)}'
                    await notify_progress(session_id)
                    
                    current_image = await apply_crop(
                        current_image,
                        current_path,
                        request.crop_method,
                        request.crop_vlm_model,
                        request.crop_aspect_ratio
                    )
                
                # Stage 4: Enhancement
                if request.enhance_enabled:
                    session['current_stage'] = f'Enhancing photo {idx + 1}/{len(photos)}'
                    await notify_progress(session_id)
                    
                    current_image = intelligent_enhancer.enhance_image(
                        current_image,
                        request.enhance_strength
                    )
                
                # Save processed image
                output_filename = f"{photo['id']}_processed.{request.output_format}"
                output_path = temp_dir / output_filename
                
                if request.output_format == 'jpeg':
                    current_image.save(output_path, 'JPEG', quality=request.jpeg_quality, optimize=True)
                else:
                    current_image.save(output_path, 'PNG', optimize=True)
                
                session['processed_count'] += 1
                await notify_progress(session_id)
                
            except Exception as e:
                logger.error(f"Error processing photo {photo['id']}: {e}")
                continue
        
        # Stage 5: Create tar.gz
        session['status'] = 'creating_archive'
        session['current_stage'] = 'Creating archive'
        await notify_progress(session_id)
        
        archive_path = await create_archive(temp_dir, session_id)
        session['output_path'] = str(archive_path)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        session['status'] = 'completed'
        session['current_stage'] = 'Complete'
        await notify_progress(session_id)
        
        logger.info(f"Batch processing completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Batch processing failed for session {session_id}: {e}")
        session['status'] = 'failed'
        session['error'] = str(e)
        await notify_progress(session_id)


async def find_matching_photos(
    min_aesthetic_score: Optional[float] = None,
    min_technical_score: Optional[float] = None,
    max_photos: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Find photos matching the specified criteria"""
    
    # Build query
    query = """
        SELECT p.id, p.filename, p.original_path, 
               a.aesthetic_score, a.technical_score
        FROM photos p
        LEFT JOIN ai_analysis a ON p.id = a.photo_id
        WHERE p.status = 'completed'
        AND p.original_path IS NOT NULL
    """
    
    params = []
    
    if min_aesthetic_score is not None:
        query += " AND a.aesthetic_score >= ?"
        params.append(min_aesthetic_score)
    
    if min_technical_score is not None:
        query += " AND a.technical_score >= ?"
        params.append(min_technical_score)
    
    # Order by combined score
    query += " ORDER BY (COALESCE(a.aesthetic_score, 0) + COALESCE(a.technical_score, 0)) DESC"
    
    if max_photos:
        query += " LIMIT ?"
        params.append(max_photos)
    
    # Execute query
    async with photo_service.db.get_connection() as conn:
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        
        photos = []
        for row in rows:
            photos.append({
                'id': row[0],
                'filename': row[1],
                'original_path': row[2],
                'aesthetic_score': row[3],
                'technical_score': row[4]
            })
        
        return photos


async def apply_rotation(
    image: Image.Image,
    image_path: Path,
    method: str,
    vlm_model: Optional[str]
) -> tuple[Image.Image, float]:
    """Apply rotation to image using specified method"""
    
    if method == "cv":
        # Use CV-based rotation
        task = analyze_rotation_cv.delay(str(image_path))
    else:
        # Use VLM-based rotation
        task = analyze_rotation_onealign.delay(str(image_path))
    
    # Wait for result
    result = await wait_for_celery_task(task)
    
    if result and 'optimal_angle' in result:
        angle = result['optimal_angle']
        
        if abs(angle) > 0.1:  # Only rotate if needed
            # Apply rotation
            rotated = image.rotate(
                -angle,  # Negative because PIL rotates clockwise
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255)
            )
            
            # Crop to remove black borders
            # Calculate the largest inscribed rectangle
            import math
            w, h = image.size
            angle_rad = math.radians(abs(angle))
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            
            new_w = int(w * cos_a - h * sin_a)
            new_h = int(h * cos_a - w * sin_a)
            
            # Center crop
            rot_w, rot_h = rotated.size
            left = (rot_w - new_w) // 2
            top = (rot_h - new_h) // 2
            
            rotated = rotated.crop((left, top, left + new_w, top + new_h))
            
            return rotated, angle
    
    return image, 0.0


async def apply_crop(
    image: Image.Image,
    image_path: Path,
    method: str,
    vlm_model: Optional[str],
    aspect_ratio: str
) -> Image.Image:
    """Apply cropping to image using specified method"""
    
    if method == "cv" or aspect_ratio == "original":
        # Simple center crop to aspect ratio
        if aspect_ratio != "original":
            return apply_aspect_ratio_crop(image, aspect_ratio)
        return image
    else:
        # Use VLM-based cropping
        # Save temp image for VLM
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG', quality=95)
            temp_path = temp_file.name
        
        try:
            # Round 1: Get description and directions
            task1 = analyze_crop_vlm_round1.delay(temp_path, vlm_model)
            round1_result = await wait_for_celery_task(task1)
            
            if round1_result:
                # Round 2: Get exact coordinates
                task2 = analyze_crop_vlm_round2.delay(temp_path, round1_result, vlm_model)
                round2_result = await wait_for_celery_task(task2)
                
                if round2_result and 'crop_coordinates' in round2_result:
                    coords = round2_result['crop_coordinates']
                    
                    # Apply crop
                    x1 = coords['x1_px']
                    y1 = coords['y1_px']
                    x2 = coords['x2_px']
                    y2 = coords['y2_px']
                    
                    cropped = image.crop((x1, y1, x2, y2))
                    
                    # Apply aspect ratio if needed
                    if aspect_ratio != "original":
                        cropped = apply_aspect_ratio_crop(cropped, aspect_ratio)
                    
                    return cropped
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    return image


def apply_aspect_ratio_crop(image: Image.Image, aspect_ratio: str) -> Image.Image:
    """Apply center crop to achieve target aspect ratio"""
    
    # Parse aspect ratio
    if ':' in aspect_ratio:
        w_ratio, h_ratio = map(float, aspect_ratio.split(':'))
        target_ratio = w_ratio / h_ratio
    else:
        return image
    
    img_w, img_h = image.size
    img_ratio = img_w / img_h
    
    if abs(img_ratio - target_ratio) < 0.01:
        return image  # Already correct ratio
    
    if img_ratio > target_ratio:
        # Image is wider, crop width
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        return image.crop((left, 0, left + new_w, img_h))
    else:
        # Image is taller, crop height
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        return image.crop((0, top, img_w, top + new_h))


async def wait_for_celery_task(task, timeout: int = 300):
    """Wait for a Celery task to complete"""
    import time
    start_time = time.time()
    
    while not task.ready():
        if time.time() - start_time > timeout:
            task.revoke(terminate=True)
            return None
        await asyncio.sleep(1)
    
    if task.successful():
        return task.get()
    return None


async def create_archive(temp_dir: Path, session_id: str) -> Path:
    """Create tar.gz archive of processed images"""
    
    archive_name = f"batch_processed_{session_id}.tar.gz"
    archive_path = Path(f"/app/data/temp/{archive_name}")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                tar.add(file_path, arcname=file_path.name)
    
    return archive_path


async def notify_progress(session_id: str):
    """Send progress notification via WebSocket"""
    if not ws_manager:
        return
    
    session = batch_sessions.get(session_id)
    if not session:
        return
    
    progress_data = {
        'session_id': session_id,
        'status': session['status'],
        'current_stage': session.get('current_stage'),
        'processed_count': session.get('processed_count', 0),
        'total_count': session.get('total_count', 0),
        'progress': (session.get('processed_count', 0) / max(session.get('total_count', 1), 1)) * 100
    }
    
    await ws_manager.broadcast({
        'type': 'batch_processing_progress',
        'data': progress_data,
        'timestamp': datetime.now().isoformat()
    })


@router.get("/batch/{session_id}/status")
async def get_batch_status(session_id: str):
    """Get status of a batch processing session"""
    
    session = batch_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        'session_id': session_id,
        'status': session['status'],
        'current_stage': session.get('current_stage'),
        'processed_count': session.get('processed_count', 0),
        'total_count': session.get('total_count', 0),
        'progress': (session.get('processed_count', 0) / max(session.get('total_count', 1), 1)) * 100,
        'created_at': session['created_at'],
        'error': session.get('error')
    }


@router.get("/batch/{session_id}/download")
async def download_batch_results(session_id: str):
    """Download the processed images as tar.gz"""
    
    session = batch_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Batch processing not completed. Current status: {session['status']}"
        )
    
    if not session.get('output_path'):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    output_path = Path(session['output_path'])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file no longer exists")
    
    return FileResponse(
        path=str(output_path),
        media_type='application/gzip',
        filename=output_path.name
    )


@router.delete("/batch/{session_id}")
async def cleanup_batch_session(session_id: str):
    """Clean up a batch processing session and its files"""
    
    session = batch_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete output file if exists
    if session.get('output_path'):
        output_path = Path(session['output_path'])
        output_path.unlink(missing_ok=True)
    
    # Remove session
    del batch_sessions[session_id]
    
    return {'message': 'Session cleaned up successfully'}


@router.get("/batch/sessions")
async def list_batch_sessions():
    """List all batch processing sessions"""
    
    sessions_list = []
    for session_id, session in batch_sessions.items():
        sessions_list.append({
            'session_id': session_id,
            'status': session['status'],
            'created_at': session['created_at'],
            'processed_count': session.get('processed_count', 0),
            'total_count': session.get('total_count', 0)
        })
    
    return {
        'sessions': sorted(sessions_list, key=lambda x: x['created_at'], reverse=True),
        'total': len(sessions_list)
    }