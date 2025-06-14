"""
Recipe Builder Routes

Interactive workflow for creating recipes by manually adjusting settings
on sample photos and saving the results as reusable recipes.
"""

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
import json
import logging
import asyncio
from datetime import datetime
from PIL import Image, ImageDraw

from tasks.ai_tasks import analyze_rotation_onealign, analyze_rotation_cv, analyze_composition_vlm, generate_crop_bbox_vlm, analyze_crop_vlm_round1, analyze_crop_vlm_round2
from services.photo_service_v2 import photo_service
from services.crop_optimizer import CropOptimizer
from services.enhance_optimizer import EnhanceOptimizer
from services.websocket_manager import WebSocketManager
from services.intelligent_enhancer import IntelligentEnhancer
import requests
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for active recipe building sessions
recipe_sessions = {}

# Pydantic models for request validation
class CropApplyRequest(BaseModel):
    aspect_ratio: Optional[str] = None
    crop_box: Optional[Dict[str, float]] = None
    use_intelligent_crop: bool = False

# Initialize services
crop_optimizer = CropOptimizer()
enhance_optimizer = EnhanceOptimizer()
intelligent_enhancer = IntelligentEnhancer()
ws_manager = None  # Will be injected from main.py

def set_websocket_manager(manager: WebSocketManager):
    """Inject websocket manager from main.py"""
    global ws_manager
    ws_manager = manager

@router.get("/recipe-builder/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models that support vision"""
    try:
        ollama_url = os.getenv('OLLAMA_HOST', 'http://immich_ollama:11434')
        
        # Query Ollama for available models
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Ollama models: {response.status_code}")
            return {
                'models': [],
                'error': f'Failed to fetch models from Ollama: {response.status_code}'
            }
        
        data = response.json()
        all_models = data.get('models', [])
        
        # Filter for vision-capable models
        vision_models = []
        vision_keywords = ['vision', 'vl', 'llava', 'bakllava', 'clip']
        
        for model in all_models:
            model_name = model.get('name', '').lower()
            model_details = model.get('details', {})
            
            # Check if model name contains vision keywords
            is_vision = any(keyword in model_name for keyword in vision_keywords)
            
            # Also check model families known to support vision
            if 'qwen' in model_name and 'vl' in model_name:
                is_vision = True
            
            # Gemma3 is a vision model!
            if 'gemma3' in model_name:
                is_vision = True
            
            if is_vision:
                vision_models.append({
                    'name': model['name'],
                    'size': model.get('size', 0),
                    'size_gb': round(model.get('size', 0) / (1024**3), 2),
                    'modified': model.get('modified_at', ''),
                    'details': model_details
                })
        
        # Sort by name
        vision_models.sort(key=lambda x: x['name'])
        
        # Add recommended flag for known good models
        for model in vision_models:
            model_lower = model['name'].lower()
            if 'qwen2.5vl' in model_lower or 'qwen2.5-vl' in model_lower:
                if '7b' in model_lower:
                    model['recommended'] = True
                    model['description'] = 'Recommended: Best balance of quality and speed'
                else:
                    model['description'] = 'Qwen 2.5 Vision Language model'
            elif 'llava' in model_lower:
                if '7b' in model_lower:
                    model['description'] = 'Good alternative vision model'
                elif '13b' in model_lower:
                    model['description'] = 'Higher quality but slower'
                else:
                    model['description'] = 'LLaVA vision model'
            elif 'llama3.2-vision' in model_lower:
                model['description'] = 'Latest Llama vision model'
            elif 'gemma3' in model_lower:
                if '12b' in model_lower:
                    model['description'] = 'Google Gemma3 vision model - larger variant'
                elif '4b' in model_lower:
                    model['description'] = 'Google Gemma3 vision model - compact variant'
                else:
                    model['description'] = 'Google Gemma3 vision model'
        
        logger.info(f"Found {len(vision_models)} vision-capable models out of {len(all_models)} total")
        
        return {
            'models': vision_models,
            'total': len(vision_models),
            'ollama_host': ollama_url
        }
        
    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to Ollama")
        return {
            'models': [],
            'error': 'Timeout connecting to Ollama service'
        }
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return {
            'models': [],
            'error': str(e)
        }

@router.post("/recipe-builder/start")
async def start_recipe_builder(
    photo_ids: List[str] = Body(..., description="List of photo IDs to use for recipe building"),
    name: str = Body("Untitled Recipe", description="Name for the recipe being built"),
    description: str = Body("", description="Description of the recipe")
):
    """Start a new recipe building session with selected photos"""
    if len(photo_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 photos allowed for recipe building"
        )
    
    if len(photo_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one photo required for recipe building"
        )
    
    # Create session
    session_id = str(uuid.uuid4())
    session = {
        'id': session_id,
        'name': name,
        'description': description,
        'photo_ids': photo_ids,
        'current_photo_index': 0,
        'current_step': 'rotate',  # rotate -> crop -> enhance
        'steps_completed': [],
        'recipe_steps': [],
        'created_at': datetime.now().isoformat(),
        'status': 'active',
        'temp_files': {}  # Maps photo_id -> temp file path
    }
    
    recipe_sessions[session_id] = session
    
    logger.info(f"Started recipe builder session {session_id} with {len(photo_ids)} photos")
    
    # Notify via WebSocket
    if ws_manager:
        await ws_manager.broadcast({
            'type': 'recipe_builder_started',
            'session_id': session_id,
            'total_photos': len(photo_ids)
        })
    
    return {
        'session_id': session_id,
        'session': session
    }


async def generate_crop_preview(photo_path: Path, crop_coords: Dict[str, Any], 
                               session_id: str, photo_id: str) -> Optional[str]:
    """Generate a preview image showing the crop box overlay on the photo"""
    try:
        # Load the image
        img = Image.open(photo_path)
        
        # Create a copy for drawing
        preview = img.copy()
        draw = ImageDraw.Draw(preview)
        
        # Get pixel coordinates
        x1 = crop_coords.get('x1_px', 0)
        y1 = crop_coords.get('y1_px', 0)
        x2 = crop_coords.get('x2_px', img.width)
        y2 = crop_coords.get('y2_px', img.height)
        
        # Draw the crop box
        # Red box with some transparency
        box_color = (255, 0, 0)
        box_width = max(3, int(img.width * 0.005))  # Scale line width with image size
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
        # Draw corner markers for emphasis
        corner_length = max(20, int(img.width * 0.02))
        
        # Top-left corner
        draw.line([x1, y1, x1 + corner_length, y1], fill=box_color, width=box_width * 2)
        draw.line([x1, y1, x1, y1 + corner_length], fill=box_color, width=box_width * 2)
        
        # Top-right corner
        draw.line([x2 - corner_length, y1, x2, y1], fill=box_color, width=box_width * 2)
        draw.line([x2, y1, x2, y1 + corner_length], fill=box_color, width=box_width * 2)
        
        # Bottom-left corner
        draw.line([x1, y2, x1 + corner_length, y2], fill=box_color, width=box_width * 2)
        draw.line([x1, y2 - corner_length, x1, y2], fill=box_color, width=box_width * 2)
        
        # Bottom-right corner
        draw.line([x2 - corner_length, y2, x2, y2], fill=box_color, width=box_width * 2)
        draw.line([x2, y2 - corner_length, x2, y2], fill=box_color, width=box_width * 2)
        
        # Save preview to temp directory
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        preview_filename = f"crop_preview_{session_id}_{photo_id}.jpg"
        preview_path = temp_dir / preview_filename
        
        preview.save(preview_path, format='JPEG', quality=90)
        logger.info(f"Generated crop preview at {preview_path}")
        
        return str(preview_path)
        
    except Exception as e:
        logger.error(f"Failed to generate crop preview: {e}")
        return None

@router.get("/recipe-builder/{session_id}/current")
async def get_current_state(session_id: str):
    """Get current state of recipe building session"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Get temp file URL if available
    temp_image_url = None
    if current_photo_id in session.get('temp_files', {}):
        temp_path = session['temp_files'][current_photo_id]
        # Generate a URL for the temp file
        import os
        temp_filename = os.path.basename(temp_path)
        temp_image_url = f"/api/files/temp/{temp_filename}"
    
    return {
        'session_id': session_id,
        'current_photo_id': current_photo_id,
        'current_photo_index': session['current_photo_index'],
        'total_photos': len(session['photo_ids']),
        'current_step': session['current_step'],
        'steps_completed': session['steps_completed'],
        'recipe_steps': session['recipe_steps'],
        'temp_image_url': temp_image_url
    }

@router.post("/recipe-builder/{session_id}/rotate/analyze")
async def analyze_rotation(
    session_id: str,
    method: str = Body("cv", description="Analysis method: 'cv' (fast, traditional CV) or 'onealign' (slower, AI-based aesthetic scoring)"),
    min_angle: float = Body(-20.0, description="Minimum rotation angle (OneAlign only)"),
    max_angle: float = Body(20.0, description="Maximum rotation angle (OneAlign only)"),
    angle_step: float = Body(0.5, description="Step size for angle search (OneAlign only)"),
    cv_method: str = Body("auto", description="CV detection method: 'auto', 'horizon', 'lines', 'faces', 'exif' (CV only)"),
    recipe_params: Optional[Dict[str, Any]] = Body(None, description="Recipe parameters for CV configuration")
):
    """
    Analyze optimal rotation for current photo.
    
    Two methods available:
    - 'cv': Fast computer vision based detection (Hough transform, horizon detection, etc.)
    - 'onealign': Slower but more aesthetically-aware AI-based scoring of multiple angles
    """
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'rotate':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not rotate"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Find photo file - try multiple locations
    photo_path = None
    
    # List of possible locations to check
    possible_paths = [
        Path(f"/app/data/inbox/{current_photo_id}"),  # Direct UUID
        Path(f"/app/data/originals/{current_photo_id}"),
        Path(f"/app/data/processed/{current_photo_id}"),
    ]
    
    # Also check for files with UUID prefix in inbox
    inbox_dir = Path("/app/data/inbox")
    if inbox_dir.exists():
        for file_path in inbox_dir.glob(f"{current_photo_id}*"):
            possible_paths.append(file_path)
    
    # Find the first existing file
    for path in possible_paths:
        if path.exists() and path.is_file():
            photo_path = path
            break
    
    if not photo_path:
        raise HTTPException(
            status_code=404, 
            detail=f"Photo file not found for ID {current_photo_id}. Checked: {[str(p) for p in possible_paths[:3]]}"
        )
    
    logger.info(f"Analyzing rotation for photo {current_photo_id} at {photo_path}")
    
    try:
        # Start appropriate Celery task based on method
        if method == "cv":
            # Use fast CV-based rotation detection
            task = analyze_rotation_cv.delay(
                str(photo_path),
                method=cv_method,
                recipe_params=recipe_params
            )
        elif method == "onealign":
            # Use slower but aesthetically-aware OneAlign analysis
            task = analyze_rotation_onealign.delay(
                str(photo_path),
                min_angle=min_angle,
                max_angle=max_angle, 
                angle_step=angle_step
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid method '{method}'. Use 'cv' or 'onealign'"
            )
        
        # Store task ID in session for progress tracking
        if 'active_tasks' not in session:
            session['active_tasks'] = {}
        session['active_tasks'][current_photo_id] = {
            'task_id': task.id,
            'task_type': 'rotation_analysis',
            'started_at': datetime.now().isoformat()
        }
        
        # Poll task until completion
        import time
        max_wait_time = 300  # 5 minutes max
        poll_interval = 1    # Check every second
        start_time = time.time()
        
        while not task.ready():
            if time.time() - start_time > max_wait_time:
                task.revoke(terminate=True)
                
                # Send timeout notification via WebSocket
                if ws_manager:
                    timeout_event = {
                        "type": "rotation_analysis_failed",
                        "data": {
                            'session_id': session_id,
                            'photo_id': current_photo_id,
                            'error': 'Analysis timed out',
                            'progress': 0,
                            'message': 'Analysis timed out after 5 minutes'
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    await ws_manager.broadcast(timeout_event)
                
                raise HTTPException(status_code=408, detail="Analysis timed out")
            
            # Check for progress updates
            if task.state == 'PROGRESS':
                progress_info = task.info
                if ws_manager and progress_info:
                    progress_event = {
                        "type": "rotation_analysis_progress",
                        "data": {
                            'session_id': session_id,
                            'photo_id': current_photo_id,
                            'progress': progress_info.get('progress', 0) * 100,
                            'phase': progress_info.get('phase', 'processing'),
                            'current_angle': progress_info.get('current_angle'),
                            'current_score': progress_info.get('current_score'),
                            'best_angle': progress_info.get('best_angle'),
                            'best_score': progress_info.get('best_score'),
                            'message': progress_info.get('message', 'Processing...')
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    await ws_manager.broadcast(progress_event)
            
            await asyncio.sleep(poll_interval)
        
        # Get the result
        if task.successful():
            result = task.get()
            
            # Store analysis result in session
            if 'analysis_results' not in session:
                session['analysis_results'] = {}
            
            session['analysis_results'][current_photo_id] = {
                'rotation': result
            }
            
            # Clean up task info
            if current_photo_id in session.get('active_tasks', {}):
                del session['active_tasks'][current_photo_id]
            
            logger.info(f"Rotation analysis completed for {current_photo_id}: {result['optimal_angle']}° (score: {result['optimal_score']:.3f})")
            
            # Send completion notification via WebSocket
            if ws_manager:
                completion_data = {
                    'session_id': session_id,
                    'photo_id': current_photo_id,
                    'optimal_angle': result['optimal_angle'],
                    'optimal_score': result['optimal_score'],
                    'display_image_url': result.get('display_image_url'),
                    'display_image_path': result.get('display_image_path'),
                    'peak_angle': result.get('peak_angle'),
                    'peak_score': result.get('peak_score'),
                    'progress': 100,
                    'message': 'Analysis completed successfully'
                }
                # Use the WebSocket manager's event structure
                event = {
                    "type": "rotation_analysis_complete",
                    "data": completion_data,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"Broadcasting completion notification: {event}")
                await ws_manager.broadcast(event)
            else:
                logger.warning("WebSocket manager not available for completion notification")
            
            return {
                'optimal_angle': result['optimal_angle'],
                'optimal_score': result['optimal_score'],
                'all_scores': result['all_scores'],
                'search_parameters': result['search_parameters'],
                'display_image_url': result.get('display_image_url'),
                'display_image_path': result.get('display_image_path')
            }
        else:
            # Task failed
            error_info = task.info if task.failed() else "Unknown error"
            logger.error(f"Rotation analysis task failed: {error_info}")
            
            # Send failure notification via WebSocket
            if ws_manager:
                failure_event = {
                    "type": "rotation_analysis_failed",
                    "data": {
                        'session_id': session_id,
                        'photo_id': current_photo_id,
                        'error': str(error_info),
                        'progress': 0,
                        'message': f'Analysis failed: {error_info}'
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await ws_manager.broadcast(failure_event)
            
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_info}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rotation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipe-builder/{session_id}/rotate/apply")
async def apply_rotation(
    session_id: str,
    angle: float = Body(..., description="Rotation angle to apply"),
    auto_detect: bool = Body(False, description="Use auto-detected optimal angle")
):
    """Apply rotation to current photo and save settings"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'rotate':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not rotate"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # If auto_detect, use the analyzed optimal angle
    if auto_detect and 'analysis_results' in session:
        if current_photo_id in session['analysis_results']:
            angle = session['analysis_results'][current_photo_id]['rotation']['optimal_angle']
        else:
            raise HTTPException(
                status_code=400,
                detail="No rotation analysis available for auto-detect"
            )
    
    # Create rotation step
    rotation_step = {
        'operation': 'rotate',
        'params': {
            'angle': angle,
            'auto_detect': auto_detect,
            'resample': 'lanczos',
            'crop_to_fit': True
        },
        'photo_id': current_photo_id,
        'timestamp': datetime.now().isoformat()
    }
    
    # Find the source photo file
    photo_path = None
    possible_paths = [
        Path(f"/app/data/inbox/{current_photo_id}"),
        Path(f"/app/data/originals/{current_photo_id}"),
        Path(f"/app/data/processed/{current_photo_id}"),
    ]
    
    # Check for files with extensions in inbox
    inbox_dir = Path("/app/data/inbox")
    if inbox_dir.exists():
        for file_path in inbox_dir.glob(f"{current_photo_id}*"):
            possible_paths.append(file_path)
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            photo_path = path
            break
    
    if not photo_path:
        raise HTTPException(
            status_code=404, 
            detail=f"Photo file not found for ID {current_photo_id}"
        )
    
    # Apply rotation and save to temp
    try:
        from PIL import Image
        import tempfile
        import os
        
        # Load the original image (handle RAW files)
        from ai_components.shared.image_utils import load_image
        img = load_image(str(photo_path))
        
        # Apply rotation using the same approach as RotationOptimizer
        if abs(angle) < 0.01:  # Skip rotation for near-zero angles
            rotated_img = img.copy()
        else:
            import math
            
            # Rotate with expansion to capture full image
            rotated_img = img.rotate(
                -angle,  # Negative because PIL rotates clockwise
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255)  # White fill for any background
            )
            
            # Calculate the largest inscribed rectangle after rotation
            # This prevents any black borders from showing
            w, h = img.size
            angle_rad = math.radians(abs(angle))
            
            # Calculate the dimensions of the largest rectangle that fits
            if angle_rad == 0:
                new_w, new_h = w, h
            else:
                # Use the formula for the largest inscribed rectangle
                cos_a = abs(math.cos(angle_rad))
                sin_a = abs(math.sin(angle_rad))
                
                # Calculate maximum dimensions that avoid black borders
                if w <= h:
                    new_w = w * cos_a - h * sin_a
                    new_h = w * sin_a + h * cos_a
                else:
                    new_w = h / (sin_a + (h/w) * cos_a)
                    new_h = new_w * h / w
                
                new_w = int(new_w)
                new_h = int(new_h)
            
            # Center crop to the calculated dimensions
            rot_w, rot_h = rotated_img.size
            left = (rot_w - new_w) // 2
            top = (rot_h - new_h) // 2
            right = left + new_w
            bottom = top + new_h
            
            rotated_img = rotated_img.crop((left, top, right, bottom))
        
        # Create temp file path
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique temp filename
        temp_filename = f"recipe_{session_id}_{current_photo_id}_rotated.jpg"
        temp_path = temp_dir / temp_filename
        
        # Save rotated image as high quality JPEG
        rotated_img.save(temp_path, 'JPEG', quality=95, optimize=False)
        
        # Store temp file path in session
        session['temp_files'][current_photo_id] = str(temp_path)
        
        logger.info(f"Applied rotation {angle}° to photo {current_photo_id}, saved to {temp_path}")
        
    except Exception as e:
        logger.error(f"Failed to apply rotation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply rotation: {str(e)}"
        )
    
    # Add to recipe steps
    session['recipe_steps'].append(rotation_step)
    
    # Move to next step
    session['current_step'] = 'crop'
    session['steps_completed'].append('rotate')
    
    return {
        'success': True,
        'applied_angle': angle,
        'next_step': 'crop',
        'step_data': rotation_step
    }

@router.post("/recipe-builder/{session_id}/crop/settings")
async def update_crop_settings(
    session_id: str,
    aspect_ratio: str = Body(..., description="Target aspect ratio (e.g., '16:9', '4:3', '3:2', '1:1')")
):
    """Update crop settings for the session"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    
    # Validate aspect ratio
    valid_ratios = ['16:9', '4:3', '3:2', '2:3', '1:1', '9:16']
    if aspect_ratio not in valid_ratios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect ratio. Must be one of: {', '.join(valid_ratios)}"
        )
    
    # Store in session
    session['crop_aspect_ratio'] = aspect_ratio
    
    logger.info(f"Updated crop aspect ratio to {aspect_ratio} for session {session_id}")
    
    return {
        'success': True,
        'aspect_ratio': aspect_ratio,
        'message': f'Crop aspect ratio set to {aspect_ratio}'
    }

@router.post("/recipe-builder/{session_id}/crop/analyze")
async def analyze_crop_composition(
    session_id: str,
    request_body: Optional[Dict[str, str]] = Body(default={'ollama_model': 'qwen2.5-vl:7b'})
):
    """Two-stage VLM crop analysis: Round 1 gets description and directions"""
    try:
        # Log the incoming request
        logger.info(f"Crop analyze request - Session: {session_id}, Body: {request_body}")
        
        # Get model from request
        if request_body and isinstance(request_body, dict):
            ollama_model = request_body.get('ollama_model', '')
        else:
            ollama_model = ''
            
        if not ollama_model:
            error_msg = "No ollama_model specified in request"
            logger.error(f"Crop analyze error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Using Ollama model: {ollama_model}")
        
        if session_id not in recipe_sessions:
            error_msg = f"Session not found: {session_id}"
            logger.error(f"Crop analyze error: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        session = recipe_sessions[session_id]
        # Allow analyzing crop even if we're still on rotate step (user might skip rotation)
        if session['current_step'] not in ['rotate', 'crop']:
            error_msg = f"Current step is {session['current_step']}, not crop or rotate"
            logger.error(f"Crop analyze error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in crop analyze: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Check if we have a temp file from rotation step
    if current_photo_id in session.get('temp_files', {}):
        photo_path = Path(session['temp_files'][current_photo_id])
        if not photo_path.exists():
            logger.warning(f"Temp file not found: {photo_path}, falling back to original")
            photo_path = None
    else:
        photo_path = None
    
    # If no temp file, find original photo file
    if not photo_path:
        possible_paths = [
            Path(f"/app/data/inbox/{current_photo_id}"),
            Path(f"/app/data/originals/{current_photo_id}"),
            Path(f"/app/data/processed/{current_photo_id}"),
        ]
        
        inbox_dir = Path("/app/data/inbox")
        if inbox_dir.exists():
            for file_path in inbox_dir.glob(f"{current_photo_id}*"):
                possible_paths.append(file_path)
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                photo_path = path
                break
        
        if not photo_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Photo file not found for ID {current_photo_id}"
            )
    
    logger.info(f"Analyzing composition for photo {current_photo_id} at {photo_path}")
    
    try:
        # Start Celery task for Round 1 of crop analysis
        task = analyze_crop_vlm_round1.delay(str(photo_path), ollama_model)
        
        # Store task in session
        if 'active_tasks' not in session:
            session['active_tasks'] = {}
        session['active_tasks'][current_photo_id] = {
            'task_id': task.id,
            'task_type': 'crop_analysis_round1',
            'started_at': datetime.now().isoformat()
        }
        
        # Wait for completion (with timeout)
        import time
        max_wait_time = 180  # 3 minutes for VLM
        poll_interval = 2    # Check every 2 seconds
        start_time = time.time()
        
        while not task.ready():
            if time.time() - start_time > max_wait_time:
                task.revoke(terminate=True)
                raise HTTPException(status_code=408, detail="Composition analysis timed out")
            
            await asyncio.sleep(poll_interval)
        
        if task.successful():
            round1_result = task.get()
            
            # Store Round 1 result in session
            if 'analysis_results' not in session:
                session['analysis_results'] = {}
            
            if current_photo_id not in session['analysis_results']:
                session['analysis_results'][current_photo_id] = {}
            
            session['analysis_results'][current_photo_id]['crop_round1'] = round1_result
            
            logger.info(f"Crop analysis Round 1 completed for {current_photo_id}")
            
            # Now start Round 2 with the Round 1 results
            task2 = analyze_crop_vlm_round2.delay(
                str(photo_path),
                round1_result,
                ollama_model=ollama_model
            )
            
            # Store Round 2 task
            session['active_tasks'][current_photo_id] = {
                'task_id': task2.id,
                'task_type': 'crop_analysis_round2',
                'started_at': datetime.now().isoformat()
            }
            
            # Wait for Round 2 completion
            start_time = time.time()
            while not task2.ready():
                if time.time() - start_time > max_wait_time:
                    task2.revoke(terminate=True)
                    raise HTTPException(status_code=408, detail="Crop analysis Round 2 timed out")
                await asyncio.sleep(poll_interval)
            
            if task2.successful():
                round2_result = task2.get()
                
                # Store Round 2 result
                session['analysis_results'][current_photo_id]['crop_round2'] = round2_result
                
                # Clean up task info
                if current_photo_id in session.get('active_tasks', {}):
                    del session['active_tasks'][current_photo_id]
                
                logger.info(f"Crop analysis Round 2 completed for {current_photo_id}")
                
                # Generate preview with crop box overlay
                preview_path = await generate_crop_preview(
                    photo_path,
                    round2_result['crop_coordinates'],
                    session_id,
                    current_photo_id
                )
                
                # Send WebSocket notification for composition analysis complete
                if ws_manager:
                    completion_event = {
                        "type": "composition_analysis_complete",
                        "data": {
                            'session_id': session_id,
                            'photo_id': current_photo_id,
                            'analysis': {
                                'round1': {
                                    'photo_description': round1_result['photo_description'],
                                    'crop_directions': round1_result['crop_directions']
                                },
                                'round2': {
                                    'crop_coordinates': round2_result['crop_coordinates'],
                                    'aspect_ratio': round2_result['aspect_ratio'],
                                    'confidence': round2_result['confidence']
                                },
                                'preview_url': f"/api/files/temp/{Path(preview_path).name}" if preview_path else None,
                                'image_dimensions': round1_result['image_dimensions']
                            }
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    await ws_manager.broadcast(completion_event)
                
                return {
                    'status': 'completed',
                    'round1': {
                        'photo_description': round1_result['photo_description'],
                        'crop_directions': round1_result['crop_directions']
                    },
                    'round2': {
                        'crop_coordinates': round2_result['crop_coordinates'],
                        'aspect_ratio': round2_result['aspect_ratio'],
                        'confidence': round2_result['confidence']
                    },
                    'preview_url': f"/api/files/temp/{Path(preview_path).name}" if preview_path else None,
                    'image_dimensions': round1_result['image_dimensions'],
                    'model_used': round1_result['model_used']
                }
            else:
                error_info = task2.info if task2.failed() else "Unknown error"
                logger.error(f"Crop analysis Round 2 failed: {error_info}")
                raise HTTPException(status_code=500, detail=f"Round 2 failed: {error_info}")
        else:
            error_info = task.info if task.failed() else "Unknown error"
            logger.error(f"Composition analysis task failed: {error_info}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_info}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Composition analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipe-builder/{session_id}/crop/generate")
async def generate_intelligent_crop(
    session_id: str,
    user_intent: str = Body(..., description="User's description of what they want the crop to achieve"),
    target_aspect_ratio: str = Body("16:9", description="Target aspect ratio"),
    ollama_model: str = Body("qwen2.5-vl:7b", description="Ollama model to use")
):
    """Stage 2: Generate precise crop bounding box based on user intent"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'crop':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not crop"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Check if composition analysis exists
    if ('analysis_results' not in session or 
        current_photo_id not in session['analysis_results'] or
        'composition' not in session['analysis_results'][current_photo_id]):
        raise HTTPException(
            status_code=400,
            detail="Composition analysis required before generating crop. Call /crop/analyze first."
        )
    
    composition_analysis = session['analysis_results'][current_photo_id]['composition']['analysis']
    
    # Find photo file
    photo_path = None
    possible_paths = [
        Path(f"/app/data/inbox/{current_photo_id}"),
        Path(f"/app/data/originals/{current_photo_id}"),
        Path(f"/app/data/processed/{current_photo_id}"),
    ]
    
    inbox_dir = Path("/app/data/inbox")
    if inbox_dir.exists():
        for file_path in inbox_dir.glob(f"{current_photo_id}*"):
            possible_paths.append(file_path)
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            photo_path = path
            break
    
    if not photo_path:
        raise HTTPException(
            status_code=404, 
            detail=f"Photo file not found for ID {current_photo_id}"
        )
    
    logger.info(f"Generating intelligent crop for photo {current_photo_id}")
    
    try:
        # Start Celery task for crop generation
        task = generate_crop_bbox_vlm.delay(
            str(photo_path), 
            user_intent, 
            composition_analysis, 
            target_aspect_ratio, 
            ollama_model
        )
        
        # Store task in session
        if 'active_tasks' not in session:
            session['active_tasks'] = {}
        session['active_tasks'][current_photo_id] = {
            'task_id': task.id,
            'task_type': 'crop_generation',
            'started_at': datetime.now().isoformat()
        }
        
        # Wait for completion
        import time
        max_wait_time = 180  # 3 minutes for VLM
        poll_interval = 2
        start_time = time.time()
        
        while not task.ready():
            if time.time() - start_time > max_wait_time:
                task.revoke(terminate=True)
                raise HTTPException(status_code=408, detail="Crop generation timed out")
            
            await asyncio.sleep(poll_interval)
        
        if task.successful():
            result = task.get()
            
            # Store crop result in session
            session['analysis_results'][current_photo_id]['crop'] = result
            
            # Clean up task info
            if current_photo_id in session.get('active_tasks', {}):
                del session['active_tasks'][current_photo_id]
            
            logger.info(f"Intelligent crop generation completed for {current_photo_id}")
            
            return {
                'status': 'completed',
                'crop_result': result['crop_result'],
                'user_intent': result['user_intent'],
                'target_aspect_ratio': result['target_aspect_ratio'],
                'image_dimensions': result['image_dimensions'],
                'model_used': result['model_used'],
                'processing_time': result['processing_time']
            }
        else:
            error_info = task.info if task.failed() else "Unknown error"
            logger.error(f"Crop generation task failed: {error_info}")
            raise HTTPException(status_code=500, detail=f"Crop generation failed: {error_info}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crop generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipe-builder/{session_id}/crop/apply")
async def apply_crop(
    session_id: str,
    request: Dict[str, Any] = Body(...)
):
    """Apply crop to current photo"""
    # Debug logging to understand the 422 error
    logger.info(f"Crop apply request - Session: {session_id}")
    logger.info(f"Raw request data: {request}")
    
    # Parse the request manually
    aspect_ratio = request.get('aspect_ratio')
    crop_box = request.get('crop_box')
    use_intelligent_crop = request.get('use_intelligent_crop', False)
    
    logger.info(f"Parsed: aspect_ratio={aspect_ratio}, use_intelligent_crop={use_intelligent_crop}, crop_box={crop_box}")
    
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'crop':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not crop"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Determine crop mode and parameters
    if use_intelligent_crop:
        # Use VLM-generated crop
        if ('analysis_results' not in session or 
            current_photo_id not in session['analysis_results'] or
            'crop_round2' not in session['analysis_results'][current_photo_id]):
            raise HTTPException(
                status_code=400,
                detail="Intelligent crop analysis required. Call /crop/analyze first."
            )
        
        round2_result = session['analysis_results'][current_photo_id]['crop_round2']
        crop_box = {
            'x': round2_result['crop_coordinates']['x1_px'],
            'y': round2_result['crop_coordinates']['y1_px'],
            'width': round2_result['crop_coordinates']['x2_px'] - round2_result['crop_coordinates']['x1_px'],
            'height': round2_result['crop_coordinates']['y2_px'] - round2_result['crop_coordinates']['y1_px']
        }
        aspect_ratio = round2_result.get('aspect_ratio', 'auto')
        mode = 'intelligent_vlm'
        reasoning = 'VLM-generated crop'
    elif crop_box:
        # crop_box already set from request
        # aspect_ratio already set from request
        mode = 'manual'
        reasoning = 'User-specified crop box'
    else:
        # crop_box already None
        # aspect_ratio already set from request
        mode = 'aspect_ratio'
        reasoning = f'Aspect ratio crop: {aspect_ratio}'
    
    # Create crop step
    crop_step = {
        'operation': 'crop',
        'params': {
            'aspect_ratio': aspect_ratio,
            'crop_box': crop_box,
            'mode': mode,
            'reasoning': reasoning
        },
        'photo_id': current_photo_id,
        'timestamp': datetime.now().isoformat()
    }
    
    # Get the source image (either temp from rotation or original)
    if current_photo_id in session.get('temp_files', {}):
        source_path = Path(session['temp_files'][current_photo_id])
        if not source_path.exists():
            logger.warning(f"Temp file not found: {source_path}, falling back to original")
            source_path = None
    else:
        source_path = None
    
    # If no temp file, find original
    if not source_path:
        possible_paths = [
            Path(f"/app/data/inbox/{current_photo_id}"),
            Path(f"/app/data/originals/{current_photo_id}"),
            Path(f"/app/data/processed/{current_photo_id}"),
        ]
        
        inbox_dir = Path("/app/data/inbox")
        if inbox_dir.exists():
            for file_path in inbox_dir.glob(f"{current_photo_id}*"):
                possible_paths.append(file_path)
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                source_path = path
                break
        
        if not source_path:
            raise HTTPException(
                status_code=404, 
                detail=f"Photo file not found for ID {current_photo_id}"
            )
    
    # Apply the crop
    try:
        from PIL import Image
        
        # Load the image
        img = Image.open(source_path)
        
        # Apply crop based on mode
        if crop_box:
            # Use specified crop box
            x = int(crop_box.get('x', 0))
            y = int(crop_box.get('y', 0))
            width = int(crop_box.get('width', img.width))
            height = int(crop_box.get('height', img.height))
            
            cropped_img = img.crop((x, y, x + width, y + height))
        else:
            # Center crop to aspect ratio
            if aspect_ratio and aspect_ratio != 'auto':
                # Parse aspect ratio (e.g., "16:9" -> (16, 9))
                if ':' in aspect_ratio:
                    w_ratio, h_ratio = map(float, aspect_ratio.split(':'))
                    target_ratio = w_ratio / h_ratio
                else:
                    target_ratio = 1.0  # Default to square
                
                # Calculate crop dimensions
                img_ratio = img.width / img.height
                
                if img_ratio > target_ratio:
                    # Image is wider than target ratio
                    new_width = int(img.height * target_ratio)
                    new_height = img.height
                    x = (img.width - new_width) // 2
                    y = 0
                else:
                    # Image is taller than target ratio
                    new_width = img.width
                    new_height = int(img.width / target_ratio)
                    x = 0
                    y = (img.height - new_height) // 2
                
                cropped_img = img.crop((x, y, x + new_width, y + new_height))
            else:
                # No crop needed
                cropped_img = img
        
        # Save cropped image to temp
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique temp filename
        temp_filename = f"recipe_{session_id}_{current_photo_id}_cropped.jpg"
        temp_path = temp_dir / temp_filename
        
        # Save cropped image
        cropped_img.save(temp_path, 'JPEG', quality=95, optimize=False)
        
        # Update temp file path in session
        session['temp_files'][current_photo_id] = str(temp_path)
        
        logger.info(f"Applied {mode} crop to photo {current_photo_id}, saved to {temp_path}")
        
    except Exception as e:
        logger.error(f"Failed to apply crop: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply crop: {str(e)}"
        )
    
    # Add to recipe steps
    session['recipe_steps'].append(crop_step)
    
    # Move to next step
    session['current_step'] = 'enhance'
    session['steps_completed'].append('crop')
    
    return {
        'success': True,
        'next_step': 'enhance',
        'step_data': crop_step,
        'applied_crop': crop_box
    }

@router.post("/recipe-builder/{session_id}/enhance/preview")
async def preview_enhancement(
    session_id: str,
    request: Dict[str, Any] = Body(...)
):
    """Generate preview of enhancement with specified strength"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'enhance':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not enhance"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Get strength from request
    strength = request.get('strength', 1.0)
    if not isinstance(strength, (int, float)):
        raise HTTPException(status_code=400, detail="Strength must be a number")
    if strength < 0 or strength > 2:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 2")
    
    # Get the temp file from previous steps
    if current_photo_id not in session.get('temp_files', {}):
        logger.error(f"No temp file found for photo {current_photo_id}. Session temp_files: {session.get('temp_files', {})}")
        raise HTTPException(
            status_code=400,
            detail="No processed image found. Complete rotation and crop first."
        )
    
    temp_path = session['temp_files'][current_photo_id]
    logger.info(f"Enhancement preview using temp file: {temp_path} for photo {current_photo_id}")
    
    try:
        # Generate enhancement preview
        original, enhanced = intelligent_enhancer.get_enhancement_preview(
            temp_path, 
            strength
        )
        
        # Save preview to temp
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        preview_filename = f"enhance_preview_{session_id}_{current_photo_id}_{int(strength*100)}.jpg"
        preview_path = temp_dir / preview_filename
        
        enhanced.save(preview_path, 'JPEG', quality=95, optimize=False)
        
        logger.info(f"Generated enhancement preview for {current_photo_id} with strength {strength}")
        
        # Ensure temp file is accessible
        if not Path(temp_path).exists():
            logger.error(f"Temp file doesn't exist at path: {temp_path}")
            raise HTTPException(status_code=500, detail="Temp file not found")
        
        return {
            'success': True,
            'strength': strength,
            'preview_url': f"/api/files/temp/{preview_filename}",
            'original_url': f"/api/files/temp/{os.path.basename(temp_path)}"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate enhancement preview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )

@router.post("/recipe-builder/{session_id}/enhance/apply")
async def apply_enhancement(
    session_id: str,
    request: Dict[str, Any] = Body(...)
):
    """Apply intelligent enhancement to current photo"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'enhance':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not enhance"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Get strength from request
    strength = request.get('strength', 1.0)
    if not isinstance(strength, (int, float)):
        raise HTTPException(status_code=400, detail="Strength must be a number")
    if strength < 0 or strength > 2:
        raise HTTPException(status_code=400, detail="Strength must be between 0 and 2")
    
    # Get the temp file from previous steps
    if current_photo_id not in session.get('temp_files', {}):
        raise HTTPException(
            status_code=400,
            detail="No processed image found. Complete rotation and crop first."
        )
    
    source_path = Path(session['temp_files'][current_photo_id])
    
    try:
        # Apply enhancement with modern enhancer
        original = Image.open(source_path)
        if original.mode != 'RGB':
            original = original.convert('RGB')
        
        enhanced = intelligent_enhancer.enhance_image(
            original,
            strength
        )
        
        # Save enhanced image to temp
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        temp_filename = f"recipe_{session_id}_{current_photo_id}_enhanced.jpg"
        temp_path = temp_dir / temp_filename
        
        enhanced.save(temp_path, 'JPEG', quality=95, optimize=False)
        
        # Update temp file path
        session['temp_files'][current_photo_id] = str(temp_path)
        
        logger.info(f"Applied intelligent enhancement to photo {current_photo_id} with strength {strength}")
        
    except Exception as e:
        logger.error(f"Failed to apply enhancement: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply enhancement: {str(e)}"
        )
    
    # Create enhancement step
    enhance_step = {
        'operation': 'enhance',
        'params': {
            'mode': 'intelligent',
            'strength': strength
        },
        'photo_id': current_photo_id,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add to recipe steps
    session['recipe_steps'].append(enhance_step)
    
    # Move to next photo or complete
    session['steps_completed'].append('enhance')
    
    if session['current_photo_index'] < len(session['photo_ids']) - 1:
        # Move to next photo
        session['current_photo_index'] += 1
        session['current_step'] = 'rotate'
        session['steps_completed'] = []
        
        return {
            'success': True,
            'next_action': 'next_photo',
            'next_photo_index': session['current_photo_index'],
            'step_data': enhance_step
        }
    else:
        # All photos processed
        session['status'] = 'complete'
        
        return {
            'success': True,
            'next_action': 'complete',
            'step_data': enhance_step,
            'total_steps': len(session['recipe_steps'])
        }

@router.post("/recipe-builder/{session_id}/save")
async def save_recipe(
    session_id: str,
    body: Optional[Dict[str, Any]] = Body(None)
):
    """Save the recipe created from the building session"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    
    # Extract finalize from body, default to True
    finalize = True
    if body and isinstance(body, dict):
        finalize = body.get('finalize', True)
    
    if not session['recipe_steps']:
        raise HTTPException(
            status_code=400,
            detail="No steps recorded in this session"
        )
    
    # Aggregate steps by operation type
    # Group similar operations and extract common parameters
    rotation_params = []
    crop_params = []
    enhance_params = []
    
    for step in session['recipe_steps']:
        if step['operation'] == 'rotate':
            rotation_params.append(step['params'])
        elif step['operation'] == 'crop':
            crop_params.append(step['params'])
        elif step['operation'] == 'enhance':
            enhance_params.append(step['params'])
    
    # Create aggregated recipe steps
    recipe_operations = []
    
    # Rotation step - average angles or use auto-detect
    if rotation_params:
        auto_detect_votes = sum(1 for p in rotation_params if p.get('auto_detect'))
        if auto_detect_votes > len(rotation_params) / 2:
            # Majority voted for auto-detect
            recipe_operations.append({
                'type': 'rotate',
                'params': {
                    'mode': 'auto',
                    'min_angle': -20,
                    'max_angle': 20,
                    'angle_step': 0.5
                }
            })
        else:
            # Use average angle
            avg_angle = sum(p['angle'] for p in rotation_params) / len(rotation_params)
            recipe_operations.append({
                'type': 'rotate',
                'params': {
                    'mode': 'manual',
                    'angle': round(avg_angle, 1)
                }
            })
    
    # Crop step - use most common aspect ratio or session setting
    if crop_params:
        # First check if we have a session-level aspect ratio setting
        session_aspect_ratio = session.get('crop_aspect_ratio')
        
        if session_aspect_ratio:
            # Use the session-level setting
            recipe_operations.append({
                'type': 'crop',
                'params': {
                    'aspect_ratio': session_aspect_ratio,
                    'mode': 'intelligent'  # Since they used the recipe builder
                }
            })
        else:
            # Fall back to most common from individual crops
            aspect_ratios = [p.get('aspect_ratio') for p in crop_params if p.get('aspect_ratio')]
            if aspect_ratios:
                # Find most common
                most_common = max(set(aspect_ratios), key=aspect_ratios.count)
                recipe_operations.append({
                    'type': 'crop',
                    'params': {
                        'aspect_ratio': most_common,
                        'mode': 'intelligent'
                    }
                })
    
    # Enhancement step - handle intelligent mode
    if enhance_params:
        # Check if all enhancements use intelligent mode
        intelligent_count = sum(1 for p in enhance_params if p.get('mode') == 'intelligent')
        
        if intelligent_count == len(enhance_params):
            # All use intelligent mode - average the strength
            avg_strength = sum(p.get('strength', 1.0) for p in enhance_params) / len(enhance_params)
            recipe_operations.append({
                'type': 'enhance',
                'params': {
                    'mode': 'intelligent',
                    'strength': round(avg_strength, 2)
                }
            })
        elif intelligent_count > 0:
            # Mixed modes - default to intelligent with average strength
            avg_strength = sum(p.get('strength', 1.0) for p in enhance_params if p.get('mode') == 'intelligent') / intelligent_count
            recipe_operations.append({
                'type': 'enhance',
                'params': {
                    'mode': 'intelligent',
                    'strength': round(avg_strength, 2)
                }
            })
        else:
            # Old-style parameters (fallback for compatibility)
            avg_params = {
                'brightness': sum(p.get('brightness', 0) for p in enhance_params) / len(enhance_params),
                'contrast': sum(p.get('contrast', 0) for p in enhance_params) / len(enhance_params),
                'saturation': sum(p.get('saturation', 0) for p in enhance_params) / len(enhance_params),
                'sharpness': sum(p.get('sharpness', 0) for p in enhance_params) / len(enhance_params),
                'denoise': sum(p.get('denoise', 0) for p in enhance_params) / len(enhance_params)
            }
            
            # Round to reasonable precision
            avg_params = {k: round(v, 1) for k, v in avg_params.items()}
            
            recipe_operations.append({
                'type': 'enhance',
                'params': avg_params
            })
    
    # Create final recipe
    recipe_id = str(uuid.uuid4())
    recipe = {
        'id': recipe_id,
        'name': session['name'],
        'description': session['description'],
        'version': '2.0',
        'created_at': datetime.now().isoformat(),
        'created_from_session': session_id,
        'sample_photos': session['photo_ids'],
        'operations': recipe_operations,
        'metadata': {
            'total_samples': len(session['photo_ids']),
            'total_steps': len(session['recipe_steps'])
        }
    }
    
    if finalize:
        # Save to recipe storage using the service
        from services.recipe_service_v2 import recipe_service
        
        try:
            # Create recipe through the service (which uses SQLite)
            saved_recipe = await recipe_service.create_recipe(
                name=session['name'],
                description=session['description'],
                operations=recipe_operations,
                style_preset='natural',  # Default style
                processing_config={
                    'version': '2.0',
                    'created_from_session': session_id,
                    'sample_photos': session['photo_ids']
                }
            )
            
            # Update recipe_id with the actual saved ID
            recipe_id = saved_recipe['id']
            recipe['id'] = recipe_id
            
            logger.info(f"Saved recipe {recipe_id} from session {session_id} to database")
            
            # Clean up session
            session['status'] = 'saved'
            session['recipe_id'] = recipe_id
            
        except Exception as e:
            logger.error(f"Failed to save recipe to database: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save recipe: {str(e)}"
            )
    
    return {
        'recipe_id': recipe_id,
        'recipe': recipe,
        'finalized': finalize
    }

@router.get("/recipe-builder/sessions")
async def list_sessions(
    active_only: bool = Query(True, description="Only show active sessions")
):
    """List all recipe building sessions"""
    sessions = []
    
    for session_id, session in recipe_sessions.items():
        if active_only and session['status'] != 'active':
            continue
        
        sessions.append({
            'id': session_id,
            'name': session['name'],
            'status': session['status'],
            'created_at': session['created_at'],
            'photos_count': len(session['photo_ids']),
            'current_photo': session['current_photo_index'] + 1,
            'current_step': session['current_step']
        })
    
    return {
        'sessions': sessions,
        'total': len(sessions)
    }

@router.delete("/recipe-builder/{session_id}")
async def cancel_session(session_id: str):
    """Cancel and delete a recipe building session"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    session['status'] = 'cancelled'
    
    # Clean up any temporary files
    # TODO: Clean up temporary processed images
    
    del recipe_sessions[session_id]
    
    logger.info(f"Cancelled recipe builder session {session_id}")
    
    return {
        'success': True,
        'message': f"Session {session_id} cancelled"
    }

@router.get("/recipe-builder/debug/available-photos")
async def debug_available_photos():
    """Debug endpoint to list available photos for recipe building"""
    inbox_dir = Path("/app/data/inbox")
    
    if not inbox_dir.exists():
        return {"error": "Inbox directory not found", "photos": []}
    
    photos = []
    for file_path in inbox_dir.glob("*"):
        if file_path.is_file():
            # Extract UUID from filename
            filename = file_path.name
            if "_" in filename:
                uuid_part = filename.split("_")[0]
                photos.append({
                    "id": uuid_part,
                    "filename": filename,
                    "full_path": str(file_path),
                    "size": file_path.stat().st_size
                })
    
    return {
        "total": len(photos),
        "photos": sorted(photos, key=lambda x: x["filename"])[:20]  # Limit to first 20
    }


@router.post("/recipe-builder/{session_id}/enhance/preview-custom")
async def preview_custom_enhancement(
    session_id: str,
    request: Dict[str, Any] = Body(...)
):
    """Generate preview of enhancement with custom individual settings"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session['current_step'] != 'enhance':
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not enhance"
        )
    
    current_photo_id = session['photo_ids'][session['current_photo_index']]
    
    # Get the temp file from previous steps
    if current_photo_id not in session.get('temp_files', {}):
        raise HTTPException(
            status_code=400,
            detail="No processed image found. Complete rotation and crop first."
        )
    
    temp_path = session['temp_files'][current_photo_id]
    
    try:
        # Extract custom settings from request
        custom_settings = EnhancementSettings(
            white_balance=request.get('white_balance', True),
            white_balance_strength=request.get('white_balance_strength', 1.0),
            exposure=request.get('exposure', True),
            exposure_strength=request.get('exposure_strength', 1.0),
            contrast=request.get('contrast', True),
            contrast_strength=request.get('contrast_strength', 1.0),
            vibrance=request.get('vibrance', True),
            vibrance_strength=request.get('vibrance_strength', 1.0),
            shadow_highlight=request.get('shadow_highlight', True),
            shadow_highlight_strength=request.get('shadow_highlight_strength', 1.0),
            overall_strength=request.get('overall_strength', 1.0)
        )
        
        # Generate enhancement preview with custom settings
        original, enhanced = intelligent_enhancer.get_enhancement_preview(
            temp_path, 
            EnhancementMode.CUSTOM,
            custom_settings
        )
        
        # Save preview to temp
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        preview_filename = f"enhance_preview_custom_{session_id}_{current_photo_id}.jpg"
        preview_path = temp_dir / preview_filename
        
        enhanced.save(preview_path, 'JPEG', quality=95, optimize=False)
        
        logger.info(f"Generated custom enhancement preview for {current_photo_id}")
        
        return {
            'success': True,
            'settings': custom_settings.__dict__,
            'preview_url': f"/api/files/temp/{preview_filename}",
            'original_url': f"/api/files/temp/{os.path.basename(temp_path)}"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate custom enhancement preview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate preview: {str(e)}"
        )


@router.post("/recipe-builder/{session_id}/enhance/apply-custom")
async def apply_custom_enhancement(
    session_id: str,
    request: Dict[str, Any] = Body(...)
):
    """Apply custom enhancement with individual settings to current photo"""
    if session_id not in recipe_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = recipe_sessions[session_id]
    if session["current_step"] != "enhance":
        raise HTTPException(
            status_code=400,
            detail=f"Current step is {session['current_step']}, not enhance"
        )
    
    current_photo_id = session["photo_ids"][session["current_photo_index"]]
    
    # Get the temp file from previous steps
    if current_photo_id not in session.get("temp_files", {}):
        raise HTTPException(
            status_code=400,
            detail="No processed image found. Complete rotation and crop first."
        )
    
    source_path = Path(session["temp_files"][current_photo_id])
    
    try:
        # Extract custom settings from request
        custom_settings = EnhancementSettings(
            white_balance=request.get("white_balance", True),
            white_balance_strength=request.get("white_balance_strength", 1.0),
            exposure=request.get("exposure", True),
            exposure_strength=request.get("exposure_strength", 1.0),
            contrast=request.get("contrast", True),
            contrast_strength=request.get("contrast_strength", 1.0),
            vibrance=request.get("vibrance", True),
            vibrance_strength=request.get("vibrance_strength", 1.0),
            shadow_highlight=request.get("shadow_highlight", True),
            shadow_highlight_strength=request.get("shadow_highlight_strength", 1.0),
            overall_strength=request.get("overall_strength", 1.0)
        )
        
        # Apply custom enhancement
        original = Image.open(source_path)
        if original.mode != "RGB":
            original = original.convert("RGB")
        
        enhanced = intelligent_enhancer.enhance_image(
            original,
            EnhancementMode.CUSTOM,
            custom_settings
        )
        
        # Save enhanced image to temp
        temp_dir = Path("/app/data/temp")
        temp_dir.mkdir(exist_ok=True)
        
        temp_filename = f"recipe_{session_id}_{current_photo_id}_enhanced.jpg"
        temp_path = temp_dir / temp_filename
        
        enhanced.save(temp_path, "JPEG", quality=95, optimize=False)
        
        # Update temp file path
        session["temp_files"][current_photo_id] = str(temp_path)
        
        logger.info(f"Applied custom enhancement to photo {current_photo_id}")
        
    except Exception as e:
        logger.error(f"Failed to apply custom enhancement: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply enhancement: {str(e)}"
        )
    
    # Create enhancement step with custom settings
    enhance_step = {
        "operation": "enhance",
        "params": {
            "mode": "custom",
            "settings": custom_settings.__dict__
        },
        "photo_id": current_photo_id,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to recipe steps
    session["recipe_steps"].append(enhance_step)
    
    # Move to next photo or complete
    session["steps_completed"].append("enhance")
    
    if session["current_photo_index"] < len(session["photo_ids"]) - 1:
        # More photos to process
        session["current_photo_index"] += 1
        session["current_step"] = "rotate"
        session["steps_completed"] = []
        
        return {
            "success": True,
            "next_action": "next_photo",
            "next_photo_index": session["current_photo_index"],
            "total_photos": len(session["photo_ids"])
        }
    else:
        # All photos processed
        session["current_step"] = "complete"
        
        return {
            "success": True,
            "next_action": "complete",
            "total_photos": len(session["photo_ids"])
        }
