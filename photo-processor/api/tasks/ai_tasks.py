"""
AI Analysis Celery tasks for background processing

This module provides scalable AI analysis capabilities:
- OneAlign (Q-Align) image quality and aesthetics assessment
- Object detection
- Scene analysis
- Image classification
- Technical quality assessment
- Content analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from celery import current_task
from celery_app import celery_app
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import numpy as np
from PIL import Image
import io
import base64

# AI models will be imported lazily inside functions to avoid loading in main API process

# Import services
from services.photo_service_sqlite import sqlite_photo_service as photo_service
from services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Global AI model instances (loaded once per worker)
_ai_models = {}
_ws_manager = None

def cleanup_gpu_memory():
    """Clean up GPU memory and unload all AI models to make space for Ollama"""
    global _ai_models
    
    logger.info("Starting GPU memory cleanup...")
    
    try:
        import torch
        if torch.cuda.is_available():
            # Unload all AI models using their unload method if available
            for model_key in list(_ai_models.keys()):
                logger.info(f"Unloading model: {model_key}")
                model = _ai_models.get(model_key)
                
                # Try to use the model's unload method if available
                if model and hasattr(model, 'unload'):
                    try:
                        model.unload()
                        logger.info(f"Successfully unloaded {model_key} using unload method")
                    except Exception as e:
                        logger.error(f"Error unloading {model_key}: {e}")
                
                # Remove from dictionary
                del _ai_models[model_key]
            
            _ai_models.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache multiple times to ensure memory is freed
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Get memory info
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            logger.info(f"GPU memory cleanup completed:")
            logger.info(f"  Allocated: {memory_allocated:.2f} GB")
            logger.info(f"  Reserved: {memory_reserved:.2f} GB")
            
            return {
                'success': True,
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved
            }
        else:
            logger.info("No GPU available, skipping GPU cleanup")
            return {'success': True, 'message': 'No GPU available'}
            
    except Exception as e:
        logger.error(f"Error during GPU cleanup: {e}")
        return {'success': False, 'error': str(e)}

def get_ai_model(model_type: str, **kwargs):
    """Get or create AI model instance (singleton per worker)"""
    global _ai_models
    
    model_key = f"{model_type}_{hash(frozenset(kwargs.items()))}"
    
    if model_key not in _ai_models:
        logger.info(f"Loading {model_type} model...")
        
        # Check if GPU is available, otherwise use CPU
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            logger.warning("GPU not available, using CPU for AI model. Performance will be slower.")
        else:
            # Clear GPU memory before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"Cleared GPU memory before loading {model_type}")
        
        if model_type == 'onealign':
            # Import OneAlign only when needed (lazy import)
            from ai_components.onealign.onealign_model import OneAlign
            _ai_models[model_key] = OneAlign(
                device=device, 
                logger=logger,
                cache_dir=kwargs.get('cache_dir', '/app/model_cache'),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown AI model type: {model_type}")
            
        logger.info(f"{model_type} model loaded successfully on {device}")
    
    return _ai_models[model_key]

def get_websocket_manager():
    """Get WebSocket manager instance"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager

# ============================================================================
# OneAlign Analysis Tasks
# ============================================================================

@celery_app.task(name='tasks.ai_tasks.cleanup_gpu_memory')
def cleanup_gpu_memory_task():
    """
    Clean up GPU memory by unloading all AI models
    This is useful to free GPU memory for Ollama or other processes
    
    Returns:
        Dict with cleanup status and memory info
    """
    logger.info("GPU memory cleanup task started")
    result = cleanup_gpu_memory()
    
    # Send websocket notification
    try:
        from services.websocket_manager import WebSocketManager
        ws_manager = WebSocketManager()
        asyncio.create_task(
            ws_manager.broadcast({
                'type': 'gpu_cleanup',
                'status': 'completed',
                'result': result
            })
        )
    except Exception as e:
        logger.error(f"Failed to send websocket notification: {e}")
    
    return result

# ============================================================================

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_photo_onealign')
def analyze_photo_onealign(self, photo_id: str, photo_path: str):
    """
    Analyze photo with OneAlign for both quality and aesthetics assessment
    
    Args:
        photo_id: Unique photo identifier
        photo_path: Path to the photo file
    
    Returns:
        Dict containing OneAlign analysis results with both quality and aesthetics scores
    """
    logger.info(f"Starting OneAlign analysis for photo {photo_id}")
    
    try:
        # Update photo status to processing
        asyncio.run(photo_service.update_photo_status(photo_id, 'processing', 
                                                     processing_message='Running OneAlign quality and aesthetics analysis...'))
        
        # Skip WebSocket notification in Celery context
        
        # Load OneAlign model
        onealign_model = get_ai_model('onealign')
        
        # Perform quality and aesthetics analysis
        logger.info(f"Running OneAlign quality and aesthetics analysis on {photo_path}")
        onealign_results = onealign_model.assess_quality_and_aesthetics(photo_path)
        
        # Extract key metrics
        quality_score = onealign_results['quality_score']
        quality_raw = onealign_results['quality_raw']
        quality_level = onealign_results['quality_level']
        quality_confidence = onealign_results['quality_confidence']
        
        aesthetics_score = onealign_results['aesthetics_score']
        aesthetics_raw = onealign_results['aesthetics_raw']
        aesthetics_level = onealign_results['aesthetics_level']
        aesthetics_confidence = onealign_results['aesthetics_confidence']
        
        combined_score = onealign_results['combined_score']
        
        # Prepare AI analysis data for storage
        # Map to database field names:
        # technical_score = quality (how well made the photo is)
        # aesthetic_score = aesthetics (how beautiful/pleasing it is)
        ai_analysis = {
            'status': 'completed',
            'analysis_type': 'onealign',
            'technical_score': quality_score,  # Technical quality score
            'aesthetic_score': aesthetics_score,  # Aesthetic appeal score
            'quality_level': quality_level,
            'confidence': (quality_confidence + aesthetics_confidence) / 2,  # Average confidence
            'combined_score': combined_score,
            'completed_at': datetime.now().isoformat(),
            'onealign_results': onealign_results,
            'model_info': onealign_results.get('model_info', {})
        }
        
        # Update photo with NIMA results
        asyncio.run(photo_service.update_photo_ai_analysis(photo_id, ai_analysis))
        
        # Mark photo as completed
        asyncio.run(photo_service.update_photo_status(photo_id, 'completed', 
                                                     processing_message='OneAlign analysis completed'))
        
        # Skip WebSocket notification in Celery context (runs synchronously)
        # The frontend will get updates through polling
        
        logger.info(f"OneAlign analysis completed for photo {photo_id}: quality={quality_score:.2f}, aesthetics={aesthetics_score:.2f}")
        
        return {
            'photo_id': photo_id,
            'status': 'completed',
            'technical_score': quality_score,  # Map to correct field name
            'aesthetic_score': aesthetics_score,  # Map to correct field name
            'combined_score': combined_score,
            'quality_level': quality_level,
            'aesthetics_level': aesthetics_level,
            'processing_time': onealign_results['inference_time'],
            'full_results': ai_analysis
        }
        
    except Exception as e:
        error_msg = f"OneAlign analysis failed: {str(e)}"
        logger.error(f"Error in OneAlign analysis for photo {photo_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update photo status to failed
        try:
            asyncio.run(photo_service.update_photo_status(photo_id, 'failed', 
                                                         processing_message=error_msg))
            
            # Update AI analysis with error
            ai_analysis = {
                'status': 'failed',
                'error': error_msg,
                'failed_at': datetime.now().isoformat()
            }
            asyncio.run(photo_service.update_photo_ai_analysis(photo_id, ai_analysis))
            
        except Exception as db_error:
            logger.error(f"Failed to update photo status in database: {db_error}")
        
        # Notify via WebSocket
        try:
            ws_manager = get_websocket_manager()
            asyncio.create_task(ws_manager.notify_photo_status_changed(
                photo_id, 'failed', error_msg
            ))
        except Exception as ws_error:
            logger.warning(f"Failed to send WebSocket notification: {ws_error}")
        
        # Re-raise the exception so Celery marks the task as failed
        raise

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_batch_onealign')
def analyze_batch_onealign(self, photo_batch: list):
    """
    Analyze multiple photos with OneAlign in batch
    
    Args:
        photo_batch: List of dicts with photo_id and photo_path
    
    Returns:
        Dict containing batch analysis results
    """
    logger.info(f"Starting OneAlign batch analysis for {len(photo_batch)} photos")
    
    results = []
    successful = 0
    failed = 0
    
    for photo_info in photo_batch:
        photo_id = photo_info['photo_id']
        photo_path = photo_info['photo_path']
        
        try:
            # Process individual photo
            result = analyze_photo_onealign.apply(args=[photo_id, photo_path])
            results.append(result.get())
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process photo {photo_id} in batch: {e}")
            results.append({
                'photo_id': photo_id,
                'status': 'failed',
                'error': str(e)
            })
            failed += 1
    
    logger.info(f"Batch OneAlign analysis completed: {successful} successful, {failed} failed")
    
    # Clean up GPU memory after batch processing
    if len(photo_batch) > 5:  # Only cleanup for larger batches
        logger.info("Cleaning up GPU memory after batch processing")
        cleanup_result = cleanup_gpu_memory()
        logger.info(f"GPU cleanup result: {cleanup_result}")
    
    return {
        'batch_id': self.request.id,
        'total_photos': len(photo_batch),
        'successful': successful,
        'failed': failed,
        'results': results
    }

@celery_app.task(bind=True, name='tasks.ai_tasks.reanalyze_photo_onealign')
def reanalyze_photo_onealign(self, photo_id: str, force: bool = False):
    """
    Re-analyze a photo with OneAlign (for reprocessing or model updates)
    
    Args:
        photo_id: Photo to reanalyze
        force: Force reanalysis even if already analyzed
    
    Returns:
        OneAlign analysis results
    """
    logger.info(f"Re-analyzing photo {photo_id} with OneAlign (force={force})")
    
    try:
        # Get photo info
        photo = asyncio.run(photo_service.get_photo(photo_id))
        if not photo:
            raise ValueError(f"Photo {photo_id} not found")
        
        # Check if already analyzed and not forcing
        if not force and photo.get('ai_analysis', {}).get('status') == 'completed':
            logger.info(f"Photo {photo_id} already analyzed, skipping (use force=True to override)")
            return {
                'photo_id': photo_id,
                'status': 'skipped',
                'reason': 'already_analyzed'
            }
        
        # Get photo path
        photo_path = photo.original_path if hasattr(photo, 'original_path') else photo.get('original_path')
        if not photo_path or not Path(photo_path).exists():
            raise ValueError(f"Photo file not found: {photo_path}")
        
        # Run analysis
        return analyze_photo_onealign.apply(args=[photo_id, photo_path]).get()
        
    except Exception as e:
        logger.error(f"Failed to reanalyze photo {photo_id}: {e}")
        raise

@celery_app.task(name='tasks.ai_tasks.cleanup_old_tasks')
def cleanup_old_tasks():
    """Clean up old task results and temporary files"""
    logger.info("Running OneAlign task cleanup...")
    
    # This could include:
    # - Cleaning up old Celery results
    # - Removing temporary files
    # - Updating stale processing statuses
    
    # For now, just log
    logger.info("OneAlign task cleanup completed")
    
    return {'status': 'completed', 'cleaned_items': 0}

# ============================================================================
# Rotation Analysis Tasks
# ============================================================================

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_rotation_cv', queue='ai_analysis')
def analyze_rotation_cv(self, image_path: str, method: str = "auto", recipe_params: Optional[Dict[str, Any]] = None):
    """
    Analyze optimal rotation for an image using traditional computer vision techniques
    (Hough transform, horizon detection, etc.) instead of AI aesthetic scoring.
    
    This is much faster than the OneAlign approach and doesn't require GPU.
    
    Args:
        image_path: Path to the image file
        method: Detection method - "auto", "horizon", "lines", "faces", "exif"
        recipe_params: Optional recipe parameters for configuration
    
    Returns:
        Dict containing rotation analysis results
    """
    logger.info(f"Starting CV-based rotation analysis for {image_path}")
    logger.info(f"Method: {method}, Recipe params: {recipe_params}")
    
    try:
        # Import the CV rotation service directly to avoid loading all services
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
        from ai_components.services.rotation_detection_service import RotationDetectionService, TransformConfig
        
        # Create service with optional recipe configuration
        if recipe_params:
            config = TransformConfig.from_recipe(recipe_params)
            rotation_service = RotationDetectionService(config)
        else:
            rotation_service = RotationDetectionService()
        
        # Load and validate source image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Analyze the image
        transform_result = rotation_service.analyze_image(
            Path(image_path),
            detect_perspective=True,
            detect_skew=True,
            detect_distortion=recipe_params.get('enable_distortion_correction', False) if recipe_params else False
        )
        
        # Generate preview if rotation is needed
        preview_url = None
        preview_path = None
        if transform_result.needs_rotation or transform_result.needs_perspective or transform_result.needs_distortion_correction:
            # Load image using the load_image function that handles RAW files
            from ai_components.shared.image_utils import load_image
            image = load_image(image_path)
            # load_image already converts to RGB, so no need to check mode
            
            # Apply transformations
            image_array = np.array(image)
            corrected_array = rotation_service.apply_transform(image_array, transform_result)
            corrected_image = Image.fromarray(corrected_array)
            
            # Save preview
            import uuid
            preview_filename = f"rotation_preview_{uuid.uuid4().hex[:8]}.jpg"
            preview_path = Path(f"/app/data/temp/{preview_filename}")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            
            corrected_image.save(preview_path, format='JPEG', quality=85)
            preview_url = f"/api/files/temp/{preview_filename}"
            logger.info(f"Saved corrected preview: {preview_path}")
        
        # Prepare result with compatibility for both CV and OneAlign formats
        result = {
            'optimal_angle': float(transform_result.rotation_angle),
            'optimal_score': float(transform_result.confidence),  # Map confidence to score for compatibility
            'needs_rotation': transform_result.needs_rotation,
            'confidence': float(transform_result.confidence),
            'method_used': transform_result.method_used,
            'scene_type': transform_result.scene_type.value,
            'needs_perspective_correction': transform_result.needs_perspective,
            'needs_skew_correction': transform_result.needs_skew_correction,
            'needs_distortion_correction': transform_result.needs_distortion_correction,
            'status': 'completed',
            'preview_url': preview_url,
            'display_image_url': preview_url,  # Add alias for compatibility
            'display_image_path': str(preview_path) if preview_path else None,
            'processing_time': 0,  # CV processing is fast, actual time tracking could be added
            # Add mock data for compatibility with OneAlign format
            'all_scores': {str(transform_result.rotation_angle): transform_result.confidence},
            'search_parameters': {
                'method': method,
                'recipe_params': recipe_params
            }
        }
        
        # Add perspective info if detected
        if transform_result.needs_perspective:
            result['perspective_correction'] = {
                'enabled': True,
                'matrix_available': transform_result.perspective_matrix is not None
            }
        
        # Add skew info if detected
        if transform_result.needs_skew_correction:
            result['skew_correction'] = {
                'enabled': True,
                'skew_angle': float(transform_result.skew_angle)
            }
        
        # Add distortion info if detected
        if transform_result.needs_distortion_correction:
            result['distortion_correction'] = {
                'enabled': True,
                'coefficients_available': transform_result.distortion_coeffs is not None
            }
        
        logger.info(f"CV rotation analysis completed: angle={result['optimal_angle']:.1f}°, method={result['method_used']}")
        
        return result
        
    except Exception as e:
        error_msg = f"CV rotation analysis failed: {str(e)}"
        logger.error(f"Error in CV rotation analysis for {image_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_rotation_onealign', queue='ai_analysis')
def analyze_rotation_onealign(self, image_path: str, min_angle: float = -20.0, max_angle: float = 20.0, angle_step: float = 0.5):
    """
    Analyze optimal rotation for an image using OneAlign with batch processing optimization
    
    Args:
        image_path: Path to the image file
        min_angle: Minimum rotation angle to test
        max_angle: Maximum rotation angle to test
        angle_step: Step size for angle testing
    
    Returns:
        Dict containing rotation analysis results
    """
    import tempfile
    import shutil
    
    logger.info(f"Starting batch rotation analysis for {image_path}")
    
    try:
        # Load and validate source image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Handle RAW files (ARW, CR2, NEF, etc.) differently
        source_image = None
        image_path_lower = str(image_path).lower()
        
        if any(ext in image_path_lower for ext in ['.arw', '.cr2', '.nef', '.raf', '.dng']):
            # This is a RAW file
            logger.info(f"Detected RAW file: {image_path}")
            try:
                import rawpy
                logger.info(f"Using rawpy to process RAW file: {image_path}")
                
                with rawpy.imread(str(image_path)) as raw:
                    # Convert to RGB array
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,
                        half_size=True,  # Use half size for faster processing
                        output_bps=8,    # 8-bit output
                        bright=1.0,      # Default brightness
                        no_auto_bright=True
                    )
                    # Convert to PIL Image
                    source_image = Image.fromarray(rgb_array)
                    logger.info(f"Successfully processed RAW file to {source_image.size}")
                    
            except ImportError as e:
                logger.error(f"rawpy not available: {e}")
                logger.warning("Attempting to use PIL for RAW file (will likely fail)")
                try:
                    from ai_components.shared.image_utils import load_image
                    source_image = load_image(image_path)
                    logger.info(f"load_image function worked for RAW file: {source_image.size}")
                except Exception as load_error:
                    raise ValueError(f"Cannot process RAW file {image_path}: rawpy not available and load_image failed with: {load_error}")
            except Exception as e:
                logger.error(f"Failed to process RAW file {image_path} with rawpy: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError(f"Cannot process RAW file {image_path}: {e}")
        else:
            # Standard image formats (JPEG, PNG, TIFF, etc.)
            logger.info(f"Processing standard image file: {image_path}")
            try:
                from ai_components.shared.image_utils import load_image
                source_image = load_image(image_path)
                logger.info(f"Successfully opened standard image: {source_image.size}")
            except Exception as e:
                logger.error(f"Failed to open image {image_path}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise ValueError(f"Invalid image file or corrupted: {e}")
        
        if source_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        if source_image.mode not in ('RGB', 'RGBA'):
            source_image = source_image.convert('RGB')
        
        # Downsample for rotation analysis to improve speed while maintaining quality
        original_size = source_image.size
        original_pixels = original_size[0] * original_size[1]
        target_pixels = 2_000_000  # At least 2MP for good quality assessment
        
        if original_pixels > target_pixels:
            # Calculate ratio to get approximately 2MP
            ratio = (target_pixels / original_pixels) ** 0.5
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            # Use BICUBIC for compatibility - LANCZOS has issues in this environment
            try:
                resample = Image.Resampling.BICUBIC
            except AttributeError:
                resample = Image.BICUBIC
            source_image = source_image.resize(new_size, resample)
            new_pixels = new_size[0] * new_size[1]
            logger.info(f"Downsampled from {original_size} ({original_pixels/1_000_000:.1f}MP) to {source_image.size} ({new_pixels/1_000_000:.1f}MP) for rotation analysis")
        else:
            logger.info(f"No downsampling needed: {original_size} ({original_pixels/1_000_000:.1f}MP) already under 2MP")
        
        # Generate rotation angles to test
        angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        logger.info(f"Generating {len(angles)} rotated images from {min_angle}° to {max_angle}°")
        
        # Create temporary directory for rotated images
        temp_dir = Path(tempfile.mkdtemp(prefix="rotation_analysis_"))
        
        try:
            # Phase 1: Generate all rotated images first
            rotated_images = []
            temp_files = []
            
            for i, angle in enumerate(angles):
                try:
                    # Rotate and crop image
                    rotated_image = _rotate_and_crop_image(source_image, float(angle))
                    
                    # Save to temporary file
                    temp_file = temp_dir / f"rotated_{angle:+06.1f}deg.jpg"
                    rotated_image.save(temp_file, format='JPEG', quality=90)
                    
                    rotated_images.append(rotated_image)
                    temp_files.append((temp_file, angle))
                    
                    # Update progress for generation phase
                    progress = (i + 1) / len(angles) * 0.3  # First 30% for generation
                    current_task.update_state(
                        state='PROGRESS',
                        meta={
                            'phase': 'generating',
                            'current': i + 1,
                            'total': len(angles),
                            'progress': progress,
                            'current_angle': float(angle),
                            'message': f'Generating rotated image {i+1}/{len(angles)}'
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to generate rotated image for angle {angle}°: {e}")
                    continue
            
            logger.info(f"Generated {len(rotated_images)} rotated images. Starting batch analysis...")
            
            # Phase 2: Load OneAlign model once and analyze all images
            onealign_model = get_ai_model('onealign')
            
            # Batch process all rotated images
            all_scores = {}
            best_angle = 0.0
            best_score = -float('inf')
            
            for i, (temp_file, angle) in enumerate(temp_files):
                try:
                    # Load the rotated image from file
                    rotated_image = Image.open(temp_file)
                    
                    # Analyze with OneAlign - get aesthetic score only
                    results = onealign_model.assess_quality_and_aesthetics(rotated_image)
                    score = results['aesthetics_score']  # Use aesthetic score for rotation decisions
                    
                    all_scores[float(angle)] = float(score)
                    
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                    
                    # Update progress for analysis phase
                    analysis_progress = (i + 1) / len(temp_files) * 0.7  # Remaining 70% for analysis
                    total_progress = 0.3 + analysis_progress  # Add to generation progress
                    
                    current_task.update_state(
                        state='PROGRESS',
                        meta={
                            'phase': 'analyzing',
                            'current': i + 1,
                            'total': len(temp_files),
                            'progress': total_progress,
                            'current_angle': float(angle),
                            'current_score': float(score),
                            'best_angle': float(best_angle),
                            'best_score': float(best_score),
                            'message': f'Analyzing image {i+1}/{len(temp_files)}'
                        }
                    )
                    
                    logger.info(f"Angle {angle}°: score {score:.3f} (best so far: {best_angle}° = {best_score:.3f})")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze rotated image for angle {angle}°: {e}")
                    continue
            
            if not all_scores:
                raise ValueError("No rotation angles could be analyzed successfully")
            
            # Calculate optimal angle using weighted average of top scores instead of single peak
            optimal_angle = _calculate_optimal_rotation_angle(all_scores)
            
            logger.info(f"Batch rotation analysis completed. Single peak: {best_angle}° (score: {best_score:.3f})")
            logger.info(f"Weighted optimal: {optimal_angle}° (using top score distribution)")
            
            # Generate the optimal rotated and cropped image for display using original resolution
            # Load original image again for display version using the same RAW handling logic
            original_display_image = None
            if any(ext in image_path_lower for ext in ['.arw', '.cr2', '.nef', '.raf', '.dng']):
                try:
                    import rawpy
                    logger.info(f"Processing RAW file for display: {image_path}")
                    
                    with rawpy.imread(str(image_path)) as raw:
                        # Convert to RGB array - use full size for display
                        rgb_array = raw.postprocess(
                            use_camera_wb=True,
                            half_size=False,  # Full size for display
                            output_bps=8,
                            bright=1.0,
                            no_auto_bright=True
                        )
                        original_display_image = Image.fromarray(rgb_array)
                        
                except ImportError:
                    logger.warning("rawpy not available for display image, using load_image")
                    from ai_components.shared.image_utils import load_image
                    original_display_image = load_image(image_path)
                except Exception as e:
                    logger.error(f"Failed to process RAW file for display: {e}")
                    from ai_components.shared.image_utils import load_image
                    original_display_image = load_image(image_path)
            else:
                from ai_components.shared.image_utils import load_image
                original_display_image = load_image(image_path)
                
            if original_display_image.mode not in ('RGB', 'RGBA'):
                original_display_image = original_display_image.convert('RGB')
            
            # Apply optimal rotation to original image for display
            logger.info(f"Generating display image: rotating {original_display_image.size} image by {optimal_angle}°")
            optimal_rotated_image = _rotate_and_crop_image(original_display_image, float(optimal_angle))
            logger.info(f"Display image after rotation/crop: {optimal_rotated_image.size}")
            
            # Save the optimal image to temp directory for frontend display
            import uuid
            display_filename = f"rotation_preview_{uuid.uuid4().hex[:8]}.jpg"
            display_path = Path(f"/app/data/temp/{display_filename}")
            display_path.parent.mkdir(parents=True, exist_ok=True)
            
            optimal_rotated_image.save(display_path, format='JPEG', quality=85)
            logger.info(f"Saved optimal rotated image for display: {display_path}")
            logger.info(f"Display image final dimensions: {optimal_rotated_image.size}, URL: /api/files/temp/{display_filename}")
            
            return {
                'optimal_angle': float(optimal_angle),
                'optimal_score': float(all_scores.get(optimal_angle, best_score)),
                'all_scores': all_scores,
                'search_parameters': {
                    'min_angle': min_angle,
                    'max_angle': max_angle,
                    'angle_step': angle_step,
                    'total_tested': len(all_scores),
                    'generation_count': len(rotated_images),
                    'analysis_count': len(temp_files)
                },
                'display_image_path': str(display_path),
                'display_image_url': f"/api/files/temp/{display_filename}",
                'status': 'completed',
                'peak_angle': float(best_angle),  # Keep track of single peak for comparison
                'peak_score': float(best_score)
            }
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        
    except Exception as e:
        error_msg = f"Rotation analysis failed: {str(e)}"
        logger.error(f"Error in rotation analysis for {image_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def _calculate_optimal_rotation_angle(all_scores: Dict[float, float]) -> float:
    """
    Calculate optimal rotation angle using weighted average of top-scoring angles
    instead of just picking the single highest peak.
    
    This approach is more robust against single-point noise and considers
    the distribution of good scores around the optimal region.
    
    Args:
        all_scores: Dictionary mapping angles to their aesthetic scores
        
    Returns:
        Weighted optimal angle in degrees
    """
    if not all_scores:
        return 0.0
    
    # Convert to numpy arrays for easier processing
    angles = np.array(list(all_scores.keys()))
    scores = np.array(list(all_scores.values()))
    
    # Normalize scores to [0, 1] range
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return float(angles[0])  # All scores are the same
    
    normalized_scores = (scores - min_score) / (max_score - min_score)
    
    # Use exponential weighting to emphasize high scores
    # Higher exponent = more focus on top scores
    exponent = 3.0
    weights = np.power(normalized_scores, exponent)
    
    # Calculate weighted average
    if np.sum(weights) == 0:
        return float(angles[np.argmax(scores)])  # Fallback to highest score
    
    weighted_angle = np.sum(angles * weights) / np.sum(weights)
    
    # Find the nearest tested angle (since we need to use an angle we actually tested)
    nearest_idx = np.argmin(np.abs(angles - weighted_angle))
    optimal_angle = angles[nearest_idx]
    
    logger.info(f"Weighted average angle: {weighted_angle:.2f}°, nearest tested: {optimal_angle:.2f}°")
    logger.info(f"Score distribution: min={min_score:.3f}, max={max_score:.3f}, weighted_score={all_scores[optimal_angle]:.3f}")
    
    return float(optimal_angle)

def _rotate_and_crop_image(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate image and crop to completely remove ALL black bars/triangles.
    
    Uses a completely different approach: rotate with expand=True, then crop out
    the exact inscribed rectangle that contains no black pixels.
    
    Args:
        image: Source PIL Image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated and cropped PIL Image with NO black areas and original aspect ratio
    """
    if abs(angle) < 0.01:  # No rotation needed
        return image
    
    # Original dimensions and aspect ratio
    orig_w, orig_h = image.size
    orig_aspect = orig_w / orig_h
    
    # Step 1: Rotate with expansion to avoid cutting off any image content
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC
    
    rotated = image.rotate(
        angle,
        resample=resample,
        expand=True,
        fillcolor=(0, 0, 0)
    )
    
    rot_w, rot_h = rotated.size
    
    # Step 2: Calculate the largest inscribed rectangle with same aspect ratio
    # This is pure geometry - no detection needed
    
    angle_rad = np.radians(abs(angle))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Use proper geometric formula for inscribed rectangle in rotated rectangle
    # The maximum axis-aligned rectangle inscribed in a rotated rectangle is given by:
    # inscribed_width = |w*cos(θ) - h*sin(θ)|
    # inscribed_height = |h*cos(θ) - w*sin(θ)|
    # where w,h are original dimensions and θ is rotation angle
    
    try:
        logger.info("Using geometric formula for inscribed rectangle...")
        
        # Calculate the inscribed rectangle dimensions
        inscribed_w = abs(orig_w * cos_a - orig_h * sin_a)
        inscribed_h = abs(orig_h * cos_a - orig_w * sin_a)
        
        logger.info(f"  CALCULATED INSCRIBED: {inscribed_w:.1f}x{inscribed_h:.1f}")
        
        # Ensure we don't get impossible dimensions
        min_w = orig_w * 0.1  # At least 10% of original
        min_h = orig_h * 0.1
        inscribed_w = max(inscribed_w, min_w)
        inscribed_h = max(inscribed_h, min_h)
        
        # Maintain original aspect ratio by fitting within inscribed dimensions
        # Calculate which dimension is the limiting factor
        scale_w = inscribed_w / orig_w
        scale_h = inscribed_h / orig_h
        scale = min(scale_w, scale_h)  # Use the more restrictive scaling
        
        crop_w = int(orig_w * scale)
        crop_h = int(orig_h * scale)
        
        logger.info(f"  ASPECT-PRESERVED CROP: {crop_w}x{crop_h} (scale: {scale:.3f})")
        
    except Exception as e:
        logger.error(f"Geometric calculation failed: {e}")
        
        # Ultra conservative fallback - just scale by cosine
        scale = cos_a * 0.9  # Conservative scaling
        crop_w = int(orig_w * scale)
        crop_h = int(orig_h * scale)
        
        logger.info(f"  FALLBACK CROP: {crop_w}x{crop_h} (scale: {scale:.3f})")
    
    # Step 3: Center the crop in the rotated image
    center_x = rot_w // 2
    center_y = rot_h // 2
    
    left = center_x - crop_w // 2
    top = center_y - crop_h // 2
    right = left + crop_w
    bottom = top + crop_h
    
    # Ensure crop bounds are within the rotated image
    left = max(0, left)
    top = max(0, top)
    right = min(rot_w, right)
    bottom = min(rot_h, bottom)
    
    # Adjust crop size if bounds were clipped
    crop_w = right - left
    crop_h = bottom - top
    
    # Step 4: Perform the final crop
    final_image = rotated.crop((left, top, right, bottom))
    
    # Calculate and log statistics
    original_area = orig_w * orig_h
    final_area = crop_w * crop_h
    area_retention = final_area / original_area
    final_aspect = crop_w / crop_h
    
    logger.info(f"Rotation {angle:.1f}°: {orig_w}x{orig_h} -> rotated {rot_w}x{rot_h} -> final {crop_w}x{crop_h}")
    logger.info(f"  Aspect: {orig_aspect:.3f} -> {final_aspect:.3f}, Area retained: {area_retention:.1%}")
    logger.info(f"  Final crop box: ({left},{top},{right},{bottom})")
    
    if area_retention < 0.3:
        logger.warning(f"Large rotation {angle:.1f}° results in {area_retention:.1%} area retention")
    
    return final_image

# ============================================================================
# VLM-Based Intelligent Cropping Tasks
# ============================================================================

from pydantic import BaseModel
from typing import List, Literal
import json

# Pydantic models for structured outputs
class FocalPoint(BaseModel):
    subject: str
    importance: Literal["high", "medium", "low"]
    location: str

class CompositionAnalysis(BaseModel):
    rule_of_thirds_compliance: Literal["high", "medium", "low"]
    balance: str
    leading_lines: str
    depth: str

class CroppingRecommendations(BaseModel):
    primary_strategy: str
    aspect_ratios: List[str]
    avoid_areas: List[str]

class PhotoCompositionAnalysis(BaseModel):
    description: str
    main_subjects: List[str]
    composition_type: Literal["portrait", "landscape", "macro", "group", "action", "architectural", "nature", "unknown"]
    focal_points: List[FocalPoint]
    composition_analysis: CompositionAnalysis
    cropping_recommendations: CroppingRecommendations

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_composition_vlm', queue='ai_analysis')
def analyze_composition_vlm(self, image_path: str, ollama_model: str = "qwen2.5-vl:7b"):
    """
    Stage 1: Analyze photo composition and content using VLM
    
    Args:
        image_path: Path to the image file
        ollama_model: Ollama model to use for analysis
    
    Returns:
        Dict containing photo description and composition analysis
    """
    logger.info(f"Starting VLM composition analysis for {image_path}")
    
    try:
        # Step 1: Clean up GPU memory to make space for Ollama
        cleanup_result = cleanup_gpu_memory()
        logger.info(f"GPU cleanup result: {cleanup_result}")
        
        # Step 2: Load and validate image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Handle RAW files like in rotation analysis
        source_image = None
        image_path_lower = str(image_path).lower()
        
        if any(ext in image_path_lower for ext in ['.arw', '.cr2', '.nef', '.raf', '.dng']):
            try:
                import rawpy
                with rawpy.imread(str(image_path)) as raw:
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        output_bps=8,
                        bright=1.0,
                        no_auto_bright=True
                    )
                    source_image = Image.fromarray(rgb_array)
            except ImportError:
                from ai_components.shared.image_utils import load_image
                source_image = load_image(image_path)
        else:
            from ai_components.shared.image_utils import load_image
            source_image = load_image(image_path)
        
        if source_image.mode not in ('RGB', 'RGBA'):
            source_image = source_image.convert('RGB')
        
        # Step 3: Prepare image with padding based on model requirements
        model_resolution_requirements = {
            'gemma3': 896,  # Uses 896x896 with Pan & Scan
            'qwen2.5': None,  # Dynamic resolution, no padding needed
            'llama3.2-vision': 1120,  # Max 1120x1120
            'llava': 336,  # 336x336 for v1.5
        }
        
        # Determine target resolution based on model
        target_resolution = None
        model_lower = ollama_model.lower()
        for model_key, resolution in model_resolution_requirements.items():
            if model_key in model_lower:
                target_resolution = resolution
                break
        
        # Store original dimensions and prepare padded image
        original_width, original_height = source_image.size
        padded_image = source_image
        padding_info = {
            'original_width': original_width,
            'original_height': original_height,
            'padded': False,
            'pad_left': 0,
            'pad_top': 0,
            'padded_width': original_width,
            'padded_height': original_height
        }
        
        # Apply padding if model requires square input
        if target_resolution and 'qwen' not in model_lower:
            # Calculate scaling to fit within target resolution while maintaining aspect ratio
            scale = min(target_resolution / original_width, target_resolution / original_height)
            
            # Only scale down if image is larger than target
            if scale < 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                padded_image = source_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                new_width = original_width
                new_height = original_height
            
            # Create square canvas with black padding
            canvas = Image.new('RGB', (target_resolution, target_resolution), (0, 0, 0))
            
            # Calculate padding to center the image
            pad_left = (target_resolution - new_width) // 2
            pad_top = (target_resolution - new_height) // 2
            
            # Paste image onto canvas
            canvas.paste(padded_image.resize((new_width, new_height), Image.Resampling.LANCZOS), (pad_left, pad_top))
            padded_image = canvas
            
            # Update padding info
            padding_info.update({
                'padded': True,
                'pad_left': pad_left,
                'pad_top': pad_top,
                'padded_width': target_resolution,
                'padded_height': target_resolution,
                'scale_factor': scale if scale < 1.0 else 1.0,
                'scaled_width': new_width,
                'scaled_height': new_height
            })
            
            logger.info(f"Padded image from {original_width}x{original_height} to {target_resolution}x{target_resolution} for model {ollama_model}")
        
        # Step 4: Save padded image to temp for Ollama access
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            padded_image.save(temp_file.name, format='JPEG', quality=95)
            temp_image_path = temp_file.name
        
        try:
            # Step 4: Call Ollama for composition analysis
            import requests
            import os
            
            ollama_url = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
            
            # Prepare the analysis prompt
            analysis_prompt = """Analyze this photograph and provide a detailed composition analysis. Be thorough and specific in your analysis."""

            # Encode image as base64
            import base64
            with open(temp_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Call Ollama API with structured output
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": analysis_prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": False,
                    "format": PhotoCompositionAnalysis.model_json_schema(),  # Use Pydantic schema
                    "options": {
                        "temperature": 0,  # Zero temperature for maximum consistency
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            ollama_result = response.json()
            
            # Step 5: Parse structured response
            import json
            try:
                # Get the content from the message
                message_content = ollama_result.get('message', {}).get('content', '')
                
                # Validate with Pydantic
                analysis_result = PhotoCompositionAnalysis.model_validate_json(message_content)
                
                # Convert to dict for response
                analysis_dict = analysis_result.model_dump()
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse structured response: {e}")
                logger.info(f"Raw response: {ollama_result}")
                
                # Fallback: create structured response
                analysis_dict = {
                    "description": "Failed to analyze image",
                    "main_subjects": ["unknown"],
                    "composition_type": "unknown",
                    "focal_points": [{"subject": "unknown", "importance": "medium", "location": "center"}],
                    "composition_analysis": {
                        "rule_of_thirds_compliance": "unknown",
                        "balance": "unknown",
                        "leading_lines": "unknown",
                        "depth": "unknown"
                    },
                    "cropping_recommendations": {
                        "primary_strategy": "center crop",
                        "aspect_ratios": ["16:9", "4:3", "1:1"],
                        "avoid_areas": []
                    },
                    "parsing_error": str(e)
                }
            
            logger.info(f"VLM composition analysis completed successfully")
            
            return {
                'status': 'completed',
                'analysis': analysis_dict,
                'image_dimensions': source_image.size,
                'model_used': ollama_model,
                'processing_time': ollama_result.get('eval_duration', 0) / 1e9 if 'eval_duration' in ollama_result else 0,
                'padding_info': padding_info  # Include padding info for coordinate conversion
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_image_path)
            except:
                pass
        
    except Exception as e:
        error_msg = f"VLM composition analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Pydantic models for crop generation
class CropBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class CropGeneration(BaseModel):
    crop_box: CropBox
    reasoning: str
    composition_benefits: List[str]
    preserved_elements: List[str]

@celery_app.task(bind=True, name='tasks.ai_tasks.generate_crop_bbox_vlm', queue='ai_analysis')
def generate_crop_bbox_vlm(self, image_path: str, user_intent: str, composition_analysis: Dict[str, Any], 
                          target_aspect_ratio: str = "16:9", ollama_model: str = "qwen2.5-vl:7b"):
    """
    Stage 2: Generate precise bounding box for cropping based on user intent and composition analysis
    
    Args:
        image_path: Path to the image file
        user_intent: User's description of what they want the crop to achieve
        composition_analysis: Result from Stage 1 analysis
        target_aspect_ratio: Desired aspect ratio (e.g., "16:9", "4:3", "1:1")
        ollama_model: Ollama model to use
    
    Returns:
        Dict containing precise crop bounding box coordinates
    """
    logger.info(f"Starting VLM crop bounding box generation for {image_path}")
    
    try:
        # Step 1: Ensure GPU memory is available
        cleanup_result = cleanup_gpu_memory()
        logger.info(f"GPU cleanup result: {cleanup_result}")
        
        # Step 2: Load image (same as Stage 1)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        source_image = None
        image_path_lower = str(image_path).lower()
        
        if any(ext in image_path_lower for ext in ['.arw', '.cr2', '.nef', '.raf', '.dng']):
            try:
                import rawpy
                with rawpy.imread(str(image_path)) as raw:
                    rgb_array = raw.postprocess(
                        use_camera_wb=True,
                        half_size=False,
                        output_bps=8,
                        bright=1.0,
                        no_auto_bright=True
                    )
                    source_image = Image.fromarray(rgb_array)
            except ImportError:
                from ai_components.shared.image_utils import load_image
                source_image = load_image(image_path)
        else:
            from ai_components.shared.image_utils import load_image
            source_image = load_image(image_path)
        
        if source_image.mode not in ('RGB', 'RGBA'):
            source_image = source_image.convert('RGB')
        
        img_width, img_height = source_image.size
        
        # Step 3: Parse target aspect ratio
        if ":" in target_aspect_ratio:
            ratio_parts = target_aspect_ratio.split(":")
            target_w_ratio = float(ratio_parts[0])
            target_h_ratio = float(ratio_parts[1])
            target_ratio = target_w_ratio / target_h_ratio
        else:
            target_ratio = 16.0 / 9.0  # Default fallback
        
        # Step 4: Apply same padding logic as composition analysis
        model_resolution_requirements = {
            'gemma3': 896,
            'qwen2.5': None,
            'llama3.2-vision': 1120,
            'llava': 336,
        }
        
        target_resolution = None
        model_lower = ollama_model.lower()
        for model_key, resolution in model_resolution_requirements.items():
            if model_key in model_lower:
                target_resolution = resolution
                break
        
        original_width, original_height = source_image.size
        padded_image = source_image
        padding_info = {
            'original_width': original_width,
            'original_height': original_height,
            'padded': False,
            'pad_left': 0,
            'pad_top': 0,
            'padded_width': original_width,
            'padded_height': original_height
        }
        
        if target_resolution and 'qwen' not in model_lower:
            scale = min(target_resolution / original_width, target_resolution / original_height)
            
            if scale < 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                padded_image = source_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                new_width = original_width
                new_height = original_height
            
            canvas = Image.new('RGB', (target_resolution, target_resolution), (0, 0, 0))
            pad_left = (target_resolution - new_width) // 2
            pad_top = (target_resolution - new_height) // 2
            canvas.paste(padded_image.resize((new_width, new_height), Image.Resampling.LANCZOS), (pad_left, pad_top))
            padded_image = canvas
            
            padding_info.update({
                'padded': True,
                'pad_left': pad_left,
                'pad_top': pad_top,
                'padded_width': target_resolution,
                'padded_height': target_resolution,
                'scale_factor': scale if scale < 1.0 else 1.0,
                'scaled_width': new_width,
                'scaled_height': new_height
            })
            
            logger.info(f"Padded image for crop generation from {original_width}x{original_height} to {target_resolution}x{target_resolution}")
        
        # Step 5: Save padded image to temp for Ollama
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            padded_image.save(temp_file.name, format='JPEG', quality=95)
            temp_image_path = temp_file.name
        
        try:
            # Step 5: Create detailed cropping prompt
            crop_prompt = f"""Based on the image and the following analysis, provide precise cropping coordinates.

PREVIOUS ANALYSIS:
{json.dumps(composition_analysis, indent=2)}

USER INTENT: {user_intent}

TARGET ASPECT RATIO: {target_aspect_ratio} (width/height = {target_ratio:.3f})

IMAGE DIMENSIONS: {padding_info['padded_width']} x {padding_info['padded_height']} pixels
{'NOTE: Image has been padded with black bars. Avoid including black padding in the crop.' if padding_info['padded'] else ''}

Please analyze the image and provide the optimal crop box coordinates that:
1. Achieves the user's intent: "{user_intent}"
2. Maintains the target aspect ratio of {target_aspect_ratio}
3. Follows good composition principles
4. Preserves the most important elements identified in the analysis

The crop box must:
- Have coordinates within image bounds (0 <= x < {padding_info['padded_width']}, 0 <= y < {padding_info['padded_height']})
- Have width/height ratio approximately equal to {target_ratio:.3f}
- Preserve the main subjects identified in the analysis
- Be as large as possible while meeting the constraints
- Avoid including black padding areas if the image has been padded"""

            # Step 6: Call Ollama for crop generation with structured output
            import requests
            import os
            import base64
            
            ollama_url = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
            
            with open(temp_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": crop_prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": False,
                    "format": CropGeneration.model_json_schema(),  # Use Pydantic schema
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            ollama_result = response.json()
            
            # Step 7: Parse and validate crop coordinates
            import json
            try:
                # Get the content from the message
                message_content = ollama_result.get('message', {}).get('content', '')
                
                # Validate with Pydantic
                crop_generation = CropGeneration.model_validate_json(message_content)
                
                # Extract crop box from model (in padded space)
                padded_x = crop_generation.crop_box.x
                padded_y = crop_generation.crop_box.y
                padded_width = crop_generation.crop_box.width
                padded_height = crop_generation.crop_box.height
                
                # Convert coordinates from padded space to original image space
                if padding_info['padded']:
                    # Remove padding offset
                    x_in_scaled = padded_x - padding_info['pad_left']
                    y_in_scaled = padded_y - padding_info['pad_top']
                    
                    # Convert from scaled space to original space
                    scale_factor = padding_info['scale_factor']
                    x = int(x_in_scaled / scale_factor)
                    y = int(y_in_scaled / scale_factor)
                    width = int(padded_width / scale_factor)
                    height = int(padded_height / scale_factor)
                    
                    logger.info(f"Converted crop from padded space ({padded_x},{padded_y},{padded_width}x{padded_height}) to original space ({x},{y},{width}x{height})")
                else:
                    # No padding, use coordinates directly
                    x = padded_x
                    y = padded_y
                    width = padded_width
                    height = padded_height
                
                # Validate coordinates are within original image bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                width = max(1, min(width, img_width - x))
                height = max(1, min(height, img_height - y))
                
                # Adjust to maintain aspect ratio
                actual_ratio = width / height
                if abs(actual_ratio - target_ratio) > 0.1:  # If ratio is off by more than 10%
                    if actual_ratio > target_ratio:
                        # Too wide, reduce width
                        width = int(height * target_ratio)
                    else:
                        # Too tall, reduce height
                        height = int(width / target_ratio)
                    
                    # Ensure bounds
                    width = min(width, img_width - x)
                    height = min(height, img_height - y)
                
                # Create final result
                crop_result = {
                    'crop_box': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    },
                    'reasoning': crop_generation.reasoning,
                    'composition_benefits': crop_generation.composition_benefits,
                    'preserved_elements': crop_generation.preserved_elements,
                    'validated': True,
                    'actual_ratio': width / height,
                    'target_ratio': target_ratio
                }
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse crop coordinates: {e}")
                logger.info(f"Raw response: {ollama_result}")
                
                # Fallback: center crop with target aspect ratio
                if img_width / img_height > target_ratio:
                    # Image is wider than target, crop width
                    height = img_height
                    width = int(height * target_ratio)
                    x = (img_width - width) // 2
                    y = 0
                else:
                    # Image is taller than target, crop height
                    width = img_width
                    height = int(width / target_ratio)
                    x = 0
                    y = (img_height - height) // 2
                
                crop_result = {
                    'crop_box': {'x': x, 'y': y, 'width': width, 'height': height},
                    'reasoning': 'Fallback center crop due to parsing error',
                    'composition_benefits': ['centered composition'],
                    'preserved_elements': ['main subject area'],
                    'validated': True,
                    'actual_ratio': width / height,
                    'target_ratio': target_ratio,
                    'fallback_used': True,
                    'parsing_error': str(e)
                }
            
            logger.info(f"VLM crop generation completed successfully")
            
            return {
                'status': 'completed',
                'crop_result': crop_result,
                'user_intent': user_intent,
                'target_aspect_ratio': target_aspect_ratio,
                'image_dimensions': [img_width, img_height],
                'model_used': ollama_model,
                'processing_time': ollama_result.get('eval_duration', 0) / 1e9 if 'eval_duration' in ollama_result else 0
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_image_path)
            except:
                pass
        
    except Exception as e:
        error_msg = f"VLM crop generation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# ============================================================================
# Two-Stage VLM Crop Analysis
# ============================================================================

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_crop_vlm_round1', queue='ai_analysis')
def analyze_crop_vlm_round1(self, image_path: str, ollama_model: str = "qwen2.5-vl:7b"):
    """
    Round 1: Get concise photo description and written crop directions
    
    Args:
        image_path: Path to the image file
        ollama_model: Ollama model to use for analysis
    
    Returns:
        Dict containing photo description and crop directions
    """
    logger.info(f"Starting VLM crop analysis Round 1 for {image_path}")
    
    try:
        # Import required Pydantic model
        from ai_components.shared.ollama_schemas import CropAnalysisRound1
        
        # Step 1: Load and prepare image (similar to existing VLM tasks)
        # NOTE: GPU cleanup removed - it was causing Ollama to use CPU instead of GPU
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        source_image = _load_image_for_analysis(image_path)
        
        # Step 3: Save image to temp for Ollama
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            source_image.save(temp_file.name, format='JPEG', quality=95)
            temp_image_path = temp_file.name
        
        try:
            # Step 4: Call Ollama for Round 1 analysis
            import requests
            import os
            import base64
            
            ollama_url = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
            
            # Prepare the analysis prompt - simple and direct
            analysis_prompt = """1. Describe this photo in one sentence.
2. In 1-2 sentences, explain where to crop for the best composition OR state if the image is already well-composed and needs no cropping."""

            # Encode image as base64
            with open(temp_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Call Ollama API with structured output
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": analysis_prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": False,
                    "format": CropAnalysisRound1.model_json_schema(),
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            ollama_result = response.json()
            
            # Step 5: Parse structured response
            import json
            message_content = ollama_result.get('message', {}).get('content', '')
            
            # Validate with Pydantic
            round1_result = CropAnalysisRound1.model_validate_json(message_content)
            
            logger.info(f"VLM crop analysis Round 1 completed successfully")
            
            return {
                'status': 'completed',
                'photo_description': round1_result.photo_description,
                'crop_directions': round1_result.crop_directions,
                'image_dimensions': source_image.size,
                'model_used': ollama_model
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_image_path)
            except:
                pass
        
    except Exception as e:
        error_msg = f"VLM crop analysis Round 1 failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_crop_vlm_round2', queue='ai_analysis')
def analyze_crop_vlm_round2(self, image_path: str, round1_result: Dict[str, Any], 
                           ollama_model: str = "qwen2.5-vl:7b"):
    """
    Round 2: Generate exact crop coordinates based on Round 1 analysis
    
    Args:
        image_path: Path to the image file
        round1_result: Result from Round 1 containing description and directions
        ollama_model: Ollama model to use
    
    Returns:
        Dict containing exact crop coordinates with auto-selected aspect ratio
    """
    logger.info(f"Starting VLM crop analysis Round 2 for {image_path}")
    
    try:
        # Import required Pydantic model
        from ai_components.shared.ollama_schemas import CropAnalysisRound2Improved
        
        # Step 1: Load image
        # NOTE: GPU cleanup removed - it was causing Ollama to use CPU instead of GPU
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        source_image = _load_image_for_analysis(image_path)
        img_width, img_height = source_image.size
        
        # Step 3: Save image to temp for Ollama
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            source_image.save(temp_file.name, format='JPEG', quality=95)
            temp_image_path = temp_file.name
        
        try:
            # Step 4: Call Ollama for Round 2 analysis
            import requests
            import os
            import base64
            
            ollama_url = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
            
            # Prepare the prompt with Round 1 results - keep it simple
            round2_prompt = f"""{round1_result['photo_description']}

Crop suggestion: {round1_result['crop_directions']}

Provide:
- center_x, center_y: Center point of the crop (0-1). If no crop needed, use 0.5, 0.5
- desired_width, desired_height: Rough crop size (0-1). If no crop needed, use 1.0, 1.0
- aspect_ratio: Choose from 1:1, 2:3, 3:2, 3:4, 4:3, 16:9, or 9:16. Use the current aspect ratio if no crop needed.

If the image doesn't need cropping, set center at 0.5,0.5 and size at 1.0,1.0."""

            # Encode image as base64
            with open(temp_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Call Ollama API with structured output
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": round2_prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": False,
                    "format": CropAnalysisRound2Improved.model_json_schema(),
                    "options": {
                        "temperature": 0,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            ollama_result = response.json()
            
            # Step 5: Parse structured response
            import json
            message_content = ollama_result.get('message', {}).get('content', '')
            
            # Validate with Pydantic
            round2_result = CropAnalysisRound2Improved.model_validate_json(message_content)
            
            # Apply aspect ratio enforcement
            enforced_coords = _enforce_aspect_ratio(
                center_x=round2_result.center_x,
                center_y=round2_result.center_y,
                desired_width=round2_result.desired_width,
                desired_height=round2_result.desired_height,
                aspect_ratio_str=round2_result.aspect_ratio,
                img_width=img_width,
                img_height=img_height
            )
            
            # Convert percentages to actual pixel coordinates
            x1_px = int(enforced_coords['x1'] * img_width)
            y1_px = int(enforced_coords['y1'] * img_height)
            x2_px = int(enforced_coords['x2'] * img_width)
            y2_px = int(enforced_coords['y2'] * img_height)
            
            # Ensure valid coordinates (should already be valid from enforcement)
            x1_px = max(0, min(x1_px, img_width))
            y1_px = max(0, min(y1_px, img_height))
            x2_px = max(x1_px + 1, min(x2_px, img_width))
            y2_px = max(y1_px + 1, min(y2_px, img_height))
            
            # Log the adjustment for debugging
            logger.info(f"VLM suggested center: ({round2_result.center_x:.3f}, {round2_result.center_y:.3f})")
            logger.info(f"VLM suggested size: {round2_result.desired_width:.3f} x {round2_result.desired_height:.3f}")
            logger.info(f"Enforced crop: ({enforced_coords['x1']:.3f}, {enforced_coords['y1']:.3f}) to ({enforced_coords['x2']:.3f}, {enforced_coords['y2']:.3f})")
            logger.info(f"Aspect ratio: {round2_result.aspect_ratio}")
            
            # Verify the aspect ratio is correct
            actual_width = x2_px - x1_px
            actual_height = y2_px - y1_px
            actual_aspect = actual_width / actual_height
            logger.info(f"Final dimensions: {actual_width}x{actual_height}, aspect ratio: {actual_aspect:.3f}")
            
            logger.info(f"VLM crop analysis Round 2 completed successfully")
            
            return {
                'status': 'completed',
                'crop_coordinates': {
                    'x1': enforced_coords['x1'],
                    'y1': enforced_coords['y1'],
                    'x2': enforced_coords['x2'],
                    'y2': enforced_coords['y2'],
                    'x1_px': x1_px,
                    'y1_px': y1_px,
                    'x2_px': x2_px,
                    'y2_px': y2_px
                },
                'aspect_ratio': round2_result.aspect_ratio,
                'confidence': round2_result.confidence,
                'image_dimensions': source_image.size,
                'model_used': ollama_model,
                'rotation_hint': round2_result.rotation_hint
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_image_path)
            except:
                pass
        
    except Exception as e:
        error_msg = f"VLM crop analysis Round 2 failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def _load_image_for_analysis(image_path: str) -> Image.Image:
    """Helper function to load images including RAW formats"""
    image_path_lower = str(image_path).lower()
    
    if any(ext in image_path_lower for ext in ['.arw', '.cr2', '.nef', '.raf', '.dng']):
        try:
            import rawpy
            with rawpy.imread(str(image_path)) as raw:
                rgb_array = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    output_bps=8,
                    bright=1.0,
                    no_auto_bright=True
                )
                source_image = Image.fromarray(rgb_array)
        except ImportError:
            from ai_components.shared.image_utils import load_image
            source_image = load_image(image_path)
    else:
        from ai_components.shared.image_utils import load_image
        source_image = load_image(image_path)
    
    if source_image.mode not in ('RGB', 'RGBA'):
        source_image = source_image.convert('RGB')
    
    return source_image


def _enforce_aspect_ratio(center_x: float, center_y: float, 
                         desired_width: float, desired_height: float,
                         aspect_ratio_str: str, img_width: int, img_height: int) -> Dict[str, float]:
    """
    Enforce exact aspect ratio while maintaining center point and maximizing crop area
    
    Args:
        center_x, center_y: Center point as percentages (0-1)
        desired_width, desired_height: Desired dimensions as percentages (0-1)
        aspect_ratio_str: Target aspect ratio like "3:4", "16:9", etc.
        img_width, img_height: Actual image dimensions in pixels
        
    Returns:
        Dict with enforced x1, y1, x2, y2 coordinates as percentages
    """
    # Parse the aspect ratio string
    aspect_parts = aspect_ratio_str.split(':')
    target_width_ratio = float(aspect_parts[0])
    target_height_ratio = float(aspect_parts[1])
    target_aspect = target_width_ratio / target_height_ratio
    
    # Convert desired dimensions to pixels
    desired_width_px = desired_width * img_width
    desired_height_px = desired_height * img_height
    
    # Calculate which dimension to use as the base
    # We'll use the larger of the two suggested dimensions and adjust the other
    current_aspect = desired_width_px / desired_height_px
    
    if current_aspect > target_aspect:
        # Width is too large, use height as base
        final_height_px = desired_height_px
        final_width_px = final_height_px * target_aspect
    else:
        # Height is too large, use width as base
        final_width_px = desired_width_px
        final_height_px = final_width_px / target_aspect
    
    # Now we have the correct dimensions, calculate the crop box
    # centered on the suggested center point
    half_width = final_width_px / 2
    half_height = final_height_px / 2
    
    # Initial crop box
    x1_px = (center_x * img_width) - half_width
    y1_px = (center_y * img_height) - half_height
    x2_px = (center_x * img_width) + half_width
    y2_px = (center_y * img_height) + half_height
    
    # Ensure the crop box fits within the image
    # If it doesn't fit, we need to shift it
    if x1_px < 0:
        x2_px -= x1_px  # Shift right by the amount we're out of bounds
        x1_px = 0
    if y1_px < 0:
        y2_px -= y1_px  # Shift down
        y1_px = 0
    if x2_px > img_width:
        x1_px -= (x2_px - img_width)  # Shift left
        x2_px = img_width
    if y2_px > img_height:
        y1_px -= (y2_px - img_height)  # Shift up
        y2_px = img_height
    
    # Final check: if the crop is still out of bounds (image too small),
    # scale it down while maintaining aspect ratio
    if x1_px < 0 or y1_px < 0 or x2_px > img_width or y2_px > img_height:
        # Calculate the maximum possible size while maintaining aspect ratio
        max_width = img_width
        max_height = img_height
        
        if max_width / max_height > target_aspect:
            # Height is limiting factor
            final_height_px = max_height
            final_width_px = final_height_px * target_aspect
        else:
            # Width is limiting factor
            final_width_px = max_width
            final_height_px = final_width_px / target_aspect
        
        # Center the crop
        x1_px = (img_width - final_width_px) / 2
        y1_px = (img_height - final_height_px) / 2
        x2_px = x1_px + final_width_px
        y2_px = y1_px + final_height_px
    
    # Convert back to percentages
    return {
        'x1': max(0, x1_px / img_width),
        'y1': max(0, y1_px / img_height),
        'x2': min(1, x2_px / img_width),
        'y2': min(1, y2_px / img_height)
    }


# ============================================================================
# Backwards Compatibility
# ============================================================================

@celery_app.task(bind=True, name='tasks.ai_tasks.analyze_photo_nima')
def analyze_photo_nima(self, photo_id: str, photo_path: str, include_technical: bool = False):
    """
    Backwards compatibility wrapper - redirects to OneAlign
    """
    logger.info(f"NIMA task called, redirecting to OneAlign for photo {photo_id}")
    return analyze_photo_onealign(photo_id, photo_path)