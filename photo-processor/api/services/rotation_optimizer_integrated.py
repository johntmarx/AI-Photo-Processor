"""
Integrated Rotation Optimizer with OneAlign

This integrates the rotation optimization with the actual OneAlign model
from the AI components.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import asyncio
import httpx
import base64
import io

from .rotation_optimizer import RotationOptimizer

logger = logging.getLogger(__name__)

class IntegratedRotationOptimizer(RotationOptimizer):
    """
    Rotation optimizer that uses the actual OneAlign service via HTTP.
    """
    
    def __init__(self, ai_service_url: str = "http://ai-components:8001"):
        """
        Initialize with connection to AI service.
        
        Args:
            ai_service_url: URL of the AI components service
        """
        super().__init__()
        self.ai_service_url = ai_service_url
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"Integrated rotation optimizer initialized with AI service at {ai_service_url}")
    
    async def score_image(self, image: Image.Image) -> float:
        """
        Score an image using OneAlign model via the AI service.
        
        Args:
            image: PIL Image to score
            
        Returns:
            Aesthetic score (1-10, higher is better)
        """
        try:
            # Convert PIL image to base64 for API
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Call OneAlign API
            response = await self.client.post(
                f"{self.ai_service_url}/analyze/onealign",
                json={
                    "image": image_base64,
                    "include_technical": False  # We only need aesthetic score
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OneAlign API error: {response.status_code} - {response.text}")
                return 5.0  # Default middle score
            
            result = response.json()
            
            # Extract aesthetic score
            aesthetic_score = result.get('aesthetic_score', 5.0)
            
            # OneAlign returns 1-10 scale
            return float(aesthetic_score)
            
        except Exception as e:
            logger.error(f"Failed to score image with OneAlign: {e}")
            # Fallback to simple heuristic
            return self._heuristic_score(image)
    
    async def score_image_from_path(self, image_path: Path) -> float:
        """
        Score an image file directly using OneAlign.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Aesthetic score (1-10)
        """
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Call OneAlign API
            response = await self.client.post(
                f"{self.ai_service_url}/analyze/onealign",
                json={
                    "image": image_base64,
                    "include_technical": False
                }
            )
            
            if response.status_code != 200:
                logger.error(f"OneAlign API error: {response.status_code} - {response.text}")
                return 5.0
            
            result = response.json()
            return float(result.get('aesthetic_score', 5.0))
            
        except Exception as e:
            logger.error(f"Failed to score image from path: {e}")
            return 5.0
    
    async def find_optimal_rotation_fast(
        self,
        image_path: Path,
        coarse_step: float = 2.0,
        fine_step: float = 0.25,
        fine_range: float = 2.0,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Fast two-phase rotation optimization.
        
        Phase 1: Coarse search with larger steps
        Phase 2: Fine search around the best coarse angle
        
        Args:
            image_path: Path to source image
            coarse_step: Step size for coarse search (default: 2°)
            fine_step: Step size for fine search (default: 0.25°)
            fine_range: Range around best coarse angle for fine search (default: ±2°)
            progress_callback: Optional progress callback
            
        Returns:
            Optimization results
        """
        logger.info(f"Fast rotation optimization for {image_path}")
        
        # Load image
        try:
            source_image = Image.open(image_path)
            if source_image.mode not in ('RGB', 'RGBA'):
                source_image = source_image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise
        
        # Phase 1: Coarse search
        coarse_angles = np.arange(self.min_angle, self.max_angle + coarse_step, coarse_step)
        logger.info(f"Phase 1: Testing {len(coarse_angles)} coarse angles")
        
        coarse_scores = {}
        best_coarse_angle = 0.0
        best_coarse_score = -float('inf')
        
        for i, angle in enumerate(coarse_angles):
            candidate = self.rotate_and_crop(source_image, angle)
            score = await self.score_image(candidate)
            coarse_scores[angle] = score
            
            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_angle = angle
            
            if progress_callback:
                progress = (i + 1) / (len(coarse_angles) + 20) * 0.5  # First 50%
                await progress_callback(progress, angle, score)
        
        logger.info(f"Best coarse angle: {best_coarse_angle}° (score: {best_coarse_score:.2f})")
        
        # Phase 2: Fine search around best angle
        fine_min = max(best_coarse_angle - fine_range, self.min_angle)
        fine_max = min(best_coarse_angle + fine_range, self.max_angle)
        fine_angles = np.arange(fine_min, fine_max + fine_step, fine_step)
        
        # Remove angles already tested
        fine_angles = [a for a in fine_angles if a not in coarse_scores]
        
        logger.info(f"Phase 2: Testing {len(fine_angles)} fine angles around {best_coarse_angle}°")
        
        all_scores = coarse_scores.copy()
        best_angle = best_coarse_angle
        best_score = best_coarse_score
        
        for i, angle in enumerate(fine_angles):
            candidate = self.rotate_and_crop(source_image, angle)
            score = await self.score_image(candidate)
            all_scores[angle] = score
            
            if score > best_score:
                best_score = score
                best_angle = angle
            
            if progress_callback:
                progress = 0.5 + (i + 1) / len(fine_angles) * 0.5  # Last 50%
                await progress_callback(progress, angle, score)
        
        logger.info(f"Optimal rotation: {best_angle}° (score: {best_score:.2f})")
        
        # Generate final image
        final_image = self.rotate_and_crop(source_image, best_angle)
        
        return {
            'optimal_angle': float(best_angle),
            'optimal_score': float(best_score),
            'all_scores': all_scores,
            'final_image': final_image,
            'original_size': source_image.size,
            'final_size': final_image.size,
            'search_parameters': {
                'coarse_step': coarse_step,
                'fine_step': fine_step,
                'fine_range': fine_range,
                'total_candidates': len(all_scores)
            },
            'phase_results': {
                'coarse': {
                    'best_angle': best_coarse_angle,
                    'best_score': best_coarse_score,
                    'candidates': len(coarse_angles)
                },
                'fine': {
                    'best_angle': best_angle,
                    'best_score': best_score,
                    'candidates': len(fine_angles)
                }
            }
        }
    
    async def close(self):
        """Clean up HTTP client"""
        await self.client.aclose()


# Global instance
_rotation_optimizer = None

def get_integrated_rotation_optimizer() -> IntegratedRotationOptimizer:
    """Get or create the integrated rotation optimizer instance"""
    global _rotation_optimizer
    if not _rotation_optimizer:
        _rotation_optimizer = IntegratedRotationOptimizer()
    return _rotation_optimizer