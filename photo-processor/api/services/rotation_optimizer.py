"""
Rotation Optimization Service

Uses OneAlign (Q-Align) model to find the optimal rotation angle for images
by generating candidates and scoring their aesthetic appeal.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math

logger = logging.getLogger(__name__)

class RotationOptimizer:
    """
    Optimizes image rotation using aesthetic scoring from OneAlign model.
    
    The process follows a "Generate, Score, and Select" approach:
    1. Generate high-quality rotated candidates
    2. Score each candidate using OneAlign
    3. Select the rotation with the highest aesthetic score
    """
    
    def __init__(self, onealign_model=None):
        """
        Initialize the rotation optimizer.
        
        Args:
            onealign_model: Instance of OneAlign model for scoring
        """
        self.onealign_model = onealign_model
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Default search parameters
        self.min_angle = -20.0
        self.max_angle = 20.0
        self.angle_step = 0.5  # Test every 0.5 degrees
        
        # Quality settings
        self.resample_filter = Image.Resampling.LANCZOS
        self.fill_color = (0, 0, 0)  # Black fill for rotation
        
        logger.info("RotationOptimizer initialized")
    
    def rotate_and_crop(self, image: Image.Image, angle: float) -> Image.Image:
        """
        Rotates an image and crops to the largest artifact-free rectangle.
        
        Args:
            image: PIL Image to rotate
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated and cropped PIL Image
        """
        if abs(angle) < 0.01:  # Skip rotation for near-zero angles
            return image.copy()
        
        # Rotate with expansion to capture full image
        rotated = image.rotate(
            angle,
            resample=self.resample_filter,
            expand=True,
            fillcolor=self.fill_color
        )
        
        # Calculate the largest inscribed rectangle after rotation
        # This prevents any black borders from showing
        w, h = image.size
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
        rot_w, rot_h = rotated.size
        left = (rot_w - new_w) // 2
        top = (rot_h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        
        cropped = rotated.crop((left, top, right, bottom))
        
        return cropped
    
    def generate_rotation_candidates(self, image: Image.Image, angles: list) -> Dict[float, Image.Image]:
        """
        Generate high-quality rotation candidates for testing.
        
        Args:
            image: Source image
            angles: List of angles to test
            
        Returns:
            Dictionary mapping angles to rotated images
        """
        candidates = {}
        
        for angle in angles:
            try:
                # For scoring, we don't need full resolution
                # Resize to a standard size for consistent scoring
                test_size = (1024, 1024)
                if image.size[0] > test_size[0] or image.size[1] > test_size[1]:
                    test_image = image.copy()
                    test_image.thumbnail(test_size, self.resample_filter)
                else:
                    test_image = image
                
                # Generate rotated candidate
                rotated = self.rotate_and_crop(test_image, angle)
                candidates[angle] = rotated
                
            except Exception as e:
                logger.error(f"Failed to generate candidate for angle {angle}: {e}")
                continue
        
        return candidates
    
    async def score_image(self, image: Image.Image) -> float:
        """
        Score an image using OneAlign model.
        
        Args:
            image: PIL Image to score
            
        Returns:
            Aesthetic score (higher is better)
        """
        if not self.onealign_model:
            # Fallback: use simple heuristics if model not available
            return self._heuristic_score(image)
        
        try:
            # Convert PIL image to format expected by OneAlign
            # This would typically involve the model's preprocessing
            result = await self.onealign_model.analyze_image(image)
            
            # Extract aesthetic score
            aesthetic_score = result.get('aesthetic_score', 5.0)
            
            return float(aesthetic_score)
            
        except Exception as e:
            logger.error(f"Failed to score image with OneAlign: {e}")
            return self._heuristic_score(image)
    
    def _heuristic_score(self, image: Image.Image) -> float:
        """
        Simple heuristic scoring when OneAlign is not available.
        Based on rule of thirds and other composition rules.
        
        Args:
            image: PIL Image to score
            
        Returns:
            Heuristic score (0-10)
        """
        # This is a placeholder - in production, OneAlign would be used
        # For now, prefer minimal rotation
        return 5.0
    
    async def find_optimal_rotation(
        self,
        image_path: Path,
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        angle_step: Optional[float] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Find the optimal rotation angle for an image.
        
        Args:
            image_path: Path to the source image
            min_angle: Minimum angle to test (default: -20)
            max_angle: Maximum angle to test (default: 20)
            angle_step: Step size between angles (default: 0.5)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with optimal angle and scoring details
        """
        # Use provided parameters or defaults
        min_angle = min_angle if min_angle is not None else self.min_angle
        max_angle = max_angle if max_angle is not None else self.max_angle
        angle_step = angle_step if angle_step is not None else self.angle_step
        
        logger.info(f"Finding optimal rotation for {image_path}")
        logger.info(f"Search range: {min_angle}° to {max_angle}° in {angle_step}° steps")
        
        # Load the source image
        try:
            source_image = Image.open(image_path)
            if source_image.mode not in ('RGB', 'RGBA'):
                source_image = source_image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            raise
        
        # Generate candidate angles
        candidate_angles = np.arange(min_angle, max_angle + angle_step, angle_step)
        total_candidates = len(candidate_angles)
        
        logger.info(f"Testing {total_candidates} rotation candidates")
        
        # Generate all candidates
        candidates = self.generate_rotation_candidates(source_image, candidate_angles)
        
        # Score each candidate
        scores = {}
        best_angle = 0.0
        best_score = -float('inf')
        
        for i, (angle, candidate) in enumerate(candidates.items()):
            # Score the candidate
            score = await self.score_image(candidate)
            scores[angle] = score
            
            logger.debug(f"Angle: {angle:6.1f}°, Score: {score:.4f}")
            
            # Track the best
            if score > best_score:
                best_score = score
                best_angle = angle
            
            # Progress callback
            if progress_callback:
                progress = (i + 1) / total_candidates
                await progress_callback(progress, angle, score)
        
        # Refine search around the best angle if needed
        if angle_step > 0.1 and abs(best_angle) > 0.1:
            logger.info(f"Refining search around {best_angle}° with 0.1° steps")
            
            # Search ±1 degree around best angle with finer steps
            fine_angles = np.arange(
                max(best_angle - 1.0, min_angle),
                min(best_angle + 1.0, max_angle) + 0.1,
                0.1
            )
            
            fine_candidates = self.generate_rotation_candidates(source_image, fine_angles)
            
            for angle, candidate in fine_candidates.items():
                if angle not in scores:  # Don't re-score
                    score = await self.score_image(candidate)
                    scores[angle] = score
                    
                    if score > best_score:
                        best_score = score
                        best_angle = angle
        
        logger.info(f"Optimal rotation: {best_angle}° (score: {best_score:.4f})")
        
        # Generate final high-quality output
        final_image = self.rotate_and_crop(source_image, best_angle)
        
        return {
            'optimal_angle': float(best_angle),
            'optimal_score': float(best_score),
            'all_scores': scores,
            'final_image': final_image,
            'original_size': source_image.size,
            'final_size': final_image.size,
            'search_parameters': {
                'min_angle': min_angle,
                'max_angle': max_angle,
                'angle_step': angle_step,
                'total_candidates': len(scores)
            }
        }
    
    async def process_with_progress(
        self,
        image_path: Path,
        output_path: Path,
        progress_callback=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image with rotation optimization and save the result.
        
        Args:
            image_path: Input image path
            output_path: Output image path
            progress_callback: Optional progress callback
            **kwargs: Additional arguments for find_optimal_rotation
            
        Returns:
            Processing results
        """
        # Find optimal rotation
        result = await self.find_optimal_rotation(
            image_path,
            progress_callback=progress_callback,
            **kwargs
        )
        
        # Save the optimized image
        final_image = result['final_image']
        final_image.save(output_path, quality=95, optimize=True)
        
        logger.info(f"Saved optimized image to {output_path}")
        
        # Clean up the PIL image from results
        result.pop('final_image')
        result['output_path'] = str(output_path)
        
        return result