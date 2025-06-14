"""
Recipe Processor Service

Applies recipe operations to images, including support for intelligent enhancement.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import json

from services.crop_optimizer import CropOptimizer
from services.enhance_optimizer import EnhanceOptimizer
from services.intelligent_enhancer import IntelligentEnhancer
try:
    from services.rotation_optimizer import RotationOptimizer
except ImportError:
    # Use the integrated version if available
    try:
        from services.rotation_optimizer_integrated import RotationOptimizer
    except ImportError:
        RotationOptimizer = None

logger = logging.getLogger(__name__)


class RecipeProcessor:
    """
    Processes images according to recipe operations.
    Supports both traditional and intelligent enhancement modes.
    """
    
    def __init__(self):
        """Initialize the recipe processor with required optimizers"""
        self.crop_optimizer = CropOptimizer()
        self.enhance_optimizer = EnhanceOptimizer()
        self.intelligent_enhancer = IntelligentEnhancer()
        
        if RotationOptimizer:
            self.rotation_optimizer = RotationOptimizer()
        else:
            self.rotation_optimizer = None
            logger.warning("RotationOptimizer not available - rotation features will be limited")
        
        logger.info("RecipeProcessor initialized with available optimizers")
    
    async def process_image_with_recipe(
        self,
        image_path: Path,
        recipe: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Process an image according to a recipe.
        
        Args:
            image_path: Path to the input image
            recipe: Recipe dictionary with operations
            output_path: Optional output path (defaults to processed directory)
            
        Returns:
            Processing result with output path and metadata
        """
        try:
            # Load the image
            image = Image.open(image_path)
            original_format = image.format
            
            logger.info(f"Processing image {image_path.name} with recipe {recipe.get('name', 'unnamed')}")
            
            # Apply each operation in sequence
            operations = recipe.get('operations', recipe.get('steps', []))
            
            for i, operation in enumerate(operations):
                if not operation.get('enabled', True):
                    continue
                    
                op_type = operation.get('type', operation.get('operation'))
                params = operation.get('params', operation.get('parameters', {}))
                
                logger.info(f"Applying operation {i+1}/{len(operations)}: {op_type}")
                
                # Apply the operation
                image = await self._apply_operation(image, op_type, params)
            
            # Save the result
            if output_path is None:
                output_dir = image_path.parent.parent / 'processed'
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{image_path.stem}_processed.jpg"
            
            # Save with high quality
            save_kwargs = {
                'quality': 95,
                'optimize': True,
                'progressive': True
            }
            
            if original_format == 'JPEG':
                save_kwargs['format'] = 'JPEG'
            
            image.save(output_path, **save_kwargs)
            
            logger.info(f"Saved processed image to {output_path}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'operations_applied': len(operations),
                'dimensions': image.size,
                'format': image.format or 'JPEG'
            }
            
        except Exception as e:
            logger.error(f"Failed to process image with recipe: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _apply_operation(
        self,
        image: Image.Image,
        op_type: str,
        params: Dict[str, Any]
    ) -> Image.Image:
        """Apply a single operation to an image"""
        
        if op_type == 'rotate':
            # Handle rotation
            mode = params.get('mode', 'manual')
            if mode == 'auto':
                # Auto-detect rotation would go here
                angle = 0  # Placeholder
            else:
                angle = params.get('angle', 0)
            
            if abs(angle) > 0.01:
                image = image.rotate(-angle, expand=True, fillcolor='white')
                logger.info(f"Rotated by {angle} degrees")
        
        elif op_type == 'crop':
            # Handle cropping
            aspect_ratio = params.get('aspect_ratio')
            crop_box = params.get('crop_box')
            mode = params.get('mode', 'manual')
            
            if crop_box:
                # Use specific crop box
                x = int(crop_box.get('x', 0))
                y = int(crop_box.get('y', 0))
                width = int(crop_box.get('width', image.width))
                height = int(crop_box.get('height', image.height))
                image = image.crop((x, y, x + width, y + height))
                logger.info(f"Cropped to box: {x},{y} {width}x{height}")
            elif aspect_ratio and aspect_ratio != 'auto':
                # Center crop to aspect ratio
                image = self._crop_to_aspect_ratio(image, aspect_ratio)
                logger.info(f"Cropped to aspect ratio: {aspect_ratio}")
        
        elif op_type == 'enhance':
            # Handle enhancement
            mode = params.get('mode', 'manual')
            
            if mode == 'intelligent':
                # Use modern intelligent enhancer
                strength = params.get('strength', 1.0)
                
                # Ensure RGB mode for enhancement
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                image = self.intelligent_enhancer.enhance_image(
                    image,
                    strength
                )
                logger.info(f"Applied modern intelligent enhancement with strength {strength}")
            else:
                # Use traditional enhancement
                image = self.enhance_optimizer.apply_enhancements(
                    image,
                    brightness=params.get('brightness', 0),
                    contrast=params.get('contrast', 0),
                    saturation=params.get('saturation', 0),
                    sharpness=params.get('sharpness', 0),
                    denoise=params.get('denoise', 0)
                )
                logger.info("Applied traditional enhancement")
        
        elif op_type == 'resize':
            # Handle resizing
            max_width = params.get('maxWidth', image.width)
            max_height = params.get('maxHeight', image.height)
            
            if image.width > max_width or image.height > max_height:
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized to fit within {max_width}x{max_height}")
        
        else:
            logger.warning(f"Unknown operation type: {op_type}")
        
        return image
    
    def _crop_to_aspect_ratio(self, image: Image.Image, aspect_ratio: str) -> Image.Image:
        """Center crop image to specified aspect ratio"""
        # Parse aspect ratio
        if ':' in aspect_ratio:
            w_ratio, h_ratio = map(float, aspect_ratio.split(':'))
            target_ratio = w_ratio / h_ratio
        else:
            target_ratio = 1.0  # Default to square
        
        # Calculate crop dimensions
        img_ratio = image.width / image.height
        
        if img_ratio > target_ratio:
            # Image is wider than target ratio
            new_width = int(image.height * target_ratio)
            new_height = image.height
            x = (image.width - new_width) // 2
            y = 0
        else:
            # Image is taller than target ratio
            new_width = image.width
            new_height = int(image.width / target_ratio)
            x = 0
            y = (image.height - new_height) // 2
        
        return image.crop((x, y, x + new_width, y + new_height))


# Singleton instance
recipe_processor = RecipeProcessor()