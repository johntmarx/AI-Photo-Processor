"""
Simple image loader for API routes that handles RAW files
No torch dependencies
"""

import rawpy
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def load_image_simple(image_path: Union[str, Path]) -> Image.Image:
    """
    Load image from file, handling RAW formats.
    Simple version without torch dependencies.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image in RGB format
    """
    image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Handle RAW formats
    raw_extensions = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.orf', '.rw2', '.raf'}
    
    if image_path.suffix.lower() in raw_extensions:
        logger.info(f"Loading RAW image: {image_path}")
        with rawpy.imread(str(image_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=False,
                output_bps=8
            )
            return Image.fromarray(rgb)
    else:
        # Regular image formats
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'RGBA'):
            if img.mode == 'CMYK':
                # Handle CMYK conversion
                img = img.convert('RGB')
            elif img.mode in ('L', 'LA', 'P'):
                # Grayscale, grayscale with alpha, or palette
                img = img.convert('RGB')
            else:
                # For any other mode, try direct conversion
                img = img.convert('RGB')
        elif img.mode == 'RGBA':
            # Convert RGBA to RGB with white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        return img