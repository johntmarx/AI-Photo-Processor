"""
Image processing utilities for RAW files and JPEG enhancement
"""
import os
import tempfile
import rawpy
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
import logging
from typing import Tuple, Optional
from schemas import BoundingBox, ColorAnalysis
from skimage import exposure, filters
from colorthief import ColorThief

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.temp_dir = "/app/temp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def convert_raw_to_rgb(self, raw_path: str) -> Tuple[np.ndarray, dict]:
        """
        Convert RAW file to RGB array with metadata
        """
        try:
            with rawpy.imread(raw_path) as raw:
                # Extract metadata
                metadata = {
                    'camera_make': getattr(raw, 'camera_make', 'Unknown'),
                    'camera_model': getattr(raw, 'camera_model', 'Unknown'),
                    'iso': getattr(raw, 'camera_iso', None),
                    'exposure_time': getattr(raw, 'camera_exposure_time', None),
                    'aperture': getattr(raw, 'camera_aperture', None),
                    'focal_length': getattr(raw, 'camera_focal_length', None),
                    'width': raw.sizes.width,
                    'height': raw.sizes.height,
                }
                
                # Process RAW with optimal settings
                rgb = raw.postprocess(
                    gamma=(2.2, 4.5),  # Gamma correction
                    no_auto_bright=True,  # Preserve exposure
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,  # 16-bit output for better processing
                    use_camera_wb=True,  # Use camera white balance
                    half_size=False,  # Full resolution
                    four_color_rgb=False,
                    dcb_enhance=True,
                    fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full
                )
                
                return rgb, metadata
                
        except Exception as e:
            logger.error(f"Error processing RAW file {raw_path}: {e}")
            raise
    
    def resize_for_ai_analysis(self, image: np.ndarray, max_size: int = 896) -> np.ndarray:
        """
        Resize image for AI analysis with letterboxing to preserve entire image
        """
        height, width = image.shape[:2]
        
        # Calculate scale factor to fit image within max_size square
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize using high-quality interpolation
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create letterboxed image (black background)
        letterboxed = np.zeros((max_size, max_size, 3), dtype=resized.dtype)
        
        # Calculate position to center the resized image
        y_offset = (max_size - new_height) // 2
        x_offset = (max_size - new_width) // 2
        
        # Place resized image in center
        letterboxed[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return letterboxed
    
    def save_temp_image_for_ai(self, image: np.ndarray) -> str:
        """
        Save image to temp file for AI analysis
        """
        # Convert to 8-bit for AI processing
        if image.dtype == np.uint16:
            image_8bit = (image / 256).astype(np.uint8)
        else:
            image_8bit = image
        
        # Convert BGR to RGB if needed
        if len(image_8bit.shape) == 3 and image_8bit.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_8bit
        
        # Save as temporary JPEG
        temp_path = os.path.join(self.temp_dir, f"ai_analysis_{os.getpid()}_{np.random.randint(1000, 9999)}.jpg")
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(temp_path, "JPEG", quality=85)
        
        return temp_path
    
    def apply_rotation(self, image: np.ndarray, rotation_degrees: float) -> np.ndarray:
        """
        Apply high-quality rotation to image
        """
        if abs(rotation_degrees) < 0.1:  # Skip if rotation is negligible
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
        
        # Calculate new image bounds to avoid cropping
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new bounds
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with high-quality interpolation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_LANCZOS4, 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        
        logger.info(f"Applied rotation: {rotation_degrees:.1f} degrees")
        return rotated
    
    def apply_smart_crop(self, image: np.ndarray, crop_box: BoundingBox) -> np.ndarray:
        """
        Apply intelligent cropping based on AI suggestions
        """
        height, width = image.shape[:2]
        
        # Convert percentage coordinates to pixels
        x = int((crop_box.x / 100) * width)
        y = int((crop_box.y / 100) * height)
        crop_width = int((crop_box.width / 100) * width)
        crop_height = int((crop_box.height / 100) * height)
        
        # Ensure crop stays within image bounds
        x = max(0, min(x, width - crop_width))
        y = max(0, min(y, height - crop_height))
        crop_width = min(crop_width, width - x)
        crop_height = min(crop_height, height - y)
        
        # Apply crop
        cropped = image[y:y+crop_height, x:x+crop_width]
        
        logger.info(f"Applied crop: ({x}, {y}) size {crop_width}x{crop_height}")
        return cropped
    
    def enhance_image(self, image: np.ndarray, color_analysis: ColorAnalysis) -> np.ndarray:
        """
        Apply smart color correction and enhancement
        """
        enhanced = image.copy()
        
        # Convert to float for processing
        if enhanced.dtype == np.uint16:
            enhanced = enhanced.astype(np.float32) / 65535.0
        elif enhanced.dtype == np.uint8:
            enhanced = enhanced.astype(np.float32) / 255.0
        
        # Apply brightness adjustment
        if color_analysis.brightness_adjustment_needed != 0:
            brightness_factor = 1.0 + (color_analysis.brightness_adjustment_needed / 100.0)
            enhanced = np.clip(enhanced * brightness_factor, 0, 1)
            logger.info(f"Applied brightness adjustment: {color_analysis.brightness_adjustment_needed}")
        
        # Apply contrast adjustment
        if color_analysis.contrast_adjustment_needed != 0:
            contrast_factor = 1.0 + (color_analysis.contrast_adjustment_needed / 100.0)
            enhanced = np.clip((enhanced - 0.5) * contrast_factor + 0.5, 0, 1)
            logger.info(f"Applied contrast adjustment: {color_analysis.contrast_adjustment_needed}")
        
        # Auto white balance correction if needed
        if color_analysis.white_balance_assessment != 'neutral':
            enhanced = self._auto_white_balance(enhanced)
            logger.info("Applied auto white balance")
        
        # Adaptive histogram equalization for better local contrast
        if color_analysis.contrast_level == 'low':
            enhanced = self._apply_clahe(enhanced)
            logger.info("Applied CLAHE for low contrast")
        
        # Convert back to appropriate bit depth
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction
        """
        # Gray world assumption
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        # Calculate correction factors
        gray_mean = (mean_r + mean_g + mean_b) / 3
        
        correction_r = gray_mean / mean_r if mean_r > 0 else 1
        correction_g = gray_mean / mean_g if mean_g > 0 else 1
        correction_b = gray_mean / mean_b if mean_b > 0 else 1
        
        # Apply corrections
        corrected = image.copy()
        corrected[:, :, 0] *= correction_r
        corrected[:, :, 1] *= correction_g
        corrected[:, :, 2] *= correction_b
        
        return np.clip(corrected, 0, 1)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization
        """
        # Convert to LAB color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0
    
    def save_high_quality_jpeg(self, image: np.ndarray, output_path: str, quality: int = 100) -> str:
        """
        Save processed image as high-quality JPEG
        """
        # Ensure image is in the right format
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            image_8bit = (image / 256).astype(np.uint8)
        else:
            image_8bit = image
        
        # Convert to PIL Image
        if len(image_8bit.shape) == 3:
            pil_image = Image.fromarray(image_8bit)
        else:
            pil_image = Image.fromarray(image_8bit)
        
        # Save with maximum quality
        pil_image.save(output_path, "JPEG", quality=quality, optimize=True)
        logger.info(f"Saved high-quality image to: {output_path}")
        
        return output_path
    
    def cleanup_temp_files(self):
        """
        Clean up temporary files
        """
        try:
            for file in os.listdir(self.temp_dir):
                if file.startswith("ai_analysis_"):
                    os.remove(os.path.join(self.temp_dir, file))
        except Exception as e:
            logger.warning(f"Error cleaning temp files: {e}")
    
    def detect_blur(self, image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
        """
        Detect if image is blurry using Laplacian variance method
        Returns: (is_blurry, variance_score)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize for consistent detection across different resolutions
        height, width = gray.shape
        if width > 1000:
            scale = 1000 / width
            new_width = 1000
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Determine if blurry
        is_blurry = variance < threshold
        
        logger.info(f"Blur detection - Variance: {variance:.2f}, Threshold: {threshold}, Is Blurry: {is_blurry}")
        
        return is_blurry, variance
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get basic image information
        """
        height, width = image.shape[:2]
        megapixels = (height * width) / 1_000_000
        
        # Determine orientation
        if width > height * 1.1:
            orientation = "landscape"
        elif height > width * 1.1:
            orientation = "portrait"
        else:
            orientation = "square"
        
        return {
            "width": width,
            "height": height,
            "megapixels": round(megapixels, 1),
            "orientation": orientation
        }