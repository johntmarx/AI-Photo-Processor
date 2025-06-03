"""
Enhanced image processing utilities with advanced auto-balance and auto-levels
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
    
    def resize_for_ai_analysis(self, image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Resize image for AI analysis while maintaining aspect ratio
        """
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        if height > width:
            new_height = min(height, max_size)
            new_width = int(width * (new_height / height))
        else:
            new_width = min(width, max_size)
            new_height = int(height * (new_width / width))
        
        # Resize using high-quality interpolation
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
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
        
        final_height, final_width = cropped.shape[:2]
        final_aspect = final_width / final_height
        logger.info(f"Applied crop: ({x}, {y}) size {final_width}x{final_height} (aspect: {final_aspect:.3f})")
        return cropped
    
    def enhance_image(self, image: np.ndarray, color_analysis: ColorAnalysis) -> np.ndarray:
        """
        Apply subtle, iPhone-like auto enhancement
        """
        enhanced = image.copy()
        
        # Convert to float32 for processing
        if enhanced.dtype == np.uint16:
            enhanced = enhanced.astype(np.float32) / 65535.0
            bit_depth = 16
        else:
            enhanced = enhanced.astype(np.float32) / 255.0
            bit_depth = 8
        
        # Step 1: Very subtle auto-levels (iPhone-like)
        enhanced = self._apply_subtle_auto_levels(enhanced)
        
        # Step 2: Gentle white balance correction only if needed
        if color_analysis.white_balance_assessment != 'neutral':
            enhanced = self._apply_gentle_white_balance(enhanced, color_analysis.white_balance_assessment)
            logger.info(f"Applied gentle white balance for {color_analysis.white_balance_assessment} cast")
        
        # Step 3: Apply very subtle AI-recommended adjustments (clamped to iPhone-like range)
        brightness_adj = np.clip(color_analysis.brightness_adjustment_needed, -15, 15)
        contrast_adj = np.clip(color_analysis.contrast_adjustment_needed, -10, 10)
        
        if abs(brightness_adj) > 3:  # Only apply if meaningful adjustment
            enhanced = self._apply_gentle_curve_adjustment(enhanced, 'brightness', brightness_adj)
            logger.info(f"Applied subtle brightness: {brightness_adj}")
        
        if abs(contrast_adj) > 3:  # Only apply if meaningful adjustment
            enhanced = self._apply_gentle_curve_adjustment(enhanced, 'contrast', contrast_adj)
            logger.info(f"Applied subtle contrast: {contrast_adj}")
        
        # Step 4: Very mild local contrast enhancement (like iPhone's smart HDR)
        enhanced = self._apply_mild_local_contrast(enhanced)
        
        # Convert back to appropriate bit depth
        if bit_depth == 16:
            enhanced = np.clip(enhanced * 65535.0, 0, 65535).astype(np.uint16)
        else:
            enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _apply_auto_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sophisticated auto-levels adjustment
        """
        # Convert to LAB for better perceptual processing
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate percentile-based levels (more robust than min/max)
        p2 = np.percentile(l_channel, 2)
        p98 = np.percentile(l_channel, 98)
        
        # Apply levels adjustment to L channel
        scale = 255.0 / (p98 - p2) if p98 > p2 else 1.0
        l_channel = np.clip((l_channel - p2) * scale, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_channel
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Apply similar adjustment to RGB channels for color preservation
        for i in range(3):
            p2 = np.percentile(image[:, :, i], 1)
            p98 = np.percentile(image[:, :, i], 99)
            if p98 > p2:
                scale = 1.0 / (p98 - p2)
                adjusted[:, :, i] = np.clip((image[:, :, i] - p2) * scale, 0, 1)
        
        return adjusted
    
    def _apply_advanced_white_balance(self, image: np.ndarray, cast_type: str) -> np.ndarray:
        """
        Apply advanced white balance using multiple algorithms
        """
        # Method 1: Gray World with LAB color space
        result_gw = self._gray_world_lab(image)
        
        # Method 2: White Patch (excluding overexposed pixels)
        result_wp = self._white_patch_retinex(image)
        
        # Method 3: Automatic color equalization
        result_ace = self._automatic_color_equalization(image)
        
        # Blend results based on cast type
        if cast_type == 'cool':
            # For cool cast (common in pools), weight white patch more
            result = 0.2 * result_gw + 0.5 * result_wp + 0.3 * result_ace
        elif cast_type == 'warm':
            # For warm cast, weight gray world more
            result = 0.5 * result_gw + 0.2 * result_wp + 0.3 * result_ace
        else:
            # Equal weighting for unknown casts
            result = 0.33 * result_gw + 0.33 * result_wp + 0.34 * result_ace
        
        return np.clip(result, 0, 1)
    
    def _gray_world_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Gray World algorithm in LAB color space
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate average of a and b channels
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        
        # Shift a and b channels to neutral (128)
        a = np.clip(a - (avg_a - 128), 0, 255).astype(np.uint8)
        b = np.clip(b - (avg_b - 128), 0, 255).astype(np.uint8)
        
        # Merge and convert back
        lab_corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
        
        return corrected.astype(np.float32) / 255.0
    
    def _white_patch_retinex(self, image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        """
        White Patch Retinex with percentile to avoid overexposed pixels
        """
        corrected = image.copy()
        
        # Find the percentile values for each channel
        for i in range(3):
            channel = image[:, :, i]
            max_val = np.percentile(channel, percentile)
            if max_val > 0:
                corrected[:, :, i] = channel / max_val
        
        return np.clip(corrected, 0, 1)
    
    def _automatic_color_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Automatic Color Equalization (ACE) algorithm
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply adaptive histogram equalization to each channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Mild adjustment to a and b channels
        for i in [1, 2]:
            channel = lab[:, :, i].astype(np.float32)
            mean = np.mean(channel)
            std = np.std(channel)
            # Normalize and scale back
            normalized = (channel - mean) / (std + 1e-6)
            lab[:, :, i] = np.clip(128 + normalized * 20, 0, 255).astype(np.uint8)
        
        # Convert back
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return corrected.astype(np.float32) / 255.0
    
    def _apply_curve_adjustment(self, image: np.ndarray, adjustment_type: str, amount: float) -> np.ndarray:
        """
        Apply smooth curve adjustments for brightness/contrast
        """
        if adjustment_type == 'brightness':
            # S-curve for brightness
            amount_normalized = amount / 100.0
            # Create lookup table
            x = np.linspace(0, 1, 256)
            if amount > 0:
                # Brighten with lifted shadows
                y = np.power(x, 1.0 - amount_normalized * 0.5)
            else:
                # Darken with preserved highlights
                y = np.power(x, 1.0 - amount_normalized * 0.5)
            
            # Apply curve
            lut = (y * 255).astype(np.uint8)
            img_uint8 = (image * 255).astype(np.uint8)
            adjusted = cv2.LUT(img_uint8, lut).astype(np.float32) / 255.0
            
        elif adjustment_type == 'contrast':
            # S-curve for contrast
            amount_normalized = amount / 100.0
            factor = (1.0 + amount_normalized)
            
            # Apply sigmoid curve for smooth contrast
            adjusted = image.copy()
            adjusted = (adjusted - 0.5) * factor + 0.5
            adjusted = np.clip(adjusted, 0, 1)
        
        return adjusted
    
    def _apply_advanced_clahe(self, image: np.ndarray, contrast_level: str) -> np.ndarray:
        """
        Apply advanced CLAHE with parameters based on contrast assessment
        """
        # Convert to LAB
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Set CLAHE parameters based on contrast level
        if contrast_level == 'low':
            clip_limit = 3.0
            grid_size = (8, 8)
        elif contrast_level == 'high':
            clip_limit = 1.0
            grid_size = (16, 16)
        else:  # normal
            clip_limit = 2.0
            grid_size = (8, 8)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0
    
    def _enhance_vibrance(self, image: np.ndarray, vibrance: float = 0.15) -> np.ndarray:
        """
        Enhance color vibrance (saturate colors selectively)
        """
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation for less saturated pixels (selective vibrance)
        s_float = s.astype(np.float32) / 255.0
        # Apply more saturation to less saturated areas
        saturation_mask = 1.0 - s_float
        s_enhanced = s_float + (saturation_mask * vibrance)
        s_enhanced = np.clip(s_enhanced * 255, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s_enhanced, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
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
    
    def save_high_quality_jpeg(self, image: np.ndarray, output_path: str, quality: int = 95) -> str:
        """
        Save processed image as high-quality JPEG with proper color profile
        """
        # Ensure image is in the right format
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit with proper scaling
            image_8bit = np.clip(image / 256.0, 0, 255).astype(np.uint8)
        else:
            image_8bit = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        if len(image_8bit.shape) == 3:
            pil_image = Image.fromarray(image_8bit)
        else:
            pil_image = Image.fromarray(image_8bit)
        
        # Save with high quality and proper color profile
        pil_image.save(output_path, "JPEG", quality=quality, optimize=True, progressive=True)
        logger.info(f"Saved high-quality image to: {output_path}")
        
        return output_path
    
    def _apply_subtle_auto_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Apply very gentle auto-levels like iPhone Photos
        """
        # Use more conservative percentiles to avoid clipping
        enhanced = image.copy()
        
        # Work on luminance for gentle adjustment
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Conservative percentiles (iPhone doesn't clip as aggressively)
        p5 = np.percentile(gray, 5)   # More conservative than 2%
        p95 = np.percentile(gray, 95)  # More conservative than 98%
        
        # Only apply if there's meaningful dynamic range to improve
        if p95 - p5 < 180:  # Image lacks contrast
            # Very gentle stretch
            for i in range(3):
                p_low = np.percentile(enhanced[:, :, i], 3)
                p_high = np.percentile(enhanced[:, :, i], 97)
                if p_high > p_low:
                    scale = 0.98 / (p_high - p_low)  # Gentle scale factor
                    enhanced[:, :, i] = np.clip((enhanced[:, :, i] - p_low) * scale + 0.01, 0, 1)
        
        return enhanced
    
    def _apply_gentle_white_balance(self, image: np.ndarray, cast_type: str) -> np.ndarray:
        """
        Apply very subtle white balance correction like iPhone
        """
        # iPhone uses very gentle corrections
        correction_strength = 0.3  # Much weaker than before
        
        if cast_type == 'cool':
            # Gently warm up cool images
            enhanced = image.copy()
            # Slightly boost red and reduce blue
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 + correction_strength * 0.1), 0, 1)  # Red
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 - correction_strength * 0.05), 0, 1)  # Blue
        elif cast_type == 'warm':
            # Gently cool down warm images
            enhanced = image.copy()
            # Slightly boost blue and reduce red
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 - correction_strength * 0.05), 0, 1)  # Red
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 + correction_strength * 0.1), 0, 1)  # Blue
        else:
            enhanced = image
            
        return enhanced
    
    def _apply_gentle_curve_adjustment(self, image: np.ndarray, adjustment_type: str, value: float) -> np.ndarray:
        """
        Apply gentle curve adjustments like iPhone Photos
        """
        enhanced = image.copy()
        strength = abs(value) / 100.0  # Convert to 0-1 range, very gentle
        
        if adjustment_type == 'brightness':
            if value > 0:
                # Gentle brightening - lift shadows more than highlights
                enhanced = enhanced + (strength * 0.5 * (1 - enhanced))  # Gentle lift
            else:
                # Gentle darkening
                enhanced = enhanced * (1 - strength * 0.3)  # Gentle darken
        elif adjustment_type == 'contrast':
            if value > 0:
                # Gentle contrast increase
                enhanced = np.clip((enhanced - 0.5) * (1 + strength * 0.4) + 0.5, 0, 1)
            else:
                # Gentle contrast decrease
                enhanced = np.clip((enhanced - 0.5) * (1 - strength * 0.3) + 0.5, 0, 1)
        
        return enhanced
    
    def _apply_mild_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply very mild local contrast enhancement like iPhone's Smart HDR
        """
        # Convert to LAB for perceptual processing
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Very mild CLAHE (much gentler than before)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Very conservative
        l_enhanced = clahe.apply(l_channel)
        
        # Blend with original (iPhone-like subtlety)
        blend_factor = 0.3  # Only 30% of the effect
        l_final = cv2.addWeighted(l_channel, 1 - blend_factor, l_enhanced, blend_factor, 0)
        
        lab[:, :, 0] = l_final
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32) / 255.0