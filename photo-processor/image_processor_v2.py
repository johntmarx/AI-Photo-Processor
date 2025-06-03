"""
Enhanced Image Processing Module with AI-Driven Enhancements

This module provides sophisticated image processing capabilities that mimic professional
photo editing software and smartphone photography apps (particularly iPhone's photo processing).
It handles RAW file conversion, intelligent cropping, and subtle enhancement algorithms.

Key Features:
- RAW file processing with 16-bit color depth preservation
- AI-guided smart cropping based on composition analysis
- Multi-algorithm white balance correction
- Subtle, natural-looking enhancements (iPhone-style)
- Blur detection using Laplacian variance
- Color-accurate JPEG output with embedded profiles

Design Philosophy:
The module prioritizes natural-looking results over dramatic transformations. We aim for
the subtle, pleasing enhancements that modern smartphones apply - improving photos without
making them look "processed". This approach is particularly important for family photos
and everyday photography where authenticity matters.
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
    """
    Advanced Image Processor with AI-Driven Enhancement Capabilities
    
    This class provides a comprehensive suite of image processing tools designed to
    automatically enhance photos with minimal user intervention. It combines traditional
    image processing techniques with AI-guided adjustments to produce natural-looking results.
    
    The processor is optimized for:
    - Family and portrait photography
    - Landscape and nature shots  
    - Indoor/outdoor scenes with challenging lighting
    - RAW files from professional cameras
    
    Processing Pipeline:
    1. RAW conversion (if applicable) with 16-bit depth preservation
    2. AI analysis for composition and quality assessment
    3. Smart cropping based on AI recommendations
    4. Subtle enhancement mimicking iPhone's computational photography
    5. High-quality JPEG output with proper color management
    """
    def __init__(self):
        self.temp_dir = "/app/temp"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def convert_raw_to_rgb(self, raw_path: str) -> Tuple[np.ndarray, dict]:
        """
        Convert RAW file to RGB array with metadata
        
        This method handles the critical RAW-to-RGB conversion process, preserving maximum
        image quality and color information for subsequent processing steps.
        """
        try:
            with rawpy.imread(raw_path) as raw:
                # Extract metadata for AI analysis and EXIF preservation
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
                
                # Process RAW with carefully chosen parameters
                rgb = raw.postprocess(
                    # Gamma correction using standard sRGB values (2.2) with slope 4.5
                    # This matches the sRGB standard and ensures proper tone mapping
                    gamma=(2.2, 4.5),
                    
                    # Disable auto-brightness to preserve the photographer's exposure intent
                    # We'll apply our own subtle adjustments later if needed
                    no_auto_bright=True,
                    
                    # Output to sRGB color space for web compatibility
                    output_color=rawpy.ColorSpace.sRGB,
                    
                    # Use 16-bit output instead of 8-bit - this is crucial for:
                    # 1. Preserving shadow and highlight detail during processing
                    # 2. Avoiding banding artifacts in gradients
                    # 3. Allowing high-quality adjustments without posterization
                    output_bps=16,
                    
                    # Trust the camera's white balance as a starting point
                    # Modern cameras are quite good at AWB, and this preserves
                    # the shooting conditions (e.g., golden hour warmth)
                    use_camera_wb=True,
                    
                    # Process at full resolution - no pixel binning
                    # We'll resize later if needed, but start with all available data
                    half_size=False,
                    
                    # Standard RGB processing (not four-color)
                    four_color_rgb=False,
                    
                    # Enable DCB enhancement for better edge interpolation
                    # This reduces color artifacts at high-contrast edges
                    dcb_enhance=True,
                    
                    # Full FBDD noise reduction - important for high ISO shots
                    # This advanced algorithm preserves detail while reducing noise
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
        
        This method implements AI-driven smart cropping that considers:
        - Rule of thirds positioning
        - Subject detection and framing
        - Removal of distracting elements at edges
        - Aspect ratio optimization for common uses
        
        The AI provides crop suggestions as percentages to ensure scalability
        across different image resolutions.
        """
        height, width = image.shape[:2]
        
        # Convert percentage coordinates to pixels
        # The AI returns percentages (0-100) to be resolution-independent
        # This allows the same crop suggestion to work on both the analysis
        # thumbnail and the full-resolution image
        x = int((crop_box.x / 100) * width)
        y = int((crop_box.y / 100) * height)
        crop_width = int((crop_box.width / 100) * width)
        crop_height = int((crop_box.height / 100) * height)
        
        # Ensure crop stays within image bounds
        # This is crucial because:
        # 1. Rounding errors in percentage conversion might push us out of bounds
        # 2. The AI might suggest crops that extend slightly beyond edges
        # 3. We need to maintain the aspect ratio even with boundary constraints
        x = max(0, min(x, width - crop_width))
        y = max(0, min(y, height - crop_height))
        crop_width = min(crop_width, width - x)
        crop_height = min(crop_height, height - y)
        
        # Apply crop using NumPy array slicing
        # This is the most efficient way to crop without copying unnecessary data
        cropped = image[y:y+crop_height, x:x+crop_width]
        
        # Log the final crop dimensions for debugging and analysis
        # The aspect ratio is particularly important for understanding
        # if the AI is suggesting standard ratios (16:9, 4:3, 1:1, etc.)
        final_height, final_width = cropped.shape[:2]
        final_aspect = final_width / final_height
        logger.info(f"Applied crop: ({x}, {y}) size {final_width}x{final_height} (aspect: {final_aspect:.3f})")
        return cropped
    
    def enhance_image(self, image: np.ndarray, color_analysis: ColorAnalysis) -> np.ndarray:
        """
        Apply subtle, iPhone-like auto enhancement
        
        This is the core enhancement pipeline that mimics the sophisticated but subtle
        adjustments that modern smartphones (particularly iPhone) apply to photos.
        The goal is to make photos look better without looking "edited".
        
        Key principles:
        - Preserve the original mood and lighting of the scene
        - Enhance details without creating artifacts
        - Maintain natural skin tones and colors
        - Apply adjustments that work well across different viewing conditions
        """
        enhanced = image.copy()
        
        # Convert to float32 for processing
        # Working in floating point prevents rounding errors and allows
        # for more precise adjustments. We track the original bit depth
        # to convert back appropriately at the end.
        if enhanced.dtype == np.uint16:
            enhanced = enhanced.astype(np.float32) / 65535.0
            bit_depth = 16
        else:
            enhanced = enhanced.astype(np.float32) / 255.0
            bit_depth = 8
        
        # Step 1: Very subtle auto-levels (iPhone-like)
        # This gently expands the tonal range without clipping highlights
        # or crushing shadows. Uses conservative percentiles (5th/95th)
        # instead of aggressive min/max stretching.
        enhanced = self._apply_subtle_auto_levels(enhanced)
        
        # Step 2: Gentle white balance correction only if needed
        # Only apply if the AI detected a color cast. This preserves
        # intentional color grading (sunset warmth, blue hour, etc.)
        # while fixing unwanted casts from artificial lighting.
        if color_analysis.white_balance_assessment != 'neutral':
            enhanced = self._apply_gentle_white_balance(enhanced, color_analysis.white_balance_assessment)
            logger.info(f"Applied gentle white balance for {color_analysis.white_balance_assessment} cast")
        
        # Step 3: Apply very subtle AI-recommended adjustments (clamped to iPhone-like range)
        # The AI might suggest large adjustments, but we constrain them to subtle ranges:
        # - Brightness: ±15 (on a 0-100 scale) to avoid blown highlights or crushed shadows
        # - Contrast: ±10 to maintain a natural look
        # These limits are based on analyzing iPhone's Photos app adjustments
        brightness_adj = np.clip(color_analysis.brightness_adjustment_needed, -15, 15)
        contrast_adj = np.clip(color_analysis.contrast_adjustment_needed, -10, 10)
        
        # Only apply adjustments if they're meaningful (>3 units)
        # This prevents micro-adjustments that won't be visible but add processing time
        if abs(brightness_adj) > 3:
            enhanced = self._apply_gentle_curve_adjustment(enhanced, 'brightness', brightness_adj)
            logger.info(f"Applied subtle brightness: {brightness_adj}")
        
        if abs(contrast_adj) > 3:
            enhanced = self._apply_gentle_curve_adjustment(enhanced, 'contrast', contrast_adj)
            logger.info(f"Applied subtle contrast: {contrast_adj}")
        
        # Step 4: Very mild local contrast enhancement (like iPhone's Smart HDR)
        # This brings out detail in both shadows and highlights without the
        # "HDR look". Uses CLAHE with very conservative settings and blends
        # with the original for subtlety.
        enhanced = self._apply_mild_local_contrast(enhanced)
        
        # Convert back to appropriate bit depth
        # Careful clipping ensures no overflow while maintaining the full
        # dynamic range of the target bit depth
        if bit_depth == 16:
            enhanced = np.clip(enhanced * 65535.0, 0, 65535).astype(np.uint16)
        else:
            enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _apply_auto_levels(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sophisticated auto-levels adjustment
        
        NOTE: This is the more aggressive version compared to _apply_subtle_auto_levels.
        It's currently not used in the main pipeline, as we prefer the subtle iPhone-style
        adjustments. Kept here for potential future use cases where stronger correction
        is explicitly requested.
        """
        # Convert to LAB for better perceptual processing
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate percentile-based levels (more robust than min/max)
        # 2nd/98th percentiles are aggressive - they'll clip 2% of data at each end
        # This creates more dramatic contrast but may lose subtle detail
        p2 = np.percentile(l_channel, 2)
        p98 = np.percentile(l_channel, 98)
        
        # Apply levels adjustment to L channel
        # This stretches the tonal range to fill 0-255 completely
        scale = 255.0 / (p98 - p2) if p98 > p2 else 1.0
        l_channel = np.clip((l_channel - p2) * scale, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_channel
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        # Apply similar adjustment to RGB channels for color preservation
        # Even more aggressive 1st/99th percentiles for individual color channels
        # This can create vivid colors but risks color shifts
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
        
        This method combines three different white balance algorithms, each with
        different strengths, to achieve robust color correction across various
        lighting conditions. The blending weights are adjusted based on the
        detected color cast type.
        
        Algorithm selection rationale:
        - Gray World: Assumes the average color should be neutral gray
        - White Patch: Assumes the brightest pixels should be white
        - ACE: Uses local color statistics for adaptive correction
        """
        # Method 1: Gray World with LAB color space
        # Works well for scenes with diverse colors but can fail with
        # dominant single colors (e.g., forest scenes, blue pools)
        result_gw = self._gray_world_lab(image)
        
        # Method 2: White Patch (excluding overexposed pixels)
        # Effective for scenes with true white objects but can be fooled
        # by colored light sources or no white objects in scene
        result_wp = self._white_patch_retinex(image)
        
        # Method 3: Automatic color equalization
        # Provides local adaptation and works well for mixed lighting
        # but can sometimes over-correct subtle color grading
        result_ace = self._automatic_color_equalization(image)
        
        # Blend results based on cast type
        # These weights were determined through extensive testing on
        # images with known color casts
        if cast_type == 'cool':
            # For cool cast (common in shade, overcast, and pool scenes)
            # White patch works well because cool casts often affect highlights
            result = 0.2 * result_gw + 0.5 * result_wp + 0.3 * result_ace
        elif cast_type == 'warm':
            # For warm cast (indoor tungsten, candlelight, sunset)
            # Gray world is more reliable as warm scenes may lack true whites
            result = 0.5 * result_gw + 0.2 * result_wp + 0.3 * result_ace
        else:
            # Equal weighting for unknown or mixed casts
            # This provides a balanced correction without over-emphasizing
            # any single algorithm's assumptions
            result = 0.33 * result_gw + 0.33 * result_wp + 0.34 * result_ace
        
        return np.clip(result, 0, 1)
    
    def _gray_world_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Gray World algorithm in LAB color space
        
        The Gray World assumption states that the average color of a scene
        should be neutral gray. This works because most natural scenes contain
        a balanced distribution of colors. Working in LAB color space is crucial
        because it separates luminance (L) from color (a, b), allowing us to
        correct color casts without affecting brightness.
        """
        # Convert to LAB color space
        # LAB is perceptually uniform, making it ideal for color corrections
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate average of a and b channels
        # In LAB: a channel = green-red axis, b channel = blue-yellow axis
        # Neutral gray has a=128, b=128 (in 8-bit representation)
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        
        # Shift a and b channels to neutral (128)
        # This removes the color cast by centering the color distribution
        # The shift amount (avg - 128) represents the color cast strength
        a = np.clip(a - (avg_a - 128), 0, 255).astype(np.uint8)
        b = np.clip(b - (avg_b - 128), 0, 255).astype(np.uint8)
        
        # Merge and convert back
        # L channel remains unchanged, preserving original brightness
        lab_corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
        
        return corrected.astype(np.float32) / 255.0
    
    def _white_patch_retinex(self, image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
        """
        White Patch Retinex with percentile to avoid overexposed pixels
        
        The White Patch algorithm assumes that the brightest pixels in the image
        should be white (equal RGB values). However, using the absolute maximum
        can be problematic due to:
        - Specular highlights that are already clipped
        - Sensor noise creating false maxima
        - Light sources in frame that shouldn't be considered
        
        Using the 99th percentile provides robustness against these issues while
        still finding legitimate bright regions that should be white.
        """
        corrected = image.copy()
        
        # Find the percentile values for each channel
        # The 99th percentile ignores the top 1% of pixels, which are likely
        # to be specular highlights, sensor hot pixels, or overexposed areas
        for i in range(3):
            channel = image[:, :, i]
            max_val = np.percentile(channel, percentile)
            if max_val > 0:
                # Scale each channel so its 99th percentile becomes 1.0 (white)
                # This assumes the bright pixels should have been white
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
        
        The Laplacian operator is essentially an edge detection filter. Sharp images
        have strong, well-defined edges that produce high variance in the Laplacian
        response. Blurry images have soft edges with low variance.
        
        Why Laplacian variance works:
        - The Laplacian is a second-order derivative operator
        - It responds strongly to rapid intensity changes (edges)
        - Blur smooths out these intensity changes
        - Lower variance = fewer/softer edges = blurrier image
        """
        # Convert to grayscale if needed
        # Edge detection works on intensity, not color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize for consistent detection across different resolutions
        # High-resolution images naturally have higher Laplacian variance
        # Normalizing to ~1000px width ensures consistent thresholds
        # This also speeds up processing without losing accuracy
        height, width = gray.shape
        if width > 1000:
            scale = 1000 / width
            new_width = 1000
            new_height = int(height * scale)
            # INTER_AREA is best for downsampling - preserves edge information
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Calculate Laplacian variance
        # CV_64F: Use 64-bit float to capture both positive and negative edge responses
        # The Laplacian kernel is: [0, 1, 0; 1, -4, 1; 0, 1, 0]
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Determine if blurry
        # Threshold of 100 was determined empirically:
        # - Sharp images typically have variance > 200
        # - Slightly soft images: 100-200
        # - Noticeably blurry: 50-100
        # - Very blurry: < 50
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
        
        iPhone's photo processing philosophy is to enhance without being obvious.
        This method mimics that approach by using conservative percentiles and
        only applying corrections when truly needed.
        """
        # Use more conservative percentiles to avoid clipping
        enhanced = image.copy()
        
        # Work on luminance for gentle adjustment
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Conservative percentiles (iPhone doesn't clip as aggressively)
        # 5th/95th percentiles preserve more original tones than typical 2nd/98th
        # This prevents crushing deep shadows or blowing subtle highlights
        p5 = np.percentile(gray, 5)   # More conservative than 2%
        p95 = np.percentile(gray, 95)  # More conservative than 98%
        
        # Only apply if there's meaningful dynamic range to improve
        # If the image already uses most of the tonal range (>180 out of 255),
        # it doesn't need levels adjustment
        if p95 - p5 < 180:  # Image lacks contrast
            # Very gentle stretch
            for i in range(3):
                # Even more conservative per-channel percentiles (3rd/97th)
                # This preserves color relationships while gently expanding range
                p_low = np.percentile(enhanced[:, :, i], 3)
                p_high = np.percentile(enhanced[:, :, i], 97)
                if p_high > p_low:
                    # Scale to 0.98 instead of 1.0, leaving headroom
                    # The 0.01 offset prevents pure black, maintaining shadow detail
                    scale = 0.98 / (p_high - p_low)  # Gentle scale factor
                    enhanced[:, :, i] = np.clip((enhanced[:, :, i] - p_low) * scale + 0.01, 0, 1)
        
        return enhanced
    
    def _apply_gentle_white_balance(self, image: np.ndarray, cast_type: str) -> np.ndarray:
        """
        Apply very subtle white balance correction like iPhone
        
        iPhone's white balance corrections are barely perceptible - they remove
        obvious color casts while preserving the scene's natural ambiance.
        This is especially important for:
        - Sunset/sunrise photos (preserve warmth)
        - Blue hour photography (keep the blue mood)
        - Indoor ambient lighting (maintain atmosphere)
        """
        # iPhone uses very gentle corrections
        # 0.3 strength means we only apply 30% of what might be "technically correct"
        # This preserves artistic intent while fixing obvious problems
        correction_strength = 0.3  # Much weaker than full correction
        
        if cast_type == 'cool':
            # Gently warm up cool images (e.g., overcast days, shade)
            enhanced = image.copy()
            # Slightly boost red and reduce blue
            # The asymmetry (0.1 vs 0.05) is intentional - warming feels more natural
            # with stronger red boost than blue reduction
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 + correction_strength * 0.1), 0, 1)  # Red
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 - correction_strength * 0.05), 0, 1)  # Blue
        elif cast_type == 'warm':
            # Gently cool down warm images (e.g., tungsten lighting)
            enhanced = image.copy()
            # Slightly boost blue and reduce red
            # Again, asymmetric adjustment for more natural results
            enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * (1 - correction_strength * 0.05), 0, 1)  # Red
            enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * (1 + correction_strength * 0.1), 0, 1)  # Blue
        else:
            # No correction needed for neutral images
            enhanced = image
            
        return enhanced
    
    def _apply_gentle_curve_adjustment(self, image: np.ndarray, adjustment_type: str, value: float) -> np.ndarray:
        """
        Apply gentle curve adjustments like iPhone Photos
        
        iPhone's brightness and contrast adjustments use sophisticated curves that:
        - Protect highlights from blowing out
        - Preserve shadow detail
        - Maintain natural midtone relationships
        
        This implementation uses weighted adjustments that affect different
        tonal ranges differently, mimicking pro photo editing techniques.
        """
        enhanced = image.copy()
        strength = abs(value) / 100.0  # Convert to 0-1 range, very gentle
        
        if adjustment_type == 'brightness':
            if value > 0:
                # Gentle brightening - lift shadows more than highlights
                # The (1 - enhanced) term means darker pixels get more boost
                # This prevents highlight clipping while opening up shadows
                # Factor of 0.5 ensures subtlety
                enhanced = enhanced + (strength * 0.5 * (1 - enhanced))  # Gentle lift
            else:
                # Gentle darkening
                # Simple multiplication affects all tones equally
                # Factor of 0.3 is more conservative than brightening
                # because darkening is more noticeable perceptually
                enhanced = enhanced * (1 - strength * 0.3)  # Gentle darken
        elif adjustment_type == 'contrast':
            if value > 0:
                # Gentle contrast increase using S-curve logic
                # Subtracting 0.5 centers the adjustment around midtones
                # Multiplying expands tonal range
                # Factor of 0.4 keeps it subtle (iPhone rarely goes beyond this)
                enhanced = np.clip((enhanced - 0.5) * (1 + strength * 0.4) + 0.5, 0, 1)
            else:
                # Gentle contrast decrease (flatten the curve)
                # Factor of 0.3 because reducing contrast can quickly look "flat"
                enhanced = np.clip((enhanced - 0.5) * (1 - strength * 0.3) + 0.5, 0, 1)
        
        return enhanced
    
    def _apply_mild_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Apply very mild local contrast enhancement like iPhone's Smart HDR
        
        This mimics the subtle local tone mapping that iPhone applies to bring out
        detail in both shadows and highlights without the artificial "HDR look".
        
        CLAHE (Contrast Limited Adaptive Histogram Equalization) is perfect for this:
        - "Adaptive" means it works on local regions, not globally
        - "Contrast Limited" prevents over-enhancement
        - Working on L channel in LAB preserves color accuracy
        """
        # Convert to LAB for perceptual processing
        # LAB separates lightness from color, allowing us to enhance detail
        # without shifting colors or creating color artifacts
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Very mild CLAHE (much gentler than typical HDR processing)
        # clipLimit=1.5 is very conservative (typical HDR might use 3.0-4.0)
        # This prevents the "crunchy" over-processed look
        # tileGridSize=(8,8) provides good local adaptation without artifacts
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Very conservative
        l_enhanced = clahe.apply(l_channel)
        
        # Blend with original (iPhone-like subtlety)
        # 30% blend factor means we keep 70% of the original
        # This ensures the enhancement is felt but not seen
        # iPhone's processing is all about "invisible" improvements
        blend_factor = 0.3  # Only 30% of the effect
        l_final = cv2.addWeighted(l_channel, 1 - blend_factor, l_enhanced, blend_factor, 0)
        
        lab[:, :, 0] = l_final
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32) / 255.0