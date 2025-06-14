"""
Intelligent Image Enhancement Service - Modernized

Complete redesign focusing on what makes photos "pop" using Instagram/VSCO-style techniques:
- Film-style tone curves for lifted shadows and compressed highlights
- True vibrance (not saturation) that protects skin tones
- Warm color grading with split toning
- Subtle glow effect for that dreamy look
- Better shadow/highlight recovery
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class IntelligentEnhancer:
    """
    Professional-grade automatic image enhancement using OpenCV.
    
    Applies multiple techniques:
    - Automatic color balance using white balance algorithms
    - Adaptive histogram equalization for local contrast
    - Smart exposure correction
    - Color vibrancy enhancement
    - Shadow/highlight recovery
    - Noise reduction while preserving details
    """
    
    def __init__(self):
        # Much gentler CLAHE with larger tile size for smoother results
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    
    def enhance_image(self, image: Image.Image, strength: float = 0.7) -> Image.Image:
        """
        Apply intelligent enhancement to image.
        
        Args:
            image: PIL Image to enhance
            strength: Enhancement strength (0.0 to 1.0, default 0.7 for natural look)
        
        Returns:
            Enhanced PIL Image
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply enhancement pipeline
        enhanced = self._apply_enhancement_pipeline(cv_image, strength)
        
        # Convert back to PIL
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)
    
    def _apply_enhancement_pipeline(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Apply the full enhancement pipeline."""
        # Store original for blending
        original = img.copy()
        
        # 1. Apply film-style tone curve FIRST (this is key!)
        curved = self._apply_film_tone_curve(img)
        
        # 2. Smart exposure adjustment
        exposed = self._smart_exposure_adjustment(curved)
        
        # 3. True vibrance enhancement (not saturation)
        vibrant = self._enhance_vibrance(exposed)
        
        # 4. Subtle local contrast
        contrasted = self._subtle_local_contrast(vibrant)
        
        # 5. Warm color grading
        graded = self._apply_color_grading(contrasted)
        
        # 6. Subtle glow effect
        glowed = self._add_subtle_glow(graded)
        
        # 7. Final sharpening
        final = self._final_sharpen(glowed)
        
        # Blend with original based on strength
        if strength < 1.0:
            result = cv2.addWeighted(original, 1.0 - strength, final, strength, 0)
        else:
            result = final
        
        return result
    
    def _apply_film_tone_curve(self, img: np.ndarray) -> np.ndarray:
        """
        Apply film-style tone curve that lifts shadows and compresses highlights.
        This is the secret sauce that makes photos "pop" like Instagram.
        """
        # Convert to float for precise calculations
        img_float = img.astype(np.float32) / 255.0
        
        # Film-style curve: lifted shadows, enhanced midtones, compressed highlights
        # Using a combination of power curves for smooth results
        
        # Shadow lifting (affects values 0-0.3)
        shadow_mask = img_float < 0.3
        img_float[shadow_mask] = 0.3 * np.power(img_float[shadow_mask] / 0.3, 0.85)
        
        # Midtone enhancement (affects values 0.3-0.7)
        midtone_mask = (img_float >= 0.3) & (img_float < 0.7)
        # Gentle S-curve for midtones
        x = (img_float[midtone_mask] - 0.3) / 0.4  # Normalize to 0-1
        # Sigmoid-like curve
        y = x + 0.05 * np.sin(2 * np.pi * x)
        img_float[midtone_mask] = 0.3 + y * 0.4
        
        # Highlight compression (affects values 0.7-1.0)
        highlight_mask = img_float >= 0.7
        x = (img_float[highlight_mask] - 0.7) / 0.3
        # Compress highlights gently
        y = np.power(x, 1.2)
        img_float[highlight_mask] = 0.7 + y * 0.3
        
        return (img_float * 255).astype(np.uint8)
    
    def _smart_exposure_adjustment(self, img: np.ndarray) -> np.ndarray:
        """
        Smart exposure using percentile-based analysis.
        More sophisticated than simple mean-based adjustment.
        """
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Use percentiles for better analysis
        p10 = np.percentile(l, 10)  # Shadow point
        p90 = np.percentile(l, 90)  # Highlight point
        
        # Only adjust if needed
        if p10 < 30:  # Too dark in shadows
            # Lift shadows selectively
            l_float = l.astype(np.float32)
            shadow_boost = np.clip(1.0 - l_float / 128, 0, 1) * 0.15
            l_float = l_float * (1 + shadow_boost)
            l = np.clip(l_float, 0, 255).astype(np.uint8)
        elif p90 > 225:  # Too bright in highlights
            # Gentle highlight recovery
            l_float = l.astype(np.float32)
            highlight_reduce = np.clip((l_float - 128) / 127, 0, 1) * 0.1
            l_float = l_float * (1 - highlight_reduce)
            l = np.clip(l_float, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_adjusted = cv2.merge([l, a, b])
        return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    def _subtle_local_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Very subtle local contrast - the key is restraint!
        """
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gentle CLAHE
        l_clahe = self.clahe.apply(l)
        
        # Blend with original - only 40% strength for natural look
        l_final = cv2.addWeighted(l, 0.6, l_clahe, 0.4, 0)
        
        # Merge back
        lab_enhanced = cv2.merge([l_final, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _enhance_vibrance(self, img: np.ndarray) -> np.ndarray:
        """
        True vibrance enhancement - the Instagram/VSCO secret.
        Boosts muted colors while protecting skin tones and preventing oversaturation.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Normalize saturation
        s_norm = s / 255.0
        
        # Vibrance formula - exponential falloff based on existing saturation
        # This gives much more natural results than linear scaling
        vibrance_amount = 0.3  # Stronger than before but still natural
        
        # Create vibrance mask - less saturated pixels get more boost
        vibrance_mask = np.exp(-2.0 * s_norm)  # Exponential falloff
        
        # Protect skin tones (red-orange hues)
        skin_protection = np.ones_like(h)
        # Skin tones are roughly 0-25 and 335-360 in OpenCV HSV (0-180 range)
        skin_mask = np.logical_or(
            np.logical_and(h >= 0, h <= 25),
            np.logical_and(h >= 168, h <= 180)
        )
        # Also check for low saturation which often indicates skin
        skin_mask = np.logical_and(skin_mask, s_norm < 0.6)
        skin_protection[skin_mask] = 0.3  # Much less effect on skin
        
        # Apply vibrance
        s_boost = 1.0 + (vibrance_amount * vibrance_mask * skin_protection)
        s_new = s * s_boost
        s_new = np.clip(s_new, 0, 255)
        
        # Merge back
        hsv_new = cv2.merge([h, s_new.astype(np.float32), v])
        return cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _apply_color_grading(self, img: np.ndarray) -> np.ndarray:
        """
        Instagram-style color grading with warm highlights and cool shadows.
        This creates that professional, cinematic look.
        """
        # Convert to LAB for precise color control
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create smooth luminance masks
        l_norm = l.astype(np.float32) / 255.0
        
        # Highlight mask with smooth falloff
        highlight_mask = np.clip((l_norm - 0.6) / 0.3, 0, 1)
        highlight_mask = highlight_mask ** 2  # Smoother curve
        
        # Shadow mask with smooth falloff
        shadow_mask = np.clip((0.4 - l_norm) / 0.3, 0, 1)
        shadow_mask = shadow_mask ** 2
        
        # Apply color grading
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        
        # Warm highlights (push towards orange/amber)
        a += highlight_mask * 3  # Red push
        b += highlight_mask * 8  # Yellow push
        
        # Cool shadows (subtle teal/blue)
        a -= shadow_mask * 2   # Slight cyan
        b -= shadow_mask * 5   # Blue push
        
        # Clip and convert back
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_graded = cv2.merge([l, a, b])
        return cv2.cvtColor(lab_graded, cv2.COLOR_LAB2BGR)
    
    def _add_subtle_glow(self, img: np.ndarray) -> np.ndarray:
        """
        Add subtle glow effect (Orton effect) for that dreamy Instagram look.
        This is what makes photos look "magical" and "pop".
        """
        # Create heavily blurred version
        blurred = cv2.GaussianBlur(img, (0, 0), 25)
        
        # Screen blend mode for glow
        img_float = img.astype(np.float32) / 255.0
        blur_float = blurred.astype(np.float32) / 255.0
        
        # Screen blend: 1 - (1 - a) * (1 - b)
        # This brightens and creates glow
        screen = 1.0 - (1.0 - img_float) * (1.0 - blur_float)
        
        # Very subtle blend - just 10% for natural glow
        result = img_float * 0.9 + screen * 0.1
        
        return (result * 255).astype(np.uint8)
    
    def _final_sharpen(self, img: np.ndarray) -> np.ndarray:
        """
        Very subtle final sharpening for crispness.
        Much gentler than before.
        """
        # Small radius unsharp mask
        blurred = cv2.GaussianBlur(img, (0, 0), 0.8)
        
        # Very gentle sharpening
        sharpened = cv2.addWeighted(img, 1.15, blurred, -0.15, 0)
        
        return sharpened
    
    def get_enhancement_preview(self, image_path: str, strength: float = 0.7) -> Tuple[Image.Image, Image.Image]:
        """
        Get before/after preview of enhancement.
        
        Returns:
            Tuple of (original, enhanced) PIL Images
        """
        original = Image.open(image_path)
        enhanced = self.enhance_image(original, strength)
        
        return original, enhanced