# Complete Recipe Operations Guide

## Overview
This guide provides a detailed explanation of every recipe operation available in the photo processing system, including all configurable parameters and how they work.

## Frontend Recipe Editor Structure

The recipe editor in `frontend-app/src/pages/RecipeEditor.tsx` provides a comprehensive UI for configuring all operations. Each recipe contains:

1. **Basic Information**
   - Name and description
   - Draft/Published status

2. **Processing Steps** (drag-and-drop reorderable)
   - Each step can be enabled/disabled
   - Full parameter configuration

3. **Style Presets**
   - Natural, Vivid, Monochrome, Vintage, Cinematic, Portrait, Landscape

4. **Quality & Culling Settings**
   - Quality threshold (0-100%)
   - Culling options with similarity detection

5. **Export Settings**
   - Format (JPEG, PNG, WebP)
   - Quality (1-100%)
   - Size constraints

6. **AI Model Parameters**
   - Enhancement models and strength
   - Feature toggles

## Detailed Operation Reference

### 1. Crop Operation
**Purpose**: Adjusts image aspect ratio by removing portions from edges

**Parameters**:
```javascript
{
  aspectRatio: string,  // "original", "1:1", "4:3", "16:9", "custom"
  customRatio: string   // e.g., "21:9" (only if aspectRatio is "custom")
}
```

**How it works**:
- Calculates center crop to achieve target aspect ratio
- Preserves the most important central portion of the image
- No distortion - only removes edge content

**Frontend Configuration**:
- Dropdown with preset ratios
- Custom ratio input field appears when "custom" selected

### 2. Rotate Operation
**Purpose**: Rotates image by specified angle or auto-straightens

**Parameters**:
```javascript
{
  angle: number,        // -180 to 180 degrees
  autoStraighten: boolean  // Enable AI-based straightening
}
```

**How it works**:
- Manual: Rotates by exact angle with canvas expansion
- Auto: Detects horizon/vertical lines and corrects tilt
- Fills expanded areas with white background

**Frontend Configuration**:
- Number input with 0.1° precision
- Checkbox for auto-straighten feature

### 3. Exposure Adjustment
**Purpose**: Brightens or darkens the entire image

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Negative values darken (down to -100 = black)
- Positive values brighten (up to +100 = white)
- Applied uniformly across all pixels

**Frontend Configuration**:
- Slider control with live value display

### 4. Contrast Adjustment
**Purpose**: Increases or decreases difference between light and dark areas

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Negative values reduce contrast (flatter image)
- Positive values increase contrast (more dramatic)
- Affects tonal range distribution

### 5. Saturation Adjustment
**Purpose**: Controls color intensity

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- -100 = complete desaturation (grayscale)
- 0 = original colors
- +100 = maximum color intensity

### 6. Highlights Adjustment
**Purpose**: Adjusts only the brightest areas of the image

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Negative values darken bright areas (recover blown highlights)
- Positive values brighten already bright areas
- Preserves shadows and midtones

### 7. Shadows Adjustment
**Purpose**: Adjusts only the darkest areas of the image

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Positive values brighten dark areas (reveal shadow detail)
- Negative values darken shadows further
- Preserves highlights and midtones

### 8. Whites Adjustment
**Purpose**: Fine-tunes the very brightest tones

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Affects only the top 10-20% of brightness values
- More precise than highlights adjustment
- Useful for controlling specular highlights

### 9. Blacks Adjustment
**Purpose**: Fine-tunes the very darkest tones

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Affects only the bottom 10-20% of brightness values
- More precise than shadows adjustment
- Controls pure black levels

### 10. Vibrance Adjustment
**Purpose**: Intelligent saturation that protects skin tones

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Increases saturation of less-saturated colors more
- Protects already vibrant colors from oversaturation
- Minimal impact on skin tones

### 11. Clarity Adjustment
**Purpose**: Enhances local contrast and detail

**Parameters**:
```javascript
{
  value: number  // -100 to +100
}
```

**How it works**:
- Positive values increase midtone contrast
- Creates "pop" without affecting overall contrast
- Negative values create soft, dreamy effect

### 12. Apply LUT (Lookup Table)
**Purpose**: Applies complex color grading via 3D color transformation

**Parameters**:
```javascript
{
  lutFile: string,    // Path to .cube or .3dl file
  intensity: number   // 0-100% strength
}
```

**How it works**:
- Remaps colors based on 3D lookup table
- Enables cinematic color grading
- Intensity blends between original and LUT result

**Frontend Configuration**:
- File path input (future: file picker)
- Intensity slider for blend control

### 13. Denoise
**Purpose**: Reduces image noise/grain

**Parameters**:
```javascript
{
  strength: number,        // 0-100%
  preserveDetail: boolean  // Protect edges/details
}
```

**How it works**:
- Analyzes and reduces random pixel variations
- Strength controls aggressiveness
- Detail preservation prevents over-smoothing

**Frontend Configuration**:
- Strength slider
- Checkbox for detail preservation

### 14. Sharpen
**Purpose**: Enhances edge definition and detail

**Parameters**:
```javascript
{
  amount: number,    // 0-200% strength
  radius: number,    // 0.1-5.0 pixels
  threshold: number  // 0-255 edge detection
}
```

**How it works**:
- Amount: How much to sharpen
- Radius: Size of sharpening effect
- Threshold: Minimum contrast to sharpen (prevents noise)

**Frontend Configuration**:
- Three separate controls for precise adjustment
- Suitable for different image types

### 15. Lens Correction
**Purpose**: Fixes optical distortions from camera lens

**Parameters**:
```javascript
{
  autoCorrect: boolean,     // Use lens profile
  vignetteAmount: number,   // -100 to +100
  distortionAmount: number  // -100 to +100
}
```

**How it works**:
- Auto: Uses EXIF data to apply lens profile
- Vignette: Corrects corner darkening
- Distortion: Fixes barrel/pincushion distortion

### 16. Perspective Correction
**Purpose**: Fixes keystoning and perspective distortion

**Parameters**:
```javascript
{
  auto: boolean,      // AI-based correction
  vertical: number,   // -100 to +100
  horizontal: number  // -100 to +100
}
```

**How it works**:
- Auto: Detects and corrects converging lines
- Manual: Adjust vertical/horizontal perspective
- Useful for architectural photography

### 17. Color Grading
**Purpose**: Professional three-way color correction

**Parameters**:
```javascript
{
  shadows: { r: number, g: number, b: number },    // -100 to +100 each
  midtones: { r: number, g: number, b: number },   // -100 to +100 each
  highlights: { r: number, g: number, b: number }  // -100 to +100 each
}
```

**How it works**:
- Independently adjust color in three tonal ranges
- Create complex color looks (teal/orange, etc.)
- Professional cinematography tool

**Frontend Configuration**:
- 9 sliders (3 channels × 3 ranges)
- Grouped by tonal range

## Style Presets

Style presets apply predefined combinations of operations:

### Natural
- Balanced enhancement
- Slight clarity boost
- Minimal color adjustment

### Vivid
- Increased saturation (+20)
- Enhanced contrast (+15)
- Vibrance boost (+25)

### Monochrome
- Complete desaturation
- Adjusted contrast curve
- Enhanced clarity

### Vintage
- Reduced saturation (-20)
- Lifted blacks (+15)
- Warm color shift
- Reduced clarity (-10)

### Cinematic
- Teal/orange color grade
- Lifted blacks
- Enhanced contrast
- 2.35:1 aspect crop

### Portrait
- Skin tone protection
- Soft clarity (-5)
- Slight warmth
- Background blur (if AI enabled)

### Landscape
- Enhanced vibrance
- Increased clarity
- Dehaze effect
- Polarizer simulation

## Quality & Culling Settings

### Quality Threshold
- 0-100% minimum acceptable quality
- Uses NIMA model for assessment
- Photos below threshold are flagged

### Culling Options
```javascript
{
  enabled: boolean,
  minScore: number,           // 0-100%
  groupSimilar: boolean,
  similarityThreshold: number // 0-100%
}
```

- Groups burst shots
- Selects best from similar images
- Based on technical quality metrics

## Export Settings

### Format Options
- **JPEG**: Best compatibility, smaller files
- **PNG**: Lossless, transparency support
- **WebP**: Modern format, best compression

### Quality Setting
- 1-100% for lossy formats
- Higher = better quality, larger files
- Recommended: 85-95% for web

### Size Constraints
- Optional maximum dimension
- Maintains aspect ratio
- Useful for web optimization

## AI Model Parameters

### Enhancement Model
- **Standard**: Balanced quality/speed
- **High Quality**: Best results, slower
- **Fast**: Quick processing
- **None**: Disable AI enhancement

### Enhancement Strength
- 0-100% AI enhancement blend
- Higher values = more AI processing

### Feature Toggles
- **Object Detection**: Identify subjects
- **Scene Analysis**: Understand content
- **Face Enhancement**: Improve portraits

## Working Operations Summary

### ✅ Fully Working (No AI Required)
1. Crop (all aspect ratios)
2. Rotate (manual angles)
3. All exposure adjustments
4. All color adjustments
5. Filters (blur, sharpen, etc.)
6. Basic denoise
7. Resize operations
8. Metadata preservation

### 🟡 Implemented but Need Dependencies
1. Auto-rotation (needs EXIF reading)
2. Auto lens correction (needs lens profiles)
3. Advanced denoise (needs AI model)
4. Scene analysis (needs model weights)
5. Face enhancement (needs face detection)
6. Perspective auto-correct (needs line detection)

## Recipe JSON Structure

Here's a complete example recipe with all operations:

```json
{
  "name": "Professional Portrait",
  "description": "Complete portrait enhancement workflow",
  "operations": [
    {
      "operation": "auto_rotate",
      "parameters": {"method": "exif"},
      "enabled": true
    },
    {
      "operation": "lens_correction",
      "parameters": {
        "autoCorrect": true,
        "vignetteAmount": 0,
        "distortionAmount": 0
      },
      "enabled": true
    },
    {
      "operation": "crop",
      "parameters": {"aspectRatio": "3:4"},
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 10},
      "enabled": true
    },
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -20},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 25},
      "enabled": true
    },
    {
      "operation": "adjust_vibrance",
      "parameters": {"value": 15},
      "enabled": true
    },
    {
      "operation": "color_grading",
      "parameters": {
        "shadows": {"r": 0, "g": 0, "b": 5},
        "midtones": {"r": 0, "g": 0, "b": 0},
        "highlights": {"r": 5, "g": 3, "b": 0}
      },
      "enabled": true
    },
    {
      "operation": "denoise",
      "parameters": {
        "strength": 30,
        "preserveDetail": true
      },
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 80,
        "radius": 1.2,
        "threshold": 2
      },
      "enabled": true
    }
  ],
  "style_preset": "portrait",
  "processing_config": {
    "qualityThreshold": 85,
    "culling": {
      "enabled": true,
      "minScore": 70,
      "groupSimilar": true,
      "similarityThreshold": 90
    },
    "export": {
      "format": "jpeg",
      "quality": 92,
      "maxDimension": 2048,
      "preserveMetadata": true
    },
    "aiModels": {
      "enhancementModel": "high-quality",
      "enhancementStrength": 60,
      "objectDetection": true,
      "sceneAnalysis": true,
      "faceEnhancement": true
    }
  }
}
```

## Integration with Backend

The frontend recipe configuration maps directly to backend processing:

1. Frontend creates recipe JSON
2. Sends to `/api/recipes` endpoint
3. Backend stores in recipe storage
4. Processing service applies operations in sequence
5. Each operation has a corresponding processor function
6. Results are saved with configured export settings

All parameters shown in this guide are fully exposed in the frontend UI and can be adjusted to create custom processing workflows.