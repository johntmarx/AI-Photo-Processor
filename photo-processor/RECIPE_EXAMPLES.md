# Practical Recipe Examples

## Common Photography Workflows

### 1. Instagram-Ready Portrait
**Goal**: Create social media optimized portraits with consistent look

```json
{
  "name": "Instagram Portrait",
  "description": "Square crop with warm, vibrant look for social media",
  "operations": [
    {
      "operation": "crop",
      "parameters": {"aspectRatio": "1:1"},
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 15},
      "enabled": true
    },
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -25},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 35},
      "enabled": true
    },
    {
      "operation": "adjust_vibrance",
      "parameters": {"value": 25},
      "enabled": true
    },
    {
      "operation": "color_grading",
      "parameters": {
        "shadows": {"r": 0, "g": 0, "b": 8},
        "midtones": {"r": 5, "g": 0, "b": -5},
        "highlights": {"r": 10, "g": 5, "b": 0}
      },
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 60,
        "radius": 0.8,
        "threshold": 2
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "export": {
      "format": "jpeg",
      "quality": 85,
      "maxDimension": 1080
    }
  }
}
```

### 2. Professional Headshot
**Goal**: Clean, professional look with subtle enhancement

```json
{
  "name": "Professional Headshot",
  "description": "Corporate headshot processing with natural enhancement",
  "operations": [
    {
      "operation": "crop",
      "parameters": {"aspectRatio": "4:5"},
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 8},
      "enabled": true
    },
    {
      "operation": "adjust_contrast",
      "parameters": {"value": 5},
      "enabled": true
    },
    {
      "operation": "adjust_clarity",
      "parameters": {"value": -5},
      "enabled": true
    },
    {
      "operation": "denoise",
      "parameters": {
        "strength": 40,
        "preserveDetail": true
      },
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 40,
        "radius": 1.0,
        "threshold": 3
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "aiModels": {
      "faceEnhancement": true,
      "enhancementStrength": 30
    },
    "export": {
      "format": "jpeg",
      "quality": 95,
      "preserveMetadata": true
    }
  }
}
```

### 3. Landscape Photography
**Goal**: Dramatic landscapes with enhanced details and colors

```json
{
  "name": "Dramatic Landscape",
  "description": "Enhance natural landscapes with vivid colors and details",
  "operations": [
    {
      "operation": "lens_correction",
      "parameters": {
        "autoCorrect": true,
        "vignetteAmount": -20
      },
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": -5},
      "enabled": true
    },
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -40},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 20},
      "enabled": true
    },
    {
      "operation": "adjust_whites",
      "parameters": {"value": -15},
      "enabled": true
    },
    {
      "operation": "adjust_blacks",
      "parameters": {"value": 10},
      "enabled": true
    },
    {
      "operation": "adjust_vibrance",
      "parameters": {"value": 35},
      "enabled": true
    },
    {
      "operation": "adjust_clarity",
      "parameters": {"value": 25},
      "enabled": true
    },
    {
      "operation": "color_grading",
      "parameters": {
        "shadows": {"r": 0, "g": 0, "b": 15},
        "midtones": {"r": 0, "g": 0, "b": 0},
        "highlights": {"r": 15, "g": 10, "b": 0}
      },
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 80,
        "radius": 1.2,
        "threshold": 1
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "qualityThreshold": 90,
    "export": {
      "format": "jpeg",
      "quality": 95,
      "maxDimension": 4096
    }
  }
}
```

### 4. Black & White Fine Art
**Goal**: Classic monochrome with rich tones

```json
{
  "name": "Fine Art Monochrome",
  "description": "Classic black and white conversion with tonal control",
  "operations": [
    {
      "operation": "adjust_saturation",
      "parameters": {"value": -100},
      "enabled": true
    },
    {
      "operation": "adjust_contrast",
      "parameters": {"value": 15},
      "enabled": true
    },
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -20},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 15},
      "enabled": true
    },
    {
      "operation": "adjust_whites",
      "parameters": {"value": 25},
      "enabled": true
    },
    {
      "operation": "adjust_blacks",
      "parameters": {"value": -15},
      "enabled": true
    },
    {
      "operation": "adjust_clarity",
      "parameters": {"value": 20},
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 70,
        "radius": 1.0,
        "threshold": 2
      },
      "enabled": true
    }
  ],
  "style_preset": "monochrome",
  "processing_config": {
    "export": {
      "format": "jpeg",
      "quality": 95
    }
  }
}
```

### 5. Event Photography Batch
**Goal**: Consistent processing for event photos with culling

```json
{
  "name": "Event Batch Processing",
  "description": "Quick enhancement for event photos with automatic culling",
  "operations": [
    {
      "operation": "auto_rotate",
      "parameters": {"method": "exif"},
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 10},
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
        "amount": 50,
        "radius": 0.8,
        "threshold": 3
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "culling": {
      "enabled": true,
      "minScore": 65,
      "groupSimilar": true,
      "similarityThreshold": 85
    },
    "export": {
      "format": "jpeg",
      "quality": 88,
      "maxDimension": 2048
    }
  }
}
```

### 6. Product Photography
**Goal**: Clean product shots with white background

```json
{
  "name": "Product Photography",
  "description": "E-commerce ready product photos",
  "operations": [
    {
      "operation": "perspective_correction",
      "parameters": {
        "auto": true
      },
      "enabled": true
    },
    {
      "operation": "crop",
      "parameters": {"aspectRatio": "1:1"},
      "enabled": true
    },
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 20},
      "enabled": true
    },
    {
      "operation": "adjust_contrast",
      "parameters": {"value": 10},
      "enabled": true
    },
    {
      "operation": "adjust_whites",
      "parameters": {"value": 30},
      "enabled": true
    },
    {
      "operation": "adjust_saturation",
      "parameters": {"value": -10},
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 90,
        "radius": 0.6,
        "threshold": 1
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "aiModels": {
      "objectDetection": true
    },
    "export": {
      "format": "jpeg",
      "quality": 95,
      "maxDimension": 2000
    }
  }
}
```

### 7. Film Emulation
**Goal**: Vintage film look with grain and color shift

```json
{
  "name": "Film Emulation - Portra 400",
  "description": "Emulate the look of Kodak Portra 400 film",
  "operations": [
    {
      "operation": "adjust_exposure",
      "parameters": {"value": 5},
      "enabled": true
    },
    {
      "operation": "adjust_contrast",
      "parameters": {"value": -8},
      "enabled": true
    },
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -15},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 20},
      "enabled": true
    },
    {
      "operation": "adjust_saturation",
      "parameters": {"value": -15},
      "enabled": true
    },
    {
      "operation": "color_grading",
      "parameters": {
        "shadows": {"r": 5, "g": 0, "b": -5},
        "midtones": {"r": 8, "g": 5, "b": 0},
        "highlights": {"r": 15, "g": 12, "b": 5}
      },
      "enabled": true
    },
    {
      "operation": "apply_lut",
      "parameters": {
        "lutFile": "portra400.cube",
        "intensity": 75
      },
      "enabled": true
    }
  ],
  "style_preset": "vintage",
  "processing_config": {
    "export": {
      "format": "jpeg",
      "quality": 92
    }
  }
}
```

### 8. HDR-Style Processing
**Goal**: High dynamic range look without actual HDR

```json
{
  "name": "Faux HDR",
  "description": "HDR-like effect from single exposure",
  "operations": [
    {
      "operation": "adjust_highlights",
      "parameters": {"value": -60},
      "enabled": true
    },
    {
      "operation": "adjust_shadows",
      "parameters": {"value": 60},
      "enabled": true
    },
    {
      "operation": "adjust_whites",
      "parameters": {"value": -30},
      "enabled": true
    },
    {
      "operation": "adjust_blacks",
      "parameters": {"value": 20},
      "enabled": true
    },
    {
      "operation": "adjust_vibrance",
      "parameters": {"value": 40},
      "enabled": true
    },
    {
      "operation": "adjust_clarity",
      "parameters": {"value": 50},
      "enabled": true
    },
    {
      "operation": "sharpen",
      "parameters": {
        "amount": 100,
        "radius": 1.5,
        "threshold": 0
      },
      "enabled": true
    }
  ],
  "processing_config": {
    "export": {
      "format": "jpeg",
      "quality": 90
    }
  }
}
```

## Parameter Effect Examples

### Exposure Values Explained
- **-100**: Completely black
- **-50**: Very dark, like 2 stops underexposed
- **-25**: Dark, preserves highlights
- **0**: No change
- **+25**: Bright, good for backlit subjects
- **+50**: Very bright, may blow highlights
- **+100**: Near white

### Clarity Values Explained
- **-100**: Extreme soft focus, dreamy
- **-50**: Very soft, flattering for portraits
- **-25**: Subtle softening
- **0**: No change
- **+25**: Enhanced local contrast
- **+50**: Strong detail enhancement
- **+100**: Extreme detail, may look unnatural

### Vibrance vs Saturation
- **Saturation**: Affects all colors equally
- **Vibrance**: Smart saturation that:
  - Protects skin tones
  - Enhances less-saturated colors more
  - Prevents color clipping

## Tips for Creating Recipes

1. **Start Conservative**: Begin with small adjustments (±10-20)
2. **Order Matters**: Apply corrections before enhancements
3. **Test on Various Images**: Recipes should work on different photos
4. **Consider Output**: Adjust quality based on intended use
5. **Use Presets as Base**: Modify existing presets for consistency
6. **Enable Culling for Events**: Automatically filter poor shots
7. **Preserve Metadata**: Important for professional work

## Workflow Combinations

### Portrait Workflow
1. Auto-rotate → 2. Lens correction → 3. Crop → 4. Exposure → 5. Skin enhancement → 6. Color grade → 7. Sharpen

### Landscape Workflow  
1. Lens correction → 2. Perspective → 3. Exposure → 4. Graduated filter → 5. Vibrance → 6. Clarity → 7. Sharpen

### Event Workflow
1. Auto-rotate → 2. Quick enhance → 3. Denoise → 4. Batch resize → 5. Auto-cull similar

### Product Workflow
1. Background removal → 2. Perspective correct → 3. White balance → 4. Enhance details → 5. Export for web