# Recipe Workflow Visual Guide

## Recipe Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND RECIPE EDITOR                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Basic Info     2. Style Preset    3. Processing Steps       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ Name        │  │ □ Natural    │  │ + Add Step          │   │
│  │ Description │  │ ☑ Portrait   │  │ ─────────────────── │   │
│  └─────────────┘  │ □ Landscape  │  │ ≡ 1. Auto Rotate    │   │
│                   └──────────────┘  │ ≡ 2. Crop (3:4)     │   │
│                                     │ ≡ 3. Enhance        │   │
│                                     │ ≡ 4. Denoise        │   │
│                                     └─────────────────────┘   │
│                                                                   │
│  4. Quality Settings    5. Export        6. AI Parameters       │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Threshold: 85%  │  │ Format: JPEG│  │ Model: High     │   │
│  │ ☑ Enable Culling│  │ Quality: 92%│  │ Strength: 60%   │   │
│  └─────────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  Save Recipe  │
                         └───────┬───────┘
                                 │
                                 ▼
```

## Backend Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHOTO UPLOAD                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. File Upload → 2. Hash Check → 3. Store Original → 4. Queue  │
│                                                                   │
└────────────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING SERVICE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Load      │ --> │    Apply    │ --> │   Generate  │       │
│  │   Recipe    │     │ Operations  │     │   Outputs   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Parse Steps │     │  For Each   │     │ • Thumbnail │       │
│  │ & Settings  │     │ Operation:  │     │ • Web Size  │       │
│  └─────────────┘     │ - Load img  │     │ • Full Res  │       │
│                      │ - Apply op  │     └─────────────┘       │
│                      │ - Save temp │                            │
│                      └─────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Operation Processing Detail

```
For Each Operation in Recipe:
┌────────────────────────────────────────────────────────────┐
│                                                              │
│  Input Image                    Operation                    │
│  ┌─────────┐     ┌──────────────────────────┐             │
│  │         │     │ Type: crop                │             │
│  │  Photo  │ --> │ Parameters:              │ --> Output  │
│  │         │     │   aspectRatio: "16:9"    │             │
│  └─────────┘     └──────────────────────────┘             │
│                               │                             │
│                               ▼                             │
│                   ┌──────────────────────┐                 │
│                   │   Operation Logic    │                 │
│                   ├──────────────────────┤                 │
│                   │ 1. Calculate dimensions                │
│                   │ 2. Determine crop area                 │
│                   │ 3. Apply transformation                │
│                   │ 4. Return processed image              │
│                   └──────────────────────┘                 │
└────────────────────────────────────────────────────────────┘
```

## Operation Categories & Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    BASIC OPERATIONS                          │
│                  (No AI Required)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ Crop           ✅ Enhance         ✅ Filters            │
│  ✅ Rotate         ✅ Color Adjust    ✅ Resize             │
│  ✅ Basic Denoise  ✅ Sharpen         ✅ Export             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 AI-ENHANCED OPERATIONS                       │
│              (Require Model Weights)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  🟡 Auto Rotation   🟡 Scene Analysis   🟡 Quality Score    │
│  🟡 Face Enhance    🟡 Object Detect    🟡 Smart Crop      │
│  🟡 Background Seg  🟡 Style Transfer   🟡 Super Resolution │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Recipe Execution Example

```
Recipe: "Portrait Enhancement"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Auto Rotate
┌────────────┐    ┌────────────┐
│   Tilted   │ -> │ Straightened│
│   Image    │    │    Image    │
└────────────┘    └────────────┘

Step 2: Crop to 3:4
┌────────────┐    ┌──────┐
│            │ -> │      │
│  16:9 img  │    │ 3:4  │
│            │    │      │
└────────────┘    └──────┘

Step 3: Enhance
┌──────┐          ┌──────┐
│ Dark │    ->    │Bright│
│ Flat │          │ Pop! │
└──────┘          └──────┘

Step 4: Denoise
┌──────┐          ┌──────┐
│Noisy │    ->    │Clean │
│Grainy│          │Smooth│
└──────┘          └──────┘

Final Output Generation:
┌──────┐     ┌───────────────────────┐
│Final │ ->  │ • thumbnail_400x400   │
│Image │     │ • web_1920x1920       │
└──────┘     │ • full_resolution     │
             └───────────────────────┘
```

## Parameter Ranges Quick Reference

| Operation | Parameter | Range | Default | Notes |
|-----------|-----------|-------|---------|-------|
| **Crop** | aspectRatio | Various | original | Presets + custom |
| **Rotate** | angle | -180 to 180 | 0 | 0.1° precision |
| **Exposure** | value | -100 to 100 | 0 | ±stops equivalent |
| **Contrast** | value | -100 to 100 | 0 | Tonal curve adjust |
| **Saturation** | value | -100 to 100 | 0 | -100 = grayscale |
| **Highlights** | value | -100 to 100 | 0 | Bright area only |
| **Shadows** | value | -100 to 100 | 0 | Dark area only |
| **Vibrance** | value | -100 to 100 | 0 | Smart saturation |
| **Clarity** | value | -100 to 100 | 0 | Local contrast |
| **Denoise** | strength | 0 to 100 | 50 | % reduction |
| **Sharpen** | amount | 0 to 200 | 50 | % strength |
| **Sharpen** | radius | 0.1 to 5.0 | 1.0 | Pixel area |
| **Sharpen** | threshold | 0 to 255 | 0 | Edge detection |
| **LUT** | intensity | 0 to 100 | 100 | Blend percentage |
| **Export** | quality | 1 to 100 | 90 | JPEG/WebP only |

## Configuration Storage

```
Frontend State (React)
        │
        ▼
Recipe Object (JSON)
        │
        ▼
API Request (/api/recipes)
        │
        ▼
Backend Storage (/data/recipes/)
        │
        ▼
Processing Queue
        │
        ▼
Applied to Photos
```

## Key Features

1. **Drag & Drop Reordering**: Operations execute in user-defined order
2. **Enable/Disable Toggle**: Skip operations without removing
3. **Live Preview**: (Future) See changes in real-time
4. **Preset System**: Quick-apply common workflows
5. **Batch Processing**: Apply to multiple photos
6. **Non-Destructive**: Original always preserved
7. **Export Control**: Full control over output format/quality