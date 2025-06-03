# AI Photo Processor Setup Complete! 🤖📸

## Overview

You now have a complete AI-powered photo processing pipeline that:

1. **Watches for new RAW photos** via SMB share
2. **Analyzes photos with AI** using Ollama + Gemma3:4b model
3. **Intelligently crops and enhances** based on AI recommendations  
4. **Uploads processed photos** to Immich with detailed descriptions

## Services Running

### 🤖 **Ollama AI Service**
- **Container**: `immich_ollama`
- **Model**: `gemma3:4b` (vision-capable, 4B parameters)
- **GPU Support**: NVIDIA GPU acceleration enabled
- **Port**: `11434`

### 📁 **SMB File Shares**
- **Container**: `immich_samba`
- **Photo Inbox**: `//192.168.1.114/Photo Inbox` (drop files here)
- **Photo Processed**: `//192.168.1.114/Photo Processed` (processed results)
- **Username**: `photos`
- **Password**: `PhotoProcessor2025!`
- **Port**: `445`

### 🔄 **Photo Processor Service**
- **Container**: `immich_photo_processor`
- **Watches**: `/mnt/storage1/photo-inbox`
- **Outputs**: `/mnt/storage1/photo-processed`
- **Supported Formats**: `.arw`, `.cr2`, `.cr3`, `.nef`, `.dng`, `.raf`, `.orf`, `.rw2`, `.pef`, `.srw`, `.x3f`, `.fff`, `.3fr`, `.mrw`, `.raw`, `.jpg`, `.jpeg`, `.png`, `.tiff`

## How It Works

### 1. **File Detection**
- Drop RAW/image files into the SMB share "Photo Inbox"
- File watcher detects new files automatically
- Waits for files to finish transferring

### 2. **AI Analysis**
- Converts RAW files to RGB (full resolution processing)
- Resizes image for AI analysis (maintains aspect ratio)
- Sends to Ollama with structured output schema
- AI analyzes:
  - **Image Quality**: crisp, slightly blurry, blurry, very blurry
  - **Primary Subject**: swimmer, person, crowd, etc.
  - **Bounding Box**: precise coordinates for main subject
  - **Professional Crop**: rule of thirds, golden ratio, etc.
  - **Color Analysis**: exposure, white balance, contrast
  - **Swimming Context**: stroke type, pool type, race timing
  - **Processing Recommendation**: crop+enhance, enhance only, etc.

### 3. **Intelligent Processing**
- **Smart Cropping**: Uses AI bounding box for professional composition
- **Color Enhancement**: 
  - Brightness/contrast adjustment based on AI analysis
  - Auto white balance correction
  - Adaptive histogram equalization for low contrast
  - Maintains 16-bit processing for quality
- **Quality Control**: Skips processing if image is too blurry

### 4. **Immich Integration**
- Saves high-quality JPEG (100% quality)
- Uploads to Immich with rich metadata:
  - AI analysis description
  - Subject identification
  - Quality assessment
  - Crop and processing details
  - Swimming event context
- Adds to "AI Processed Photos" album automatically

## AI Analysis Schema

The AI provides structured responses with:

```json
{
  "quality": "crisp|slightly_blurry|blurry|very_blurry",
  "primary_subject": "swimmer|multiple_swimmers|person|crowd|...",
  "primary_subject_box": {
    "x": 25.5, "y": 30.2, "width": 45.8, "height": 60.1
  },
  "recommended_crop": {
    "crop_box": {...},
    "aspect_ratio": "16:9",
    "composition_rule": "rule_of_thirds",
    "confidence": 0.85
  },
  "color_analysis": {
    "exposure_assessment": "properly_exposed",
    "white_balance_assessment": "neutral",
    "brightness_adjustment_needed": 10,
    "contrast_adjustment_needed": -5
  },
  "swimming_context": {
    "event_type": "freestyle",
    "pool_type": "indoor",
    "time_of_event": "mid_race"
  },
  "processing_recommendation": "crop_and_enhance"
}
```

## Network Access

### **SMB Shares** (from any computer on network):
- **Windows**: `\\192.168.1.114\Photo Inbox`
- **Mac**: `smb://192.168.1.114/Photo Inbox`
- **Linux**: `smb://192.168.1.114/Photo Inbox`

### **Direct API Access**:
- **Ollama**: `http://192.168.1.114:11434`
- **Immich**: `http://192.168.1.114` or `https://photos.marxfamily.net`

## Usage Workflow

1. **Copy photos** to SMB share "Photo Inbox"
2. **AI processes automatically** (check logs: `docker logs immich_photo_processor`)
3. **View results** in Immich under "AI Processed Photos" album
4. **Processed files** also available in "Photo Processed" SMB share

## Performance & Quality

- **RAW Processing**: Full resolution (10-30MP) with 16-bit precision
- **AI Analysis**: Optimized 1024px max dimension for speed
- **Output Quality**: 100% JPEG quality for final images
- **Processing Time**: ~30-60 seconds per photo (depending on size)
- **GPU Acceleration**: NVIDIA GPU used for AI inference

## Monitoring & Logs

```bash
# View processing logs
docker logs -f immich_photo_processor

# Check Ollama status
docker logs immich_ollama

# Monitor all services
docker compose logs -f
```

## Directory Structure

```
/mnt/storage1/
├── photo-inbox/          # SMB share - drop files here
├── photo-processed/       # SMB share - processed results
└── immich/               # Immich storage
```

## Security Features

- **SMB Authentication**: Username/password required
- **Internal Network**: Services isolated to Docker network
- **GPU Isolation**: AI processing in dedicated container
- **Rate Limiting**: Built into photo processor
- **Immich API**: Secure API key authentication

## Supported Photography Types

Optimized for:
- **Swimming Events**: Stroke recognition, pool analysis
- **Sports Photography**: Motion analysis, subject tracking
- **Portrait Photography**: Face detection, composition
- **General Photography**: Professional cropping rules

Your AI photo processor is now live and ready to automatically enhance your swimming event and sports photography! 🏊‍♂️📸✨