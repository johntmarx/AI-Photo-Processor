# NIMA Data Standards - Photo Processor v2.0

## Overview

This document defines the data structures and standards for NIMA (Neural Image Assessment) scoring integrated into our photo processing pipeline. NIMA provides aesthetic and technical quality assessment for uploaded photos.

## Photo Data Structure with NIMA

### Database Schema (Backend - snake_case)

```python
{
    "id": "uuid-string",
    "filename": "string",
    "status": "processing|completed|failed|pending",
    "created_at": "datetime",
    "processed_at": "datetime|null",
    "original_path": "string",
    "file_hash": "string",
    "file_size": "integer",
    "session_id": "string|null",
    "ai_analysis": {
        "status": "pending|processing|completed|failed|not_available",
        "queued_at": "iso_string|null",
        "started_at": "iso_string|null",
        "completed_at": "iso_string|null",
        "aesthetic_score": "float|null",       # 1.0-10.0
        "technical_score": "float|null",       # 1.0-10.0 (optional)
        "quality_level": "string|null",        # poor|below_average|average|good|excellent
        "confidence": "float|null",            # 0.0-1.0
        "nima_results": {
            "aesthetic": {
                "quality_score": "float",
                "quality_std": "float",
                "quality_distribution": "array[float]",  # 10 values (1-10 rating distribution)
                "quality_level": "string",
                "confidence": "float",
                "model_type": "aesthetic",
                "inference_time": "float",
                "image_size": "array[int]"  # [width, height]
            },
            "technical": {  # Optional - only if include_technical=True
                "quality_score": "float",
                "quality_std": "float",
                "quality_distribution": "array[float]",
                "quality_level": "string",
                "confidence": "float",
                "model_type": "technical",
                "inference_time": "float",
                "image_size": "array[int]"
            }
        },
        "model_info": {
            "aesthetic_model": "string",       # "NIMA-aesthetic-v1.0"
            "technical_model": "string|null",  # "NIMA-technical-v1.0"
            "inference_time": "float"
        },
        "error": "string|null",               # Error message if failed
        "task_id": "string|null"              # Celery task ID for tracking
    },
    "metadata": {}  # Additional metadata
}
```

### API Response Format (Frontend - camelCase)

```typescript
interface Photo {
    id: string;
    filename: string;
    status: 'processing' | 'completed' | 'failed' | 'pending';
    createdAt: string;  // ISO string
    processedAt?: string;  // ISO string
    originalPath: string;
    fileHash: string;
    fileSize: number;
    sessionId?: string;
    aiAnalysis: AIAnalysis;
    metadata: Record<string, any>;
}

interface AIAnalysis {
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'not_available';
    queuedAt?: string;      // ISO string
    startedAt?: string;     // ISO string
    completedAt?: string;   // ISO string
    aestheticScore?: number;  // 1.0-10.0
    technicalScore?: number;  // 1.0-10.0 (optional)
    qualityLevel?: QualityLevel;
    confidence?: number;     // 0.0-1.0
    nimaResults?: NimaResults;
    modelInfo?: ModelInfo;
    error?: string;
    taskId?: string;
}

type QualityLevel = 'poor' | 'below_average' | 'average' | 'good' | 'excellent';

interface NimaResults {
    aesthetic: NimaAnalysis;
    technical?: NimaAnalysis;  // Optional
}

interface NimaAnalysis {
    qualityScore: number;
    qualityStd: number;
    qualityDistribution: number[];  // 10 values
    qualityLevel: QualityLevel;
    confidence: number;
    modelType: 'aesthetic' | 'technical';
    inferenceTime: number;
    imageSize: [number, number];  // [width, height]
}

interface ModelInfo {
    aestheticModel: string;
    technicalModel?: string;
    inferenceTime: number;
}
```

## Status Flow

### Photo Processing Status

1. **Upload** → `status: 'processing'`, `aiAnalysis.status: 'pending'`
2. **NIMA Queued** → `aiAnalysis.status: 'processing'`, `aiAnalysis.queuedAt: timestamp`
3. **NIMA Processing** → `aiAnalysis.startedAt: timestamp`
4. **NIMA Complete** → `status: 'completed'`, `aiAnalysis.status: 'completed'`, `aiAnalysis.completedAt: timestamp`
5. **NIMA Failed** → `status: 'failed'`, `aiAnalysis.status: 'failed'`, `aiAnalysis.error: message`

### WebSocket Events

```typescript
// NIMA analysis started
{
    type: "nima_analysis_started",
    data: {
        photoId: string,
        message: "Starting NIMA aesthetic analysis..."
    },
    timestamp: string
}

// NIMA analysis completed
{
    type: "nima_analysis_completed",
    data: {
        photoId: string,
        aestheticScore: number,
        qualityLevel: QualityLevel,
        confidence: number,
        message: string
    },
    timestamp: string
}

// Photo status changed
{
    type: "photo_status_changed",
    data: {
        photoId: string,
        status: PhotoStatus,
        message?: string
    },
    timestamp: string
}

// AI analysis progress (for future use)
{
    type: "ai_analysis_progress",
    data: {
        photoId: string,
        analysisType: "nima" | "object_detection" | "scene_analysis",
        progress: number,  // 0.0-1.0
        stage?: string
    },
    timestamp: string
}
```

## Celery Task Structure

### Task Naming Convention

- `tasks.ai_tasks.analyze_photo_nima` - Single photo NIMA analysis
- `tasks.ai_tasks.analyze_batch_nima` - Batch NIMA analysis
- `tasks.ai_tasks.reanalyze_photo_nima` - Re-analyze existing photo

### Task Queue Routing

- **Queue**: `ai_analysis` (GPU-intensive tasks)
- **Priority**: 5 (default)
- **Retry**: 3 attempts with exponential backoff
- **Timeout**: 300 seconds (5 minutes)

### Task Result Format

```python
{
    "photo_id": "string",
    "status": "completed|failed",
    "aesthetic_score": "float",
    "quality_level": "string",
    "confidence": "float",
    "processing_time": "float",
    "full_results": "dict"  # Complete AI analysis object
}
```

## Quality Score Interpretation

### Aesthetic Score (1.0-10.0)

- **9.0-10.0**: Excellent - Professional quality, exceptional composition
- **7.5-8.9**: Good - High quality, pleasing aesthetics
- **5.0-7.4**: Average - Acceptable quality, some issues
- **3.0-4.9**: Below Average - Noticeable quality issues
- **1.0-2.9**: Poor - Significant quality problems

### Confidence Score (0.0-1.0)

- **0.8-1.0**: High confidence - Reliable assessment
- **0.6-0.79**: Medium confidence - Generally reliable
- **0.4-0.59**: Low confidence - Less reliable
- **0.0-0.39**: Very low confidence - Unreliable assessment

## Database Updates

### Required Photo Service Methods

```python
# Update AI analysis data
update_photo_ai_analysis(photo_id: str, ai_analysis: dict)

# Update photo status with processing message
update_photo_status(photo_id: str, status: str, processing_message: str = None)

# Update photo metadata
update_photo_metadata(photo_id: str, metadata: dict)

# Get photos by AI analysis status
get_photos_by_ai_status(status: str) -> List[Photo]

# Get photos needing reanalysis
get_photos_for_reanalysis(force: bool = False) -> List[Photo]
```

## Frontend Integration

### Photo Display Updates

- Show NIMA score badge on photo cards
- Display quality level with color coding
- Show processing status during analysis
- Real-time updates via WebSocket

### Quality Level Color Coding

```css
.quality-excellent { color: #10b981; }  /* Green */
.quality-good { color: #3b82f6; }       /* Blue */
.quality-average { color: #f59e0b; }    /* Yellow */
.quality-below-average { color: #ef4444; } /* Red */
.quality-poor { color: #dc2626; }       /* Dark Red */
```

## Migration Strategy

### Phase 1: NIMA Integration (Current)
- ✅ Set up Celery infrastructure
- ✅ Implement NIMA analysis tasks
- ✅ Update upload flow to trigger analysis
- ✅ Add WebSocket notifications
- ⏳ Update data standards

### Phase 2: UI Integration
- Update photo grid to show NIMA scores
- Add filtering by quality level
- Implement real-time status updates

### Phase 3: Advanced Features
- Batch reanalysis tools
- Quality-based auto-culling
- Export filtering by quality
- Historical quality trends

## Error Handling

### Common Error Scenarios

1. **Model Loading Failed**: Retry with exponential backoff
2. **Image File Not Found**: Mark as failed, notify user
3. **GPU Memory Error**: Retry on different worker
4. **Timeout**: Retry with longer timeout
5. **Invalid Image Format**: Mark as failed with specific error

### Error Recovery

- Failed tasks are retried up to 3 times
- Permanent failures are logged and user is notified
- System continues processing other photos
- Failed photos can be manually requeued

## Performance Considerations

### Expected Processing Times

- **NIMA Aesthetic**: ~2-5 seconds per image
- **NIMA Technical**: ~2-5 seconds per image
- **Both**: ~4-8 seconds per image

### Scaling Guidelines

- 1 GPU worker can process ~10-15 images per minute
- Memory usage: ~2GB per worker
- Recommended: 2-4 workers per GPU
- Monitor queue depth and add workers as needed

## Compliance and Data Flow

All data transformations follow the established transformation middleware:
- Backend stores data in snake_case
- API responses convert to camelCase
- WebSocket events use camelCase
- Database queries use snake_case

This ensures consistency across the entire application stack.