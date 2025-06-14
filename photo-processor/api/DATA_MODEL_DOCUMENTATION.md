# Photo Processor Data Model Documentation

## Overview

This document describes the data models and API endpoints used in the Photo Processor application.

## Database Schema

### Photos Table
```sql
CREATE TABLE photos (
    id TEXT PRIMARY KEY,                    -- UUID for photo
    filename TEXT NOT NULL,                 -- Original filename
    status TEXT NOT NULL DEFAULT 'pending', -- Status: pending, processing, completed, failed, rejected
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- File paths
    original_path TEXT NULL,     -- Path to original file
    processed_path TEXT NULL,    -- Path to processed file
    thumbnail_path TEXT NULL,    -- Path to thumbnail file
    web_path TEXT NULL,         -- Path to web-optimized version
    
    -- Metadata
    file_hash TEXT NULL,        -- SHA256 hash of file
    file_size INTEGER DEFAULT 0,
    recipe_id TEXT NULL,        -- ID of recipe used for processing
    session_id TEXT NULL,       -- Upload session ID
    processing_time REAL DEFAULT 0,
    error_message TEXT NULL,
    metadata TEXT DEFAULT '{}'  -- JSON metadata
);
```

### AI Analysis Table
```sql
CREATE TABLE ai_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT NOT NULL,
    analysis_type TEXT NOT NULL DEFAULT 'nima',
    status TEXT NOT NULL DEFAULT 'pending',
    
    -- Timestamps
    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    
    -- NIMA specific fields
    aesthetic_score REAL NULL,
    technical_score REAL NULL,
    quality_level TEXT NULL,
    confidence REAL NULL,
    
    -- Results
    nima_results TEXT NULL,     -- JSON results
    model_info TEXT NULL,       -- JSON model info
    error TEXT NULL,
    task_id TEXT NULL,
    
    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
);
```

## API Models

### Photo Model
```python
class Photo(BaseModel):
    id: str                              # Photo ID (UUID)
    filename: str                        # Original filename
    original_path: str                   # Path to original file
    processed_path: Optional[str]        # Path to processed file
    thumbnail_path: Optional[str]        # Path to thumbnail
    web_path: Optional[str]             # Path to web version
    status: Literal["pending", "processing", "completed", "failed", "rejected"]
    created_at: datetime
    processed_at: Optional[datetime]
    file_size: int
```

### PhotoDetail Model
```python
class PhotoDetail(Photo):
    recipe_id: Optional[str]
    recipe_name: Optional[str]
    ai_analysis: Optional[Dict[str, Any]]
    processing_time: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

## API Response Transformations

### Path Transformations
When photos are returned via the API, file paths are transformed from absolute server paths to API URLs:

- `/app/data/thumbnails/photo_thumb.jpg` → `/api/files/thumbnails/photo_thumb.jpg`
- `/app/data/processed/photo.jpg` → `/api/files/processed/photo.jpg`
- `/app/data/web/photo_web.jpg` → `/api/files/web/photo_web.jpg`

### Field Name Transformations
All API responses use camelCase field names:

- `original_path` → `originalPath`
- `processed_path` → `processedPath`
- `thumbnail_path` → `thumbnailPath`
- `web_path` → `webPath`
- `created_at` → `createdAt`
- `processed_at` → `processedAt`
- `file_size` → `fileSize`

## API Endpoints

### Photo Endpoints
- `GET /api/photos` - List photos with pagination
- `GET /api/photos/{id}` - Get photo details
- `GET /api/photos/{id}/thumbnail` - Get thumbnail image (legacy)
- `GET /api/photos/{id}/preview` - Get preview image
- `GET /api/photos/{id}/download` - Download original
- `POST /api/photos/upload` - Upload single photo
- `DELETE /api/photos/{id}` - Delete photo

### File Serving Endpoints
- `GET /api/files/{type}/{path}` - Serve static files
  - Types: `thumbnails`, `processed`, `web`, `inbox`, `originals`
  - Example: `/api/files/thumbnails/abc123_thumb.jpg`

### Upload Endpoints
- `POST /api/upload/session` - Create upload session
- `POST /api/upload/session/{id}/file` - Upload file to session
- `POST /api/upload/session/{id}/complete` - Complete session
- `POST /api/upload/single` - Single file upload
- `POST /api/upload/batch` - Batch upload

### Processing Endpoints
- `GET /api/processing/queue` - Get queue status
- `POST /api/processing/queue/{id}` - Add to queue
- `DELETE /api/processing/queue/{id}` - Remove from queue
- `GET /api/processing/status/{id}` - Get processing status

## Frontend Integration

### Accessing Images
The frontend should use the paths returned in the API response:

```javascript
// API returns:
{
  "id": "abc123",
  "thumbnailPath": "/api/files/thumbnails/abc123_thumb.jpg",
  "webPath": "/api/files/web/abc123_web.jpg",
  "processedPath": "/api/files/processed/abc123.jpg"
}

// Frontend usage:
<img src={photo.thumbnailPath} />  // Uses the returned path directly
```

### Legacy Endpoints
The following endpoints are maintained for backward compatibility:
- `/api/photos/{id}/thumbnail` - Returns the thumbnail image directly
- `/api/photos/{id}/preview` - Returns the web-optimized version

## Processing Flow

1. **Upload**: File uploaded to `/app/data/inbox/`
2. **Processing**: File copied to `/app/data/processed/`
3. **Thumbnail**: Generated at `/app/data/thumbnails/{id}_thumb.jpg`
4. **Web Version**: Generated at `/app/data/web/{id}_web.jpg`
5. **Database Update**: All paths stored in database
6. **API Transform**: Paths converted to `/api/files/...` URLs

## Notes

- All file operations are asynchronous
- Thumbnails are 225x400px max
- Web versions are 1920px max (longest edge)
- File paths in database are absolute server paths
- API always returns URL paths for client access
- The `/api/files` endpoint validates paths for security