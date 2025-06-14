# Photo Processor API Endpoints

Base URL: `http://localhost:8000`

## Core Endpoints

- `GET /` - API root info
- `GET /health` - Health check  
- `WS /ws` - WebSocket for real-time updates
- `GET /docs` - Swagger UI documentation
- `GET /openapi.json` - OpenAPI specification

## Photos (`/api/photos`)

- `GET /api/photos/` - List photos with pagination
  - Query params: `page`, `page_size`, `status`, `sort_by`, `order`
- `GET /api/photos/{photo_id}` - Get photo details
- `GET /api/photos/{photo_id}/comparison` - Get before/after comparison
- `GET /api/photos/{photo_id}/ai-analysis` - Get AI analysis results
- `POST /api/photos/{photo_id}/reprocess` - Reprocess a photo
- `POST /api/photos/upload` - Legacy upload endpoint
- `DELETE /api/photos/{photo_id}` - Delete a photo

## Processing (`/api/processing`)

- `GET /api/processing/status` - Current processing status
- `GET /api/processing/queue` - Queue status
- `GET /api/processing/settings` - Processing settings
- `PUT /api/processing/settings` - Update settings
- `PUT /api/processing/pause` - Pause processing
- `PUT /api/processing/resume` - Resume processing
- `POST /api/processing/batch` - Batch process photos
- `POST /api/processing/approve/{photo_id}` - Approve processing
- `POST /api/processing/reject/{photo_id}` - Reject processing
- `POST /api/processing/reorder` - Reorder queue
- `DELETE /api/processing/queue/{photo_id}` - Remove from queue

## Recipes (`/api/recipes`)

- `GET /api/recipes/` - List all recipes
- `GET /api/recipes/presets/` - Get preset recipes
- `GET /api/recipes/{recipe_id}` - Get recipe details
- `POST /api/recipes/` - Create new recipe
- `PUT /api/recipes/{recipe_id}` - Update recipe
- `DELETE /api/recipes/{recipe_id}` - Delete recipe
- `POST /api/recipes/{recipe_id}/apply` - Apply to photos
- `POST /api/recipes/{recipe_id}/duplicate` - Duplicate recipe
- `POST /api/recipes/{recipe_id}/preview` - Preview on photo

## Upload (`/api/upload`)

- `POST /api/upload/session` - Create upload session
- `POST /api/upload/session/{session_id}/file` - Upload file to session
- `GET /api/upload/session/{session_id}/progress` - Get upload progress
- `POST /api/upload/session/{session_id}/complete` - Complete session
- `DELETE /api/upload/session/{session_id}` - Cancel session
- `POST /api/upload/single` - Single file upload (legacy)
- `POST /api/upload/batch` - Batch upload
- `GET /api/upload/stats` - Upload statistics
- `GET /api/upload/supported-formats` - Supported file types
- `POST /api/upload/cleanup` - Clean up old sessions

## Statistics (`/api/stats`)

- `GET /api/stats/dashboard` - Dashboard statistics
- `GET /api/stats/processing` - Processing statistics
- `GET /api/stats/storage` - Storage statistics
- `GET /api/stats/activity` - Recent activity
- `GET /api/stats/performance` - Performance metrics
- `GET /api/stats/trends` - Usage trends
- `GET /api/stats/ai-performance` - AI model performance
- `GET /api/stats/errors` - Error statistics

## Static Files

- `GET /images/originals/{filename}` - Original images
- `GET /images/processed/{filename}` - Processed images
- `GET /images/thumbnails/{filename}` - Thumbnail images

## Notes

- All list endpoints use trailing slashes (e.g., `/api/photos/`)
- Single resource endpoints don't use trailing slashes (e.g., `/api/photos/{id}`)
- WebSocket endpoint at `/ws` for real-time updates
- No authentication required (local network use)