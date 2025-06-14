# Photo Processor Data Standards Document

## Overview

This document defines the comprehensive data standards for the Photo Processor system, including all data types, their exact shapes, transformations, and identified inconsistencies between frontend and backend implementations.

## Critical Data Type Mismatches and Inconsistencies

### 1. Photo/PhotoDetail Objects

#### Backend (Python - models/photo.py)
```python
class Photo:
    id: str                    # Hash-based ID
    filename: str
    original_path: str
    processed_path: Optional[str]
    thumbnail_path: Optional[str]
    web_path: Optional[str]
    status: Literal["pending", "processing", "completed", "failed", "rejected"]
    created_at: datetime       # Python datetime object
    processed_at: Optional[datetime]
    file_size: int

class PhotoDetail(Photo):
    recipe_id: Optional[str]
    recipe_name: Optional[str]  # Frontend doesn't have this field
    ai_analysis: Optional[Dict[str, Any]]
    processing_time: Optional[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface Photo {
    id: string
    filename: string
    original_path: string
    processed_path?: string
    thumbnail_path?: string
    web_path?: string
    file_size: number
    created_at: string          // String format, not Date
    processed_at?: string       // String format, not Date
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'rejected'
}

interface PhotoDetail extends Photo {
    metadata?: Record<string, any>
    processing_history?: ProcessingRecord[]  // Backend doesn't have this
    recipe_id?: string
    ai_analysis?: AIAnalysis    // Different structure than backend
    error_message?: string
    processing_time?: number
    // Missing: recipe_name
}
```

**Inconsistencies:**
- Backend has `recipe_name` in PhotoDetail, frontend doesn't
- Frontend has `processing_history` field, backend doesn't
- Date fields are datetime objects in backend, strings in frontend
- AI analysis structure differs (backend uses generic Dict, frontend has typed AIAnalysis)

### 2. Queue/QueueStatus/QueueItem Structures

#### Backend (Python - models/processing.py)
```python
class QueueItem:
    photo_id: str
    filename: Optional[str]
    position: Optional[int]
    added_at: Optional[datetime]    # Two different time fields
    created_at: Optional[datetime]
    priority: Literal["low", "normal", "high"]
    recipe_id: Optional[str]
    manual_approval: bool
    estimated_time: Optional[float]

class QueueStatus:
    pending: List[QueueItem]
    processing: List[QueueItem]
    completed: List[QueueItem]
    is_paused: bool
    stats: ProcessingStatus
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface QueueItem {
    photo_id: string
    filename?: string
    position?: number
    added_at?: string      // String, not datetime
    created_at?: string    // String, not datetime
    priority: 'low' | 'normal' | 'high'
    recipe_id?: string
    manual_approval?: boolean
    estimated_time?: number
    // Additional fields for current_item
    status?: 'pending' | 'processing' | 'completed' | 'failed'
    started_at?: string
    progress?: number
}

interface QueueStatus {
    pending: QueueItem[]
    processing: QueueItem[]
    completed: QueueItem[]
    is_paused: boolean
    stats: ProcessingStatus
    // Computed client-side
    total?: number
    current_item?: QueueItem
}
```

**Inconsistencies:**
- Frontend adds extra fields to QueueItem (status, started_at, progress) that backend doesn't have
- Frontend computes additional fields (total, current_item) client-side
- Datetime fields are objects in backend, strings in frontend

### 3. Processing Status and Statistics

#### Backend (Python - models/processing.py)
```python
class ProcessingStatus:
    is_paused: bool            # Note: both is_paused and is_running
    current_photo: Optional[Dict[str, Any]]
    queue_length: int
    processing_rate: float
    average_time: float
    errors_today: int
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface ProcessingStatus {
    is_running?: boolean       // Frontend has is_running
    is_paused?: boolean       
    current_photo?: any
    queue_length?: number
    progress?: number          // Frontend only
    processing_rate?: number
    average_time?: number
    estimated_completion?: string  // Frontend only
    errors_today?: number
}
```

**Inconsistencies:**
- Backend doesn't have `is_running`, `progress`, or `estimated_completion` fields
- All fields are optional in frontend but required in backend

### 4. Dashboard Statistics

#### Backend Response (from stats routes)
```python
# Actual response from /stats/dashboard endpoint
{
    "total_photos": int,
    "processed_today": int,
    "processing_rate": float,
    "storage_used": {
        "total": {"bytes": int, "formatted": str},
        "originals": {"bytes": int, "formatted": str},
        "processed": {"bytes": int, "formatted": str},
        "disk": {"total": str, "used": str, "free": str, "percentUsed": float}
    },
    "queue_length": int,
    "success_rate": float,
    "recent_activity": List[Dict],
    "system_status": {
        "processing": str,
        "aiModels": str,
        "storage": str
    }
}
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface DashboardStats {
    // Support both snake_case and camelCase from API
    total_photos?: number
    totalPhotos?: number
    processed_today?: number
    processedToday?: number
    processing_rate?: number
    processingRate?: number
    storage_used?: number      // Simple number in some places
    storageUsed?: {            // Complex object in others
        total?: { bytes: number; formatted: string }
        originals?: { bytes: number; formatted: string }
        processed?: { bytes: number; formatted: string }
        disk?: { total: string; used: string; free: string; percentUsed: number }
    }
    queue_length?: number
    inQueue?: number           // Duplicate field name
    success_rate?: number
    successRate?: number
    averageProcessingTime?: number  // Not in backend
    failedToday?: number           // Not in backend
    recentActivity?: any[]
    systemStatus?: {
        processing: string
        aiModels: string
        storage: string
    }
}
```

**Inconsistencies:**
- Frontend supports both snake_case and camelCase for compatibility
- `storage_used`/`storageUsed` has inconsistent types (number vs object)
- Frontend has extra fields not provided by backend
- Duplicate field names for same data (queue_length vs inQueue)

### 5. Recipe Structures

#### Backend (Python)
```python
# Recipe structure in services
{
    "id": str,
    "name": str,
    "description": str,
    "operations": List[Dict],     # Note: "operations"
    "style_preset": str,
    "processing_config": Dict,
    "is_default": bool,
    "created_at": datetime,
    "updated_at": datetime,
    "is_preset": bool,
    "usage_count": int
}
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface Recipe {
    id: string
    name: string
    description: string
    steps: ProcessingStep[]      // Note: "steps" not "operations"
    created_at: string
    updated_at: string
    is_preset: boolean
    usage_count: number
    // Missing: style_preset, processing_config, is_default
}

interface RecipePreset {
    id: string
    name: string
    description: string
    category?: string
    steps?: ProcessingStep[]     // Optional
    operations?: ProcessingStep[] // Also accepts operations
}
```

**Inconsistencies:**
- Backend uses "operations", frontend uses "steps" (though RecipePreset supports both)
- Backend has additional fields (style_preset, processing_config, is_default)
- Date fields are datetime in backend, strings in frontend

### 6. WebSocket Event Formats

#### Backend (Python - websocket_manager.py)
```python
# Processing events
{
    "type": "processing_started",
    "data": {
        "photo_id": str,
        "filename": str,
        "recipe_id": Optional[str]
    },
    "timestamp": str  # ISO format
}

# Queue events
{
    "type": "queue_updated",
    "pending": int,           # Direct properties
    "processing": int,
    "completed": int,
    "timestamp": str
}

# AI events use camelCase
{
    "type": "culling_started",
    "photoIds": List[str],    # camelCase
    "count": int,
    "timestamp": str
}
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface WebSocketEvent {
    type: 'processing_started' | 'processing_completed' | 'processing_failed' | 
          'queue_updated' | 'stats_updated' | 'photo_uploaded' | 'recipe_updated' |
          'connection'
    data: any                  // Generic, no structure
    timestamp: string
}
```

**Inconsistencies:**
- Backend mixes naming conventions (snake_case vs camelCase)
- Queue events have data at root level, not in `data` property
- Frontend doesn't define specific event data structures
- AI events use different naming convention than other events

### 7. Pagination Response Format

#### Backend (Python - models/photo.py)
```python
class PhotoList:
    photos: List[Photo]
    total: int
    page: int
    page_size: int         # snake_case
    has_next: bool
    has_prev: bool
```

#### Frontend (TypeScript - types/api.ts)
```typescript
interface PaginatedResponse<T> {
    items?: T[]            // Generic version
    photos?: T[]           // Photos API specific
    total: number
    page: number
    per_page?: number      // Different name
    page_size?: number     // Also accepts backend name
    pages?: number         // Additional field
    has_next?: boolean
    has_prev?: boolean
}
```

**Inconsistencies:**
- Backend uses `photos`, frontend supports both `items` and `photos`
- Backend uses `page_size`, frontend uses `per_page` (but accepts both)
- Frontend adds computed `pages` field

### 8. File Paths and Naming Conventions

#### Backend
- Uses absolute file system paths: `/data/photos/original/abc123.jpg`
- Paths stored in database with full system path
- Hash-based IDs for photos

#### Frontend
- Expects relative URLs: `/api/photos/abc123/original`
- Must transform paths to API endpoints
- No direct file system access

**Transformation Required:**
- Backend paths must be converted to API URLs for frontend
- Frontend cannot use raw file paths from backend

### 9. Status Enums and Values

#### Inconsistent Status Values Across System

**Photo Status:**
- Backend/Frontend: `pending`, `processing`, `completed`, `failed`, `rejected`

**Processing Status:**
- Sometimes uses `is_paused` (boolean)
- Sometimes uses `status` field with string values
- Mix of `is_running` and `is_paused` booleans

**System Status:**
- String values: `"running"`, `"paused"`, `"error"`
- No standardized enum

### 10. Timestamps and Date Formats

#### Backend
- Uses Python `datetime` objects
- Serializes to ISO 8601 format: `2024-01-15T10:30:00.000Z`
- Timezone aware in some places, naive in others

#### Frontend
- Expects string timestamps
- Parses ISO 8601 format
- May display in local timezone

**Issues:**
- Inconsistent timezone handling
- No standardized format documentation
- Mix of field names: `created_at`, `createdAt`, `timestamp`

## Recommendations for Standardization

### 1. Unified Naming Convention
- **Choose one:** Stick to snake_case throughout the API
- Transform at the frontend boundary if needed
- Document the chosen convention

### 2. Consistent Date Handling
- Always use ISO 8601 format with timezone
- Standardize on UTC for storage
- Document timezone expectations

### 3. Standardized Response Envelopes
```typescript
interface StandardResponse<T> {
    data: T
    timestamp: string
    version: string
}

interface StandardListResponse<T> {
    items: T[]
    pagination: {
        page: number
        page_size: number
        total: number
        total_pages: number
    }
}
```

### 4. Explicit WebSocket Event Types
Define specific interfaces for each event type instead of using `any`:

```typescript
interface ProcessingStartedEvent {
    type: 'processing_started'
    data: {
        photo_id: string
        filename: string
        recipe_id?: string
    }
    timestamp: string
}
```

### 5. API Versioning
- Add version prefix to API routes: `/api/v1/photos`
- Include version in response headers
- Plan for backward compatibility

### 6. Field Alignment
For each data type, ensure:
- Backend and frontend have same fields
- Optional vs required fields match
- No duplicate fields for same data
- Clear documentation of computed fields

### 7. Error Response Standardization
```typescript
interface StandardError {
    error: {
        code: string
        message: string
        details?: any
    }
    timestamp: string
}
```

## Implementation Priority

1. **High Priority** - Fix naming inconsistencies in WebSocket events
2. **High Priority** - Standardize photo status enums
3. **Medium Priority** - Align Recipe structure (operations vs steps)
4. **Medium Priority** - Fix dashboard stats field names
5. **Low Priority** - Add missing fields to maintain feature parity

## Testing Checklist

- [ ] Verify all API responses match documented types
- [ ] Test WebSocket events with actual payloads
- [ ] Validate timezone handling in dates
- [ ] Check pagination consistency across endpoints
- [ ] Ensure error responses follow standard format
- [ ] Test data transformations in frontend
- [ ] Verify optional vs required fields
- [ ] Test with missing/null values

## Migration Strategy

1. **Phase 1**: Document current state (this document)
2. **Phase 2**: Add compatibility layer in frontend
3. **Phase 3**: Gradually update backend endpoints
4. **Phase 4**: Remove compatibility code
5. **Phase 5**: Update all clients to new standard