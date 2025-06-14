# Photo Processor Data Standards v1.0

## PURPOSE AND USAGE

This document defines the SINGLE SOURCE OF TRUTH for all data structures in the Photo Processor system. 

### MANDATORY COMPLIANCE RULES:
1. **ALL** data structures MUST conform to these standards
2. **ANY** changes to data structures MUST be reflected in this document FIRST
3. **ALL** affected files listed in this document MUST be updated when changes are made
4. **NO** data structure changes are allowed without updating this document

### How to Use This Document:
1. Before creating any new data structure, CHECK if it already exists here
2. Before modifying any data structure, UPDATE this document first
3. After updating this document, UPDATE all files listed in the "Used By" sections
4. Run the validation checklist after any changes

---

## NAMING CONVENTIONS

### Field Names
- **API (Python)**: snake_case (e.g., `file_size`, `created_at`)
- **Frontend (TypeScript)**: camelCase (e.g., `fileSize`, `createdAt`)
- **Database**: snake_case
- **WebSocket Events**: snake_case for event types, camelCase for data payloads

### Status Values
- Always lowercase
- Use consistent enums across the system

---

## CORE DATA STRUCTURES

### 1. Photo

**Description**: Basic photo information

**Backend Model** (Python):
```python
class Photo(BaseModel):
    id: str
    filename: str
    original_path: str
    processed_path: Optional[str]
    thumbnail_path: Optional[str]
    web_path: Optional[str]
    file_size: int  # in bytes
    created_at: datetime
    processed_at: Optional[datetime]
    status: Literal["pending", "processing", "completed", "failed", "rejected"]
    error_message: Optional[str]
    processing_time: Optional[float]  # in seconds
    recipe_used: Optional[str]  # recipe_id
```

**Frontend Interface** (TypeScript):
```typescript
interface Photo {
    id: string
    filename: string
    originalPath: string
    processedPath?: string
    thumbnailPath?: string
    webPath?: string
    fileSize: number  // in bytes
    createdAt: string  // ISO 8601
    processedAt?: string  // ISO 8601
    status: 'pending' | 'processing' | 'completed' | 'failed' | 'rejected'
    errorMessage?: string
    processingTime?: number  // in seconds
    recipeUsed?: string  // recipe_id
}
```

**Transformation Rules**:
- Convert snake_case to camelCase
- Convert datetime objects to ISO 8601 strings
- File paths: Convert absolute paths to relative API URLs

**Used By**:
- Backend: 
  - `/api/models/photo.py`
  - `/api/services/photo_service_v2.py`
  - `/api/routes/photos.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/services/api.ts`
  - `/src/pages/Photos.tsx`
  - `/src/components/photos/PhotoGrid.tsx`

---

### 2. PhotoDetail

**Description**: Extended photo information with metadata and AI analysis

**Backend Model** (Python):
```python
class PhotoDetail(Photo):
    metadata: Optional[Dict[str, Any]]
    processing_history: List[ProcessingRecord]
    recipe_id: Optional[str]
    recipe_name: Optional[str]  # Added for frontend compatibility
    ai_analysis: Optional[AIAnalysisResult]
    immich_asset_id: Optional[str]
```

**Frontend Interface** (TypeScript):
```typescript
interface PhotoDetail extends Photo {
    metadata?: Record<string, any>
    processingHistory?: ProcessingRecord[]
    recipeId?: string
    recipeName?: string
    aiAnalysis?: AIAnalysisResult
    immichAssetId?: string
}
```

**Used By**:
- Backend:
  - `/api/models/photo.py`
  - `/api/services/photo_service_v2.py`
  - `/api/routes/photos.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/components/photos/PhotoDialog.tsx`
  - `/src/hooks/usePhoto.ts`

---

### 3. QueueItem

**Description**: Item in the processing queue

**Backend Model** (Python):
```python
class QueueItem(BaseModel):
    photo_id: str
    filename: str
    position: int  # 1-based position in queue
    created_at: datetime
    priority: Literal["low", "normal", "high"]
    recipe_id: Optional[str]
    manual_approval: bool = False
    estimated_time: float = 30.0  # seconds
    # Runtime fields (added by service)
    status: Optional[Literal["pending", "processing", "completed", "failed"]]
    started_at: Optional[datetime]
    progress: Optional[float]  # 0-100
```

**Frontend Interface** (TypeScript):
```typescript
interface QueueItem {
    photoId: string
    filename: string
    position: number  // 1-based position in queue
    createdAt: string  // ISO 8601
    priority: 'low' | 'normal' | 'high'
    recipeId?: string
    manualApproval?: boolean
    estimatedTime?: number  // seconds
    // Runtime fields
    status?: 'pending' | 'processing' | 'completed' | 'failed'
    startedAt?: string  // ISO 8601
    progress?: number  // 0-100
}
```

**Used By**:
- Backend:
  - `/api/models/processing.py`
  - `/api/services/processing_service_v2.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/pages/Processing.tsx`

---

### 4. QueueStatus

**Description**: Complete queue status with statistics

**Backend Model** (Python):
```python
class QueueStatus(BaseModel):
    pending: List[QueueItem]
    processing: List[QueueItem]
    completed: List[QueueItem]  # last 20
    is_paused: bool
    stats: ProcessingStatus
```

**Frontend Interface** (TypeScript):
```typescript
interface QueueStatus {
    pending: QueueItem[]
    processing: QueueItem[]
    completed: QueueItem[]  // last 20
    isPaused: boolean
    stats: ProcessingStatus
    // Computed fields (frontend only)
    total?: number  // sum of all items
    currentItem?: QueueItem  // first item in processing array
}
```

**Transformation Rules**:
- `total` is computed as: `pending.length + processing.length + completed.length`
- `currentItem` is computed as: `processing[0]` or `undefined`

**Used By**:
- Backend:
  - `/api/models/processing.py`
  - `/api/services/processing_service_v2.py`
  - `/api/routes/processing.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/pages/Processing.tsx`

---

### 5. ProcessingStatus

**Description**: Current processing statistics

**Backend Model** (Python):
```python
class ProcessingStatus(BaseModel):
    is_paused: bool
    current_photo: Optional[Dict[str, str]]  # {id, filename}
    queue_length: int
    processing_rate: float  # photos per minute
    average_time: float  # seconds
    errors_today: int
```

**Frontend Interface** (TypeScript):
```typescript
interface ProcessingStatus {
    isPaused: boolean
    currentPhoto?: {
        id: string
        filename: string
    }
    queueLength: number
    processingRate: number  // photos per minute
    averageTime: number  // seconds
    errorsToday: number
    // Computed fields
    isRunning?: boolean  // computed as !isPaused
    progress?: number  // overall progress percentage
}
```

**Used By**:
- Backend:
  - `/api/models/processing.py`
  - `/api/services/processing_service_v2.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/pages/Processing.tsx`
  - `/src/components/dashboard/StatsOverview.tsx`

---

### 6. Recipe

**Description**: Processing recipe configuration

**Backend Model** (Python):
```python
class Recipe(BaseModel):
    id: str
    name: str
    description: str
    operations: List[ProcessingOperation]  # Note: backend uses "operations"
    created_at: datetime
    updated_at: datetime
    is_preset: bool = False
    usage_count: int = 0
    category: Optional[str]
```

**Frontend Interface** (TypeScript):
```typescript
interface Recipe {
    id: string
    name: string
    description: string
    steps: ProcessingStep[]  // Note: frontend uses "steps"
    createdAt: string  // ISO 8601
    updatedAt: string  // ISO 8601
    isPreset: boolean
    usageCount: number
    category?: string
}
```

**CRITICAL**: Backend uses `operations` while frontend uses `steps`. This MUST be transformed.

**Used By**:
- Backend:
  - `/api/models/recipe.py`
  - `/api/services/recipe_service.py`
  - `/recipe_storage.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/pages/Recipes.tsx`
  - `/src/pages/RecipeEditor.tsx`

---

### 7. ProcessingOperation / ProcessingStep

**Description**: Individual operation in a recipe

**Backend Model** (Python):
```python
class ProcessingOperation(BaseModel):
    type: str  # "crop", "rotate", "enhance", etc.
    parameters: Dict[str, Any]
    order: int
    source: str  # "ai" or "user"
    enabled: bool = True
```

**Frontend Interface** (TypeScript):
```typescript
interface ProcessingStep {
    operation: string  // Note: frontend uses "operation" for backend's "type"
    parameters: Record<string, any>
    enabled: boolean
    // Missing from frontend: order, source
}
```

**Used By**:
- Backend:
  - `/schemas.py`
  - `/recipe_storage.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/components/recipes/StepEditor.tsx`

---

### 8. AIAnalysisResult

**Description**: AI analysis results for a photo

**Backend Model** (Python):
```python
class AIAnalysisResult(BaseModel):
    subjects: List[str]
    composition_score: float  # 0-10
    quality_score: float  # 0-10
    suggested_crop: Optional[CropSuggestion]
    recommendations: List[str]
    color_analysis: ColorAnalysis
    scene_detection: SceneDetection
    face_detection: Optional[List[FaceDetection]]
    # Metadata
    analysis_version: str
    processing_time: float
    model_versions: Dict[str, str]
```

**Frontend Interface** (TypeScript):
```typescript
interface AIAnalysisResult {
    subjects: string[]
    compositionScore: number  // 0-10
    qualityScore: number  // 0-10
    suggestedCrop?: CropSuggestion
    recommendations: string[]
    colorAnalysis?: ColorAnalysis
    sceneDetection?: SceneDetection
    faceDetection?: FaceDetection[]
    // Metadata often omitted in frontend
    status?: 'available' | 'not_available' | 'error'  // Frontend addition
}
```

**Used By**:
- Backend:
  - `/schemas.py`
  - `/ai_analyzer.py`
- Frontend:
  - `/src/types/api.ts`
  - `/src/components/photos/PhotoDialog.tsx`

---

### 9. WebSocket Events

**Description**: Real-time event structure

**Standard Format**:
```typescript
interface WebSocketEvent {
    type: string  // snake_case event type
    data: any  // Event-specific payload
    timestamp: string  // ISO 8601
}
```

**Event Types and Payloads**:

1. **processing_started**
   ```typescript
   {
       photoId: string
       recipeId?: string
       filename: string
   }
   ```

2. **processing_completed**
   ```typescript
   {
       photoId: string
       success: boolean
       processedPath?: string
       filename: string
   }
   ```

3. **queue_updated**
   ```typescript
   {
       pending: number
       processing: number
       completed: number
   }
   ```

**Used By**:
- Backend:
  - `/api/services/websocket_manager.py`
- Frontend:
  - `/src/providers/WebSocketProvider.tsx`
  - `/src/hooks/useWebSocket.ts`

---

### 10. Pagination Response

**Description**: Paginated list response

**Backend Model** (Python):
```python
class PaginatedResponse(BaseModel):
    items: List[Any]  # Or specific type
    total: int
    page: int
    page_size: int
    pages: int
```

**Frontend Interface** (TypeScript):
```typescript
interface PaginatedResponse<T> {
    items: T[]  // Sometimes "photos" for photo endpoints
    total: number
    page: number
    pageSize: number  // Note: camelCase
    pages: number
}
```

**Used By**:
- Backend: All list endpoints
- Frontend: All paginated lists

---

## VALIDATION CHECKLIST

After ANY changes to data structures:

1. [ ] Update this document FIRST
2. [ ] Update all backend models
3. [ ] Update all frontend types
4. [ ] Update transformation logic in API routes
5. [ ] Update WebSocket event handlers
6. [ ] Test data flow end-to-end
7. [ ] Verify no TypeScript errors
8. [ ] Verify no Python type errors
9. [ ] Update any affected tests

---

## TRANSFORMATION UTILITIES

### Backend to Frontend (Python):
```python
def to_frontend_format(obj: BaseModel) -> dict:
    """Convert backend model to frontend format"""
    data = obj.dict()
    # Convert snake_case to camelCase
    # Convert datetime to ISO strings
    # Convert file paths to API URLs
    return transformed_data
```

### Frontend to Backend (TypeScript):
```typescript
function toBackendFormat(obj: any): any {
    // Convert camelCase to snake_case
    // Convert ISO strings to datetime format
    // Convert API URLs to file paths
    return transformedData
}
```

---

## MIGRATION NOTES

### Priority Issues to Fix:
1. **HIGH**: Recipe operations vs steps mismatch
2. **HIGH**: QueueStatus structure differences
3. **MEDIUM**: Pagination field naming
4. **MEDIUM**: WebSocket event consistency
5. **LOW**: AIAnalysisResult metadata fields

### Breaking Changes:
- Changing Recipe.operations to Recipe.steps will break existing recipes
- QueueStatus changes will affect Processing page
- Any snake_case to camelCase conversions need API middleware

---

## CURRENT IMPLEMENTATION STATUS

### Transformation Layer Active (v1.1)
As of January 7, 2024, a transformation layer is active in `/api/middleware/transform.py` that handles:
- snake_case ↔ camelCase conversion
- datetime ↔ ISO string conversion
- Recipe operations ↔ steps field mapping
- Computed field additions (QueueStatus.total, currentItem)

### Routes Using Transformation
- ✅ `/api/processing/*` - All routes transformed
- ✅ `/api/photos/*` - All routes transformed
- ✅ `/api/recipes/*` - All routes transformed
- ✅ WebSocket events - All using standard format
- ⚠️ `/api/stats/*` - Not yet implemented
- ⚠️ `/api/upload/*` - Partially implemented

## VERSION HISTORY

- v1.1 (2024-01-07): Transformation layer implemented
  - Added middleware/transform.py
  - Updated all main routes
  - Standardized WebSocket events
  - Fixed frontend type definitions
  - See DATA_STANDARDS_UPDATE_LOG.md for details

- v1.0 (2024-01-07): Initial comprehensive documentation
  - Documented all major data structures
  - Identified key inconsistencies
  - Created transformation rules
  - Listed all affected files

---

## ENFORCEMENT

This document is enforced by:
1. Code review requirements
2. CI/CD type checking
3. Runtime validation
4. Developer discipline

**Remember**: This document is the CONTRACT. Break the contract, break the system.