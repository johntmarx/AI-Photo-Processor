# Photo Processor Architecture Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Photo Processor Suite                           │
├─────────────────────────┬────────────────────┬─────────────────────────┤
│      Frontend UI        │   Core Processor   │    Storage Layer        │
│  ┌─────────────────┐   │ ┌────────────────┐│  ┌──────────────────┐  │
│  │  React Dashboard│   │ │ File Monitor   ││  │ Local File Store │  │
│  │  - Live Preview │   │ │ - Hot Folders  ││  │ - Originals      │  │
│  │  - Queue Mgmt   │   │ │ - Watch API    ││  │ - Processed      │  │
│  │  - Settings     │   │ └────────────────┘│  │ - Recipes        │  │
│  └─────────────────┘   │ ┌────────────────┐│  └──────────────────┘  │
│  ┌─────────────────┐   │ │ Process Engine ││  ┌──────────────────┐  │
│  │ WebSocket Server│   │ │ - Queue Manager││  │ Immich Client    │  │
│  │ - Real-time     │   │ │ - Worker Pool  ││  │ - API v2         │  │
│  │ - Status Updates│   │ │ - Recipe Engine││  │ - Batch Upload   │  │
│  └─────────────────┘   │ └────────────────┘│  │ - Metadata Sync  │  │
│                         │ ┌────────────────┐│  └──────────────────┘  │
│                         │ │ AI Pipeline    ││  ┌──────────────────┐  │
│                         │ │ - Detection    ││  │ External Storage │  │
│                         │ │ - Segmentation ││  │ - S3 Compatible  │  │
│                         │ │ - Enhancement  ││  │ - NAS/SMB        │  │
│                         │ └────────────────┘│  └──────────────────┘  │
└─────────────────────────┴────────────────────┴─────────────────────────┘
```

## Core Components Detailed Design

### 1. File Management Service

```python
class FileManagementService:
    """
    Handles all file operations with original preservation
    """
    
    def __init__(self):
        self.original_store = OriginalFileStore()
        self.processed_store = ProcessedFileStore()
        self.recipe_store = RecipeStore()
    
    def process_file(self, file_path: Path) -> ProcessResult:
        # 1. Calculate file hash for deduplication
        file_hash = self.calculate_hash(file_path)
        
        # 2. Check if already processed
        if self.is_processed(file_hash):
            return ProcessResult(status="duplicate", hash=file_hash)
        
        # 3. Store original file
        original_id = self.original_store.store(file_path, file_hash)
        
        # 4. Create processing recipe
        recipe = ProcessingRecipe(
            original_id=original_id,
            timestamp=datetime.now(),
            operations=[]
        )
        
        # 5. Process file (non-destructive)
        processed_data = self.apply_processing(file_path, recipe)
        
        # 6. Store processed version
        processed_id = self.processed_store.store(
            processed_data, 
            original_id,
            recipe
        )
        
        # 7. Save recipe for future reference
        self.recipe_store.save(recipe)
        
        return ProcessResult(
            status="success",
            original_id=original_id,
            processed_id=processed_id,
            recipe_id=recipe.id
        )
```

### 2. Processing Recipe System

```python
@dataclass
class ProcessingRecipe:
    """
    Stores all processing operations for reproducibility
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    original_id: str
    timestamp: datetime
    operations: List[ProcessingOperation]
    ai_metadata: Dict[str, Any]
    user_overrides: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
    
    def replay(self, original_file: Path) -> ProcessedImage:
        """Replay all operations on original file"""
        image = load_image(original_file)
        for operation in self.operations:
            image = operation.apply(image)
        return image

@dataclass
class ProcessingOperation:
    """
    Individual processing operation
    """
    type: str  # crop, rotate, enhance, etc.
    parameters: Dict[str, Any]
    timestamp: datetime
    source: str  # 'ai', 'user', 'preset'
```

### 3. Dual Upload Strategy

```python
class ImmichUploadService:
    """
    Handles uploading both original and processed files to Immich
    """
    
    def upload_with_original(
        self, 
        original_path: Path,
        processed_path: Path,
        recipe: ProcessingRecipe,
        metadata: Dict[str, Any]
    ) -> UploadResult:
        # 1. Upload original file
        original_asset = self.immich_client.upload_asset(
            file_path=original_path,
            album_name="Originals",
            metadata={
                **metadata,
                "is_original": True,
                "processing_recipe_id": recipe.id
            }
        )
        
        # 2. Upload processed file
        processed_asset = self.immich_client.upload_asset(
            file_path=processed_path,
            album_name="Processed",
            metadata={
                **metadata,
                "is_processed": True,
                "original_asset_id": original_asset.id,
                "processing_recipe": recipe.to_json(),
                "ai_analysis": recipe.ai_metadata
            }
        )
        
        # 3. Link assets in Immich (if API supports)
        self.immich_client.link_assets(
            original_asset.id,
            processed_asset.id,
            relationship="processed_from"
        )
        
        return UploadResult(
            original_asset_id=original_asset.id,
            processed_asset_id=processed_asset.id,
            recipe_stored=True
        )
```

### 4. Frontend Communication Layer

```python
class ProcessingWebSocketHandler:
    """
    Real-time communication with frontend
    """
    
    async def handle_connection(self, websocket):
        await websocket.accept()
        
        # Subscribe to processing events
        async for event in self.processing_queue.events():
            await websocket.send_json({
                "type": event.type,
                "data": {
                    "file_id": event.file_id,
                    "status": event.status,
                    "progress": event.progress,
                    "preview_url": event.preview_url,
                    "ai_suggestions": event.ai_suggestions
                }
            })
    
    async def handle_user_action(self, action):
        """Handle user overrides and approvals"""
        if action.type == "approve_processing":
            await self.processing_queue.approve(action.file_id)
        elif action.type == "modify_crop":
            await self.processing_queue.update_recipe(
                action.file_id,
                {"crop": action.crop_params}
            )
```

### 5. AI Pipeline Integration

```python
class ModernAIAnalyzer:
    """
    Integrates state-of-the-art AI models
    """
    
    def __init__(self):
        # Initialize models
        self.detector = self.load_rtdetr()  # RT-DETR for object detection
        self.segmenter = self.load_sam2()   # SAM2 for segmentation
        self.embedder = self.load_clip()    # CLIP for semantic search
        self.aesthetic_scorer = self.load_nima()  # NIMA for quality
        self.vlm = self.load_qwen_vl()      # Qwen2.5-VL for analysis
    
    async def analyze_image(self, image_path: Path) -> AIAnalysis:
        # 1. Object Detection
        detections = await self.detector.detect(image_path)
        
        # 2. Segmentation for main subjects
        segments = await self.segmenter.segment(
            image_path, 
            prompts=detections.boxes
        )
        
        # 3. Aesthetic quality scoring
        quality_scores = await self.aesthetic_scorer.score(image_path)
        
        # 4. Generate embeddings for search
        embeddings = await self.embedder.encode(image_path)
        
        # 5. Comprehensive VLM analysis
        vlm_analysis = await self.vlm.analyze(
            image_path,
            prompt="""Analyze this image and provide:
            1. Composition recommendations
            2. Color correction needs
            3. Suggested crops
            4. Content description
            5. Technical quality assessment"""
        )
        
        return AIAnalysis(
            objects=detections,
            segments=segments,
            quality=quality_scores,
            embeddings=embeddings,
            recommendations=vlm_analysis
        )
```

## Data Flow

### 1. New Photo Ingestion
```
New Photo → Hash Check → Store Original → AI Analysis → Generate Recipe → 
Process Image → Store Processed → Upload Both → Update Frontend
```

### 2. Manual Review Flow
```
Frontend Request → Load Preview → Show AI Suggestions → User Modifies → 
Update Recipe → Reprocess → Store New Version → Upload Updates
```

### 3. Batch Processing Flow
```
Select Photos → Load Recipes → Apply Preset → Queue Processing → 
Monitor Progress → Review Results → Bulk Upload
```

## Storage Strategy

### Local Storage Structure
```
/photo-processor-data/
├── originals/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── {hash}_DSC0001.NEF
│   │   │   └── {hash}_DSC0002.CR2
├── processed/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── {hash}_processed_v1.jpg
│   │   │   └── {hash}_processed_v2.jpg
├── recipes/
│   ├── 2024/
│   │   ├── 01/
│   │   │   ├── {recipe_id}.json
├── cache/
│   ├── thumbnails/
│   ├── previews/
│   └── ai_results/
```

### Database Schema
```sql
-- Processing history
CREATE TABLE processing_history (
    id UUID PRIMARY KEY,
    original_file_hash VARCHAR(64) UNIQUE,
    original_path TEXT,
    processed_at TIMESTAMP,
    recipe_id UUID,
    status VARCHAR(50),
    error_message TEXT
);

-- Recipes
CREATE TABLE recipes (
    id UUID PRIMARY KEY,
    original_hash VARCHAR(64),
    created_at TIMESTAMP,
    operations JSONB,
    ai_metadata JSONB,
    user_overrides JSONB
);

-- Upload tracking
CREATE TABLE uploads (
    id UUID PRIMARY KEY,
    original_hash VARCHAR(64),
    immich_original_id VARCHAR(255),
    immich_processed_id VARCHAR(255),
    uploaded_at TIMESTAMP
);
```

## Error Handling and Recovery

### 1. Processing Failures
- Automatic retry with exponential backoff
- Fallback to basic processing if AI fails
- Store partial results for debugging

### 2. Upload Failures
- Queue failed uploads for retry
- Maintain local cache until confirmed
- Support offline mode with sync

### 3. Data Integrity
- SHA256 checksums for all files
- Regular integrity checks
- Automated backup verification

## Performance Considerations

### 1. Concurrent Processing
- Worker pool with configurable size
- Priority queue for urgent files
- Resource-aware scheduling

### 2. Caching Strategy
- AI results cached for 24 hours
- Thumbnail generation on-demand
- Preview cache with LRU eviction

### 3. Memory Management
- Stream large files instead of loading
- Dispose of processed images after upload
- Monitor memory usage and adjust workers