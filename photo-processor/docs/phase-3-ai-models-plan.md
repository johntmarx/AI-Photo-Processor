# Phase 3: AI Model Upgrades Plan

## Status: IN PROGRESS 🚧
**Prerequisites Complete**: Frontend and API infrastructure ready for AI enhancements
**Current Progress**: All 5 AI models implemented as standalone components with 100% test coverage

## Overview
Integrate state-of-the-art AI models to enhance photo analysis and processing capabilities. Key focus on making AI configurable through the recipe system.

## Key Requirements
- **Multiple AI Models**: RT-DETR, SAM2, CLIP, NIMA, Qwen2.5-VL
- **Configurable Recipes**: AI parameters adjustable per recipe step
- **Custom Prompts**: User-definable prompts with variables
- **Hyperparameter Control**: Temperature, top_p, detection thresholds
- **GPU Optimization**: Efficient model loading and caching
- **Conditional Logic**: Recipe branching based on AI outputs

## Architecture Enhancement

### Recipe Schema Extension
```json
{
  "recipe": {
    "name": "Sports Photography AI Enhanced",
    "steps": [
      {
        "type": "ai_analysis",
        "model": "qwen2.5-vl",
        "prompt": "Analyze this sports photo. Identify: 1) Main subjects 2) Action type 3) Key moments 4) Suggested crops for ${aspect_ratio}",
        "parameters": {
          "temperature": 0.7,
          "max_tokens": 500,
          "top_p": 0.9
        },
        "output_variables": ["subjects", "action_type", "crops"]
      },
      {
        "type": "object_detection",
        "model": "rt-detr",
        "parameters": {
          "confidence_threshold": 0.6,
          "classes": ["person", "ball", "equipment"]
        },
        "conditional": {
          "if": "subjects.count > 3",
          "then": "crop_to_main_action"
        }
      },
      {
        "type": "segmentation",
        "model": "sam2",
        "prompt_from": "object_detection.results",
        "parameters": {
          "quality": "high",
          "use_gpu": true
        }
      },
      {
        "type": "quality_assessment",
        "model": "nima",
        "parameters": {
          "aspects": ["technical", "aesthetic"]
        },
        "threshold": 7.5
      }
    ]
  }
}
```

### Model Integration Architecture
```
┌─────────────────────┐
│   Recipe Engine     │
│  (Enhanced with AI) │
└──────────┬──────────┘
           │
     ┌─────▼─────┐
     │   Model   │
     │  Manager  │
     └─────┬─────┘
           │
    ┌──────┴──────────────────────────┐
    │                                  │
┌───▼────┐  ┌────────┐  ┌────────┐  ┌─▼──────┐
│RT-DETR │  │  SAM2  │  │  CLIP  │  │  NIMA  │
└────────┘  └────────┘  └────────┘  └────────┘
                                        │
                                   ┌────▼─────┐
                                   │Qwen2.5-VL│
                                   └──────────┘
```

## Implementation Phases

### Phase 3.1: Model Infrastructure (Week 1)
1. **Model Manager Service**
   - Dynamic model loading/unloading
   - GPU memory management
   - Model caching strategy
   - Fallback mechanisms

2. **Recipe Schema Updates**
   - Extend recipe format for AI configuration
   - Add validation for AI parameters
   - Implement variable system
   - Support conditional logic

3. **API Extensions**
   - New endpoints for model management
   - Recipe validation with AI steps
   - Model capability queries

### Phase 3.2: Model Integration (Week 2)
1. **RT-DETR Integration**
   - Object detection pipeline
   - Bounding box extraction
   - Class filtering
   - Confidence thresholds

2. **SAM2 Integration**
   - Segmentation masks
   - Interactive prompting
   - Multi-object support
   - Quality levels

3. **CLIP Integration**
   - Image embeddings
   - Text-image similarity
   - Semantic search prep
   - Style matching

4. **NIMA Integration**
   - Technical quality scoring
   - Aesthetic assessment
   - Multi-aspect evaluation

5. **Qwen2.5-VL Integration**
   - Advanced scene understanding
   - Detailed descriptions
   - Custom prompt handling
   - Multi-modal analysis

### Phase 3.3: Advanced Features
1. **Prompt Templates**
   - Variable substitution
   - Metadata injection
   - Context awareness
   - Multi-language support

2. **Conditional Processing**
   - If/then logic in recipes
   - Multiple condition support
   - Variable comparisons
   - Action branching

3. **Model Orchestration**
   - Pipeline optimization
   - Parallel processing
   - Result aggregation
   - Error handling

## Technical Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **CPU**: Multi-core for parallel processing
- **RAM**: 16GB+ for model loading
- **Storage**: 50GB+ for model weights

### Software Dependencies
```python
# Core ML frameworks
torch>=2.0.0
transformers>=4.35.0
timm>=0.9.0

# Model-specific
ultralytics  # RT-DETR
segment-anything-2  # SAM2
clip-torch  # CLIP
pytorch-nima  # NIMA

# Infrastructure
accelerate  # Distributed computing
safetensors  # Efficient model loading
tritonclient  # Model serving
```

### Docker Updates
```dockerfile
# AI-enhanced processor
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install ML dependencies
COPY requirements-ai.txt .
RUN pip install --no-cache-dir -r requirements-ai.txt

# Model cache directory
ENV MODEL_CACHE_DIR=/models
VOLUME /models
```

## API Enhancements

### New Endpoints
```python
# Model management
GET  /api/models              # List available models
GET  /api/models/{model_id}   # Model details and capabilities
POST /api/models/load         # Pre-load a model
POST /api/models/unload       # Unload from memory

# AI-enhanced recipes
POST /api/recipes/validate    # Validate recipe with AI steps
POST /api/recipes/preview     # Preview AI analysis results
GET  /api/recipes/templates   # Get prompt templates

# AI analysis
POST /api/analyze/scene       # Detailed scene analysis
POST /api/analyze/quality     # Quality assessment
POST /api/analyze/similarity  # Find similar images
```

### WebSocket Events
```javascript
// New AI-related events
{
  "type": "model_loaded",
  "data": { "model": "rt-detr", "memory_usage": "2.3GB" }
}

{
  "type": "ai_analysis_progress",
  "data": { 
    "photo_id": "123",
    "step": "object_detection",
    "progress": 0.45,
    "found_objects": 3
  }
}

{
  "type": "quality_score",
  "data": {
    "photo_id": "123",
    "technical": 8.2,
    "aesthetic": 7.9
  }
}
```

## Testing Strategy

### Unit Tests
- Model wrapper classes
- Prompt template engine
- Conditional logic processor
- Parameter validation

### Integration Tests
- Full AI pipeline processing
- Recipe execution with AI steps
- Model switching/fallbacks
- GPU/CPU mode switching

### Performance Tests
- Model loading times
- Inference benchmarks
- Memory usage monitoring
- Concurrent request handling

### AI-Specific Tests
- Prompt injection prevention
- Output validation
- Error handling
- Timeout management

## Success Metrics

### Technical
- Model inference time < 2s per image
- GPU memory usage < 8GB with all models
- 99% uptime for AI services
- Successful fallback rate > 95%

### Quality
- Object detection accuracy > 90%
- Segmentation quality score > 0.85
- User satisfaction with AI suggestions > 4.5/5
- Processing time reduction > 30%

### Usage
- AI features used in > 60% of recipes
- Custom prompts created by > 40% of users
- Conditional logic used in > 20% of recipes

## Migration Path

### For Existing Users
1. **Backward Compatibility**
   - Existing recipes continue to work
   - AI features are opt-in
   - Gradual migration tools

2. **Model Download**
   ```bash
   # Script to download required models
   python download_models.py --models all
   # Or selective download
   python download_models.py --models rt-detr,sam2
   ```

3. **Recipe Migration**
   - Automated suggestions for AI enhancement
   - Template library for common use cases
   - Conversion wizard in UI

## Risk Mitigation

### Technical Risks
1. **GPU Memory Exhaustion**
   - Mitigation: Dynamic model unloading
   - Fallback: CPU inference mode

2. **Model Download Failures**
   - Mitigation: Mirror repositories
   - Fallback: Essential models only

3. **Inference Timeouts**
   - Mitigation: Configurable timeouts
   - Fallback: Skip AI steps

### Operational Risks
1. **Increased Complexity**
   - Mitigation: Comprehensive documentation
   - Fallback: Simple mode without AI

2. **Resource Costs**
   - Mitigation: Efficient batching
   - Fallback: Cloud inference option

## Next Steps

1. **Week 1 Goals**
   - Set up GPU development environment
   - Implement model manager service
   - Extend recipe schema
   - Create model loading infrastructure

2. **Week 2 Goals**
   - Integrate all 5 AI models
   - Implement prompt template system
   - Add conditional logic processing
   - Complete API enhancements

3. **Testing & Polish**
   - Comprehensive testing suite
   - Performance optimization
   - Documentation updates
   - UI enhancements for AI features

---

**Note**: AI parameters in recipes allow users to configure prompts, select models, and tune hyperparameters for each processing step, providing unprecedented control over AI-enhanced photo processing.