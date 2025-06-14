# Photo Processor Implementation Roadmap

## Current State Analysis

### Existing Strengths
- Functional AI analysis with Ollama/Gemma3
- RAW file processing capability
- Immich integration working
- Docker containerization
- Basic testing infrastructure

### Critical Issues to Address
1. **Original files are being lost** - only processed JPEGs uploaded
2. No user interface for monitoring/control
3. Aggressive automatic processing without user input
4. No way to reprocess or undo changes
5. Limited AI model (Gemma3 4B)
6. No recipe/versioning system

## Implementation Phases

### Phase 0: Critical Fix - Original Preservation (Week 1) ✅ COMPLETED
**Goal**: Stop losing original files immediately

#### Tasks:
1. **Modify main.py upload flow**
   - [x] Change from move to copy for original files
   - [x] Implement dual upload (original + processed)
   - [x] Add original file tracking in database
   - [x] Create "Originals" album in Immich

2. **Update ImmichClient**
   - [x] Add method for uploading with relationships
   - [x] Implement metadata linking between versions
   - [x] Add original preservation flag

3. **Testing**
   - [x] Verify originals are preserved
   - [x] Test dual upload functionality
   - [x] Ensure backward compatibility
   - [x] All 41 unit tests passing

**Deliverables**: 
- ✅ Updated photo processor that preserves originals (main_v2.py)
- ✅ Recipe storage system (recipe_storage.py)
- ✅ Enhanced Immich client (immich_client_v2.py)
- ✅ Comprehensive test suite

### Phase 1: Frontend Infrastructure (Weeks 2-3) ✅ COMPLETED
**Goal**: Build API backend and test infrastructure for frontend development

#### Tasks:
1. **Recipe System Implementation** ✅
   - [x] Design recipe data model (recipe_storage.py)
   - [x] Create RecipeStorage class with JSON persistence
   - [x] Implement recipe serialization/deserialization
   - [x] Add recipe CRUD operations

2. **API Backend Development** ✅
   - [x] Design RESTful API structure (40+ endpoints)
   - [x] Implement FastAPI backend with CORS support
   - [x] Add WebSocket support for real-time updates
   - [x] Create comprehensive API documentation
   - [x] Implement all route handlers (photos, processing, recipes, stats)

3. **Service Layer Architecture** ✅
   - [x] Create service layer separation
   - [x] Implement WebSocket manager for broadcasting
   - [x] Design photo, processing, recipe, and stats services
   - [x] Add proper error handling and validation

4. **Test Infrastructure** ✅
   - [x] Create comprehensive test suite (90+ test cases)
   - [x] Set up Docker-based testing environment
   - [x] Implement async test support
   - [x] Add integration and unit test separation
   - [x] Configure coverage reporting

**Deliverables**: ✅ COMPLETED
- ✅ Recipe-based processing system (14/14 tests passing)
- ✅ RESTful API with WebSocket support (21/21 core tests passing)
- ✅ Complete test infrastructure with Docker support
- ✅ Production-ready backend architecture

### Phase 2: Frontend Development (Weeks 4-5) ✅ COMPLETED
**Goal**: Create user interface for control and monitoring
**Note**: No authentication required - designed for local network access

#### Tasks:
1. **React App Setup** ✅
   - [x] Initialize React project with TypeScript
   - [x] Set up development environment with Vite
   - [x] Configure build pipeline and Docker
   - [x] Implement routing structure (React Router)

2. **Core Components** ✅
   - [x] Dashboard with real-time statistics
   - [x] Processing queue viewer with live updates
   - [x] Photo viewer with before/after comparison
   - [x] Recipe editor with preset management
   - [x] Settings panel for processing configuration

3. **Real-time Features** ✅
   - [x] WebSocket integration for live updates
   - [x] Live processing progress indicators
   - [x] Toast notification system
   - [x] Real-time queue status monitoring

4. **User Controls** ✅
   - [x] Manual processing approval/rejection
   - [x] Recipe application to photos
   - [x] Quality threshold adjustment
   - [x] Batch operations interface
   - [x] Processing pause/resume controls

**Deliverables**: ✅ COMPLETED
- ✅ Fully functional React frontend with TypeScript (74/74 tests passing)
- ✅ Real-time monitoring dashboard with WebSocket updates
- ✅ Manual control interface with approval/rejection flow
- ✅ Local network deployment ready with nginx
- ✅ Complete component library (10+ components)
- ✅ Responsive design with TailwindCSS
- ✅ Comprehensive testing with Vitest + React Testing Library

### Phase 3: AI Model Upgrades (Weeks 6-7) 🚧 IN PROGRESS
**Goal**: Integrate state-of-the-art AI models with configurable parameters

#### Tasks:
1. **Model Integration**
   - [ ] RT-DETR for object detection
   - [ ] SAM2 for segmentation
   - [ ] CLIP for embeddings
   - [ ] NIMA for quality scoring
   - [ ] Qwen2.5-VL for analysis

2. **Recipe Enhancement for AI Configuration**
   - [ ] Add AI model selection per recipe step
   - [ ] Implement prompt customization for each model
   - [ ] Configure model hyperparameters (temperature, top_p, etc.)
   - [ ] Add model-specific parameters (detection thresholds, etc.)
   - [ ] Create prompt template system with variables
   - [ ] Support conditional logic based on AI outputs

3. **Data Translation Layer**
   - [ ] Implement model data converters
   - [ ] Create unified pipeline orchestrator
   - [ ] Add model fallback system
   - [ ] Optimize for GPU usage

4. **Enhanced Features**
   - [ ] Semantic search capability
   - [ ] Advanced composition analysis
   - [ ] Style transfer options
   - [ ] Content-aware cropping

5. **Performance Optimization**
   - [ ] Implement model caching
   - [ ] Add batch processing
   - [ ] Optimize memory usage
   - [ ] Create model loader service

**Deliverables**:
- Modern AI pipeline with 5+ models
- Configurable AI parameters in recipes (prompts, models, hyperparameters)
- Semantic search functionality
- Improved processing quality
- Recipe template system with AI customization

### Phase 4: Advanced Features (Week 8)
**Goal**: Add power user features

#### Tasks:
1. **Advanced UI Features**
   - [ ] Split-screen comparison view
   - [ ] Timeline/history browser
   - [ ] Keyboard shortcuts
   - [ ] Drag-and-drop upload

2. **Automation**
   - [ ] Rule-based processing
   - [ ] Scheduled operations
   - [ ] Watch folder improvements
   - [ ] Auto-categorization

3. **Integration Enhancements**
   - [ ] External storage support (S3)
   - [ ] Plugin system design
   - [ ] Export capabilities
   - [ ] Backup automation

4. **Quality Assurance**
   - [ ] Comprehensive testing
   - [ ] Performance profiling
   - [ ] Security audit
   - [ ] Documentation completion

**Deliverables**:
- Feature-complete photo processor
- Comprehensive documentation
- Production-ready deployment

## Development Guidelines

### Code Structure
```
photo-processor/
├── backend/
│   ├── api/           # FastAPI routes
│   ├── core/          # Business logic
│   ├── models/        # Data models
│   ├── services/      # External services
│   └── workers/       # Background tasks
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── stores/
├── ai-models/
│   ├── detection/
│   ├── segmentation/
│   ├── quality/
│   └── embeddings/
├── shared/
│   ├── types/         # Shared TypeScript types
│   └── schemas/       # Shared data schemas
└── deployment/
    ├── docker/
    ├── k8s/
    └── scripts/
```

### Testing Strategy

1. **Unit Tests**
   - Model converters
   - Recipe operations
   - API endpoints
   - React components

2. **Integration Tests**
   - Full pipeline processing
   - API + Frontend interaction
   - Database operations
   - AI model integration

3. **E2E Tests**
   - User workflows
   - File upload to processed
   - Recipe creation and replay
   - Batch operations

### Deployment Plan

#### Development Environment
```yaml
# docker-compose.dev.yml
services:
  backend:
    build: ./backend
    volumes:
      - ./backend:/app
      - ./data:/data
    environment:
      - DEV_MODE=true
    ports:
      - "8000:8000"
      
  frontend:
    build: ./frontend
    volumes:
      - ./frontend:/app
      - /app/node_modules
    ports:
      - "3000:3000"
      
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=photo_processor
      - POSTGRES_USER=dev
      - POSTGRES_PASSWORD=dev
    ports:
      - "5432:5432"
```

#### Production Deployment
```yaml
# docker-compose.prod.yml
services:
  backend:
    image: photo-processor-backend:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              
  frontend:
    image: photo-processor-frontend:latest
    deploy:
      replicas: 2
      
  nginx:
    image: nginx:alpine
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    ports:
      - "80:80"
      - "443:443"
```

## Migration Strategy

### For Existing Users

1. **Backup Current Data**
   ```bash
   # Backup script
   #!/bin/bash
   tar -czf backup_$(date +%Y%m%d).tar.gz \
     ./processed \
     ./originals \
     ./photo-processor.db
   ```

2. **Migration Steps**
   - Stop current processor
   - Backup all data
   - Deploy new version
   - Run migration scripts
   - Verify functionality
   - Resume processing

3. **Rollback Plan**
   - Keep old version available
   - Database migration rollback scripts
   - Data restoration procedures
   - Communication plan

## Success Metrics

### Technical Metrics
- **Original Preservation Rate**: 100%
- **Processing Speed**: <5 seconds average
- **AI Accuracy**: >90% acceptable decisions
- **System Uptime**: 99.9%
- **Memory Usage**: <4GB per worker

### User Experience Metrics
- **Time to First Use**: <5 minutes
- **Manual Interventions**: <10% of photos
- **User Satisfaction**: >4.5/5 rating
- **Feature Adoption**: >80% using recipes

### Business Metrics
- **Storage Efficiency**: <2x original size
- **Processing Cost**: <$0.001 per photo
- **Support Tickets**: <5% of users
- **Active Users**: >90% monthly

## Risk Management

### Technical Risks
1. **GPU Memory Issues**
   - Mitigation: Dynamic model loading
   - Fallback: CPU processing

2. **Storage Growth**
   - Mitigation: Compression strategies
   - Fallback: External storage

3. **API Rate Limits**
   - Mitigation: Request queuing
   - Fallback: Retry logic

### Operational Risks
1. **Data Loss**
   - Mitigation: Multiple backups
   - Fallback: Recovery procedures

2. **Performance Degradation**
   - Mitigation: Monitoring alerts
   - Fallback: Horizontal scaling

## Timeline Summary

| Week | Phase | Key Deliverables | Status |
|------|-------|------------------|--------|
| 1 | Phase 0 | Original preservation fix | ✅ COMPLETED |
| 2-3 | Phase 1 | Backend API, WebSocket, Tests | ✅ COMPLETED |
| 4-5 | Phase 2 | Frontend React UI | ✅ COMPLETED |
| 6-7 | Phase 3 | AI upgrades with configurable recipes | 🚧 NEXT |
| 8 | Phase 4 | Polish, deployment | 📋 PLANNED |

## Current Status: Phase 2 Complete ✅

### ✅ Completed (100% functional):
**Phase 0 - Original Preservation**:
- **Original File Preservation**: 41/41 tests passing
- **Recipe Storage System**: JSON-based with full CRUD

**Phase 1 - Backend Infrastructure**:
- **FastAPI Backend**: 40+ endpoints with validation
- **WebSocket Real-time Updates**: Fully functional
- **Comprehensive Test Suite**: 90+ test cases
- **Docker Infrastructure**: Development and testing ready

**Phase 2 - Frontend Development**:
- **React TypeScript Frontend**: 74/74 tests passing
- **Real-time Dashboard**: WebSocket integration complete
- **Recipe Management UI**: Full CRUD with presets
- **Photo Management**: Upload, view, compare, process
- **Processing Controls**: Pause/resume, queue management
- **Responsive Design**: Mobile-friendly with TailwindCSS

### 🚧 Next Phase: AI Model Upgrades
- Integrate modern AI models (RT-DETR, SAM2, CLIP, etc.)
- Add configurable AI parameters to recipes
- Implement prompt customization per recipe step
- Enable model selection and hyperparameter tuning

## Next Steps

1. **Immediate Actions** (Phase 3 - AI Upgrades)
   - Research and evaluate AI model options
   - Design recipe schema for AI configuration
   - Set up GPU-optimized Docker containers
   - Implement model loader service
   - Create prompt template system

2. **Development Environment Ready**
   - Full stack running: `docker-compose up`
   - Frontend: `http://localhost:80`
   - Backend API: `http://localhost:8000`
   - WebSocket endpoint: `ws://localhost:8000/ws`
   - API docs: `http://localhost:8000/docs`

3. **Architecture Notes**
   - Frontend and backend fully integrated
   - Real-time updates working end-to-end
   - Recipe system ready for AI parameter extensions
   - Infrastructure prepared for GPU workloads

4. **Recipe AI Configuration Design**
   - Each recipe step can specify:
     - Model selection (e.g., "model": "qwen2.5-vl")
     - Custom prompts with variables
     - Hyperparameters (temperature, max_tokens, etc.)
     - Model-specific settings (detection confidence, etc.)
   - Support for conditional branching based on AI outputs
   - Prompt templates with photo metadata injection