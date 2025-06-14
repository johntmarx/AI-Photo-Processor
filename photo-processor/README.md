# AI Photo Processor for Immich

An intelligent photo processing service that automatically analyzes, enhances, and uploads photos to your Immich instance. Designed specifically for high-volume photography workflows, particularly sports and event photography.

## 🚀 Quick Start

```bash
# Clone and start the application
git clone <repository-url>
cd immich/photo-processor
docker-compose up -d

# Open the web interface
open http://localhost
```

## 🎯 Project Status: Phase 2 Complete ✅

- **Phase 0**: Original preservation ✅ (41/41 tests passing)
- **Phase 1**: Backend API infrastructure ✅ (21/21 tests passing) 
- **Phase 2**: Frontend development ✅ (74/74 tests passing)
- **Phase 3**: AI model upgrades 🚧 (Ready to start)

📊 **See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed progress**

## 📚 Documentation

### Project Management
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - Current project status and metrics
- [implementation-roadmap.md](docs/implementation-roadmap.md) - Complete development roadmap
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Guide for migrating from v1 to v2

### Architecture & Design
- [architecture-design.md](docs/architecture-design.md) - System architecture overview
- [frontend-implementation.md](docs/frontend-implementation.md) - Frontend architecture details
- [component-api-reference.md](docs/component-api-reference.md) - API endpoint reference
- [ai-models-implementation.md](docs/ai-models-implementation.md) - AI model integration plans

### Development & Testing
- [TESTING.md](TESTING.md) - Comprehensive testing guide
- [lessons-learned-from-testing.md](docs/lessons-learned-from-testing.md) - Testing insights
- [FINAL_TEST_RESULTS.md](FINAL_TEST_RESULTS.md) - Complete test results

## Features

### Core Processing
- **Automatic File Monitoring**: Watches designated folders for new photos (RAW and standard formats)
- **AI-Powered Analysis**: Uses Ollama with Gemma3 model to understand photo content, composition, and quality
- **Intelligent Processing**: 
  - Smart cropping based on subject detection
  - Automatic rotation correction
  - Color enhancement and white balance
  - Blur detection and quality assessment
- **RAW File Support**: Handles ARW, CR2, CR3, NEF, DNG, ORF, RW2, and more
- **Immich Integration**: Seamlessly uploads processed photos with AI-generated metadata
- **Duplicate Prevention**: SHA256-based hash tracking prevents reprocessing
- **Original Preservation**: Dual upload system preserves original files

### Web Interface (New in Phase 2)
- **Real-time Dashboard**: Live statistics and processing status
- **Photo Management**: Upload, view, compare original vs processed
- **Recipe System**: Create, edit, and apply processing recipes
- **Processing Queue**: Manual control with pause/resume/approve
- **WebSocket Updates**: Real-time notifications and progress
- **Responsive Design**: Works on desktop and mobile devices

### API (New in Phase 1)
- **RESTful API**: 40+ endpoints for complete control
- **WebSocket Support**: Real-time event streaming
- **No Authentication**: Designed for secure local networks
- **Comprehensive Documentation**: Auto-generated API docs

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Frontend  │────▶│    FastAPI       │────▶│  Photo Processor│
│  (React + TS)   │     │  Backend API     │     │   (Core Logic)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         │                       ▼                         ▼
         │              ┌──────────────────┐     ┌─────────────────┐
         └─────────────▶│    WebSocket     │     │     Immich      │
                        │  (Real-time)     │     │   (API upload)  │
                        └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │    Ollama    │
                                                   │ (AI Analysis)│
                                                   └──────────────┘
```

## Components

### Core Services
- **`main_v2.py`**: Enhanced service orchestrator with recipe support
- **`ai_analyzer.py`**: AI integration for image analysis using Ollama
- **`image_processor_v2.py`**: Advanced image processing with RAW conversion
- **`immich_client_v2.py`**: Enhanced Immich client with dual upload support
- **`recipe_storage.py`**: Recipe management system
- **`hash_tracker.py`**: Duplicate detection using file hashing

### API Backend
- **`api/main.py`**: FastAPI application with WebSocket support
- **`api/routes/`**: RESTful endpoints (photos, processing, recipes, stats)
- **`api/services/`**: Business logic and WebSocket manager
- **`api/models/`**: Pydantic models for validation

### Frontend Application
- **`frontend-app/`**: React TypeScript application
- **Components**: Dashboard, PhotoGrid, RecipeEditor, ProcessingQueue
- **Real-time**: WebSocket provider for live updates
- **State Management**: React Query for server state

## Requirements

- Python 3.12+
- Ollama with Gemma3 model
- Immich instance (self-hosted)
- Docker (for containerized deployment)
- 8GB+ RAM recommended for RAW processing

## Installation

### Docker Deployment (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd immich/photo-processor
```

2. Start the full stack application:
```bash
docker-compose up -d

# Access the application:
# Frontend: http://localhost:80
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

3. Individual components can be run separately:
```bash
# Just the photo processor
docker-compose up -d processor

# Just the API backend
docker-compose up -d api

# Just the frontend
docker-compose up -d frontend
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export IMMICH_API_URL="http://localhost:2283"
export IMMICH_API_KEY="your-api-key"
export INPUT_FOLDER="/path/to/watch/folder"
export OUTPUT_FOLDER="/path/to/processed/folder"
```

3. Run the service:
```bash
python main.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://ollama:11434` |
| `OLLAMA_MODEL` | AI model to use | `gemma2:9b` |
| `IMMICH_API_URL` | Immich API endpoint | `http://immich-server:2283` |
| `IMMICH_API_KEY` | Immich API key | Required |
| `INPUT_FOLDER` | Folder to watch for new photos | `/import` |
| `OUTPUT_FOLDER` | Folder for processed photos | `/processed` |
| `LOGLEVEL` | Logging level | `INFO` |
| `MAX_FILE_SIZE_MB` | Max file size to process | `500` |
| `PROCESS_EXISTING` | Process existing files on startup | `true` |

### Processing Configuration

The processor uses intelligent defaults but can be fine-tuned:

- **Image Quality**: Output JPEG quality is set to 95%
- **Max Dimensions**: 4000x4000 pixels (maintains aspect ratio)
- **Color Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Blur Threshold**: Configurable in image processor

## Usage

### Basic Workflow

1. **Drop photos** into the configured input folder
2. **AI analyzes** each photo for content and composition
3. **Processing** applies intelligent enhancements
4. **Upload** to Immich with metadata and tags
5. **Archive** original files to output folder

### Supported File Types

**RAW Formats**: ARW, CR2, CR3, NEF, DNG, ORF, RW2, RAF, SRW, X3F, IIQ, 3FR, DCR, K25, KDC, MDC, PEF, RAW, RWL, SR2, SRF, MRW, MEF

**Standard Formats**: JPG, JPEG, PNG, HEIC, HEIF, WEBP, TIFF, BMP

### AI Analysis Features

The AI analyzer detects:
- **Subjects**: People, faces, objects
- **Composition**: Rule of thirds, symmetry, leading lines
- **Scene Type**: Sports, landscape, portrait, etc.
- **Quality Issues**: Blur, exposure problems
- **Suggested Crops**: Intelligent framing recommendations

## Testing

The project includes comprehensive test suites for all components:

### Backend Tests
```bash
# Core processor tests (41 tests)
pytest

# API tests (21 tests)
cd api && pytest

# With coverage
pytest --cov=. --cov-report=html
```

### Frontend Tests
```bash
# Frontend tests (74 tests)
cd frontend-app
npm test

# With coverage
npm run test:coverage

# In watch mode
npm run test:watch
```

### Docker-based Testing
```bash
# Run all tests in isolated containers
./run_tests_docker.sh
```

**Total Test Coverage**: 136/136 tests passing (100%)

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Monitoring

The service includes:
- **Health Checks**: Service availability monitoring
- **Processing Metrics**: Files processed, success/failure rates
- **Detailed Logging**: Configurable log levels
- **Error Tracking**: Failed file quarantine

## Troubleshooting

### Common Issues

1. **"Ollama not responding"**
   - Ensure Ollama is running: `curl http://localhost:11434/api/tags`
   - Check if Gemma3 model is installed: `ollama list`

2. **"Failed to upload to Immich"**
   - Verify API key is correct
   - Check Immich server is accessible
   - Ensure user has upload permissions

3. **"RAW file processing failed"**
   - Check available memory (8GB+ recommended)
   - Verify RAW format is supported
   - Check file isn't corrupted

4. **"Duplicate file detected"**
   - This is normal - prevents reprocessing
   - To reprocess, delete entry from `hashes.json`

### Debug Mode

Enable detailed logging:
```bash
export LOGLEVEL=DEBUG
python main.py
```

## Performance Tuning

- **Batch Processing**: Enable for multiple files
- **Memory Management**: Adjust `MAX_FILE_SIZE_MB` for your system
- **CPU Optimization**: Uses multicore for image processing
- **GPU Acceleration**: Ollama can use GPU if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

- Immich team for the excellent photo management platform
- Ollama for accessible AI model serving
- Contributors to rawpy, Pillow, and other dependencies