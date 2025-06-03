# AI Photo Processor for Immich

An intelligent photo processing service that automatically analyzes, enhances, and uploads photos to your Immich instance. Designed specifically for high-volume photography workflows, particularly sports and event photography.

## Features

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
- **High Performance**: Processes high-resolution RAW files efficiently

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   File System   │────▶│  Photo Processor │────▶│     Immich      │
│  (watch folder) │     │                  │     │   (API upload)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    Ollama    │
                        │ (AI Analysis)│
                        └──────────────┘
```

## Components

### Core Services

- **`main.py`**: Service orchestrator with file watching and processing pipeline
- **`ai_analyzer.py`**: AI integration for image analysis using Ollama
- **`image_processor_v2.py`**: Advanced image processing with RAW conversion and enhancement
- **`immich_client.py`**: Immich API client for uploads and metadata management
- **`hash_tracker.py`**: Duplicate detection using file hashing
- **`schemas.py`**: Pydantic models for structured data validation

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

2. Build the Docker image:
```bash
docker build -t photo-processor .
```

3. Run with docker-compose (see parent directory's docker-compose.yml):
```bash
cd ..
docker-compose up -d photo-processor
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

Run the comprehensive test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test categories
pytest tests/unit/
pytest tests/integration/
```

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