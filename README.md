# Immich Photo Management System with AI Processing

A comprehensive photo management solution combining Immich's powerful media server with AI-powered photo processing, secure user registration, and automated workflows.

## System Overview

This setup includes:
- **Immich**: Self-hosted photo and video management solution
- **AI Photo Processor**: Intelligent photo analysis and enhancement pipeline
- **Registration Portal**: Secure self-registration system with storage quotas
- **SMB File Sharing**: Network-accessible photo processing folders
- **Cloudflare Tunnel**: Secure external access

## Quick Start

### 1. Initial Setup

```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your configuration:
# - CLOUDFLARE_TUNNEL_TOKEN
# - REGISTRATION_SECRET
# - IMMICH_API_KEY (generate after first login)

# Initialize directories
./fix-permissions.sh

# Start all services
docker-compose up -d
```

### 2. Access Points

- **Immich Web Interface**: https://your-domain.com
- **Registration Portal**: https://your-domain.com/register
- **Local Access**: http://your-server-ip:2283
- **SMB Shares**: \\your-server-ip\photo-inbox (configure credentials)

## Key Features

### AI Photo Processing Pipeline

The photo processor automatically:
- **Monitors** folders for new photos (supports 25+ RAW formats)
- **Analyzes** images using AI vision model (Gemma3:4b) for:
  - Subject detection and framing
  - Image quality assessment
  - Professional composition recommendations
  - Color analysis and correction needs
- **Enhances** photos with:
  - Smart AI-guided cropping
  - Automatic rotation correction
  - 16-bit color enhancement
  - Adaptive histogram equalization
- **Uploads** to Immich with metadata and album organization

Processing workflow:
1. Drop photos in SMB inbox or `/import` folder
2. AI analyzes and enhances within 30-60 seconds
3. Processed photos appear in Immich "AI Processed Photos" album
4. Access results via SMB processed folder

### User Registration System

Secure self-registration with:
- Secret key validation (server-side only)
- Storage quota selection (50GB - 1TB)
- Rate limiting (5 attempts per IP/15 min)
- Automatic Immich user creation

### Auto-Import Feature

For bulk imports without AI processing:
1. Place files in `./import` directory
2. Configure in Immich: Administration → External Libraries
3. Add library pointing to `/mnt/media/import`

## Architecture

### Docker Services

- **immich-server**: Main web interface (port 2283)
- **immich-microservices**: Background jobs and processing
- **immich-machine-learning**: Face recognition, object detection
- **immich_photo_processor**: AI photo enhancement pipeline
- **immich_ollama**: AI vision model service
- **immich_samba**: SMB file sharing
- **immich_registration**: User registration portal
- **postgres**: Database (PostgreSQL 16)
- **redis**: Cache and job queue
- **cloudflared**: Cloudflare tunnel

### Directory Structure

```
./library/          # Main Immich storage
./import/           # Manual import directory
./photo-processor/  # AI processing service
  ├── input/       # Photos to process
  ├── output/      # Processed results
  └── working_files/
./immich-registration/  # Registration portal
./postgres/         # Database storage
```

## Configuration

### Environment Variables

Key settings in `.env`:
```bash
# Cloudflare
CLOUDFLARE_TUNNEL_TOKEN=your_token_here

# Registration
REGISTRATION_SECRET=your_secret_key
REGISTRATION_PORT=3000

# Immich
IMMICH_API_KEY=your_api_key
IMMICH_URL=http://immich-server:3001

# AI Processing
OLLAMA_BASE_URL=http://immich_ollama:11434
MAX_FILE_SIZE_MB=500
PROCESS_INTERVAL=10
```

### Processing Settings

- **Supported Formats**: NEF, CR2, ARW, RAF, ORF, DNG, RW2 + 18 more RAW formats
- **Output**: JPEG 100% quality, max 4000x4000 (maintains aspect ratio)
- **AI Model**: Gemma3:4b with NVIDIA GPU acceleration
- **Duplicate Prevention**: SHA256 hash tracking

## Administration

### Generate Immich API Key

1. Log in to Immich as admin
2. Go to User Settings → API Keys
3. Create new key and add to `.env`

### Monitor Services

```bash
# View all service logs
docker-compose logs -f

# Check specific service
docker logs immich_photo_processor -f

# Service health
docker-compose ps
```

### Troubleshooting

- **Photo Processing Issues**: Check `docker logs immich_photo_processor`
- **Registration Errors**: Verify REGISTRATION_SECRET and API key
- **SMB Access**: Ensure ports 139/445 are accessible
- **GPU Issues**: Verify NVIDIA drivers and Docker GPU support

## Security Notes

- Registration portal requires secret key (never exposed to client)
- All services run behind Cloudflare tunnel
- Immich API key required for automated operations
- SMB shares use authenticated access
- Input validation and rate limiting on all endpoints

## Performance

- Photo processing: ~30-60 seconds per image
- Supports images up to 500MB (configurable)
- GPU acceleration for AI inference
- Concurrent processing capabilities
- Efficient duplicate detection

## Backup Recommendations

Critical data to backup:
- `./postgres/` - Database
- `./library/` - Photo storage
- `.env` - Configuration
- Custom album metadata

## Support

For detailed setup instructions:
- AI Processor: See `AI-PHOTO-PROCESSOR-SETUP.md`
- Registration: See `REGISTRATION-SETUP.md`
- Testing: See `photo-processor/TESTING.md`

## License

This project combines open-source components:
- Immich: AGPL-3.0
- Custom services: See individual component licenses