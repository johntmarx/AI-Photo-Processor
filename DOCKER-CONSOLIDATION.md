# Docker Compose Consolidation Guide

## Overview
This guide explains the consolidation of multiple docker-compose files into a single, unified configuration.

## ⚠️ CRITICAL DATA LOCATIONS

**These directories contain your precious data and MUST be preserved:**

| Data Type | Location | Description |
|-----------|----------|-------------|
| **Photos** | `/mnt/storage1/immich` | Your entire Immich photo library |
| **Database** | `./postgres` | All metadata, users, albums, settings |
| **ML Models** | `model-cache` volume | Machine learning models |
| **Ollama Models** | `ollama-data` volume | AI language models |
| **Import Folder** | `./import` | Photos to be imported |
| **Photo Processor** | `./photo-processor/data` | Processing data and cache |

## What Was Consolidated

### Previous Structure:
- `/home/john/immich/docker-compose.yml` - Main Immich services
- `/home/john/immich/photo-processor/docker-compose.yml` - Photo processor v1
- `/home/john/immich/photo-processor/docker-compose.v2.yml` - Photo processor v2
- `/home/john/immich/photo-processor/docker-compose.frontend.yml` - Frontend services

### New Structure:
- `/home/john/immich/docker-compose.yml` - ALL services in one file

## Services Included

### Core Immich Services:
- `immich-server` - Main application
- `immich-machine-learning` - ML processing
- `database` - PostgreSQL
- `redis` - Cache/messaging

### Web Services:
- `nginx` - Reverse proxy
- `cloudflared` - Cloudflare tunnel
- `immich-registration` - User registration

### AI Services:
- `ollama` - Qwen2.5-VL and other models
- `photo-processor` - AI-powered photo processing
- `photo-processor-api` - REST API for photo processing

### File Services:
- `samba` - Network file sharing

## Network Configuration

All services now use a single network: `immich_default`

This ensures:
- All services can communicate
- No network isolation issues
- Simplified troubleshooting

## Migration Steps

### 1. Pre-Migration Checklist
- [ ] Verify all photo data at `/mnt/storage1/immich`
- [ ] Verify database at `./postgres`
- [ ] Back up `.env` file
- [ ] Note any custom configurations

### 2. Run Migration Script
```bash
cd /home/john/immich
./migrate-to-consolidated.sh
```

The script will:
1. Verify all data paths exist
2. Backup current docker-compose.yml
3. Stop all containers
4. Install consolidated configuration
5. Start all services

### 3. Post-Migration Verification
```bash
# Check all services are running
docker ps

# Check logs for any errors
docker-compose logs -f

# Verify Immich web interface
# http://your-server

# Verify photo processor
docker logs immich_photo_processor

# Check Ollama/Qwen2.5-VL
docker exec immich_ollama ollama list
```

## Rollback Plan

If anything goes wrong:

```bash
# Stop all services
docker-compose down

# Restore backup (replace timestamp)
cp docker-compose.yml.backup.20240101_120000 docker-compose.yml

# Start services
docker-compose up -d
```

## Environment Variables

All settings remain in `.env` file:

| Variable | Purpose | Example |
|----------|---------|---------|
| `UPLOAD_LOCATION` | Photo storage | `/mnt/storage1/immich` |
| `DB_DATA_LOCATION` | Database | `./postgres` |
| `EXTERNAL_PATH` | Import folder | `./import` |
| `STORAGE1_PATH` | Additional storage | `/mnt/storage1/immich` |
| `STORAGE2_PATH` | Additional storage | `/mnt/storage2/immich` |

## Benefits of Consolidation

1. **Single Point of Control**: One docker-compose.yml for all services
2. **Unified Networking**: All services on same network
3. **Easier Updates**: Update all services with one command
4. **Consistent Configuration**: All services use same environment
5. **Simplified Backup**: One configuration to backup

## Troubleshooting

### Services won't start
```bash
# Check for port conflicts
sudo netstat -tulpn | grep -E ':(80|443|3001|11434|8100)'

# Check Docker logs
docker-compose logs [service-name]
```

### Network issues
```bash
# Verify network exists
docker network ls | grep immich

# Recreate if needed
docker-compose down
docker network rm immich_default
docker-compose up -d
```

### Permission issues
```bash
# Fix ownership (adjust UID:GID as needed)
sudo chown -R 1000:1000 /mnt/storage1/immich
sudo chown -R 1000:1000 ./photo-processor
```

## Maintenance Commands

```bash
# Update all services
docker-compose pull
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Restart a service
docker-compose restart [service-name]

# Stop everything
docker-compose down

# Stop and remove volumes (DANGER!)
# docker-compose down -v  # DON'T DO THIS unless you want to lose data!
```

## Important Notes

1. **NEVER** run `docker-compose down -v` as it will delete named volumes
2. **ALWAYS** backup before major changes
3. **VERIFY** data paths in .env match your actual data locations
4. The consolidated setup preserves ALL existing data locations
5. All volume mounts are identical to the original configuration

## Support

If you encounter issues:
1. Check service logs: `docker-compose logs [service]`
2. Verify all paths in `.env`
3. Ensure adequate disk space
4. Check file permissions