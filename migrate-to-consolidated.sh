#!/bin/bash

# Migration script to safely consolidate docker-compose files
# This script helps transition from multiple docker-compose files to a single consolidated one

set -e  # Exit on error

echo "==========================================="
echo "Docker Compose Consolidation Migration Tool"
echo "==========================================="
echo ""
echo "⚠️  CRITICAL: This script will help you migrate to a consolidated docker-compose.yml"
echo "⚠️  Your data will NOT be touched - we're only changing configuration"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as the correct user
echo "1. Checking environment..."
if [ ! -f ".env" ]; then
    print_error ".env file not found! Please run this from the immich directory."
    exit 1
fi
print_status "Found .env file"

# Source the .env file to get paths
export $(cat .env | grep -v '^#' | xargs)

# Verify critical paths exist
echo ""
echo "2. Verifying critical data paths..."

# Check photo storage
if [ -d "${UPLOAD_LOCATION}" ]; then
    PHOTO_COUNT=$(find "${UPLOAD_LOCATION}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.heic" -o -iname "*.raw" \) | wc -l)
    print_status "Photo storage found at: ${UPLOAD_LOCATION} (contains ~${PHOTO_COUNT} images)"
else
    print_error "Photo storage NOT FOUND at: ${UPLOAD_LOCATION}"
    echo "    This is CRITICAL - your photos should be here!"
    exit 1
fi

# Check database
if [ -d "${DB_DATA_LOCATION}" ]; then
    print_status "Database found at: ${DB_DATA_LOCATION}"
else
    print_warning "Database directory not found at: ${DB_DATA_LOCATION}"
    echo "    This might be OK if using a named volume"
fi

# Check import directory
if [ -d "${EXTERNAL_PATH}" ]; then
    print_status "Import directory found at: ${EXTERNAL_PATH}"
else
    print_warning "Import directory not found at: ${EXTERNAL_PATH}"
fi

# Backup current docker-compose.yml
echo ""
echo "3. Backing up current configuration..."
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
if [ -f "docker-compose.yml" ]; then
    cp docker-compose.yml "docker-compose.yml.backup.${BACKUP_DATE}"
    print_status "Backed up current docker-compose.yml to docker-compose.yml.backup.${BACKUP_DATE}"
fi

# Show current running containers
echo ""
echo "4. Current running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep immich || true

# Prompt for confirmation
echo ""
echo "==========================================="
echo "MIGRATION PLAN:"
echo "1. Stop all current containers"
echo "2. Replace docker-compose.yml with consolidated version"
echo "3. Start services with new configuration"
echo ""
echo "Your data at these locations will NOT be touched:"
echo "  - Photos: ${UPLOAD_LOCATION}"
echo "  - Database: ${DB_DATA_LOCATION}"
echo "  - Import: ${EXTERNAL_PATH}"
echo "==========================================="
echo ""
read -p "Do you want to proceed with migration? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    print_warning "Migration cancelled by user"
    exit 0
fi

# Stop current services
echo ""
echo "5. Stopping current services..."
print_warning "Stopping all Immich containers..."
docker-compose down || true

# Also stop any services from other compose files
if [ -f "photo-processor/docker-compose.yml" ]; then
    (cd photo-processor && docker-compose down) || true
fi

# Wait for containers to stop
sleep 5

# Install new docker-compose file
echo ""
echo "6. Installing consolidated docker-compose.yml..."
if [ -f "docker-compose.consolidated.yml" ]; then
    cp docker-compose.consolidated.yml docker-compose.yml
    print_status "Installed new consolidated docker-compose.yml"
else
    print_error "docker-compose.consolidated.yml not found!"
    exit 1
fi

# Show what will be started
echo ""
echo "7. Services that will be started:"
echo "  - immich-server (core application)"
echo "  - immich-machine-learning (ML processing)"
echo "  - database (PostgreSQL)"
echo "  - redis (cache)"
echo "  - nginx (web proxy)"
echo "  - cloudflared (tunnel)"
echo "  - ollama (AI models)"
echo "  - photo-processor (photo processing)"
echo "  - photo-processor-api (API service)"
echo "  - immich-registration (user registration)"
echo "  - samba (file sharing)"

# Start services
echo ""
read -p "Ready to start services? (yes/no): " -r
echo

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "8. Starting services..."
    docker-compose up -d
    
    # Wait and check status
    echo ""
    echo "Waiting for services to start..."
    sleep 10
    
    echo ""
    echo "9. Checking service status..."
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep immich || true
    
    echo ""
    print_status "Migration completed!"
    echo ""
    echo "Please verify:"
    echo "1. Check that all services are running: docker ps"
    echo "2. Access Immich web interface"
    echo "3. Verify your photos are still accessible"
    echo "4. Check that photo processor is working"
    echo ""
    echo "If anything goes wrong, restore the backup:"
    echo "  cp docker-compose.yml.backup.${BACKUP_DATE} docker-compose.yml"
    echo "  docker-compose up -d"
else
    print_warning "Skipping service start. You can start manually with: docker-compose up -d"
fi

echo ""
echo "==========================================="
echo "Migration script completed"
echo "==========================================="