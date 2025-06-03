#!/bin/bash
# Script to fix permissions on photo processing directories

echo "Fixing permissions on photo processing directories..."

# Fix ownership and permissions on the directories
sudo chown -R john:john /mnt/storage1/photo-test-inbox
sudo chown -R john:john /mnt/storage1/photo-test-processed

# Set directory permissions to 755 (rwxr-xr-x)
sudo chmod 755 /mnt/storage1/photo-test-inbox
sudo chmod 755 /mnt/storage1/photo-test-processed
sudo chmod 755 /mnt/storage1/photo-test-processed/originals

# Set file permissions to 644 (rw-r--r--)
find /mnt/storage1/photo-test-processed -type f -exec sudo chmod 644 {} \;
find /mnt/storage1/photo-test-inbox -type f -exec sudo chmod 644 {} \;

echo "Permissions fixed!"
echo ""
echo "Current permissions:"
ls -la /mnt/storage1/photo-test-inbox
echo ""
ls -la /mnt/storage1/photo-test-processed