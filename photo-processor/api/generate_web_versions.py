#!/usr/bin/env python3
"""
Generate web-optimized versions for existing photos
"""
import asyncio
import json
from pathlib import Path
import logging
from PIL import Image
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def generate_web_version(image_path: Path, web_path: Path, max_size: int = 1920) -> bool:
    """Generate a web-optimized version of an image"""
    try:
        logger.info(f"Generating web version for {image_path} -> {web_path}")
        
        # Open the image
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            
            # Calculate new size if image is larger than max_size
            width, height = img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Save web version with good quality
            img.save(web_path, 'JPEG', quality=90, optimize=True)
            
        logger.info(f"Web version generated successfully: {web_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate web version: {e}")
        return False

async def main():
    """Generate web versions for all existing photos"""
    photos_db = Path("../data/photos.json")
    processed_dir = Path("../data/processed")
    web_dir = Path("../data/web")
    
    # Create web directory if it doesn't exist
    web_dir.mkdir(parents=True, exist_ok=True)
    
    if not photos_db.exists():
        logger.error("Photos database not found")
        return
    
    # Load photos database
    with open(photos_db) as f:
        data = json.load(f)
    
    # Get photos dict
    photos = data.get('photos', {})
    
    # Filter completed photos without web versions
    photos_to_process = []
    for photo_id, photo_data in photos.items():
        if photo_data.get('status') in ['completed', 'processed'] and not photo_data.get('web_path'):
            photos_to_process.append((photo_id, photo_data))
    
    logger.info(f"Found {len(photos_to_process)} photos without web versions")
    
    # Process photos
    updated_count = 0
    for photo_id, photo_data in photos_to_process:
        processed_path = photo_data.get('processed_path')
        if not processed_path:
            logger.warning(f"Photo {photo_id} has no processed path")
            continue
        
        processed_file = Path(processed_path)
        if not processed_file.exists():
            logger.warning(f"Processed file not found: {processed_file}")
            continue
        
        # Generate web version
        web_path = web_dir / f"{photo_id}_web.jpg"
        if await generate_web_version(processed_file, web_path):
            # Update database
            photo_data['web_path'] = str(web_path)
            updated_count += 1
            logger.info(f"Updated photo {photo_id} with web path")
    
    # Save updated database
    if updated_count > 0:
        with open(photos_db, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Updated {updated_count} photos with web versions")
    else:
        logger.info("No photos were updated")

if __name__ == "__main__":
    asyncio.run(main())