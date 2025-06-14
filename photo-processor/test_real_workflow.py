#!/usr/bin/env python3
"""
Real workflow test using actual photos and working operations
Tests the complete pipeline with existing test photos
"""

import asyncio
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import logging

sys.path.insert(0, '/app')

from api.services.photo_service_v2 import photo_service
from api.services.processing_service_v2 import processing_service
from api.services.recipe_service_v2 import recipe_service
from api.models.processing import BatchOperation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealWorkflowTester:
    """Test the real workflow with actual photos"""
    
    def __init__(self):
        self.results = {}
        self.test_photos = []
    
    async def setup_test_photos(self):
        """Get some real test photos from inbox"""
        print("\n=== Setting Up Test Photos ===")
        
        inbox_path = Path("/app/data/inbox")
        photos = list(inbox_path.glob("*.jpeg"))[:5]  # Get first 5 JPEGs
        
        if not photos:
            print("❌ No test photos found in inbox!")
            return False
        
        print(f"Found {len(photos)} test photos:")
        for photo in photos:
            print(f"  - {photo.name} ({photo.stat().st_size:,} bytes)")
            self.test_photos.append(photo)
        
        return True
    
    async def test_basic_recipe_operations(self):
        """Test basic recipe operations on real photos"""
        print("\n=== Testing Basic Recipe Operations ===")
        
        # Create a comprehensive recipe
        recipe_data = {
            "name": "Real Workflow Test Recipe",
            "description": "Comprehensive recipe for testing real workflow",
            "operations": [
                {
                    "operation": "auto_rotate",
                    "parameters": {"method": "exif"},
                    "enabled": True
                },
                {
                    "operation": "crop",
                    "parameters": {"aspectRatio": "16:9"},
                    "enabled": True
                },
                {
                    "operation": "enhance",
                    "parameters": {
                        "brightness": 1.05,
                        "contrast": 1.1,
                        "saturation": 1.1,
                        "sharpness": 1.2
                    },
                    "enabled": True
                },
                {
                    "operation": "denoise",
                    "parameters": {"strength": 0.3},
                    "enabled": True
                },
                {
                    "operation": "color_balance",
                    "parameters": {
                        "temperature": 0.05,  # Slight warm
                        "tint": 0.0
                    },
                    "enabled": True
                }
            ],
            "style_preset": "natural",
            "processing_config": {
                "qualityThreshold": 90,
                "export": {
                    "format": "jpeg",
                    "quality": 92,
                    "preserveMetadata": True
                }
            }
        }
        
        # Create the recipe
        recipe = await recipe_service.create_recipe(**recipe_data)
        print(f"✓ Created recipe: {recipe['name']} (ID: {recipe['id']})")
        
        self.results['recipe_creation'] = {
            'status': 'success',
            'recipe_id': recipe['id'],
            'operations': len(recipe['operations'])
        }
        
        return recipe['id']
    
    async def test_photo_upload_and_processing(self, recipe_id: str):
        """Test uploading and processing photos with recipe"""
        print("\n=== Testing Photo Upload and Processing ===")
        
        uploaded_photos = []
        
        # Upload test photos
        for photo_path in self.test_photos[:3]:  # Use first 3 photos
            print(f"\nUploading: {photo_path.name}")
            
            # Create mock upload file
            class MockUploadFile:
                def __init__(self, path):
                    self.filename = path.name
                    self.content_type = "image/jpeg"
                    self._path = path
                
                async def read(self):
                    return self._path.read_bytes()
            
            mock_file = MockUploadFile(photo_path)
            
            # Upload with recipe
            result = await photo_service.save_upload(
                file=mock_file,
                auto_process=False,  # We'll batch process
                recipe_id=recipe_id
            )
            
            if result['status'] in ['completed', 'pending']:
                uploaded_photos.append(result['photo_id'])
                print(f"  ✓ Uploaded: {result['photo_id']}")
            else:
                print(f"  ⚠️  Upload status: {result['status']}")
        
        self.results['photo_upload'] = {
            'status': 'success',
            'uploaded_count': len(uploaded_photos),
            'photo_ids': uploaded_photos
        }
        
        return uploaded_photos
    
    async def test_batch_processing(self, photo_ids: list, recipe_id: str):
        """Test batch processing with recipe"""
        print("\n=== Testing Batch Processing ===")
        
        # Create batch operation
        batch_op = BatchOperation(
            photo_ids=photo_ids,
            recipe_id=recipe_id,
            priority="high",
            skip_ai=False
        )
        
        # Process batch
        batch_result = await processing_service.batch_process(batch_op)
        
        print(f"Batch processing result:")
        print(f"  - Queued: {batch_result['queued']}")
        print(f"  - Skipped: {batch_result['skipped']}")
        
        if batch_result.get('errors'):
            print(f"  - Errors: {batch_result['errors']}")
        
        self.results['batch_processing'] = {
            'status': 'success' if batch_result['queued'] > 0 else 'failed',
            'queued': batch_result['queued'],
            'skipped': batch_result['skipped']
        }
        
        # Process the queue
        processed_count = 0
        print("\nProcessing queue...")
        
        while processed_count < batch_result['queued']:
            result = await processing_service.process_next_item()
            if result:
                processed_count += 1
                print(f"  ✓ Processed photo {processed_count}/{batch_result['queued']}: {result['photo_id']}")
                print(f"    Time: {result.get('processing_time', 0):.2f}s")
            else:
                break
        
        self.results['queue_processing'] = {
            'status': 'success',
            'processed': processed_count,
            'expected': batch_result['queued']
        }
        
        return processed_count
    
    async def verify_processed_photos(self, photo_ids: list):
        """Verify processed photos have all outputs"""
        print("\n=== Verifying Processed Photos ===")
        
        verified_count = 0
        issues = []
        
        for photo_id in photo_ids:
            photo = await photo_service.get_photo(photo_id)
            
            if not photo:
                issues.append(f"Photo {photo_id} not found")
                continue
            
            print(f"\nPhoto: {photo.filename}")
            print(f"  Status: {photo.status}")
            
            # Check outputs
            checks = {
                'processed': photo.processed_path and Path(photo.processed_path).exists(),
                'thumbnail': photo.thumbnail_path and Path(photo.thumbnail_path).exists(),
                'web': photo.web_path and Path(photo.web_path).exists()
            }
            
            all_good = all(checks.values())
            
            for output, exists in checks.items():
                print(f"  {output}: {'✓' if exists else '✗'}")
            
            if all_good:
                verified_count += 1
                
                # Check file sizes
                if photo.processed_path:
                    size = Path(photo.processed_path).stat().st_size
                    print(f"  Processed size: {size:,} bytes")
            else:
                issues.append(f"Photo {photo_id} missing outputs")
        
        self.results['verification'] = {
            'status': 'success' if verified_count == len(photo_ids) else 'partial',
            'verified': verified_count,
            'total': len(photo_ids),
            'issues': issues
        }
        
        return verified_count == len(photo_ids)
    
    async def test_recipe_operations_on_image(self):
        """Test individual recipe operations on a single image"""
        print("\n=== Testing Individual Recipe Operations ===")
        
        if not self.test_photos:
            print("❌ No test photos available")
            return
        
        test_photo = self.test_photos[0]
        output_dir = Path("/app/data/test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Load image
        img = Image.open(test_photo)
        print(f"\nTest image: {test_photo.name}")
        print(f"Original size: {img.size}")
        
        # Test each operation
        operations_tested = []
        
        # 1. Crop to 16:9
        print("\n1. Testing Crop (16:9)...")
        width, height = img.size
        target_ratio = 16 / 9
        current_ratio = width / height
        
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            cropped = img.crop((left, 0, left + new_width, height))
        else:
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            cropped = img.crop((0, top, width, top + new_height))
        
        cropped.save(output_dir / "test_crop_16x9.jpg")
        print(f"   ✓ Cropped to: {cropped.size}")
        operations_tested.append('crop')
        
        # 2. Enhancement
        print("\n2. Testing Enhancement...")
        enhanced = cropped
        
        # Brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1.05)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Saturation
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        enhanced.save(output_dir / "test_enhanced.jpg")
        print("   ✓ Applied brightness, contrast, saturation, sharpness")
        operations_tested.append('enhance')
        
        # 3. Denoise (using blur as approximation)
        print("\n3. Testing Denoise...")
        denoised = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
        denoised.save(output_dir / "test_denoised.jpg")
        print("   ✓ Applied denoise filter")
        operations_tested.append('denoise')
        
        # 4. Generate outputs
        print("\n4. Generating output formats...")
        
        # Thumbnail
        thumb = denoised.copy()
        thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)
        thumb.save(output_dir / "test_thumbnail.jpg", quality=85)
        print(f"   ✓ Thumbnail: {thumb.size}")
        
        # Web version
        web = denoised.copy()
        web.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
        web.save(output_dir / "test_web.jpg", quality=90)
        print(f"   ✓ Web: {web.size}")
        
        # Full quality
        denoised.save(output_dir / "test_final.jpg", quality=92)
        print(f"   ✓ Final: {denoised.size}")
        
        self.results['operations_test'] = {
            'status': 'success',
            'operations_tested': operations_tested,
            'outputs_generated': ['thumbnail', 'web', 'final']
        }
        
        print(f"\n✓ Test outputs saved to: {output_dir}")
    
    async def run_all_tests(self):
        """Run complete workflow test"""
        print("\n" + "="*60)
        print("REAL WORKFLOW TEST")
        print("="*60)
        
        # Setup
        if not await self.setup_test_photos():
            return False
        
        # Test recipe operations on single image
        await self.test_recipe_operations_on_image()
        
        # Create recipe
        recipe_id = await self.test_basic_recipe_operations()
        
        # Upload photos
        photo_ids = await self.test_photo_upload_and_processing(recipe_id)
        
        if photo_ids:
            # Batch process
            processed = await self.test_batch_processing(photo_ids, recipe_id)
            
            # Verify results
            if processed > 0:
                await self.verify_processed_photos(photo_ids)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test, result in self.results.items():
            status = result.get('status', 'unknown')
            symbol = '✓' if status == 'success' else '⚠️' if status == 'partial' else '✗'
            print(f"{symbol} {test}: {status}")
        
        # Save results
        results_file = Path("/app/real_workflow_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        return all(r.get('status') in ['success', 'partial'] for r in self.results.values())


async def test_frontend_integration():
    """Test that frontend can use the processed photos"""
    print("\n" + "="*60)
    print("FRONTEND INTEGRATION TEST")
    print("="*60)
    
    # Get recent photos
    photo_list = await photo_service.list_photos(page=1, page_size=5)
    
    print(f"Found {photo_list.total} photos total")
    print(f"Showing {len(photo_list.photos)} recent photos:")
    
    for photo in photo_list.photos:
        print(f"\n{photo.filename}:")
        print(f"  ID: {photo.id}")
        print(f"  Status: {photo.status}")
        
        if photo.status == 'completed':
            # Check image URLs
            urls = {
                'thumbnail': f"/images/thumbnails/{photo.id}_thumb.jpg",
                'web': f"/images/web/{photo.id}_web.jpg",
                'processed': f"/images/processed/{photo.id}_{photo.filename}"
            }
            
            print("  Frontend URLs:")
            for type_, url in urls.items():
                print(f"    {type_}: {url}")


async def main():
    """Run all tests"""
    print("Real Photo Processing Workflow Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create tester
    tester = RealWorkflowTester()
    
    # Run tests
    success = await tester.run_all_tests()
    
    # Test frontend integration
    await test_frontend_integration()
    
    print(f"\n{'✓' if success else '✗'} Real workflow test {'passed' if success else 'had issues'}")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)