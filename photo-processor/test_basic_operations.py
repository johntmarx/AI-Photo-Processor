#!/usr/bin/env python3
"""
Basic operation tests for photo processing
Tests the fundamental operations without AI dependencies
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import cv2
import asyncio

sys.path.insert(0, '/app')


class BasicOperationTester:
    """Test basic image operations that don't require AI models"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results = {}
        print(f"Using temp directory: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, name: str = "test.jpg", size=(800, 600)) -> Path:
        """Create a simple test image"""
        img = Image.new('RGB', size, color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes for testing
        draw.rectangle([100, 100, 300, 300], fill='red', outline='darkred')
        draw.ellipse([400, 200, 600, 400], fill='green', outline='darkgreen')
        draw.text((350, 50), "TEST IMAGE", fill='black')
        
        path = self.temp_dir / name
        img.save(path, 'JPEG', quality=95)
        return path
    
    def test_crop_operation(self):
        """Test cropping with different aspect ratios"""
        print("\n=== Testing Crop Operation ===")
        
        try:
            test_img = self.create_test_image("crop_test.jpg", size=(1000, 800))
            img = Image.open(test_img)
            
            crop_configs = [
                ("original", None),
                ("square", (1, 1)),
                ("16:9", (16, 9)),
                ("4:3", (4, 3)),
                ("portrait", (3, 4)),
            ]
            
            results = []
            for name, ratio in crop_configs:
                if ratio is None:
                    # Keep original
                    cropped = img
                else:
                    # Calculate crop dimensions
                    width, height = img.size
                    target_ratio = ratio[0] / ratio[1]
                    current_ratio = width / height
                    
                    if current_ratio > target_ratio:
                        # Too wide, crop width
                        new_width = int(height * target_ratio)
                        new_height = height
                        left = (width - new_width) // 2
                        top = 0
                    else:
                        # Too tall, crop height
                        new_width = width
                        new_height = int(width / target_ratio)
                        left = 0
                        top = (height - new_height) // 2
                    
                    cropped = img.crop((left, top, left + new_width, top + new_height))
                
                output_path = self.temp_dir / f"crop_{name}.jpg"
                cropped.save(output_path)
                
                results.append({
                    'aspect': name,
                    'size': cropped.size,
                    'path': output_path.name
                })
                
                print(f"  ✓ {name}: {cropped.size[0]}x{cropped.size[1]}")
            
            self.results['crop'] = {
                'status': 'success',
                'tested_aspects': len(crop_configs),
                'results': results
            }
            
        except Exception as e:
            print(f"  ✗ Crop test failed: {e}")
            self.results['crop'] = {'status': 'failed', 'error': str(e)}
    
    def test_rotation_operation(self):
        """Test basic rotation without AI detection"""
        print("\n=== Testing Rotation Operation ===")
        
        try:
            test_img = self.create_test_image("rotation_test.jpg")
            img = Image.open(test_img)
            
            angles = [0, 90, 180, 270, -45, 15]
            results = []
            
            for angle in angles:
                rotated = img.rotate(-angle, expand=True, fillcolor='white')
                
                output_path = self.temp_dir / f"rotate_{angle}.jpg"
                rotated.save(output_path)
                
                results.append({
                    'angle': angle,
                    'size': rotated.size,
                    'path': output_path.name
                })
                
                print(f"  ✓ Rotated {angle}°: {rotated.size[0]}x{rotated.size[1]}")
            
            self.results['rotation'] = {
                'status': 'success',
                'tested_angles': len(angles),
                'results': results
            }
            
        except Exception as e:
            print(f"  ✗ Rotation test failed: {e}")
            self.results['rotation'] = {'status': 'failed', 'error': str(e)}
    
    def test_enhance_operation(self):
        """Test image enhancement operations"""
        print("\n=== Testing Enhancement Operations ===")
        
        try:
            test_img = self.create_test_image("enhance_test.jpg")
            img = Image.open(test_img)
            
            enhancements = [
                ('brightness', ImageEnhance.Brightness, [0.5, 1.0, 1.5]),
                ('contrast', ImageEnhance.Contrast, [0.5, 1.0, 1.5]),
                ('saturation', ImageEnhance.Color, [0.0, 1.0, 1.5]),
                ('sharpness', ImageEnhance.Sharpness, [0.5, 1.0, 2.0]),
            ]
            
            results = []
            for name, enhancer_class, values in enhancements:
                enhancer = enhancer_class(img)
                
                for value in values:
                    enhanced = enhancer.enhance(value)
                    
                    output_path = self.temp_dir / f"enhance_{name}_{value}.jpg"
                    enhanced.save(output_path)
                    
                    results.append({
                        'enhancement': name,
                        'value': value,
                        'path': output_path.name
                    })
                    
                print(f"  ✓ {name}: tested {len(values)} values")
            
            self.results['enhance'] = {
                'status': 'success',
                'enhancements_tested': len(enhancements),
                'total_variations': len(results)
            }
            
        except Exception as e:
            print(f"  ✗ Enhancement test failed: {e}")
            self.results['enhance'] = {'status': 'failed', 'error': str(e)}
    
    def test_filter_operations(self):
        """Test image filter operations"""
        print("\n=== Testing Filter Operations ===")
        
        try:
            test_img = self.create_test_image("filter_test.jpg")
            img = Image.open(test_img)
            
            filters = [
                ('blur', ImageFilter.BLUR),
                ('sharpen', ImageFilter.SHARPEN),
                ('edge_enhance', ImageFilter.EDGE_ENHANCE),
                ('smooth', ImageFilter.SMOOTH),
                ('detail', ImageFilter.DETAIL),
                ('gaussian_blur', ImageFilter.GaussianBlur(radius=2)),
            ]
            
            results = []
            for name, filter_obj in filters:
                filtered = img.filter(filter_obj)
                
                output_path = self.temp_dir / f"filter_{name}.jpg"
                filtered.save(output_path)
                
                results.append({
                    'filter': name,
                    'path': output_path.name
                })
                
                print(f"  ✓ Applied {name} filter")
            
            self.results['filters'] = {
                'status': 'success',
                'filters_tested': len(filters)
            }
            
        except Exception as e:
            print(f"  ✗ Filter test failed: {e}")
            self.results['filters'] = {'status': 'failed', 'error': str(e)}
    
    def test_resize_operation(self):
        """Test image resizing for different outputs"""
        print("\n=== Testing Resize Operations ===")
        
        try:
            test_img = self.create_test_image("resize_test.jpg", size=(2000, 1500))
            img = Image.open(test_img)
            
            resize_configs = [
                ('thumbnail', (400, 400)),
                ('web', (1920, 1080)),
                ('social', (1200, 1200)),
                ('mobile', (1080, 1920)),
            ]
            
            results = []
            for name, max_size in resize_configs:
                # Calculate size maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                output_path = self.temp_dir / f"resize_{name}.jpg"
                img.save(output_path, optimize=True, quality=90)
                
                results.append({
                    'name': name,
                    'target': max_size,
                    'actual': img.size,
                    'path': output_path.name
                })
                
                print(f"  ✓ {name}: {img.size[0]}x{img.size[1]} (target: {max_size})")
                
                # Reload original for next resize
                img = Image.open(test_img)
            
            self.results['resize'] = {
                'status': 'success',
                'configs_tested': len(resize_configs)
            }
            
        except Exception as e:
            print(f"  ✗ Resize test failed: {e}")
            self.results['resize'] = {'status': 'failed', 'error': str(e)}
    
    def test_color_adjustments(self):
        """Test color balance and adjustments"""
        print("\n=== Testing Color Adjustments ===")
        
        try:
            test_img = self.create_test_image("color_test.jpg")
            img = Image.open(test_img)
            
            # Convert to numpy for color operations
            img_array = np.array(img)
            
            adjustments = []
            
            # Warm filter (increase red/yellow)
            warm = img_array.copy()
            warm[:, :, 0] = np.clip(warm[:, :, 0] * 1.1, 0, 255)  # Red
            warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.05, 0, 255)  # Green
            warm_path = self.temp_dir / "color_warm.jpg"
            Image.fromarray(warm.astype(np.uint8)).save(warm_path)
            adjustments.append(('warm', warm_path.name))
            
            # Cool filter (increase blue)
            cool = img_array.copy()
            cool[:, :, 2] = np.clip(cool[:, :, 2] * 1.15, 0, 255)  # Blue
            cool_path = self.temp_dir / "color_cool.jpg"
            Image.fromarray(cool.astype(np.uint8)).save(cool_path)
            adjustments.append(('cool', cool_path.name))
            
            # Vibrance (selective saturation)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Increase saturation
            vibrant = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            vibrant_path = self.temp_dir / "color_vibrant.jpg"
            Image.fromarray(vibrant).save(vibrant_path)
            adjustments.append(('vibrant', vibrant_path.name))
            
            # Black and white
            bw = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            bw_rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
            bw_path = self.temp_dir / "color_bw.jpg"
            Image.fromarray(bw_rgb).save(bw_path)
            adjustments.append(('black_white', bw_path.name))
            
            for name, path in adjustments:
                print(f"  ✓ Applied {name} adjustment")
            
            self.results['color_adjustments'] = {
                'status': 'success',
                'adjustments_tested': len(adjustments)
            }
            
        except Exception as e:
            print(f"  ✗ Color adjustment test failed: {e}")
            self.results['color_adjustments'] = {'status': 'failed', 'error': str(e)}
    
    def test_metadata_operations(self):
        """Test metadata preservation and modification"""
        print("\n=== Testing Metadata Operations ===")
        
        try:
            from PIL.ExifTags import TAGS
            
            test_img = self.create_test_image("metadata_test.jpg")
            img = Image.open(test_img)
            
            # Add some metadata
            exif = img.getexif()
            
            # Common EXIF tags
            exif[271] = "Test Camera"  # Make
            exif[272] = "Test Model"   # Model
            exif[306] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")  # DateTime
            
            # Save with metadata
            output_path = self.temp_dir / "metadata_preserved.jpg"
            img.save(output_path, exif=exif)
            
            # Read back and verify
            img_check = Image.open(output_path)
            exif_check = img_check.getexif()
            
            metadata_found = []
            for tag, value in exif_check.items():
                tag_name = TAGS.get(tag, tag)
                metadata_found.append((tag_name, value))
                
            print(f"  ✓ Metadata operations tested")
            print(f"  ✓ Found {len(metadata_found)} metadata tags")
            
            self.results['metadata'] = {
                'status': 'success',
                'tags_written': 3,
                'tags_found': len(metadata_found)
            }
            
        except Exception as e:
            print(f"  ✗ Metadata test failed: {e}")
            self.results['metadata'] = {'status': 'failed', 'error': str(e)}
    
    async def test_all_operations(self):
        """Run all basic operation tests"""
        print("\n" + "="*60)
        print("TESTING BASIC PHOTO OPERATIONS")
        print("="*60)
        
        # Run all tests
        self.test_crop_operation()
        self.test_rotation_operation()
        self.test_enhance_operation()
        self.test_filter_operations()
        self.test_resize_operation()
        self.test_color_adjustments()
        self.test_metadata_operations()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        success_count = 0
        failed_count = 0
        
        for operation, result in self.results.items():
            status = result.get('status', 'unknown')
            symbol = '✓' if status == 'success' else '✗'
            print(f"{symbol} {operation}: {status}")
            
            if status == 'success':
                success_count += 1
            else:
                failed_count += 1
        
        print(f"\nTotal: {success_count} passed, {failed_count} failed")
        print("="*60)
        
        # Save detailed results
        results_file = self.temp_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        return self.results


async def test_recipe_application():
    """Test applying a recipe to an actual image"""
    print("\n" + "="*60)
    print("TESTING RECIPE APPLICATION")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test image
        test_img = temp_dir / "recipe_test.jpg"
        img = Image.new('RGB', (1600, 1200), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.rectangle([400, 300, 1200, 900], fill='orange', outline='darkorange')
        draw.text((700, 550), "RECIPE TEST", fill='black')
        img.save(test_img)
        
        # Define a recipe with basic operations
        recipe = {
            'id': 'basic-test',
            'name': 'Basic Operations Recipe',
            'operations': [
                {
                    'operation': 'crop',
                    'parameters': {'aspectRatio': '16:9'},
                    'enabled': True
                },
                {
                    'operation': 'enhance',
                    'parameters': {
                        'brightness': 1.1,
                        'contrast': 1.2,
                        'saturation': 1.15
                    },
                    'enabled': True
                },
                {
                    'operation': 'sharpen',
                    'parameters': {'strength': 1.5},
                    'enabled': True
                },
                {
                    'operation': 'resize',
                    'parameters': {'maxWidth': 1920, 'maxHeight': 1080},
                    'enabled': True
                }
            ]
        }
        
        print(f"Applying recipe: {recipe['name']}")
        print(f"Operations: {len(recipe['operations'])}")
        
        # Apply each operation
        current_img = Image.open(test_img)
        
        for i, op in enumerate(recipe['operations']):
            if not op['enabled']:
                continue
                
            print(f"\nApplying operation {i+1}: {op['operation']}")
            
            if op['operation'] == 'crop':
                # Apply 16:9 crop
                width, height = current_img.size
                target_ratio = 16 / 9
                current_ratio = width / height
                
                if current_ratio > target_ratio:
                    new_width = int(height * target_ratio)
                    left = (width - new_width) // 2
                    current_img = current_img.crop((left, 0, left + new_width, height))
                else:
                    new_height = int(width / target_ratio)
                    top = (height - new_height) // 2
                    current_img = current_img.crop((0, top, width, top + new_height))
                
                print(f"  ✓ Cropped to: {current_img.size}")
                
            elif op['operation'] == 'enhance':
                params = op['parameters']
                
                if 'brightness' in params:
                    enhancer = ImageEnhance.Brightness(current_img)
                    current_img = enhancer.enhance(params['brightness'])
                    
                if 'contrast' in params:
                    enhancer = ImageEnhance.Contrast(current_img)
                    current_img = enhancer.enhance(params['contrast'])
                    
                if 'saturation' in params:
                    enhancer = ImageEnhance.Color(current_img)
                    current_img = enhancer.enhance(params['saturation'])
                
                print(f"  ✓ Enhanced with: {params}")
                
            elif op['operation'] == 'sharpen':
                enhancer = ImageEnhance.Sharpness(current_img)
                current_img = enhancer.enhance(op['parameters']['strength'])
                print(f"  ✓ Sharpened with strength: {op['parameters']['strength']}")
                
            elif op['operation'] == 'resize':
                max_size = (op['parameters']['maxWidth'], op['parameters']['maxHeight'])
                current_img.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"  ✓ Resized to: {current_img.size}")
        
        # Save final result
        output_path = temp_dir / "recipe_output.jpg"
        current_img.save(output_path, quality=90, optimize=True)
        
        print(f"\n✓ Recipe applied successfully")
        print(f"Final image: {output_path.name} ({current_img.size})")
        
        # Calculate file sizes
        original_size = test_img.stat().st_size
        final_size = output_path.stat().st_size
        
        print(f"Original size: {original_size:,} bytes")
        print(f"Final size: {final_size:,} bytes")
        print(f"Reduction: {(1 - final_size/original_size)*100:.1f}%")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


async def main():
    """Run all tests"""
    print("Photo Processing - Basic Operations Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create tester
    tester = BasicOperationTester()
    
    try:
        # Run basic operation tests
        results = await tester.test_all_operations()
        
        # Test recipe application
        await test_recipe_application()
        
        # Check if all tests passed
        all_passed = all(
            r.get('status') == 'success'
            for r in results.values()
        )
        
        print(f"\n{'✓' if all_passed else '✗'} All tests {'passed' if all_passed else 'had failures'}")
        
        return all_passed
        
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)