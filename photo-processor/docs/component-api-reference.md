# Component API Reference

## Overview
This document captures the actual API methods discovered during testing. Use this as the source of truth for component integration.

## Core Components

### HashTracker
**Location**: `hash_tracker.py`

**Actual Methods**:
- `is_already_processed(filepath: str) -> bool`
- `mark_as_processed(filepath: str) -> None`

**NOT**: `is_processed()` or `mark_processed()`

### ImageProcessor
**Location**: `image_processor_v2.py`

**Actual Methods**:
- `convert_raw_to_rgb(input_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, dict]`
- `save_high_quality_jpeg(image: np.ndarray, output_path: str, quality: int = 95) -> None`
- `enhance_image(image: np.ndarray, color_analysis: ColorAnalysis) -> np.ndarray`
- `is_raw_file(filepath: str) -> bool`

**DOES NOT HAVE**:
- `load_image()` - Use cv2.imread directly
- `save_image()` - Use save_high_quality_jpeg instead

### AIAnalyzer
**Location**: `ai_analyzer.py`

**Class Name**: `AIAnalyzer` (NOT `AIPhotoAnalyzer`)

**Methods**:
- `analyze_photo(image_path: str) -> Dict[str, Any]`

### EnhancedImmichClient
**Location**: `immich_client_v2.py`

**Key Methods**:
- `upload_photo_pair(original_path: Path, processed_path: Path, recipe: ProcessingRecipe, original_album: str = "Original Files", processed_album: str = "Processed Photos") -> DualUploadResult`
- `upload_asset(file_path: Path, metadata: Dict[str, Any], album_name: Optional[str] = None) -> UploadResult`
- `find_or_create_album(album_name: str) -> str`
- `check_asset_exists(file_hash: str) -> bool`

### RecipeStorage
**Location**: `recipe_storage.py`

**Methods**:
- `save_recipe(recipe: ProcessingRecipe) -> bool`
- `load_recipe(recipe_id: str) -> Optional[ProcessingRecipe]`
- `find_recipe_by_hash(original_hash: str) -> Optional[ProcessingRecipe]`
- `list_recipes() -> List[Dict[str, Any]]`

## Data Models

### ColorAnalysis (from schemas.py)
**Required Fields**:
- `exposure_assessment`: Literal['underexposed', 'properly_exposed', 'overexposed']
- `dominant_colors`: List[str]
- `white_balance_assessment`: Literal['cool', 'neutral', 'warm']
- `contrast_level`: Literal['low', 'normal', 'high']  # NOT 'good'
- `brightness_adjustment_needed`: int  # NOT float, range -100 to 100
- `contrast_adjustment_needed`: int  # NOT float, range -100 to 100

### ProcessingRecipe
**Fields**:
- `id`: str (auto-generated UUID)
- `original_hash`: str
- `original_filename`: str
- `operations`: List[ProcessingOperation]
- `ai_metadata`: Dict[str, Any]
- `created_at`: datetime
- `version`: str

### ProcessingOperation
**Fields**:
- `type`: str (e.g., 'rotate', 'crop', 'enhance')
- `parameters`: Dict[str, Any]
- `order`: int
- `source`: str (e.g., 'ai', 'user')

## Image Processing Patterns

### Loading Images
```python
import cv2
import numpy as np

# For regular images
image = cv2.imread(str(file_path))
if image is None:
    raise ValueError(f"Failed to load image: {file_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# For RAW files
if image_processor.is_raw_file(str(file_path)):
    image, metadata = image_processor.convert_raw_to_rgb(str(file_path))
```

### Applying Transformations
```python
# Rotation
if angle != 0:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

# Cropping (using normalized coordinates 0-1)
height, width = image.shape[:2]
x1 = int(crop['x1'] * width)
y1 = int(crop['y1'] * height)
x2 = int(crop['x2'] * width)
y2 = int(crop['y2'] * height)
image = image[y1:y2, x1:x2]

# Enhancement
color_analysis = ColorAnalysis(...)
image = image_processor.enhance_image(image, color_analysis)
```

### Saving Images
```python
# Always use save_high_quality_jpeg for processed images
image_processor.save_high_quality_jpeg(
    image,
    str(output_path),
    quality=95
)
```

## Testing Patterns

### Mocking cv2
```python
# Correct way - use numpy arrays
mock_cv2 = MagicMock()
mock_image = np.zeros((1000, 1500, 3), dtype=np.uint8)
mock_cv2.imread.return_value = mock_image
mock_cv2.cvtColor.return_value = mock_image
mock_cv2.getRotationMatrix2D.return_value = np.eye(2, 3)
mock_cv2.warpAffine.return_value = mock_image
mock_cv2.COLOR_BGR2RGB = 4

# Inject into sys.modules for runtime imports
sys.modules['cv2'] = mock_cv2
```

### Creating Test Fixtures
```python
# Temporary directories
@pytest.fixture
def temp_dirs(self):
    base_dir = tempfile.mkdtemp()
    dirs = {
        'inbox': Path(base_dir) / 'inbox',
        'originals': Path(base_dir) / 'originals',
        'processed': Path(base_dir) / 'processed',
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    yield dirs
    shutil.rmtree(base_dir)

# Test images
@pytest.fixture
def test_image(self, temp_dirs):
    image_path = temp_dirs['inbox'] / 'test_photo.jpg'
    image_path.write_bytes(b'fake image data')
    return image_path
```

## Common Pitfalls to Avoid

1. **Don't assume method names** - Always check the actual implementation
2. **Don't use float where int is expected** - Schemas are strict
3. **Don't mock with wrong types** - cv2 expects numpy arrays
4. **Don't move files** - Always copy originals first
5. **Don't stop on single failures** - Continue processing other files

This reference should be kept updated as new components are added or APIs change.