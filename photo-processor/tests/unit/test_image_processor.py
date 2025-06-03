"""
Comprehensive unit tests for Image Processor module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np
from PIL import Image
import tempfile
import os
from image_processor import ImageProcessor
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis


class TestImageProcessor:
    """Test suite for ImageProcessor class"""

    @pytest.fixture
    def image_processor(self):
        """Create ImageProcessor instance for testing"""
        return ImageProcessor()

    @pytest.fixture
    def sample_rgb_array(self):
        """Create sample RGB array for testing"""
        # Create a 1000x1000 RGB array
        return np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_photo_analysis(self):
        """Sample photo analysis for testing"""
        return PhotoAnalysis(
            description="Test swimming photo",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=10.0, y=15.0, width=80.0, height=70.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.85
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "white"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=10,
                contrast_adjustment_needed=5
            ),
            swimming_context={
                "event_type": "freestyle",
                "pool_type": "indoor",
                "time_of_event": "mid_race",
                "lane_number": 4
            },
            processing_recommendation="crop_and_enhance"
        )

    def test_initialization(self, image_processor):
        """Test ImageProcessor initialization"""
        assert image_processor is not None

    @patch('image_processor.rawpy.imread')
    def test_load_raw_image_success(self, mock_imread, image_processor):
        """Test successful RAW image loading"""
        # Setup mock
        mock_raw = Mock()
        mock_rgb_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        mock_raw.postprocess.return_value = mock_rgb_array
        mock_imread.return_value = mock_raw

        # Test
        result = image_processor.load_raw_image("/test/image.arw")

        # Assertions
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000, 1000, 3)
        mock_imread.assert_called_once_with("/test/image.arw")
        mock_raw.postprocess.assert_called_once()

    @patch('image_processor.rawpy.imread')
    def test_load_raw_image_file_not_found(self, mock_imread, image_processor):
        """Test RAW image loading with file not found"""
        mock_imread.side_effect = FileNotFoundError("File not found")

        result = image_processor.load_raw_image("/nonexistent/image.arw")

        assert result is None

    @patch('image_processor.rawpy.imread')
    def test_load_raw_image_general_exception(self, mock_imread, image_processor):
        """Test RAW image loading with general exception"""
        mock_imread.side_effect = Exception("Processing error")

        result = image_processor.load_raw_image("/test/image.arw")

        assert result is None

    def test_resize_for_analysis_landscape(self, image_processor, sample_rgb_array):
        """Test resizing landscape image for analysis"""
        # Create landscape image (wider than tall)
        landscape_image = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
        
        result = image_processor.resize_for_analysis(landscape_image, max_dimension=1024)

        # Should maintain aspect ratio and max dimension
        assert result.shape[1] == 1024  # Width should be max dimension
        assert result.shape[0] == int(800 * 1024 / 1200)  # Height scaled proportionally
        assert result.shape[2] == 3  # RGB channels

    def test_resize_for_analysis_portrait(self, image_processor):
        """Test resizing portrait image for analysis"""
        # Create portrait image (taller than wide)
        portrait_image = np.random.randint(0, 255, (1200, 800, 3), dtype=np.uint8)
        
        result = image_processor.resize_for_analysis(portrait_image, max_dimension=1024)

        # Should maintain aspect ratio and max dimension
        assert result.shape[0] == 1024  # Height should be max dimension
        assert result.shape[1] == int(800 * 1024 / 1200)  # Width scaled proportionally
        assert result.shape[2] == 3  # RGB channels

    def test_resize_for_analysis_square(self, image_processor):
        """Test resizing square image for analysis"""
        square_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        result = image_processor.resize_for_analysis(square_image, max_dimension=512)

        assert result.shape == (512, 512, 3)

    def test_resize_for_analysis_already_small(self, image_processor):
        """Test resizing image that's already smaller than max dimension"""
        small_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        result = image_processor.resize_for_analysis(small_image, max_dimension=1024)

        # Should return original image unchanged
        np.testing.assert_array_equal(result, small_image)

    def test_crop_image_valid_bounds(self, image_processor, sample_rgb_array):
        """Test cropping with valid bounds"""
        crop_box = BoundingBox(x=10.0, y=15.0, width=50.0, height=60.0)
        
        result = image_processor.crop_image(sample_rgb_array, crop_box)

        # Calculate expected dimensions
        img_height, img_width = sample_rgb_array.shape[:2]
        expected_width = int(img_width * 0.5)  # 50% width
        expected_height = int(img_height * 0.6)  # 60% height

        assert result.shape[0] == expected_height
        assert result.shape[1] == expected_width
        assert result.shape[2] == 3

    def test_crop_image_bounds_clamping(self, image_processor, sample_rgb_array):
        """Test cropping with bounds that exceed image dimensions"""
        # Crop box that goes beyond image bounds
        crop_box = BoundingBox(x=80.0, y=80.0, width=50.0, height=50.0)
        
        result = image_processor.crop_image(sample_rgb_array, crop_box)

        # Should clamp to image bounds
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_crop_image_zero_dimensions(self, image_processor, sample_rgb_array):
        """Test cropping with zero width or height"""
        crop_box = BoundingBox(x=10.0, y=15.0, width=0.0, height=50.0)
        
        result = image_processor.crop_image(sample_rgb_array, crop_box)

        # Should return None for invalid crop
        assert result is None

    def test_enhance_colors_brightness_increase(self, image_processor, sample_rgb_array):
        """Test color enhancement with brightness increase"""
        color_analysis = ColorAnalysis(
            dominant_colors=["blue"],
            exposure_assessment="underexposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=20,
            contrast_adjustment_needed=0
        )

        result = image_processor.enhance_colors(sample_rgb_array, color_analysis)

        assert result is not None
        assert result.shape == sample_rgb_array.shape
        # Brightness increased - pixels should generally be brighter
        assert np.mean(result) > np.mean(sample_rgb_array)

    def test_enhance_colors_brightness_decrease(self, image_processor, sample_rgb_array):
        """Test color enhancement with brightness decrease"""
        color_analysis = ColorAnalysis(
            dominant_colors=["blue"],
            exposure_assessment="overexposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=-20,
            contrast_adjustment_needed=0
        )

        result = image_processor.enhance_colors(sample_rgb_array, color_analysis)

        assert result is not None
        assert result.shape == sample_rgb_array.shape
        # Brightness decreased - pixels should generally be darker
        assert np.mean(result) < np.mean(sample_rgb_array)

    def test_enhance_colors_contrast_adjustment(self, image_processor, sample_rgb_array):
        """Test color enhancement with contrast adjustment"""
        color_analysis = ColorAnalysis(
            dominant_colors=["blue"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="low",
            brightness_adjustment_needed=0,
            contrast_adjustment_needed=30
        )

        result = image_processor.enhance_colors(sample_rgb_array, color_analysis)

        assert result is not None
        assert result.shape == sample_rgb_array.shape
        # Contrast increased - standard deviation should be higher
        assert np.std(result) > np.std(sample_rgb_array)

    def test_enhance_colors_low_contrast_clahe(self, image_processor):
        """Test CLAHE enhancement for low contrast images"""
        # Create low contrast image
        low_contrast_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        color_analysis = ColorAnalysis(
            dominant_colors=["gray"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="low",
            brightness_adjustment_needed=0,
            contrast_adjustment_needed=0
        )

        with patch('image_processor.cv2.createCLAHE') as mock_clahe:
            mock_clahe_instance = Mock()
            mock_clahe.return_value = mock_clahe_instance
            mock_clahe_instance.apply.return_value = low_contrast_image[:,:,0]

            result = image_processor.enhance_colors(low_contrast_image, color_analysis)

            assert result is not None
            mock_clahe.assert_called()

    @patch('image_processor.cv2.cvtColor')
    def test_auto_white_balance(self, mock_cvtcolor, image_processor, sample_rgb_array):
        """Test automatic white balance correction"""
        # Mock LAB conversion
        mock_cvtcolor.side_effect = [
            np.random.randint(0, 255, sample_rgb_array.shape, dtype=np.uint8),  # RGB to LAB
            sample_rgb_array  # LAB to RGB
        ]

        result = image_processor.auto_white_balance(sample_rgb_array)

        assert result is not None
        assert result.shape == sample_rgb_array.shape
        assert mock_cvtcolor.call_count == 2

    def test_save_image_jpeg(self, image_processor, sample_rgb_array):
        """Test saving image as JPEG"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            output_path = tmp_file.name

        try:
            success = image_processor.save_image(sample_rgb_array, output_path, quality=95)

            assert success is True
            assert os.path.exists(output_path)
            
            # Verify image can be loaded
            saved_image = Image.open(output_path)
            assert saved_image.format == 'JPEG'
            assert saved_image.size == (sample_rgb_array.shape[1], sample_rgb_array.shape[0])

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_image_invalid_path(self, image_processor, sample_rgb_array):
        """Test saving image to invalid path"""
        invalid_path = "/nonexistent/directory/image.jpg"
        
        success = image_processor.save_image(sample_rgb_array, invalid_path)

        assert success is False

    def test_process_photo_crop_and_enhance(self, image_processor, sample_photo_analysis):
        """Test full photo processing with crop and enhance"""
        with patch.object(image_processor, 'load_raw_image') as mock_load, \
             patch.object(image_processor, 'crop_image') as mock_crop, \
             patch.object(image_processor, 'enhance_colors') as mock_enhance, \
             patch.object(image_processor, 'save_image') as mock_save:

            # Setup mocks
            mock_rgb = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            mock_cropped = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
            mock_enhanced = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

            mock_load.return_value = mock_rgb
            mock_crop.return_value = mock_cropped
            mock_enhance.return_value = mock_enhanced
            mock_save.return_value = True

            # Test
            result = image_processor.process_photo(
                "/test/input.arw",
                "/test/output.jpg",
                sample_photo_analysis
            )

            # Assertions
            assert result is True
            mock_load.assert_called_once_with("/test/input.arw")
            mock_crop.assert_called_once()
            mock_enhance.assert_called_once()
            mock_save.assert_called_once()

    def test_process_photo_enhance_only(self, image_processor):
        """Test photo processing with enhance only"""
        # Create analysis for enhance only
        analysis = PhotoAnalysis(
            description="Test photo",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.85
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            ),
            swimming_context={},
            processing_recommendation="enhance_only"
        )

        with patch.object(image_processor, 'load_raw_image') as mock_load, \
             patch.object(image_processor, 'crop_image') as mock_crop, \
             patch.object(image_processor, 'enhance_colors') as mock_enhance, \
             patch.object(image_processor, 'save_image') as mock_save:

            mock_rgb = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            mock_enhanced = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

            mock_load.return_value = mock_rgb
            mock_enhance.return_value = mock_enhanced
            mock_save.return_value = True

            result = image_processor.process_photo("/test/input.arw", "/test/output.jpg", analysis)

            assert result is True
            mock_load.assert_called_once()
            mock_crop.assert_not_called()  # Should not crop for enhance_only
            mock_enhance.assert_called_once()
            mock_save.assert_called_once()

    def test_process_photo_no_processing(self, image_processor):
        """Test photo processing with no processing recommendation"""
        analysis = PhotoAnalysis(
            description="Test photo",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.85
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            ),
            swimming_context={},
            processing_recommendation="no_processing"
        )

        with patch.object(image_processor, 'load_raw_image') as mock_load, \
             patch.object(image_processor, 'save_image') as mock_save:

            mock_rgb = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            mock_load.return_value = mock_rgb
            mock_save.return_value = True

            result = image_processor.process_photo("/test/input.arw", "/test/output.jpg", analysis)

            assert result is True
            mock_load.assert_called_once()
            mock_save.assert_called_once()

    def test_process_photo_load_failure(self, image_processor, sample_photo_analysis):
        """Test photo processing when image loading fails"""
        with patch.object(image_processor, 'load_raw_image') as mock_load:
            mock_load.return_value = None

            result = image_processor.process_photo(
                "/test/input.arw",
                "/test/output.jpg",
                sample_photo_analysis
            )

            assert result is False

    def test_process_photo_save_failure(self, image_processor, sample_photo_analysis):
        """Test photo processing when saving fails"""
        with patch.object(image_processor, 'load_raw_image') as mock_load, \
             patch.object(image_processor, 'save_image') as mock_save:

            mock_rgb = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
            mock_load.return_value = mock_rgb
            mock_save.return_value = False

            result = image_processor.process_photo(
                "/test/input.arw",
                "/test/output.jpg",
                sample_photo_analysis
            )

            assert result is False

    def test_get_image_info(self, image_processor):
        """Test getting image information"""
        sample_image = np.random.randint(0, 255, (1200, 800, 3), dtype=np.uint8)
        
        info = image_processor.get_image_info(sample_image)

        assert info['width'] == 800
        assert info['height'] == 1200
        assert info['channels'] == 3
        assert info['dtype'] == 'uint8'
        assert 'file_size_mb' in info

    def test_calculate_crop_coordinates_percentage(self, image_processor):
        """Test calculating crop coordinates from percentages"""
        image_shape = (1000, 800, 3)  # height, width, channels
        crop_box = BoundingBox(x=10.0, y=20.0, width=50.0, height=60.0)
        
        x1, y1, x2, y2 = image_processor._calculate_crop_coordinates(image_shape, crop_box)

        assert x1 == 80  # 10% of 800
        assert y1 == 200  # 20% of 1000
        assert x2 == 480  # x1 + 50% of 800
        assert y2 == 800  # y1 + 60% of 1000

    def test_pixel_value_clamping(self, image_processor):
        """Test that pixel values are properly clamped to valid range"""
        # Create image with values that could overflow
        test_image = np.array([[[300, -50, 128]]], dtype=np.float32)
        
        result = image_processor._clamp_pixel_values(test_image)

        assert result[0, 0, 0] == 255  # Clamped from 300
        assert result[0, 0, 1] == 0    # Clamped from -50
        assert result[0, 0, 2] == 128  # Unchanged
        assert result.dtype == np.uint8