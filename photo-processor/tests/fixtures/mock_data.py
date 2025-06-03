"""
Mock data and fixtures for testing
"""
import numpy as np
from PIL import Image
import tempfile
import os
from schemas import PhotoAnalysis, BoundingBox, CropSuggestion, ColorAnalysis, SwimmingContext


class MockImageData:
    """Mock image data generator for testing"""
    
    @staticmethod
    def create_rgb_array(width=1000, height=1000, channels=3):
        """Create a random RGB array for testing"""
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def create_low_contrast_image(width=100, height=100):
        """Create a low contrast image for testing CLAHE enhancement"""
        return np.full((height, width, 3), 128, dtype=np.uint8)
    
    @staticmethod
    def create_high_contrast_image(width=100, height=100):
        """Create a high contrast image for testing"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Create alternating black and white stripes
        image[:, ::2] = 255
        return image
    
    @staticmethod
    def create_test_image_file(width=800, height=600, format='JPEG'):
        """Create a temporary test image file"""
        image = Image.fromarray(MockImageData.create_rgb_array(width, height), 'RGB')
        temp_file = tempfile.NamedTemporaryFile(suffix=f'.{format.lower()}', delete=False)
        image.save(temp_file.name, format=format)
        return temp_file.name
    
    @staticmethod
    def create_mock_raw_file(size_mb=50):
        """Create a mock RAW file with specified size"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.arw', delete=False)
        # Write fake RAW data
        data = b'FAKE_RAW_DATA' * (size_mb * 1024 * 1024 // 13)
        temp_file.write(data)
        temp_file.close()
        return temp_file.name


class MockPhotoAnalysisData:
    """Mock photo analysis data for testing"""
    
    @staticmethod
    def crisp_swimmer_analysis():
        """Create a crisp swimmer photo analysis"""
        return PhotoAnalysis(
            description="A crisp photo of a swimmer performing freestyle stroke with excellent technique",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=25.0, y=30.0, width=50.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=10.0, y=15.0, width=80.0, height=70.0),
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=0.95
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "white", "turquoise"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=5
            ),
            swimming_context=SwimmingContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=4
            ),
            processing_recommendation="crop_and_enhance"
        )
    
    @staticmethod
    def blurry_photo_analysis():
        """Create a blurry photo analysis"""
        return PhotoAnalysis(
            description="A blurry photo with motion blur, not suitable for processing",
            quality="very_blurry",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=30.0, y=35.0, width=40.0, height=30.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=20.0, y=25.0, width=60.0, height=50.0),
                aspect_ratio="16:9",
                composition_rule="center_crop",
                confidence=0.3
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "gray"],
                exposure_assessment="underexposed",
                white_balance_assessment="cool",
                contrast_level="low",
                brightness_adjustment_needed=20,
                contrast_adjustment_needed=15
            ),
            swimming_context=SwimmingContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="start",
                lane_number=3
            ),
            processing_recommendation="no_processing"
        )
    
    @staticmethod
    def multiple_swimmers_analysis():
        """Create analysis for photo with multiple swimmers"""
        return PhotoAnalysis(
            description="Dynamic photo showing multiple swimmers in a competitive race",
            quality="slightly_blurry",
            primary_subject="multiple_swimmers",
            primary_subject_box=BoundingBox(x=10.0, y=20.0, width=80.0, height=60.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=5.0, y=10.0, width=90.0, height=80.0),
                aspect_ratio="16:9",
                composition_rule="golden_ratio",
                confidence=0.75
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "white", "green"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=-5,
                contrast_adjustment_needed=10
            ),
            swimming_context=SwimmingContext(
                event_type="relay",
                pool_type="outdoor",
                time_of_event="mid_race",
                lane_number=None
            ),
            processing_recommendation="crop_and_enhance"
        )
    
    @staticmethod
    def backstroke_analysis():
        """Create analysis for backstroke swimming photo"""
        return PhotoAnalysis(
            description="Clean backstroke technique captured at the perfect moment",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=35.0, y=25.0, width=30.0, height=50.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=20.0, y=10.0, width=60.0, height=80.0),
                aspect_ratio="4:3",
                composition_rule="rule_of_thirds",
                confidence=0.88
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "white"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="warm",
                contrast_level="good",
                brightness_adjustment_needed=3,
                contrast_adjustment_needed=0
            ),
            swimming_context=SwimmingContext(
                event_type="backstroke",
                pool_type="indoor",
                time_of_event="finish",
                lane_number=2
            ),
            processing_recommendation="crop_and_enhance"
        )
    
    @staticmethod
    def enhance_only_analysis():
        """Create analysis that only needs enhancement"""
        return PhotoAnalysis(
            description="Well-composed photo that only needs color enhancement",
            quality="crisp",
            primary_subject="swimmer",
            primary_subject_box=BoundingBox(x=40.0, y=30.0, width=20.0, height=40.0),
            recommended_crop=CropSuggestion(
                crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                aspect_ratio="16:9",
                composition_rule="subject_focus",
                confidence=0.92
            ),
            color_analysis=ColorAnalysis(
                dominant_colors=["blue", "turquoise"],
                exposure_assessment="underexposed",
                white_balance_assessment="cool",
                contrast_level="low",
                brightness_adjustment_needed=15,
                contrast_adjustment_needed=20
            ),
            swimming_context=SwimmingContext(
                event_type="butterfly",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=5
            ),
            processing_recommendation="enhance_only"
        )


class MockOllamaResponses:
    """Mock Ollama API responses for testing"""
    
    @staticmethod
    def successful_model_list():
        """Mock successful model list response"""
        from unittest.mock import Mock
        
        mock_model = Mock()
        mock_model.model = "gemma3:4b"
        mock_model.modified_at = "2025-06-01T18:45:22.535806942Z"
        mock_model.size = 3338801804
        
        mock_response = Mock()
        mock_response.models = [mock_model]
        return mock_response
    
    @staticmethod
    def successful_chat_response(analysis_data):
        """Mock successful chat response with analysis data"""
        from unittest.mock import Mock
        import json
        
        mock_response = Mock()
        mock_response.message.content = json.dumps(analysis_data.model_dump())
        return mock_response
    
    @staticmethod
    def failed_chat_response():
        """Mock failed chat response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.message.content = "Invalid JSON response"
        return mock_response


class MockImmichResponses:
    """Mock Immich API responses for testing"""
    
    @staticmethod
    def successful_ping():
        """Mock successful ping response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        return mock_response
    
    @staticmethod
    def failed_ping():
        """Mock failed ping response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        return mock_response
    
    @staticmethod
    def successful_album_creation():
        """Mock successful album creation response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "test-album-id-123",
            "albumName": "AI Processed Photos",
            "description": "Auto-processed photos from AI photo pipeline"
        }
        return mock_response
    
    @staticmethod
    def existing_albums_list():
        """Mock existing albums list response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "album-1",
                "albumName": "Family Photos",
                "description": "Family memories"
            },
            {
                "id": "album-2",
                "albumName": "AI Processed Photos",
                "description": "Auto-processed photos"
            },
            {
                "id": "album-3",
                "albumName": "Vacation",
                "description": "Vacation photos"
            }
        ]
        return mock_response
    
    @staticmethod
    def successful_upload():
        """Mock successful photo upload response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "asset-id-456",
            "deviceAssetId": "ai-processed-DSC09123.ARW",
            "deviceId": "ai-photo-processor",
            "originalFileName": "DSC09123_processed.jpg",
            "fileSize": 2048576
        }
        return mock_response
    
    @staticmethod
    def successful_album_add():
        """Mock successful album asset addition response"""
        from unittest.mock import Mock
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "added": 1
        }
        return mock_response


class TestFileGenerator:
    """Generate test files for testing"""
    
    @staticmethod
    def create_test_arw_files(directory, count=3):
        """Create test ARW files in specified directory"""
        files_created = []
        for i in range(count):
            filename = f"DSC0900{i:02d}.ARW"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'wb') as f:
                # Write fake RAW file header and data
                f.write(b'SONY_ARW_HEADER')
                f.write(b'FAKE_RAW_DATA' * 10000)  # ~130KB file
            files_created.append(filepath)
        return files_created
    
    @staticmethod
    def create_test_mixed_files(directory):
        """Create mixed test files (various formats)"""
        files_created = []
        
        formats = [
            ("IMG_001.CR2", b'CANON_CR2_HEADER'),
            ("DSC_002.NEF", b'NIKON_NEF_HEADER'),
            ("_MG_003.DNG", b'ADOBE_DNG_HEADER'),
            ("PHOTO_004.JPG", b'\xff\xd8\xff\xe0'),  # JPEG header
            ("IMAGE_005.PNG", b'\x89PNG\r\n\x1a\n'),  # PNG header
            ("UNSUPPORTED.TXT", b'This is not an image file'),
        ]
        
        for filename, header in formats:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'wb') as f:
                f.write(header)
                f.write(b'FAKE_IMAGE_DATA' * 1000)
            files_created.append(filepath)
        
        return files_created
    
    @staticmethod
    def cleanup_test_files(file_paths):
        """Clean up test files"""
        for filepath in file_paths:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
            except OSError:
                pass  # Ignore cleanup errors


# Utility functions for tests
def assert_photo_analysis_equal(analysis1, analysis2):
    """Assert that two PhotoAnalysis objects are equal"""
    assert analysis1.description == analysis2.description
    assert analysis1.quality == analysis2.quality
    assert analysis1.primary_subject == analysis2.primary_subject
    assert analysis1.processing_recommendation == analysis2.processing_recommendation
    
    # Compare bounding boxes
    assert analysis1.primary_subject_box.x == analysis2.primary_subject_box.x
    assert analysis1.primary_subject_box.y == analysis2.primary_subject_box.y
    assert analysis1.primary_subject_box.width == analysis2.primary_subject_box.width
    assert analysis1.primary_subject_box.height == analysis2.primary_subject_box.height
    
    # Compare crop suggestions
    assert analysis1.recommended_crop.aspect_ratio == analysis2.recommended_crop.aspect_ratio
    assert analysis1.recommended_crop.composition_rule == analysis2.recommended_crop.composition_rule
    assert analysis1.recommended_crop.confidence == analysis2.recommended_crop.confidence


def create_temp_directory_structure():
    """Create temporary directory structure for testing"""
    temp_dir = tempfile.mkdtemp()
    
    structure = {
        'base': temp_dir,
        'inbox': os.path.join(temp_dir, 'inbox'),
        'processed': os.path.join(temp_dir, 'processed'),
        'temp': os.path.join(temp_dir, 'temp'),
        'failed': os.path.join(temp_dir, 'failed'),
    }
    
    for path in structure.values():
        if path != temp_dir:  # Don't create base dir twice
            os.makedirs(path, exist_ok=True)
    
    return structure