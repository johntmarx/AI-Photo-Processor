"""
Comprehensive unit tests for Schemas and Data Validation
"""
import pytest
from pydantic import ValidationError
from schemas import (
    BoundingBox, CropSuggestion, ColorAnalysis, 
    SwimmingEventContext, PhotoAnalysis
)


class TestBoundingBox:
    """Test suite for BoundingBox schema"""

    def test_valid_bounding_box(self):
        """Test creation of valid bounding box"""
        bbox = BoundingBox(x=10.0, y=20.0, width=50.0, height=30.0)
        
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 50.0
        assert bbox.height == 30.0

    def test_bounding_box_zero_values(self):
        """Test bounding box with zero values"""
        bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
        
        assert bbox.x == 0.0
        assert bbox.y == 0.0
        assert bbox.width == 0.0
        assert bbox.height == 0.0

    def test_bounding_box_negative_coordinates(self):
        """Test bounding box with negative coordinates (should be allowed)"""
        bbox = BoundingBox(x=-10.0, y=-5.0, width=50.0, height=30.0)
        
        assert bbox.x == -10.0
        assert bbox.y == -5.0

    def test_bounding_box_negative_dimensions(self):
        """Test bounding box with negative dimensions (should be allowed for certain use cases)"""
        bbox = BoundingBox(x=10.0, y=20.0, width=-5.0, height=-3.0)
        
        assert bbox.width == -5.0
        assert bbox.height == -3.0

    def test_bounding_box_float_precision(self):
        """Test bounding box with high precision floats"""
        bbox = BoundingBox(x=10.123456, y=20.987654, width=50.111111, height=30.999999)
        
        assert bbox.x == pytest.approx(10.123456)
        assert bbox.y == pytest.approx(20.987654)
        assert bbox.width == pytest.approx(50.111111)
        assert bbox.height == pytest.approx(30.999999)

    def test_bounding_box_integer_conversion(self):
        """Test bounding box with integer inputs"""
        bbox = BoundingBox(x=10, y=20, width=50, height=30)
        
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 50.0
        assert bbox.height == 30.0

    def test_bounding_box_string_conversion(self):
        """Test bounding box with string inputs that can be converted to float"""
        bbox = BoundingBox(x="10.5", y="20.7", width="50.2", height="30.9")
        
        assert bbox.x == 10.5
        assert bbox.y == 20.7
        assert bbox.width == 50.2
        assert bbox.height == 30.9

    def test_bounding_box_invalid_string(self):
        """Test bounding box with invalid string inputs"""
        with pytest.raises(ValidationError):
            BoundingBox(x="invalid", y=20.0, width=50.0, height=30.0)

    def test_bounding_box_missing_fields(self):
        """Test bounding box with missing required fields"""
        with pytest.raises(ValidationError):
            BoundingBox(x=10.0, y=20.0, width=50.0)  # Missing height


class TestCropSuggestion:
    """Test suite for CropSuggestion schema"""

    def test_valid_crop_suggestion(self):
        """Test creation of valid crop suggestion"""
        crop_box = BoundingBox(x=10.0, y=15.0, width=80.0, height=70.0)
        crop = CropSuggestion(
            crop_box=crop_box,
            aspect_ratio="16:9",
            composition_rule="rule_of_thirds",
            confidence=0.85
        )
        
        assert crop.crop_box == crop_box
        assert crop.aspect_ratio == "16:9"
        assert crop.composition_rule == "rule_of_thirds"
        assert crop.confidence == 0.85

    def test_crop_suggestion_valid_aspect_ratios(self):
        """Test crop suggestion with various valid aspect ratios"""
        crop_box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        
        valid_ratios = ["16:9", "4:3", "3:2", "1:1", "2:3", "9:16"]
        for ratio in valid_ratios:
            crop = CropSuggestion(
                crop_box=crop_box,
                aspect_ratio=ratio,
                composition_rule="rule_of_thirds",
                confidence=0.8
            )
            assert crop.aspect_ratio == ratio

    def test_crop_suggestion_valid_composition_rules(self):
        """Test crop suggestion with various valid composition rules"""
        crop_box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        
        valid_rules = ["rule_of_thirds", "golden_ratio", "center_crop", "subject_focus"]
        for rule in valid_rules:
            crop = CropSuggestion(
                crop_box=crop_box,
                aspect_ratio="16:9",
                composition_rule=rule,
                confidence=0.8
            )
            assert crop.composition_rule == rule

    def test_crop_suggestion_confidence_bounds(self):
        """Test crop suggestion confidence value bounds"""
        crop_box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        
        # Test minimum confidence
        crop_min = CropSuggestion(
            crop_box=crop_box,
            aspect_ratio="16:9",
            composition_rule="rule_of_thirds",
            confidence=0.0
        )
        assert crop_min.confidence == 0.0
        
        # Test maximum confidence
        crop_max = CropSuggestion(
            crop_box=crop_box,
            aspect_ratio="16:9",
            composition_rule="rule_of_thirds",
            confidence=1.0
        )
        assert crop_max.confidence == 1.0

    def test_crop_suggestion_invalid_confidence(self):
        """Test crop suggestion with invalid confidence values"""
        crop_box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        
        # Test confidence > 1.0
        with pytest.raises(ValidationError):
            CropSuggestion(
                crop_box=crop_box,
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=1.5
            )
        
        # Test negative confidence
        with pytest.raises(ValidationError):
            CropSuggestion(
                crop_box=crop_box,
                aspect_ratio="16:9",
                composition_rule="rule_of_thirds",
                confidence=-0.1
            )


class TestColorAnalysis:
    """Test suite for ColorAnalysis schema"""

    def test_valid_color_analysis(self):
        """Test creation of valid color analysis"""
        color = ColorAnalysis(
            dominant_colors=["blue", "white", "green"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=10,
            contrast_adjustment_needed=-5
        )
        
        assert color.dominant_colors == ["blue", "white", "green"]
        assert color.exposure_assessment == "properly_exposed"
        assert color.white_balance_assessment == "neutral"
        assert color.contrast_level == "good"
        assert color.brightness_adjustment_needed == 10
        assert color.contrast_adjustment_needed == -5

    def test_color_analysis_valid_exposures(self):
        """Test color analysis with valid exposure assessments"""
        valid_exposures = ["underexposed", "properly_exposed", "overexposed"]
        
        for exposure in valid_exposures:
            color = ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment=exposure,
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            )
            assert color.exposure_assessment == exposure

    def test_color_analysis_valid_white_balance(self):
        """Test color analysis with valid white balance assessments"""
        valid_wb = ["cool", "neutral", "warm"]
        
        for wb in valid_wb:
            color = ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment="properly_exposed",
                white_balance_assessment=wb,
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            )
            assert color.white_balance_assessment == wb

    def test_color_analysis_valid_contrast_levels(self):
        """Test color analysis with valid contrast levels"""
        valid_contrast = ["low", "good", "high"]
        
        for contrast in valid_contrast:
            color = ColorAnalysis(
                dominant_colors=["blue"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level=contrast,
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            )
            assert color.contrast_level == contrast

    def test_color_analysis_adjustment_bounds(self):
        """Test color analysis adjustment value bounds"""
        # Test extreme values
        color = ColorAnalysis(
            dominant_colors=["red"],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=-100,
            contrast_adjustment_needed=100
        )
        
        assert color.brightness_adjustment_needed == -100
        assert color.contrast_adjustment_needed == 100

    def test_color_analysis_invalid_adjustment_bounds(self):
        """Test color analysis with invalid adjustment values"""
        # Test brightness adjustment out of bounds
        with pytest.raises(ValidationError):
            ColorAnalysis(
                dominant_colors=["red"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=150,  # Too high
                contrast_adjustment_needed=0
            )
        
        with pytest.raises(ValidationError):
            ColorAnalysis(
                dominant_colors=["red"],
                exposure_assessment="properly_exposed",
                white_balance_assessment="neutral",
                contrast_level="good",
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=-150  # Too low
            )

    def test_color_analysis_empty_dominant_colors(self):
        """Test color analysis with empty dominant colors list"""
        color = ColorAnalysis(
            dominant_colors=[],
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=0,
            contrast_adjustment_needed=0
        )
        
        assert color.dominant_colors == []

    def test_color_analysis_many_dominant_colors(self):
        """Test color analysis with many dominant colors"""
        colors_list = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown"]
        color = ColorAnalysis(
            dominant_colors=colors_list,
            exposure_assessment="properly_exposed",
            white_balance_assessment="neutral",
            contrast_level="good",
            brightness_adjustment_needed=0,
            contrast_adjustment_needed=0
        )
        
        assert color.dominant_colors == colors_list


class TestSwimmingContext:
    """Test suite for SwimmingContext schema"""

    def test_valid_swimming_context(self):
        """Test creation of valid swimming context"""
        context = SwimmingEventContext(
            event_type="freestyle",
            pool_type="indoor",
            time_of_event="mid_race",
            lane_number=4
        )
        
        assert context.event_type == "freestyle"
        assert context.pool_type == "indoor"
        assert context.time_of_event == "mid_race"
        assert context.lane_number == 4

    def test_swimming_context_valid_event_types(self):
        """Test swimming context with valid event types"""
        valid_events = ["freestyle", "backstroke", "breaststroke", "butterfly", "medley", "relay", "diving"]
        
        for event in valid_events:
            context = SwimmingEventContext(
                event_type=event,
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=1
            )
            assert context.event_type == event

    def test_swimming_context_valid_pool_types(self):
        """Test swimming context with valid pool types"""
        valid_pools = ["indoor", "outdoor", "short_course", "long_course"]
        
        for pool in valid_pools:
            context = SwimmingEventContext(
                event_type="freestyle",
                pool_type=pool,
                time_of_event="mid_race",
                lane_number=1
            )
            assert context.pool_type == pool

    def test_swimming_context_valid_time_events(self):
        """Test swimming context with valid time of events"""
        valid_times = ["start", "mid_race", "finish", "warmup", "cooldown"]
        
        for time_event in valid_times:
            context = SwimmingEventContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event=time_event,
                lane_number=1
            )
            assert context.time_of_event == time_event

    def test_swimming_context_lane_number_bounds(self):
        """Test swimming context with valid lane numbers"""
        for lane in range(1, 11):  # Lanes 1-10
            context = SwimmingEventContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=lane
            )
            assert context.lane_number == lane

    def test_swimming_context_invalid_lane_number(self):
        """Test swimming context with invalid lane numbers"""
        # Lane 0 should be invalid
        with pytest.raises(ValidationError):
            SwimmingEventContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=0
            )
        
        # Lane > 10 should be invalid
        with pytest.raises(ValidationError):
            SwimmingEventContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=11
            )

    def test_swimming_context_optional_fields(self):
        """Test swimming context with optional fields None"""
        context = SwimmingEventContext(
            event_type=None,
            pool_type=None,
            time_of_event=None,
            lane_number=None
        )
        
        assert context.event_type is None
        assert context.pool_type is None
        assert context.time_of_event is None
        assert context.lane_number is None


class TestPhotoAnalysis:
    """Test suite for PhotoAnalysis schema"""

    def test_valid_photo_analysis(self):
        """Test creation of valid photo analysis"""
        analysis = PhotoAnalysis(
            description="A crisp photo of a swimmer performing freestyle",
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
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            ),
            swimming_context=SwimmingEventContext(
                event_type="freestyle",
                pool_type="indoor",
                time_of_event="mid_race",
                lane_number=4
            ),
            processing_recommendation="crop_and_enhance"
        )
        
        assert analysis.description == "A crisp photo of a swimmer performing freestyle"
        assert analysis.quality == "crisp"
        assert analysis.primary_subject == "swimmer"
        assert analysis.processing_recommendation == "crop_and_enhance"

    def test_photo_analysis_valid_qualities(self):
        """Test photo analysis with valid quality values"""
        valid_qualities = ["crisp", "slightly_blurry", "blurry", "very_blurry"]
        
        for quality in valid_qualities:
            analysis = PhotoAnalysis(
                description="Test photo",
                quality=quality,
                primary_subject="person",
                primary_subject_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                recommended_crop=CropSuggestion(
                    crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                    aspect_ratio="16:9",
                    composition_rule="rule_of_thirds",
                    confidence=0.8
                ),
                color_analysis=ColorAnalysis(
                    dominant_colors=["blue"],
                    exposure_assessment="properly_exposed",
                    white_balance_assessment="neutral",
                    contrast_level="good",
                    brightness_adjustment_needed=0,
                    contrast_adjustment_needed=0
                ),
                swimming_context=SwimmingEventContext(),
                processing_recommendation="enhance_only"
            )
            assert analysis.quality == quality

    def test_photo_analysis_valid_subjects(self):
        """Test photo analysis with valid primary subjects"""
        valid_subjects = ["swimmer", "multiple_swimmers", "person", "crowd", "pool", "equipment"]
        
        for subject in valid_subjects:
            analysis = PhotoAnalysis(
                description="Test photo",
                quality="crisp",
                primary_subject=subject,
                primary_subject_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                recommended_crop=CropSuggestion(
                    crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                    aspect_ratio="16:9",
                    composition_rule="rule_of_thirds",
                    confidence=0.8
                ),
                color_analysis=ColorAnalysis(
                    dominant_colors=["blue"],
                    exposure_assessment="properly_exposed",
                    white_balance_assessment="neutral",
                    contrast_level="good",
                    brightness_adjustment_needed=0,
                    contrast_adjustment_needed=0
                ),
                swimming_context=SwimmingEventContext(),
                processing_recommendation="enhance_only"
            )
            assert analysis.primary_subject == subject

    def test_photo_analysis_valid_processing_recommendations(self):
        """Test photo analysis with valid processing recommendations"""
        valid_recommendations = ["crop_and_enhance", "crop_only", "enhance_only", "no_processing"]
        
        for recommendation in valid_recommendations:
            analysis = PhotoAnalysis(
                description="Test photo",
                quality="crisp",
                primary_subject="swimmer",
                primary_subject_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                recommended_crop=CropSuggestion(
                    crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                    aspect_ratio="16:9",
                    composition_rule="rule_of_thirds",
                    confidence=0.8
                ),
                color_analysis=ColorAnalysis(
                    dominant_colors=["blue"],
                    exposure_assessment="properly_exposed",
                    white_balance_assessment="neutral",
                    contrast_level="good",
                    brightness_adjustment_needed=0,
                    contrast_adjustment_needed=0
                ),
                swimming_context=SwimmingEventContext(),
                processing_recommendation=recommendation
            )
            assert analysis.processing_recommendation == recommendation

    def test_photo_analysis_json_schema(self):
        """Test that PhotoAnalysis can generate JSON schema"""
        schema = PhotoAnalysis.model_json_schema()
        
        assert isinstance(schema, dict)
        assert 'properties' in schema
        assert 'description' in schema['properties']
        assert 'quality' in schema['properties']
        assert 'primary_subject' in schema['properties']
        assert 'processing_recommendation' in schema['properties']

    def test_photo_analysis_serialization(self):
        """Test PhotoAnalysis serialization to JSON"""
        analysis = PhotoAnalysis(
            description="Test photo",
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
                brightness_adjustment_needed=0,
                contrast_adjustment_needed=0
            ),
            swimming_context=SwimmingEventContext(),
            processing_recommendation="crop_and_enhance"
        )
        
        json_data = analysis.model_dump()
        assert isinstance(json_data, dict)
        assert json_data['description'] == "Test photo"
        assert json_data['quality'] == "crisp"

    def test_photo_analysis_deserialization(self):
        """Test PhotoAnalysis deserialization from JSON"""
        json_data = {
            "description": "Test photo",
            "quality": "crisp",
            "primary_subject": "swimmer",
            "primary_subject_box": {"x": 25.0, "y": 30.0, "width": 50.0, "height": 40.0},
            "recommended_crop": {
                "crop_box": {"x": 10.0, "y": 15.0, "width": 80.0, "height": 70.0},
                "aspect_ratio": "16:9",
                "composition_rule": "rule_of_thirds",
                "confidence": 0.85
            },
            "color_analysis": {
                "dominant_colors": ["blue", "white"],
                "exposure_assessment": "properly_exposed",
                "white_balance_assessment": "neutral",
                "contrast_level": "good",
                "brightness_adjustment_needed": 0,
                "contrast_adjustment_needed": 0
            },
            "swimming_context": {},
            "processing_recommendation": "crop_and_enhance"
        }
        
        analysis = PhotoAnalysis.model_validate(json_data)
        assert analysis.description == "Test photo"
        assert analysis.quality == "crisp"
        assert analysis.primary_subject == "swimmer"

    def test_photo_analysis_missing_required_field(self):
        """Test PhotoAnalysis validation with missing required fields"""
        with pytest.raises(ValidationError):
            PhotoAnalysis(
                # Missing description
                quality="crisp",
                primary_subject="swimmer",
                primary_subject_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                recommended_crop=CropSuggestion(
                    crop_box=BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0),
                    aspect_ratio="16:9",
                    composition_rule="rule_of_thirds",
                    confidence=0.8
                ),
                color_analysis=ColorAnalysis(
                    dominant_colors=["blue"],
                    exposure_assessment="properly_exposed",
                    white_balance_assessment="neutral",
                    contrast_level="good",
                    brightness_adjustment_needed=0,
                    contrast_adjustment_needed=0
                ),
                swimming_context=SwimmingEventContext(),
                processing_recommendation="enhance_only"
            )