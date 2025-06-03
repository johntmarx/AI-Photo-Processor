"""
Pydantic Schema Definitions for AI-Powered Photo Analysis

This module defines the data models used to structure AI responses from the
Llama 3.2 Vision model via Ollama. These schemas ensure consistent, validated
data structures for photo analysis results.

Key Design Principles:

1. Structured Output:
   - Forces the AI model to provide responses in a predictable format
   - Enables programmatic processing of analysis results
   - Prevents free-form responses that would be hard to parse

2. Validation and Constraints:
   - Pydantic provides automatic validation of data types and ranges
   - Field constraints ensure values are within expected bounds
   - Enums restrict certain fields to predefined options

3. Domain-Specific Modeling:
   - Schemas are tailored for swimming/sports photography
   - Includes swimming-specific context (stroke types, pool types)
   - Extensible to other sports or photo types

4. Percentage-Based Coordinates:
   - Bounding boxes use percentages (0-100) instead of pixels
   - Makes the system resolution-independent
   - Simplifies calculations across different image sizes

Integration Points:
   - Used by ai_analyzer.py to parse Ollama responses
   - Consumed by image_processor_v2.py for applying transformations
   - Passed to immich_client.py for metadata generation

Why Pydantic:
   - Industry standard for data validation in Python
   - Automatic JSON serialization/deserialization
   - Clear error messages for validation failures
   - Type hints provide IDE support and documentation
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum


class SubjectType(str, Enum):
    """
    Enumeration of possible primary subjects in photos.
    
    This enum helps the AI categorize what it sees in the image,
    which influences subsequent processing decisions like crop
    suggestions and metadata tagging.
    
    The categories are ordered from most specific (individual swimmer)
    to most general (other), allowing for hierarchical classification.
    
    Swimming-specific subjects are prioritized given the primary use case.
    """
    SWIMMER = "swimmer"                    # Single swimmer in frame
    MULTIPLE_SWIMMERS = "multiple_swimmers" # Multiple swimmers (relay, practice)
    PERSON = "person"                      # Non-swimming individual
    MULTIPLE_PEOPLE = "multiple_people"    # Groups not swimming
    CROWD = "crowd"                        # Large audience/spectators
    POOL = "pool"                          # Pool infrastructure focus
    STARTING_BLOCKS = "starting_blocks"    # Equipment focus
    LANE_ROPES = "lane_ropes"              # Pool dividers focus
    OTHER = "other"                        # Catch-all for unusual subjects

class BoundingBox(BaseModel):
    """
    Represents a rectangular region in an image using percentage coordinates.
    
    Using percentages (0-100) instead of pixel coordinates makes the system
    resolution-independent. A bounding box at (10, 20, 50, 60) means:
    - Starts 10% from the left edge
    - Starts 20% from the top edge  
    - Spans 50% of the image width
    - Spans 60% of the image height
    
    This approach works regardless of whether the image is 800x600 or 4000x3000.
    
    Validation ensures all values are within 0-100 range to prevent
    invalid crops that would extend outside the image boundaries.
    """
    x: float = Field(..., ge=0, le=100, description="Left edge as percentage of image width")
    y: float = Field(..., ge=0, le=100, description="Top edge as percentage of image height") 
    width: float = Field(..., ge=0, le=100, description="Width as percentage of image width")
    height: float = Field(..., ge=0, le=100, description="Height as percentage of image height")

class CropSuggestion(BaseModel):
    """
    Represents a professional cropping and rotation suggestion for an image.
    
    This model combines:
    1. Spatial cropping (what part of the image to keep)
    2. Aspect ratio (shape of the final image)
    3. Rotation correction (fixing tilted horizons or angles)
    
    Common aspect ratios and their uses:
    - '16:9': Widescreen, modern displays, video
    - '4:3': Traditional photo, older displays
    - '3:2': Classic 35mm film ratio, DSLR standard
    - '1:1': Square format, social media friendly
    - '5:4': Medium format, portrait orientation
    
    Rotation is limited to ±45 degrees as larger rotations would likely
    indicate the image needs to be rotated by 90° increments instead.
    
    Default rotation is 0.0 (no rotation) since most photos are already level.
    """
    crop_box: BoundingBox = Field(..., description="Suggested crop area")
    aspect_ratio: str = Field(..., description="Suggested aspect ratio (e.g., '16:9', '4:3', '3:2', '1:1')")
    rotation_degrees: float = Field(0.0, ge=-45, le=45, description="Rotation in degrees (-45 to 45) to straighten/optimize composition")

class ColorAnalysis(BaseModel):
    """
    Comprehensive color and lighting analysis of an image.
    
    This model provides both assessment and correction suggestions for:
    - Exposure (overall brightness)
    - White balance (color temperature)
    - Contrast (tonal range)
    - Dominant colors (for metadata/searching)
    
    Adjustment values use a -100 to +100 scale where:
    - Negative values: Decrease the parameter
    - Zero: No adjustment needed
    - Positive values: Increase the parameter
    
    This standardized scale works across different image processing libraries.
    
    Why these parameters:
    - Indoor pools often have challenging lighting (fluorescent, mixed sources)
    - Action shots may be underexposed due to fast shutter speeds
    - Pool water can create color casts that need correction
    - High contrast helps subjects stand out from backgrounds
    """
    exposure_assessment: Literal['underexposed', 'properly_exposed', 'overexposed'] = Field(..., description="Overall exposure level")
    dominant_colors: List[str] = Field(..., description="List of dominant color names in the image")
    white_balance_assessment: Literal['cool', 'neutral', 'warm'] = Field(..., description="White balance assessment")
    contrast_level: Literal['low', 'normal', 'high'] = Field(..., description="Overall contrast level")
    brightness_adjustment_needed: int = Field(..., ge=-100, le=100, description="Suggested brightness adjustment (-100 to +100)")
    contrast_adjustment_needed: int = Field(..., ge=-100, le=100, description="Suggested contrast adjustment (-100 to +100)")

class SwimmingEventContext(BaseModel):
    """
    Swimming-specific contextual information extracted from the image.
    
    This model captures domain-specific details that are valuable for:
    - Organizing photos by event type
    - Understanding the moment captured (start, finish, etc.)
    - Providing rich metadata for searching
    
    All fields are optional because not every photo will clearly show
    these details. The AI should only populate fields it can confidently
    determine from visual cues.
    
    Event types cover the major swimming disciplines plus related activities.
    Pool types help understand the setting (competitive vs casual).
    Time of event helps contextualize the action (dramatic start vs victory).
    
    Lane numbers (1-10) cover standard pool configurations:
    - 6 lanes: Small pools, high school
    - 8 lanes: Standard competition pools
    - 10 lanes: Large venue pools, championships
    """
    event_type: Optional[Literal['freestyle', 'backstroke', 'breaststroke', 'butterfly', 'individual_medley', 'relay', 'diving', 'water_polo', 'unknown']] = None
    pool_type: Optional[Literal['indoor', 'outdoor', 'competition', 'training', 'unknown']] = None
    time_of_event: Optional[Literal['warm_up', 'race_start', 'mid_race', 'finish', 'celebration', 'unknown']] = None
    lane_number: Optional[int] = Field(None, ge=1, le=10, description="Lane number if identifiable")

class PhotoAnalysis(BaseModel):
    """
    Complete structured analysis of a swimming/sports photo.
    
    This is the top-level model that combines all analysis components into
    a single, comprehensive result. It's designed to capture everything the
    AI determines about the photo in a structured, actionable format.
    
    The model includes:
    1. Subject identification and location
    2. Professional composition suggestions
    3. Technical corrections needed
    4. Domain-specific context
    5. Human-readable description
    
    Required vs Optional Fields:
    - Required fields ensure we always get core analysis results
    - Optional fields allow flexibility for different photo types
    - Default values (like empty lists) prevent null pointer issues
    
    Usage Flow:
    1. AI analyzer populates this model from vision model output
    2. Image processor uses crop/color data to enhance the photo
    3. Immich client uses all fields to create rich metadata
    """
    # Primary subject identification
    # These fields help focus processing on the most important element
    primary_subject: SubjectType = Field(..., description="Type of primary subject in the photo")
    primary_subject_box: BoundingBox = Field(..., description="Bounding box around the primary subject")
    subject_confidence: float = Field(..., ge=0, le=1, description="Confidence in subject identification as decimal (0.0 to 1.0, not percentage)")
    
    # Professional cropping suggestions
    # Primary recommendation is applied, alternatives stored for future use
    recommended_crop: CropSuggestion = Field(..., description="Professional cropping recommendation")
    alternative_crops: List[CropSuggestion] = Field(default=[], description="Alternative cropping options")
    
    # Color and lighting analysis
    # Provides both assessment and correction values
    color_analysis: ColorAnalysis = Field(..., description="Color and lighting assessment")
    
    # Swimming-specific context
    # Populated when swimming elements are detected
    swimming_context: SwimmingEventContext = Field(..., description="Swimming event context if applicable")
    
    # Overall description
    # Human-readable summary for display and search
    description: str = Field(..., description="Natural language description of the photo")
    
    # Technical metadata
    # Helps understand image quality and composition
    estimated_megapixels: Optional[float] = Field(None, description="Estimated megapixel count")
    orientation: Literal['landscape', 'portrait', 'square'] = Field(..., description="Image orientation")