"""
Pydantic schemas for structured AI responses from Ollama
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum


class SubjectType(str, Enum):
    SWIMMER = "swimmer"
    MULTIPLE_SWIMMERS = "multiple_swimmers"
    PERSON = "person"
    MULTIPLE_PEOPLE = "multiple_people"
    CROWD = "crowd"
    POOL = "pool"
    STARTING_BLOCKS = "starting_blocks"
    LANE_ROPES = "lane_ropes"
    OTHER = "other"

class BoundingBox(BaseModel):
    """Bounding box coordinates as percentages (0-100) of image dimensions"""
    x: float = Field(..., ge=0, le=100, description="Left edge as percentage of image width")
    y: float = Field(..., ge=0, le=100, description="Top edge as percentage of image height") 
    width: float = Field(..., ge=0, le=100, description="Width as percentage of image width")
    height: float = Field(..., ge=0, le=100, description="Height as percentage of image height")

class CropSuggestion(BaseModel):
    """Professional cropping suggestion"""
    crop_box: BoundingBox = Field(..., description="Suggested crop area")
    aspect_ratio: str = Field(..., description="Suggested aspect ratio (e.g., '16:9', '4:3', '3:2', '1:1')")
    rotation_degrees: float = Field(0.0, ge=-45, le=45, description="Rotation in degrees (-45 to 45) to straighten/optimize composition")

class ColorAnalysis(BaseModel):
    """Color and lighting analysis"""
    exposure_assessment: Literal['underexposed', 'properly_exposed', 'overexposed'] = Field(..., description="Overall exposure level")
    dominant_colors: List[str] = Field(..., description="List of dominant color names in the image")
    white_balance_assessment: Literal['cool', 'neutral', 'warm'] = Field(..., description="White balance assessment")
    contrast_level: Literal['low', 'normal', 'high'] = Field(..., description="Overall contrast level")
    brightness_adjustment_needed: int = Field(..., ge=-100, le=100, description="Suggested brightness adjustment (-100 to +100)")
    contrast_adjustment_needed: int = Field(..., ge=-100, le=100, description="Suggested contrast adjustment (-100 to +100)")

class SwimmingEventContext(BaseModel):
    """Swimming event specific context"""
    event_type: Optional[Literal['freestyle', 'backstroke', 'breaststroke', 'butterfly', 'individual_medley', 'relay', 'diving', 'water_polo', 'unknown']] = None
    pool_type: Optional[Literal['indoor', 'outdoor', 'competition', 'training', 'unknown']] = None
    time_of_event: Optional[Literal['warm_up', 'race_start', 'mid_race', 'finish', 'celebration', 'unknown']] = None
    lane_number: Optional[int] = Field(None, ge=1, le=10, description="Lane number if identifiable")

class PhotoAnalysis(BaseModel):
    """Complete structured analysis of a swimming/sports photo"""
    # Primary subject identification
    primary_subject: SubjectType = Field(..., description="Type of primary subject in the photo")
    primary_subject_box: BoundingBox = Field(..., description="Bounding box around the primary subject")
    subject_confidence: float = Field(..., ge=0, le=1, description="Confidence in subject identification as decimal (0.0 to 1.0, not percentage)")
    
    # Professional cropping suggestions
    recommended_crop: CropSuggestion = Field(..., description="Professional cropping recommendation")
    alternative_crops: List[CropSuggestion] = Field(default=[], description="Alternative cropping options")
    
    # Color and lighting analysis
    color_analysis: ColorAnalysis = Field(..., description="Color and lighting assessment")
    
    # Swimming-specific context
    swimming_context: SwimmingEventContext = Field(..., description="Swimming event context if applicable")
    
    # Overall description
    description: str = Field(..., description="Natural language description of the photo")
    
    # Technical metadata
    estimated_megapixels: Optional[float] = Field(None, description="Estimated megapixel count")
    orientation: Literal['landscape', 'portrait', 'square'] = Field(..., description="Image orientation")