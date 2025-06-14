"""
Photo models for API responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

class Photo(BaseModel):
    """Basic photo information"""
    id: str = Field(..., description="Unique photo identifier (hash)")
    filename: str = Field(..., description="Original filename")
    original_path: str = Field(..., description="Path to original file")
    processed_path: Optional[str] = Field(None, description="Path to processed file")
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail file")
    web_path: Optional[str] = Field(None, description="Path to web-optimized version")
    status: Literal["pending", "processing", "completed", "failed", "rejected"] = Field(
        ..., description="Current processing status"
    )
    created_at: datetime = Field(..., description="When photo was added")
    processed_at: Optional[datetime] = Field(None, description="When processing completed")
    file_size: int = Field(..., description="Original file size in bytes")
    
class PhotoDetail(Photo):
    """Detailed photo information including AI analysis and recipe"""
    recipe_id: Optional[str] = Field(None, description="Applied recipe ID")
    recipe_name: Optional[str] = Field(None, description="Applied recipe name")
    ai_analysis: Optional[Dict[str, Any]] = Field(None, description="AI analysis results")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class PhotoList(BaseModel):
    """Paginated photo list response"""
    photos: List[Photo] = Field(..., description="List of photos")
    total: int = Field(..., description="Total number of photos")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")

class PhotoComparison(BaseModel):
    """Photo comparison data for before/after view"""
    photo_id: str
    original_url: Optional[str] = Field(None, description="URL to original photo")
    processed_url: Optional[str] = Field(None, description="URL to processed photo")
    original: Optional[Dict[str, Any]] = Field(None, description="Original photo info")
    processed: Optional[Dict[str, Any]] = Field(None, description="Processed photo info")
    ai_overlays: Optional[Dict[str, Any]] = Field(None, description="AI detection overlays")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Comparison statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class PhotoUploadResponse(BaseModel):
    """Response after uploading a photo"""
    photo_id: str
    filename: str
    status: str = "queued"
    message: str = "Photo added to processing queue"
    queue_position: int