"""
Processing models for queue and status management
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class QueueItem(BaseModel):
    """Item in the processing queue"""
    photo_id: str = Field(..., description="Photo identifier")
    filename: Optional[str] = Field(None, description="Photo filename")
    position: Optional[int] = Field(None, description="Position in queue")
    added_at: Optional[datetime] = Field(None, description="When added to queue")
    created_at: Optional[datetime] = Field(None, description="When created")
    priority: Literal["low", "normal", "high"] = Field("normal", description="Processing priority")
    recipe_id: Optional[str] = Field(None, description="Recipe to apply")
    manual_approval: bool = Field(False, description="Requires manual approval")
    estimated_time: Optional[float] = Field(None, description="Estimated processing time")

class ProcessingStatus(BaseModel):
    """Current processing status"""
    is_paused: bool = Field(..., description="Whether processing is paused")
    current_photo: Optional[Dict[str, Any]] = Field(None, description="Currently processing photo")
    queue_length: int = Field(..., description="Number of items in queue")
    processing_rate: float = Field(..., description="Photos per minute")
    average_time: float = Field(..., description="Average processing time in seconds")
    errors_today: int = Field(0, description="Number of errors today")

class QueueStatus(BaseModel):
    """Complete queue status"""
    pending: List[QueueItem] = Field(..., description="Items waiting to be processed")
    processing: List[QueueItem] = Field(..., description="Items currently being processed")
    completed: List[QueueItem] = Field(..., description="Recently completed items")
    is_paused: bool = Field(..., description="Whether processing is paused")
    stats: ProcessingStatus = Field(..., description="Processing statistics")

class ProcessingApproval(BaseModel):
    """Manual approval request"""
    photo_id: str
    ai_suggestions: Dict[str, Any] = Field(..., description="AI processing suggestions")
    preview_url: Optional[str] = Field(None, description="Preview of processed result")
    adjustments: Dict[str, Any] = Field(default_factory=dict, description="Manual adjustments")

class ProcessingResult(BaseModel):
    """Result of processing operation"""
    photo_id: str
    success: bool
    processed_path: Optional[str] = None
    processing_time: float
    operations_applied: List[str] = Field(default_factory=list)
    error: Optional[str] = None

class BatchOperation(BaseModel):
    """Batch processing operation"""
    photo_ids: List[str] = Field(..., description="Photos to process")
    recipe_id: Optional[str] = Field(None, description="Recipe to apply to all")
    priority: Literal["low", "normal", "high"] = Field("normal")
    skip_ai: bool = Field(False, description="Skip AI analysis")

class ProcessingSettings(BaseModel):
    """Global processing settings"""
    auto_process: bool = Field(True, description="Automatically process new photos")
    require_approval: bool = Field(False, description="Require manual approval")
    max_concurrent: int = Field(1, description="Max concurrent processing")
    quality_threshold: float = Field(7.0, description="Minimum quality score")
    enable_ai: bool = Field(True, description="Enable AI analysis")