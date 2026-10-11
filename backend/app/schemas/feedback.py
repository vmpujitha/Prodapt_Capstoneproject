from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class FeedbackCreate(BaseModel):
    query: str
    case_ids: List[str] = Field(default_factory=list)
    recommended_crop: Optional[str] = None
    rating: int = Field(..., ge=-1, le=1, description="+1 for positive, -1 for negative, 0 for neutral")
    comment: Optional[str] = None
    helpful_case_ids: List[str] = Field(default_factory=list)

class FeedbackResponse(BaseModel):
    id: int
    status: str
    message: str

class FeedbackStats(BaseModel):
    total_feedback: int
    positive_count: int
    negative_count: int
    neutral_count: int
    top_helpful_cases: List[Dict[str, Any]]
    crop_ratings: Dict[str, Dict[str, int]]
