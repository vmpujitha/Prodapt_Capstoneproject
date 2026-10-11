from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from backend.app.schemas.prediction import RiskAssessment, YieldPredictionInterval
from backend.app.schemas.search import MetadataFilters

class CropRecommendation(BaseModel):
    crop: str
    score: float = Field(..., description="Overall suitability score (0-1)")
    median_yield: float = Field(..., description="Historical median yield in t/ha")
    mean_yield: float = Field(default=0.0, description="Historical mean yield in t/ha")
    yield_percentile: float = Field(default=50.0, description="Mean yield percentile in historical cases")
    similarity: float = Field(default=0.0, description="Average similarity of matching cases")
    case_count: int = Field(default=1, description="Number of supporting historical cases")
    supporting_cases: List[str] = Field(default_factory=list, description="Case IDs supporting this recommendation")
    risk: Optional[RiskAssessment] = None
    predicted_yield: Optional[YieldPredictionInterval] = None
    score_breakdown: Optional[Dict[str, float]] = None

class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[MetadataFilters] = Field(default_factory=MetadataFilters)
    top_k: int = Field(default=5, ge=1, le=20)
    llm: Optional[Dict[str, Optional[str]]] = None
    embedding: Optional[Dict[str, Optional[str]]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[CropRecommendation]
    explanation: str
    supporting_cases_count: int
    llm_used: Optional[Dict[str, Optional[str]]] = None
    fallback_triggered: bool = False
    degraded_mode: bool = False
