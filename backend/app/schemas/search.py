from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class MetadataFilters(BaseModel):
    crop: Optional[str] = None
    season: Optional[str] = None
    state: Optional[str] = None
    crop_year_min: Optional[int] = None
    crop_year_max: Optional[int] = None
    ph_class: Optional[str] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    n_level: Optional[str] = None
    p_level: Optional[str] = None
    k_level: Optional[str] = None
    temp_class: Optional[str] = None
    rainfall_class: Optional[str] = None
    humidity_class: Optional[str] = None

class ScoreBreakdown(BaseModel):
    retrieval: float = Field(..., description="Normalized retrieval score (hybrid/semantic/keyword)")
    proximity: float = Field(..., description="Condition proximity based on query soil/weather")
    yield_pct: float = Field(..., description="Yield percentile within crop")
    recency: float = Field(..., description="Year recency score")
    feedback: float = Field(..., description="Feedback boost")
    prevalence: float = Field(0.5, description="How widely the crop is grown in the state/season (share of planted area, softened)")
    final: float = Field(..., description="Composite final score")

class CaseDocument(BaseModel):
    case_id: str
    narrative: str
    metadata: Dict[str, Any]
    scores: ScoreBreakdown

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[MetadataFilters] = Field(default_factory=MetadataFilters)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["hybrid", "semantic", "keyword"] = Field(default="hybrid")
    embedding: Optional[Dict[str, Optional[str]]] = None
    user_conditions: Optional[Dict[str, Optional[float]]] = None

class SearchResponse(BaseModel):
    results: List[CaseDocument]
    total_found: int
    mode_used: str = Field(..., description="Retrieval mode actually used (may degrade to keyword)")
    mode_requested: Optional[str] = None
    collection_used: Optional[str] = None
    note: Optional[str] = None
