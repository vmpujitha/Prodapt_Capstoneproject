from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from backend.app.schemas.classification import QueryClassification
from backend.app.schemas.search import CaseDocument, MetadataFilters
from backend.app.schemas.recommendation import CropRecommendation
from backend.app.schemas.analysis import ImpactAnalysisResponse, YieldGapAnalysis

class JudgeVerdict(BaseModel):
    grounded: int = Field(default=5, ge=1, le=5, description="Grounded in retrieved cases (1-5)")
    consistent_with_data: int = Field(default=5, ge=1, le=5, description="Consistent with model output (1-5)")
    actionable: int = Field(default=5, ge=1, le=5, description="Actionable agronomic advice (1-5)")
    safe: int = Field(default=5, ge=1, le=5, description="Safe and non-hazardous advice (1-5)")
    passed: bool = Field(default=True, description="Whether all scores >= 3 and mean >= 3.5")
    critique: str = Field(default="Recommendation is sound and well-grounded.", description="Judge critique or validation note")

class TraceStep(BaseModel):
    node: str
    ms: int
    summary: str

class PipelineQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language farming query")
    filters: Optional[MetadataFilters] = Field(default_factory=MetadataFilters)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: Literal["hybrid", "semantic", "keyword"] = Field(default="hybrid")
    llm: Optional[Dict[str, Optional[str]]] = None
    embedding: Optional[Dict[str, Optional[str]]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "query": "Which crops do well in Punjab during Kharif with N around 90, P 40, K 40, pH 6.5 and about 1200 mm rainfall?",
                "filters": {},
                "top_k": 8,
                "mode": "hybrid",
                "llm": {"provider": "ollama", "model": "llama3.1:8b"},
                "embedding": {"provider": "ollama", "model": "nomic-embed-text"}
            }]
        }
    }

class PipelineQueryResponse(BaseModel):
    classification: QueryClassification
    similar_cases: List[CaseDocument]
    recommendations: List[CropRecommendation]
    impact_analysis: Optional[ImpactAnalysisResponse] = None
    yield_gap: Optional[YieldGapAnalysis] = Field(None, description="Present for 'why do yields differ' questions")
    explanation: str
    judge: Optional[JudgeVerdict] = None
    agent_trace: List[TraceStep]
    warnings: List[str] = Field(default_factory=list, description="Guardrail notes, unknown values, retrieval degradation")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters used for retrieval")
    search_mode_used: Optional[str] = Field(None, description="Retrieval mode actually used (may degrade to keyword)")
    out_of_scope: bool = False
    is_synthetic_data: bool = False
    llm_used: Optional[Dict[str, Optional[str]]] = None
    fallback_triggered: bool = False
    degraded_mode: bool = False
    disclaimer: str = (
        "Disclaimer: This advisory is generated from historical crop, soil, and weather data patterns "
        "for decision support. Field outcomes may vary based on localized micro-climate, pests, and seed variety. "
        "Consult local agronomy extension services for critical inputs."
    )
