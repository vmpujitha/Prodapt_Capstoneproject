from typing import Dict, Any, List, Optional, TypedDict, Annotated
import operator

from backend.app.schemas.classification import QueryClassification
from backend.app.schemas.search import CaseDocument, MetadataFilters
from backend.app.schemas.recommendation import CropRecommendation
from backend.app.schemas.prediction import YieldPredictionInterval, RiskAssessment
from backend.app.schemas.analysis import ImpactAnalysisResponse, YieldGapAnalysis
from backend.app.schemas.query import JudgeVerdict

class AdvisoryState(TypedDict, total=False):
    # Input parameters
    query: str
    filters: Optional[MetadataFilters]
    top_k: int
    mode: str
    preferred_llm_provider: Optional[str]
    preferred_llm_model: Optional[str]
    preferred_embed_provider: Optional[str]
    preferred_embed_model: Optional[str]

    # Pipeline stages
    is_valid: bool
    rejection_reason: Optional[str]
    rejection_code: Optional[str]
    rejection_status: Optional[int]
    out_of_scope: bool
    classification: Optional[QueryClassification]
    warnings: List[str]
    ranked_cases: Optional[List[CaseDocument]]
    filters_applied: Dict[str, Any]
    effective_search_mode: str

    # Parallel node outputs
    soil_weather_analysis: Optional[ImpactAnalysisResponse]
    yield_gap: Optional[YieldGapAnalysis]
    yield_predictions: Optional[Dict[str, YieldPredictionInterval]]
    risk_assessments: Optional[Dict[str, RiskAssessment]]

    # Synthesis & validation
    recommendations: Optional[List[CropRecommendation]]
    explanation: Optional[str]
    judge_verdict: Optional[JudgeVerdict]
    revision_count: int

    # Metadata & trace (trace is concatenated automatically via operator.add)
    trace: Annotated[List[Dict[str, Any]], operator.add]
    llm_used: Optional[Dict[str, str]]
    fallback_triggered: bool
    degraded_mode: bool
    disclaimer: str
