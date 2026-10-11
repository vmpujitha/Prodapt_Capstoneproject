from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SensitivityFactor(BaseModel):
    factor: str
    baseline_value: float
    current_predicted_yield: float
    plus_20_predicted_yield: float
    minus_20_predicted_yield: float
    sensitivity_delta: float

class GlobalImportance(BaseModel):
    feature: str
    importance: float

class ImpactAnalysisRequest(BaseModel):
    crop: str
    season: str
    state: str
    n: float
    p: float
    k: float
    ph: float
    temperature: float
    rainfall: float
    humidity: float
    fertilizer: Optional[float] = None
    pesticide: Optional[float] = None
    area: Optional[float] = None
    crop_year: Optional[int] = None

class ImpactAnalysisResponse(BaseModel):
    crop: str
    yield_unit: str = "t/ha"
    importance_method: str = "permutation importance on held-out rows of this crop"
    global_importance: List[GlobalImportance]
    what_if_sensitivities: List[SensitivityFactor]
    summary: str

class FarmProfile(BaseModel):
    name: str = "Farm A"
    crop: str
    season: str
    state: str
    n: float
    p: float
    k: float
    ph: float
    temperature: float
    rainfall: float
    humidity: float
    fertilizer: Optional[float] = None
    pesticide: Optional[float] = None
    area: Optional[float] = None
    actual_yield: Optional[float] = None

class CompareRequest(BaseModel):
    profiles: List[FarmProfile] = Field(..., min_length=2, max_length=5)

class YieldGapFactor(BaseModel):
    factor: str
    farm1_val: float
    farm2_val: float
    difference: float
    estimated_impact_on_yield: float
    direction: str
    explanation: str

class CompareResponse(BaseModel):
    crop: str
    yield_unit: str = "t/ha"
    profiles: List[Dict[str, Any]]
    yield_gap: float
    ranked_factors: List[YieldGapFactor]
    comparison_summary: str


class YieldGapGroup(BaseModel):
    name: str
    case_count: int
    median_yield: float
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Median conditions of the group")
    example_case_ids: List[str] = Field(default_factory=list)
    example_cases: List[Dict[str, Any]] = Field(default_factory=list, description="case_id, narrative, metadata")


class YieldGapAnalysis(BaseModel):
    crop: str
    season: Optional[str] = None
    grouping: str = Field(..., description="'state' (named regions) or 'yield_quartile' (best vs worst seasons)")
    yield_unit: str = "t/ha"
    groups: List[YieldGapGroup]
    yield_gap: float = Field(..., description="Median yield of the second group minus the first")
    ranked_factors: List[YieldGapFactor]
    summary: str
    caveat: str
