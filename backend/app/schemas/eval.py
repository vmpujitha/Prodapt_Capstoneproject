from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EvalMetricSummary(BaseModel):
    metric_name: str
    score: float
    description: str


class EvalSampleResult(BaseModel):
    id: Optional[str] = None
    query: str
    scenario: str
    expected_crops: List[str]
    expected_crops_resolved: List[str] = Field(default_factory=list, description="Expected crops mapped to dataset names")
    retrieved_cases: List[str]
    retrieved_crops: List[str] = Field(default_factory=list)
    recommended_crops: List[str] = Field(default_factory=list)
    hit_at_k: bool = Field(..., description="An expected crop is among the top-k retrieved cases' crops")
    recommendation_hit: bool = False
    judge_passed: Optional[bool] = None
    degraded_mode: bool = False
    # DeepEval scores (0-1); None when the metric could not be run
    relevancy_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    contextual_relevancy_score: Optional[float] = None
    contextual_precision_score: Optional[float] = None
    geval_score: Optional[float] = None
    metric_reasons: Dict[str, str] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    reason: Optional[str] = None


class EvalRunRequest(BaseModel):
    llm: Optional[Dict[str, Optional[str]]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=100, description="Number of golden queries (default: all)")
    resume: bool = Field(default=True, description="Resume from checkpoint if a previous run was interrupted")


class EvalRunResponse(BaseModel):
    run_id: str
    status: str
    total_samples: int
    message: str


class EvalStatusResponse(BaseModel):
    run_id: str
    status: str  # pending | running | completed | failed
    progress: float  # 0.0 to 1.0
    total_samples: int
    completed_samples: int
    results: Optional[List[EvalSampleResult]] = None
    aggregate_metrics: Optional[Dict[str, Optional[float]]] = None
    deepeval_status: Optional[str] = Field(None, description="'ran', or why the DeepEval LLM metrics were skipped")
    judge_model: Optional[str] = None
    is_synthetic_data: Optional[bool] = None
    resumed_samples: int = Field(default=0, description="Number of samples loaded from a previous checkpoint")
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
