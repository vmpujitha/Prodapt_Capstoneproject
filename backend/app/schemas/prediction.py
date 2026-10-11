from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class YieldPredictionInterval(BaseModel):
    value: float = Field(..., description="Expected crop yield")
    low: float = Field(..., description="Lower bound (10th percentile of held-out residuals)")
    high: float = Field(..., description="Upper bound (90th percentile of held-out residuals)")
    unit: str = Field(default="t/ha", description="Yield unit (t/ha; nuts/ha for Coconut)")

class YieldPredictionRequest(BaseModel):
    crop: str
    season: str
    state: str
    n: float = Field(..., ge=0, le=500)
    p: float = Field(..., ge=0, le=500)
    k: float = Field(..., ge=0, le=500)
    ph: float = Field(..., ge=0, le=14)
    temperature: float = Field(..., ge=-10, le=55)
    rainfall: float = Field(..., ge=0, le=5000)
    humidity: float = Field(..., ge=0, le=100)
    fertilizer: Optional[float] = Field(default=None, ge=0, description="kg/ha; crop median if omitted")
    pesticide: Optional[float] = Field(default=None, ge=0, description="kg/ha; crop median if omitted")
    area: Optional[float] = Field(default=None, ge=0, description="ha; crop median if omitted")
    crop_year: Optional[int] = Field(default=None, ge=1950, description="latest dataset year if omitted")

class YieldPredictionResponse(BaseModel):
    crop: str
    prediction: YieldPredictionInterval
    model_type: str
    trained_on: str
    metrics: Dict[str, Any] = Field(default_factory=dict)

class RiskFactor(BaseModel):
    factor: str
    impact: str
    description: str

class RiskPredictionRequest(BaseModel):
    crop: str
    season: str
    state: str
    n: float = Field(..., ge=0, le=500)
    p: float = Field(..., ge=0, le=500)
    k: float = Field(..., ge=0, le=500)
    ph: float = Field(..., ge=0, le=14)
    temperature: float = Field(..., ge=-10, le=55)
    rainfall: float = Field(..., ge=0, le=5000)
    humidity: float = Field(..., ge=0, le=100)
    fertilizer: Optional[float] = None
    pesticide: Optional[float] = None
    area: Optional[float] = None
    crop_year: Optional[int] = None

class RiskAssessment(BaseModel):
    level: str = Field(..., description="low | medium | high")
    probability: float = Field(..., description="P(yield < 25th percentile)")
    factors: List[RiskFactor] = Field(default_factory=list)

class RiskPredictionResponse(BaseModel):
    crop: str
    risk: RiskAssessment
