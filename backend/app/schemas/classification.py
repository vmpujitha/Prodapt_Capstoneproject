from typing import Optional, Literal
from pydantic import BaseModel, Field

class SoilCondition(BaseModel):
    n: Optional[float] = Field(None, description="Nitrogen content (kg/ha)")
    p: Optional[float] = Field(None, description="Phosphorus content (kg/ha)")
    k: Optional[float] = Field(None, description="Potassium content (kg/ha)")
    ph: Optional[float] = Field(None, description="Soil pH (0-14)")
    n_level: Optional[str] = Field(None, description="low | medium | high")
    p_level: Optional[str] = Field(None, description="low | medium | high")
    k_level: Optional[str] = Field(None, description="low | medium | high")
    ph_class: Optional[str] = Field(None, description="acidic | neutral | alkaline")

class WeatherCondition(BaseModel):
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    rainfall: Optional[float] = Field(None, description="Rainfall in mm")
    humidity: Optional[float] = Field(None, description="Relative humidity %")
    temp_class: Optional[str] = Field(None, description="cool | moderate | hot")
    rainfall_class: Optional[str] = Field(None, description="low | moderate | high")
    humidity_class: Optional[str] = Field(None, description="low | moderate | high")

class InputCondition(BaseModel):
    fertilizer: Optional[float] = Field(None, description="Fertilizer amount in kg/ha")
    pesticide: Optional[float] = Field(None, description="Pesticide amount in kg/ha")

class QueryClassification(BaseModel):
    intent: Literal[
        "recommend_crop",
        "explain_yield_difference",
        "alternative_crop",
        "similar_cases",
        "out_of_scope"
    ] = Field(default="recommend_crop", description="Classified intent of the farming query")
    crop: Optional[str] = Field(None, description="Target crop if specified")
    season: Optional[str] = Field(None, description="Agricultural season (e.g., Kharif, Rabi, Zaid)")
    state: Optional[str] = Field(None, description="Geographical state/region")
    soil: SoilCondition = Field(default_factory=SoilCondition)
    weather: WeatherCondition = Field(default_factory=WeatherCondition)
    inputs: InputCondition = Field(default_factory=InputCondition)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Classification confidence")

class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    llm: Optional[dict] = None

class ClassifyResponse(BaseModel):
    classification: QueryClassification
    warnings: list[str] = Field(default_factory=list, description="'Did you mean' suggestions for unknown values")
    llm_used: Optional[dict] = None
    fallback_triggered: bool = False
    degraded_mode: bool = False
