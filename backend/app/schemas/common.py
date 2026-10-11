from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: Optional[str] = Field(None, description="LLM provider: ollama | openai")
    model: Optional[str] = Field(None, description="Model identifier")

class EmbeddingConfig(BaseModel):
    provider: Optional[str] = Field(None, description="Embedding provider: ollama | openai")
    model: Optional[str] = Field(None, description="Embedding model identifier")

class ProviderStatus(BaseModel):
    provider: str
    available: bool
    reason: Optional[str] = None
    chat_models: List[str] = Field(default_factory=list)
    embedding_models: List[str] = Field(default_factory=list)
    default_chat_model: Optional[str] = None
    default_embedding_model: Optional[str] = None

class LLMUsedInfo(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
