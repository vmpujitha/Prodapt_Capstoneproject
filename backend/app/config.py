import os
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings
from typing import List

# Base workspace path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.1:8b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    DEFAULT_LLM_PROVIDER: str = "ollama"
    DEFAULT_EMBEDDING_PROVIDER: str = "ollama"

    # Judge Settings (blank = use the chat provider/model selected for the request)
    JUDGE_PROVIDER: str = ""
    JUDGE_MODEL: str = ""

    # Paths
    CHROMA_DIR: str = str(BASE_DIR / "data" / "chroma")
    DATA_CSV_PATH: str = str(BASE_DIR / "data" / "raw" / "crop_yield.csv")
    KAGGLE_SOURCE_DIR: str = str(BASE_DIR / "data_set")
    SYNTHETIC_DATA_PATH: str = str(BASE_DIR / "data" / "raw" / "SYNTHETIC_crop_yield.csv")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "data" / "processed" / "cases.parquet")
    DATASET_SOURCE_PATH: str = str(BASE_DIR / "data" / "processed" / "dataset_source.json")
    THRESHOLDS_PATH: str = str(BASE_DIR / "data" / "processed" / "thresholds.json")
    BM25_INDEX_DIR: str = str(BASE_DIR / "data" / "processed")
    MODELS_DIR: str = str(BASE_DIR / "models")
    FEEDBACK_DB_PATH: str = str(BASE_DIR / "data" / "feedback.db")
    EVAL_RESULTS_DIR: str = str(BASE_DIR / "data" / "eval")

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"

    # Retrieval & Ranking Parameters
    HYBRID_ALPHA: float = 0.5
    RANKING_W1: float = 0.35  # retrieval score weight
    RANKING_W2: float = 0.25  # condition proximity weight
    RANKING_W3: float = 0.20  # yield percentile weight
    RANKING_W4: float = 0.10  # recency weight
    RANKING_W5: float = 0.10  # feedback boost weight
    RANKING_W6: float = 0.30  # crop prevalence (share of planted area in the state/season); 0 disables

    # Guardrails
    RATE_LIMIT_PER_MINUTE: int = 60

    # Evaluation
    EVAL_TIMEOUT_SECONDS: int = 180
    EVAL_MAX_RETRIES: int = 1
    # Comma-separated DeepEval metric names to run.  "all" = the full 5-metric suite.
    # Default is the 3 cheapest, most informative metrics (saves ~12 LLM calls/query).
    EVAL_METRICS: str = "answer_relevancy,crop_recommendation_quality"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        """Relative paths (e.g. ./data/chroma in .env) are resolved against the project root."""
        for name in ("CHROMA_DIR", "DATA_CSV_PATH", "KAGGLE_SOURCE_DIR", "SYNTHETIC_DATA_PATH",
                     "PROCESSED_DATA_PATH", "DATASET_SOURCE_PATH", "THRESHOLDS_PATH", "BM25_INDEX_DIR",
                     "MODELS_DIR", "FEEDBACK_DB_PATH", "EVAL_RESULTS_DIR"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                setattr(self, name, str((BASE_DIR / value).resolve()))
        return self

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

settings = Settings()
