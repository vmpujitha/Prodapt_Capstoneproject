import os
import time
import logging
import httpx
from typing import List, Optional, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from backend.app.config import settings
from backend.app.schemas.common import ProviderStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL-cached provider health checks
# ---------------------------------------------------------------------------
_STATUS_CACHE_TTL = 30  # seconds

_ollama_cache: Tuple[float, Tuple[bool, Optional[str], List[str], List[str]]] = (0.0, (False, "not checked yet", [], []))
_openai_cache: Tuple[float, Tuple[bool, Optional[str], List[str], List[str]]] = (0.0, (False, "not checked yet", [], []))


def check_ollama_status() -> Tuple[bool, Optional[str], List[str], List[str]]:
    """
    Pings Ollama server at OLLAMA_BASE_URL/api/tags with a 2-second timeout.
    Returns (available, reason, chat_models, embedding_models).
    Result is cached for 30 seconds to avoid repeated HTTP pings during eval runs.
    """
    global _ollama_cache
    now = time.monotonic()
    if now - _ollama_cache[0] < _STATUS_CACHE_TTL:
        return _ollama_cache[1]

    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                chat_models = [m for m in models if "embed" not in m.lower()]
                embed_models = [m for m in models if "embed" in m.lower()]
                if not chat_models:
                    chat_models = [settings.OLLAMA_CHAT_MODEL]
                if not embed_models:
                    embed_models = [settings.OLLAMA_EMBED_MODEL]
                result = (True, None, chat_models, embed_models)
            else:
                result = (False, f"Ollama returned HTTP status {resp.status_code}", [], [])
    except Exception as e:
        result = (False, f"Ollama not reachable at {settings.OLLAMA_BASE_URL} ({type(e).__name__})", [], [])

    _ollama_cache = (now, result)
    return result


def check_openai_status() -> Tuple[bool, Optional[str], List[str], List[str]]:
    """
    Checks OpenAI API key configuration.
    Result is cached for 30 seconds.
    """
    global _openai_cache
    now = time.monotonic()
    if now - _openai_cache[0] < _STATUS_CACHE_TTL:
        return _openai_cache[1]

    key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not key or not key.strip() or "your_" in key.lower():
        result = (False, "OPENAI_API_KEY not set", [], [])
    else:
        result = (True, None, ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], ["text-embedding-3-small", "text-embedding-3-large"])

    _openai_cache = (now, result)
    return result


def invalidate_status_cache() -> None:
    """Force the next check_ollama_status / check_openai_status to hit the network."""
    global _ollama_cache, _openai_cache
    _ollama_cache = (0.0, _ollama_cache[1])
    _openai_cache = (0.0, _openai_cache[1])

def list_providers() -> List[ProviderStatus]:
    """
    Returns the operational availability of Ollama and OpenAI providers.
    """
    ollama_ok, ollama_reason, ollama_chat, ollama_embed = check_ollama_status()
    openai_ok, openai_reason, openai_chat, openai_embed = check_openai_status()

    providers = [
        ProviderStatus(
            provider="ollama",
            available=ollama_ok,
            reason=ollama_reason,
            chat_models=ollama_chat if ollama_ok else [settings.OLLAMA_CHAT_MODEL],
            embedding_models=ollama_embed if ollama_ok else [settings.OLLAMA_EMBED_MODEL],
            default_chat_model=settings.OLLAMA_CHAT_MODEL,
            default_embedding_model=settings.OLLAMA_EMBED_MODEL
        ),
        ProviderStatus(
            provider="openai",
            available=openai_ok,
            reason=openai_reason,
            chat_models=openai_chat if openai_ok else ["gpt-4o-mini", "gpt-4o"],
            embedding_models=openai_embed if openai_ok else ["text-embedding-3-small"],
            default_chat_model=settings.OPENAI_CHAT_MODEL,
            default_embedding_model=settings.OPENAI_EMBED_MODEL
        )
    ]
    return providers

def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0
) -> BaseChatModel:
    """
    Instantiates a LangChain Chat Model for the chosen provider.
    """
    prov = (provider or settings.DEFAULT_LLM_PROVIDER).lower()
    
    if prov == "openai":
        from langchain_openai import ChatOpenAI
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        model_name = model or settings.OPENAI_CHAT_MODEL
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    elif prov == "ollama":
        from langchain_ollama import ChatOllama
        model_name = model or settings.OLLAMA_CHAT_MODEL
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=model_name,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

class EmbeddingProviderUnavailable(RuntimeError):
    """Raised when the requested embedding provider cannot be used (no key / server down)."""


def default_embed_model(provider: Optional[str]) -> str:
    """Configured default embedding model for `provider`."""
    prov = (provider or settings.DEFAULT_EMBEDDING_PROVIDER).lower()
    return settings.OPENAI_EMBED_MODEL if prov == "openai" else settings.OLLAMA_EMBED_MODEL


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Embeddings:
    """
    Instantiates a LangChain Embeddings model for the requested provider.

    Never substitutes another provider: vectors from different models are not comparable,
    and each Chroma collection is tied to one provider/model. Raises
    EmbeddingProviderUnavailable when the provider cannot be used; callers degrade to
    keyword (BM25) retrieval instead.
    """
    prov = (provider or settings.DEFAULT_EMBEDDING_PROVIDER).lower()
    model_name = model or default_embed_model(prov)

    if prov == "openai":
        available, reason, _, _ = check_openai_status()
        if not available:
            raise EmbeddingProviderUnavailable(f"OpenAI embeddings unavailable: {reason}")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name, api_key=settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""))

    if prov == "ollama":
        available, reason, _, _ = check_ollama_status()
        if not available:
            raise EmbeddingProviderUnavailable(f"Ollama embeddings unavailable: {reason}")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=model_name)

    raise ValueError(f"Unsupported embedding provider: {provider}")
