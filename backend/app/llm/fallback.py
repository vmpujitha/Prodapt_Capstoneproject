import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.config import settings
from backend.app.llm.factory import get_chat_model, check_ollama_status, check_openai_status

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

def invoke_with_fallback(
    prompt: Any,
    structured_schema: Optional[Type[T]] = None,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    degraded_fallback_fn: Optional[Callable[[], T | str]] = None
) -> Tuple[Any, Optional[Dict[str, str]], bool, bool]:
    """
    Executes an LLM prompt through an automatic fallback hierarchy:
    1. Preferred provider/model
    2. Alternate provider (if available)
    3. Degraded mode (deterministic fallback function)

    Returns: (result, llm_used_dict, fallback_triggered, degraded_mode)
    """
    primary_prov = preferred_provider or settings.DEFAULT_LLM_PROVIDER
    primary_model = preferred_model or (
        settings.OLLAMA_CHAT_MODEL if primary_prov == "ollama" else settings.OPENAI_CHAT_MODEL
    )

    alt_prov = "openai" if primary_prov == "ollama" else "ollama"
    alt_model = settings.OPENAI_CHAT_MODEL if alt_prov == "openai" else settings.OLLAMA_CHAT_MODEL

    # Candidate sequence
    candidates = [
        (primary_prov, primary_model),
        (alt_prov, alt_model)
    ]

    ollama_avail, _, _, _ = check_ollama_status()
    openai_avail, _, _, _ = check_openai_status()
    avail_map = {"ollama": ollama_avail, "openai": openai_avail}

    for idx, (prov, mdl) in enumerate(candidates):
        if not avail_map.get(prov, False):
            continue
        try:
            logger.info(f"Attempting LLM invocation with {prov}:{mdl}")
            chat_model = get_chat_model(provider=prov, model=mdl)
            if structured_schema:
                try:
                    runnable = chat_model.with_structured_output(structured_schema)
                    res = runnable.invoke(prompt)
                except Exception as struct_err:
                    logger.warning(f"Structured output failed on {prov}:{mdl}: {struct_err}. Trying standard invocation...")
                    raw_res = chat_model.invoke(prompt)
                    # Attempt parsing text/json into schema
                    res = structured_schema.model_validate_json(raw_res.content)
            else:
                raw_res = chat_model.invoke(prompt)
                res = raw_res.content if hasattr(raw_res, "content") else str(raw_res)

            fallback_triggered = (idx > 0)
            return res, {"provider": prov, "model": mdl}, fallback_triggered, False
        except Exception as err:
            logger.warning(f"LLM provider {prov}:{mdl} invocation failed: {err}")

    # Fallback to Degraded Mode
    logger.warning("All configured LLM providers unavailable or failed. Entering Degraded Mode.")
    fallback_result = None
    if degraded_fallback_fn:
        fallback_result = degraded_fallback_fn()
    elif structured_schema:
        # Default empty model if available
        try:
            fallback_result = structured_schema()
        except Exception:
            fallback_result = None

    return fallback_result, None, True, True
