"""
DeepEval judge model backed by the project's LangChain chat models (Ollama or OpenAI).
"""
import os
import json
import re
import logging
from typing import Any, Optional, Type

from pydantic import BaseModel

from backend.app.config import settings
from backend.app.llm.factory import get_chat_model

# DeepEval sends anonymous telemetry by default; keep evaluation fully local unless the user opts in.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _content(message: Any) -> str:
    return message.content if hasattr(message, "content") else str(message)


def _parse_into(schema: Type[BaseModel], text: str) -> BaseModel:
    match = _JSON_BLOCK_RE.search(text)
    return schema.model_validate(json.loads(match.group(0) if match else text))


try:
    from deepeval.models.base_model import DeepEvalBaseLLM

    class CustomDeepEvalLLM(DeepEvalBaseLLM):
        """
        DeepEvalBaseLLM wrapper around a LangChain chat model so DeepEval metrics can be
        scored by Ollama or OpenAI. When DeepEval passes a Pydantic `schema`, the model is
        asked for structured output, with a JSON-parsing fallback for models without tool calling.
        """
        def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
            self.provider = (provider or settings.JUDGE_PROVIDER or settings.DEFAULT_LLM_PROVIDER).lower()
            self.model_name = model_name or settings.JUDGE_MODEL or (
                settings.OLLAMA_CHAT_MODEL if self.provider == "ollama" else settings.OPENAI_CHAT_MODEL
            )
            self._chat_model = None
            super().__init__(model=self.model_name)

        def load_model(self):
            if self._chat_model is None:
                self._chat_model = get_chat_model(self.provider, self.model_name, temperature=0.0)
            return self._chat_model

        def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
            model = self.load_model()
            if schema is None:
                return _content(model.invoke(prompt))
            try:
                return model.with_structured_output(schema).invoke(prompt)
            except Exception as e:
                logger.info(f"Structured output failed on {self.get_model_name()} ({e}); parsing JSON from text.")
                return _parse_into(schema, _content(model.invoke(prompt)))

        async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None) -> Any:
            model = self.load_model()
            if schema is None:
                return _content(await model.ainvoke(prompt))
            try:
                return await model.with_structured_output(schema).ainvoke(prompt)
            except Exception as e:
                logger.info(f"Structured output failed on {self.get_model_name()} ({e}); parsing JSON from text.")
                return _parse_into(schema, _content(await model.ainvoke(prompt)))

        def get_model_name(self) -> str:
            return f"{self.provider}:{self.model_name}"

    DEEPEVAL_IMPORT_ERROR: Optional[str] = None

except Exception as e:  # deepeval missing or incompatible
    logger.warning(f"DeepEval is not available: {e}")
    CustomDeepEvalLLM = None  # type: ignore
    DEEPEVAL_IMPORT_ERROR = str(e)
