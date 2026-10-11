"""
Test configuration: isolates mutable state (feedback DB, evaluation outputs) in a temporary
directory so test runs never write into the real data/ folder. Must run before any
backend module is imported, because settings are read at import time.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="agri_tests_")
os.environ["FEEDBACK_DB_PATH"] = os.path.join(_TMP, "feedback.db")
os.environ["EVAL_RESULTS_DIR"] = os.path.join(_TMP, "eval")
# Keep tests independent of whichever judge model a developer configured.
os.environ["JUDGE_PROVIDER"] = ""
os.environ["JUDGE_MODEL"] = ""

import pytest

_OFFLINE = (False, "provider disabled in tests", [], [])


@pytest.fixture(autouse=True)
def offline_llm_providers(monkeypatch):
    """
    Tests never call a real Ollama/OpenAI server, even when one is running on the machine.
    Tests that exercise the LLM path patch these again with a fake chat model.
    """
    from backend.app.llm import factory, fallback
    from backend.app.api.v1 import router
    from backend.app.evaluation import deepeval_runner
    for module in (factory, fallback, router, deepeval_runner):
        monkeypatch.setattr(module, "check_ollama_status", lambda: _OFFLINE, raising=False)
        monkeypatch.setattr(module, "check_openai_status", lambda: _OFFLINE, raising=False)
