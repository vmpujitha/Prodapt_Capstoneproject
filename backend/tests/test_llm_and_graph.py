import pytest
from unittest.mock import MagicMock, patch
from backend.app.agents.graph import get_advisory_graph, get_graph_mermaid_diagram
from backend.app.llm.fallback import invoke_with_fallback
from backend.app.schemas.classification import QueryClassification

def test_fallback_chain_degraded_mode():
    def dummy_fallback():
        return "Deterministic Degraded Text"

    res, llm_used, fb_triggered, degraded = invoke_with_fallback(
        prompt="Test Prompt",
        preferred_provider="non_existent_provider",
        degraded_fallback_fn=dummy_fallback
    )
    assert res == "Deterministic Degraded Text"
    assert llm_used is None
    assert fb_triggered is True
    assert degraded is True

def test_graph_happy_path():
    graph = get_advisory_graph()
    state = {
        "query": "Which crops do well in Punjab during Kharif with N around 90, P 40, K 40, pH 6.5 and about 1200 mm rainfall?",
        "top_k": 5,
        "mode": "hybrid",
        "trace": []
    }
    result = graph.invoke(state)
    assert result["is_valid"] is True
    assert len(result["recommendations"]) > 0
    assert result["judge_verdict"].passed is True
    assert len(result["trace"]) >= 5

def test_graph_rejection_path():
    graph = get_advisory_graph()
    state = {
        "query": "How do I build a crypto trading bot in Python?",
        "trace": []
    }
    result = graph.invoke(state)
    assert result["is_valid"] is False
    assert "rejection_reason" in result

def test_mermaid_diagram_export():
    diagram = get_graph_mermaid_diagram()
    assert "graph TD" in diagram or "classDef" in diagram or "guardrail_node" in diagram
