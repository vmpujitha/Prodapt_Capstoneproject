import logging
from langgraph.graph import StateGraph, START, END

from backend.app.agents.state import AdvisoryState
from backend.app.agents.nodes import (
    guardrail_node,
    classifier_node,
    retriever_node,
    analyst_node,
    yield_node,
    risk_node,
    recommender_node,
    judge_node,
    finalize_node
)

logger = logging.getLogger(__name__)

def check_guardrail_condition(state: AdvisoryState) -> str:
    """Routes to finalize immediately if input was rejected by guardrails."""
    if not state.get("is_valid", True):
        return "finalize_node"
    return "classifier_node"

def check_classifier_condition(state: AdvisoryState) -> str:
    """Out-of-scope queries skip retrieval and analysis entirely."""
    if state.get("out_of_scope"):
        return "finalize_node"
    return "retriever_node"

def check_judge_condition(state: AdvisoryState) -> str:
    """
    Always advances to finalize after the judge pass.
    The revision loop is disabled to keep LLM calls at exactly 3 per query
    (classifier + recommender + judge) for predictable eval latency.
    """
    return "finalize_node"

def build_crop_advisory_graph():
    """
    Constructs and compiles the multi-agent agricultural advisory StateGraph.
    """
    builder = StateGraph(AdvisoryState)

    # Add agent nodes
    builder.add_node("guardrail_node", guardrail_node)
    builder.add_node("classifier_node", classifier_node)
    builder.add_node("retriever_node", retriever_node)
    builder.add_node("analyst_node", analyst_node)
    builder.add_node("yield_node", yield_node)
    builder.add_node("risk_node", risk_node)
    builder.add_node("recommender_node", recommender_node)
    builder.add_node("judge_node", judge_node)
    builder.add_node("finalize_node", finalize_node)

    # Set flow
    builder.add_edge(START, "guardrail_node")

    # Conditional routing after guardrail
    builder.add_conditional_edges(
        "guardrail_node",
        check_guardrail_condition,
        {
            "classifier_node": "classifier_node",
            "finalize_node": "finalize_node"
        }
    )

    builder.add_conditional_edges(
        "classifier_node",
        check_classifier_condition,
        {
            "retriever_node": "retriever_node",
            "finalize_node": "finalize_node"
        }
    )

    # Parallel fan-out from retriever to analyst, yield, risk nodes
    builder.add_edge("retriever_node", "analyst_node")
    builder.add_edge("retriever_node", "yield_node")
    builder.add_edge("retriever_node", "risk_node")

    # Fan-in to recommender node
    builder.add_edge("analyst_node", "recommender_node")
    builder.add_edge("yield_node", "recommender_node")
    builder.add_edge("risk_node", "recommender_node")

    builder.add_edge("recommender_node", "judge_node")
    builder.add_edge("judge_node", "finalize_node")

    builder.add_edge("finalize_node", END)

    compiled_graph = builder.compile()
    return compiled_graph

_COMPILED_GRAPH = None

def get_advisory_graph():
    """Returns singleton compiled LangGraph StateGraph instance."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_crop_advisory_graph()
    return _COMPILED_GRAPH

def get_graph_mermaid_diagram() -> str:
    """Returns Mermaid markdown representation of compiled LangGraph."""
    graph = get_advisory_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Error extracting Mermaid from LangGraph: {e}")
        return """
graph TD
    __start__([START]) --> guardrail_node
    guardrail_node -->|Valid| classifier_node
    guardrail_node -->|Invalid| finalize_node
    classifier_node -->|In scope| retriever_node
    classifier_node -->|Out of scope| finalize_node
    retriever_node --> analyst_node
    retriever_node --> yield_node
    retriever_node --> risk_node
    analyst_node --> recommender_node
    yield_node --> recommender_node
    risk_node --> recommender_node
    recommender_node --> judge_node
    judge_node --> finalize_node
    finalize_node --> __end__([END])
"""
