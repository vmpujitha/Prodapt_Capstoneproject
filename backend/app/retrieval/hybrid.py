import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from langchain_core.documents import Document

from backend.app.config import settings
from backend.app.retrieval.vectorstore import search_vectorstore
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.retrieval.ranking import prioritize_cases, diversify_by_crop
from backend.app.schemas.search import MetadataFilters, CaseDocument

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    vector_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    alpha: float = 0.5,
    k_rrf: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combines ranked results using Reciprocal Rank Fusion:
    RRF(d) = (1 - alpha) * (1 / (k_rrf + rank_vec)) + alpha * (1 / (k_rrf + rank_bm25))
    """
    doc_map: Dict[str, Document] = {}
    rrf_scores: Dict[str, float] = defaultdict(float)

    # Add vector ranks
    for rank, (doc, _) in enumerate(vector_results):
        case_id = doc.metadata.get("case_id", doc.page_content)
        doc_map[case_id] = doc
        rrf_scores[case_id] += (1.0 - alpha) * (1.0 / (k_rrf + rank + 1))

    # Add BM25 ranks
    for rank, (doc, _) in enumerate(bm25_results):
        case_id = doc.metadata.get("case_id", doc.page_content)
        doc_map[case_id] = doc
        rrf_scores[case_id] += alpha * (1.0 / (k_rrf + rank + 1))

    if not rrf_scores:
        return []

    # Min-max normalization of RRF scores to [0, 1]
    max_rrf = max(rrf_scores.values())
    min_rrf = min(rrf_scores.values())
    spread = max(1e-6, max_rrf - min_rrf)

    merged: List[Tuple[Document, float]] = []
    for case_id, score in rrf_scores.items():
        doc = doc_map[case_id]
        norm_score = (score - min_rrf) / spread if max_rrf > min_rrf else 0.8
        merged.append((doc, norm_score))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

def search_with_mode(
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
    filters: Optional[MetadataFilters] = None,
    user_conditions: Optional[Dict[str, Optional[float]]] = None,
    feedback_boosts: Optional[Dict[str, float]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Tuple[List[CaseDocument], str, Optional[str]]:
    """
    Runs keyword, semantic or hybrid (RRF) retrieval with the same metadata filters on both
    retrievers, then prioritizes the results.

    If the vector side is unusable (embedding provider down or collection not built), it
    degrades to keyword search. Returns (ranked_cases, effective_mode, note) where `note`
    explains any degradation.
    """
    # A wide candidate pool lets condition proximity / yield percentile do the fine ranking.
    fetch_k = max(top_k * 10, 100)
    effective_mode = mode
    note: Optional[str] = None

    vec_docs: List[Tuple[Document, float]] = []
    if mode in ("semantic", "hybrid"):
        try:
            vec_docs = search_vectorstore(query=query, top_k=fetch_k, filters=filters,
                                          provider=provider, model=model)
        except Exception as e:
            note = f"Semantic search unavailable ({e}); used keyword (BM25) search only."
            logger.warning(note)
            effective_mode = "keyword"

    if effective_mode == "keyword":
        scored_docs = get_bm25_retriever().search(query=query, top_k=fetch_k, filters=filters)
    elif effective_mode == "semantic":
        scored_docs = vec_docs
    else:
        bm25_docs = get_bm25_retriever().search(query=query, top_k=fetch_k, filters=filters)
        scored_docs = reciprocal_rank_fusion(vector_results=vec_docs, bm25_results=bm25_docs,
                                             alpha=settings.HYBRID_ALPHA)

    ranked_cases = prioritize_cases(
        retrieved_pairs=scored_docs,
        user_conditions=user_conditions,
        feedback_boosts=feedback_boosts
    )
    if filters is None or not filters.crop:
        ranked_cases = diversify_by_crop(ranked_cases, top_k, max_per_crop=max(2, (top_k + 2) // 3))
    return ranked_cases[:top_k], effective_mode, note


def hybrid_search(
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",
    filters: Optional[MetadataFilters] = None,
    user_conditions: Optional[Dict[str, Optional[float]]] = None,
    feedback_boosts: Optional[Dict[str, float]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> List[CaseDocument]:
    """Same as `search_with_mode` but returns only the ranked cases."""
    cases, _, _ = search_with_mode(query, top_k, mode, filters, user_conditions, feedback_boosts, provider, model)
    return cases
