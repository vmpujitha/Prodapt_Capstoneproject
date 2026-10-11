import os
import re
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from backend.app.config import settings
from backend.app.retrieval.filters import matches_python_filters
from backend.app.schemas.search import MetadataFilters

logger = logging.getLogger(__name__)

# Bumped whenever tokenization changes, so stale on-disk indexes are rebuilt.
TOKENIZER_VERSION = 2

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "about", "around",
    "during", "which", "what", "do", "does", "did", "is", "are", "was", "were", "be", "it", "its",
    "my", "i", "me", "we", "our", "can", "could", "should", "would", "will", "well", "best", "good",
    "how", "why", "when", "where", "that", "this", "these", "those", "than", "then", "there",
    "grown", "grow", "crop", "crops", "soil", "weather", "inputs", "area", "production", "yield",
    "kg", "ha", "mm", "t", "c",
}
YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def simple_tokenize(text: str) -> List[str]:
    """
    Lowercased alphanumeric tokens without stopwords or bare numbers (years are kept).
    Numbers are dropped because matching digits lexically ("1200" mm vs "1200" t of
    production) is noise; numeric closeness is scored by condition proximity in ranking.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and (not t.isdigit() or YEAR_RE.match(t))]

class BM25Retriever:
    """
    Cached BM25 retriever supporting in-memory metadata pre-filtering.
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus_tokens = [simple_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def save(self, cache_path: str):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((TOKENIZER_VERSION, self.documents, self.corpus_tokens, self.bm25), f)
        logger.info(f"Saved BM25 index to: {cache_path}")

    @classmethod
    def load(cls, cache_path: str) -> Optional["BM25Retriever"]:
        if not Path(cache_path).exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if len(payload) != 4 or payload[0] != TOKENIZER_VERSION:
                logger.info("BM25 cache was built with an older tokenizer; rebuilding.")
                return None
            _, documents, corpus_tokens, bm25 = payload
            inst = cls.__new__(cls)
            inst.documents = documents
            inst.corpus_tokens = corpus_tokens
            inst.bm25 = bm25
            return inst
        except Exception as e:
            logger.warning(f"Failed to load cached BM25 index: {e}")
            return None

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[MetadataFilters] = None
    ) -> List[Tuple[Document, float]]:
        """
        Executes BM25 search with metadata pre-filtering and score min-max normalization.
        """
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        # Calculate scores across entire corpus
        raw_scores = self.bm25.get_scores(query_tokens)

        # Filter candidates based on metadata
        candidate_indices = []
        for idx, doc in enumerate(self.documents):
            if matches_python_filters(doc.metadata, filters):
                candidate_indices.append(idx)

        if not candidate_indices:
            return []

        candidate_scores = [raw_scores[idx] for idx in candidate_indices]
        max_score = max(candidate_scores) if candidate_scores else 1.0
        min_score = min(candidate_scores) if candidate_scores else 0.0
        score_range = max(1e-6, max_score - min_score)

        # Pair and sort
        scored_docs = []
        for idx in candidate_indices:
            score = raw_scores[idx]
            # Min-max normalization
            norm_score = max(0.0, min(1.0, (score - min_score) / score_range)) if max_score > 0 else 0.0
            scored_docs.append((self.documents[idx], norm_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

_GLOBAL_BM25: Optional[BM25Retriever] = None


def _cache_is_stale(cache_file: str) -> bool:
    """The on-disk BM25 cache is stale if the processed dataset was written after it."""
    processed = Path(settings.PROCESSED_DATA_PATH)
    cache = Path(cache_file)
    return not cache.exists() or (processed.exists() and processed.stat().st_mtime > cache.stat().st_mtime)


def get_bm25_retriever(documents: Optional[List[Document]] = None) -> BM25Retriever:
    """Returns singleton BM25Retriever, loading a fresh on-disk cache or building it."""
    global _GLOBAL_BM25
    cache_file = os.path.join(settings.BM25_INDEX_DIR, "bm25_index.pkl")

    if _GLOBAL_BM25 is None:
        cached = None if _cache_is_stale(cache_file) else BM25Retriever.load(cache_file)
        if cached is not None:
            _GLOBAL_BM25 = cached
        else:
            if not documents:
                from backend.app.data.ingest import get_active_dataframe, df_to_documents
                documents = df_to_documents(get_active_dataframe())
            logger.info(f"Building BM25 index over {len(documents)} documents...")
            _GLOBAL_BM25 = BM25Retriever(documents)
            _GLOBAL_BM25.save(cache_file)

    return _GLOBAL_BM25


def reset_bm25_cache() -> None:
    """Drops the in-memory index so the next call reloads or rebuilds it."""
    global _GLOBAL_BM25
    _GLOBAL_BM25 = None
