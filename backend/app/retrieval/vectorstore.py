import re
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma

from backend.app.config import settings
from backend.app.llm.factory import get_embeddings, default_embed_model
from backend.app.retrieval.filters import build_chroma_filter
from backend.app.schemas.search import MetadataFilters

logger = logging.getLogger(__name__)

EMBED_RETRY_ATTEMPTS = 3


class CollectionMissing(RuntimeError):
    """Raised when searching a provider/model collection that has not been built yet."""


def get_collection_name(provider: str, model: str) -> str:
    """Generates standardized collection name: cases__{provider}__{model_slug}"""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", model.lower())
    return f"cases__{provider.lower()}__{slug}"


def resolve_embedding_target(provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
    """Returns the (provider, model) pair to use, applying configured defaults."""
    prov = (provider or settings.DEFAULT_EMBEDDING_PROVIDER).lower()
    return prov, (model or default_embed_model(prov))


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns persistent ChromaDB client."""
    chroma_path = Path(settings.CHROMA_DIR)
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def build_or_get_vectorstore(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Chroma:
    """Returns a LangChain Chroma vectorstore bound to the provider/model collection."""
    prov, mdl = resolve_embedding_target(provider, model)
    return Chroma(
        client=get_chroma_client(),
        collection_name=get_collection_name(prov, mdl),
        embedding_function=get_embeddings(provider=prov, model=mdl),
        collection_metadata={"embedding_provider": prov, "embedding_model": mdl},
    )


def check_collection_exists(provider: Optional[str], model: Optional[str]) -> Tuple[bool, int]:
    """Checks whether a collection exists and returns (exists_and_non_empty, count)."""
    prov, mdl = resolve_embedding_target(provider, model)
    try:
        count = get_chroma_client().get_collection(get_collection_name(prov, mdl)).count()
        return count > 0, count
    except Exception:
        return False, 0


def index_documents(
    documents: List[Document],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 200,
    force_reindex: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Batched, idempotent ingestion into the provider/model Chroma collection. Documents are
    upserted by case_id, so re-running only rewrites existing entries. Raises
    EmbeddingProviderUnavailable if the provider cannot embed.
    """
    prov, mdl = resolve_embedding_target(provider, model)
    col_name = get_collection_name(prov, mdl)
    vectorstore = build_or_get_vectorstore(provider=prov, model=mdl)  # fails fast if provider is down

    client = get_chroma_client()
    if force_reindex:
        try:
            client.delete_collection(col_name)
            logger.info(f"Deleted existing collection '{col_name}' for clean reindex.")
        except Exception:
            logger.info(f"Collection '{col_name}' did not exist; building from scratch.")
        vectorstore = build_or_get_vectorstore(provider=prov, model=mdl)

    total = len(documents)
    ids = [doc.metadata.get("case_id", f"ID-{i}") for i, doc in enumerate(documents)]

    # Resume: without force, skip documents that are already in the collection.
    existing = set() if force_reindex else set(client.get_or_create_collection(col_name).get(include=[])["ids"])
    pending = [(doc, cid) for doc, cid in zip(documents, ids) if cid not in existing]
    done = total - len(pending)
    if existing:
        logger.info(f"Collection '{col_name}' already has {len(existing)} documents; indexing the remaining {len(pending)}.")

    n_batches = (len(pending) + batch_size - 1) // batch_size
    logger.info(f"Indexing {len(pending)} documents into '{col_name}' ({n_batches} batches of {batch_size})...")
    if progress_cb:
        progress_cb(done, total)

    for b, i in enumerate(range(0, len(pending), batch_size), start=1):
        chunk = pending[i:i + batch_size]
        _add_with_retry(vectorstore, [d for d, _ in chunk], [c for _, c in chunk])
        done += len(chunk)
        logger.info(f"Indexed batch {b}/{n_batches} ({done}/{total} docs)")
        if progress_cb:
            progress_cb(done, total)

    logger.info(f"Finished indexing collection: {col_name}")
    return total


def _add_with_retry(vectorstore: Chroma, docs: List[Document], ids: List[str],
                    attempts: int = EMBED_RETRY_ATTEMPTS) -> None:
    """
    Adds one batch, retrying with backoff. Local embedding servers occasionally drop a
    request (e.g. Ollama's runner restarting under memory pressure); if a batch keeps
    failing it is split in half, down to single documents, before giving up.
    """
    for attempt in range(1, attempts + 1):
        try:
            vectorstore.add_documents(documents=docs, ids=ids)
            return
        except Exception as e:
            if attempt == attempts:
                if len(docs) == 1:
                    raise
                logger.warning(f"Batch of {len(docs)} failed {attempts} times ({e}); splitting it.")
                mid = len(docs) // 2
                _add_with_retry(vectorstore, docs[:mid], ids[:mid], attempts)
                _add_with_retry(vectorstore, docs[mid:], ids[mid:], attempts)
                return
            wait = min(2 ** attempt, 20)
            logger.warning(f"Embedding batch of {len(docs)} failed (attempt {attempt}/{attempts}: {e}); retrying in {wait}s.")
            time.sleep(wait)


def search_vectorstore(
    query: str,
    top_k: int = 20,
    filters: Optional[MetadataFilters] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> List[Tuple[Document, float]]:
    """
    Similarity search with an optional metadata `where` filter. Returns (Document, similarity)
    pairs with similarity in (0, 1]. Raises CollectionMissing / EmbeddingProviderUnavailable.
    """
    prov, mdl = resolve_embedding_target(provider, model)
    exists, _ = check_collection_exists(prov, mdl)
    if not exists:
        raise CollectionMissing(
            f"Collection '{get_collection_name(prov, mdl)}' has not been built. "
            f"Use POST /api/v1/admin/reindex (or 'Build index' in the UI)."
        )
    vectorstore = build_or_get_vectorstore(provider=prov, model=mdl)
    results = vectorstore.similarity_search_with_score(query=query, k=top_k, filter=build_chroma_filter(filters))
    # Map L2 distance (>= 0) to a similarity in (0, 1]
    return [(doc, round(1.0 / (1.0 + max(0.0, float(dist))), 4)) for doc, dist in results]


# ---------------------------------------------------------------------------
# Background index-build jobs (used by the admin API and the UI "Build index" button)
# ---------------------------------------------------------------------------
_INDEX_JOBS: Dict[str, Dict[str, Any]] = {}
_INDEX_LOCK = threading.Lock()


def start_index_job(documents: List[Document], provider: Optional[str], model: Optional[str],
                    force_reindex: bool = True) -> Dict[str, Any]:
    """Starts indexing in a daemon thread and returns the job record (one job per collection)."""
    prov, mdl = resolve_embedding_target(provider, model)
    col_name = get_collection_name(prov, mdl)
    with _INDEX_LOCK:
        for job in _INDEX_JOBS.values():
            if job["collection"] == col_name and job["status"] == "running":
                return job
        job_id = f"index_{uuid.uuid4().hex[:8]}"
        job = {
            "job_id": job_id, "collection": col_name, "provider": prov, "model": mdl,
            "status": "running", "indexed": 0, "total": len(documents), "progress": 0.0,
            "error": None, "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": None,
        }
        _INDEX_JOBS[job_id] = job

    def progress(done: int, total: int) -> None:
        job["indexed"], job["progress"] = done, round(done / max(1, total), 4)

    def run() -> None:
        try:
            index_documents(documents, prov, mdl, force_reindex=force_reindex, progress_cb=progress)
            job["status"] = "completed"
        except Exception as e:
            logger.exception(f"Index job {job_id} failed: {e}")
            job["status"], job["error"] = "failed", str(e)
        finally:
            job["completed_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=run, daemon=True).start()
    return job


def get_index_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _INDEX_JOBS.get(job_id)


def get_running_index_job(provider: Optional[str], model: Optional[str]) -> Optional[Dict[str, Any]]:
    col_name = get_collection_name(*resolve_embedding_target(provider, model))
    for job in _INDEX_JOBS.values():
        if job["collection"] == col_name and job["status"] == "running":
            return job
    return None
