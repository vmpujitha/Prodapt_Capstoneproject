import sys
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.data.ingest import get_active_dataframe, df_to_documents
from backend.app.llm.factory import EmbeddingProviderUnavailable
from backend.app.retrieval.vectorstore import index_documents, resolve_embedding_target, get_collection_name
from backend.app.retrieval.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_indices(provider: str | None = None, model: str | None = None, force: bool = False,
                  fraction: float = 1.0) -> int:
    """Builds the BM25 index and the provider/model Chroma collection. Returns an exit code."""
    df = get_active_dataframe()
    docs = df_to_documents(df)
    logger.info(f"Loaded {len(docs)} case documents.")

    bm25_path = Path(settings.BM25_INDEX_DIR) / "bm25_index.pkl"
    BM25Retriever(docs).save(str(bm25_path))

    if fraction < 1.0:
        # Seeded random sample, so every year, state and crop stays represented.
        import random
        docs = random.Random(42).sample(docs, k=int(len(docs) * fraction))
        logger.info(f"Vector-indexing a {fraction:.0%} sample: {len(docs)} documents (BM25 covers all cases).")

    prov, mdl = resolve_embedding_target(provider, model)
    logger.info(f"Building Chroma collection {get_collection_name(prov, mdl)} (force={force})...")
    try:
        total = index_documents(documents=docs, provider=prov, model=mdl, batch_size=200, force_reindex=force)
    except EmbeddingProviderUnavailable as e:
        logger.error(f"{e}. The BM25 index was built; semantic/hybrid search will fall back to keyword "
                     f"search until this collection is built (start Ollama / set OPENAI_API_KEY and re-run).")
        return 2
    logger.info(f"Indexed {total} documents.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the BM25 index and a Chroma vector collection.")
    parser.add_argument("--provider", default=None, help="ollama | openai (default: DEFAULT_EMBEDDING_PROVIDER)")
    parser.add_argument("--model", default=None, help="Embedding model (default: the provider's configured model)")
    parser.add_argument("--force", action="store_true", help="Drop and rebuild the collection")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Index only a seeded random share of the cases (e.g. 0.7); re-run without it to complete")
    args = parser.parse_args()
    sys.exit(build_indices(provider=args.provider, model=args.model, force=args.force, fraction=args.fraction))
