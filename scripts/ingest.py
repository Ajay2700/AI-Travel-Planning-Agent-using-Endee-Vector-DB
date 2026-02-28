import json
import os
import sys
from typing import List, Dict, Any

# Ensure project root is importable when run as `python scripts/ingest.py`
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retriever.embedder import Embedder
from retriever.vector_store import EndeeVectorStore
from utils.config import logger


def load_travel_data(path: str) -> List[Dict[str, Any]]:
    """Load travel data documents from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("travel_data.json must contain a JSON list")
        logger.info("Loaded %d travel documents from %s", len(data), path)
        return data
    except FileNotFoundError:
        logger.error("travel data file not found at %s", path)
        raise
    except Exception:
        logger.exception("Failed to load travel data from %s", path)
        raise


def ingest() -> None:
    """Main ingestion pipeline."""
    data_path = os.path.join(PROJECT_ROOT, "data", "travel_data.json")
    docs = load_travel_data(data_path)

    embedder = Embedder()
    store = EndeeVectorStore()

    # Combine title + description as the text to embed
    texts = [
        f"{doc.get('title', '')} - {doc.get('description', '')}".strip()
        for doc in docs
    ]
    embeddings = embedder.embed_batch(texts)

    store.add_documents(docs, embeddings)
    logger.info("Ingestion finished successfully")


if __name__ == "__main__":
    ingest()

