from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL, logger


class Embedder:
    """Wrapper around sentence-transformers for text embeddings."""

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(name)
            logger.info("Embedder initialized with model '%s'", name)
        except Exception as exc:  # pragma: no cover - startup failure
            logger.exception("Failed to load embedding model '%s'", name)
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a dense vector."""
        if not text:
            raise ValueError("Text for embedding must be non-empty")
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed text")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if not texts:
            raise ValueError("Texts for embedding must be non-empty")
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed batch of texts")
            raise

