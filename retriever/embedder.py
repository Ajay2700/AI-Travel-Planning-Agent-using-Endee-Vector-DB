from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL, logger


class Embedder:
    """
    Wrapper around sentence-transformers for generating text embeddings.
    
    Embeddings are dense vector representations of text that capture semantic meaning.
    We use 'all-MiniLM-L6-v2' by default - it's a good balance of quality and speed,
    producing 384-dimensional vectors that work well for semantic search.
    
    The model is loaded once at initialization and reused for all embeddings,
    which is efficient for batch operations during ingestion.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the embedding model.
        
        The model will be downloaded from HuggingFace on first use if not cached.
        This can take a minute or two the first time, but subsequent runs are fast.
        """
        name = model_name or EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(name)
            logger.info("Embedder initialized with model '%s'", name)
        except Exception as exc:  # pragma: no cover - startup failure
            logger.exception("Failed to load embedding model '%s'", name)
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a dense vector.
        
        This is used for query embeddings during search. The resulting vector
        can be compared with document vectors using cosine similarity to find
        semantically similar content.
        """
        if not text:
            raise ValueError("Text for embedding must be non-empty")
        try:
            # convert_to_numpy=True gives us numpy arrays, which work well with Endee
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed text")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts efficiently.
        
        This is much faster than calling embed_text() in a loop because
        sentence-transformers can process batches in parallel. Used during
        data ingestion to embed all travel documents at once.
        """
        if not texts:
            raise ValueError("Texts for embedding must be non-empty")
        try:
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed batch of texts")
            raise

