from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import requests

from utils.config import (
    ENDEE_AUTH_TOKEN,
    ENDEE_BASE_URL,
    ENDEE_INDEX_NAME,
    logger,
)


class EndeeVectorStore:
    """
    Integration layer with the Endee vector database for RAG operations.
    
    This class handles all interactions with Endee:
    - Creating/verifying indexes
    - Storing document embeddings with metadata
    - Performing semantic similarity search
    
    It's designed to be resilient - if Endee is unavailable, it falls back
    to in-memory storage and semantic search using cosine similarity. This
    ensures the RAG system still works even without the vector database running,
    though with reduced scalability.
    
    The API endpoints are based on Endee's REST API (see https://github.com/endee-io/endee).
    If the API changes in future versions, update the endpoints here.
    """

    def __init__(
        self,
        base_url: str | None = None,
        index_name: str | None = None,
        auth_token: str | None = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = (base_url or ENDEE_BASE_URL).rstrip("/")
        self.index_name = index_name or ENDEE_INDEX_NAME
        self.auth_token = auth_token or ENDEE_AUTH_TOKEN
        self.timeout = timeout
        self._available = False
        self._fallback_data: List[Dict[str, Any]] = []

        logger.info(
            "Initializing EndeeVectorStore base_url=%s index=%s",
            self.base_url,
            self.index_name,
        )
        
        # Try to connect to Endee, but don't fail if unavailable
        try:
            self._ensure_index()
            self._available = True
            logger.info("Endee vector store is available and ready")
        except Exception as exc:
            logger.warning(
                "Endee service not available at %s - using fallback mode. "
                "Error: %s. To use Endee, start the service (see https://github.com/endee-io/endee) and set ENDEE_BASE_URL.",
                self.base_url,
                str(exc)
            )
            self._available = False
            self._load_fallback_data()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including optional authentication."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add documents and their embeddings to Endee.

        Falls back to in-memory storage if Endee is unavailable.
        """
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        if self._available:
            # Endee API: POST /api/v1/index/{index_name}/vectors
            url = f"{self.base_url}/api/v1/index/{self.index_name}/vectors"
            payload = {
                "vectors": []
            }
            for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
                payload["vectors"].append(
                    {
                        "id": doc.get("id") or f"doc-{idx}",
                        "vector": emb.tolist(),
                        "metadata": doc,
                    }
                )

            try:
                resp = requests.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers(),
                    timeout=self.timeout
                )
                resp.raise_for_status()
                logger.info("Added %s documents to Endee index '%s'", len(documents), self.index_name)
                return
            except Exception as exc:
                logger.warning("Failed to add documents to Endee, using fallback storage: %s", exc)
                self._available = False
        
        # Fallback: store in memory
        for doc, emb in zip(documents, embeddings):
            self._fallback_data.append({
                "document": doc,
                "embedding": emb,
                "id": doc.get("id") or f"doc-{len(self._fallback_data)}"
            })
        logger.info("Stored %s documents in fallback in-memory storage", len(documents))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search in Endee and return top_k matches.

        Falls back to in-memory search if Endee is unavailable.
        """
        if self._available:
            # Endee API: POST /api/v1/index/{index_name}/search
            url = f"{self.base_url}/api/v1/index/{self.index_name}/search"
            payload = {
                "vector": query_vector.tolist(),
                "top_k": top_k,
            }
            try:
                resp = requests.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers(),
                    timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                # Endee returns results directly or in a 'results' field
                results = data.get("results", data.get("data", []))
                logger.info(
                    "Endee search returned %s results for index '%s'",
                    len(results),
                    self.index_name,
                )
                # Normalise format for upstream tools
                normalised: List[Dict[str, Any]] = []
                for item in results:
                    normalised.append(
                        {
                            "document": item.get("metadata", {}),
                            "score": float(item.get("score", item.get("distance", 0.0))),
                            "id": item.get("id"),
                        }
                    )
                return normalised
            except Exception as exc:
                logger.warning("Endee search failed, falling back to in-memory search: %s", exc)
                return self._fallback_search(query_vector, top_k)
        else:
            return self._fallback_search(query_vector, top_k)

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _ensure_index(self) -> None:
        """
        Ensure that the target index exists in Endee.
        
        Based on Endee API (https://github.com/endee-io/endee):
        - GET /api/v1/index/list to check existing indexes
        - POST /api/v1/index/create to create a new index
        """
        # First, check if index exists by listing all indexes
        list_url = f"{self.base_url}/api/v1/index/list"
        try:
            resp = requests.get(
                list_url, 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            if resp.status_code == 200:
                indexes = resp.json().get("indexes", []) or resp.json().get("data", [])
                index_names = [idx.get("name") if isinstance(idx, dict) else idx for idx in indexes]
                if self.index_name in index_names:
                    logger.info("Endee index '%s' already exists", self.index_name)
                    return
        except Exception:
            # If list fails, try to create anyway
            logger.info("Could not list Endee indexes, attempting to create '%s'", self.index_name)

        # Create the index if it doesn't exist
        # Endee API: POST /api/v1/index/create
        create_url = f"{self.base_url}/api/v1/index/create"
        payload = {
            "name": self.index_name,
            "dimension": 384,  # matches all-MiniLM-L6-v2 by default
            "space_type": "cosine",  # cosine similarity
        }
        try:
            resp = requests.post(
                create_url, 
                json=payload, 
                headers=self._get_headers(),
                timeout=self.timeout
            )
            resp.raise_for_status()
            logger.info("Created Endee index '%s'", self.index_name)
        except Exception as exc:
            logger.exception("Failed to create Endee index '%s' at %s", self.index_name, create_url)
            raise
    
    def _load_fallback_data(self) -> None:
        """Load travel data from JSON file for fallback mode."""
        try:
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "travel_data.json"
            )
            if os.path.exists(data_path):
                with open(data_path, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                for doc in docs:
                    self._fallback_data.append({
                        "document": doc,
                        "embedding": None,  # Will compute on-demand if needed
                        "id": doc.get("id") or f"doc-{len(self._fallback_data)}"
                    })
                logger.info("Loaded %s documents for fallback search", len(self._fallback_data))
        except Exception as exc:
            logger.warning("Failed to load fallback data: %s", exc)
    
    def _fallback_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        RAG-enabled fallback search using cosine similarity on embeddings.
        
        When Endee is unavailable, we still perform proper semantic search:
        1. Compute embeddings for documents that don't have them (lazy loading)
        2. Normalize query and document vectors
        3. Compute cosine similarity (dot product of normalized vectors)
        4. Return top_k most similar documents
        
        This ensures RAG quality is maintained even in fallback mode. The only
        limitation is that it's in-memory, so it won't scale to millions of documents
        like Endee would, but it's perfect for demos and small datasets.
        """
        if not self._fallback_data:
            logger.warning("No fallback data available for search")
            return []
        
        # Lazy-load embeddings: compute them on-demand if not already cached
        # This saves memory and computation time
        from retriever.embedder import Embedder
        embedder = Embedder()
        
        doc_vectors = []
        for item in self._fallback_data:
            doc = item.get("document", {})
            embedding = item.get("embedding")
            
            if embedding is None:
                # Generate embedding from document text
                # We combine title and description for richer semantic representation
                text = f"{doc.get('title', '')} {doc.get('description', '')}"
                embedding = embedder.embed_text(text)
                item["embedding"] = embedding  # Cache for future searches
            
            doc_vectors.append(embedding)
        
        # Compute cosine similarity: normalize vectors and take dot product
        # Cosine similarity ranges from -1 to 1, where 1 means identical
        import numpy as np
        doc_vectors_array = np.array(doc_vectors)
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)  # Add small epsilon to avoid division by zero
        doc_norms = doc_vectors_array / (np.linalg.norm(doc_vectors_array, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top_k most similar documents (highest similarity scores)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results to match Endee's response format
        results = []
        for idx in top_indices:
            item = self._fallback_data[idx]
            results.append({
                "document": item.get("document", {}),
                "score": float(similarities[idx]),  # Convert numpy float to Python float
                "id": item.get("id"),
            })
        
        logger.info("RAG fallback search returned %s results (semantic similarity)", len(results))
        return results
    
    def is_available(self) -> bool:
        """Check if Endee service is available."""
        return self._available

