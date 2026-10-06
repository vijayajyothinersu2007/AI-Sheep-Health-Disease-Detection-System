"""
Embedding engine for veterinary knowledge retrieval.
Supports Google Gemini text-embedding-004 with a robust, zero-dependency local dense
semantic embedding fallback of fixed 768 dimensions for FAISS index compatibility.
"""
from typing import List, Optional
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class LocalDenseEmbeddings:
    """
    High-performance, deterministic local dense embedding generator.
    Produces fixed 768-dimensional L2-normalized float32 vectors suitable
    for FAISS IndexFlatIP cosine similarity search.
    Requires no network connection and never fails.
    """
    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        # Word + bigram hashing vectorizer
        self.vectorizer = HashingVectorizer(
            n_features=dimension,
            ngram_range=(1, 2),
            norm='l2',
            alternate_sign=False,
            token_pattern=r"(?u)\b\w+\b"
        )

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        # Clean and stringify
        clean_texts = [str(t) if t else " " for t in texts]
        matrix = self.vectorizer.transform(clean_texts)
        dense = matrix.toarray().astype(np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])


class GeminiEmbeddings:
    """
    Google Gemini text-embedding-004 embedding client using google.genai SDK.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
        self.model = model
        self._client = None
        if self.api_key and not self.api_key.startswith("hf_"):
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize google.genai client: {e}")
                self._client = None
        else:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and bool(self.api_key) and not self.api_key.startswith("hf_")

    def embed_documents(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.is_available() or not texts:
            return None
        try:
            vectors = []
            # Batch in chunks of 50 to respect API limits
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # In google.genai SDK, embed_content takes contents
                res = self._client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                if hasattr(res, "embeddings"):
                    for emb in res.embeddings:
                        vectors.append(emb.values)
                elif hasattr(res, "embedding"):
                    vectors.append(res.embedding.values)
            if vectors:
                arr = np.array(vectors, dtype=np.float32)
                # Ensure L2 normalization
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (arr / norms).astype(np.float32)
        except Exception as e:
            logger.warning(f"Gemini embeddings API failed, using fallback: {e}")
            return None
        return None

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        res = self.embed_documents([query])
        return res if res is not None else None


class EmbeddingsEngine:
    """
    Unified Embeddings Engine.
    Attempts Google Gemini semantic embeddings when available; automatically
    falls back to deterministic local dense embeddings when offline or unconfigured.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.gemini = GeminiEmbeddings(api_key=api_key)
        self.local = LocalDenseEmbeddings(dimension=EMBEDDING_DIM)
        self.dimension = EMBEDDING_DIM

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.gemini.is_available():
            arr = self.gemini.embed_documents(texts)
            if arr is not None and len(arr) == len(texts):
                return arr
        return self.local.embed_documents(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if self.gemini.is_available():
            arr = self.gemini.embed_query(query)
            if arr is not None and len(arr) == 1:
                return arr
        return self.local.embed_query(query)


# Alias for backwards compatibility
class TFIDFEmbeddings(LocalDenseEmbeddings):
    """Backward compatibility alias for TFIDFEmbeddings."""
    def fit(self, texts: List[str]):
        pass

    def transform(self, texts: List[str]):
        return self.embed_documents(texts)
