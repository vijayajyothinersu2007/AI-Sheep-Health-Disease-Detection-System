"""
FAISS Vector Database for Veterinary Knowledge Base.
Persists FAISS Index and chunk metadata to vectorstore/index.faiss and vectorstore/metadata.pkl.
Supports fast cosine similarity retrieval with disease metadata boosting.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle
import numpy as np
import faiss

from .documents import Document
from .embeddings import EmbeddingsEngine
from .document_loader import load_knowledge_base
from .text_chunker import chunk_documents
from .config import (
    KNOWLEDGE_BASE_DIR,
    VECTORSTORE_DIR,
    INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIM
)


class VectorStore:
    """
    FAISS-based vector database with local disk persistence.
    Uses normalized dense vectors with faiss.IndexFlatIP for exact cosine similarity search.
    """
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        embeddings: Optional[EmbeddingsEngine] = None,
        index: Optional[faiss.Index] = None,
        dimension: int = EMBEDDING_DIM
    ):
        self.dimension = dimension
        self.embeddings = embeddings or EmbeddingsEngine()
        self.documents: List[Document] = documents or []
        self.index: faiss.Index = index if index is not None else faiss.IndexFlatIP(self.dimension)
        
        # If documents are provided and index is empty, populate index
        if self.documents and self.index.ntotal == 0:
            self._index_documents()

    def _index_documents(self):
        """Generates embeddings and populates the FAISS index."""
        texts = [doc.page_content for doc in self.documents]
        vectors = self.embeddings.embed_documents(texts)
        if len(vectors) > 0:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)

    def save(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        """Persists the FAISS index and documents metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write FAISS index file
        faiss.write_index(self.index, str(index_path))
        
        # Write metadata pickle file
        meta_payload = {
            "documents": self.documents,
            "dimension": self.dimension,
            "total_chunks": len(self.documents)
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(meta_payload, f)

    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        embeddings: Optional[EmbeddingsEngine] = None
    ) -> Optional["VectorStore"]:
        """Loads a persisted FAISS vectorstore from disk if files exist."""
        if not index_path.exists() or not metadata_path.exists():
            return None
        try:
            loaded_index = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
                meta_payload = pickle.load(f)
            
            docs = meta_payload.get("documents", [])
            dim = meta_payload.get("dimension", EMBEDDING_DIM)
            
            store = cls(
                documents=docs,
                embeddings=embeddings or EmbeddingsEngine(),
                index=loaded_index,
                dimension=dim
            )
            return store
        except Exception:
            return None

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
        boost_disease: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Performs semantic retrieval against FAISS index.
        Applies a metadata boost when chunk disease matches the target disease context.
        """
        if not self.documents or self.index.ntotal == 0:
            return []

        # Generate normalized query vector
        q_vec = self.embeddings.embed_query(query)
        faiss.normalize_L2(q_vec)
        
        # Search a wider pool (top 3*k) to apply metadata re-ranking/boosting
        search_k = min(self.index.ntotal, max(k * 3, 10))
        distances, indices = self.index.search(q_vec, search_k)
        
        results: List[Tuple[Document, float]] = []
        seen_chunks = set()
        
        boost_lower = boost_disease.lower() if boost_disease else ""
        
        for idx_pos, doc_idx in enumerate(indices[0]):
            if doc_idx < 0 or doc_idx >= len(self.documents):
                continue
                
            raw_score = float(distances[0][idx_pos])
            doc = self.documents[doc_idx]
            
            chunk_id = doc.metadata.get("chunk_id", str(doc_idx))
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            
            # Metadata relevance boost for target disease
            final_score = raw_score
            if boost_lower:
                doc_disease = str(doc.metadata.get("disease", "")).lower()
                doc_title = str(doc.metadata.get("title", "")).lower()
                doc_section = str(doc.metadata.get("section", "")).lower()
                
                if boost_lower in doc_disease or boost_lower in doc_title:
                    final_score += 0.35
                elif boost_lower in doc_section:
                    final_score += 0.15
                    
            results.append((doc, final_score))
            
        # Re-sort results by final score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        boost_disease: Optional[str] = None
    ) -> List[Document]:
        """Returns the top-k most relevant documents."""
        results = self.similarity_search_with_scores(query, k=k, boost_disease=boost_disease)
        return [doc for doc, _ in results]


def get_or_build_vectorstore(force_rebuild: bool = False) -> VectorStore:
    """
    Loads FAISS vectorstore from disk if available;
    otherwise ingests documents from knowledge_base, chunks, creates FAISS index, and persists.
    """
    if not force_rebuild and INDEX_PATH.exists() and METADATA_PATH.exists():
        loaded = VectorStore.load(INDEX_PATH, METADATA_PATH)
        if loaded is not None and len(loaded.documents) > 0:
            return loaded

    # Build fresh FAISS vectorstore
    raw_docs = load_knowledge_base()
    chunked_docs = chunk_documents(raw_docs)
    vectorstore = VectorStore(documents=chunked_docs)
    vectorstore.save(INDEX_PATH, METADATA_PATH)
    return vectorstore
