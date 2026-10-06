"""
Backward compatibility wrapper for FAISS vector store.
"""
from .vector_store import VectorStore, get_or_build_vectorstore

__all__ = ["VectorStore", "get_or_build_vectorstore"]
