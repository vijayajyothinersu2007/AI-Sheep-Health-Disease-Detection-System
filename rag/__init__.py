"""
RAG Package for AI Sheep Health & Disease Detection System.
Provides end-to-end semantic retrieval with FAISS and Gemini LLM grounded generation.
"""
from .documents import Document
from .document_loader import load_knowledge_base, parse_metadata_section
from .text_chunker import chunk_documents
from .embeddings import (
    EmbeddingsEngine,
    GeminiEmbeddings,
    LocalDenseEmbeddings,
    TFIDFEmbeddings
)
from .vector_store import VectorStore, get_or_build_vectorstore
from .retriever import SheepHealthRetriever
from .llm import LLMClient
from .rag_pipeline import ClinicalChatResponse, ClinicalReport, RAGPipeline
from .response_formatter import (
    format_retrieved_sources_html,
    format_retrieved_sources_markdown,
    sanitize_sources_to_markdown,
    render_chat_message_html,
    format_point_by_point_markdown
)
from .ingest import run_ingestion

__all__ = [
    "Document",
    "load_knowledge_base",
    "parse_metadata_section",
    "chunk_documents",
    "EmbeddingsEngine",
    "GeminiEmbeddings",
    "LocalDenseEmbeddings",
    "TFIDFEmbeddings",
    "VectorStore",
    "get_or_build_vectorstore",
    "SheepHealthRetriever",
    "LLMClient",
    "ClinicalChatResponse",
    "ClinicalReport",
    "RAGPipeline",
    "format_retrieved_sources_html",
    "format_retrieved_sources_markdown",
    "sanitize_sources_to_markdown",
    "render_chat_message_html",
    "format_point_by_point_markdown",
    "run_ingestion"
]
