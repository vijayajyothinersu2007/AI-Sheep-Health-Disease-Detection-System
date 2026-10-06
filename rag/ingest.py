"""
Offline ingestion script for Sheep AI Knowledge Base.
Loads all markdown documents from knowledge_base/, chunks them, generates embeddings,
builds the FAISS vector index, and persists to vectorstore/index.faiss and vectorstore/metadata.pkl.

Usage:
    python -m rag.ingest
"""
import sys
import time
from pathlib import Path

from .config import (
    KNOWLEDGE_BASE_DIR,
    VECTORSTORE_DIR,
    INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_DIM
)
from .document_loader import load_knowledge_base
from .text_chunker import chunk_documents
from .embeddings import EmbeddingsEngine
from .vector_store import VectorStore


def run_ingestion() -> VectorStore:
    print("=" * 65)
    print("  SHEEP HEALTH RAG KNOWLEDGE BASE — OFFLINE INGESTION PIPELINE")
    print("=" * 65)
    start_time = time.time()

    # 1. Scan and Load Knowledge Base Documents
    print(f"\n[1/5] Scanning knowledge base directory: {KNOWLEDGE_BASE_DIR}")
    raw_docs = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    if not raw_docs:
        print("ERROR: No markdown documents found in knowledge_base directory.")
        sys.exit(1)
    print(f"      Loaded {len(raw_docs)} documents.")

    # 2. Chunk Documents
    print(f"\n[2/5] Chunking documents into semantic sections...")
    chunks = chunk_documents(raw_docs)
    print(f"      Created {len(chunks)} chunks.")

    # 3. Initialize Embeddings Engine
    print(f"\n[3/5] Initializing embeddings engine (dim={EMBEDDING_DIM})...")
    embeddings = EmbeddingsEngine()

    # 4. Build FAISS Vector Index
    print(f"\n[4/5] Building FAISS IndexFlatIP and embedding all chunks...")
    vectorstore = VectorStore(documents=chunks, embeddings=embeddings, dimension=EMBEDDING_DIM)
    print(f"      Indexed {vectorstore.index.ntotal} vectors in FAISS index.")

    # 5. Persist to Disk
    print(f"\n[5/5] Persisting vectorstore to disk...")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save(INDEX_PATH, METADATA_PATH)
    print(f"      Saved FAISS index to: {INDEX_PATH}")
    print(f"      Saved metadata to:    {METADATA_PATH}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("  INGESTION SUMMARY")
    print("=" * 65)
    print(f"  • Documents Loaded:      {len(raw_docs)}")
    print(f"  • Chunks Created:        {len(chunks)}")
    print(f"  • Embeddings Dimension:  {EMBEDDING_DIM}")
    print(f"  • Total Vectors Indexed: {vectorstore.index.ntotal}")
    print(f"  • FAISS Index File:      {INDEX_PATH}")
    print(f"  • Metadata File:         {METADATA_PATH}")
    print(f"  • Time Taken:            {elapsed:.2f} seconds")
    print("=" * 65 + "\n")

    return vectorstore


if __name__ == "__main__":
    run_ingestion()
