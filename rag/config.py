"""
RAG configuration and constants for AI Sheep Health & Disease Detection System.
Loads environment variables from .env and defines paths, model names, and indexing parameters.
"""
from pathlib import Path
import os
import dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root if it exists
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    dotenv.load_dotenv(dotenv_path)

# Knowledge Base & Vectorstore Paths
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
FALLBACK_KNOWLEDGE_BASE_DIR = BASE_DIR / "Data" / "knowledge_base"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

# Legacy paths for backward compatibility
DATA_DIR = BASE_DIR / "Data"
VECTORSTORE_CACHE_PATH = DATA_DIR / "vectorstore_cache.pkl"

# Retrieval & Chunking Parameters
DEFAULT_TOP_K = 4
MAX_RETRIEVED_CHUNKS = 6
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
EMBEDDING_DIM = 768

# Environment Keys & API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Model Names
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
