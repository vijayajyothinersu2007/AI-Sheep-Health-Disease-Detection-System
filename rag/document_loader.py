"""
Knowledge Base document loader.
Reads markdown and text files from knowledge_base/ and extracts structured metadata.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from .documents import Document
from .config import KNOWLEDGE_BASE_DIR, FALLBACK_KNOWLEDGE_BASE_DIR

def parse_metadata_section(text: str) -> Dict[str, Any]:
    """
    Parses '## Metadata' section from markdown document if present.
    Returns metadata dict.
    """
    metadata: Dict[str, Any] = {}
    meta_pattern = r"##\s*Metadata\s*\n(.*?)(?=\n##|\Z)"
    match = re.search(meta_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        meta_block = match.group(1)
        for line in meta_block.splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
            if ":" in line:
                key, val = line.split(":", 1)
                clean_key = key.replace("**", "").replace("*", "").strip().lower()
                clean_val = val.replace("**", "").replace("*", "").strip()
                metadata[clean_key] = clean_val
                
    return metadata

def load_knowledge_base(directory: Optional[Path] = None) -> List[Document]:
    """
    Loads all .md and .txt files recursively from knowledge base directory.
    Checks primary directory, falling back to legacy Data/knowledge_base if needed.
    """
    if directory is None:
        if KNOWLEDGE_BASE_DIR.exists() and any(KNOWLEDGE_BASE_DIR.rglob("*.md")):
            directory = KNOWLEDGE_BASE_DIR
        elif FALLBACK_KNOWLEDGE_BASE_DIR.exists():
            directory = FALLBACK_KNOWLEDGE_BASE_DIR
        else:
            directory = KNOWLEDGE_BASE_DIR

    documents: List[Document] = []
    
    if not directory.exists():
        return documents
        
    for file_path in directory.rglob("*.*"):
        if file_path.suffix.lower() not in [".md", ".txt"]:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception:
                continue
                
        doc_metadata = parse_metadata_section(content)
        
        # Extract title from first # header if available
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()
        
        doc_metadata.setdefault("document_name", file_path.name)
        doc_metadata.setdefault("title", doc_title)
        doc_metadata.setdefault("category", file_path.parent.name)
        doc_metadata.setdefault("file_path", str(file_path))
        doc_metadata.setdefault("source", "ICAR / IVRI / CSWRI / DAHD Veterinary Knowledge Base")
        doc_metadata.setdefault("source_url", "https://ivri.nic.in")
        
        # Also infer disease from filename or folder if not explicitly in metadata
        if "disease" not in doc_metadata:
            if "sheep_diseases" in str(file_path) or "diseases" in str(file_path):
                doc_metadata["disease"] = file_path.stem.replace("_", " ").title()
            else:
                doc_metadata["disease"] = "General Sheep Health"
                
        documents.append(Document(page_content=content, metadata=doc_metadata))
        
    return documents
