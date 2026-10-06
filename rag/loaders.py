"""
Backward compatibility wrapper for document loader.
"""
from .document_loader import parse_metadata_section, load_knowledge_base

__all__ = ["parse_metadata_section", "load_knowledge_base"]
