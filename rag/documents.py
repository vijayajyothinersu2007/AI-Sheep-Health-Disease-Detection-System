"""
Document schema for Knowledge Base chunks and metadata.
"""
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    """Represents a chunk of veterinary knowledge with associated metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        disease = self.metadata.get("disease", "Unknown")
        topic = self.metadata.get("topic", "General")
        preview = self.page_content[:60].replace("\n", " ")
        return f"<Document disease='{disease}' topic='{topic}': '{preview}...'>"
