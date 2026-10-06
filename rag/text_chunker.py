"""
Header-aware and semantic text chunker for veterinary documents.
Splits markdown documents into coherent knowledge chunks while preserving headers and metadata.
"""
from typing import List, Dict, Any
import re
from .documents import Document
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits documents into coherent knowledge chunks based on markdown section headers (##, ###)
    and paragraph boundaries, preserving metadata and adding breadcrumbs.
    """
    chunked_docs: List[Document] = []
    
    for doc in documents:
        raw_text = doc.page_content
        base_meta = doc.metadata.copy()
        
        # Split document by markdown section headers (## or ###)
        sections = re.split(r"\n(?=#{2,3}\s+)", raw_text)
        
        for sec_idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
                
            # Skip standalone Metadata section as it's already captured in metadata
            if section.lower().startswith("## metadata"):
                continue
                
            # Extract section heading if present
            header_match = re.match(r"^(#{2,3}\s+([^\n]+))", section)
            sec_header = ""
            if header_match:
                sec_header = header_match.group(2).strip()
            
            # If section is small enough, keep as single chunk
            if len(section) <= chunk_size + 150:
                meta = base_meta.copy()
                meta["section"] = sec_header or base_meta.get("title", "General")
                meta["chunk_id"] = f"{base_meta.get('document_name', 'doc')}_{sec_idx}"
                chunked_docs.append(Document(page_content=section, metadata=meta))
            else:
                # Sub-split long section by paragraphs
                paragraphs = section.split("\n\n")
                current_chunk = ""
                sub_idx = 0
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                        
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n" + para
                        else:
                            current_chunk = para
                    else:
                        if current_chunk:
                            meta = base_meta.copy()
                            meta["section"] = sec_header or base_meta.get("title", "General")
                            meta["chunk_id"] = f"{base_meta.get('document_name', 'doc')}_{sec_idx}_{sub_idx}"
                            
                            # Prepend section header if not already present
                            full_text = current_chunk
                            if sec_header and not full_text.startswith("#"):
                                full_text = f"### {sec_header}\n" + full_text
                                
                            chunked_docs.append(Document(page_content=full_text, metadata=meta))
                            sub_idx += 1
                        
                        current_chunk = para
                
                if current_chunk:
                    meta = base_meta.copy()
                    meta["section"] = sec_header or base_meta.get("title", "General")
                    meta["chunk_id"] = f"{base_meta.get('document_name', 'doc')}_{sec_idx}_{sub_idx}"
                    full_text = current_chunk
                    if sec_header and not full_text.startswith("#"):
                        full_text = f"### {sec_header}\n" + full_text
                    chunked_docs.append(Document(page_content=full_text, metadata=meta))
                    
    return chunked_docs
