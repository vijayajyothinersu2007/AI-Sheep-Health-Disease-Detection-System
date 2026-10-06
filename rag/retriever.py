"""
Retriever module: Builds context-aware queries from image classification metadata
and fetches grounded veterinary documents from the FAISS VectorStore.
"""
from typing import List, Tuple, Optional, Dict, Any
from .documents import Document
from .vector_store import VectorStore, get_or_build_vectorstore
from .config import DEFAULT_TOP_K

class SheepHealthRetriever:
    """
    Intelligent retriever that bridges image classification output to the FAISS knowledge base.
    """
    def __init__(self, vectorstore: Optional[VectorStore] = None):
        self.vectorstore = vectorstore or get_or_build_vectorstore()

    def build_contextual_query(
        self,
        disease: str,
        question: Optional[str] = None,
        top_candidates: Optional[List[str]] = None
    ) -> str:
        """
        Builds a comprehensive retrieval query using predicted condition and query aspects.
        """
        if question and question.strip():
            # If user asked a specific question, enrich it with disease context
            q_clean = question.strip()
            return f"{disease} sheep health {q_clean}. Symptoms, treatment, medicines, home care, prevention, warning signs."
        else:
            # Default prediction analysis query covering all required report facets
            candidates_text = ""
            if top_candidates:
                candidates_text = " Differential candidates: " + ", ".join(top_candidates)
                
            return (
                f"Sheep disease: {disease}.{candidates_text} "
                f"Need clinical information about: overview, causes, transmission, "
                f"symptoms, veterinary treatment, medicines, safe supportive home care, "
                f"home remedies evidence, prevention, vaccination schedule, feeding, "
                f"hydration, emergency warning signs, when to contact veterinarian."
            )

    def retrieve_with_scores(
        self,
        query: str = "",
        prediction_context: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.05
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k documents along with their similarity scores.
        """
        boost_disease = None
        if prediction_context:
            raw_disease = prediction_context.get("raw_class") or prediction_context.get("disease", "")
            short_disease = prediction_context.get("short_name") or raw_disease
            boost_disease = short_disease
            
            top_k_diseases = []
            for item in prediction_context.get("top_k", []):
                d_name = item.get("short_name") or item.get("raw_class") or item.get("disease")
                if d_name and d_name != short_disease:
                    top_k_diseases.append(d_name)
                    
            search_query = self.build_contextual_query(
                disease=short_disease,
                question=query if query.strip() else None,
                top_candidates=top_k_diseases[:2]
            )
        else:
            search_query = query if query.strip() else "General sheep health, symptoms, nutrition, vaccination, diseases"
            
        results = self.vectorstore.similarity_search_with_scores(
            query=search_query,
            k=top_k,
            boost_disease=boost_disease
        )
        return [(doc, score) for doc, score in results if score >= min_score]

    def retrieve_question_aware(
        self,
        question: str,
        intent: str,
        disease: str,
        prediction_context: Optional[Dict[str, Any]] = None,
        top_k: int = 4
    ) -> List[Document]:
        """
        Retrieves knowledge chunks with question-aware query expansion,
        intent-specific section reranking, cross-disease filtering, and deduplication.
        """
        from .intent_detector import IntentType, get_intent_keywords

        # 1. Expand query specifically for the question and intent
        intent_kws = get_intent_keywords(intent)
        expanded_query = f"{disease} {question} {intent_kws}"

        # 2. Retrieve candidate chunks pool (larger pool to filter/rerank)
        pool_size = max(top_k * 3, 10)
        boost_disease = disease if disease and disease != "Sheep Health" else None
        raw_results = self.vectorstore.similarity_search_with_scores(
            query=expanded_query,
            k=pool_size,
            boost_disease=boost_disease
        )

        # 3. Intent section keywords for reranking
        intent_section_keywords = {
            IntentType.HOME_CARE: ["home care", "supportive care", "first aid", "comfort", "wound management", "nursing"],
            IntentType.MEDICINE_TREATMENT: ["treatment", "medicine", "allopathic", "medical", "prescription", "antibiotic"],
            IntentType.SYMPTOMS: ["symptom", "signs", "clinical", "manifestation", "lesion"],
            IntentType.CAUSES: ["cause", "etiology", "underlying", "transmission"],
            IntentType.PREVENTION: ["prevention", "biosecurity", "sanitation", "quarantine"],
            IntentType.VACCINATION: ["vaccin", "schedule", "calendar", "immuniz"],
            IntentType.FEEDING_NUTRITION: ["nutrition", "feeding", "diet", "drought", "acidosis", "bloat", "fodder"],
            IntentType.TRANSMISSION: ["transmission", "spread", "contagious", "vector"],
            IntentType.SEVERITY_EMERGENCY: ["emergency", "warning", "critical", "when to call", "triage"],
            IntentType.DISEASE_OVERVIEW: ["overview", "description", "disease overview"],
        }
        target_keywords = intent_section_keywords.get(intent, [])

        filtered_results = []
        seen_chunks = set()
        target_disease_clean = disease.lower().strip() if disease and disease != "Sheep Health" else ""

        for doc, score in raw_results:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            sec = str(doc.metadata.get("section", "")).lower()
            doc_disease = str(doc.metadata.get("disease", "")).lower()
            title = str(doc.metadata.get("title", "")).lower()
            content = doc.page_content.lower()

            adjusted_score = score

            # Intent alignment boost on section header and content
            if any(kw in sec for kw in target_keywords):
                adjusted_score += 0.45
            elif any(kw in content for kw in target_keywords):
                adjusted_score += 0.15

            # Disease relevance boost and cross-disease penalty
            if target_disease_clean:
                if target_disease_clean in doc_disease or target_disease_clean in title:
                    adjusted_score += 0.35
                elif doc_disease and doc_disease not in ["general", "sheep health", "all"]:
                    # Chunk belongs explicitly to a different disease
                    if target_disease_clean not in doc_disease and target_disease_clean not in content:
                        adjusted_score -= 0.60

            filtered_results.append((doc, adjusted_score))

        # Re-sort results by adjusted score descending
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in filtered_results[:top_k]]

    def retrieve(
        self,
        query: str = "",
        prediction_context: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Retrieves the most relevant grounded knowledge chunks based on query and prediction context.
        """
        boost_disease = None
        
        if prediction_context:
            raw_disease = prediction_context.get("raw_class") or prediction_context.get("disease", "")
            short_disease = prediction_context.get("short_name") or raw_disease
            boost_disease = short_disease
            
            top_k_diseases = []
            for item in prediction_context.get("top_k", []):
                d_name = item.get("short_name") or item.get("raw_class") or item.get("disease")
                if d_name and d_name != short_disease:
                    top_k_diseases.append(d_name)
                    
            search_query = self.build_contextual_query(
                disease=short_disease,
                question=query if query.strip() else None,
                top_candidates=top_k_diseases[:2]
            )
        else:
            search_query = query if query.strip() else "General sheep health, symptoms, nutrition, vaccination, diseases"
            
        docs = self.vectorstore.similarity_search(
            query=search_query,
            k=top_k,
            boost_disease=boost_disease
        )
        
        # If low confidence and we have a second candidate, ensure we also retrieve candidate 2
        if prediction_context and prediction_context.get("is_uncertain") and len(prediction_context.get("top_k", [])) > 1:
            second_cand = prediction_context["top_k"][1].get("short_name")
            if second_cand:
                extra_docs = self.vectorstore.similarity_search(
                    query=f"{second_cand} symptoms treatment differential diagnosis",
                    k=2,
                    boost_disease=second_cand
                )
                existing_ids = {d.metadata.get("chunk_id") for d in docs}
                for ed in extra_docs:
                    if ed.metadata.get("chunk_id") not in existing_ids:
                        docs.append(ed)
                        existing_ids.add(ed.metadata.get("chunk_id"))
                        if len(docs) >= top_k + 2:
                            break
                            
        return docs

    def format_context_for_prompt(self, documents: List[Document]) -> str:
        """
        Formats retrieved knowledge chunks cleanly with headers and citations for LLM prompt injection.
        """
        if not documents:
            return "No specific veterinary documents retrieved."
            
        formatted_chunks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "ICAR / IVRI Knowledge Base")
            disease = doc.metadata.get("disease", "General")
            section = doc.metadata.get("section", "Clinical Guidance")
            title = doc.metadata.get("title", doc.metadata.get("document_name", "Document"))
            
            chunk_header = f"=== DOCUMENT CHUNK [{i}] | Disease: {disease} | Section: {section} | Source: {source} ({title}) ==="
            formatted_chunks.append(f"{chunk_header}\n{doc.page_content.strip()}")
            
        return "\n\n".join(formatted_chunks)
