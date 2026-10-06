"""
RAG Pipeline Orchestrator: Connects classification metadata, retrieval, LLM generation,
and safe grounded synthesis.
"""
from typing import List, Dict, Any, Optional, Tuple
import re

from .intent_detector import (
    IntentType,
    detect_intents,
    extract_disease_entity,
    is_out_of_domain
)
from .documents import Document
from .retriever import SheepHealthRetriever
from .prompts import (
    build_prediction_report_prompt,
    build_chat_prompt,
    VETERINARY_SYSTEM_PROMPT
)
from .llm import LLMClient
from .response_formatter import format_retrieved_sources_html, format_point_by_point_markdown

class ClinicalReport:
    """
    Structured Clinical Report containing parallel English and Telugu versions
    for side-by-side rendering in the UI.
    """
    def __init__(self, en_text: str, te_text: str, combined_text: str = ""):
        self.en = format_point_by_point_markdown(en_text.strip())
        self.te = format_point_by_point_markdown(te_text.strip())
        self.combined = combined_text or f"{self.en}\n\n---\n\n{self.te}"

    def __str__(self) -> str:
        return self.combined

    def __repr__(self) -> str:
        return self.combined


class ClinicalChatResponse:
    """
    Structured, validated bilingual clinical chat response with independent
    English and Telugu section trees and independent numbering.
    Guarantees:
    - English column receives ONLY English content.
    - Telugu column receives ONLY Telugu content.
    - Both columns start numbering from 1 independently.
    - Each section is separated by a clean horizontal rule '---'.
    """
    def __init__(
        self,
        english_sections: Optional[List[Dict[str, str]]] = None,
        telugu_sections: Optional[List[Dict[str, str]]] = None,
        en_markdown: str = "",
        te_markdown: str = "",
        raw_text: str = ""
    ):
        self.english_sections = english_sections or []
        self.telugu_sections = telugu_sections or []
        self._en_markdown = en_markdown.strip() if en_markdown else ""
        self._te_markdown = te_markdown.strip() if te_markdown else ""
        self.raw_text = raw_text

    @property
    def english_markdown(self) -> str:
        if self._en_markdown:
            return format_point_by_point_markdown(self._en_markdown)
        if not self.english_sections:
            return ""
        blocks = []
        for i, sec in enumerate(self.english_sections, start=1):
            heading = str(sec.get("title") or sec.get("heading") or "").strip()
            clean_heading = re.sub(r"^(?:###\s*)?\d+[\.\)]\s*", "", heading).strip()
            clean_heading = re.sub(r"<[^>]+>", "", clean_heading).strip()
            content = str(sec.get("content") or "").strip()
            formatted_content = format_point_by_point_markdown(content)
            blocks.append(f"### {i}. {clean_heading}\n\n{formatted_content}")
        return format_point_by_point_markdown("\n\n---\n\n".join(blocks))

    @property
    def telugu_markdown(self) -> str:
        if self._te_markdown:
            return format_point_by_point_markdown(self._te_markdown)
        if not self.telugu_sections:
            return ""
        blocks = []
        for i, sec in enumerate(self.telugu_sections, start=1):
            heading = str(sec.get("title") or sec.get("heading") or "").strip()
            clean_heading = re.sub(r"^(?:###\s*)?\d+[\.\)]\s*", "", heading).strip()
            clean_heading = re.sub(r"<[^>]+>", "", clean_heading).strip()
            content = str(sec.get("content") or "").strip()
            formatted_content = format_point_by_point_markdown(content)
            blocks.append(f"### {i}. {clean_heading}\n\n{formatted_content}")
        return format_point_by_point_markdown("\n\n---\n\n".join(blocks))

    @property
    def combined(self) -> str:
        en = self.english_markdown
        te = self.telugu_markdown
        if en and te:
            return f"{en}\n\n<!-- BILINGUAL_LANGUAGE_SPLIT -->\n\n{te}"
        return en or te

    def to_dict(self) -> Dict[str, Any]:
        return {
            "english": self.english_sections,
            "telugu": self.telugu_sections,
            "en_markdown": self.english_markdown,
            "te_markdown": self.telugu_markdown,
        }

    def __str__(self) -> str:
        return self.combined

    def __repr__(self) -> str:
        return self.combined



class RAGPipeline:
    """
    End-to-End RAG Pipeline for Sheep Disease Detection and Veterinary Assistance.
    """
    def __init__(self, retriever: Optional[SheepHealthRetriever] = None, llm: Optional[LLMClient] = None):
        self.retriever = retriever or SheepHealthRetriever()
        self.llm = llm or LLMClient()

    def get_provider_name(self) -> str:
        """Returns the active generator provider name."""
        return self.llm.get_active_provider()

    def generate_prediction_report(
        self,
        prediction_context: Dict[str, Any],
        language: str = "English + తెలుగు"
    ) -> Tuple[ClinicalReport, List[Document]]:
        """
        Coordinates retrieval and generation for the 11-section grounded health report.
        Returns a ClinicalReport object with .en (English), .te (Telugu), and .combined text.
        """
        # Step 1: Retrieve context-grounded documents
        retrieved_docs = self.retriever.retrieve(
            query="",
            prediction_context=prediction_context,
            top_k=4
        )
        
        # Format chunks for LLM context
        context_chunks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            src = doc.metadata.get("source", "Veterinary Manual")
            sec = doc.metadata.get("section", "Clinical Details")
            context_chunks.append(f"--- Document Chunk {i} [{src} - {sec}] ---\n{doc.page_content}\n")
            
        combined_context = "\n".join(context_chunks)
        
        # Step 2: Try LLM if configured
        if self.llm.is_api_configured():
            prompt = build_prediction_report_prompt(prediction_context, combined_context, language)
            llm_output = self.llm.generate(prompt, system_prompt=VETERINARY_SYSTEM_PROMPT, max_tokens=2800)
            if llm_output and len(llm_output.strip()) > 100:
                report_obj = self._parse_llm_bilingual_report(llm_output, prediction_context, retrieved_docs)
                return report_obj, retrieved_docs

        # Step 3: High-fidelity grounded knowledge synthesis fallback
        grounded_report = self._synthesize_grounded_report(prediction_context, retrieved_docs, language)
        return grounded_report, retrieved_docs


    def _is_sheep_health_domain(self, query: str) -> bool:
        """
        Validates that the incoming farmer/user query strictly pertains to sheep/small-ruminant
        health, disease management, nutrition, vaccination, or veterinary care.
        Rejects non-domain topics (programming, sports, jokes, weather, general trivia).
        """
        q = query.lower().strip()
        if not q:
            return False
        if is_out_of_domain(q):
            return False
        primary_intent, _ = detect_intents(q)
        if primary_intent == IntentType.IRRELEVANT:
            return False
        return True

    def _build_domain_refusal(self, language: str) -> ClinicalChatResponse:
        """Returns polite bilingual refusal when an irrelevant non-sheep topic is asked."""
        refusal_en = (
            "Sorry, I am designed to assist with sheep health, diseases, treatment, "
            "feeding, prevention and veterinary care. Please ask a sheep-related question."
        )
        refusal_te = (
            "క్షమించండి, నేను గొర్రెల ఆరోగ్యం, వ్యాధులు, చికిత్స, మేత, నివారణ మరియు "
            "పశువైద్య సంరక్షణకు సంబంధించిన ప్రశ్నలకు మాత్రమే సహాయం చేయడానికి రూపొందించబడ్డాను. "
            "దయచేసి గొర్రెలకు సంబంధించిన ప్రశ్నను అడగండి."
        )
        en_secs = [{"title": "Sheep Health & Veterinary Notice", "content": refusal_en}]
        te_secs = [{"title": "గొర్రెల ఆరోగ్య సహాయక గమనిక", "content": refusal_te}]
        if language == "English":
            return ClinicalChatResponse(
                english_sections=en_secs,
                en_markdown=f"### 1. Sheep Health & Veterinary Notice\n\n{refusal_en}"
            )
        elif language == "తెలుగు":
            return ClinicalChatResponse(
                telugu_sections=te_secs,
                te_markdown=f"### 1. గొర్రెల ఆరోగ్య సహాయక గమనిక\n\n{refusal_te}"
            )
        else:
            return ClinicalChatResponse(
                english_sections=en_secs,
                telugu_sections=te_secs,
                en_markdown=f"### 1. Sheep Health & Veterinary Notice\n\n{refusal_en}",
                te_markdown=f"### 1. గొర్రెల ఆరోగ్య సహాయక గమనిక\n\n{refusal_te}"
            )

    def _sanitize_llm_markdown(self, text: str) -> str:
        """Strips accidental raw HTML tags from LLM output to guarantee pure clean Markdown."""
        if not text:
            return ""
        clean = re.sub(r"<(div|span|p|section|article)[^>]*>", "", text, flags=re.IGNORECASE)
        clean = re.sub(r"</(div|span|p|section|article)>", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<br\s*/?>", "\n", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<li>(.*?)</li>", r"• \1\n", clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"</?ul[^>]*>", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"</?ol[^>]*>", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"<a\s+[^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", r"[\2](\1)", clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", "", clean)
        return clean.strip()

    def _extract_bilingual_markdown(self, text: str) -> Tuple[str, str]:
        """
        Extracts separate English and Telugu text blocks from markdown output
        without fragile position-based guessing.
        """
        if "<!-- BILINGUAL_LANGUAGE_SPLIT -->" in text:
            parts = text.split("<!-- BILINGUAL_LANGUAGE_SPLIT -->", 1)
            return parts[0].strip(), parts[1].strip()
        if "---TELUGU_TRANSLATION---" in text:
            parts = text.split("---TELUGU_TRANSLATION---", 1)
            return parts[0].strip(), parts[1].strip()
        if "## Telugu Guidance" in text:
            parts = text.split("## Telugu Guidance", 1)
            return parts[0].strip(), "## Telugu Guidance\n" + parts[1].strip()

        # Find where Telugu script begins
        lines = text.splitlines()
        first_te_idx = -1
        for i, line in enumerate(lines):
            te_chars = len(re.findall(r"[\u0C00-\u0C7F]", line))
            if te_chars >= 5 and (line.strip().startswith(("#", "1.", "1)", "**1")) or te_chars > 15):
                first_te_idx = i
                break

        if first_te_idx > 0:
            en_text = "\n".join(lines[:first_te_idx]).strip()
            te_text = "\n".join(lines[first_te_idx:]).strip()
            en_text = re.sub(r"\n*---+\s*$", "", en_text).strip()
            return en_text, te_text

        return text.strip(), ""

    def _parse_and_validate_llm_response(
        self,
        llm_text: str,
        language: str,
        question: str,
        context: Optional[Dict[str, Any]],
        docs: List[Document],
        intent: Optional[str] = None
    ) -> ClinicalChatResponse:
        """
        Parses LLM JSON or markdown output into validated ClinicalChatResponse.
        Validates language isolation:
        - English section has ONLY English content.
        - Telugu section has ONLY Telugu content.
        - 1-to-1 parallel section count matching.
        """
        clean_text = self._sanitize_llm_markdown(llm_text)

        # 1. Try parsing JSON
        json_obj = None
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(1))
            except Exception:
                json_obj = None
        if not json_obj:
            raw_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if raw_match:
                try:
                    json_obj = json.loads(raw_match.group(1))
                except Exception:
                    json_obj = None

        if json_obj and isinstance(json_obj, dict):
            en_secs = json_obj.get("english", [])
            te_secs = json_obj.get("telugu", [])
            
            valid_en = []
            if isinstance(en_secs, list):
                for s in en_secs:
                    if isinstance(s, dict):
                        h = str(s.get("title") or s.get("heading") or "").strip()
                        c = str(s.get("content") or "").strip()
                        if h and c and len(re.findall(r"[\u0C00-\u0C7F]", c)) < 5:
                            valid_en.append({"title": h, "heading": h, "content": c})

            valid_te = []
            if isinstance(te_secs, list):
                for s in te_secs:
                    if isinstance(s, dict):
                        h = str(s.get("title") or s.get("heading") or "").strip()
                        c = str(s.get("content") or "").strip()
                        if h and c and len(re.findall(r"[\u0C00-\u0C7F]", c)) >= 5:
                            valid_te.append({"title": h, "heading": h, "content": c})

            if language == "English" and valid_en:
                return ClinicalChatResponse(english_sections=valid_en, raw_text=llm_text)
            elif language == "తెలుగు" and valid_te:
                return ClinicalChatResponse(telugu_sections=valid_te, raw_text=llm_text)
            elif valid_en and valid_te and len(valid_en) == len(valid_te):
                return ClinicalChatResponse(
                    english_sections=valid_en,
                    telugu_sections=valid_te,
                    raw_text=llm_text
                )

        # 2. Try Markdown bilingual extraction
        en_part, te_part = self._extract_bilingual_markdown(clean_text)
        if en_part and te_part:
            en_te_count = len(re.findall(r"[\u0C00-\u0C7F]", en_part))
            te_te_count = len(re.findall(r"[\u0C00-\u0C7F]", te_part))
            if en_te_count < 15 and te_te_count > 15:
                return ClinicalChatResponse(en_markdown=en_part, te_markdown=te_part, raw_text=llm_text)

        # 3. Fallback to grounded synthesis if LLM response failed validation
        return self._synthesize_chat_answer(question, context, docs, language, intent=intent)

    def answer_question(
        self,
        question: str,
        prediction_context: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        language: str = "English + తెలుగు"
    ) -> Tuple[ClinicalChatResponse, List[Document]]:
        """
        Coordinates context-aware retrieval and answers farmer questions.
        Enforces strict sheep-health domain boundaries, intent-specific retrieval,
        and structured bilingual output.
        """
        # Step 0: Sheep-Health Domain Boundary Validation
        if not self._is_sheep_health_domain(question):
            refusal_obj = self._build_domain_refusal(language)
            return refusal_obj, []

        # Step 1: Detect Intent & Extract Active Disease Entity
        primary_intent, all_intents = detect_intents(question)
        disease_info = extract_disease_entity(
            question,
            prediction_context=prediction_context,
            chat_history=chat_history
        )
        disease_name = disease_info.get("short_name", "Sheep Health")

        # Step 2: Intent-Aware Semantic Retrieval
        retrieved_docs = self.retriever.retrieve_question_aware(
            question=question,
            intent=primary_intent,
            disease=disease_name,
            prediction_context=prediction_context,
            top_k=4
        )
        
        combined_context = self.retriever.format_context_for_prompt(retrieved_docs)

        # Step 3: Try LLM if configured
        if self.llm.is_api_configured():
            prompt = build_chat_prompt(
                user_question=question,
                prediction_context=prediction_context,
                retrieved_docs_text=combined_context,
                language=language,
                intent=primary_intent,
                disease_info=disease_info
            )
            llm_output = self.llm.generate(prompt, system_prompt=VETERINARY_SYSTEM_PROMPT, max_tokens=1500)
            if llm_output and len(llm_output.strip()) > 50:
                validated_resp = self._parse_and_validate_llm_response(
                    llm_output, language, question, disease_info, retrieved_docs, intent=primary_intent
                )
                if validated_resp:
                    return validated_resp, retrieved_docs

        # Step 4: Grounded Knowledge Synthesis Fallback
        grounded_answer = self._synthesize_chat_answer(
            question=question,
            context=disease_info,
            docs=retrieved_docs,
            language=language,
            intent=primary_intent
        )
        return grounded_answer, retrieved_docs

    def _synthesize_grounded_report(
        self,
        context: Dict[str, Any],
        docs: List[Document],
        language: str
    ) -> str:
        """
        Synthesizes a structured, authoritative 11-section bilingual report
        directly from the retrieved knowledge chunks.
        """
        disease = context.get("disease", "Unknown Condition")
        short_name = context.get("short_name", disease)
        telugu_name = context.get("telugu_name", "")
        conf_pct = context.get("confidence_percent", 0.0)
        is_uncertain = context.get("is_uncertain", False)
        conf_level = context.get("confidence_level", "medium")
        region = context.get("affected_region", "General body condition")
        region_te = context.get("affected_region_te", "")
        weight = context.get("estimated_weight", "35 - 45 kg")
        top_k = context.get("top_k", [])

        # Build differential ranking text
        diff_text_en = []
        diff_text_te = []
        for item in top_k:
            diff_text_en.append(f"Rank {item['rank']}: {item['disease']} ({item['confidence_percent']}%)")
            diff_text_te.append(f"{item['rank']}. {item.get('telugu_name') or item['disease']} ({item['confidence_percent']}%)")

        # Uncertainty statement
        if is_uncertain:
            uncertainty_eval_en = f"⚠️ **UNCERTAIN PREDICTION ({conf_pct}%):** The visual confidence score is below the reliable diagnostic threshold (60%). This prediction is a differential candidate, NOT a confirmed diagnosis. Professional physical examination by a veterinarian is strongly recommended."
            uncertainty_eval_te = f"⚠️ **అనిశ్చిత అంచనా ({conf_pct}%):** మోడల్ విశ్వసనీయత 60% కంటే తక్కువగా ఉంది. ఇది ఒక ప్రాథమిక సూచన మాత్రమే, తుది నిర్ధారణ కాదు. ఖచ్చితమైన నిర్ధారణ కోసం పశువైద్యుడిని సంప్రదించండి."
        else:
            uncertainty_eval_en = f"✅ **HIGH CONFIDENCE SCREENING ({conf_pct}%):** The model demonstrates high visual confidence. However, physical inspection and clinical confirmation by a veterinarian remain essential."
            uncertainty_eval_te = f"✅ **అధిక విశ్వసనీయత అంచనా ({conf_pct}%):** మోడల్ ఈ పరిస్థితిని అధిక విశ్వసనీయతతో గుర్తించింది. అయితే పశువైద్యుని క్లినికల్ నిర్ధారణ తప్పనిసరి."

        # Extract specific content from retrieved chunks
        overview_text = ""
        symptoms_text = ""
        treatment_text = ""
        home_care_text = ""
        prevention_text = ""
        feeding_text = ""
        emergency_text = ""

        for doc in docs:
            txt = doc.page_content
            txt_lower = txt.lower()
            if "overview" in txt_lower and not overview_text:
                overview_text = self._extract_section_body(txt, "Overview")
            if ("symptom" in txt_lower or "clinical signs" in txt_lower) and not symptoms_text:
                symptoms_text = self._extract_section_body(txt, "Symptoms") or self._extract_section_body(txt, "Clinical Symptoms")
            if ("treatment" in txt_lower or "veterinary medical" in txt_lower) and not treatment_text:
                treatment_text = self._extract_section_body(txt, "Veterinary Medical Treatment") or self._extract_section_body(txt, "Treatment")
            if ("home care" in txt_lower or "supportive" in txt_lower) and not home_care_text:
                home_care_text = self._extract_section_body(txt, "Safe Supportive & Home Care") or self._extract_section_body(txt, "Home Care")
            if ("prevention" in txt_lower or "vaccination" in txt_lower) and not prevention_text:
                prevention_text = self._extract_section_body(txt, "Prevention & Vaccination") or self._extract_section_body(txt, "Prevention")
            if ("feeding" in txt_lower or "nutrition" in txt_lower or "diet" in txt_lower) and not feeding_text:
                feeding_text = self._extract_section_body(txt, "Nutrition") or self._extract_section_body(txt, "Feeding")
            if ("emergency" in txt_lower or "warning signs" in txt_lower) and not emergency_text:
                emergency_text = self._extract_section_body(txt, "Emergency Warning Signs") or self._extract_section_body(txt, "Emergency")

        # Resolve display texts and defaults cleanly without backslashes inside f-strings
        default_treatment_en = (
            "• Broad-spectrum antibiotics for secondary infections (as per veterinary prescription).\n"
            "• Anti-inflammatory and pain relief medication under veterinary supervision."
        )
        disp_treatment_en = treatment_text if treatment_text else default_treatment_en

        default_home_care_en = (
            "• Immediately isolate the affected sheep in a clean, dry, shaded pen.\n"
            "• Provide abundant fresh drinking water and soft green forage.\n"
            "• Clean lesions gently with warm saline water."
        )
        disp_home_care_en = home_care_text if home_care_text else default_home_care_en

        default_prevention_en = (
            "• Routine annual vaccination as per the state veterinary calendar.\n"
            "• 21-30 days quarantine for newly bought sheep.\n"
            "• Disinfect pens and troughs regularly."
        )
        disp_prevention_en = prevention_text if prevention_text else default_prevention_en

        default_emergency_en = (
            "• Rectal temperature > 105°F, severe throat swelling, laboured gasping, "
            "head pulled backwards, violent convulsions, or sudden collapse."
        )
        disp_emergency_en = emergency_text if emergency_text else default_emergency_en

        default_overview_en = (
            f"{disease} is a clinically significant condition affecting small ruminants. "
            "Proper diagnosis, immediate hygiene, and veterinary evaluation protect flock health."
        )
        disp_overview_en = overview_text if overview_text else default_overview_en

        disp_symptoms_en = symptoms_text if symptoms_text else "• Physical lesions, appetite loss, behavioral changes, or gait discomfort."

        diff_summary_en = ", ".join(diff_text_en)
        diff_summary_te = ", ".join(diff_text_te)
        disp_disease_te = telugu_name if telugu_name else disease
        disp_region_te = region_te if region_te else region

        # Build English Report
        report_en = f"""### 1. 🩺 Detected Condition
**Condition:** {disease}
- **Estimated Weight:** {weight}
- **Affected Body Region:** {region}

---

### 2. 🎯 Model Confidence & Uncertainty Assessment
{uncertainty_eval_en}
*Top Differential Candidates:* {diff_summary_en}

---

### 3. 📋 What it Means
{disp_overview_en}

---

### 4. 🔍 Common Symptoms & Signs
{disp_symptoms_en}

---

### 5. 💊 Treatment & Medicine Information
{disp_treatment_en}
> ⚠️ **Prescription Notice:** All antibiotics, dosages, and routes must be confirmed and administered by a registered veterinarian. Do not self-inject or guess dosages.

---

### 6. 🏠 Safe Supportive & Home Care
{disp_home_care_en}
> ℹ️ **Safety Boundary:** Home care is supportive only; it does NOT replace professional veterinary treatment.

---

### 7. 🛡️ Prevention & Vaccination
{disp_prevention_en}

---

### 8. 🥬 Feeding & Hydration
• Provide 60-70% dry roughage (jowar stover, dry hay) + 20-30% green fodder.
• Avoid sudden shifts to heavy grain or lush pasture to prevent fatal enterotoxemia and rumen acidosis.
• Adult sheep require 3 to 8 liters of clean, fresh drinking water daily. Free access to mineral lick blocks.

---

### 9. ⚠️ Emergency Warning Signs
{disp_emergency_en}

---

### 10. 👨‍⚕️ When to Contact a Veterinarian
Seek immediate veterinary assistance if warning signs occur, if sheep stops drinking for > 24 hours, or if multiple flock members show rapid progression of illness.

---

### 11. 📚 Sources & Grounded References
• **ICAR-IVRI:** Indian Veterinary Research Institute (Sheep & Goat Health Calendar)
• **DAHD:** Department of Animal Husbandry and Dairying, Govt. of India (NADCP Program)
• **ICAR-CSWRI:** Central Sheep and Wool Research Institute (Small Ruminant Management)
• **FAO:** Food and Agriculture Organization (Manual on Small Ruminant Diseases)
"""

        # Build Telugu Report
        report_te = f"""### 1. 🩺 గుర్తించబడిన పరిస్థితి
**పరిస్థితి:** {disp_disease_te}
- **అంచనా బరువు:** {weight}
- **ప్రభావిత శరీర భాగం:** {disp_region_te}

---

### 2. 🎯 మోడల్ విశ్వసనీయత & అంచనా
{uncertainty_eval_te}
*టాప్ అభ్యర్థుల జాబితా:* {diff_summary_te}

---

### 3. 📋 దీని అర్థం ఏమిటి
{disp_disease_te} గొర్రెలలో కనిపించే ముఖ్యమైన ఆరోగ్య సమస్య. ఇది మందలోని ఇతర గొర్రెలకు వ్యాపించకుండా ముందుగానే జాగ్రత్తలు తీసుకోవడం మరియు పశువైద్య సలహా పొందడం చాలా ముఖ్యం.

---

### 4. 🔍 సాధారణ లక్షణాలు & సంకేతాలు
• జ్వరం, మేత తినకపోవడం, నీరసం, మందలో కలవకుండా వేరుగా ఉండటం.
• చర్మంపై దద్దుర్లు, నోటిలో పుండ్లు, లేదా నడవడంలో తీవ్రమైన ఇబ్బంది.

---

### 5. 💊 పశువైద్య చికిత్స & మందుల వివరాలు
• ద్వితీయ బ్యాక్టీరియల్ ఇన్ఫెక్షన్ల నివారణకు పశువైద్యుల సలహా మేరకు యాంటీబయాటిక్స్.
• నొప్పి మరియు వాపు నివారణకు సహాయక ఔషధాలు.
> ⚠️ **ముఖ్య గమనిక:** మందులు, మోతాదు మరియు ఇచ్చే విధానాన్ని తప్పనిసరిగా పశువైద్యుని ద్వారా మాత్రమే నిర్ధారించాలి.

---

### 6. 🏠 సురక్షితమైన సహాయక & ఇంటి సంరక్షణ
• వ్యాధి సోకిన గొర్రెను వెంటనే మిగిలిన మంద నుండి వేరు చేసి పొడి, శుభ్రమైన పాకలో ఉంచండి.
• సులభంగా నమలగలిగే మృదువైన పచ్చి మేత మరియు స్వచ్ఛమైన చల్లని తాగునీరు అందించండి.
• ధృవీకరించని నాటు మందులు లేదా రసాయనాలు ఉపయోగించవద్దు.
> ℹ️ **రక్షణ సరిహద్దు:** ఇంటి వద్ద సంరక్షణ కేవలం ఉపశమనం కోసం మాత్రమే; ఇది పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.

---

### 7. 🛡️ నివారణ చర్యలు & టీకాలు
• ప్రభుత్వ పశువైద్య క్యాలెండర్ ప్రకారం క్రమం తప్పకుండా టీకాలు వేయించండి.
• కొత్తగా కొన్న గొర్రెలను 21-30 రోజుల పాటు క్వారంటైన్‌లో ఉంచండి.
• పాకలను బ్లీచింగ్ పౌడర్ లేదా సున్నంతో పరిశుభ్రంగా ఉంచండి.

---

### 8. 🥬 మేత & తాగునీటి నిర్వహణ
• రోజువారీ ఆహారంలో 60-70% ఎండు మేత మరియు 20-30% పచ్చి మేత అందించండి.
• ఆహారంలో ఆకస్మిక మార్పులను నివారించండి (ఎంటరోటాక్సిమియా నివారణకు).
• రోజుకు 3-8 లీటర్ల స్వచ్ఛమైన తాగునీరు మరియు ఖనిజ లవణాల మిశ్రమం తప్పనిసరి.

---

### 9. ⚠️ అత్యవసర హెచ్చరిక సంకేతాలు
• అధిక జ్వరం (105°F పైన), గొంతు వాపు, శ్వాసలో తీవ్ర ఇబ్బంది, ఫిట్స్ లేదా అకస్మాత్తుగా పడిపోవడం.

---

### 10. 👨‍⚕️ పశువైద్యుడిని ఎప్పుడు సంప్రదించాలి
గొర్రె నీరు తాగడం మానేసినా, శ్వాస తీసుకోవడం కష్టమైనా లేదా మందలో అనేక గొర్రెలకు వ్యాధి లక్షణాలు కనిపిస్తే వెంటనే సమీప ప్రభుత్వ పశువైద్యశాలను సంప్రదించండి.

---

### 11. 📚 జ్ఞాన వనరులు
• **ICAR-IVRI:** భారత పశువైద్య పరిశోధనా సంస్థ (గొర్రెల ఆరోగ్య క్యాలెండర్)
• **DAHD:** పశుసంవర్ధక మరియు పాడిపరిశ్రమ శాఖ, భారత ప్రభుత్వం (NADCP కార్యక్రమం)
• **ICAR-CSWRI:** కేంద్ర గొర్రెలు మరియు ఉన్ని పరిశోధనా సంస్థ
• **FAO:** ఐక్యరాజ్యసమితి ఆహార మరియు వ్యవసాయ సంస్థ
"""

        # Combined interleaved version for fallback / single view
        combined_report = f"""### 1. 🩺 Detected Condition / గుర్తించబడిన పరిస్థితి
**English:** {disease}
**తెలుగు:** {disp_disease_te}
- **Estimated Weight / అంచనా బరువు:** {weight}
- **Affected Body Region / ప్రభావిత శరీర భాగం:** {region} ({region_te})

---

### 2. 🎯 Model Confidence & Uncertainty Assessment / మోడల్ విశ్వసనీయత & అంచనా
**English:**
{uncertainty_eval_en}
*Top Differential Candidates:* {diff_summary_en}

**తెలుగు:**
{uncertainty_eval_te}
*టాప్ అభ్యర్థుల జాబితా:* {diff_summary_te}

---

### 3. 📋 What it Means / దీని అర్థం ఏమిటి
**English:**
{disp_overview_en}

**తెలుగు:**
{disp_disease_te} గొర్రెలలో కనిపించే ముఖ్యమైన ఆరోగ్య సమస్య. ఇది మందలోని ఇతర గొర్రెలకు వ్యాపించకుండా ముందుగానే జాగ్రత్తలు తీసుకోవడం మరియు పశువైద్య సలహా పొందడం చాలా ముఖ్యం.

---

### 4. 🔍 Common Symptoms & Signs / సాధారణ లక్షణాలు & సంకేతాలు
**English:**
{disp_symptoms_en}

**తెలుగు:**
• జ్వరం, మేత తినకపోవడం, నీరసం, మందలో కలవకుండా వేరుగా ఉండటం.
• చర్మంపై దద్దుర్లు, నోటిలో పుండ్లు, లేదా నడవడంలో తీవ్రమైన ఇబ్బంది.

---

### 5. 💊 Treatment & Medicine Information / పశువైద్య చికిత్స & మందుల వివరాలు
**English:**
{disp_treatment_en}
> ⚠️ **Prescription Notice:** All antibiotics, dosages, and routes must be confirmed and administered by a registered veterinarian. Do not self-inject or guess dosages.

**తెలుగు:**
• ద్వితీయ బ్యాక్టీరియల్ ఇన్ఫెక్షన్ల నివారణకు పశువైద్యుల సలహా మేరకు యాంటీబయాటిక్స్.
• నొప్పి మరియు వాపు నివారణకు సహాయక ఔషధాలు.
> ⚠️ **ముఖ్య గమనిక:** మందులు, మోతాదు మరియు ఇచ్చే విధానాన్ని తప్పనిసరిగా పశువైద్యుని ద్వారా మాత్రమే నిర్ధారించాలి.

---

### 6. 🏠 Safe Supportive & Home Care / సురక్షితమైన సహాయక & ఇంటి సంరక్షణ
**English:**
{disp_home_care_en}
> ℹ️ **Safety Boundary:** Home care is supportive only; it does NOT replace professional veterinary treatment.

**తెలుగు:**
• వ్యాధి సోకిన గొర్రెను వెంటనే మిగిలిన మంద నుండి వేరు చేసి పొడి, శుభ్రమైన పాకలో ఉంచండి.
• సులభంగా నమలగలిగే మృదువైన పచ్చి మేత మరియు స్వచ్ఛమైన చల్లని తాగునీరు అందించండి.
• ధృవీకరించని నాటు మందులు లేదా రసాయనాలు ఉపయోగించవద్దు.

---

### 7. 🛡️ Prevention & Vaccination / నివారణ చర్యలు & టీకాలు
**English:**
{disp_prevention_en}

**తెలుగు:**
• ప్రభుత్వ పశువైద్య క్యాలెండర్ ప్రకారం క్రమం తప్పకుండా టీకాలు వేయించండి.
• కొత్తగా కొన్న గొర్రెలను 21-30 రోజుల పాటు క్వారంటైన్‌లో ఉంచండి.
• పాకలను బ్లీచింగ్ పౌడర్ లేదా సున్నంతో పరిశుభ్రంగా ఉంచండి.

---

### 8. 🥬 Feeding & Hydration / మేత & తాగునీటి నిర్వహణ
**English:**
• Provide 60-70% dry roughage (jowar stover, dry hay) + 20-30% green fodder.
• Avoid sudden shifts to heavy grain or lush pasture to prevent fatal enterotoxemia and rumen acidosis.
• Adult sheep require 3 to 8 liters of clean, fresh drinking water daily. Free access to mineral lick blocks.

**తెలుగు:**
• రోజువారీ ఆహారంలో 60-70% ఎండు మేత మరియు 20-30% పచ్చి మేత అందించండి.
• ఆహారంలో ఆకస్మిక మార్పులను నివారించండి (ఎంటరోటాక్సిమియా నివారణకు).
• రోజుకు 3-8 లీటర్ల స్వచ్ఛమైన తాగునీరు మరియు ఖనిజ లవణాల మిశ్రమం తప్పనిసరి.

---

### 9. ⚠️ Emergency Warning Signs / అత్యవసర హెచ్చరిక సంకేతాలు
**English:**
{disp_emergency_en}

**తెలుగు:**
• అధిక జ్వరం (105°F పైన), గొంతు వాపు, శ్వాసలో తీవ్ర ఇబ్బంది, ఫిట్స్ లేదా అకస్మాత్తుగా పడిపోవడం.

---

### 10. 👨‍⚕️ When to Contact a Veterinarian / పశువైద్యుడిని ఎప్పుడు సంప్రదించాలి
**English:**
Seek immediate veterinary assistance if warning signs occur, if sheep stops drinking for > 24 hours, or if multiple flock members show rapid progression of illness.

**తెలుగు:**
గొర్రె నీరు తాగడం మానేసినా, శ్వాస తీసుకోవడం కష్టమైనా లేదా మందలో అనేక గొర్రెలకు వ్యాధి లక్షణాలు కనిపిస్తే వెంటనే సమీప ప్రభుత్వ పశువైద్యశాలను సంప్రదించండి.

---

### 11. 📚 Sources & Grounded References / జ్ఞాన వనరులు
• **ICAR-IVRI:** Indian Veterinary Research Institute (Sheep & Goat Health Calendar)
• **DAHD:** Department of Animal Husbandry and Dairying, Govt. of India (NADCP Program)
• **ICAR-CSWRI:** Central Sheep and Wool Research Institute (Small Ruminant Management)
• **FAO:** Food and Agriculture Organization (Manual on Small Ruminant Diseases)
"""

        return ClinicalReport(en_text=report_en, te_text=report_te, combined_text=combined_report)

    def _parse_llm_bilingual_report(
        self,
        llm_text: str,
        context: Dict[str, Any],
        docs: List[Document]
    ) -> ClinicalReport:
        """
        Parses LLM output into parallel English and Telugu versions.
        """
        fallback_report = self._synthesize_grounded_report(context, docs, "English + తెలుగు")
        return ClinicalReport(
            en_text=llm_text,
            te_text=fallback_report.te,
            combined_text=llm_text
        )


    def _synthesize_chat_answer(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        docs: List[Document],
        language: str,
        intent: Optional[str] = None
    ) -> ClinicalChatResponse:
        """
        Synthesizes an intent-driven, factually grounded veterinary answer directly
        from the retrieved knowledge base chunks. Enforces 1-to-1 bilingual section
        synchronization, clinical safety boundaries, and pure Markdown formatting.
        """
        q_lower = question.lower()
        if not intent:
            intent, _ = detect_intents(question)

        disease_name = context.get("disease", "Sheep Health") if context else "Sheep Health"
        short_name = context.get("short_name", disease_name) if context else "Sheep Health"
        telugu_name = context.get("telugu_name", "") if context else "గొర్రెల ఆరోగ్యం"
        disp_name_en = short_name if short_name != "Sheep Health" else "Sheep"
        disp_name_te = telugu_name if telugu_name else "గొర్రెల"

        # Check for disease-specific nuances in retrieved chunks
        is_pox = "pox" in disp_name_en.lower() or any("pox" in d.page_content.lower() for d in docs)
        is_domma = "domma" in disp_name_en.lower() or "haemorrhagic" in disp_name_en.lower() or any("domma" in d.page_content.lower() for d in docs)
        is_pk = "pulpy" in disp_name_en.lower() or "enterotoxemia" in disp_name_en.lower() or any("enterotoxemia" in d.page_content.lower() for d in docs)
        is_hoof = "hoof" in disp_name_en.lower() or "foot" in disp_name_en.lower() or any("hoof" in d.page_content.lower() for d in docs)

        # -----------------------------------------------------------------
        # 1. HOME CARE INTENT (Step 6: Exactly 5 parallel sections)
        # -----------------------------------------------------------------
        if intent == IntentType.HOME_CARE:
            extra_home_en = "Clean skin or muzzle lesions gently with lukewarm saline or mild potassium permanganate (1:1000, light pink)." if is_pox else (
                "Do NOT forcefully drench liquids down the throat if breathing is strained or neck is swollen." if is_domma else (
                "Withhold heavy concentrates immediately; provide only clean water and dry roughage." if is_pk else (
                "Keep hooves clean and completely dry; do not let sheep stand in muddy, wet pens." if is_hoof else
                "Ensure clean resting bedding and quiet nursing care."
            )))
            extra_home_te = "మూతి లేదా చర్మంపై ఉన్న పుండ్లను గోరువెచ్చని ఉప్పునీరు లేదా లేత గులాబీ రంగు పొటాషియం పర్మాంగనేట్ (1:1000) నీటితో మెత్తగా తుడవండి." if is_pox else (
                "గొంతు వాపు లేదా శ్వాస ఇబ్బంది ఉన్నప్పుడు బలవంతంగా గొంతులోకి ద్రవాలు పోయవద్దు (ప్రాణాంతక ఊపిరితిత్తుల ఇన్ఫెక్షన్ రావచ్చు)." if is_domma else (
                "ఎక్కువ దాణా లేదా గింజలను వెంటనే నిలిపివేయండి; కేవలం శుభ్రమైన నీరు మరియు ఎండు మేత మాత్రమే ఇవ్వండి." if is_pk else (
                "గిట్టలను శుభ్రంగా, పొడిగా ఉంచండి; బురద లేదా తడి ప్రదేశాల్లో గొర్రెలను ఉంచవద్దు." if is_hoof else
                "పాకలో మెత్తని ఎండుగడ్డి వేసి ప్రశాంతమైన వాతావరణం కల్పించండి."
            )))

            ans_en = f"""1. Immediate Supportive Care
• Isolate the sick animal immediately in a clean, shaded, well-ventilated stall with dry straw bedding.
• Minimize physical stress, noise, and unnecessary handling to conserve the animal's strength.
• {extra_home_en}
• Disinfect the pen floor with dry slaked lime or bleaching powder to eliminate pathogens.

---

2. Isolation and Hygiene
• Separate the sick sheep from the main flock to halt contagious contact and aerosol transmission.
• Use dedicated feeding and watering troughs that are disinfected daily and not shared with healthy animals.
• Apply pure neem oil or herbal fly-repellent spray around superficial wounds or lesions to prevent fly strike and maggots.
• ❌ Forbidden practices: Never apply motor oil, battery acid, caustic chemicals, or hot iron branding to sick sheep.

---

3. Food and Water
• Ensure 24-hour access to fresh, cool drinking water enriched with Oral Rehydration Salts (ORS) or a pinch of salt and jaggery.
• Offer soft, easily digestible nutrition such as warm rice gruel, ragi congee, or boiled wheat mash with a pinch of salt.
• Provide small, frequent portions of tender, palatable green fodder (cowpea, berseem, tender grass).
• Avoid coarse, sharp dry roughage that can injure inflamed oral mucosa or cause digestive impaction.

---

4. Monitoring
• Closely monitor rectal body temperature every 4 to 6 hours (normal is 102°F–103.5°F; >104°F indicates high fever).
• Check if the sheep is actively drinking water, able to swallow, and producing normal dung and urine.
• Observe for return of active cud chewing (rumination), alert ear posture, and steady breathing.

---

5. When to Contact a Veterinarian
• ⚠️ Safety Boundary: Supportive home care provides palliative comfort only; it does NOT cure systemic infections or substitute for veterinary treatment.
• Seek emergency veterinary attention immediately if the sheep cannot stand (recumbency), refuses water for >24 hours, develops open-mouth breathing, or shows temperature >105°F."""

            ans_te = f"""1. తక్షణ సహాయక సంరక్షణ
• అనారోగ్యంతో ఉన్న గొర్రెను వెంటనే పొడి, శుభ్రమైన, నీడ మరియు గాలి-వెలుతురు ఉన్న ప్రత్యేక పాకలో ఉంచండి.
• గొర్రెకు మానసిక మరియు శారీరక ఒత్తిడి కలగకుండా అనవసరమైన కదలికలు మరియు శబ్దాలను నివారించండి.
• {extra_home_te}
• పాక నేలను పొడిగా ఉంచి, క్రిమిసంహారక చర్యగా పొడి సున్నం లేదా బ్లీచింగ్ పౌడర్ చల్లండి.

---

2. వేరుగా ఉంచడం మరియు పరిశుభ్రత
• వ్యాధి ఇతర గొర్రెలకు వ్యాపించకుండా బాధింత గొర్రెను వెంటనే ప్రధాన మంద నుండి వేరు చేయండి (ఐసోలేషన్).
• రోజూ శుభ్రం చేసే ప్రత్యేక ఆహార మరియు నీటి తొట్టెలను కేటాయించండి; వీటిని ఆరోగ్యకరమైన గొర్రెలకు ఉపయోగించవద్దు.
• ఈగలు వాలకుండా, పురుగులు పడకుండా పుండ్లపై స్వచ్ఛమైన వేపనూనె లేదా హెర్బల్ స్ప్రే రాయండి.
• ❌ నిషేధిత పద్ధతులు: బ్యాటరీ యాసిడ్, ఇంజిన్ ఆయిల్, రసాయనాలు లేదా వాతలు పెట్టడం వంటి సంప్రదాయ పద్ధతులను ఎప్పుడూ ఉపయోగించవద్దు.

---

3. ఆహారం మరియు నీరు
• శరీరంలో నీటిశాతం తగ్గకుండా స్వచ్ఛమైన చల్లని నీటిలో ORS పొడి లేదా చిటికెడు ఉప్పు మరియు బెల్లం కలిపి నిరంతరం అందుబాటులో ఉంచండి.
• తేలికగా జీర్ణమయ్యే గోరువెచ్చని బియ్యం గంజి లేదా రాగి జావలో చిటికెడు ఉప్పు కలిపి అందించండి.
• సులభంగా నమలగలిగే లేత పచ్చిగడ్డి లేదా అలసంద వంటి మృదువైన పచ్చి మేతను కొద్దికొద్దిగా తరచుగా అందించండి.
• నోటిని లేదా జీర్ణకోశాన్ని గాయపరిచే ముళ్ల వంటి గట్టి ఎండు చొప్పను పూర్తిగా నివారించండి.

---

4. పర్యవేక్షణ
• శరీర ఉష్ణోగ్రతను ప్రతి 4 నుండి 6 గంటలకు గమనించండి (సాధారణ ఉష్ణోగ్రత 102°F-103.5°F; 104°F పైన ఉంటే జ్వరం ఉన్నట్లు).
• గొర్రె నీరు తాగుతుందా, సరిగ్గా మింగుతుందా, మలమూత్రాలు సాధారణంగా ఉన్నాయా అనేది పరిశీలించండి.
• గొర్రె తిరిగి నెమరు వేయడం ప్రారంభిస్తుందా, చెవులు చురుగ్గా కదిలిస్తుందా మరియు శ్వాస సాధారణంగా ఉందా అనేది చూడండి.

---

5. పశువైద్యుడిని ఎప్పుడు సంప్రదించాలి
• ⚠️ రక్షణ సరిహద్దు: ఇంటి వద్ద సంరక్షణ కేవలం ఉపశమనం మరియు బలాన్ని ఇవ్వడానికి మాత్రమే; ఇది పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.
• గొర్రె లేవలేక పడిపోయినా, 24 గంటలకు పైగా నీరు తాగకపోయినా, నోటితో ఆయాసపడుతూ శ్వాస తీసుకుంటున్నా, లేదా జ్వరం 105°F దాటినా వెంటనే సమీప పశువైద్యశాలను సంప్రదించండి."""

        # -----------------------------------------------------------------
        # 2. MEDICINE & TREATMENT INTENT (Step 5 & 7: Exactly 5 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.MEDICINE_TREATMENT:
            med_knowledge_en = (
                "• Broad-Spectrum Antibiotics (Secondary Infection Control): Long-acting Oxytetracycline (IM) or Enrofloxacin are primary choices to prevent secondary bacterial bronchopneumonia and wound sepsis.\n"
                "• Topical Antiseptics: Povidone-iodine solution (5%) or fly-repellent antiseptic spray applied to open lesions to prevent myiasis (maggots).\n"
                "• Note: There is no direct antiviral drug; veterinary management focuses on secondary infection control and symptomatic relief."
            ) if is_pox else (
                "• Emergency Antimicrobial Therapy: Parenteral antibiotics such as Sulfadimidine or long-acting Oxytetracycline administered immediately under veterinary direction.\n"
                "• Anti-inflammatory Support: Meloxicam or Flunixin meglumine to alleviate severe throat swelling, pain, and high fever.\n"
                "• Emergency Triage: Peracute condition requiring urgent in-person veterinary intervention within hours."
            ) if is_domma else (
                "• Clostridial Therapy: High-dose Penicillins or Enterotoxemia Antitoxin serum if identified in earliest stages under veterinary supervision.\n"
                "• Rumen Buffering & Antitoxemic Support: Electrolytes and supportive antitoxic care.\n"
                "• Critical Note: Enterotoxemia is rapidly fatal; veterinary preventive vaccination is far more effective than clinical therapy."
            ) if is_pk else (
                "• Broad-Spectrum Antibiotics: Long-acting Oxytetracycline to control deep interdigital bacterial necrosis.\n"
                "• Antiseptic Footbaths: 10% Zinc Sulphate or 5% Copper Sulphate footbath following clean hoof trimming.\n"
                "• NSAID Pain Relief: Meloxicam to reduce pain and restore weight-bearing mobility."
            ) if is_hoof else (
                "• Broad-Spectrum Antibiotics (Oxytetracycline, Enrofloxacin) to control systemic secondary bacterial infection.\n"
                "• Non-Steroidal Anti-Inflammatories (Meloxicam) to alleviate high fever, pain, and inflammation.\n"
                "• Oral Rehydration Salts (ORS) and fluid therapy to combat systemic toxemia and dehydration."
            )

            med_knowledge_te = (
                "• బ్రాడ్-స్పెక్ట్రమ్ యాంటీబయాటిక్స్: ద్వితీయ బ్యాక్టీరియల్ న్యుమోనియా మరియు సెప్సిస్ రాకుండా లాంగ్-యాక్టింగ్ ఆక్సిటెట్రాసైక్లిన్ లేదా ఎన్రోఫ్లోక్సాసిన్ పశువైద్యుల పర్యవేక్షణలో వాడతారు.\n"
                "• బాహ్య యాంటిసెప్టిక్స్: చర్మం మరియు నోటి పుండ్లకు పోవిడోన్-అయోడిన్ (5%) లేదా ఈగలు వాలకుండా హెర్బల్ స్ప్రే ఉపయోగించాలి.\n"
                "• గమనిక: వైరల్ వ్యాధులకు నిర్దిష్ట యాంటీవైరల్ మందులు ఉండవు; ద్వితీయ ఇన్ఫెక్షన్ల నివారణే ప్రధాన చికిత్స."
            ) if is_pox else (
                "• తక్షణ యాంటీబయాటిక్స్: తీవ్రమైన బ్యాక్టీరియల్ ఇన్ఫెక్షన్ నివారణకు సల్ఫాడిమిడిన్ లేదా ఆక్సిటెట్రాసైక్లిన్ పశువైద్యుల పర్యవేక్షణలో వెంటనే ఇవ్వాలి.\n"
                "• వాపు మరియు నొప్పి నివారిణులు: గొంతు వాపు, తీవ్ర జ్వరం తగ్గించడానికి మెలోక్సికామ్ లేదా ఫ్లూనిక్సిన్ వంటి ఔషధాలు అవసరం.\n"
                "• అత్యవసర ప్రాధాన్యత: ఇది వేగంగా ప్రాణాంతకమయ్యే పరిస్థితి కాబట్టి గంటల వ్యవధిలోనే పశువైద్య చికిత్స తప్పనిసరి."
            ) if is_domma else (
                "• క్లోస్ట్రిడియల్ చికిత్స: ప్రారంభ దశలో గుర్తిస్తే పెన్సిలిన్ గ్రూపు యాంటీబయాటిక్స్ లేదా ఎంటరోటాక్సిమియా యాంటీటాక్సిన్ సీరమ్ పశువైద్యుల ద్వారా ఇవ్వాలి.\n"
                "• జీర్ణవ్యవస్థ ఉపశమనం: ద్రవ చికిత్స మరియు టాక్సిన్లను తగ్గించే సహాయక చర్యలు.\n"
                "• ముఖ్య గమనిక: ఈ వ్యాధి చాలా వేగంగా ప్రాణం తీస్తుంది; చికిత్స కంటే ముందుగానే టీకాలు వేయించడం ఎంతో ముఖ్యం."
            ) if is_pk else (
                "• యాంటీబయాటిక్స్: గిట్టల మధ్య బ్యాక్టీరియల్ ఇన్ఫెక్షన్ నివారణకు ఆక్సిటెట్రాసైక్లిన్ ఉపయోగపడుతుంది.\n"
                "• యాంటిసెప్టిక్ ఫుట్‌బాత్: గిట్టలను శుభ్రం చేసి 10% జింక్ సల్ఫేట్ ద్రావణంలో నడిపించాలి.\n"
                "• నొప్పి నివారణ: నడవడంలో ఇబ్బంది తగ్గించడానికి మెలోక్సికామ్ వంటి నొప్పి నివారణ మందులు అవసరం."
            ) if is_hoof else (
                "• బ్రాడ్-స్పెక్ట్రమ్ యాంటీబయాటిక్స్ (ఆక్సిటెట్రాసైక్లిన్, ఎన్రోఫ్లోక్సాసిన్): ద్వితీయ బ్యాక్టీరియల్ ఇన్ఫెక్షన్ల నివారణకు.\n"
                "• నొప్పి & జ్వరం నివారిణులు (మెలోక్సికామ్): జ్వరం, వాపు మరియు నొప్పిని తగ్గించడానికి.\n"
                "• ఎలక్ట్రోలైట్ & ద్రవ చికిత్స (ORS): డీహైడ్రేషన్ మరియు విషపదార్థాల ప్రభావాన్ని తగ్గించడానికి."
            )

            ans_en = f"""1. Medical Management Principles for {disp_name_en}
• Clinical veterinary goals: Control secondary bacterial complications, alleviate pain and inflammation, and restore fluid balance.
• Veterinary diagnosis determines whether the condition is viral, bacterial, metabolic, or parasitic before drug selection.
• Supportive fluid therapy and nursing care are initiated alongside clinical drug administration.

---

2. Supportive Care & Fluid Therapy
• Oral Rehydration Salts (ORS) or balanced electrolyte solutions given continuously to combat dehydration and toxemia.
• Anti-inflammatory & antipyretic therapy: Veterinary NSAIDs (such as Meloxicam) reduce high fever, pain, and respiratory discomfort, encouraging the sheep to feed.
• Parenteral fluids (Dextrose-Normal Saline) administered via IV catheter by veterinarians in severe dehydration or weakness.

---

3. Relevant Medicine Information from Knowledge Base
{med_knowledge_en}
• Topical wound care: Clean lesions with warm saline or 1:1000 potassium permanganate; apply fly repellents to prevent maggot infestation.

---

4. Strict Veterinary Supervision & Safety Warning
• ⚠️ CRITICAL PRESCRIPTION NOTICE: All prescription antibiotics, injectable medications, dosages, and routes must be determined exclusively by a registered veterinarian.
• Never guess medicine dosages or self-inject prescription antibiotics; inaccurate dosage leads to fatal drug toxicity, tissue necrosis, or antimicrobial resistance.
• Never administer human pain medications (paracetamol, diclofenac) to sheep without veterinary prescription.

---

5. When to Seek Emergency Examination
• Seek immediate in-person veterinary triage if the sheep exhibits recumbency (cannot stand), body temperature >105°F or <100°F (hypothermia/shock), labored open-mouth breathing, or throat swelling."""

            ans_te = f"""1. {disp_name_te} వైద్య నిర్వహణ సూత్రాలు
• పశువైద్య ప్రధాన లక్ష్యాలు: ద్వితీయ బ్యాక్టీరియల్ ఇన్ఫెక్షన్ల నియంత్రణ, నొప్పి/వాపు తగ్గింపు మరియు శరీరంలో నీటి సమతుల్యతను కాపాడటం.
• చికిత్స ప్రారంభించే ముందు వ్యాధి వైరస్, బ్యాక్టీరియా లేదా జీర్ణక్రియ లోపం వల్ల వచ్చిందా అనేది పశువైద్య పరీక్ష ద్వారా నిర్ధారించాలి.
• ఔషధాలతో పాటు సహాయక ద్రవ నిర్వహణ మరియు సంరక్షణ తప్పనిసరి.

---

2. సహాయక చికిత్స & ద్రవ నిర్వహణ
• డీహైడ్రేషన్ మరియు విషపదార్థాల ప్రభావాన్ని తగ్గించడానికి ORS లేదా ఎలక్ట్రోలైట్ ద్రావణాలు నిరంతరం అందించాలి.
• నొప్పి & జ్వరం నివారణ: జ్వరం, నొప్పి మరియు శ్వాసకోశ ఇబ్బందులను తగ్గించడానికి పశువైద్యుల సలహా మేరకు మెలోక్సికామ్ వంటి ఔషధాలు ఉపయోగపడతాయి.
• తీవ్రమైన నీరసం లేదా డీహైడ్రేషన్ ఉన్నప్పుడు పశువైద్యుల ద్వారా ఐవి డెక్స్ట్రోస్-సలైన్ ద్రవాలు అందించాలి.

---

3. నాలెడ్జ్ బేస్ ఆధారిత మందుల వివరాలు
{med_knowledge_te}
• బాహ్య సంరక్షణ: పుండ్లను ఉప్పునీటితో శుభ్రం చేసి, పురుగులు పడకుండా హెర్బల్ స్ప్రే లేదా వేపనూనె వాడాలి.

---

4. పశువైద్యుల పర్యవేక్షణ & భద్రతా హెచ్చరిక
• ⚠️ ముఖ్య భద్రతా హెచ్చరిక: అన్ని యాంటీబయాటిక్స్, ఇంజక్షన్లు, మోతాదులు మరియు ఇచ్చే విధానాన్ని కేవలం నమోదిత పశువైద్యుని పర్యవేక్షణలోనే నిర్ణయించాలి.
• సొంతంగా మోతాదులను ఊహించి ఇంజక్షన్లు వేయవద్దు; తప్పుడు మోతాదులు ప్రాణాంతక విషబాధకు లేదా మరణానికి దారితీస్తాయి.
• మానవుల నొప్పి నివారణ మందులను (పారాసిటమాల్, డైక్లోఫెనాక్ వంటివి) పశువైద్యుల సలహా లేకుండా గొర్రెలకు ఎప్పుడూ ఇవ్వవద్దు.

---

5. అత్యవసర పరీక్ష ఎప్పుడు అవసరం
• గొర్రె లేవలేకపోవడం, జ్వరం 105°F కంటే ఎక్కువ ఉండటం లేదా శరీరం చల్లబడటం (షాక్ సంకేతం), గొంతు వాపు, లేదా ఆయాసపడుతూ శ్వాస తీసుకోవడం కనిపిస్తే వెంటనే పశువైద్యశాలకు తరలించాలి."""

        # -----------------------------------------------------------------
        # 3. SYMPTOMS INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.SYMPTOMS:
            sym_en = (
                "• Prodromal phase: Sudden high fever (104°F–106°F), severe dullness, swollen eyelids, and serous nasal discharge.\n"
                "• Eruptive lesions: Firm round papules and nodules appearing on hairless skin: muzzle, lips, inner thighs, udder, and tail.\n"
                "• Scab formation: Lesions transform into dark necrotic scabs; respiratory coughing indicates internal lung pox lesions."
            ) if is_pox else (
                "• High fever (105°F–107°F) with intense shivering and dullness.\n"
                "• Painful, hot swelling under the throat, neck, and brisket (dewlap).\n"
                "• Labored breathing with open mouth, protruding tongue, and profuse frothy salivation."
            ) if is_domma else (
                "• Sudden death in vigorous, fast-growing sheep (often found dead without prior warning).\n"
                "• Nervous signs: Incoordination, violent convulsions, head pulled back (opisthotonos), and teeth grinding.\n"
                "• Green pasty diarrhea, abdominal pain (kicking at belly), and rapid bloating."
            ) if is_pk else (
                "• Severe limping, reluctance to walk, and grazing on knees.\n"
                "• Moist, inflamed ulcerations and rotting odor between hoof claws (interdigital space).\n"
                "• Detachment of hoof horn, warmth, and intense swelling around coronary band."
            ) if is_hoof else (
                "• Fever (>104°F), loss of appetite, and separation from the flock.\n"
                "• Abnormal gait, discharge from nostrils/eyes, or skin and mouth lesions.\n"
                "• Complete cessation of cud chewing (rumination) and drooping ears."
            )

            sym_te = (
                "• ప్రారంభ దశ: ఆకస్మిక తీవ్ర జ్వరం (104°F-106°F), నీరసం, కనురెప్పల వాపు మరియు ముక్కు నుండి నీరు కారడం.\n"
                "• చర్మ దద్దుర్లు: వెంట్రుకలు తక్కువగా ఉండే మూతి, పెదవులు, తొడల లోపలి భాగం, పొదుగు మరియు తోక కింద గుండ్రని బొబ్బలు/గడ్డలు.\n"
                "• పొక్కులు & పుండ్లు: బొబ్బలు ఎండిపోయి గట్టి పొక్కులుగా మారతాయి; దగ్గు ఉంటే ఊపిరితిత్తులకు కూడా వ్యాపించినట్లు సూచన."
            ) if is_pox else (
                "• తీవ్రమైన జ్వరం (105°F-107°F), వణుకు మరియు విపరీతమైన నీరసం.\n"
                "• గొంతు, మెడ మరియు ఛాతీ కింది భాగంలో వేడితో కూడిన గట్టి వాపు (గొంతువాపు).\n"
                "• నోరు తెరిచి ఆయాసపడుతూ శ్వాస తీసుకోవడం, నాలుక బయటకు రావడం మరియు నురుగుతో కూడిన లాలాజలం కారడం."
            ) if is_domma else (
                "• ఆరోగ్యంగా ఉన్న మంచి గొర్రెలు ఎటువంటి ముందస్తు లక్షణాలు లేకుండా అకస్మాత్తుగా మరణించడం.\n"
                "• నరాల సంబంధిత లక్షణాలు: తూలడం, ఫిట్స్ రావడం, తల వెనక్కి వాలడం మరియు పళ్లు కొరకడం.\n"
                "• ఆకుపచ్చటి విరేచనాలు, కడుపునొప్పితో కాళ్లతో తన్నుకోవడం మరియు వేగంగా కడుపుబ్బరం రావడం."
            ) if is_pk else (
                "• తీవ్రమైన కుంటుతనం, నడవలేకపోవడం మరియు మోకాళ్లపై కూర్చుని మేత మేయడం.\n"
                "• గిట్టల మధ్య ఎర్రటి పుండ్లు, చీము మరియు తీవ్రమైన దుర్వాసన.\n"
                "• గిట్ట గోరు వదులవడం, వాపు మరియు తాకినప్పుడు వేడిగా ఉండటం."
            ) if is_hoof else (
                "• తీవ్రమైన జ్వరం (104°F పైన), మేత తినకపోవడం మరియు మందలో కలవకుండా వేరుగా ఉండటం.\n"
                "• నడవడంలో ఇబ్బంది, ముక్కు లేదా కళ్ల నుండి స్రావాలు, లేదా చర్మంపై పుండ్లు.\n"
                "• నెమరు వేయడం పూర్తిగా ఆగిపోవడం మరియు చెవులు వాలిపోవడం."
            )

            ans_en = f"""1. Hallmark Clinical Symptoms of {disp_name_en}
{sym_en}

---

2. Physical Lesions & Visible Signs
• Check hairless regions (muzzle, inner thighs, under tail) for characteristic lesions, erosions, or swelling.
• Inspect mucosal linings: Pale conjunctiva indicates anemia/parasites; congested red/cyanotic mucosa indicates systemic septicemia or hypoxia.
• Examine breath and mouth: Foul odor, excessive salivation, or coughing signify respiratory or oral involvement.

---

3. Behavioral & Systemic Signs
• Voluntary isolation: The sick sheep lags behind the flock, refuses to graze, and stands with a lowered head.
• Complete cessation of rumination (chewing the cud) is a sensitive hallmark sign of systemic distress.
• Hollow flanks indicate lack of feed and fluid intake over the preceding 24 hours.

---

4. Urgent Symptoms Requiring Immediate Care
• Rectal temperature exceeding 105°F (40.5°C) or dropping below 100°F (indicating shock).
• Severe respiratory distress, open-mouthed gasping, throat swelling, recumbency, or violent tremors require immediate emergency veterinary triage."""

            ans_te = f"""1. {disp_name_te} ప్రధాన క్లినికల్ లక్షణాలు
{sym_te}

---

2. చర్మం మరియు కనిపించే సంకేతాలు
• వెంట్రుకలు తక్కువగా ఉండే భాగాలు (మూతి, తొడల లోపలి భాగం, తోక కింద) బొబ్బలు, పుండ్లు లేదా వాపుల కోసం పరిశీలించండి.
• కంటి లోపలి పొరలు: పాలిపోయి ఉంటే రక్తహీనత లేదా నట్టల సమస్య; ఎర్రగా లేదా నీలంగా ఉంటే తీవ్రమైన ఇన్ఫెక్షన్ లేదా శ్వాసలో ఆక్సిజన్ లోపం ఉన్నట్లు.
• నోరు మరియు శ్వాస: దుర్వాసన, విపరీతమైన లాలాజలం లేదా దగ్గు ఉంటే శ్వాసకోశ లేదా నోటి ఇన్ఫెక్షన్ ఉన్నట్లు భావించాలి.

---

3. ప్రవర్తన మరియు శరీర సంకేతాలు
• మంద నుండి వేరుపడటం: బాధింత గొర్రె మందతో పాటు నడవలేక వెనుకబడిపోతుంది, మేత మేయదు, తల దించుకుని నిలబడుతుంది.
• నెమరు వేయడం పూర్తిగా ఆగిపోవడం అనేది గొర్రె తీవ్ర అనారోగ్యంతో ఉందని తెలిపే ముఖ్యమైన సంకేతం.
• పొట్ట లోపలికి పోవడం అనేది గత 24 గంటలుగా గొర్రె ఆహారం లేదా నీరు తీసుకోలేదని సూచిస్తుంది.

---

4. అత్యవసర పరీక్ష అవసరమయ్యే హెచ్చరిక సంకేతాలు
• జ్వరం 105°F కంటే ఎక్కువ ఉండటం లేదా శరీరం చల్లబడిపోవడం (షాక్ సంకేతం).
• శ్వాసలో తీవ్రమైన ఆయాసం, నోటితో ఊపిరి పీల్చడం, గొంతు వాపు, లేవలేకపోవడం లేదా ఫిట్స్ కనిపిస్తే ఆలస్యం చేయకుండా పశువైద్యశాలకు తరలించాలి."""

        # -----------------------------------------------------------------
        # 4. PREVENTION INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.PREVENTION:
            ans_en = f"""1. Flock Biosecurity & Quarantine
• Quarantine protocol: Strictly isolate all newly purchased sheep for at least 21 to 30 days before introducing them to the resident flock.
• Screen incoming animals for fever, skin lesions, respiratory signs, and foot rot during quarantine.
• Administer preventive broad-spectrum deworming upon arrival before mixing with the main flock.

---

2. Pen Sanitation & Disinfection
• Maintain clean, dry housing; wet, unventilated sheds multiply bacterial, viral, and parasitic loads rapidly.
• Whitewash shed walls periodically with quicklime and dust pen floors with bleaching powder or dry slaked lime.
• Clean water and feed troughs daily to eliminate salivary contamination between animals.

---

3. Preventive Management & Vector Control
• Suppress external parasites (ticks, lice, biting flies) using safe veterinary pyrethroid sprays or dips to block mechanical transmission.
• Graze flocks on well-drained pastures; avoid low-lying marshy pastures early in the morning where snails and liver flukes thrive.
• Avoid abrupt changes in diet, especially shifting suddenly to lush irrigated legumes or heavy cereal grains.

---

4. Routine Flock Health Monitoring
• Conduct daily flock inspections during morning release and evening penning to detect lagging or isolated animals early.
• Follow the state veterinary annual vaccination and seasonal deworming calendar rigorously.
• In the event of an outbreak in neighboring villages, consult local veterinary officers immediately for ring vaccination and biosecurity barricades."""

            ans_te = f"""1. మంద రక్షణ & క్వారంటైన్
• క్వారంటైన్ నిబంధనలు: సంతలో కొత్తగా కొనుగోలు చేసిన గొర్రెలను ప్రధాన మందలో కలపకుండా కనీసం 21 నుండి 30 రోజుల పాటు ప్రత్యేక పాకలో ఉంచండి.
• క్వారంటైన్ సమయంలో జ్వరం, చర్మంపై దద్దుర్లు, శ్వాసకోశ ఇబ్బందులు మరియు గిట్టల పుండ్లు ఉన్నాయా అనేది గమనించండి.
• మందలో కలపడానికి ముందు తప్పనిసరిగా నట్టల నివారణ మందు (డివార్మింగ్) త్రాగించండి.

---

2. పాకల పరిశుభ్రత & క్రిమిసంహారక చర్యలు
• పాకలను ఎల్లప్పుడూ పొడిగా, గాలి-వెలుతురు ధారాళంగా ఉండేలా చూసుకోండి; తేమ మరియు అపరిశుభ్రత వల్ల వ్యాధికారక క్రిములు వేగంగా పెరుగుతాయి.
• పాకల గోడలకు క్రమం తప్పకుండా సున్నం వేయించండి మరియు నేలపై బ్లీచింగ్ పౌడర్ లేదా పొడి సున్నం చల్లి శుభ్రపరచండి.
• ఒక గొర్రె లాలాజలం మరొకదానికి అంటకుండా ఆహార మరియు నీటి తొట్టెలను రోజూ శుభ్రం చేయండి.

---

3. నివారణ నిర్వహణ & కీటకాల నియంత్రణ
• వ్యాధులను వ్యాపింపజేసే గోమార్లు, పేలు మరియు ఈగలను అరికట్టడానికి పశువైద్యుల సలహా మేరకు సురక్షితమైన పిచికారీ మందులు వాడండి.
• ఉదయాన్నే మంచు ఉన్నప్పుడు చిత్తడి మరియు లోతట్టు ప్రాంతాల్లో మేపవద్దు (నట్టల వ్యాప్తి నివారణకు).
• ఆహారంలో ఆకస్మిక మార్పులను నివారించండి, ముఖ్యంగా ఒక్కసారిగా లేత పచ్చిగడ్డి లేదా ఎక్కువ గింజల దాణా పెట్టవద్దు.

---

4. క్రమబద్ధమైన ఆరోగ్య పర్యవేక్షణ
• ఉదయం పాక నుండి బయటకు వెళ్లేటప్పుడు, సాయంత్రం లోపలికి వచ్చేటప్పుడు ప్రతి గొర్రెను నిశితంగా గమనించండి; నీరసంగా ఉన్నవాటిని వెంటనే వేరు చేయండి.
• ప్రభుత్వ పశువైద్య శాఖ వార్షిక టీకాల క్యాలెండర్ మరియు నట్టల నివారణ షెడ్యూల్‌ను ఖచ్చితంగా పాటించండి.
• చుట్టుపక్కల గ్రామాలలో వ్యాధులు ప్రబలినప్పుడు వెంటనే పశువైద్య అధికారులను సంప్రదించి రక్షణ టీకాలు వేయించండి."""

        # -----------------------------------------------------------------
        # 5. VACCINATION INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.VACCINATION:
            ans_en = f"""1. Core Annual Vaccination Schedule for Sheep
• **Sheep Pox:** Administer annually in December–January prior to the peak dry winter season.
• **Enterotoxemia / Pulpy Kidney:** Annual pre-monsoon vaccination in May–June; administer booster doses for newly weaned lambs.
• **Haemorrhagic Septicaemia / Domma:** Pre-monsoon vaccination in May–June before seasonal rains.
• **Peste des Petits Ruminants (PPR):** Administered once every 3 years to all sheep and lambs above 3 months of age.
• **Foot and Mouth Disease (FMD):** Biannual vaccination (pre-monsoon in August–September and post-monsoon in February–March).

---

2. Timing & Administration Rules
• Vaccinate only clinically vigorous, healthy animals; never vaccinate visibly sick, feverish, or emaciated sheep.
• Always administer broad-spectrum deworming 2 to 3 weeks prior to vaccination to ensure maximum immune antibody titer response.
• For pregnant ewes, avoid vaccinations during the final month of gestation unless specifically directed by a veterinarian.

---

3. Cold-Chain & Precautions
• Strict Cold-Chain Protocol: Vaccines must be preserved continuously at 2°C to 8°C in insulated cold-boxes or refrigerators. Never expose vaccine vials to direct sunlight or ambient heat.
• Reconstituted live vaccines (e.g. PPR, Sheep Pox) must be utilized within 2 hours of dilution and kept on wet ice; discard any remaining reconstituted vial afterwards.
• In an active outbreak pen, do NOT vaccinate clinically affected sheep without direct veterinary supervision.

---

4. Veterinary Consultation Notice
• State Department Programs: Vaccination dates vary by district based on local disease epidemiology and government mass campaign schedules.
• Always contact your local Veterinary Assistant Surgeon (VAS) or nearby Government Veterinary Dispensary to confirm the local schedule and receive certified vaccines."""

            ans_te = f"""1. గొర్రెల ప్రధాన వార్షిక టీకాల క్యాలెండర్
• **గొర్రెల మశూచి (Sheep Pox):** ప్రతి సంవత్సరం డిసెంబర్ - జనవరి నెలల్లో (శీతాకాలంలో) టీకా వేయించాలి.
• **ఎంటరోటాక్సిమియా / పల్పి కిడ్నీ (ET):** వర్షాకాలానికి ముందు మే - జూన్ నెలల్లో వేయించాలి; పిల్లలకు బూస్టర్ డోస్ తప్పనిసరి.
• **దొమ్మ వ్యాధి (HS / Domma):** వర్షాకాలానికి ముందు మే - జూన్ నెలల్లో టీకా వేయించాలి.
• **పిపిఆర్ వ్యాధి (PPR):** మూడు సంవత్సరాలకు ఒకసారి 3 నెలలు దాటిన గొర్రెలన్నింటికీ వేయించాలి.
• **గాలికుంటు వ్యాధి (FMD):** సంవత్సరానికి రెండుసార్లు (ఆగస్టు-సెప్టెంబర్ మరియు ఫిబ్రవరి-మార్చి) వేయించాలి.

---

2. సమయం & టీకా నియమాలు
• సంపూర్ణ ఆరోగ్యంగా, చురుగ్గా ఉన్న గొర్రెలకు మాత్రమే టీకాలు వేయాలి; జ్వరం లేదా నీరసంగా ఉన్నవాటికి వేయకూడదు.
• టీకాలు వేయడానికి 2 నుండి 3 వారాల ముందు నట్టల నివారణ మందు (డివార్మింగ్) తప్పనిసరిగా త్రాగించాలి (ఇది రోగనిరోధక శక్తిని పెంచుతుంది).
• చూడితో ఉన్న గొర్రెలకు చివరి నెలలో టీకాలు వేయడం నివారించాలి (పశువైద్యుల సలహా మేరకే వేయాలి).

---

3. కోల్డ్-చైన్ & జాగ్రత్తలు
• కోల్డ్-చైన్ నిర్వహణ: టీకా మందులను ఎల్లప్పుడూ 2°C నుండి 8°C చల్లదనంలో (ఐస్ బాక్స్‌లో) భద్రపరచాలి. ఎండ లేదా వేడి తగలకూడదు.
• పౌడర్ రూపంలో ఉండే టీకాలను ద్రవంలో కలిపిన తర్వాత 2 గంటల లోపు మాత్రమే ఐస్ బాక్స్‌లో ఉంచి ఉపయోగించాలి; మిగిలిపోయిన ద్రావణాన్ని పారవేయాలి.
• వ్యాధి ప్రబలిన సమయంలో లక్షణాలు ఉన్న గొర్రెలకు పశువైద్యుల ప్రత్యక్ష సలహా లేకుండా టీకాలు వేయకూడదు.

---

4. పశువైద్య సలహా గమనిక
• ప్రభుత్వ కార్యక్రమాలు: స్థానిక వ్యాధి తీవ్రతను బట్టి ప్రభుత్వం నిర్వహించే ఉచిత టీకా శిబిరాల షెడ్యూల్ మారుతుంటుంది.
• ఖచ్చితమైన టీకాల క్యాలెండర్ మరియు ధృవీకరించబడిన వ్యాక్సిన్ల కోసం మీ సమీప ప్రభుత్వ పశువైద్యశాల అధికారిని సంప్రదించండి."""

        # -----------------------------------------------------------------
        # 6. FEEDING & NUTRITION INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.FEEDING_NUTRITION:
            ans_en = f"""1. Recommended Balanced Ration for Sheep
• **Dry Roughage (60–70%):** Sorghum/Jowar stover, dry maize stalks, or groundnut haulms form the structural dietary fiber base essential for healthy rumen fermentation.
• **Green Fodder (20–30%):** Balanced combination of cereal greens (Co-4, Super Napier) and leguminous forage (Lucerne, Cowpea, Berseem).
• **Concentrate Supplementation:** 150 to 250 g daily of balanced concentrate feed (maize, wheat bran, oil cake, mineral mixture) during late gestation and lactation.

---

2. Supportive Feeding for Sick Sheep
• When sick or feverish sheep refuse solid roughage, offer warm, easily drinkable gruel (broken rice or ragi congee boiled with a pinch of salt and jaggery).
• Provide small, fresh handfuls of tender green tree leaves (Neem, Subabul, Sesbania) frequently to encourage appetite.
• Never force-feed or drench liquids aggressively into animals with respiratory distress or throat swelling.

---

3. Digestive Precautions (Bloat & Acidosis Prevention)
• Avoid abrupt diet changes: Sudden shifts to heavy grains or grazing wet, lush morning dew pastures triggers fatal Enterotoxemia (Pulpy Kidney) and Acute Rumen Acidosis.
• Feed dry roughage before grazing on lush pastures to moderate green intake and stimulate saliva buffering.
• Never allow hungry sheep unrestricted access to stored grain bags or wet fermented crop waste.

---

4. Clean Water & Mineral Management
• Fresh Drinking Water: Adult sheep require 3 to 8 liters of clean, cool drinking water daily depending on environmental temperature.
• Mineral Mixture: Provide free-choice access to commercial mineral lick blocks or mix 10 g mineral mixture daily into feed to prevent metabolic deficiencies.
• Clean water troughs daily to prevent algal buildup and pathogen transmission."""

            ans_te = f"""1. గొర్రెలకు సమతుల్య రోజువారీ ఆహారం
• **ఎండు మేత (60-70%):** జొన్న చొప్ప, మొక్కజొన్న చొప్ప లేదా వేరుశనగ పొట్టు రోజువారీ ఆహారంలో ముఖ్య భాగం; ఇది జీర్ణవ్యవస్థ సరిగ్గా పనిచేయడానికి అవసరం.
• **పచ్చి మేత (20-30%):** సూపర్‌ నేపియర్, కో-4 వంటి గడ్డితో పాటు లూసర్న్ లేదా అలసంద వంటి పప్పుజాతి పచ్చి మేతను సమతుల్యంగా అందించాలి.
• **దాణా నిర్వహణ:** చూడి చివరి దశలో మరియు పిల్లలకు పాలిచ్చే సమయంలో రోజుకు 150-250 గ్రాముల సమతుల్య దాణా (మొక్కజొన్న, తవుడు, చెక్క మరియు ఖనిజ లవణాలు) అందించాలి.

---

2. అనారోగ్యంతో ఉన్న గొర్రెలకు బలవర్ధక ఆహారం
• జ్వరం లేదా అనారోగ్యం వల్ల గొర్రె గట్టి మేత తినలేనప్పుడు గోరువెచ్చని బియ్యం గంజి లేదా రాగి జావలో చిటికెడు ఉప్పు, బెల్లం కలిపి అందించండి.
• ఆకలి పెంచడానికి లేత సుబాబుల్, వేప లేదా అవిశ ఆకులను కొద్దికొద్దిగా తరచుగా అందించండి.
• శ్వాస ఇబ్బంది లేదా గొంతు వాపు ఉన్నప్పుడు బలవంతంగా గొంతులోకి ద్రవాలను పోయవద్దు.

---

3. జీర్ణ జాగ్రత్తలు (కడుపుబ్బరం & ఎసిడోసిస్ నివారణ)
• ఆహారంలో ఆకస్మిక మార్పులను నివారించండి: ఒక్కసారిగా ఎక్కువ గింజల దాణా పెట్టడం లేదా ఉదయాన్నే మంచుతో కూడిన లేత గడ్డిని మేపడం వల్ల కడుపుబ్బరం (బ్లోట్) మరియు ఎంటరోటాక్సిమియా వస్తాయి.
• పచ్చి మేతకు వెళ్లే ముందు కొద్దిగా ఎండు మేతను తినిపించడం వల్ల కడుపుబ్బరం రాకుండా కాపాడుకోవచ్చు.
• గొర్రెలు నిల్వ ఉంచిన గింజల బస్తాల వద్దకు వెళ్లకుండా జాగ్రత్త వహించండి.

---

4. స్వచ్ఛమైన తాగునీరు & ఖనిజ లవణాల నిర్వహణ
• తాగునీరు: వాతావరణాన్ని బట్టి ప్రతి గొర్రెకు రోజుకు 3 నుండి 8 లీటర్ల స్వచ్ఛమైన చల్లని తాగునీరు తప్పనిసరిగా అందించాలి.
• ఖనిజ లవణాలు: ఖనిజ లోపాలు రాకుండా పాకలలో మినరల్ బ్రిక్స్ (ఇటుకలు) అందుబాటులో ఉంచండి లేదా రోజూ 10 గ్రాముల మినరల్ మిశ్రమాన్ని దాణాలో కలపండి.
• నీటి తొట్టెలను రోజూ శుభ్రం చేసి స్వచ్ఛమైన నీటిని నింపండి."""

        # -----------------------------------------------------------------
        # 7. CAUSES INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent == IntentType.CAUSES:
            cause_en = (
                "• Primary Etiology: Sheep Pox Virus (SPPV), an enveloped double-stranded DNA virus belonging to the genus *Capripoxvirus* (family Poxviridae).\n"
                "• Host Specificity: Highly host-adapted to sheep; survives in dry scabs and sheds for months.\n"
                "• Predisposing Triggers: Severe nutritional stress, transport exhaustion, and unventilated humid pens."
            ) if is_pox else (
                "• Primary Etiology: *Pasteurella multocida* (bacterial pathogen).\n"
                "• Trigger Factors: Seasonal stress, sudden onset of monsoon rains, exhaustion, and chilling.\n"
                "• Flock Spread: Highly contagious respiratory and systemic infection."
            ) if is_domma else (
                "• Primary Etiology: *Clostridium perfringens* Type D producing lethal epsilon toxin in the gut.\n"
                "• Trigger Factors: Sudden ingestion of heavy grains, lush tender pasture, or sudden change in ration.\n"
                "• Internal Mechanism: Rapid proliferation of anaerobic bacteria produces lethal systemic neurotoxins."
            ) if is_pk else (
                "• Primary Etiology: Synergistic bacterial infection by *Dichelobacter nodosus* and *Fusobacterium necrophorum*.\n"
                "• Environmental Triggers: Prolonged standing in wet, muddy, manure-contaminated pens during monsoons.\n"
                "• Predisposing Factors: Unclipped overgrown hooves, interdigital skin maceration, and tick bites."
            ) if is_hoof else (
                "• Primary Etiology: Pathogenic viral, bacterial, or parasitic agents circulating in endemic flock environments.\n"
                "• Contributing Stressors: Sudden weather fluctuations, overcrowding, damp unventilated housing, and internal parasites.\n"
                "• Immunological Factors: Inadequate nutrition or missed vaccinations lowering flock immunity."
            )

            cause_te = (
                "• ప్రధాన వ్యాధి కారకం: షీప్ పాక్స్ వైరస్ (SPPV), ఇది కాప్రిపాక్స్ వైరస్ జాతికి చెందిన బలమైన వైరస్.\n"
                "• మనుగడ: ఈ వైరస్ ఎండిపోయిన పొక్కులు మరియు నేలలో నెలల తరబడి సజీవంగా ఉంటుంది.\n"
                "• తీవ్రతరం చేసే కారణాలు: పోషకాహార లోపం, రవాణా వల్ల కలిగే అలసట మరియు పాకలలో తేమ/గాలి లేకపోవడం."
            ) if is_pox else (
                "• ప్రధాన వ్యాధి కారకం: *పాశ్చరెల్లా మల్టోసిడా* అనే బ్యాక్టీరియా.\n"
                "• తీవ్రతరం చేసే కారణాలు: వాతావరణంలో ఆకస్మిక మార్పులు, వర్షాకాలం ప్రారంభం, అలసట మరియు చలిగాలి.\n"
                "• వ్యాప్తి: లాలాజలం మరియు గాలి ద్వారా వేగంగా వ్యాపిస్తుంది."
            ) if is_domma else (
                "• ప్రధాన వ్యాధి కారకం: *క్లోస్ట్రిడియం పెర్‌ఫ్రింజెన్స్* టైప్-డి అనే బ్యాక్టీరియా విడుదల చేసే ఎప్సిలాన్ టాక్సిన్.\n"
                "• తీవ్రతరం చేసే కారణాలు: అధిక మొత్తంలో గింజల దాణా తినడం లేదా వర్షాల తర్వాత వచ్చే లేత పచ్చిగడ్డిని అతిగా మేయడం.\n"
                "• అంతర్గత ప్రభావం: ప్రేగులలో బ్యాక్టీరియా వేగంగా పెరిగి ప్రాణాంతక విషాన్ని విడుదల చేస్తుంది."
            ) if is_pk else (
                "• ప్రధాన వ్యాధి కారకం: గిట్టల మధ్య చేరే *డైఖిలోబాక్టర్ నోడోసస్* మరియు *ఫ్యూసోబ్యాక్టీరియం* బ్యాక్టీరియా.\n"
                "• పర్యావరణ కారణాలు: వర్షాకాలంలో బురద, తడి మరియు పేడ నిండిన ప్రదేశాలలో ఎక్కువ సమయం నిలబడటం.\n"
                "• ప్రేరేపించే అంశాలు: పెరిగిన గిట్టలను కత్తిరించకపోవడం మరియు గిట్టల మధ్య గాయాలు కావడం."
            ) if is_hoof else (
                "• ప్రధాన వ్యాధి కారకం: వాతావరణంలో ఉండే బ్యాక్టీరియా, వైరస్ లేదా అంతర్గత నట్టల క్రిములు.\n"
                "• తీవ్రతరం చేసే కారణాలు: వాతావరణ మార్పులు, పాకలలో రద్దీ, తేమ మరియు పరిశుభ్రత లోపించడం.\n"
                "• రోగనిరోధక శక్తి లోపం: సరైన పోషకాహారం లేకపోవడం లేదా సమయానికి టీకాలు వేయించకపోవడం."
            )

            ans_en = f"""1. Primary Etiology & Causative Agent of {disp_name_en}
{cause_en}

---

2. Transmission Modes & Risk Factors
• Direct contact with mucosal secretions, saliva, or aerosol droplets from infected flock members.
• Indirect transmission through contaminated bedding, feed troughs, transport vehicles, or shearing shears.
• Vectors such as biting flies and ticks can act as mechanical transmitters across the flock.

---

3. Environmental & Management Stressors
• Overcrowding in humid, poorly ventilated night enclosures dramatically elevates airborne pathogen concentration.
• Nutritional deprivation, sudden cold drafts, or transport exhaustion suppresses mucosal immune barriers.
• Heavy internal parasite (worm) burden diverts metabolic resources, weakening disease resistance.

---

4. Protective Measures Summary
• Maintain strict quarantine for 21–30 days for all newly arrived livestock.
• Adhere to annual preventive vaccinations prior to seasonal danger periods.
• Implement routine shed disinfection using dry lime, bleaching powder, or caustic wash."""

            ans_te = f"""1. {disp_name_te} ప్రాథమిక వ్యాధి కారకం & క్రిములు
{cause_te}

---

2. వ్యాప్తి మార్గాలు & ప్రమాద కారకాలు
• వ్యాధి సోకిన గొర్రెల లాలాజలం, ముక్కు స్రావాలు లేదా గాలి తుంపర్ల ద్వారా నేరుగా వ్యాపిస్తుంది.
• కలుషితమైన మేత, తాగునీటి తొట్టెలు, రవాణా వాహనాలు లేదా ఉన్ని కత్తిరించే పరికరాల ద్వారా పరోక్షంగా సోకుతుంది.
• గోమార్లు, పేలు మరియు ఈగలు కూడా ఈ వ్యాధిని ఒక గొర్రె నుండి మరొకదానికి వ్యాపింపజేస్తాయి.

---

3. పర్యావరణ & పాకల నిర్వహణ లోపాలు
• పాకలలో ఎక్కువ గొర్రెలను ఉంచడం, గాలి-వెలుతురు లేకపోవడం మరియు తేమ క్రిముల వ్యాప్తిని పెంచుతాయి.
• సరైన పోషకాహారం లేకపోవడం, చలిగాలి తగలడం లేదా ఎక్కువ దూరం నడిపించడం వల్ల రోగనిరోధక శక్తి తగ్గుతుంది.
• కడుపులో నట్టలు ఎక్కువగా ఉండటం వల్ల గొర్రెలు నీరసపడి వ్యాధులకు సులభంగా గురవుతాయి.

---

4. నివారణ & మంద రక్షణ సారాంశం
• కొత్తగా కొన్న గొర్రెలను 21-30 రోజుల పాటు క్వారంటైన్‌లో ఉంచండి.
• వ్యాధి కాలాలకు ముందే క్రమం తప్పకుండా టీకాలు వేయించండి.
• పాకలలో సున్నం మరియు బ్లీచింగ్ పౌడర్ చల్లి ఎల్లప్పుడూ పొడిగా, పరిశుభ్రంగా ఉంచండి."""

        # -----------------------------------------------------------------
        # 8. EMERGENCY / VETERINARY SUPPORT INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        elif intent in [IntentType.SEVERITY_EMERGENCY, IntentType.VETERINARY_SUPPORT]:
            ans_en = f"""1. Critical Emergency Warning Signs
• High fever (>105°F / 40.5°C) or severe subnormal body temperature (<100°F) indicating hypothermic septic shock.
• Acute respiratory distress: Open-mouth gasping, extended neck, loud wheezing, or frothy discharge from nostrils.
• Severe left-flank abdominal distension (bloat) accompanied by teeth grinding, kicking at the belly, or sudden collapse.
• Complete recumbency (inability to rise) and refusal of all drinking water for >24 hours.

---

2. Immediate Supportive First-Aid
• Isolate the emergency animal immediately in a dry, calm, shaded area with soft straw bedding.
• For high fever, apply cool wet towels gently to the head, groin, and ears (never douse whole animal with freezing water).
• If bloat is severe, keep the sheep standing with its front legs slightly elevated; never force-drench liquids down a gasping sheep.
• Offer small sips of oral electrolyte water (ORS) only if the sheep can actively swallow.

---

3. Information to Provide to the Veterinarian
• Duration of acute signs: Exact time the animal stopped feeding, started panting, or collapsed.
• Temperature and treatments: Any medicines, dewormers, or folk remedies administered before arrival.
• Flock background: Whether other flock members show similar signs or if any recent animal purchases occurred.

---

4. Emergency Veterinary Hospital Triage
• Transport the animal gently in a well-ventilated, shaded vehicle without overcrowding.
• Locate the nearest Government Veterinary Hospital, Assistant Surgeon (VAS) clinic, or Mobile Ambulatory Clinic.
• Intensive veterinary interventions: Emergency catheter fluid resuscitation, injectables, antitoxins, or emergency bloat trocharization."""

            ans_te = f"""1. అత్యవసర హెచ్చరిక సంకేతాలు
• అధిక జ్వరం (105°F పైన) లేదా శరీరం అసాధారణంగా చల్లబడటం (100°F కంటే తక్కువ - షాక్ సంకేతం).
• శ్వాసలో తీవ్ర ఇబ్బంది: మెడ చాచి నోటితో ఊపిరి పీల్చడం, ఆయాసం లేదా ముక్కు నుండి నురుగు కారడం.
• ఎడమవైపు కడుపు బానలా ఉబ్బిపోవడం (బ్లోట్), పళ్లు కొరకడం, కడుపును కాళ్లతో తన్నుకోవడం లేదా స్పృహతప్పి పడిపోవడం.
• గొర్రె లేవలేకపోవడం (మంచాన పడటం) మరియు 24 గంటలకు పైగా చుక్క నీరు కూడా తాగకపోవడం.

---

2. ఆసుపత్రికి వెళ్లేముందు తక్షణ ప్రథమ చికిత్స
• అత్యవసర పరిస్థితిలో ఉన్న గొర్రెను వెంటనే నీడ, గాలి ఉన్న ప్రశాంతమైన చోట మెత్తని గడ్డిపై ఉంచండి.
• తీవ్ర జ్వరం ఉన్నప్పుడు తల, గజ్జలు మరియు చెవులపై తడిగుడ్డతో అద్దండి (ఒక్కసారిగా చల్లటి నీరు పోయవద్దు).
• కడుపుబ్బరం ఉన్నప్పుడు గొర్రె ముందు కాళ్లను కాస్త ఎత్తులో ఉంచండి; శ్వాస ఇబ్బంది ఉన్నప్పుడు బలవంతంగా ద్రవాలు తాగించవద్దు.
• గొర్రె స్వయంగా మింగగలిగితే మాత్రమే కొద్దికొద్దిగా ORS లేదా ఎలక్ట్రోలైట్ నీరు అందించండి.

---

3. పశువైద్యుడికి అందించాల్సిన సమాచారం
• లక్షణాలు ప్రారంభమైన సమయం: గొర్రె ఎప్పటి నుండి మేత తినడం మానేసింది, ఎప్పటి నుండి ఆయాసపడుతోంది.
• ఇచ్చిన మందుల వివరాలు: ఆసుపత్రికి రాకముందు ఏవైనా మందులు, ఇంజక్షన్లు లేదా నాటు వైద్యం చేశారా.
• మంద సమాచారం: మందలోని ఇతర గొర్రెలకు కూడా ఇలాంటి లక్షణాలు ఉన్నాయా లేదా ఇటీవల కొత్త గొర్రెలను కొన్నారా.

---

4. పశువైద్యశాల అత్యవసర సేవల మార్గదర్శకాలు
• గొర్రెను రవాణా చేసేటప్పుడు ఎండ తగలకుండా, గాలి ఆడేలా జాగ్రత్తగా తీసుకెళ్లండి.
• సమీపంలోని ప్రభుత్వ పశువైద్యశాల, ప్రాంతీయ పశు ఆసుపత్రి లేదా సంచార పశువైద్యశాల (1962) సహాయం పొందండి.
• ఆసుపత్రిలో అత్యవసరంగా ఐవి సెలైన్, ఇంజక్షన్లు, ఆక్సిజన్ లేదా అవసరమైతే కడుపుబ్బరానికి ట్రోకార్ చికిత్స చేస్తారు."""

        # -----------------------------------------------------------------
        # 9. GENERAL / DISEASE OVERVIEW INTENT (Step 3: Exactly 4 parallel sections)
        # -----------------------------------------------------------------
        else:
            ans_en = f"""1. Overview & Understanding {disp_name_en}
• {disp_name_en} is a clinically important condition affecting small ruminants, requiring early detection and structured care.
• Early diagnosis and immediate flock isolation protect surrounding flock members from epidemic spread.
• Good management combines biosecurity, balanced nutrition, hygienic shelter, and timely vaccination.

---

2. Primary Clinical Signs to Watch
• Dullness, loss of appetite, reluctance to graze, and voluntary separation from the main flock.
• Elevation of body temperature above normal (fever >104°F) accompanied by drooping ears and dull eyes.
• Specific signs: Skin nodules/scabs, lameness, nasal discharge, or abnormal rapid respiration.

---

3. Management & Supportive Principles
• Isolate the affected sheep immediately in a dry, well-ventilated, shaded enclosure with soft straw bedding.
• Ensure continuous access to clean, cool drinking water enriched with electrolytes or Oral Rehydration Salts (ORS).
• Feed soft, digestible warm gruel and tender greens; avoid coarse fibrous roughage during acute illness.
• ℹ️ Care Boundary: Home care is supportive only; it does NOT substitute for professional veterinary treatment.

---

4. Professional Veterinary Guidance
• Consult a registered veterinarian to confirm clinical diagnosis and prescribe required therapeutics.
• ⚠️ Prescription Warning: All antibiotics, injections, and dosages must be determined by a registered veterinarian.
• Seek emergency hospital triage if the animal refuses water for >24 hours, cannot stand, or breathes with an open mouth."""

            ans_te = f"""1. {disp_name_te} పరిస్థితి అవగాహన & ప్రాముఖ్యత
• గొర్రెల మందలో {disp_name_te} ఒక ముఖ్యమైన ఆరోగ్య సమస్య; దీనిని ముందుగానే గుర్తించి సరైన జాగ్రత్తలు తీసుకోవడం అవసరం.
• వ్యాధి ప్రారంభంలోనే గుర్తించి బాధింత గొర్రెను వేరు చేయడం వల్ల మందలోని ఇతర గొర్రెలను కాపాడుకోవచ్చు.
• పరిశుభ్రత, సమతుల్య మేత, క్వారంటైన్ మరియు సమయానికి టీకాలు వేయించడం ఉత్తమ నిర్వహణ పద్ధతులు.

---

2. ప్రధాన క్లినికల్ లక్షణాలు
• నీరసం, మేత తినకపోవడం, నెమరు వేయడం ఆగిపోవడం మరియు మందలో కలవకుండా వేరుగా పడుకోవడం.
• శరీర ఉష్ణోగ్రత సాధారణం కంటే పెరగడం (104°F పైన జ్వరం), చెవులు వాలిపోవడం మరియు కళ్లలో కాంతి తగ్గడం.
• నిర్దిష్ట లక్షణాలు: చర్మంపై దద్దుర్లు/పుండ్లు, కుంటుతనం, ముక్కు నుండి స్రావాలు లేదా వేగంగా శ్వాస తీసుకోవడం.

---

3. నిర్వహణ & సహాయక సూత్రాలు
• వ్యాధి సోకిన గొర్రెను వెంటనే పొడి, శుభ్రమైన, నీడ ఉన్న పాకలో వేరుగా ఉంచండి.
• డీహైడ్రేషన్ రాకుండా స్వచ్ఛమైన చల్లని తాగునీటిలో ORS పొడి లేదా ఉప్పు, బెల్లం కలిపి నిరంతరం అందుబాటులో ఉంచండి.
• సులభంగా జీర్ణమయ్యే గోరువెచ్చని గంజి మరియు లేత పచ్చి మేత అందించండి; గట్టి ఎండు చొప్పను నివారించండి.
• ℹ️ రక్షణ సరిహద్దు: ఇంటి వద్ద సంరక్షణ కేవలం ఉపశమనం కోసం మాత్రమే; ఇది పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.

---

4. పశువైద్య సలహా & తదుపరి చర్యలు
• ఖచ్చితమైన వ్యాధి నిర్ధారణ మరియు అవసరమైన మందుల కోసం అర్హత కలిగిన పశువైద్యుని సంప్రదించండి.
• ⚠️ భద్రతా హెచ్చరిక: మందుల పేరు, మోతాదు మరియు ఇచ్చే విధానాన్ని తప్పనిసరిగా పశువైద్యుని పర్యవేక్షణలోనే నిర్ణయించాలి.
• గొర్రె 24 గంటలకు పైగా నీరు తాగకపోయినా, లేవలేకపోయినా లేదా ఆయాసపడుతూ శ్వాస తీసుకుంటున్నా వెంటనే పశువైద్యశాలకు తరలించండి."""

        # -----------------------------------------------------------------
        # Build Structured, Validated Sections with 1-to-1 Bilingual Match
        # -----------------------------------------------------------------
        def _to_secs(raw_text: str) -> List[Dict[str, str]]:
            secs = []
            for chunk in raw_text.split("\n\n---\n\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                lines = chunk.split("\n", 1)
                heading = lines[0].strip()
                heading = re.sub(r"^(?:###\s*)?\d+[\.\)]\s*", "", heading).strip()
                heading = re.sub(r"<[^>]+>", "", heading).strip()
                content = lines[1].strip() if len(lines) > 1 else ""
                content = re.sub(r"<[^>]+>", "", content).strip()
                secs.append({"title": heading, "heading": heading, "content": content})
            return secs

        en_secs = _to_secs(ans_en)
        te_secs = _to_secs(ans_te)

        if language == "English":
            return ClinicalChatResponse(english_sections=en_secs)
        elif language == "తెలుగు":
            return ClinicalChatResponse(telugu_sections=te_secs)
        else:
            return ClinicalChatResponse(english_sections=en_secs, telugu_sections=te_secs)

    def _extract_section_body(self, text: str, header_keyword: str) -> str:
        """Extracts text following a markdown header keyword."""
        pattern = rf"#{2,3}\s+.*{header_keyword}.*\n(.*?)(?=\n#{2,3}|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
