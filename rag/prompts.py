"""
Prompt templates and system guidelines for grounded veterinary generation.
Enforces clinical safety, strict factual grounding, and natural bilingual (English + Telugu) output.
"""
from typing import Optional, Dict, Any, List


VETERINARY_SYSTEM_PROMPT = """You are an expert small-ruminant veterinary health knowledge assistant assisting Indian sheep farmers, extension workers, and field veterinarians.

CRITICAL INSTRUCTIONS & SAFETY BOUNDARIES:
1. FACTUAL GROUNDING: Base your answers strictly on the provided RETRIEVED VETERINARY KNOWLEDGE. Do not hallucinate facts.
2. NO INVENTED MEDICINES OR DOSAGES: Only mention medication classes and principles present in the retrieved knowledge. Never guess or fabricate exact drug dosages.
3. CLEAR CARE BOUNDARIES: You must strictly separate:
   - HOME / SUPPORTIVE CARE (bedding, hydration, soft forage, mild salt/potassium permanganate wash)
   - VETERINARY CLINICAL TREATMENT (prescription antibiotics, anti-inflammatories, vaccinations)
   - EMERGENCY CARE (immediate hospital triage)
4. DO NOT PRESENT UNPROVEN REMEDIES: Do not present folk practices or unverified remedies as clinical medicine.
5. CONFIDENCE & UNCERTAINTY AWARENESS:
   - When classifier confidence is HIGH (>= 60%), explain the predicted condition directly while reminding the user this is an AI visual screening.
   - When classifier confidence is MEDIUM (45% - 60%), explicitly state that the visual prediction is uncertain, discuss the top candidates, and recommend professional veterinary clinical examination.
   - When classifier confidence is LOW (< 45%), explicitly state that the model cannot confirm a diagnosis from the photograph, provide general safe sheep-care guidance, and advise in-person veterinary review or uploading a clearer, well-lit photo.
6. BILINGUAL DELIVERY: Provide natural, fluent Telugu (తెలుగు) alongside English for every section so rural sheep farmers can easily understand and act on the advice. Avoid rigid word-by-word machine translation.
"""

def build_prediction_report_prompt(
    prediction_context: dict,
    retrieved_docs_text: str,
    language: str = "English + తెలుగు"
) -> str:
    """
    Constructs the prompt for generating the comprehensive 11-section grounded health report.
    """
    disease = prediction_context.get("disease", "Unknown")
    confidence_pct = prediction_context.get("confidence_percent", 0.0)
    conf_level = prediction_context.get("confidence_level", "medium")
    is_uncertain = prediction_context.get("is_uncertain", False)
    gender = prediction_context.get("gender", "Adult Ewe / Ram")
    weight = prediction_context.get("estimated_weight", "35 - 45 kg")
    region = prediction_context.get("affected_region", "General body condition")
    
    top_k_str = ""
    for item in prediction_context.get("top_k", []):
        top_k_str += f"- Rank {item.get('rank')}: {item.get('disease')} ({item.get('confidence_percent')}%) [Telugu: {item.get('telugu_name')}]\n"

    prompt = f"""
=========================================
PREDICTION CONTEXT (FROM IMAGE SCREENING):
=========================================
- Animal: Sheep
- Detected Condition / Disease: {disease}
- Model Softmax Confidence: {confidence_pct}% ({conf_level.upper()} CONFIDENCE)
- Prediction Uncertainty Flag: {"UNCERTAIN PREDICTION — VETERINARY REVIEW RECOMMENDED" if is_uncertain else "HIGH CONFIDENCE SCREENING"}
- Differential Top Candidates:
{top_k_str}
- Estimated Weight: {weight}
- Gender Screening: {gender}
- Affected Body Region: {region}

=========================================
RETRIEVED TRUSTED VETERINARY KNOWLEDGE:
=========================================
{retrieved_docs_text}

=========================================
REQUESTED LANGUAGE: {language}
=========================================

Generate a comprehensive, grounded clinical explanation formatted EXACTLY into the following 11 sections. Provide natural parallel English and Telugu (తెలుగు) for each section:

1. 🩺 Detected Condition / గుర్తించబడిన పరిస్థితి
2. 🎯 Model Confidence & Uncertainty Assessment / మోడల్ విశ్వసనీయత & అంచనా
3. 📋 What it Means / దీని అర్థం ఏమిటి
4. 🔍 Common Symptoms & Signs / సాధారణ లక్షణాలు & సంకేతాలు
5. 💊 Treatment & Medicine Information / పశువైద్య చికిత్స & మందుల వివరాలు (Note: State veterinary prescription requirements)
6. 🏠 Safe Supportive & Home Care / సురక్షితమైన సహాయక & ఇంటి వద్ద సంరక్షణ (Clearly separated from prescription treatment)
7. 🛡️ Prevention & Vaccination / నివారణ చర్యలు & టీకాలు
8. 🥬 Feeding & Hydration / మేత & తాగునీటి నిర్వహణ
9. ⚠️ Emergency Warning Signs / అత్యవసర హెచ్చరిక సంకేతాలు
10. 👨‍⚕️ When to Contact a Veterinarian / పశువైద్యుడిని ఎప్పుడు సంప్రదించాలి
11. 📚 Sources & Grounded References / జ్ఞాన వనరులు (Include references to ICAR, IVRI, DAHD, FAO)
"""
    return prompt

def build_chat_prompt(
    user_question: str,
    prediction_context: Optional[dict],
    retrieved_docs_text: str,
    language: str = "English + తెలుగు",
    intent: Optional[str] = None,
    disease_info: Optional[dict] = None
) -> str:
    """
    Constructs a question-aware prompt for answering farmer questions.
    Instructs the LLM to answer ONLY what the user asked, using intent-specific guidance,
    strict factual grounding, and structured bilingual formatting without dumping entire documents.
    """
    pred_summary = "No previous image screening available (General Veterinary Knowledge Mode)."
    active_disease = "Sheep Health"
    if disease_info:
        active_disease = disease_info.get("short_name", disease_info.get("disease", "Sheep Health"))
    elif prediction_context:
        active_disease = prediction_context.get("short_name", prediction_context.get("disease", "Sheep Health"))

    if prediction_context:
        disease = prediction_context.get("disease", active_disease)
        conf = prediction_context.get("confidence_percent", 0.0)
        is_unc = prediction_context.get("is_uncertain", False)
        pred_summary = (
            f"Active Detected Condition: {disease} (Confidence: {conf}%"
            f"{', UNCERTAIN' if is_unc else ''}). "
            f"Interpreting user question in context of this detected condition."
        )
    elif active_disease != "Sheep Health":
        pred_summary = f"Active Disease Context: {active_disease}."

    intent_directives = {
        "HOME_CARE": """INTENT DIRECTIVE — HOME CARE:
- Answer ONLY practical, safe supportive home care for sheep.
- Focus on: isolation, clean/dry shelter, hydration (ORS/clean water), easily digestible soft feed (gruel/tender grass), basic wound hygiene, monitoring, and safety boundary (when to contact vet).
- DO NOT provide prescription drugs, antibiotics, or complete disease textbooks.
- Emphasize: Home supportive care provides palliative comfort and does NOT replace professional veterinary medical therapy.""",

        "MEDICINE_TREATMENT": """INTENT DIRECTIVE — MEDICINE & TREATMENT:
- Answer ONLY veterinary medical treatment principles and medicine information available in the retrieved knowledge base.
- Focus on: supportive vs disease-specific therapy, broad-spectrum antibiotics to control secondary infection, anti-inflammatory/antipyretics under veterinary guidance, and fluid therapy.
- CRITICAL SAFETY: State that exact medicine choice, dose, route, and duration must be determined and administered exclusively by a registered veterinarian. NEVER invent drugs or dosages.
- DO NOT dump unrelated causes, full symptoms list, or home care textbooks.""",

        "SYMPTOMS": """INTENT DIRECTIVE — SYMPTOMS:
- Return ONLY symptoms and clinical signs supported by the retrieved knowledge base (e.g. fever, skin lesions, nodules, weakness, nasal/ocular discharge, condition-specific signs).
- DO NOT provide treatment, causes, prevention, or vaccination unless specifically asked.""",

        "CAUSES": """INTENT DIRECTIVE — CAUSES:
- Explain ONLY the causes, underlying etiology, pathogen, and transmission risk factors supported by the retrieved knowledge base.
- DO NOT dump treatment or vaccination schedules.""",

        "PREVENTION": """INTENT DIRECTIVE — PREVENTION:
- Focus ONLY on prevention, biosecurity, pen sanitation, quarantine for newly purchased animals, and flock protective management.
- DO NOT dump detailed medical treatments or clinical symptoms.""",

        "VACCINATION": """INTENT DIRECTIVE — VACCINATION:
- Answer specifically about vaccination schedule, timing, and rules from the verified knowledge base.
- State: 'Vaccination timing depends on the local veterinary vaccination programme and disease risk. Please confirm the schedule with the local veterinary department/registered veterinarian.'
- DO NOT fabricate dates or dosages.""",

        "FEEDING_NUTRITION": """INTENT DIRECTIVE — FEEDING & NUTRITION:
- Answer ONLY about suitable feed (dry roughage 60-70%, green fodder 20-30%), clean water, digestive precautions (preventing bloat/acidosis/enterotoxemia), and supportive feeding for sick sheep (warm gruel, soft leaves).
- DO NOT dump disease medical treatment.""",

        "TRANSMISSION": """INTENT DIRECTIVE — TRANSMISSION & SPREAD:
- Answer ONLY about contagion level, transmission routes (direct contact, aerosol, scabs, biting flies), and flock spread prevention.""",

        "SEVERITY_EMERGENCY": """INTENT DIRECTIVE — EMERGENCY & WARNING SIGNS:
- Give clear emergency warning signs (fever >105°F or subnormal temp, recumbency, labored open-mouth breathing, complete refusal of water for >24 hrs, severe bloat).
- Prioritize animal safety and recommend immediate professional veterinary hospital triage.""",

        "DISEASE_OVERVIEW": """INTENT DIRECTIVE — DISEASE OVERVIEW:
- Provide a concise, clear overview of the disease (definition, cause, hallmark signs, management approach, and vet guidance). Keep it structured and easy for a farmer to understand."""
    }

    directive = intent_directives.get(intent or "", "Answer ONLY what the user asked concisely, practically, and accurately based on the retrieved knowledge.")

    prompt = f"""
=========================================
ACTIVE CLINICAL CONTEXT:
=========================================
{pred_summary}

=========================================
RETRIEVED VETERINARY KNOWLEDGE:
=========================================
{retrieved_docs_text}

=========================================
FARMER / USER QUESTION:
=========================================
"{user_question}"

=========================================
REQUESTED LANGUAGE: {language}
=========================================

{directive}

CORE GROUNDING RULES:
1. ANSWER THE EXACT QUESTION:
   - Answer ONLY what the user asked. Do NOT dump the entire retrieved document.
   - Do NOT return unrelated sections (e.g. do not return causes or treatment when asked about home care; do not return treatment when asked about symptoms).
   - If the user asked a compound question (e.g. home care AND when to visit a vet), provide clearly separated sections covering both parts.

2. MEDICAL SAFETY:
   - Base all advice strictly on the retrieved knowledge base.
   - NEVER invent veterinary facts, medicines, dosages, or vaccination schedules.
   - If prescription medicines are discussed, state clearly: "Exact medicine choice, dose, route and duration must be determined by a registered veterinarian."
   - If supportive home care is discussed, state clearly: "Home care does not replace veterinary treatment."

3. LANGUAGE ISOLATION:
   - The "english" list must contain ONLY English text (zero Telugu characters).
   - The "telugu" list must contain ONLY natural, fluent Telugu text (తెలుగు).
   - The number of sections, headings, and meaning in "english" and "telugu" must match 1-to-1 in the exact same order.

4. PURE MARKDOWN ONLY (NO RAW HTML):
   - Output clean Markdown only. NEVER output raw HTML tags (no <div>, <ul>, <li>, <span>, <a>, <p>).

RESPONSE FORMAT:
Return your response as a strictly valid JSON object with matching parallel sections:
{{
  "english": [
    {{
      "title": "Section Title in English",
      "content": "Clear, practical, focused answer grounded strictly in retrieved knowledge..."
    }}
  ],
  "telugu": [
    {{
      "title": "సహజమైన తెలుగు శీర్షిక",
      "content": "రైతులకు సులభంగా అర్థమయ్యే స్వచ్ఛమైన తెలుగు వివరణ..."
    }}
  ]
}}

CRITICAL: The number of items in "english" and "telugu" MUST BE EXACTLY EQUAL. Each english[i] section must correspond directly to telugu[i].
"""
    return prompt

