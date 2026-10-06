"""
Intent Detection and Disease Entity Extraction for Sheep Veterinary RAG.
Classifies user questions into specific veterinary intents, extracts disease context,
and provides targeted retrieval keywords.
"""
import re
from typing import Dict, Any, List, Optional, Tuple

class IntentType:
    HOME_CARE = "HOME_CARE"
    MEDICINE_TREATMENT = "MEDICINE_TREATMENT"
    SYMPTOMS = "SYMPTOMS"
    CAUSES = "CAUSES"
    PREVENTION = "PREVENTION"
    VACCINATION = "VACCINATION"
    FEEDING_NUTRITION = "FEEDING_NUTRITION"
    TRANSMISSION = "TRANSMISSION"
    DIAGNOSIS = "DIAGNOSIS"
    SEVERITY_EMERGENCY = "SEVERITY_EMERGENCY"
    DISEASE_OVERVIEW = "DISEASE_OVERVIEW"
    GENERAL_SHEEP_HEALTH = "GENERAL_SHEEP_HEALTH"
    OTHER_SHEEP_VETERINARY = "OTHER_SHEEP_VETERINARY"
    IRRELEVANT = "IRRELEVANT"


# Known disease aliases for extraction
DISEASE_MAPPINGS = [
    {
        "canonical": "Sheep Pox",
        "short_name": "Sheep Pox",
        "telugu_name": "గొర్రెల పొంగు వ్యాధి",
        "patterns": [r"\bsheep\s*pox\b", r"\bpox\b", r"\bsppv\b", r"మశూచి", r"పొంగు", r"వరికోల"]
    },
    {
        "canonical": "Domma / Haemorrhagic Septicaemia (HS)",
        "short_name": "Domma",
        "telugu_name": "దొమ్మ / హెమరేజిక్ సెప్టిసీమియా",
        "patterns": [r"\bdomma\b", r"\bhaemorrhagic\s*septicaemia\b", r"\bhemorrhagic\s*septicemia\b", r"\bhs\b", r"దొమ్మ", r"గొంతువాపు"]
    },
    {
        "canonical": "Pulpy Kidney Disease / Enterotoxemia",
        "short_name": "Pulpy Kidney",
        "telugu_name": "పల్పి కిడ్నీ వ్యాధి / ఎంటరోటాక్సిమియా",
        "patterns": [r"\bpulpy\s*kidney\b", r"\benterotoxemia\b", r"\benterotoxaemia\b", r"\bet\b", r"పల్పి\s*కిడ్నీ", r"ఎంటరోటాక్సిమియా"]
    },
    {
        "canonical": "Hoof Wounds / Lameness",
        "short_name": "Hoof Wounds",
        "telugu_name": "గిట్టల గాయాలు / కుంటువ్యాధి",
        "patterns": [r"\bhoof\s*wound[s]?\b", r"\bfoot\s*rot\b", r"\bhoof\s*rot\b", r"\blameness\b", r"\blimping\b", r"గిట్టల", r"కుంటు"]
    },
    {
        "canonical": "Excessive Salivation (Clinical Sign)",
        "short_name": "Excessive Salivation",
        "telugu_name": "అధిక లాలాజలం కారడం",
        "patterns": [r"\bexcessive\s*salivation\b", r"\bsalivation\b", r"\bsaliva\b", r"\bdrooling\b", r"లాలాజలం", r"నోటి\s*నుండి\s*నీరు"]
    },
    {
        "canonical": "Foot and Mouth Disease (FMD)",
        "short_name": "Foot and Mouth",
        "telugu_name": "గాలికుంటు వ్యాధి",
        "patterns": [r"\bfoot\s*and\s*mouth\b", r"\bfmd\b", r"గాలికుంటు"]
    },
    {
        "canonical": "Peste des Petits Ruminants (PPR)",
        "short_name": "PPR",
        "telugu_name": "పిపిఆర్ వ్యాధి",
        "patterns": [r"\bppr\b", r"\bpeste\s*des\s*petits\b", r"పిపిఆర్"]
    },
    {
        "canonical": "Healthy Sheep",
        "short_name": "Healthy",
        "telugu_name": "ఆరోగ్యకరమైన గొర్రె",
        "patterns": [r"\bhealthy\s*sheep\b", r"\bnormal\s*sheep\b", r"ఆరోగ్య"]
    }
]


NON_DOMAIN_PATTERNS = [
    r"\b(joke|jokes|tell me a joke|funny|humor|comedy|meme|riddle)\b",
    r"(జోక్|హాస్యం|నవ్వుల|కథ)",
    r"\b(weather|temperature today|forecast|rain today|climate today|monsoon forecast)\b",
    r"(వాతావరణం|వర్షం|ఎండ)",
    r"\b(cricket|football|soccer|ipl|world cup|virat kohli|dhoni|sachin|messi|ronaldo)\b",
    r"\b(python|java|javascript|c\+\+|html|css|sql|rust|golang|php|ruby|typescript)\b",
    r"\b(programming|code|coding|software|algorithm|developer|github|debugging)\b",
    r"\b(elon musk|bill gates|mark zuckerberg|ambani|adani|trump|modi|biden)\b",
    r"\b(movie|cinema|actor|actress|hollywood|bollywood|tollywood|song|lyrics|dance)\b",
    r"\b(capital of|president of|prime minister of|currency of|population of)\b",
    r"\b(who is\b|who was\b|who won\b|write a poem|write a story|write a program|write an essay|write my resume)\b",
    r"\b(cook biryani|recipe for|how to make biryani|baking|cooking|pancake)\b",
    r"\b(solve this math|algebra|calculus|quantum|astrophysics)\b",
    r"\b(car\b|automobile|iphone|android|laptop|gadget|stock market|cryptocurrency|bitcoin)\b",
    r"\b(flight|train ticket|hotel booking|railway)\b",
]

SHEEP_DOMAIN_ANCHORS = [
    r"\b(sheep|lamb|ewe|ram|flock|livestock|ruminant|capripox|domma|mutton|wool)\b",
    r"\b(veterinary|vet|veterinarian|animal husbandry|icar|ivri|cswri|dahd)\b",
    r"(గొర్రె|గొర్రెల|పొట్టేలు|మేక|మంద|జీవాలు|పశువైద్య|టీకా|నట్టలు)",
    r"\b(sick|ill|hoof|pox|fever|bloat|acidosis|salivation|wound|infection|graze|grazing|fodder)\b"
]


def is_out_of_domain(query: str) -> bool:
    """Checks if query is strictly out-of-domain."""
    q = query.lower().strip()
    if not q:
        return True
    # Check non-domain topic signatures
    for pat in NON_DOMAIN_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    return False


def detect_intents(question: str) -> Tuple[str, List[str]]:
    """
    Classifies the user question into primary intent and a list of all matched intents.
    Supports English, Telugu script, and transliterated Telugu.
    """
    q = question.lower().strip()

    if not q or is_out_of_domain(q):
        return IntentType.IRRELEVANT, [IntentType.IRRELEVANT]

    matched_intents: List[str] = []

    # 1. HOME_CARE patterns
    home_patterns = [
        r"\b(home\s*care|home\s*remed|at\s*home|supportive\s*care|what\s*(can|should)\s*i\s*do\s*at\s*home|nursing\s*care|first\s*aid\s*at\s*home)\b",
        r"(ఇంటి\s*వద్ద|ఇంటి\s*సంరక్షణ|ఇంటి\s*వైద్యం|గృహ\s*సంరక్షణ|సహాయక\s*చర్యలు)",
        r"\b(intlo|inti\s*daggara|inti\s*vaddhana|home\s*care\s*ela|care\s*ela)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in home_patterns):
        matched_intents.append(IntentType.HOME_CARE)

    # 2. MEDICINE_TREATMENT patterns
    med_patterns = [
        r"\b(medicine|medicines|treatment|cure|drug|drugs|antibiotic|antibiotics|injection|dosage|dose|tablet|ointment|therapy|medication|clinical\s*treatment|how\s*to\s*treat)\b",
        r"(మందులు|చికిత్స|ఇంజెక్షన్|నయం|ఔషధం|టాబ్లెట్)",
        r"\b(mandulu|chikitsa|medicine\s*enti|treatment\s*enti|ela\s*tagginchali|treatment\s*ela)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in med_patterns):
        matched_intents.append(IntentType.MEDICINE_TREATMENT)

    # 3. SYMPTOMS patterns
    symptom_patterns = [
        r"\b(symptom[s]?|sign[s]?|clinical\s*sign[s]?|how\s*to\s*identify|how\s*to\s*recogni[sz]e|what\s*happens\s*in|manifestation|look\s*like)\b",
        r"(లక్షణాలు|సంకేతాలు|ఎలా\s*గుర్తించాలి)",
        r"\b(lakshanalu|symptoms\s*enti|ela\s*untundi|gurthinchali)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in symptom_patterns):
        matched_intents.append(IntentType.SYMPTOMS)

    # 4. CAUSES patterns
    cause_patterns = [
        r"\b(cause[s]?|why\s*does|why\s*do|reason[s]?|etiology|how\s*do\s*sheep\s*get|origin|how\s*does\s*it\s*occur)\b",
        r"(కారణం|కారణాలు|ఎందుకు\s*వస్తుంది|ఎలా\s*వస్తుంది)",
        r"\b(enduku\s*vastundi|karanam|karanalu|ela\s*vastundi)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in cause_patterns):
        matched_intents.append(IntentType.CAUSES)

    # 5. PREVENTION patterns (separate from vaccination if specifically about biosecurity/avoiding)
    prev_patterns = [
        r"\b(prevent|prevention|preventative|preventive|protect|biosecurity|quarantine|stop\s*spread|how\s*to\s*avoid)\b",
        r"(నివారణ|రాకుండా|నివారించడం|రక్షణ|పాకల\s*పరిశుభ్రత)",
        r"\b(nivarana|rakunda|ela\s*aapali|ela\s*rakshinchali)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in prev_patterns):
        matched_intents.append(IntentType.PREVENTION)

    # 6. VACCINATION patterns
    vax_patterns = [
        r"\b(vaccin[e|es|ation|ate]?|immuniz|shot[s]?|calendar|schedule)\b",
        r"(టీకా|టీకాలు|వ్యాక్సిన్|షెడ్యూల్|క్యాలెండర్)",
        r"\b(teeka|teekalu|vaccine\s*eppudu|eppudu\s*ivvali)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in vax_patterns):
        matched_intents.append(IntentType.VACCINATION)

    # 7. FEEDING_NUTRITION patterns
    feed_patterns = [
        r"\b(feed|feeding|food|nutrition|fodder|diet|graze|grazing|roughage|hay|grass|water|drinking|acidosis|bloat|concentrate|congee|gruel)\b",
        r"(మేత|ఆహారం|పోషణ|పచ్చి\s*మేత|ఎండు\s*మేత|నీరు|దాణా|కడుపుబ్బరం|జావ|గంజి)",
        r"\b(meta|aaharam|dana|gaddi|food\s*enti|em\s*pettali|em\s*tinali)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in feed_patterns):
        matched_intents.append(IntentType.FEEDING_NUTRITION)

    # 8. TRANSMISSION patterns
    trans_patterns = [
        r"\b(contagious|transmission|spread|spreads|infectious|pass\s*to\s*other|communicable|flock\s*transmission)\b",
        r"(వ్యాప్తి|సోకుతుందా|ఇతర\s*గొర్రెలకు|అంటువ్యాధి)",
        r"\b(contagious\s*aa|vyapti|spread\s*avtunda|sokutunda|pakkanunna)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in trans_patterns):
        matched_intents.append(IntentType.TRANSMISSION)

    # 9. SEVERITY_EMERGENCY patterns
    emerg_patterns = [
        r"\b(emergency|when\s*to\s*call|when\s*to\s*contact|when\s*should\s*i\s*(visit|see|call)|serious|how\s*serious|danger|critical|dying|cannot\s*stand|not\s*breathing|not\s*eating|urgent|triage)\b",
        r"(అత్యవసరం|వైద్యుడిని\s*ఎప్పుడు|తీవ్రత|ప్రమాదం|లేవలేకపోవడం|డాక్టర్\s*వద్దకు)",
        r"\b(emergency|doctor\s*eppudu|serious\s*aa|vet\s*eppudu|pramadama|chachipotunda)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in emerg_patterns):
        matched_intents.append(IntentType.SEVERITY_EMERGENCY)

    # 10. DIAGNOSIS patterns
    diag_patterns = [
        r"\b(diagnosis|differential|test|confirm|how\s*to\s*diagnose|clinical\s*test)\b",
        r"(వ్యాధి\s*నిర్ధారణ|పరీక్ష)",
        r"\b(diagnosis|pariksha|ela\s*telusukovali)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in diag_patterns):
        matched_intents.append(IntentType.DIAGNOSIS)

    # 11. DISEASE_OVERVIEW patterns
    overview_patterns = [
        r"\b(tell\s*me\s*about|what\s*is|overview|explain|information\s*about|details\s*of)\b",
        r"(గురించి\s*చెప్పండి|అంటే\s*ఏమిటి|సమగ్ర\s*వివరాలు|సమాచారం)",
        r"\b(gurinchi\s*cheppandi|enti|overview|vivaralu)\b"
    ]
    if any(re.search(pat, q, re.IGNORECASE) for pat in overview_patterns):
        matched_intents.append(IntentType.DISEASE_OVERVIEW)

    # If no specific intent matched, check if it's general sheep health or other
    if not matched_intents:
        if any(re.search(anchor, q, re.IGNORECASE) for anchor in SHEEP_DOMAIN_ANCHORS):
            return IntentType.GENERAL_SHEEP_HEALTH, [IntentType.GENERAL_SHEEP_HEALTH]
        return IntentType.IRRELEVANT, [IntentType.IRRELEVANT]

    # Resolve compound priority
    # If both HOME_CARE and SEVERITY_EMERGENCY are present, keep both!
    primary_intent = matched_intents[0]

    # Priority ordering for primary intent if multiple
    priority_order = [
        IntentType.HOME_CARE,
        IntentType.MEDICINE_TREATMENT,
        IntentType.VACCINATION,
        IntentType.FEEDING_NUTRITION,
        IntentType.SEVERITY_EMERGENCY,
        IntentType.SYMPTOMS,
        IntentType.CAUSES,
        IntentType.TRANSMISSION,
        IntentType.PREVENTION,
        IntentType.DIAGNOSIS,
        IntentType.DISEASE_OVERVIEW
    ]
    for p in priority_order:
        if p in matched_intents:
            primary_intent = p
            break

    return primary_intent, matched_intents


def extract_disease_entity(
    question: str,
    prediction_context: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, str]:
    """
    Extracts or resolves active sheep disease context from:
    1. Question text
    2. Prediction context (image screening)
    3. Recent chat history (follow-up context)
    """
    q = question.strip()

    # 1. Search in question directly
    for d in DISEASE_MAPPINGS:
        for pat in d["patterns"]:
            if re.search(pat, q, re.IGNORECASE):
                return {
                    "disease": d["canonical"],
                    "short_name": d["short_name"],
                    "telugu_name": d["telugu_name"]
                }

    # 2. Check prediction_context if present
    if prediction_context:
        d_name = prediction_context.get("short_name") or prediction_context.get("disease")
        if d_name:
            for d in DISEASE_MAPPINGS:
                if d["short_name"].lower() == str(d_name).lower() or d["canonical"].lower() == str(d_name).lower():
                    return {
                        "disease": d["canonical"],
                        "short_name": d["short_name"],
                        "telugu_name": prediction_context.get("telugu_name") or d["telugu_name"]
                    }
            return {
                "disease": str(d_name),
                "short_name": str(d_name),
                "telugu_name": prediction_context.get("telugu_name", "")
            }

    # 3. Check chat history backwards for recent disease mentions
    if chat_history:
        for msg in reversed(chat_history):
            content = msg.get("content", "")
            for d in DISEASE_MAPPINGS:
                for pat in d["patterns"]:
                    if re.search(pat, content, re.IGNORECASE):
                        return {
                            "disease": d["canonical"],
                            "short_name": d["short_name"],
                            "telugu_name": d["telugu_name"]
                        }

    # Default to General Sheep Health
    return {
        "disease": "Sheep Health",
        "short_name": "Sheep Health",
        "telugu_name": "గొర్రెల ఆరోగ్యం"
    }


def get_intent_keywords(intent: str) -> str:
    """Returns query expansion keywords for intent-guided vector retrieval."""
    keywords_map = {
        IntentType.HOME_CARE: "safe supportive home care isolation hydration feeding wound cleaning comfort shelter hygiene nursing",
        IntentType.MEDICINE_TREATMENT: "veterinary medical treatment medicines prescription antibiotics nsaids pain fever fluid therapy allopathic",
        IntentType.SYMPTOMS: "clinical symptoms signs fever skin lesions papules nodules weakness nasal discharge appetite appearance",
        IntentType.CAUSES: "underlying causes etiology virus pathogen risk factors transmission overcrowding environmental origin",
        IntentType.PREVENTION: "prevention biosecurity sanitation quarantine hygiene management protective measures flock safety",
        IntentType.VACCINATION: "vaccination vaccine schedule immunization timing doses booster cold chain annual calendar",
        IntentType.FEEDING_NUTRITION: "feeding nutrition balanced diet green fodder dry roughage gruel water minerals bloat acidosis",
        IntentType.TRANSMISSION: "contagious transmission spread contact vectors aerosol flock infection communicable",
        IntentType.DIAGNOSIS: "differential diagnosis examination clinical identification confirmation tests",
        IntentType.SEVERITY_EMERGENCY: "emergency warning signs critical severe triage danger veterinarian immediate hospital",
        IntentType.DISEASE_OVERVIEW: "overview description small ruminant condition flock health management",
        IntentType.GENERAL_SHEEP_HEALTH: "sheep health management welfare prevention care",
    }
    return keywords_map.get(intent, "veterinary guidance sheep health")
