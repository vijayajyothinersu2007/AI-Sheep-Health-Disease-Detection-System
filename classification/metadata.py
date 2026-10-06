"""
Metadata, taxonomy, bilingual dictionaries, and prediction context builder.
"""
from typing import List, Tuple, Dict, Any, Optional

CLASS_NAMES = [
    'Domma', 'Excessive Salivation', 'Healthy', 'Hoof Wounds', 'Pulpy_Kidney', 'Sheep Pox'
]

UNCERTAINTY_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.45

DISPLAY_NAMES = {
    'Domma': 'Domma / Haemorrhagic Septicaemia (HS)',
    'Excessive Salivation': 'Excessive Salivation (Clinical Sign)',
    'Healthy': 'Healthy Sheep',
    'Hoof Wounds': 'Hoof Wounds / Lameness',
    'Pulpy_Kidney': 'Pulpy Kidney Disease / Enterotoxemia',
    'Pulpy Kidney': 'Pulpy Kidney Disease / Enterotoxemia',
    'Sheep Pox': 'Sheep Pox',
}

SHORT_NAMES = {
    'Domma': 'Domma',
    'Excessive Salivation': 'Excessive Salivation',
    'Healthy': 'Healthy',
    'Hoof Wounds': 'Hoof Wounds',
    'Pulpy_Kidney': 'Pulpy Kidney',
    'Pulpy Kidney': 'Pulpy Kidney',
    'Sheep Pox': 'Sheep Pox',
}

TELUGU_NAMES = {
    'Domma': 'దొమ్మ / హెమరేజిక్ సెప్టిసీమియా',
    'Excessive Salivation': 'అధిక లాలాజలం కారడం (లక్షణం)',
    'Healthy': 'ఆరోగ్యకరమైన గొర్రె',
    'Hoof Wounds': 'గిట్టల గాయాలు / కుంటువ్యాధి',
    'Pulpy_Kidney': 'పల్పి కిడ్నీ వ్యాధి / ఎంటరోటాక్సిమియా',
    'Pulpy Kidney': 'పల్పి కిడ్నీ వ్యాధి / ఎంటరోటాక్సిమియా',
    'Sheep Pox': 'గొర్రెల పొంగు వ్యాధి',
}

AFFECTED_REGIONS = {
    'Sheep Pox': {
        'en': 'Skin / Respiratory system',
        'te': 'చర్మం / శ్వాసకోశ వ్యవస్థ'
    },
    'Domma': {
        'en': 'Throat / Respiratory tract & blood circulation',
        'te': 'గొంతు / శ్వాసకోశ మార్గం మరియు రక్తప్రసరణ వ్యవస్థ'
    },
    'Excessive Salivation': {
        'en': 'Mouth / Oral cavity & swallowing tract',
        'te': 'నోరు / నోటి గుహ మరియు గొంతు'
    },
    'Hoof Wounds': {
        'en': 'Hooves / Feet & lower legs',
        'te': 'గిట్టలు / కాళ్లు మరియు పాదాలు'
    },
    'Pulpy_Kidney': {
        'en': 'Digestive tract & kidneys (systemic toxemia)',
        'te': 'జీర్ణవ్యవస్థ మరియు మూత్రపిండాలు (రక్తవిషబాధ)'
    },
    'Healthy': {
        'en': 'General body condition (Normal)',
        'te': 'సాధారణ శరీర పరిస్థితి (ఆరోగ్యకరం)'
    }
}
AFFECTED_REGIONS['Pulpy Kidney'] = AFFECTED_REGIONS['Pulpy_Kidney']

def build_prediction_context(
    ranked_predictions: List[Tuple[str, float]],
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Constructs a structured, standardized prediction context dictionary
    to be passed seamlessly into the RAG retriever and prompt builder.
    """
    if not ranked_predictions:
        ranked_predictions = [('Healthy', 0.50)]
    
    top_class, top_conf = ranked_predictions[0]
    conf_percent = round(top_conf * 100.0, 1)
    
    # Determine confidence level category
    if top_conf >= UNCERTAINTY_THRESHOLD:
        confidence_level = "high"
    elif top_conf >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    is_uncertain = top_conf < UNCERTAINTY_THRESHOLD
    health_status = "healthy" if top_class == "Healthy" else "unhealthy"
    
    region = AFFECTED_REGIONS.get(top_class, {
        'en': 'General body condition',
        'te': 'సాధారణ శరీర పరిస్థితి'
    })
    
    top_k_list = []
    for rank, (c_name, c_conf) in enumerate(ranked_predictions[:3], start=1):
        top_k_list.append({
            "rank": rank,
            "raw_class": c_name,
            "disease": DISPLAY_NAMES.get(c_name, c_name),
            "short_name": SHORT_NAMES.get(c_name, c_name),
            "telugu_name": TELUGU_NAMES.get(c_name, ""),
            "confidence": round(float(c_conf), 4),
            "confidence_percent": round(float(c_conf * 100.0), 1)
        })
    
    context = {
        "animal": "sheep",
        "health_status": health_status,
        "disease": DISPLAY_NAMES.get(top_class, top_class),
        "raw_class": top_class,
        "short_name": SHORT_NAMES.get(top_class, top_class),
        "telugu_name": TELUGU_NAMES.get(top_class, ""),
        "confidence": round(float(top_conf), 4),
        "confidence_percent": conf_percent,
        "confidence_level": confidence_level,
        "is_uncertain": is_uncertain,
        "gender": "Adult Ewe / Ram",
        "estimated_weight": "35 – 45 kg",
        "affected_region": region['en'],
        "affected_region_te": region['te'],
        "top_k": top_k_list
    }
    
    if extra_metadata:
        context.update(extra_metadata)
        
    return context
