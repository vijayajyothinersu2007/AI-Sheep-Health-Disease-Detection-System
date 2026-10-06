"""
Class names, display names, Telugu translations, and affected anatomical regions
for sheep disease classification.
"""

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
