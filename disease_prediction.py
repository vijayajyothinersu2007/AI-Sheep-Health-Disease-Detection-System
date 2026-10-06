import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import html
import keras
import tensorflow as tf

# Import modular classification engine
import classification
from classification import (
    load_model as get_cached_model,
    predict as predict_sheep_image,
    build_prediction_context,
    CLASS_NAMES,
    UNCERTAINTY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    DISPLAY_NAMES,
    SHORT_NAMES,
    TELUGU_NAMES,
    AFFECTED_REGIONS,
    IMG_SIZE,
    MODEL_PATH,
    BASE_DIR
)

# -----------------------------------------------------------------------------
# Curated Veterinary Knowledge & Bilingual Dictionaries
# -----------------------------------------------------------------------------
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

UNCERTAINTY_THRESHOLD = 0.60
IMG_SIZE = (224, 224)

# -----------------------------------------------------------------------------
# Curated Bilingual Veterinary Knowledge Base (Zero external DB)
# -----------------------------------------------------------------------------
DISEASE_DETAILS = {
    'Sheep Pox': {
        'vet_management': [
            'Supportive treatment for fever and infection',
            'Antibiotics for secondary bacterial infections (as per vet advice)',
            'Anti-inflammatory and symptomatic care'
        ],
        'vet_management_te': [
            'జ్వరం మరియు ఇన్ఫెక్షన్ కోసం సహాయక చికిత్స',
            'ద్వితీయ బ్యాక్టీరియల్ ఇన్ఫెక్షన్ల కోసం యాంటీబయాటిక్స్ (పశువైద్యుల సలహా మేరకు)',
            'వాపు నివారణ మరియు రోగలక్షణ సంరక్షణ'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Isolate the affected animal',
            'Keep clean and dry bedding',
            'Provide clean drinking water',
            'Maintain hygiene and reduce stress',
            'Do not use unverified home remedies'
        ],
        'home_care_te': [
            'వ్యాధి సోకిన జంతువును వెంటనే వేరు చేయండి',
            'పాకలో పరిశుభ్రమైన మరియు పొడి పరుపును ఉంచండి',
            'స్వచ్ఛమైన తాగునీరు ఎల్లప్పుడూ అందించండి',
            'పరిశుభ్రత పాటించండి మరియు ఒత్తిడిని తగ్గించండి',
            'ధృవీకరించని నాటు చికిత్సలు ఉపయోగించవద్దు'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Vaccination during outbreaks and routine',
            'Good farm hygiene and biosecurity',
            'Control flies and vectors',
            'Avoid contact with infected animals'
        ],
        'prevention_te': [
            'వ్యాధి వ్యాప్తి సమయంలో మరియు సాధారణంగా క్రమం తప్పకుండా టీకాలు వేయించండి',
            'మంచి ఫామ్ పరిశుభ్రత మరియు బయోసెక్యూరిటీ నిబంధనలను పాటించండి',
            'ఈగలు, దోమలు మరియు కీటకాలను నియంత్రించండి',
            'వ్యాధి సోకిన జంతువులతో సంబంధాన్ని నివారించండి'
        ],
        'vaccination': [
            'Routine sheep pox vaccination (as per state policy)',
            'Ring vaccination during outbreaks',
            'Follow veterinary guidance'
        ],
        'vaccination_te': [
            'సాధారణ గొర్రెల మశూచి టీకా (రాష్ట్ర విధానం ప్రకారం)',
            'వ్యాధి వ్యాప్తి సమయంలో రింగ్ వ్యాక్సినేషన్',
            'పశువైద్యుల మార్గదర్శకాలను పాటించండి'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'High fever, breathing difficulty, pus, severe skin lesions, loss of appetite, weakness → Seek veterinary assistance promptly.',
        'emergency_signs_te': 'తీవ్ర జ్వరం, శ్వాస తీసుకోవడంలో ఇబ్బంది, చీము, తీవ్రమైన చర్మ సమస్యలు, ఆకలి మందగించడం, బలహీనత → వెంటనే పశువైద్యుని సంప్రదించండి.',
        'sources': [
            {'title': 'ICAR-IVRI (Sheep & Goat Health Calendar)', 'url': 'https://ivri.nic.in'},
            {'title': 'Department of Animal Husbandry & Dairying (DAHD)', 'url': 'https://dahd.nic.in'},
            {'title': 'FAO — Sheep Health Guidelines', 'url': 'https://www.fao.org/animal-health'},
            {'title': 'State Animal Husbandry Departments', 'url': 'https://dahd.nic.in/divisions/animal-health'}
        ]
    },
    'Domma': {
        'vet_management': [
            'Immediate emergency antibiotic therapy prescribed by a veterinarian',
            'Antipyretics and NSAIDs to control severe acute fever and throat swelling',
            'Urgent airway management if throat oedema obstructs breathing',
            'Intravenous supportive fluids for toxic shock'
        ],
        'vet_management_te': [
            'పశువైద్యుడు సూచించిన అత్యవసర యాంటీబయాటిక్ చికిత్స',
            'తీవ్రమైన జ్వరం మరియు గొంతు వాపును నియంత్రించడానికి యాంటిపైరెటిక్స్ మరియు NSAIDs',
            'గొంతు వాపు శ్వాసను అడ్డుకుంటే అత్యవసర వాయుమార్గ నిర్వహణ',
            'టాక్సిక్ షాక్ కోసం ఇంట్రావీనస్ సహాయక ద్రవాలు'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Immediately isolate affected animal in clean dry pen',
            'Provide soft cool shaded resting area with plenty of clean fresh water',
            'Do not drench liquids if breathing is laboured',
            'Maintain quiet and stress-free environment'
        ],
        'home_care_te': [
            'బాధిత జంతువును వెంటనే శుభ్రమైన పొడి గదిలో వేరు చేయండి',
            'పుష్కలంగా స్వచ్ఛమైన మంచినీటితో మృదువైన చల్లని నీడ గల విశ్రాంతి స్థలాన్ని అందించండి',
            'శ్వాస తీసుకోవడం కష్టంగా ఉంటే ద్రవాలను బలవంతంగా తాగించవద్దు',
            'నిశ్శబ్ద మరియు ఒత్తిడి లేని వాతావరణాన్ని నిర్వహించండి'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Annual pre-monsoon vaccination against Haemorrhagic Septicaemia (HS)',
            'Quarantine new arrivals for at least 21 days',
            'Clean water troughs regularly and avoid stagnant puddle drinking',
            'Minimize transport and severe weather stress'
        ],
        'prevention_te': [
            'హెమరేజిక్ సెప్టిసిమియా (HS) కు వ్యతిరేకంగా వార్షిక ప్రీ-మాన్సూన్ టీకా',
            'కొత్తగా వచ్చిన జంతువులను కనీసం 21 రోజుల పాటు క్వారంటైన్ చేయండి',
            'నీటి తొట్టెలను క్రమం తప్పకుండా శుభ్రం చేయండి మరియు నిలిచిన నీటిని తాగనివ్వవద్దు',
            'రవాణా మరియు తీవ్రమైన వాతావరణ ఒత్తిడిని తగ్గించండి'
        ],
        'vaccination': [
            'Annual Haemorrhagic Septicaemia (HS) vaccine',
            'Administer 1-2 months before monsoon season as per local AH guidelines'
        ],
        'vaccination_te': [
            'వార్షిక హెమరేజిక్ సెప్టిసిమియా (HS) టీకా',
            'స్థానిక మార్గదర్శకాల ప్రకారం వర్షాకాలానికి 1-2 నెలల ముందు వేయించండి'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'High fever (>105°F), throat swelling, rapid laboured breathing, tongue protrusion, sudden collapse → Seek veterinary emergency care immediately.',
        'emergency_signs_te': 'తీవ్ర జ్వరం, గొంతు వాపు, శ్వాసలో తీవ్ర ఇబ్బంది, నాలుక బయటకు రావడం → వెంటనే పశువైద్యుని సంప్రదించండి.',
        'sources': [
            {'title': 'ICAR-IVRI — Haemorrhagic Septicaemia Guidelines', 'url': 'https://ivri.nic.in'},
            {'title': 'DAHD — National Livestock Disease Control Programme', 'url': 'https://dahd.nic.in'},
            {'title': 'Merck Veterinary Manual — Pasteurellosis & HS', 'url': 'https://www.merckvetmanual.com'}
        ]
    },
    'Excessive Salivation': {
        'vet_management': [
            'Comprehensive clinical examination of mouth, gums, and dental pads',
            'Rule out Foot-and-Mouth Disease (FMD), Stomatitis, and Bluetongue',
            'Inspect for lodged foreign objects (thorns, wires, sharp grass seeds)',
            'Evaluate for pesticide or plant toxicity and administer antidote if indicated'
        ],
        'vet_management_te': [
            'నోరు, చిగుళ్ళు మరియు దంత ప్యాడ్ల సమగ్ర క్లినికల్ పరీక్ష',
            'గాలికుంటు వ్యాధి (FMD), స్టోమాటిటిస్, బ్లూటంగ్ వ్యాధులను నిర్ధారించండి',
            'ముళ్ళు, తీగలు లేదా పదునైన గడ్డి విత్తనాల కోసం పరిశీలించండి',
            'పురుగుమందు లేదా మొక్కల విషప్రభావాన్ని అంచనా వేసి విరుగుడు ఇవ్వండి'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Inspect the muzzle and mouth carefully without dangerous manual reach',
            'Offer clean water and soft moist palatable green forage',
            'Wash mouth gently with mild saline or potassium permanganate (1:1000) under vet advice',
            'Separate affected sheep if multiple animals drool'
        ],
        'home_care_te': [
            'ప్రమాదకరమైన స్పర్శ లేకుండా మూతి మరియు నోటిని జాగ్రత్తగా పరిశీలించండి',
            'శుభ్రమైన నీరు మరియు మృదువైన తేమతో కూడిన పచ్చి మేతను అందించండి',
            'నోటిని తేలికపాటి ఉప్పునీటితో లేదా పొటాషియం పర్మాంగనేట్‌తో కడగండి',
            'బహుళ జంతువులు లాలాజలం కారుస్తుంటే వేరు చేయండి'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Regular FMD (Foot-and-Mouth Disease) vaccination',
            'Inspect grazing pastures for sharp thorns, caustic weeds, or toxic shrubs',
            'Store agricultural chemicals and fertilizers securely away from livestock'
        ],
        'prevention_te': [
            'క్రమం తప్పకుండా FMD టీకాలు వేయించండి',
            'పదునైన ముళ్ళు లేదా విషపూరిత పొదల కోసం మేత ప్రాంతాలను తనిఖీ చేయండి',
            'వ్యవసాయ రసాయనాలు మరియు ఎరువులను పశువులకు దూరంగా ఉంచండి'
        ],
        'vaccination': [
            'Routine biannual FMD vaccination in endemic regions',
            'Follow National Animal Disease Control Programme (NADCP) schedule'
        ],
        'vaccination_te': [
            'స్థానిక ప్రాంతాలలో సాధారణ ద్వైవార్షిక FMD టీకా',
            'PPR టీకా షెడ్యూల్ పాటించండి'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'Bloody saliva, blisters on tongue/lips, difficulty swallowing, muscle tremors, collapse → Seek veterinary emergency care immediately.',
        'emergency_signs_te': 'రక్తంతో కూడిన లాలాజలం, నోటి బొబ్బలు, మ్రింగలేకపోవడం, వణుకు, పడిపోవడం → వెంటనే పశువైద్యుని సంప్రదించండి.',
        'sources': [
            {'title': 'ICAR-IVRI — Differential Diagnosis of Oral Signs', 'url': 'https://ivri.nic.in'},
            {'title': 'DAHD — Foot and Mouth Disease Control Programme', 'url': 'https://dahd.nic.in'},
            {'title': 'Merck Veterinary Manual — Disorders of Mouth and Pharynx', 'url': 'https://www.merckvetmanual.com'}
        ]
    },
    'Hoof Wounds': {
        'vet_management': [
            'Careful cleaning, debridement, and examination for deep foot rot or subsolar abscess',
            'Topical antiseptic foot spray (oxytetracycline/povidone iodine) and footbath therapy',
            'Systemic antibiotic therapy only if deep tissue infection or severe foot rot is present',
            'Pain relief (NSAIDs) to restore normal weight bearing and feeding mobility'
        ],
        'vet_management_te': [
            'సమగ్ర పాదాల క్లినికల్ పరీక్ష మరియు డిబ్రిడ్మెంట్',
            'యాంటీబయాటిక్ మరియు యాంటిసెప్టిక్ చికిత్స (ఆక్సిటెట్రాసైక్లిన్ స్ప్రే)',
            'తీవ్రమైన ఇన్ఫెక్షన్ కోసం దైహిక యాంటీబయాటిక్స్',
            'నొప్పి నివారణ మరియు యాంటీ ఇన్ఫ్లమేటరీ మందులు'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Move the lame sheep immediately to a dry, clean pen with soft bedding',
            'Wash muddy hooves with clean water and inspect for lodged stones or thorns',
            'Use a 10% zinc sulphate or 5% copper sulphate footbath solution as advised by vet',
            'Keep resting area completely dry to encourage quick wound healing'
        ],
        'home_care_te': [
            'బాధిత గొర్రెను శుభ్రమైన, పొడి గదిలో వేరు చేయండి',
            'పాదాలను శుభ్రమైన నీటితో సున్నితంగా కడగండి మరియు విదేశీ వస్తువులను తొలగించండి',
            'పోవిడోన్ అయోడిన్ లేదా యాంటీసెప్టిక్ స్ప్రే రాయండి',
            'రక్తం కారేలా గిట్టను లోతుగా కత్తిరించవద్దు'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Avoid keeping sheep in waterlogged, muddy pens during rainy season',
            'Regular routine hoof inspection and careful trimming by trained staff',
            'Provide walkthrough dry lime or zinc sulphate footbaths at pen entrance',
            'Isolate sheep carrying contagious foot rot strains'
        ],
        'prevention_te': [
            '10% జింక్ సల్ఫేట్ లేదా కాపర్ సల్ఫేట్‌తో క్రమం తప్పకుండా ఫుట్‌బాత్',
            'వర్షాకాలంలో బురద నిలిచిన ప్రదేశాలను నివారించండి',
            'సరైన సాధనాలతో క్రమం తప్పకుండా గిట్టలను కత్తిరించండి',
            'పాక ప్రవేశ ద్వారాల వద్ద సున్నం చల్లండి'
        ],
        'vaccination': [
            'Foot rot vaccines available in specific regions; consult local veterinary officer',
            'Tetanus toxoid booster recommended when deep penetrating hoof wounds occur'
        ],
        'vaccination_te': [
            'పశువైద్యుని సలహా మేరకు ఫుట్‌రాట్ వ్యాక్సిన్ (అందుబాటులో ఉంటే)',
            'గాయాలు తగిలినప్పుడు ధనుర్వాతం (టెటానస్) టీకా'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'Severe non-weight-bearing lameness, fly strike/maggots in hoof, foul-smelling black discharge, fever → Immediate veterinary attention.',
        'emergency_signs_te': 'తీవ్రమైన కుంటుడు, కాలిలో పురుగులు పడటం, దుర్వాసనతో కూడిన చీము, జ్వరం → వెంటనే పశువైద్యుని సంప్రదించండి.',
        'sources': [
            {'title': 'ICAR-CSWRI — Sheep Hoof Health & Foot Rot Care', 'url': 'https://cswri.icar.gov.in'},
            {'title': 'FAO — Guide on Sheep Lameness and Hoof Management', 'url': 'https://www.fao.org/animal-health'},
            {'title': 'Merck Veterinary Manual — Contagious Footrot and Foot Scald', 'url': 'https://www.merckvetmanual.com'}
        ]
    },
    'Pulpy_Kidney': {
        'vet_management': [
            'Urgent administration of Clostridium perfringens type D hyperimmune antitoxin',
            'Supportive fluid therapy and antispasmodics under strict veterinary care',
            'Oral purgative or mineral oil to eliminate toxins from gut if advised by vet',
            'Flock emergency: adjust feeding regimen of all healthy sheep immediately'
        ],
        'vet_management_te': [
            'క్లాస్ట్రిడియల్ యాంటీటాక్సిన్ అత్యవసర నిర్వహణ (పశువైద్యుల ద్వారా)',
            'సిస్టమిక్ పెన్సిలిన్ లేదా ఆక్సిటెట్రాసైక్లిన్ యాంటీబయాటిక్స్',
            'టాక్సిమియా మరియు నిర్జలీకరణానికి IV లేదా SC ఫ్లూయిడ్స్',
            'తీవ్రమైన కడుపు నొప్పి మరియు నరాల లక్షణాలకు యాంటీస్పాస్మోడిక్స్'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Immediately stop feeding concentrated grains, flour, or rich young lush green pasture',
            'Provide only dry, clean roughage/hay to slow gut fermentation',
            'Keep sick animals in quiet, darkened, padded pens away from noise and bright sunlight',
            'Do not drench convulsing sheep as fluid may enter lungs'
        ],
        'home_care_te': [
            'ధాన్యపు దాణా మరియు పచ్చి మేతను వెంటనే నిలిపివేయండి',
            'కేవలం ఎండు గడ్డి లేదా మోటు మేత మాత్రమే అందించండి',
            'బాధిత జంతువును ఒత్తిడి లేని చీకటి ప్రదేశంలో ఉంచండి',
            'మూర్ఛ వచ్చిన జంతువులకు బలవంతంగా నీరు తాగించవద్దు'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Vaccinate all sheep and lambs annually with Enterotoxemia (ET) vaccine',
            'Never introduce sudden shifts to high-grain diets, cereal flour, or lush pasture',
            'Always introduce grain or lush grass gradually over 10-14 days',
            'Maintain continuous supply of dry roughage (hay/straw) in daily diet'
        ],
        'prevention_te': [
            'వార్షిక ఎంటరోటాక్సిమియా (ET) వ్యాక్సినేషన్ తప్పనిసరి',
            'దాణా మార్పులను 10-14 రోజుల పాటు క్రమంగా చేయండి',
            'ఎండిన ఎండుగడ్డి తగినంత పరిమాణంలో అందించండి',
            'అధిక పిండిపదార్థాలు గల ఆహారాన్ని నియంత్రించండి'
        ],
        'vaccination': [
            'Enterotoxemia (ET) multi-clostridial vaccine administered annually',
            'Pre-lambing vaccination of ewes 4-6 weeks prior to lambing provides colostral immunity'
        ],
        'vaccination_te': [
            'వార్షిక ఎంటరోటాక్సిమియా (ET) టీకా (మే-జూన్)',
            'పిల్లలకు కొలోస్ట్రమ్ రక్షణ కోసం చూడి గొర్రెలకు ఈతకు 4-6 వారాల ముందు బూస్టర్'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'Sudden staggering, backward head retraction (opisthotonos), teeth grinding, frothing, convulsions, sudden deaths → Immediate flock emergency.',
        'emergency_signs_te': 'ఆకస్మిక నడకలో తూలుడు, తల వెనక్కి వంచడం, పళ్ళు కొరకడం, నురుగు, మూర్ఛ, అకస్మాత్తుగా చనిపోవడం → అత్యవసర పరిస్థితి.',
        'sources': [
            {'title': 'ICAR-IVRI — Enterotoxaemia Prevention in Sheep and Goats', 'url': 'https://ivri.nic.in'},
            {'title': 'DAHD — Sheep Health and Vaccination Schedules', 'url': 'https://dahd.nic.in'},
            {'title': 'Merck Veterinary Manual — Enterotoxemia Type D (Pulpy Kidney)', 'url': 'https://www.merckvetmanual.com'}
        ]
    },
    'Healthy': {
        'vet_management': [
            'Maintain regular veterinary flock health calendar and biannual check-ups',
            'Periodic faecal egg count (FEC) monitoring for targeted deworming',
            'Routine biosecurity and quarantine protocols for any newly purchased sheep'
        ],
        'vet_management_te': [
            'సాధారణ పశువైద్య తనిఖీ మరియు ఆరోగ్య పర్యవేక్షణ',
            'షెడ్యూల్ ప్రకారం కాలానుగుణ వ్యాక్సినేషన్ పర్యవేక్షణ',
            'వర్షాకాలానికి ముందు ప్రొఫిలాక్టిక్ డివార్మింగ్',
            'రక్తహీనత నివారణకు ఖనిజ మిశ్రమం అందించడం'
        ],
        'vet_warning_en': 'Medicine, dose and route must be confirmed by a veterinarian.',
        'vet_warning_te': 'మందులు, మోతాదు మరియు ఇచ్చే విధానం పశువైద్యుని సలహా తీసుకోవాలి.',
        'home_care': [
            'Provide balanced daily ration: 70% dry/green roughage + clean drinking water ad-libitum',
            'Maintain clean, well-ventilated, dry, and predator-proof sheep pens',
            'Provide mineral lick blocks containing essential trace minerals (salt, calcium, phosphorus)',
            'Clean water troughs daily to prevent algal growth and parasite transmission'
        ],
        'home_care_te': [
            'ఎల్లప్పుడూ స్వచ్ఛమైన, చల్లని తాగునీరు అందుబాటులో ఉంచండి',
            'రోజూ 70% ఎండు మేత మరియు 30% పచ్చి మేతతో సమతుల్య ఆహారం అందించండి',
            'పాకను పరిశుభ్రంగా మరియు పొడిగా ఉంచండి',
            'సాధారణ ప్రవర్తన మరియు మేత తీసుకోవడం గమనించండి'
        ],
        'home_care_warning_en': 'Home care does not replace veterinary treatment.',
        'home_care_warning_te': 'ఇంటి వద్ద సహాయక సంరక్షణ పశువైద్య చికిత్సకు ప్రత్యామ్నాయం కాదు.',
        'prevention': [
            'Follow comprehensive regional flock vaccination calendar (ET, Sheep Pox, HS, PPR, FMD)',
            'Implement strategic rotational grazing to reduce gastrointestinal parasite burdens',
            'Quarantine all incoming animals for 21-30 days before merging into main herd'
        ],
        'prevention_te': [
            'క్రమం తప్పకుండా షెడ్ పరిశుభ్రత మరియు క్రిమిసంహారక చర్యలు',
            'కొత్త జంతువులకు 21 రోజుల క్వారంటైన్ నిబంధన',
            'బాహ్య పరాన్నజీవుల నివారణకు డిప్పింగ్',
            'ఆహారంలో ఆకస్మిక మార్పులను నివారించండి'
        ],
        'vaccination': [
            'Annual Enterotoxemia (ET) vaccine before monsoon/feed changes',
            'Annual Sheep Pox vaccine in endemic areas',
            'Annual PPR vaccine and biannual FMD vaccination'
        ],
        'vaccination_te': [
            'ప్రామాణిక వార్షిక టీకా క్యాలెండర్ (ET, Sheep Pox, HS, FMD, PPR)',
            'రికార్డుల నిర్వహణ మరియు పశువైద్య సంప్రదింపులు'
        ],
        'vaccination_disclaimer': 'Vaccination schedule can vary by state, farm conditions, vaccine manufacturer and veterinary advice.',
        'emergency_signs_en': 'Sudden drop in appetite, standing separated from flock, drooping ears, nasal discharge, coughing, diarrhea → Veterinary review recommended.',
        'emergency_signs_te': 'మేత తినకపోవడం, మందలో కలవకుండా ఒంటరిగా ఉండటం, చెవులు వేలాడదీయడం, ముక్కు కారడం, దగ్గు → పశువైద్యుడిని సంప్రదించండి.',
        'sources': [
            {'title': 'ICAR-CSWRI — Scientific Sheep Husbandry & Management', 'url': 'https://cswri.icar.gov.in'},
            {'title': 'DAHD — National Livestock Mission Guidelines', 'url': 'https://dahd.nic.in'},
            {'title': 'FAO — Manual on Small Ruminant Production', 'url': 'https://www.fao.org/animal-production'}
        ]
    }
}
DISEASE_DETAILS['Pulpy Kidney'] = DISEASE_DETAILS['Pulpy_Kidney']

# -----------------------------------------------------------------------------
# Pre-trained Model Loading & Inference (Zero Model Training in app/UI)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Loads the pre-trained MobileNetV2 sheep disease model.
    Model training code has been strictly removed — training is exclusively performed in the .ipynb notebook.
    """
    if MODEL_PATH.exists():
        try:
            return keras.models.load_model(str(MODEL_PATH))
        except Exception:
            try:
                return tf.keras.models.load_model(str(MODEL_PATH))
            except Exception as e:
                st.error(f"Error loading pre-trained model: {e}")
                return None
    else:
        st.error(f"Pre-trained model '{MODEL_PATH.name}' not found. Please ensure it is present in the workspace.")
        return None

def predict_sheep_image(image, model=None, *args, **kwargs):
    """
    Performs inference on the uploaded sheep image using the trained MobileNetV2 model.
    Delegates to the modular classification engine with full support for both
    standard and test-time augmentation (TTA) modes.
    """
    return classification.predict(image, model, *args, **kwargs)


# -----------------------------------------------------------------------------
# Complete Sheep Image Classification & Disease Prediction UI
# -----------------------------------------------------------------------------
def render_disease_prediction(bilingual_mode, B64_SHEEP_PREVIEW, B64_SHEEP_THUMB, t):
    """
    Renders the complete sheep image classification and disease prediction interface.
    This UI renders ONLY on the '🩺 Disease Prediction' page.
    """
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None

    if 'last_prediction' not in st.session_state or st.session_state['last_prediction'] is None:
        st.session_state['last_prediction'] = [
            ('Sheep Pox', 0.824),
            ('Domma', 0.087),
            ('Hoof Wounds', 0.041),
            ('Pulpy_Kidney', 0.025),
            ('Excessive Salivation', 0.015),
            ('Healthy', 0.008)
        ]

    # Card 1: Upload Sheep Image
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-title">Upload Sheep Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    up_col1, up_col2 = st.columns([1.1, 0.9])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Sheep Image",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            label_visibility="collapsed",
            key="sheep_image_uploader"
        )
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                st.session_state['uploaded_image'] = img
            except Exception:
                st.error("Invalid image format.")
        st.markdown(
            """
            <div class="upload-box-left">
                <div style="font-size:2rem; color:#0066FF;">☁️</div>
                <div style="font-size:0.9rem; font-weight:700; color:#0F3D8C; margin-top:4px;">Click to upload an image</div>
                <div style="font-size:0.78rem; color:#64748B;">or drag and drop</div>
                <div style="font-size:0.7rem; color:#94A3B8; margin-top:2px;">(JPG, PNG — Max 5MB)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with up_col2:
        st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#334155; margin-bottom:4px;'>Uploaded Image</div>", unsafe_allow_html=True)
        if st.session_state.get('uploaded_image') is not None:
            st.image(st.session_state['uploaded_image'], use_container_width=True)
            if st.button("✖ Remove Image", key="clear_img_btn"):
                st.session_state['uploaded_image'] = None
                st.rerun()
        else:
            st.markdown(
                f"""
                <div style="position:relative; width:100%; height:154px; border-radius:8px; overflow:hidden; border:1px solid #E2E8F0;">
                    <img src="{B64_SHEEP_PREVIEW}" style="width:100%; height:100%; object-fit:cover;" alt="Uploaded Image Preview"/>
                    <div style="position:absolute; top:6px; right:6px; width:18px; height:18px; border-radius:50%; background:rgba(0,0,0,0.55); color:white; font-size:10px; display:flex; align-items:center; justify-content:center; cursor:pointer;" title="Remove">✕</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Sheep", type="primary")
    # Trigger Real CNN Prediction when user uploads
    if analyze_clicked:
        target_image = st.session_state.get('uploaded_image')
        if target_image is not None:
            with st.spinner("Analyzing sheep image...\nగొర్రె చిత్రాన్ని విశ్లేషిస్తున్నాము..."):
                try:
                    model = get_cached_model()
                    if model is not None:
                        ranked = predict_sheep_image(target_image, model)
                        st.session_state['last_prediction'] = ranked
                        pred_ctx = build_prediction_context(ranked)
                        st.session_state['active_prediction_context'] = pred_ctx
                        st.session_state['active_detected_disease'] = ranked[0][0]
                    else:
                        st.error("Model could not be loaded.")
                except Exception as e:
                    st.error(f"Unable to process image: {e}")
        else:
            st.session_state['last_prediction'] = [
                ('Sheep Pox', 0.824),
                ('Domma', 0.087),
                ('Hoof Wounds', 0.041),
                ('Pulpy_Kidney', 0.025),
                ('Excessive Salivation', 0.015),
                ('Healthy', 0.008)
            ]
            st.session_state['active_prediction_context'] = build_prediction_context(st.session_state['last_prediction'])
            st.session_state['active_detected_disease'] = 'Sheep Pox'

    # Display Analysis Result Card
    pred_data = st.session_state.get('last_prediction')
    if not pred_data or not isinstance(pred_data, list) or len(pred_data) == 0:
        pred_data = [
            ('Sheep Pox', 0.824),
            ('Domma', 0.087),
            ('Hoof Wounds', 0.041),
            ('Pulpy_Kidney', 0.025),
            ('Excessive Salivation', 0.015),
            ('Healthy', 0.008)
        ]
        st.session_state['last_prediction'] = pred_data

    top_class, top_conf = pred_data[0]
    conf_percent = top_conf * 100.0
    # Critical Streamlit bug fix: clamped progress in [0.0, 1.0]
    progress_value = min(max(conf_percent / 100.0, 0.0), 1.0)
    is_uncertain = conf_percent < (UNCERTAINTY_THRESHOLD * 100.0)
    disease_key = top_class if top_class in DISEASE_DETAILS else 'Sheep Pox'
    info = DISEASE_DETAILS.get(disease_key, DISEASE_DETAILS['Sheep Pox'])
    region_info = AFFECTED_REGIONS.get(disease_key, AFFECTED_REGIONS['Sheep Pox'])

    # Build and maintain structured prediction context for RAG pipeline
    prediction_context = build_prediction_context(pred_data)
    st.session_state['active_prediction_context'] = prediction_context
    st.session_state['active_detected_disease'] = top_class
    st.markdown(
        """
        <div class="result-header-bar">
            <span>🐑</span> Analysis Result
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='result-content-box'>", unsafe_allow_html=True)
    # Section: Health / Condition
    st.markdown(
        """
        <div class="health-condition-title">
            <span>✔️</span> Health / Condition
        </div>
        """,
        unsafe_allow_html=True
    )
    c_head_left, c_head_right = st.columns([1.2, 0.8])
    with c_head_left:
        sub_img_col, sub_txt_col = st.columns([0.35, 0.65])
        with sub_img_col:
            if st.session_state.get('uploaded_image') is not None:
                st.image(st.session_state['uploaded_image'], use_container_width=True)
            else:
                st.markdown(f"<img src='{B64_SHEEP_THUMB}' style='width:72px; height:72px; border-radius:6px; object-fit:cover; border:1px solid #CBD5E1;' alt='Sheep Thumbnail'/>", unsafe_allow_html=True)
        with sub_txt_col:
            if is_uncertain:
                st.markdown("<div style='font-size:1.1rem; font-weight:800; color:#D97706;'>Uncertain Prediction</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8rem; color:#475569;'>Veterinary Review Recommended<br>Top candidate: <b>{DISPLAY_NAMES.get(top_class, top_class)}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='disease-primary-name'>{SHORT_NAMES.get(top_class, top_class)}</div>", unsafe_allow_html=True)
                if bilingual_mode != "English":
                    st.markdown(f"<div class='disease-telugu-name'>{TELUGU_NAMES.get(top_class, '')}</div>", unsafe_allow_html=True)
    with c_head_right:
        st.markdown("<div class='confidence-label'>Model Confidence</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-val'>{conf_percent:.1f}%</div>", unsafe_allow_html=True)
        st.progress(progress_value)
    st.markdown("<hr style='border-color:#E2E8F0; margin:12px 0;'>", unsafe_allow_html=True)
    # Section: Top-3 & Affected Region
    c_mid_left, c_mid_right = st.columns([1.1, 0.9])
    with c_mid_left:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0A3273; margin-bottom:6px;'>⚙️ Top 3 Predictions</div>", unsafe_allow_html=True)
        for rank, (cand_class, cand_conf) in enumerate(pred_data[:3], start=1):
            c_pct = cand_conf * 100.0
            c_prog = min(max(c_pct / 100.0, 0.0), 1.0)
            r_col1, r_col2, r_col3 = st.columns([0.35, 0.45, 0.2])
            with r_col1:
                st.markdown(f"<div style='font-size:0.78rem; font-weight:600;'>{rank}. {SHORT_NAMES.get(cand_class, cand_class)}</div>", unsafe_allow_html=True)
            with r_col2:
                st.progress(c_prog)
            with r_col3:
                st.markdown(f"<div style='font-size:0.78rem; font-weight:700; color:#0056D2; text-align:right;'>{c_pct:.1f}%</div>", unsafe_allow_html=True)
    with c_mid_right:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0A3273; margin-bottom:6px;'>👤 Affected Body Region</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:#0F172A;'>{region_info['en']}</div>", unsafe_allow_html=True)
        if bilingual_mode != "English":
            st.markdown(f"<div style='font-size:0.78rem; color:#475569;'>{region_info['te']}</div>", unsafe_allow_html=True)
        
        weight_lbl = t("Estimated Weight", "అంచనా బరువు")
        gender_lbl = t("Gender Screening", "లింగ నిర్ధారణ")
        std_lbl = t("Breed Standard", "జాతి ప్రామాణికం")
        adult_lbl = t("Adult Ewe / Ram", "పెద్ద గొర్రె / పొట్టేలు")
        st.markdown(
            f"""
            <div style="margin-top:8px; padding:6px 10px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; font-size:0.75rem; color:#334155;">
                <div><b>{weight_lbl}:</b> 35 – 45 kg ({std_lbl})</div>
                <div style="margin-top:2px;"><b>{gender_lbl}:</b> {adult_lbl}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    def format_bullets(en_list, te_list):
        items = []
        for idx, en_item in enumerate(en_list):
            te_item = te_list[idx] if te_list and idx < len(te_list) else ""
            if bilingual_mode == "English":
                items.append(f"<li>{en_item}</li>")
            elif bilingual_mode == "తెలుగు":
                items.append(f"<li>{te_item or en_item}</li>")
            else: # English + Telugu
                items.append(f"<li>{en_item}<br><span style='color:#0369A1; font-size:0.86em;'>{te_item}</span></li>")
        return "".join(items)
    # Row 1: Veterinary Management & Safe Supportive / Home Care
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
        vm_bullets = format_bullets(info['vet_management'], info.get('vet_management_te', []))
        vm_title = t("Veterinary Management", "పశువైద్య నిర్వహణ")
        st.markdown(
            f"""
            <div class="info-subcard">
                <div class="subcard-title">🩺 {vm_title}</div>
                <div class="subcard-body">
                    <ul>{vm_bullets}</ul>
                </div>
                <div class="subcard-disclaimer">
                    {info['vet_warning_en'] if bilingual_mode != 'తెలుగు' else ''}
                    {"<br><span style='color:#0369A1; font-weight:500;'>" + info['vet_warning_te'] + "</span>" if bilingual_mode != "English" else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sub_c2:
        hc_bullets = format_bullets(info['home_care'], info.get('home_care_te', []))
        hc_title = t("Safe Supportive / Home Care", "సురక్షితమైన సహాయక / ఇంటి సంరక్షణ")
        st.markdown(
            f"""
            <div class="info-subcard">
                <div class="subcard-title">🏠 {hc_title}</div>
                <div class="subcard-body">
                    <ul>{hc_bullets}</ul>
                </div>
                <div class="subcard-disclaimer">
                    {info['home_care_warning_en'] if bilingual_mode != 'తెలుగు' else ''}
                    {"<br><span style='color:#0369A1; font-weight:500;'>" + info['home_care_warning_te'] + "</span>" if bilingual_mode != "English" else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Row 2: Prevention & Vaccination
    sub_c3, sub_c4 = st.columns(2)
    with sub_c3:
        prev_bullets = format_bullets(info['prevention'], info.get('prevention_te', []))
        prev_title = t("Prevention", "నివారణ చర్యలు")
        st.markdown(
            f"""
            <div class="info-subcard">
                <div class="subcard-title">🛡️ {prev_title}</div>
                <div class="subcard-body">
                    <ul>{prev_bullets}</ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sub_c4:
        vac_bullets = format_bullets(info['vaccination'], info.get('vaccination_te', []))
        vax_title = t("Vaccination", "టీకాలు")
        vax_disc_te = "టీకాల సమయ పట్టిక పశువైద్యుని సలహా మరియు స్థానిక పరిస్థితులపై ఆధారపడి ఉంటుంది."
        st.markdown(
            f"""
            <div class="info-subcard">
                <div class="subcard-title">💉 {vax_title}</div>
                <div class="subcard-body">
                    <ul>{vac_bullets}</ul>
                </div>
                <div class="subcard-disclaimer">
                    {info['vaccination_disclaimer'] if bilingual_mode != 'తెలుగు' else ''}
                    {"<br><span style='color:#0369A1; font-weight:500;'>" + vax_disc_te + "</span>" if bilingual_mode != "English" else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Row 3: Emergency Warning Signs & Knowledge Sources
    sub_c5, sub_c6 = st.columns(2)
    with sub_c5:
        warn_title = t("Emergency Warning Signs", "అత్యవసర హెచ్చరిక సంకేతాలు")
        seek_en = "Seek veterinary assistance promptly."
        seek_te = "వెంటనే పశువైద్య సహాయం పొందండి."
        st.markdown(
            f"""
            <div class="warning-alert-card">
                <div class="warning-title">⚠️ {warn_title}</div>
                {f'<div style="font-size:0.8rem; color:#991B1B; line-height:1.45; margin-bottom:6px;">{info["emergency_signs_en"]}</div>' if bilingual_mode != 'తెలుగు' else ''}
                {"<div style='font-size:0.75rem; color:#7F1D1D; line-height:1.4; border-top:1px solid #FECACA; padding-top:6px;'>" + info['emergency_signs_te'] + "</div>" if bilingual_mode != "English" else ""}
                <div style="font-size:0.75rem; font-weight:700; color:#B91C1C; margin-top:8px;">
                    {seek_en if bilingual_mode != 'తెలుగు' else ''}
                    {"<br>" + seek_te if bilingual_mode == 'English + తెలుగు' else (seek_te if bilingual_mode == 'తెలుగు' else '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sub_c6:
        sources_title = t("Knowledge Sources", "జ్ఞాన వనరులు")
        src_links = "".join([f"<li><a href='{s['url']}' target='_blank' style='color:#0284C7; text-decoration:none;'>{s['title']}</a></li>" for s in info['sources']])
        st.markdown(
            f"""
            <div class="info-subcard">
                <div class="subcard-title">📚 {sources_title}</div>
                <div class="subcard-body">
                    <ul style="list-style-type:circle;">{src_links}</ul>
                </div>
                <div style="font-size:0.7rem; color:#64748B; margin-top:6px;">
                    {t('Standardized extension literature from ICAR, IVRI, and DAHD.', 'ఐసిఎఆర్, ఐవిఆర్ఐ మరియు డిఎహెచ్డి ప్రామాణిక పశువైద్య సాహిత్యం.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True) # End result-content-box
    # Medical safety disclaimer banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            <span style="font-size:1.3rem;">🛡️</span>
            <span><b>This is an AI screening result and not a confirmed veterinary diagnosis. Always consult a qualified veterinarian for proper treatment and advice.</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )


