"""
Google Maps URL generator for nearby veterinary facilities.
"""
from urllib.parse import quote
from typing import List, Dict, Any
from .location_search import get_formatted_location_for_maps

FACILITY_TEMPLATES = [
    {
        "title": "Government Veterinary Hospital / Dispensary",
        "icon": "🏥",
        "query_prefix": "Government veterinary hospital dispensary near ",
        "desc": "Subsidized clinical care, routine vaccinations, and emergency triage."
    },
    {
        "title": "Private Veterinary Clinics & Doctors",
        "icon": "🩺",
        "query_prefix": "Veterinary clinic doctor near ",
        "desc": "Small ruminant veterinarians, specialized diagnoses, and field assistance."
    },
    {
        "title": "Animal Husbandry Department Office",
        "icon": "🏛️",
        "query_prefix": "Animal husbandry department office near ",
        "desc": "Government livestock welfare schemes, subsidized fodder seeds, and disease control officers."
    },
    {
        "title": "Veterinary Pharmacy & Livestock Medicines",
        "icon": "💊",
        "query_prefix": "Veterinary pharmacy livestock medicine store near ",
        "desc": "Veterinarian-prescribed antibiotics, dewormers, mineral mixtures, and antiseptic wound sprays."
    },
    {
        "title": "Sheep & Cattle Feed Suppliers / Fodder Centers",
        "icon": "🌾",
        "query_prefix": "Sheep feed fodder supplier store near ",
        "desc": "Commercial concentrates, silage, mineral blocks, and dry roughage."
    },
    {
        "title": "State Forest Department Range Office",
        "icon": "🌲",
        "query_prefix": "Forest department range office near ",
        "desc": "Grazing rights inquiries, local forest transit rules, and legal grazing boundary verification."
    }
]

def build_facility_cards(target_location: str) -> List[Dict[str, Any]]:
    """
    Constructs facility card metadata with live Google Maps links based on user's target location.
    Dynamically appends state and country context (e.g. 'Tadepalligudem, Andhra Pradesh, India'
    or 'Warangal, Telangana, India' or 'Some Village, India') for accurate Google Maps searching.
    """
    loc_clean = target_location.strip() or "Kakinada"
    formatted_location = get_formatted_location_for_maps(loc_clean)
    cards = []
    
    for fac in FACILITY_TEMPLATES:
        query_str = f"{fac['query_prefix']}{formatted_location}"
        maps_link = f"https://www.google.com/maps/search/?api=1&query={quote(query_str)}"
        cards.append({
            "title": fac["title"],
            "icon": fac["icon"],
            "desc": fac["desc"],
            "query": query_str,
            "maps_url": maps_link
        })
        
    return cards
