"""
Curated Andhra Pradesh locations dataset covering districts, mandals, livestock hubs, and towns.
"""

AP_REGIONS_DATABASE = [
    # West Godavari / Eluru / East Godavari Hubs
    "Tadepalligudem", "Tadepallegudem", "Tadepalligudem Rural", "Tadepalligudem Mandal", "Tadepalli", "Tadepalle", "Penumantra", "Pentapadu",
    "Tanuku", "Tanuku Rural", "Tanuku Mandal", "Attili", "Iragavaram", "Peravali",
    "Bhimavaram", "Bhimavaram Rural", "Bhimavaram Mandal", "Palakollu", "Narasapuram",
    "Akividu", "Kalla", "Undi", "Mogalthur", "Achanta", "Veeravasaram",
    "Eluru", "Eluru Rural", "Eluru Mandal", "Denduluru", "Pedavegi", "Chintalapudi",
    "Jangareddygudem", "Koyyalagudem", "Polavaram", "Gopalapuram", "Nidadavole",
    
    # Kakinada / Konaseema / East Godavari
    "Kakinada", "Kakinada Rural", "Kakinada Port", "Kakinada Urban",
    "Samalkota", "Samalkota Rural", "Peddapuram", "Peddapuram Rural", "Peddapuram Mandal",
    "Pithapuram", "Gollaprolu", "U.Kothapalli", "Thallarevu", "Karapa",
    "Rajahmundry", "Rajahmundry Rural", "Rajahmundry Urban", "Kadiam",
    "Rajanagaram", "Korukonda", "Gokavaram", "Seethanagaram",
    "Amalapuram", "Amalapuram Rural", "Razole", "Kothapeta", "Ravulapalem",
    "Mummidivaram", "Allavaram", "Katrenikona", "I.Polavaram", "Malkipuram",
    "Mandapeta", "Ramachandrapuram", "Rayavaram", "Bicavolu", "Anaparthy",
    "Tuni", "Tuni Rural", "Kotananduru", "Rowthulapudi", "Sankhavaram",
    "Prathipadu", "Yeleswaram", "Jaggampeta", "Gandepalli", "Annavaram",
    
    # Krishna / NTR / Guntur
    "Vijayawada", "Vijayawada Rural", "Vijayawada Central", "Penamaluru",
    "Gannavaram", "Gudivada", "Gudivada Rural", "Machilipatnam", "Nuzvid",
    "Kankipadu", "Vuyyuru", "Pamarru", "Avanigadda", "Jaggayyapeta", "Nandigama",
    "Guntur", "Guntur Rural", "Tenali", "Tenali Rural", "Mangalagiri",
    "Chilakaluripet", "Narasaraopet", "Sattenapalle", "Bapatla", "Repalle",
    "Vinukonda", "Macherla", "Piduguralla", "Ponnur",
    
    # Visakhapatnam / North Coastal
    "Visakhapatnam", "Gajuwaka", "Anakapalle", "Pendurthi", "Bheemunipatnam",
    "Chodavaram", "Narsipatnam", "Yelamanchili", "Payakaraopeta",
    "Vizianagaram", "Bobbili", "Parvathipuram", "Salur", "Cheepurupalli",
    "Srikakulam", "Amadalavalasa", "Palasa", "Tekkali", "Narasannapeta", "Sompeta",
    
    # Rayalaseema (Kurnool / Nandyal / Anantapur / Kadapa / Chittoor / Tirupati)
    "Kurnool", "Kurnool Rural", "Adoni", "Yemmiganur", "Alur", "Pattikonda", "Dhone",
    "Nandyal", "Allagadda", "Banaganapalle", "Atmakur", "Nandikotkur", "Koilkuntla",
    "Anantapur", "Anantapur Rural", "Dharmavaram", "Guntakal", "Tadipatri", "Kadiri",
    "Hindupur", "Madakasira", "Penukonda", "Rayadurg", "Uravakonda", "Kalyandurg",
    "Kadapa", "Proddatur", "Pulivendula", "Jammalamadugu", "Mydukur", "Badvel", "Rajampet",
    "Tirupati", "Tirupati Rural", "Chandragiri", "Srikalahasti", "Puttur", "Nagari",
    "Chittoor", "Madanapalle", "Punganur", "Palamaner", "Kuppam",
    
    # Prakasam / Nellore
    "Ongole", "Ongole Rural", "Markapur", "Chirala", "Kandukur", "Giddalur", "Podili", "Kanigiri",
    "Nellore", "Nellore Rural", "Kavali", "Gudur", "Venkatagiri", "Atmakur (Nellore)", "Sullurpeta"
]

TELANGANA_REGIONS_DATABASE = [
    # Hyderabad / Ranga Reddy / Medchal-Malkajgiri
    "Hyderabad", "Secunderabad", "Cyberabad", "Gachibowli", "Madhapur", "Kukatpally",
    "LB Nagar", "Uppal", "Malkajgiri", "Medchal", "Shamshabad", "Rajendranagar",
    "Ibrahimpatnam", "Maheshwaram", "Keesara", "Ghatkesar",

    # Warangal / Hanamkonda / Jangaon
    "Warangal", "Warangal Rural", "Warangal Urban", "Hanamkonda", "Kazipet",
    "Jangaon", "Mahabubabad", "Parkal", "Narsampet", "Wardhannapet", "Station Ghanpur",

    # Karimnagar / Peddapalli / Jagtial / Rajanna Sircilla
    "Karimnagar", "Karimnagar Rural", "Huzurabad", "Jammikunta", "Manakondur", "Choppadandi",
    "Peddapalli", "Ramagundam", "Godavarikhani", "Manthani", "Sultanabad",
    "Jagtial", "Korutla", "Metpally", "Dharmapuri", "Raikal",
    "Sircilla", "Vemulawada", "Yellareddypet",

    # Khammam / Bhadradri Kothagudem
    "Khammam", "Khammam Rural", "Khammam Urban", "Sathupalli", "Madhira", "Wyra", "Palair",
    "Kothagudem", "Palwancha", "Bhadrachalam", "Yellandu", "Manuguru", "Aswaraopeta",

    # Nizamabad / Kamareddy
    "Nizamabad", "Nizamabad Rural", "Nizamabad Urban", "Bodhan", "Armoor", "Bheemgal",
    "Kamareddy", "Banswada", "Yellareddy", "Jukkal",

    # Nalgonda / Suryapet / Yadadri Bhuvanagiri
    "Nalgonda", "Nalgonda Rural", "Miryalaguda", "Devarakonda", "Nakrekal", "Nagarjuna Sagar",
    "Suryapet", "Kodad", "Huzurnagar", "Thungathurthy",
    "Bhongir", "Yadagirigutta", "Alair", "Choutuppal", "Mothkur",

    # Mahbubnagar / Nagarkurnool / Wanaparthy / Jogulamba Gadwal / Narayanpet
    "Mahbubnagar", "Mahbubnagar Rural", "Jadcherla", "Bhoothpur", "Devarkadra",
    "Nagarkurnool", "Achampet", "Kalwakurthy", "Kollapur",
    "Wanaparthy", "Pebbair", "Kothakota", "Atmakur (Wanaparthy)",
    "Gadwal", "Alampur", "Ieeja",
    "Narayanpet", "Makthal", "Kosgi",

    # Adilabad / Mancherial / Nirmal / Kumuram Bheem Asifabad
    "Adilabad", "Adilabad Rural", "Utnoor", "Boath", "Bela",
    "Mancherial", "Bellampalli", "Mandamarri", "Chennur", "Luxettipet",
    "Nirmal", "Bhainsa", "Khanapur", "Mudhole",
    "Asifabad", "Kagaznagar", "Sirpur",

    # Medak / Sangareddy / Siddipet
    "Medak", "Narsapur", "Ramayampet", "Chegunta",
    "Sangareddy", "Zaheerabad", "Patancheru", "Narayankhed", "Sadasivpet", "Andole",
    "Siddipet", "Siddipet Rural", "Gajwel", "Dubbak", "Husnabad",

    # Vikarabad / Mulugu / Jayashankar Bhupalpally
    "Vikarabad", "Tandur", "Pargi", "Kodangal",
    "Mulugu", "Venkatapur", "Eturnagaram",
    "Bhupalpally", "Kataram", "Mahadevpur"
]

ALL_REGIONS_DATABASE = AP_REGIONS_DATABASE + TELANGANA_REGIONS_DATABASE

def search_locations(query: str, max_results: int = 10) -> list:
    """
    Performs prefix and substring matching across both AP and Telangana locations dataset.
    Prioritizes prefix matches over general substring matches.
    """
    if not query or not query.strip():
        return []
        
    q = query.strip().lower()
    
    prefix_matches = []
    substring_matches = []
    
    for loc in ALL_REGIONS_DATABASE:
        loc_lower = loc.lower()
        if loc_lower.startswith(q):
            prefix_matches.append(loc)
        elif q in loc_lower:
            substring_matches.append(loc)
            
    # Combine prefix matches first, then substring matches (deduplicated preserving order)
    combined = prefix_matches + [m for m in substring_matches if m not in prefix_matches]
    return combined[:max_results]


def determine_state(location_name: str) -> str:
    """
    Determines whether a location belongs to Andhra Pradesh or Telangana when possible.
    Returns 'Andhra Pradesh', 'Telangana', or '' (if unknown).
    """
    if not location_name or not location_name.strip():
        return ""

    loc_clean = location_name.strip()
    loc_lower = loc_clean.lower()

    # Explicit user input check
    if "andhra" in loc_lower or " ap" in loc_lower or loc_lower.endswith(", ap") or loc_lower.startswith("ap "):
        return "Andhra Pradesh"
    if "telangana" in loc_lower or " tg" in loc_lower or " ts" in loc_lower or loc_lower.endswith(", ts") or loc_lower.endswith(", tg"):
        return "Telangana"

    # Exact or word boundary matches in Telangana
    for tg_loc in TELANGANA_REGIONS_DATABASE:
        tg_l = tg_loc.lower()
        if tg_l == loc_lower or tg_l in loc_lower.split(","):
            return "Telangana"

    # Exact or word boundary matches in AP
    for ap_loc in AP_REGIONS_DATABASE:
        ap_l = ap_loc.lower()
        if ap_l == loc_lower or ap_l in loc_lower.split(","):
            return "Andhra Pradesh"

    # Substring matches in Telangana
    for tg_loc in TELANGANA_REGIONS_DATABASE:
        if tg_loc.lower() in loc_lower:
            return "Telangana"

    # Substring matches in AP
    for ap_loc in AP_REGIONS_DATABASE:
        if ap_loc.lower() in loc_lower:
            return "Andhra Pradesh"

    return ""


def get_formatted_location_for_maps(location_name: str) -> str:
    """
    Resolves the location with proper state and country context for dynamic Google Maps queries.
    e.g.
    'Tadepalligudem' -> 'Tadepalligudem, Andhra Pradesh, India'
    'Warangal' -> 'Warangal, Telangana, India'
    'Some Small Village' -> 'Some Small Village, India' (or with state if detected/entered)
    """
    loc_clean = location_name.strip()
    if not loc_clean:
        return "Kakinada, Andhra Pradesh, India"

    loc_lower = loc_clean.lower()

    # If user already typed 'India'
    if "india" in loc_lower:
        return loc_clean

    state = determine_state(loc_clean)
    if state:
        if state.lower() in loc_lower:
            return f"{loc_clean}, India"
        else:
            return f"{loc_clean}, {state}, India"
    else:
        # Fallback for any unknown village / locality
        return f"{loc_clean}, India"

