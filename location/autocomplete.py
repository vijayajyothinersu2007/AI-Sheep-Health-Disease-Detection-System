"""
Autocomplete helper for interactive location search in Streamlit.
"""
from typing import List
from .location_search import search_locations

def get_location_suggestions(input_text: str, max_results: int = 10) -> List[str]:
    """
    Returns ranked auto-complete suggestions for partial user inputs.
    e.g. 'tade' -> ['Tadepalligudem', 'Tadepallegudem', 'Tadepalligudem Rural', 'Tadepalligudem Mandal']
    """
    if not input_text or len(input_text.strip()) < 2:
        return []
    return search_locations(input_text, max_results=max_results)
