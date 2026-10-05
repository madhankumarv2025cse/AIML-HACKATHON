import re
import unicodedata

# Legal entity designators to strip / normalize
LEGAL_SUFFIXES = r"\b(ltd|limited|pvt|private|corp|corporation|co|company|inc|incorporated|llc|plc|gmbh|sa|sarl|bv|nv|pty|holdings|holding|group|enterprises|enterprise|services|solutions|intl|international)\b"

# Common street/address abbreviations mapping
ADDRESS_ABBREVIATIONS = {
    r"\broad\b": "rd",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bsuite\b": "ste",
    r"\bapartment\b": "apt",
    r"\bbuilding\b": "bldg",
    r"\bfloor\b": "fl",
    r"\bsquare\b": "sq",
    r"\bhighway\b": "hwy",
    r"\bjunction\b": "jct",
    r"\bplaza\b": "plz",
    r"\bparkway\b": "pkwy",
    r"\broute\b": "rt",
}

def clean_text(text: str) -> str:
    """Basic unicode normalization and lowercasing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # NFKD normalization to separate base characters and diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()

def normalize_country(country: str) -> str:
    """Normalize open-set country names."""
    cleaned = clean_text(country)
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()

def normalize_name(name: str) -> str:
    """Standardize and normalize business names while preserving distinct terms."""
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    
    # Standardize & -> and
    cleaned = re.sub(r'\s*&\s*', ' and ', cleaned)
    
    # Standardize punctuation to spaces except alphanumeric
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    
    # Normalize legal suffixes
    cleaned = re.sub(LEGAL_SUFFIXES, '', cleaned)
    
    # Whitespace cleanup
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def normalize_address(address: str) -> str:
    """Normalize address strings while keeping numbers, postal codes, and key features."""
    cleaned = clean_text(address)
    if not cleaned:
        return ""
    
    # Replace & -> and
    cleaned = re.sub(r'\s*&\s*', ' and ', cleaned)
    
    # Replace non-alphanumeric (keep numbers, letters, whitespace)
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    
    # Apply abbreviation replacements
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, repl, cleaned)
        
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_postal_code(address: str) -> str:
    """Extract potential PIN / Zip / Postal code from address (digits of length 5-7)."""
    if not isinstance(address, str):
        return ""
    matches = re.findall(r'\b\d{5,7}\b', address)
    return matches[0] if matches else ""

def extract_numbers(text: str) -> set:
    """Extract numeric tokens from string for number matching."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))
