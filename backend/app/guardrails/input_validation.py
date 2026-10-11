import re
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from fastapi import HTTPException, Request

# In-memory IP rate limiter
_REQUEST_TIMESTAMPS: Dict[str, List[float]] = defaultdict(list)

# Known domain lists for fuzzy matching and verification. These defaults are replaced
# (in place, so imported references stay valid) by the values found in the active
# dataset via `set_known_values`, which the ingestion layer calls on load.
KNOWN_CROPS = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane",
    "Groundnut", "Chickpea", "Mustard", "Millets", "Soybean"
]

KNOWN_SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]

KNOWN_STATES = [
    "Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka",
    "Andhra Pradesh", "Telangana", "Gujarat", "Tamil Nadu", "West Bengal",
    "Madhya Pradesh", "Rajasthan", "Bihar", "Odisha"
]

# Common names / spellings mapped to dataset crop names. Only entries whose target
# exists in the active dataset are used.
CROP_SYNONYMS: Dict[str, str] = {
    "soybean": "Soyabean", "soya": "Soyabean", "chickpea": "Gram", "chana": "Gram",
    "sesame": "Sesamum", "til": "Sesamum", "sorghum": "Jowar", "pearl millet": "Bajra",
    "finger millet": "Ragi", "pigeon pea": "Arhar/Tur", "red gram": "Arhar/Tur", "lentil": "Masoor",
    "mung": "Moong(Green Gram)", "green gram": "Moong(Green Gram)", "black gram": "Urad",
    "chilli": "Dry Chillies", "chillies": "Dry Chillies", "cassava": "Tapioca", "paddy": "Rice",
    "mustard": "Rapeseed &Mustard", "rapeseed": "Rapeseed &Mustard", "cotton": "Cotton(Lint)",
    "millet": "Small Millets", "millets": "Small Millets", "cowpea": "Cowpea(Lobia)",
    "groundnuts": "Groundnut", "peanut": "Groundnut", "pepper": "Black Pepper",
}

# Words too generic to identify a crop on their own when derived from compound names.
_GENERIC_CROP_WORDS = {
    "other", "others", "pulses", "seed", "seeds", "total", "kharif", "rabi", "summer", "lint",
    "oilseeds", "cereals", "beans", "dry", "small", "sweet", "black", "gram", "&", "and",
}

_CROP_ALIASES: Dict[str, str] = {}


def _build_crop_aliases(crops: List[str]) -> Dict[str, str]:
    """Maps lowercase mentions (full names, name parts, synonyms) to dataset crop names."""
    aliases: Dict[str, str] = {}
    known = set(crops)
    for crop in crops:
        aliases[crop.lower()] = crop
        cleaned = re.sub(r"[()/&]", " ", crop.lower())
        aliases.setdefault(" ".join(cleaned.split()), crop)
        for part in re.split(r"[()/&,]", crop.lower()):
            part = " ".join(part.split())
            if part and part not in _GENERIC_CROP_WORDS and len(part) > 2:
                aliases.setdefault(part, crop)
    for synonym, target in CROP_SYNONYMS.items():
        if target in known:
            aliases.setdefault(synonym, target)
    return aliases


def set_known_values(crops: List[str], seasons: List[str], states: List[str]) -> None:
    """Replaces the known crop/season/state vocabularies with values from the active dataset."""
    if crops:
        KNOWN_CROPS[:] = crops
    if seasons:
        KNOWN_SEASONS[:] = seasons
    if states:
        KNOWN_STATES[:] = states
    _CROP_ALIASES.clear()
    _CROP_ALIASES.update(_build_crop_aliases(KNOWN_CROPS))


def find_crop_mention(text: str) -> Optional[str]:
    """Returns the dataset crop name mentioned in `text` (longest alias wins), or None."""
    if not _CROP_ALIASES:
        _CROP_ALIASES.update(_build_crop_aliases(KNOWN_CROPS))
    lowered = text.lower()
    for alias in sorted(_CROP_ALIASES, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", lowered):
            return _CROP_ALIASES[alias]
    return None


def mentioned_values(text: str, choices: List[str]) -> List[str]:
    """Known values (e.g. states) mentioned in `text`, in order of first appearance."""
    lowered = text.lower()
    found = []
    for choice in choices:
        m = re.search(r"(?<![a-z])" + re.escape(choice.lower()) + r"(?![a-z])", lowered)
        if m:
            found.append((m.start(), choice))
    return [choice for _, choice in sorted(found)]


def fuzzy_find_value(text: str, choices: List[str], cutoff: float = 0.8, min_len: int = 5) -> Optional[str]:
    """
    Finds a misspelled mention of a known multi-letter name (e.g. 'Rajastan' -> 'Rajasthan')
    by comparing word uni/bigrams of at least `min_len` characters. Intended for proper nouns
    such as states; too risky for crop names ('price' ~ 'rice').
    """
    import difflib
    words = re.findall(r"[a-zA-Z]+", text)
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    lowered = {c.lower(): c for c in choices}
    for gram in grams:
        if len(gram) < min_len:
            continue
        match = difflib.get_close_matches(gram.lower(), lowered.keys(), n=1, cutoff=cutoff)
        if match:
            return lowered[match[0]]
    return None


def max_crop_year() -> int:
    return datetime.date.today().year + 1


# Reference list of Indian states / UTs, used only to tell users when a place they mention
# has no historical coverage in the active dataset.
INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman And Nicobar Islands", "Chandigarh", "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi", "Jammu And Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]


def uncovered_state_mentions(text: str) -> List[str]:
    """States/UTs named in `text` that do not appear in the active dataset."""
    known = {s.lower() for s in KNOWN_STATES}
    return [s for s in mentioned_values(text, INDIAN_STATES_AND_UTS) if s.lower() not in known]


# Injection & jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"dan\s+mode",
    r"jailbreak",
    r"drop\s+table",
    r"select\s+.*\s+from",
    r"<script\b",
    r"eval\(",
    r"exec\("
]

# Agricultural keywords for domain relevance check
AGRI_KEYWORDS = [
    "crop", "yield", "farm", "soil", "wheat", "rice", "cotton", "maize",
    "sugarcane", "groundnut", "chickpea", "mustard", "millet", "soybean",
    "nitrogen", "phosphorus", "potassium", "ph", "rain", "rainfall", "temperature",
    "humidity", "fertilizer", "pesticide", "kharif", "rabi", "zaid", "harvest",
    "cultivation", "sow", "sowing", "season", "hectare", "plant", "produce", "agriculture",
    "field", "irrigation", "drought", "npk", "punjab", "haryana", "grow", "recommend"
]

def check_rate_limit(client_ip: str, limit_per_minute: int = 60) -> bool:
    """In-memory sliding window rate limiter."""
    now = time.time()
    cutoff = now - 60.0
    timestamps = [t for t in _REQUEST_TIMESTAMPS[client_ip] if t > cutoff]
    timestamps.append(now)
    _REQUEST_TIMESTAMPS[client_ip] = timestamps
    return len(timestamps) <= limit_per_minute

def sanitize_and_validate_query_text(query: str) -> str:
    """
    Validates text length, strips control characters, and blocks prompt injection.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail={"code": "EMPTY_QUERY", "message": "Query cannot be empty or whitespace only."})

    # Strip control characters (keep printable ASCII and standard unicode)
    cleaned = "".join(ch for ch in query if ch.isprintable() or ch in ("\n", "\t", "\r")).strip()

    if len(cleaned) < 5:
        raise HTTPException(status_code=422, detail={"code": "QUERY_TOO_SHORT", "message": "Query must be at least 5 characters long."})

    if len(cleaned) > 1000:
        raise HTTPException(status_code=422, detail={"code": "QUERY_TOO_LONG", "message": "Query exceeds maximum limit of 1000 characters."})

    # Prompt injection check
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail={"code": "SECURITY_VIOLATION", "message": "Input contains disallowed instructional override or injection patterns."}
            )

    return cleaned

def validate_numeric_bounds(
    n: Optional[float] = None,
    p: Optional[float] = None,
    k: Optional[float] = None,
    ph: Optional[float] = None,
    temperature: Optional[float] = None,
    rainfall: Optional[float] = None,
    humidity: Optional[float] = None,
    crop_year: Optional[int] = None
) -> None:
    """
    Validates physical feasibility of agricultural parameters.
    Throws 422 with a specific error message if invalid.
    """
    if ph is not None and not (0.0 <= ph <= 14.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_PH", "message": f"pH value {ph} is out of allowable range (0.0 - 14.0)."})

    if n is not None and not (0.0 <= n <= 500.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_NITROGEN", "message": f"Nitrogen value {n} kg/ha is out of allowable range (0 - 500 kg/ha)."})

    if p is not None and not (0.0 <= p <= 500.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_PHOSPHORUS", "message": f"Phosphorus value {p} kg/ha is out of allowable range (0 - 500 kg/ha)."})

    if k is not None and not (0.0 <= k <= 500.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_POTASSIUM", "message": f"Potassium value {k} kg/ha is out of allowable range (0 - 500 kg/ha)."})

    if temperature is not None and not (-10.0 <= temperature <= 55.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_TEMPERATURE", "message": f"Temperature {temperature}°C is out of allowable range (-10°C to 55°C)."})

    if rainfall is not None and not (0.0 <= rainfall <= 5000.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_RAINFALL", "message": f"Rainfall {rainfall} mm is out of allowable range (0 - 5000 mm)."})

    if humidity is not None and not (0.0 <= humidity <= 100.0):
        raise HTTPException(status_code=422, detail={"code": "INVALID_HUMIDITY", "message": f"Humidity {humidity}% is out of allowable range (0 - 100%)."})

    if crop_year is not None and not (1950 <= crop_year <= max_crop_year()):
        raise HTTPException(status_code=422, detail={"code": "INVALID_YEAR", "message": f"Crop year {crop_year} is out of allowable range (1950 - {max_crop_year()})."})

def is_domain_relevant(query: str) -> bool:
    """Checks if the query has at least one agricultural or climatic keyword."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    for token in tokens:
        if token in AGRI_KEYWORDS:
            return True
    lowered = query.lower()
    if find_crop_mention(query):
        return True
    return any(re.search(r"\b" + re.escape(s.lower()) + r"\b", lowered) for s in KNOWN_STATES)

def fuzzy_match_token(token: str, choices: List[str], min_ratio: float = 0.7) -> Tuple[Optional[str], Optional[str]]:
    """
    Fuzzy matches a token to known choices using Levenshtein distance ratio.
    Returns (exact_or_matched_val, suggestion_if_close).
    """
    import difflib
    token_clean = token.strip()
    matches = difflib.get_close_matches(token_clean, choices, n=1, cutoff=min_ratio)
    if matches:
        return matches[0], None
    
    # Suggestion if slightly lower ratio
    closer = difflib.get_close_matches(token_clean, choices, n=1, cutoff=0.5)
    if closer:
        return None, closer[0]
    return None, None


def _load_vocabulary_from_processed_data() -> None:
    """Initialises the vocabularies from the processed dataset, if it exists, at import time."""
    try:
        import pandas as pd
        from pathlib import Path
        from backend.app.config import settings
        path = Path(settings.PROCESSED_DATA_PATH)
        if path.exists():
            df = pd.read_parquet(path, columns=["crop", "season", "state"])
            set_known_values(sorted(df["crop"].dropna().unique().tolist()),
                             sorted(df["season"].dropna().unique().tolist()),
                             sorted(df["state"].dropna().unique().tolist()))
    except Exception:
        pass  # defaults stay until the ingestion layer calls set_known_values


_load_vocabulary_from_processed_data()
