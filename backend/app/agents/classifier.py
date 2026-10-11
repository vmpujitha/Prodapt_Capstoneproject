import re
import logging
from typing import Dict, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate

from backend.app.schemas.classification import (
    QueryClassification,
    SoilCondition,
    WeatherCondition,
    InputCondition
)
from backend.app.guardrails.input_validation import (
    KNOWN_CROPS,
    KNOWN_SEASONS,
    KNOWN_STATES,
    fuzzy_match_token,
    find_crop_mention,
    mentioned_values,
    fuzzy_find_value,
    is_domain_relevant
)
from backend.app.data.features import load_thresholds, classify_ph, classify_value_by_tertile
from backend.app.config import settings
from backend.app.llm.fallback import invoke_with_fallback
from backend.app.agents.prompts import CLASSIFIER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_MONTH_RANGE_RE = re.compile(rf"(?<![a-z]){_MONTH_RE}\.?\s*(?:-|–|to|till|until|through|thru)\s*{_MONTH_RE}(?![a-z])")
_MONTH_SINGLE_RE = re.compile(rf"(?<![a-z]){_MONTH_RE}(?![a-z])")
# Indian cropping calendar: Kharif (monsoon) Jun-Oct, Rabi (winter) Nov-Mar, Summer/Zaid Apr-Jun.
_SEASON_MONTHS = {
    "Kharif": {6, 7, 8, 9, 10},
    "Rabi": {11, 12, 1, 2, 3},
    "Summer": {3, 4, 5, 6},
}


def season_from_months(text: str) -> Optional[str]:
    """
    Maps months stated in the query ("jun-september", "from June to Sept", "in July") to the
    cropping season they fall in. Returns a season only if at least 75% of the months fall in
    one season and no other season fits equally well; otherwise None.
    """
    lowered = text.lower()
    months = set()
    for a, b in _MONTH_RANGE_RE.findall(lowered):
        start, end = _MONTHS[a], _MONTHS[b]
        months.update(range(start, end + 1) if start <= end else list(range(start, 13)) + list(range(1, end + 1)))
    # A bare "may" is usually the verb, so single mentions of it are ignored.
    for m in _MONTH_SINGLE_RE.findall(lowered):
        if m != "may":
            months.add(_MONTHS[m])
    if not months:
        return None

    fits = {s: len(months & ms) / len(months) for s, ms in _SEASON_MONTHS.items()}
    best = max(fits.values())
    winners = [s for s, f in fits.items() if f == best]
    if best < 0.75 or len(winners) != 1:
        return None
    season = winners[0]
    if season == "Summer" and season not in KNOWN_SEASONS:
        season = "Zaid" if "Zaid" in KNOWN_SEASONS else None
    return season if season in KNOWN_SEASONS else None


def extract_conditions_regex(query: str) -> QueryClassification:
    """
    Deterministic rule-based and regex extractor used for Degraded Mode or offline fallback.
    Extracts crops, seasons, states, pH, NPK, rainfall, temperature, humidity, fertilizer, pesticide.
    """
    text = query.lower()
    
    # Check domain relevance
    if not is_domain_relevant(query):
        return QueryClassification(
            intent="out_of_scope",
            confidence=0.9
        )

    # 1. Intent heuristic
    intent = "recommend_crop"
    if any(w in text for w in ["why", "difference", "compare", "yield gap", "explain"]):
        intent = "explain_yield_difference"
    elif any(w in text for w in ["alternative", "substitute", "instead of", "replace"]):
        intent = "alternative_crop"
    elif any(w in text for w in ["similar", "historical", "cases", "examples"]):
        intent = "similar_cases"

    # 2. Extract Crop
    matched_crop = find_crop_mention(text)

    # 3. Extract Season
    seasons = mentioned_values(text, KNOWN_SEASONS)
    matched_season = seasons[0] if seasons else season_from_months(text)

    # 4. Extract State
    states = mentioned_values(text, KNOWN_STATES)
    matched_state = states[0] if states else fuzzy_find_value(query, KNOWN_STATES)

    # 5. Extract Soil (pH, N, P, K)
    ph_val = None
    ph_match = re.search(r"\b(?:ph|soil\s*ph)\s*(?:around|about|is|of|level|[:=~])?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if ph_match:
        try:
            ph_val = float(ph_match.group(1))
        except ValueError:
            pass

    n_val = None
    n_match = re.search(r"\b(?:n|nitrogen)\s*(?:around|about|is|of|level|[:=~])?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if n_match:
        try:
            n_val = float(n_match.group(1))
        except ValueError:
            pass

    p_val = None
    p_match = re.search(r"\b(?:p|phosphorus)\s*(?:around|about|is|of|level|[:=~])?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if p_match:
        try:
            p_val = float(p_match.group(1))
        except ValueError:
            pass

    k_val = None
    k_match = re.search(r"\b(?:k|potassium|potash)\s*(?:around|about|is|of|level|[:=~])?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if k_match:
        try:
            k_val = float(k_match.group(1))
        except ValueError:
            pass

    # 6. Extract Weather (Temp, Rainfall, Humidity)
    temp_val = None
    temp_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:°c|celsius|c|deg\s*c|degrees)\b", text)
    if not temp_match:
        temp_match = re.search(r"\b(?:temp|temperature)\s*[:=~]?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if temp_match:
        try:
            temp_val = float(temp_match.group(1))
        except ValueError:
            pass

    rain_val = None
    rain_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:mm|millimeters?)\b", text)
    if not rain_match:
        rain_match = re.search(r"\b(?:rain|rainfall|precipitation)\s*(?:of|around|about|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if not rain_match:
        rain_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:mm)?\s*(?:rain|rainfall|precipitation)\b", text)
    if rain_match:
        try:
            rain_val = float(rain_match.group(1))
        except ValueError:
            pass

    hum_val = None
    hum_match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:humidity)?\b", text)
    if not hum_match:
        hum_match = re.search(r"\b(?:humidity)\s*[:=~]?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if hum_match:
        try:
            hum_val = float(hum_match.group(1))
        except ValueError:
            pass

    # 7. Extract Inputs
    fert_val = None
    fert_match = re.search(r"\b(?:fertilizer)\s*[:=~]?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if fert_match:
        try:
            fert_val = float(fert_match.group(1))
        except ValueError:
            pass

    pest_val = None
    pest_match = re.search(r"\b(?:pesticide)\s*[:=~]?\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if pest_match:
        try:
            pest_val = float(pest_match.group(1))
        except ValueError:
            pass

    # Derive classes using thresholds
    thresholds = load_thresholds(settings.THRESHOLDS_PATH)
    n_cfg = thresholds.get("n", {"low_max": 60.0, "medium_max": 100.0})
    p_cfg = thresholds.get("p", {"low_max": 40.0, "medium_max": 60.0})
    k_cfg = thresholds.get("k", {"low_max": 35.0, "medium_max": 55.0})
    t_cfg = thresholds.get("temperature", {"cool_max": 22.0, "moderate_max": 28.0})
    r_cfg = thresholds.get("rainfall", {"low_max": 450.0, "moderate_max": 900.0})
    h_cfg = thresholds.get("humidity", {"low_max": 55.0, "moderate_max": 75.0})

    soil = SoilCondition(
        n=n_val,
        p=p_val,
        k=k_val,
        ph=ph_val,
        ph_class=classify_ph(ph_val) if ph_val else None,
        n_level=classify_value_by_tertile(n_val, n_cfg.get("low_max", 60.0), n_cfg.get("medium_max", 100.0)) if n_val else None,
        p_level=classify_value_by_tertile(p_val, p_cfg.get("low_max", 40.0), p_cfg.get("medium_max", 60.0)) if p_val else None,
        k_level=classify_value_by_tertile(k_val, k_cfg.get("low_max", 35.0), k_cfg.get("medium_max", 55.0)) if k_val else None,
    )

    weather = WeatherCondition(
        temperature=temp_val,
        rainfall=rain_val,
        humidity=hum_val,
        temp_class=classify_value_by_tertile(temp_val, t_cfg.get("cool_max", 22.0), t_cfg.get("moderate_max", 28.0), ("cool", "moderate", "hot")) if temp_val else None,
        rainfall_class=classify_value_by_tertile(rain_val, r_cfg.get("low_max", 450.0), r_cfg.get("moderate_max", 900.0), ("low", "moderate", "high")) if rain_val else None,
        humidity_class=classify_value_by_tertile(hum_val, h_cfg.get("low_max", 55.0), h_cfg.get("moderate_max", 75.0), ("low", "moderate", "high")) if hum_val else None,
    )

    inputs = InputCondition(
        fertilizer=fert_val,
        pesticide=pest_val
    )

    return QueryClassification(
        intent=intent,
        crop=matched_crop,
        season=matched_season,
        state=matched_state,
        soil=soil,
        weather=weather,
        inputs=inputs,
        confidence=0.85
    )

def normalize_classification(clf: QueryClassification) -> QueryClassification:
    """
    Fuzzy normalizes extracted categoricals against known ground truth dataset values.
    """
    if clf.crop:
        norm_crop = find_crop_mention(clf.crop)
        if not norm_crop:
            norm_crop, _ = fuzzy_match_token(clf.crop, KNOWN_CROPS)
        if norm_crop:
            clf.crop = norm_crop

    if clf.season:
        norm_season, _ = fuzzy_match_token(clf.season, KNOWN_SEASONS)
        if norm_season:
            clf.season = norm_season

    if clf.state:
        norm_state, _ = fuzzy_match_token(clf.state, KNOWN_STATES)
        if norm_state:
            clf.state = norm_state

    return clf

def merge_with_rule_based(llm: QueryClassification, rules: QueryClassification) -> QueryClassification:
    """
    Fills fields the LLM left empty with the deterministic extraction. Small local models often
    get the intent right but drop explicit details (e.g. crop, state, season); values the LLM
    did return are kept. Intent is only taken from the rules if the LLM said out_of_scope for a
    question the domain check accepted.
    """
    merged = llm.model_copy(deep=True)
    for field in ("crop", "season", "state"):
        if getattr(merged, field) in (None, "") and getattr(rules, field):
            setattr(merged, field, getattr(rules, field))
    for group in ("soil", "weather", "inputs"):
        target, source = getattr(merged, group), getattr(rules, group)
        for name, value in source.model_dump().items():
            if getattr(target, name) is None and value is not None:
                setattr(target, name, value)
    if merged.intent == "out_of_scope" and rules.intent != "out_of_scope":
        merged.intent = rules.intent
    return merged


def classify_query(
    query: str,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None
) -> Tuple[QueryClassification, Optional[Dict[str, str]], bool, bool]:
    """
    Classifies the user's query into agricultural intent and parameter conditions.
    Uses LLM structured output with fallback to regex extractor.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    formatted_prompt = prompt_template.format_messages(query=query)

    def degraded_fallback():
        return extract_conditions_regex(query)

    result, llm_used, fallback_triggered, degraded_mode = invoke_with_fallback(
        prompt=formatted_prompt,
        structured_schema=QueryClassification,
        preferred_provider=preferred_provider,
        preferred_model=preferred_model,
        degraded_fallback_fn=degraded_fallback
    )

    if not isinstance(result, QueryClassification):
        result = extract_conditions_regex(query)
        degraded_mode = True
    elif not degraded_mode:
        result = merge_with_rule_based(result, extract_conditions_regex(query))
        # Months the user explicitly stated are deterministic; small LLMs often miss or misplace them.
        if not mentioned_values(query, KNOWN_SEASONS):
            month_season = season_from_months(query)
            if month_season:
                result.season = month_season

    normalized_result = normalize_classification(result)
    return normalized_result, llm_used, fallback_triggered, degraded_mode
