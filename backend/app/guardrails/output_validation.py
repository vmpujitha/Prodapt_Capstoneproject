import re
from typing import Iterable, List, Optional, Set, Tuple

DISCLAIMER_TEXT = (
    "Disclaimer: This advisory is generated from historical crop, soil, and weather data patterns "
    "for decision support. Field outcomes may vary based on localized micro-climate, pests, and seed variety. "
    "Consult local agronomy extension services for critical inputs."
)

CASE_ID_RE = re.compile(r"CASE-\d{6}")
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\-\[(])")
# Small integers (list numbering, "top 3", "2 crops") carry no data claim.
FREE_INTEGER_MAX = 10
# More than this share of numeric sentences unverifiable -> use the deterministic template.
MAX_UNVERIFIED_SHARE = 0.5


def validate_and_sanitize_citations(text: str, valid_case_ids: List[str]) -> str:
    """Replaces case IDs that are not in the retrieved set with a neutral marker."""
    valid_set: Set[str] = set(valid_case_ids)
    cleaned_text = text
    for cited in set(CASE_ID_RE.findall(text)):
        if cited not in valid_set:
            cleaned_text = cleaned_text.replace(cited, "[Unverified Case Reference]")
    return cleaned_text


def collect_numbers(values: Iterable) -> Set[float]:
    """Flattens numeric values (and numbers inside strings) into a set of floats."""
    allowed: Set[float] = set()
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            allowed.add(float(value))
            if 0 <= value <= 1:
                allowed.add(float(value) * 100)  # scores/probabilities quoted as percentages
        elif isinstance(value, str):
            for match in NUMBER_RE.findall(CASE_ID_RE.sub(" ", value)):
                allowed.add(float(match.replace(",", "")))
    return allowed


def _is_verified(number: float, allowed: Set[float]) -> bool:
    if number.is_integer() and abs(number) <= FREE_INTEGER_MAX:
        return True
    return any(abs(number - a) <= max(0.051, 0.02 * abs(a)) for a in allowed)


def strip_unverified_numbers(text: str, allowed: Set[float]) -> Tuple[str, int, int]:
    """
    Removes sentences quoting numbers that cannot be traced to `allowed`.
    Returns (cleaned_text, removed_sentences, numeric_sentences).
    """
    removed = numeric = 0
    out_lines: List[str] = []
    for line in text.split("\n"):
        kept: List[str] = []
        for sentence in SENTENCE_SPLIT_RE.split(line):
            numbers = [float(n.replace(",", "")) for n in NUMBER_RE.findall(CASE_ID_RE.sub(" ", sentence))]
            if numbers:
                numeric += 1
                if not all(_is_verified(n, allowed) for n in numbers):
                    removed += 1
                    continue
            kept.append(sentence)
        if kept or not line.strip():
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines), removed, numeric


def format_verified_response(
    explanation: str,
    valid_case_ids: List[str],
    degraded_mode: bool = False,
    allowed_numbers: Optional[Set[float]] = None,
    fallback_text: Optional[str] = None,
) -> str:
    """
    Applies the output guardrails to an explanation:
    1. citations of case IDs outside the retrieved set are neutralised;
    2. when `allowed_numbers` is given, sentences quoting untraceable numbers are removed,
       and if most numeric sentences fail (or nothing remains) `fallback_text` is used.
    """
    sanitized = validate_and_sanitize_citations(explanation, valid_case_ids)

    if allowed_numbers is not None and not degraded_mode:
        cleaned, removed, numeric = strip_unverified_numbers(sanitized, allowed_numbers)
        too_many = numeric > 0 and removed / numeric > MAX_UNVERIFIED_SHARE
        if fallback_text and (too_many or not cleaned.strip()):
            return (f"[Output guardrail: the generated explanation quoted {removed} unverifiable figure(s); "
                    f"showing the data-derived summary instead]\n\n{fallback_text}")
        if removed:
            cleaned += f"\n\n_[Output guardrail removed {removed} sentence(s) with figures not found in the retrieved data.]_"
        sanitized = cleaned

    if degraded_mode:
        sanitized = f"[Deterministic Rule-Based Advisory]\n\n{sanitized}"
    return sanitized
