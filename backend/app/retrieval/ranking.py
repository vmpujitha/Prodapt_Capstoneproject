import math
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.documents import Document

from backend.app.config import settings
from backend.app.schemas.search import CaseDocument, ScoreBreakdown
from backend.app.feedback.reranker import crop_feedback_key

# Normalization ranges for condition proximity calculation
CONDITION_SCALES = {
    "n": 250.0,
    "p": 120.0,
    "k": 120.0,
    "ph": 6.0,
    "temperature": 35.0,
    "rainfall": 2000.0,
    "humidity": 80.0
}

def compute_condition_proximity(
    doc_metadata: Dict[str, Any],
    user_conditions: Optional[Dict[str, Optional[float]]]
) -> float:
    """
    Computes condition proximity: 1 - normalized Euclidean distance across fields supplied by user.
    If user provided no numerical conditions, returns a neutral 0.5.
    """
    if not user_conditions:
        return 0.5

    diff_sq_sum = 0.0
    active_count = 0

    for key, scale in CONDITION_SCALES.items():
        user_val = user_conditions.get(key)
        if user_val is not None and not math.isnan(user_val):
            doc_val = doc_metadata.get(key)
            if doc_val is not None and not math.isnan(doc_val):
                normalized_diff = (float(user_val) - float(doc_val)) / scale
                diff_sq_sum += (normalized_diff ** 2)
                active_count += 1

    if active_count == 0:
        return 0.5

    dist = math.sqrt(diff_sq_sum / active_count)
    proximity = max(0.0, 1.0 - dist)
    return round(proximity, 4)

_YEAR_RANGE: Optional[Tuple[int, int]] = None
_PREVALENCE_CACHE: Dict[str, Any] = {"df_id": None, "map": {}}


def is_aggregate_crop(crop: Optional[str]) -> bool:
    """Dataset roll-up labels such as 'Oilseeds Total' or 'Other Kharif Pulses' are not recommendable crops."""
    name = (crop or "").strip().lower()
    return name.startswith("other ") or name.endswith(" total") or name == "total"


def _prevalence_map() -> Dict[Tuple[str, str, str], float]:
    """
    (state, season, crop) -> sqrt(crop's planted area / area of the most-planted crop in that
    state and season), in (0, 1]. The square root softens the long tail so a minor crop is not
    ranked near zero. Rebuilt only when the active dataframe object changes.
    """
    try:
        from backend.app.data.ingest import get_active_dataframe
        df = get_active_dataframe()
    except Exception:
        return {}
    if _PREVALENCE_CACHE["df_id"] == id(df):
        return _PREVALENCE_CACHE["map"]
    result: Dict[Tuple[str, str, str], float] = {}
    if df is not None and {"state", "season", "crop", "area"} <= set(df.columns):
        data = df[~df["crop"].map(is_aggregate_crop)]
        area = data.groupby(["state", "season", "crop"])["area"].sum()
        peak = area.groupby(level=["state", "season"]).transform("max")
        for key, value in (area / peak.where(peak > 0)).dropna().items():
            result[key] = round(math.sqrt(float(value)), 4)
    _PREVALENCE_CACHE.update(df_id=id(df), map=result)
    return result


def get_dataset_year_range() -> Tuple[int, int]:
    """(min, max) crop_year of the active dataset, read from thresholds.json."""
    global _YEAR_RANGE
    if _YEAR_RANGE is None:
        from backend.app.data.features import load_thresholds
        years = load_thresholds(settings.THRESHOLDS_PATH).get("crop_year", {})
        _YEAR_RANGE = (int(years.get("min", 1997)), int(years.get("max", 2020)))
    return _YEAR_RANGE


def reset_year_range_cache() -> None:
    global _YEAR_RANGE
    _YEAR_RANGE = None


def compute_recency_score(crop_year: Optional[int], min_year: Optional[int] = None,
                          max_year: Optional[int] = None) -> float:
    """
    Linear recency score in [0.1, 1.0]: the dataset's earliest year scores 0.1 and its
    latest year (or anything newer) scores 1.0.
    """
    if crop_year is None:
        return 0.5
    if min_year is None or max_year is None:
        min_year, max_year = get_dataset_year_range()
    if max_year <= min_year:
        return 1.0
    if crop_year >= max_year:
        return 1.0
    if crop_year <= min_year:
        return 0.1
    return round(0.1 + 0.9 * (crop_year - min_year) / (max_year - min_year), 4)

def prioritize_cases(
    retrieved_pairs: List[Tuple[Document, float]],
    user_conditions: Optional[Dict[str, Optional[float]]] = None,
    feedback_boosts: Optional[Dict[str, float]] = None
) -> List[CaseDocument]:
    """
    Applies the multi-component ranking formula:
    score = w1*retrieval + w2*proximity + w3*yield_pct + w4*recency + w5*feedback + w6*prevalence
    Roll-up labels (e.g. 'Oilseeds Total') are dropped.
    """
    w1 = settings.RANKING_W1
    w2 = settings.RANKING_W2
    w3 = settings.RANKING_W3
    w4 = settings.RANKING_W4
    w5 = settings.RANKING_W5
    w6 = settings.RANKING_W6
    total_w = w1 + w2 + w3 + w4 + w5 + w6
    prevalence_map = _prevalence_map() if w6 > 0 else {}

    feedback_boosts = feedback_boosts or {}
    results: List[CaseDocument] = []

    for doc, ret_score in retrieved_pairs:
        meta = doc.metadata
        case_id = meta.get("case_id", "UNKNOWN")
        if is_aggregate_crop(meta.get("crop")):
            continue

        # 1. Retrieval score [0, 1]
        norm_retrieval = max(0.0, min(1.0, float(ret_score)))

        # 2. Condition proximity [0, 1]
        proximity = compute_condition_proximity(meta, user_conditions)

        # 3. Yield percentile [0, 1]
        yield_pct_raw = float(meta.get("yield_percentile_within_crop", 50.0))
        norm_yield_pct = max(0.0, min(1.0, yield_pct_raw / 100.0))

        # 4. Recency [0, 1]
        recency = compute_recency_score(meta.get("crop_year"))

        # 5. Feedback boost: case-level + crop-level, bounded to [-0.1, +0.1], mapped to [0, 1]
        raw_fb = feedback_boosts.get(case_id, 0.0) + feedback_boosts.get(crop_feedback_key(meta.get("crop")), 0.0)
        raw_fb = max(-0.1, min(0.1, raw_fb))
        norm_fb = max(0.0, min(1.0, 0.5 + (raw_fb * 5.0)))

        # 6. Prevalence: how widely this crop is grown in the case's state and season (neutral 0.5 if unknown)
        prevalence = prevalence_map.get((meta.get("state"), meta.get("season"), meta.get("crop")), 0.5)

        # Composite score
        final_score = (
            w1 * norm_retrieval +
            w2 * proximity +
            w3 * norm_yield_pct +
            w4 * recency +
            w5 * norm_fb +
            w6 * prevalence
        ) / total_w

        breakdown = ScoreBreakdown(
            retrieval=round(norm_retrieval, 3),
            proximity=round(proximity, 3),
            yield_pct=round(norm_yield_pct, 3),
            recency=round(recency, 3),
            feedback=round(norm_fb, 3),
            prevalence=round(prevalence, 3),
            final=round(final_score, 3)
        )

        results.append(
            CaseDocument(
                case_id=case_id,
                narrative=doc.page_content,
                metadata=meta,
                scores=breakdown
            )
        )

    # Sort by final score descending
    results.sort(key=lambda x: x.scores.final, reverse=True)
    return results


def diversify_by_crop(cases: List[CaseDocument], top_k: int, max_per_crop: int = 3) -> List[CaseDocument]:
    """
    Picks the top_k highest-scoring cases while allowing at most `max_per_crop` per crop, so
    one crop cannot crowd out the alternatives. If fewer than top_k cases remain after the
    cap, the best skipped cases are added back.
    """
    picked: List[CaseDocument] = []
    skipped: List[CaseDocument] = []
    per_crop: Dict[str, int] = {}
    for case in cases:
        crop = case.metadata.get("crop", "")
        if per_crop.get(crop, 0) < max_per_crop:
            picked.append(case)
            per_crop[crop] = per_crop.get(crop, 0) + 1
        else:
            skipped.append(case)
        if len(picked) == top_k:
            return picked
    return picked + skipped[: top_k - len(picked)]
