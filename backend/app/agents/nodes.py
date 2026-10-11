import time
import statistics
import logging
from collections import Counter
from typing import Dict, Any, List, Optional

from fastapi import HTTPException

from backend.app.agents.state import AdvisoryState
from backend.app.guardrails.input_validation import (
    sanitize_and_validate_query_text,
    is_domain_relevant,
    validate_numeric_bounds,
    fuzzy_match_token,
    mentioned_values,
    uncovered_state_mentions,
    KNOWN_CROPS,
    KNOWN_SEASONS,
    KNOWN_STATES,
)
from backend.app.guardrails.output_validation import DISCLAIMER_TEXT
from backend.app.agents.classifier import classify_query, extract_conditions_regex
from backend.app.retrieval.hybrid import search_with_mode
from backend.app.agents.recommender import aggregate_crop_recommendations, generate_crop_recommendation
from backend.app.agents.judge import evaluate_recommendation
from backend.app.ml.yield_model import get_yield_predictor
from backend.app.ml.risk_model import get_risk_predictor
from backend.app.ml.impact import perform_impact_analysis
from backend.app.ml.yield_gap import analyze_yield_gap
from backend.app.data.ingest import get_active_dataframe
from backend.app.feedback.reranker import get_feedback_boost_map
from backend.app.schemas.classification import QueryClassification
from backend.app.schemas.search import MetadataFilters, CaseDocument
from backend.app.schemas.analysis import ImpactAnalysisRequest
from backend.app.schemas.prediction import YieldPredictionInterval, RiskAssessment

logger = logging.getLogger(__name__)

CONDITION_FIELDS = ["n", "p", "k", "ph", "temperature", "rainfall", "humidity", "fertilizer", "pesticide"]
MAX_CANDIDATE_CROPS = 5


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def user_conditions_from(clf: Optional[QueryClassification]) -> Dict[str, float]:
    """Numeric soil/weather/input values the user actually stated."""
    if not clf:
        return {}
    values = {**clf.soil.model_dump(), **clf.weather.model_dump(), **clf.inputs.model_dump()}
    return {f: float(values[f]) for f in CONDITION_FIELDS if values.get(f) is not None}


def is_yield_gap_question(clf: Optional[QueryClassification]) -> bool:
    return bool(clf and clf.intent == "explain_yield_difference")


def classification_warnings(clf: Optional[QueryClassification], query: str = "") -> List[str]:
    """
    'Did you mean' messages for crop/season/state values not present in the dataset, and
    notices for Indian states named in the query that the dataset does not cover.
    """
    warnings = [f"'{s}' is not covered by the historical dataset ({len(KNOWN_STATES)} states); "
                f"results are drawn from other states." for s in uncovered_state_mentions(query)]
    if not clf:
        return warnings
    for label, value, choices in (("crop", clf.crop, KNOWN_CROPS), ("season", clf.season, KNOWN_SEASONS),
                                  ("state", clf.state, KNOWN_STATES)):
        if value and value not in choices and not (label == "state" and value in uncovered_state_mentions(query)):
            _, suggestion = fuzzy_match_token(value, choices)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            warnings.append(f"Unknown {label} '{value}' (not in the historical dataset); it was not used as a filter.{hint}")
    return warnings


def excluded_crops(clf: Optional[QueryClassification]) -> List[str]:
    """For alternative-crop questions, the crop the user wants to replace is excluded."""
    if clf and clf.intent == "alternative_crop" and clf.crop:
        return [clf.crop]
    return []


def candidate_crops(state: AdvisoryState) -> List[str]:
    """Crops the recommender will rank, in recommendation order."""
    clf = state.get("classification")
    recs = aggregate_crop_recommendations(state.get("ranked_cases") or [], exclude_crops=excluded_crops(clf))
    crops = [r.crop for r in recs][:MAX_CANDIDATE_CROPS]
    if not crops and clf and clf.crop in KNOWN_CROPS:
        crops = [clf.crop]
    return crops


def farm_profile(crop: str, clf: Optional[QueryClassification], ranked_cases: List[CaseDocument]) -> Dict[str, Any]:
    """
    Model input profile for `crop`: values the user stated, with the rest taken from the
    retrieved similar cases of that crop (median numerics, most common season/state).
    Anything still missing is filled by the model from the crop's historical medians.
    """
    profile: Dict[str, Any] = {"crop": crop, **user_conditions_from(clf)}
    crop_cases = [c for c in ranked_cases if c.metadata.get("crop") == crop] or ranked_cases

    for field in ("season", "state"):
        stated = getattr(clf, field, None) if clf else None
        known = KNOWN_SEASONS if field == "season" else KNOWN_STATES
        if stated in known:
            profile[field] = stated
        elif crop_cases:
            profile[field] = Counter(c.metadata.get(field) for c in crop_cases).most_common(1)[0][0]

    for field in CONDITION_FIELDS + ["area"]:
        if field not in profile:
            vals = [float(c.metadata[field]) for c in crop_cases if c.metadata.get(field) is not None]
            if vals:
                profile[field] = statistics.median(vals)
    return profile


def build_filters(explicit: MetadataFilters, clf: Optional[QueryClassification], level: int) -> MetadataFilters:
    """
    Merges explicit request filters (always kept) with classifier-derived ones.
    level 0: + crop/season/state + derived soil/weather classes
    level 1: + crop/season/state
    level 2: + state only
    level 3: explicit filters only
    Classifier values are used only if they exist in the dataset; for alternative-crop
    questions the user's current crop is never used as a filter.
    """
    def known(value, choices):
        return value if value in choices else None

    derived: Dict[str, Any] = {}
    if clf and level <= 2:
        derived["state"] = known(clf.state, KNOWN_STATES)
    if clf and level <= 1:
        derived["season"] = known(clf.season, KNOWN_SEASONS)
    # The crop filter is kept longest; for yield-gap questions it is never relaxed, since cases of
    # other crops cannot explain a gap for this one.
    if clf and clf.intent != "alternative_crop" and (level <= 1 or clf.intent == "explain_yield_difference"):
        derived["crop"] = known(clf.crop, KNOWN_CROPS)
    if clf and level == 0:
        derived.update(
            ph_class=clf.soil.ph_class, n_level=clf.soil.n_level, p_level=clf.soil.p_level,
            k_level=clf.soil.k_level, temp_class=clf.weather.temp_class,
            rainfall_class=clf.weather.rainfall_class, humidity_class=clf.weather.humidity_class,
        )

    merged = explicit.model_dump()
    for key, value in derived.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return MetadataFilters(**merged)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def guardrail_node(state: AdvisoryState) -> Dict[str, Any]:
    """Input guardrails before any LLM call: text checks, injection screen, domain check, numeric bounds."""
    t0 = time.time()
    query = state.get("query", "")

    try:
        cleaned_query = sanitize_and_validate_query_text(query)
        if not is_domain_relevant(cleaned_query):
            return {
                "query": cleaned_query,
                "is_valid": False,
                "out_of_scope": True,
                "rejection_reason": "Query does not appear related to agriculture, crops, soil, or weather.",
                "trace": [{"node": "guardrail_node", "ms": _ms(t0), "summary": "Query out of agricultural scope"}]
            }

        # Deterministic extraction (no LLM) of numbers stated in the text, checked against physical ranges.
        extracted = extract_conditions_regex(cleaned_query)
        validate_numeric_bounds(
            n=extracted.soil.n, p=extracted.soil.p, k=extracted.soil.k, ph=extracted.soil.ph,
            temperature=extracted.weather.temperature, rainfall=extracted.weather.rainfall,
            humidity=extracted.weather.humidity
        )
        return {
            "query": cleaned_query,
            "is_valid": True,
            "rejection_reason": None,
            "trace": [{"node": "guardrail_node", "ms": _ms(t0), "summary": "Input validation & security screening passed"}]
        }
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {"code": "VALIDATION_FAILED", "message": str(e.detail)}
        return {
            "is_valid": False,
            "rejection_reason": detail.get("message"),
            "rejection_code": detail.get("code", "VALIDATION_FAILED"),
            "rejection_status": e.status_code,
            "trace": [{"node": "guardrail_node", "ms": _ms(t0), "summary": f"Validation rejected: {detail.get('message')}"}]
        }


def classifier_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    clf, llm_used, fb_triggered, degraded = classify_query(
        query=state["query"],
        preferred_provider=state.get("preferred_llm_provider"),
        preferred_model=state.get("preferred_llm_model")
    )
    summary = f"Intent: {clf.intent}, Crop: {clf.crop}, State: {clf.state}, Season: {clf.season}"
    update: Dict[str, Any] = {
        "classification": clf,
        "warnings": classification_warnings(clf, state["query"]),
        "llm_used": llm_used,
        "fallback_triggered": fb_triggered,
        "degraded_mode": degraded,
        "trace": [{"node": "classifier_node", "ms": _ms(t0), "summary": summary}]
    }
    if clf.intent == "out_of_scope":
        update["out_of_scope"] = True
    return update


def _retrieve_with_relaxation(state: AdvisoryState, clf: Optional[QueryClassification], explicit: MetadataFilters,
                              top_k: int) -> Dict[str, Any]:
    """Runs search, relaxing classifier-derived filters until at least min(3, top_k) cases match."""
    min_results = min(3, top_k)
    out: Dict[str, Any] = {}
    for level in range(4):
        filters = build_filters(explicit, clf, level)
        cases, mode_used, note = search_with_mode(
            query=state["query"], top_k=top_k, mode=state.get("mode", "hybrid"), filters=filters,
            user_conditions=user_conditions_from(clf), feedback_boosts=get_feedback_boost_map(),
            provider=state.get("preferred_embed_provider"), model=state.get("preferred_embed_model")
        )
        out = {"cases": cases, "mode": mode_used, "note": note, "filters": filters, "level": level}
        if len(cases) >= min_results:
            break
    return out


def retriever_node(state: AdvisoryState) -> Dict[str, Any]:
    """
    Hybrid retrieval + prioritization with progressive filter relaxation. When a yield-gap
    question names several states (e.g. "Punjab vs Bihar"), a balanced set of cases is
    retrieved for each state.
    """
    t0 = time.time()
    clf = state.get("classification")
    top_k = state.get("top_k", 8)
    explicit = state.get("filters") or MetadataFilters()

    states = mentioned_values(state["query"], KNOWN_STATES)
    if clf and clf.intent == "explain_yield_difference" and len(states) > 1 and not explicit.state:
        states = states[:3]
        per_state = max(1, -(-top_k // len(states)))
        runs = [_retrieve_with_relaxation(state, clf.model_copy(update={"state": s}), explicit, per_state)
                for s in states]
        ranked_cases = sorted((c for r in runs for c in r["cases"]), key=lambda c: c.scores.final, reverse=True)[:top_k]
        run = runs[0]
        active = {k: v for k, v in run["filters"].model_dump().items() if v is not None and k != "state"}
        active["state"] = states
        level = max(r["level"] for r in runs)
        notes = {r["note"] for r in runs if r["note"]}
    else:
        run = _retrieve_with_relaxation(state, clf, explicit, top_k)
        ranked_cases = run["cases"]
        active = {k: v for k, v in run["filters"].model_dump().items() if v is not None}
        level = run["level"]
        notes = {run["note"]} if run["note"] else set()

    summary = f"Retrieved and ranked {len(ranked_cases)} cases ({run['mode']} search); filters: {active or 'none'}"
    if level:
        summary += f" (relaxed {level} step(s) to find enough cases)"
    return {
        "ranked_cases": ranked_cases,
        "filters_applied": active,
        "effective_search_mode": run["mode"],
        "warnings": list(state.get("warnings") or []) + sorted(notes),
        "trace": [{"node": "retriever_node", "ms": _ms(t0), "summary": summary}]
    }


def analyst_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    clf = state.get("classification")
    ranked_cases = state.get("ranked_cases") or []
    crops = candidate_crops(state)
    target_crop = clf.crop if clf and clf.crop in KNOWN_CROPS and not excluded_crops(clf) else (crops[0] if crops else None)
    if not target_crop:
        return {"soil_weather_analysis": None,
                "trace": [{"node": "analyst_node", "ms": _ms(t0), "summary": "No crop to analyse (no cases retrieved)"}]}

    profile = get_yield_predictor().complete_profile(farm_profile(target_crop, clf, ranked_cases))
    analysis = perform_impact_analysis(ImpactAnalysisRequest(**profile))
    summary = f"Computed feature importance & what-if sensitivity for {target_crop}"
    update: Dict[str, Any] = {"soil_weather_analysis": analysis}

    if is_yield_gap_question(clf):
        gap = analyze_yield_gap(
            get_active_dataframe(), crop=target_crop,
            states=mentioned_values(state["query"], KNOWN_STATES),
            season=clf.season if clf.season in KNOWN_SEASONS else None,
            state=clf.state if clf.state in KNOWN_STATES else None,
        )
        update["yield_gap"] = gap
        if gap:
            summary += f"; yield-gap analysis ({' vs '.join(g.name for g in gap.groups)}): {gap.yield_gap:+.2f} {gap.yield_unit}"
        else:
            summary += "; not enough records for a yield-gap analysis"

    update["trace"] = [{"node": "analyst_node", "ms": _ms(t0), "summary": summary}]
    return update


def yield_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    clf = state.get("classification")
    ranked_cases = state.get("ranked_cases") or []
    predictor = get_yield_predictor()
    predictions: Dict[str, YieldPredictionInterval] = {
        crop: predictor.predict_one(farm_profile(crop, clf, ranked_cases)) for crop in candidate_crops(state)
    }
    return {
        "yield_predictions": predictions,
        "trace": [{"node": "yield_node", "ms": _ms(t0), "summary": f"Predicted yield intervals for {len(predictions)} crop(s)"}]
    }


def risk_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    clf = state.get("classification")
    ranked_cases = state.get("ranked_cases") or []
    predictor = get_risk_predictor()
    stated = user_conditions_from(clf)
    assessments: Dict[str, RiskAssessment] = {}
    for crop in candidate_crops(state):
        # Envelope warnings only cover conditions the user actually stated.
        assessments[crop] = predictor.assess_risk(farm_profile(crop, clf, ranked_cases), check_fields=stated.keys())
    return {
        "risk_assessments": assessments,
        "trace": [{"node": "risk_node", "ms": _ms(t0), "summary": f"Evaluated suitability risk for {len(assessments)} crop(s)"}]
    }


def recommender_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    clf = state.get("classification")
    ranked_cases = state.get("ranked_cases") or []
    yield_preds = state.get("yield_predictions") or {}
    risk_preds = state.get("risk_assessments") or {}

    gap = state.get("yield_gap")
    if gap:
        ranked_cases = [c for c in ranked_cases if c.metadata.get("crop") == gap.crop] or ranked_cases

    raw_recs, explanation, llm_used, fb_triggered, degraded = generate_crop_recommendation(
        query=state["query"],
        ranked_cases=ranked_cases,
        yield_gap=gap,
        preferred_provider=state.get("preferred_llm_provider"),
        preferred_model=state.get("preferred_llm_model"),
        yield_predictions=yield_preds,
        risk_assessments=risk_preds,
        exclude_crops=excluded_crops(clf),
    )

    enriched_recs = []
    for r in raw_recs:
        r_copy = r.model_copy()
        r_copy.predicted_yield = yield_preds.get(r.crop)
        r_copy.risk = risk_preds.get(r.crop)
        enriched_recs.append(r_copy)

    summary = f"Synthesized {len(enriched_recs)} crop recommendations"
    update: Dict[str, Any] = {
        "recommendations": enriched_recs,
        "explanation": explanation,
        "trace": [{"node": "recommender_node", "ms": _ms(t0), "summary": summary}]
    }
    if llm_used:
        update["llm_used"] = llm_used
    if fb_triggered:
        update["fallback_triggered"] = fb_triggered
    if degraded:
        update["degraded_mode"] = degraded
    return update


def judge_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    verdict, llm_used, fb_triggered, degraded = evaluate_recommendation(
        query=state["query"],
        recommendations=state.get("recommendations") or [],
        explanation=state.get("explanation", ""),
        ranked_cases=state.get("ranked_cases") or [],
        preferred_provider=state.get("preferred_llm_provider"),
        preferred_model=state.get("preferred_llm_model"),
        extra_valid_ids=[cid for g in (state.get("yield_gap").groups if state.get("yield_gap") else []) for cid in g.example_case_ids],
    )
    avg = round((verdict.grounded + verdict.consistent_with_data + verdict.actionable + verdict.safe) / 4, 2)
    update: Dict[str, Any] = {
        "judge_verdict": verdict,
        "revision_count": state.get("revision_count", 0) + 1,
        "trace": [{"node": "judge_node", "ms": _ms(t0), "summary": f"Judge verdict: passed={verdict.passed}, avg_score={avg}"}]
    }
    if fb_triggered:
        update["fallback_triggered"] = fb_triggered
    if degraded:
        update["degraded_mode"] = degraded
    return update


def finalize_node(state: AdvisoryState) -> Dict[str, Any]:
    t0 = time.time()
    update: Dict[str, Any] = {"disclaimer": DISCLAIMER_TEXT}
    if state.get("out_of_scope"):
        update["explanation"] = (
            "This assistant only answers questions about crops, soil, weather and farm inputs, "
            "so no historical cases were retrieved for this query."
        )
        summary = "Out-of-scope query: returned scope notice without retrieval"
    elif not state.get("is_valid", True):
        summary = "Rejected by input guardrails"
    else:
        summary = f"Finalized advisory after {state.get('revision_count', 0)} judge pass(es)"
    update["trace"] = [{"node": "finalize_node", "ms": _ms(t0), "summary": summary}]
    return update
