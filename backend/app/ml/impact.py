import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from backend.app.data.schema import yield_unit
from backend.app.ml.features import ALL_FEATURES
from backend.app.ml.yield_model import get_yield_predictor
from backend.app.schemas.analysis import (
    ImpactAnalysisRequest,
    ImpactAnalysisResponse,
    GlobalImportance,
    SensitivityFactor,
    CompareRequest,
    CompareResponse,
    YieldGapFactor
)

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "rainfall": "Rainfall (mm)",
    "temperature": "Temperature (°C)",
    "n": "Nitrogen (N kg/ha)",
    "fertilizer": "Fertilizer (kg/ha)",
    "ph": "Soil pH",
    "p": "Phosphorus (P kg/ha)",
    "k": "Potassium (K kg/ha)",
    "humidity": "Humidity (%)",
    "pesticide": "Pesticide (kg/ha)",
    "area": "Cultivated area (ha)",
    "crop_year": "Crop year (trend)",
    "state": "State (location)",
    "season": "Season",
}

WHAT_IF_FACTORS = ["rainfall", "n", "temperature", "fertilizer", "ph", "p", "k", "pesticide"]
COMPARE_FACTORS = ["rainfall", "n", "p", "k", "fertilizer", "ph", "temperature", "humidity", "pesticide"]
MAX_IMPORTANCE_ROWS = 400

_IMPORTANCE_CACHE: Dict[str, List[GlobalImportance]] = {}


def compute_global_importance(crop: str) -> List[GlobalImportance]:
    """
    Permutation importance of each input for `crop`, measured as the drop in R² (log-yield
    scale) when that column is shuffled, on up to 400 of the crop's historical rows.
    Results are cached per crop and normalised to percentages of the total positive drop.
    """
    if crop in _IMPORTANCE_CACHE:
        return _IMPORTANCE_CACHE[crop]

    from backend.app.data.ingest import get_active_dataframe
    predictor = get_yield_predictor()
    df = get_active_dataframe()
    subset = df[df["crop"] == crop]
    if len(subset) < 20:
        subset = df
    subset = subset.sample(n=min(len(subset), MAX_IMPORTANCE_ROWS), random_state=42)

    features = [f for f in ALL_FEATURES if f != "crop"]
    try:
        result = permutation_importance(
            predictor.pipeline, subset[ALL_FEATURES], np.log1p(subset["yield"]),
            n_repeats=5, random_state=42, scoring="r2"
        )
    except Exception as e:
        logger.warning(f"Permutation importance failed for {crop}: {e}")
        return []

    raw = {f: max(0.0, float(result.importances_mean[ALL_FEATURES.index(f)])) for f in features}
    total = sum(raw.values())
    if total <= 0:
        return []
    importances = sorted(
        (GlobalImportance(feature=FEATURE_LABELS.get(f, f), importance=round(v / total * 100, 1))
         for f, v in raw.items() if v > 0),
        key=lambda g: g.importance, reverse=True
    )
    _IMPORTANCE_CACHE[crop] = importances
    return importances


def clear_importance_cache() -> None:
    _IMPORTANCE_CACHE.clear()


def compute_what_if_sensitivity(data: Dict[str, Any]) -> List[SensitivityFactor]:
    """
    Varies one factor at a time by ±20% (±10% for pH) around the farm profile and
    reports the predicted yield at each end.
    """
    predictor = get_yield_predictor()
    profile = predictor.complete_profile(data)
    baseline_pred = predictor.predict_one(profile).value

    sensitivities: List[SensitivityFactor] = []
    for factor in WHAT_IF_FACTORS:
        base_val = float(profile.get(factor, 0.0))
        if base_val <= 0:
            continue
        step = 0.1 if factor == "ph" else 0.2
        plus = dict(profile, **{factor: min(14.0, base_val * (1 + step)) if factor == "ph" else base_val * (1 + step)})
        minus = dict(profile, **{factor: base_val * (1 - step)})
        pred_plus = predictor.predict_one(plus).value
        pred_minus = predictor.predict_one(minus).value

        sensitivities.append(
            SensitivityFactor(
                factor=FEATURE_LABELS.get(factor, factor.title()),
                baseline_value=round(base_val, 2),
                current_predicted_yield=round(baseline_pred, 2),
                plus_20_predicted_yield=round(pred_plus, 2),
                minus_20_predicted_yield=round(pred_minus, 2),
                sensitivity_delta=round(pred_plus - pred_minus, 3)
            )
        )

    sensitivities.sort(key=lambda s: abs(s.sensitivity_delta), reverse=True)
    return sensitivities


def perform_impact_analysis(req: ImpactAnalysisRequest) -> ImpactAnalysisResponse:
    """Combines global (permutation) importance and local what-if sensitivity."""
    data_dict = req.model_dump()
    global_imp = compute_global_importance(req.crop)
    sensitivities = compute_what_if_sensitivity(data_dict)
    unit = yield_unit(req.crop)

    if sensitivities and abs(sensitivities[0].sensitivity_delta) > 0:
        top = sensitivities[0]
        summary = (
            f"For {req.crop}, the model's predicted yield is most sensitive to {top.factor}: moving it "
            f"from -20% to +20% changes the prediction by {top.sensitivity_delta:+.2f} {unit}."
        )
    else:
        summary = f"For {req.crop}, the model's prediction barely changes when single inputs are varied ±20%."
    if global_imp:
        summary += f" Across historical {req.crop} records, the most important input is {global_imp[0].feature}."
    summary += " These are model associations from historical data, not causal effects."

    return ImpactAnalysisResponse(
        crop=req.crop,
        yield_unit=unit,
        global_importance=global_imp,
        what_if_sensitivities=sensitivities,
        summary=summary
    )


def compare_farm_profiles(req: CompareRequest) -> CompareResponse:
    """
    Explains the yield gap between the first (reference) farm profile and the other profile
    whose yield differs most from it (with two profiles: farm 2 minus farm 1). Each factor's
    impact is the predicted yield change when only that factor of the reference farm is set
    to the other farm's value.
    """
    predictor = get_yield_predictor()
    crop = req.profiles[0].crop
    unit = yield_unit(crop)

    evaluated = []
    for profile in req.profiles:
        filled = predictor.complete_profile(profile.model_dump())
        y = profile.actual_yield if profile.actual_yield is not None else predictor.predict_one(filled).value
        evaluated.append({"name": profile.name, "yield": round(float(y), 2), "conditions": filled,
                          "yield_source": "actual" if profile.actual_yield is not None else "predicted"})

    ref = evaluated[0]
    other = max(evaluated[1:], key=lambda e: abs(e["yield"] - ref["yield"]))
    yield_gap = round(other["yield"] - ref["yield"], 2)
    base_pred = predictor.predict_one(ref["conditions"]).value

    ranked_factors: List[YieldGapFactor] = []
    for factor in COMPARE_FACTORS:
        v1 = float(ref["conditions"][factor])
        v2 = float(other["conditions"][factor])
        diff = round(v2 - v1, 2)
        if abs(diff) < 1e-4:
            continue
        sim_yield = predictor.predict_one(dict(ref["conditions"], **{factor: v2})).value
        impact = round(sim_yield - base_pred, 2)
        direction = "Positive Driver" if impact > 0 else ("Negative Drag" if impact < 0 else "Neutral")
        ranked_factors.append(
            YieldGapFactor(
                factor=FEATURE_LABELS.get(factor, factor.title()),
                farm1_val=v1,
                farm2_val=v2,
                difference=diff,
                estimated_impact_on_yield=impact,
                direction=direction,
                explanation=(
                    f"{other['name']}'s {FEATURE_LABELS.get(factor, factor)} ({v2:g}) instead of {ref['name']}'s "
                    f"({v1:g}) {'raises' if impact > 0 else 'lowers' if impact < 0 else 'does not change'} the "
                    f"predicted yield" + (f" by {abs(impact):.2f} {unit}." if impact else ".")
                )
            )
        )

    ranked_factors.sort(key=lambda x: abs(x.estimated_impact_on_yield), reverse=True)
    top = ranked_factors[0].factor if ranked_factors else "none of the compared inputs"
    summary = (
        f"Comparing {ref['name']} ({ref['yield']:.2f} {unit}, {ref['yield_source']}) with {other['name']} "
        f"({other['yield']:.2f} {unit}, {other['yield_source']}) for {crop}: the yield gap is {yield_gap:+.2f} {unit}. "
        f"The largest single-factor contribution in the model comes from {top}."
    )

    return CompareResponse(
        crop=crop,
        yield_unit=unit,
        profiles=evaluated,
        yield_gap=yield_gap,
        ranked_factors=ranked_factors,
        comparison_summary=summary
    )
