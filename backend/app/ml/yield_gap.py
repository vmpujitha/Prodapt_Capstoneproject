"""
Yield-gap analysis for "why do yields differ" questions: compares groups of historical
records of one crop (two or more named states, or a crop's best vs worst seasons) and uses
the yield model to rank which conditions account for the gap.
"""
import logging
from typing import List, Optional

import pandas as pd

from backend.app.data.ingest import df_to_documents
from backend.app.data.schema import yield_unit
from backend.app.ml.impact import COMPARE_FACTORS, compare_farm_profiles
from backend.app.schemas.analysis import (
    CompareRequest, FarmProfile, YieldGapAnalysis, YieldGapGroup,
)

logger = logging.getLogger(__name__)

MIN_GROUP_ROWS = 3
PROFILE_FIELDS = COMPARE_FACTORS + ["area"]

STATE_CAVEAT = (
    "Soil N/P/K/pH in this dataset is one value per state, so soil differences between states are "
    "confounded with everything else that differs between them (irrigation, varieties, practices). "
    "Factor contributions are model associations, not proven causes."
)
QUARTILE_CAVEAT = (
    "High- and low-yield seasons are compared within the same historical records; factor contributions "
    "are model associations, not proven causes."
)


def _group(name: str, frame: pd.DataFrame) -> YieldGapGroup:
    conditions = {f: round(float(frame[f].median()), 2) for f in PROFILE_FIELDS if f in frame}
    conditions["season"] = str(frame["season"].mode().iloc[0])
    conditions["state"] = str(frame["state"].mode().iloc[0])
    recent = frame.sort_values("crop_year", ascending=False)
    examples = [{"case_id": d.metadata["case_id"], "narrative": d.page_content, "metadata": d.metadata}
                for d in df_to_documents(recent.head(3))]
    return YieldGapGroup(
        name=name,
        case_count=int(len(frame)),
        median_yield=round(float(frame["yield"].median()), 2),
        conditions=conditions,
        example_case_ids=recent["case_id"].head(3).tolist(),
        example_cases=examples,
    )


def analyze_yield_gap(df: pd.DataFrame, crop: str, states: List[str], season: Optional[str] = None,
                      state: Optional[str] = None) -> Optional[YieldGapAnalysis]:
    """
    Builds the yield-gap analysis for `crop`. With two or more `states`, each state's records form
    a group (first state = reference). Otherwise the crop's bottom-25% and top-25% yield seasons
    (within `state` if it has enough records) are compared. Returns None when there is too little data.
    """
    subset = df[df["crop"] == crop]
    if season and (subset["season"] == season).sum() >= MIN_GROUP_ROWS:
        subset = subset[subset["season"] == season]
    else:
        season = None
    if subset.empty:
        return None

    groups: List[YieldGapGroup] = []
    if len(states) >= 2:
        for s in states:
            rows = subset[subset["state"] == s]
            if len(rows) >= MIN_GROUP_ROWS:
                groups.append(_group(s, rows))
        grouping, caveat = "state", STATE_CAVEAT

    if len(groups) < 2:
        base = subset[subset["state"] == state] if state and (subset["state"] == state).sum() >= 8 else subset
        q25, q75 = base["yield"].quantile([0.25, 0.75])
        low, high = base[base["yield"] <= q25], base[base["yield"] >= q75]
        if len(low) < MIN_GROUP_ROWS or len(high) < MIN_GROUP_ROWS:
            return None
        where = f" in {state}" if base is not subset else ""
        groups = [_group(f"Low-yield seasons{where} (bottom 25%)", low),
                  _group(f"High-yield seasons{where} (top 25%)", high)]
        grouping, caveat = "yield_quartile", QUARTILE_CAVEAT

    profiles = [
        FarmProfile(name=g.name, crop=crop, season=g.conditions["season"], state=g.conditions["state"],
                    actual_yield=g.median_yield, **{f: g.conditions[f] for f in PROFILE_FIELDS})
        for g in groups
    ]
    comparison = compare_farm_profiles(CompareRequest(profiles=profiles[:5]))
    ref = groups[0]
    # compare_farm_profiles compares the reference with the group whose yield differs most from it
    other_name = max(comparison.profiles[1:], key=lambda p: abs(p["yield"] - ref.median_yield))["name"]
    other = next(g for g in groups if g.name == other_name)
    unit = yield_unit(crop)

    movers = [f for f in comparison.ranked_factors if f.estimated_impact_on_yield != 0][:3]
    drivers = " ".join(f.explanation for f in movers) or "none of the measured soil, weather or input differences moves the model's prediction noticeably"
    season_txt = f" ({season})" if season else ""
    summary = (
        f"{crop}{season_txt}: {ref.name} has a median yield of {ref.median_yield:.2f} {unit} over {ref.case_count} "
        f"records versus {other.median_yield:.2f} {unit} for {other.name} ({other.case_count} records), "
        f"a gap of {comparison.yield_gap:+.2f} {unit}. Largest contributing differences (model estimates): {drivers}"
    )
    return YieldGapAnalysis(
        crop=crop, season=season, grouping=grouping, yield_unit=unit, groups=groups,
        yield_gap=comparison.yield_gap, ranked_factors=comparison.ranked_factors,
        summary=summary, caveat=caveat,
    )
