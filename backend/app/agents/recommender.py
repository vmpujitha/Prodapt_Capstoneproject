import statistics
import logging
from typing import List, Dict, Tuple, Optional, Iterable, Set
from collections import defaultdict
from langchain_core.prompts import ChatPromptTemplate

from backend.app.data.schema import yield_unit
from backend.app.schemas.search import CaseDocument
from backend.app.schemas.recommendation import CropRecommendation
from backend.app.schemas.prediction import YieldPredictionInterval, RiskAssessment
from backend.app.schemas.analysis import YieldGapAnalysis
from backend.app.guardrails.output_validation import format_verified_response, collect_numbers
from backend.app.llm.fallback import invoke_with_fallback
from backend.app.agents.prompts import RECOMMENDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def aggregate_crop_recommendations(
    ranked_cases: List[CaseDocument],
    top_n: int = 5,
    exclude_crops: Optional[Iterable[str]] = None
) -> List[CropRecommendation]:
    """
    Aggregates retrieved cases by crop: case count, mean/median yield, mean yield percentile,
    mean similarity and mean final case score (plus a small bonus for multiple supporting cases).
    """
    excluded = {c.lower() for c in (exclude_crops or [])}
    crop_groups: Dict[str, List[CaseDocument]] = defaultdict(list)
    for case in ranked_cases:
        crop = case.metadata.get("crop", "Unknown")
        if crop.lower() not in excluded:
            crop_groups[crop].append(case)

    recommendations: List[CropRecommendation] = []
    for crop, cases in crop_groups.items():
        yields = [float(c.metadata.get("yield", 0.0)) for c in cases]
        yield_pcts = [float(c.metadata.get("yield_percentile_within_crop", 50.0)) for c in cases]
        sim_scores = [c.scores.retrieval for c in cases]
        final_scores = [c.scores.final for c in cases]

        mean_sim = statistics.mean(sim_scores)
        overall_score = statistics.mean(final_scores)
        mean_yield_pct = statistics.mean(yield_pcts)
        count_boost = min(0.05, (len(cases) - 1) * 0.01)

        recommendations.append(
            CropRecommendation(
                crop=crop,
                score=round(min(1.0, max(0.0, overall_score + count_boost)), 3),
                median_yield=round(statistics.median(yields), 2),
                mean_yield=round(statistics.mean(yields), 2),
                yield_percentile=round(mean_yield_pct, 1),
                similarity=round(mean_sim, 3),
                case_count=len(cases),
                supporting_cases=[c.case_id for c in cases][:5],
                score_breakdown={
                    "retrieval_similarity": round(mean_sim, 3),
                    "mean_case_score": round(overall_score, 3),
                    "historical_yield_percentile": round(mean_yield_pct, 1),
                    "multi_case_bonus": round(count_boost, 3),
                }
            )
        )

    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations[:top_n]


def build_deterministic_recommendation_text(
    recommendations: List[CropRecommendation],
    ranked_cases: List[CaseDocument],
    yield_predictions: Optional[Dict[str, YieldPredictionInterval]] = None,
    risk_assessments: Optional[Dict[str, RiskAssessment]] = None,
) -> str:
    """Template explanation built only from aggregates and model outputs (degraded mode / guardrail fallback)."""
    if not recommendations:
        return "No historical farming cases matched the specified criteria."

    yield_predictions = yield_predictions or {}
    risk_assessments = risk_assessments or {}
    lines = ["### Crop Suitability Analysis (Historical Evidence)\n"]
    for i, rec in enumerate(recommendations, 1):
        unit = yield_unit(rec.crop)
        cases_str = ", ".join(f"[{cid}]" for cid in rec.supporting_cases)
        lines.append(
            f"{i}. **{rec.crop}** (suitability score {rec.score:.2f})\n"
            f"   - Historical median yield of matched cases: {rec.median_yield:.2f} {unit} (mean {rec.mean_yield:.2f} {unit}).\n"
            f"   - Mean yield percentile within {rec.crop} records: {rec.yield_percentile:.1f}.\n"
            f"   - Supporting records: {cases_str} ({rec.case_count} matched cases)."
        )
        pred = yield_predictions.get(rec.crop)
        if pred:
            lines.append(f"   - Model-predicted yield: {pred.value:.2f} {unit} (10th-90th percentile range {pred.low:.2f}-{pred.high:.2f}).")
        risk = risk_assessments.get(rec.crop)
        if risk:
            factor_text = " ".join(f.description for f in risk.factors) or "No out-of-range conditions among the values you gave."
            lines.append(f"   - Low-yield risk: {risk.level} (probability {risk.probability:.2f}). {factor_text}")
        lines.append("")

    top = recommendations[0]
    lines.append(
        f"**Summary**: **{top.crop}** ranks first with a suitability score of {top.score:.2f}, "
        f"based on {top.case_count} matching historical record(s)."
    )
    return "\n".join(lines)


def yield_gap_context(gap: YieldGapAnalysis) -> str:
    """Plain-text rendering of a yield-gap analysis for the LLM prompt (and number verification)."""
    lines = [gap.summary]
    for g in gap.groups:
        cond = ", ".join(f"{k} {v}" for k, v in g.conditions.items())
        lines.append(f"- {g.name}: median yield {g.median_yield} {gap.yield_unit}, {g.case_count} records "
                     f"(e.g. {', '.join(g.example_case_ids)}); median conditions: {cond}")
    lines.append("Factor contributions (model estimate of the yield change if only that factor differed):")
    for f in gap.ranked_factors:
        lines.append(f"- {f.explanation}")
    lines.append(f"Caveat: {gap.caveat}")
    return "\n".join(lines)


def build_yield_gap_text(gap: YieldGapAnalysis) -> str:
    """Template explanation of a yield gap (degraded mode / guardrail fallback)."""
    unit = gap.yield_unit
    lines = ["### Yield Gap Analysis (Historical Evidence)\n"]
    for g in gap.groups:
        cites = ", ".join(f"[{c}]" for c in g.example_case_ids)
        lines.append(f"- **{g.name}**: median yield {g.median_yield:.2f} {unit} across {g.case_count} records (e.g. {cites}).")
    lines.append(f"\n**Gap**: {gap.yield_gap:+.2f} {unit}.\n")
    movers = [f for f in gap.ranked_factors if f.estimated_impact_on_yield != 0]
    if movers:
        lines.append("**Differences that account for it** (model estimate of the yield change from that factor alone):")
        for i, f in enumerate(movers[:5], 1):
            lines.append(f"{i}. {f.explanation}")
    else:
        lines.append("None of the measured soil, weather or input differences changes the model's prediction noticeably; "
                     "the gap is likely driven by factors this dataset does not record.")
    lines.append(f"\n_{gap.caveat}_")
    return "\n".join(lines)


def _allowed_numbers(
    query: str,
    ranked_cases: List[CaseDocument],
    recommendations: List[CropRecommendation],
    yield_predictions: Dict[str, YieldPredictionInterval],
    risk_assessments: Dict[str, RiskAssessment],
) -> Set[float]:
    values: List = [query]
    for case in ranked_cases:
        values.extend(v for v in case.metadata.values() if isinstance(v, (int, float)))
        values.append(case.narrative)
        values.extend(case.scores.model_dump().values())
    for rec in recommendations:
        values.extend([rec.score, rec.median_yield, rec.mean_yield, rec.yield_percentile, rec.similarity, rec.case_count])
        values.extend((rec.score_breakdown or {}).values())
    for pred in yield_predictions.values():
        values.extend([pred.value, pred.low, pred.high])
    for risk in risk_assessments.values():
        values.append(risk.probability)
        values.extend(f.description for f in risk.factors)
    return collect_numbers(values)


def generate_crop_recommendation(
    query: str,
    ranked_cases: List[CaseDocument],
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    yield_predictions: Optional[Dict[str, YieldPredictionInterval]] = None,
    risk_assessments: Optional[Dict[str, RiskAssessment]] = None,
    critique: Optional[str] = None,
    exclude_crops: Optional[Iterable[str]] = None,
    yield_gap: Optional[YieldGapAnalysis] = None,
) -> Tuple[List[CropRecommendation], str, Optional[Dict[str, str]], bool, bool]:
    """
    Aggregates retrieved cases into ranked crop recommendations and asks the LLM for an
    explanation grounded in those aggregates and model outputs. `critique` carries the
    judge's feedback on a previous draft. Output passes the citation and number guardrails.
    """
    yield_predictions = yield_predictions or {}
    risk_assessments = risk_assessments or {}
    recommendations = aggregate_crop_recommendations(ranked_cases, exclude_crops=exclude_crops)
    valid_case_ids = [c.case_id for c in ranked_cases]
    if yield_gap:  # records summarised by the gap analysis are valid evidence too
        valid_case_ids += [cid for g in yield_gap.groups for cid in g.example_case_ids]

    context_str = "\n".join(
        f"Case [{c.case_id}]: {c.narrative} (final rank score {c.scores.final:.2f})" for c in ranked_cases[:6]
    )
    rec_lines = []
    for r in recommendations:
        unit = yield_unit(r.crop)
        line = (f"- {r.crop}: score {r.score}, median yield {r.median_yield} {unit}, "
                f"{r.case_count} supporting cases {r.supporting_cases}")
        if r.crop in yield_predictions:
            p = yield_predictions[r.crop]
            line += f"; model-predicted yield {p.value} {unit} (range {p.low}-{p.high})"
        if r.crop in risk_assessments:
            rk = risk_assessments[r.crop]
            line += f"; low-yield risk {rk.level} (p={rk.probability})"
            if rk.factors:
                line += " because " + "; ".join(f.description for f in rk.factors)
        rec_lines.append(line)

    human = ("User Query: {query}\n\nCandidate Recommendations (with model predictions):\n{recs}\n\n"
             "Retrieved Historical Cases:\n{context}")
    variables = {"query": query, "recs": "\n".join(rec_lines) or "(none)", "context": context_str or "(none)"}
    if yield_gap:
        human = ("User Query: {query}\n\nThe user is asking WHY yields differ. Do not recommend other crops. "
                 "Explain the yield gap using only this analysis and the cases below.\n\n"
                 "Yield gap analysis:\n{gap}\n\nRetrieved Historical Cases:\n{context}")
        variables = {"query": query, "gap": yield_gap_context(yield_gap), "context": context_str or "(none)"}
    if critique:
        human += ("\n\nA reviewer rejected your previous draft with this critique. Address it while "
                  "using only the data above:\n{critique}")
        variables["critique"] = critique

    prompt = ChatPromptTemplate.from_messages([
        ("system", RECOMMENDER_SYSTEM_PROMPT),
        ("human", human)
    ]).format_messages(**variables)

    template_text = (build_yield_gap_text(yield_gap) if yield_gap else build_deterministic_recommendation_text(
        recommendations, ranked_cases, yield_predictions, risk_assessments
    ))

    result_text, llm_used, fallback_triggered, degraded_mode = invoke_with_fallback(
        prompt=prompt,
        structured_schema=None,
        preferred_provider=preferred_provider,
        preferred_model=preferred_model,
        degraded_fallback_fn=lambda: template_text
    )

    final_text = format_verified_response(
        explanation=str(result_text),
        valid_case_ids=valid_case_ids,
        degraded_mode=degraded_mode,
        allowed_numbers=_allowed_numbers(query, ranked_cases, recommendations, yield_predictions, risk_assessments)
        | (collect_numbers([yield_gap_context(yield_gap)]) if yield_gap else set()),
        fallback_text=template_text,
    )
    return recommendations, final_text, llm_used, fallback_triggered, degraded_mode
