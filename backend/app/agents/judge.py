import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate

from backend.app.config import settings
from backend.app.schemas.query import JudgeVerdict
from backend.app.schemas.recommendation import CropRecommendation
from backend.app.schemas.search import CaseDocument
from backend.app.llm.fallback import invoke_with_fallback
from backend.app.agents.prompts import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def rule_based_judge(
    recommendations: List[CropRecommendation],
    ranked_cases: List[CaseDocument],
    explanation: str = "",
    extra_valid_ids: Optional[List[str]] = None
) -> JudgeVerdict:
    """
    Deterministic validator used when no LLM judge is available. Checks:
    - grounded: every supporting / cited case ID is in the retrieved set;
    - consistent_with_data: each crop's historical median yield is compatible with the
      model's prediction interval (within half of the low bound to double the high bound);
    - actionable: recommendations carry yield predictions and risk assessments;
    - safe: the advisory states no input dosages (the template never does; LLM text is
      covered by the output guardrails).
    """
    if not recommendations:
        return JudgeVerdict(grounded=4, consistent_with_data=4, actionable=3, safe=5, passed=True,
                            critique="No candidate crops matched the criteria; returned an empty result.")

    valid_cids = {c.case_id for c in ranked_cases} | set(extra_valid_ids or [])
    cited = {cid for r in recommendations for cid in r.supporting_cases} | set(re.findall(r"CASE-\d{6}", explanation))
    unknown = sorted(cited - valid_cids)
    grounded = 5 if not unknown else 2

    inconsistent = []
    for r in recommendations:
        p = r.predicted_yield
        if p and not (p.low * 0.5 <= r.median_yield <= max(p.high * 2.0, 0.01)):
            inconsistent.append(f"{r.crop} (historical median {r.median_yield} vs predicted {p.low}-{p.high})")
    consistent = 5 if not inconsistent else (4 if len(inconsistent) == 1 else 3)

    complete = sum(1 for r in recommendations if r.predicted_yield and r.risk)
    actionable = 5 if complete == len(recommendations) else (4 if complete else 3)

    notes = []
    notes.append("all cited case IDs are in the retrieved set" if not unknown else f"unknown case IDs cited: {unknown}")
    notes.append("historical yields agree with model intervals" if not inconsistent
                 else "historical and predicted yields diverge for " + "; ".join(inconsistent))
    notes.append(f"{complete}/{len(recommendations)} recommendations include yield and risk estimates")
    scores = [grounded, consistent, actionable, 5]
    return JudgeVerdict(
        grounded=grounded, consistent_with_data=consistent, actionable=actionable, safe=5,
        passed=all(x >= 3 for x in scores) and sum(scores) / 4 >= 3.5,
        critique="Rule-based validation: " + "; ".join(notes) + "."
    )

def evaluate_recommendation(
    query: str,
    recommendations: List[CropRecommendation],
    explanation: str,
    ranked_cases: List[CaseDocument],
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    extra_valid_ids: Optional[List[str]] = None
) -> Tuple[JudgeVerdict, Optional[Dict[str, str]], bool, bool]:
    """
    Evaluates the draft advisory recommendation using LLM-as-judge or rule-based validation.
    """
    # A configured JUDGE_PROVIDER (optionally a different/larger model) takes precedence;
    # otherwise the judge uses the chat provider/model selected for this request.
    if settings.JUDGE_PROVIDER:
        judge_prov, judge_mdl = settings.JUDGE_PROVIDER, (settings.JUDGE_MODEL or None)
    else:
        judge_prov, judge_mdl = preferred_provider, preferred_model

    cases_summary = "\n".join([f"- [{c.case_id}] {c.narrative}" for c in ranked_cases[:6]])
    rec_lines = []
    for r in recommendations:
        line = f"- {r.crop}: score={r.score}, median_yield={r.median_yield}, supporting={r.supporting_cases}"
        if r.predicted_yield:
            line += f", predicted_yield={r.predicted_yield.value} ({r.predicted_yield.low}-{r.predicted_yield.high})"
        if r.risk:
            line += f", risk={r.risk.level} (p={r.risk.probability})"
        rec_lines.append(line)
    recs_summary = "\n".join(rec_lines)

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", (
            "User Question: {query}\n\n"
            "Retrieved Cases:\n{cases}\n\n"
            "Structured Recommendations:\n{recs}\n\n"
            "Draft Advisory Explanation:\n{explanation}\n\n"
            "Evaluate and return structured JudgeVerdict."
        ))
    ]).format_messages(
        query=query,
        cases=cases_summary,
        recs=recs_summary,
        explanation=explanation
    )

    def degraded_fallback():
        return rule_based_judge(recommendations, ranked_cases, explanation, extra_valid_ids)

    result, llm_used, fallback_triggered, degraded_mode = invoke_with_fallback(
        prompt=prompt,
        structured_schema=JudgeVerdict,
        preferred_provider=judge_prov,
        preferred_model=judge_mdl,
        degraded_fallback_fn=degraded_fallback
    )

    if not isinstance(result, JudgeVerdict):
        result = rule_based_judge(recommendations, ranked_cases, explanation, extra_valid_ids)
        degraded_mode = True

    # Enforce pass criteria
    scores = [result.grounded, result.consistent_with_data, result.actionable, result.safe]
    passed = all(s >= 3 for s in scores) and (sum(scores) / len(scores) >= 3.5)
    result.passed = passed

    return result, llm_used, fallback_triggered, degraded_mode
