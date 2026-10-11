"""
Evaluation harness: runs the golden set through the full LangGraph pipeline, computes
retrieval hit-rate@k, and scores answers with DeepEval metrics using a local or OpenAI
judge model.

Features:
  - Resumable: each sample result is checkpointed to disk; an interrupted run can be
    resumed without re-evaluating already-completed samples.
  - Configurable metrics: EVAL_METRICS in settings controls which DeepEval metrics run
    (default: 3 core metrics instead of all 5, saving ~12 LLM calls per query).
  - Cached health checks: Ollama/OpenAI availability is cached for 30s so repeated
    invoke_with_fallback calls inside a single graph run don't re-ping the server.
"""
import json
import uuid
import time
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.app.config import settings
from backend.app.agents.graph import get_advisory_graph
from backend.app.guardrails.input_validation import find_crop_mention
from backend.app.llm.factory import check_ollama_status, check_openai_status
from backend.app.schemas.eval import EvalSampleResult, EvalStatusResponse

logger = logging.getLogger(__name__)

RETRIEVAL_K = 5
METRIC_THRESHOLD = 0.5
EVAL_CONCURRENCY = 3

_EVAL_JOBS: Dict[str, Dict[str, Any]] = {}

GEVAL_CRITERIA = (
    "Judge the crop recommendation answer for a farming question. A high-quality answer "
    "(1) recommends crops suited to the stated soil, weather, season and region, "
    "(2) supports each recommendation with the retrieved historical cases and their yields, "
    "(3) mentions relevant risks, and (4) gives practical, safe advice without inventing data."
)

# All supported metric names in canonical order
ALL_METRIC_NAMES = [
    "answer_relevancy", "faithfulness", "contextual_relevancy",
    "contextual_precision", "crop_recommendation_quality",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configured_metric_names() -> List[str]:
    """Parse EVAL_METRICS from settings into a list of metric names."""
    raw = settings.EVAL_METRICS.strip().lower()
    if raw == "all":
        return list(ALL_METRIC_NAMES)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    valid = [n for n in names if n in ALL_METRIC_NAMES]
    if not valid and names:
        logger.warning(f"No valid metric names in EVAL_METRICS='{settings.EVAL_METRICS}'; running all metrics.")
        return list(ALL_METRIC_NAMES)
    return valid


# ---------------------------------------------------------------------------
# Checkpoint helpers for resume support
# ---------------------------------------------------------------------------
def _checkpoint_dir() -> Path:
    return Path(settings.EVAL_RESULTS_DIR) / ".checkpoints"


def _checkpoint_fingerprint(golden_set: List[Dict[str, Any]], limit: Optional[int],
                             provider: Optional[str], model: Optional[str]) -> str:
    """Stable hash of the eval configuration so we can match a checkpoint to a run."""
    ids = [s.get("id", s["query"]) for s in golden_set]
    key = json.dumps({"ids": ids, "limit": limit, "provider": provider, "model": model}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _checkpoint_path(fingerprint: str) -> Path:
    return _checkpoint_dir() / f"eval_{fingerprint}.json"


def _save_checkpoint(fingerprint: str, completed: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Write checkpoint atomically (write to .tmp then rename)."""
    cp_dir = _checkpoint_dir()
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(fingerprint)
    tmp = path.with_suffix(".tmp")
    data = {"meta": meta, "completed": completed}
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def _load_checkpoint(fingerprint: str) -> Optional[Dict[str, Any]]:
    path = _checkpoint_path(fingerprint)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"Corrupt checkpoint {path}: {e}; starting fresh.")
        return None


def _delete_checkpoint(fingerprint: str) -> None:
    path = _checkpoint_path(fingerprint)
    path.unlink(missing_ok=True)


def load_golden_set(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(__file__).resolve().parent / "golden_set.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def call_with_timeout(fn: Callable[[], Any], timeout_s: float, retries: int) -> Any:
    """Runs `fn` with a timeout, retrying on failure; re-raises the last error."""
    last_err: Optional[BaseException] = None
    for attempt in range(retries + 1):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout:
            last_err = TimeoutError(f"timed out after {timeout_s}s")
        except Exception as e:
            last_err = e
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 5))
    raise last_err  # type: ignore[misc]


def resolve_judge(provider: Optional[str], model: Optional[str]) -> Tuple[Any, Optional[str]]:
    """Returns (DeepEval judge model or None, status message)."""
    from backend.app.evaluation.judge import CustomDeepEvalLLM, DEEPEVAL_IMPORT_ERROR
    if CustomDeepEvalLLM is None:
        return None, f"skipped: deepeval not importable ({DEEPEVAL_IMPORT_ERROR})"

    candidates = [provider or settings.JUDGE_PROVIDER or settings.DEFAULT_LLM_PROVIDER]
    candidates.append("openai" if candidates[0] == "ollama" else "ollama")
    status = {"ollama": check_ollama_status(), "openai": check_openai_status()}
    for idx, prov in enumerate(candidates):
        available, reason, _, _ = status.get(prov, (False, "unknown provider", [], []))
        if available:
            judge = CustomDeepEvalLLM(provider=prov, model_name=model if idx == 0 else None)
            return judge, "ran"
    reasons = "; ".join(f"{p}: {status[p][1]}" for p in ("ollama", "openai"))
    return None, f"skipped: no LLM available to act as DeepEval judge ({reasons})"


def build_metrics(judge: Any, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build only the requested DeepEval metric objects."""
    from deepeval.metrics import (
        AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric,
        ContextualPrecisionMetric, GEval,
    )
    from deepeval.test_case import LLMTestCaseParams

    names = set(metric_names or ALL_METRIC_NAMES)
    common = dict(model=judge, threshold=METRIC_THRESHOLD, async_mode=False)
    metrics: Dict[str, Any] = {}

    if "answer_relevancy" in names:
        metrics["answer_relevancy"] = AnswerRelevancyMetric(include_reason=True, **common)
    if "faithfulness" in names:
        metrics["faithfulness"] = FaithfulnessMetric(include_reason=True, **common)
    if "contextual_relevancy" in names:
        metrics["contextual_relevancy"] = ContextualRelevancyMetric(include_reason=True, **common)
    if "contextual_precision" in names:
        metrics["contextual_precision"] = ContextualPrecisionMetric(include_reason=True, **common)
    if "crop_recommendation_quality" in names:
        metrics["crop_recommendation_quality"] = GEval(
            name="CropRecommendationQuality",
            criteria=GEVAL_CRITERIA,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            **common,
        )

    logger.info(f"DeepEval metrics enabled: {list(metrics.keys())}")
    return metrics


def expected_output_for(sample: Dict[str, Any], resolved: List[str]) -> str:
    crops = ", ".join(resolved or sample.get("expected_crops", []))
    criteria = sample.get("criteria")
    text = f"The answer should recommend or discuss crops such as {crops}, grounded in similar historical cases."
    return f"{text} {criteria}" if criteria else text


def evaluate_sample(sample: Dict[str, Any], state: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> EvalSampleResult:
    """Scores one pipeline run: retrieval hit@k, recommendation hit, and DeepEval metrics."""
    resolved = [find_crop_mention(c) or c for c in sample.get("expected_crops", [])]
    expected = {c.lower() for c in resolved}

    cases = state.get("ranked_cases") or []
    recs = state.get("recommendations") or []
    retrieved_crops = [c.metadata.get("crop", "") for c in cases]
    rec_crops = [r.crop for r in recs]
    verdict = state.get("judge_verdict")

    result = EvalSampleResult(
        id=sample.get("id"),
        query=sample["query"],
        scenario=sample.get("scenario", "recommendation"),
        expected_crops=sample.get("expected_crops", []),
        expected_crops_resolved=resolved,
        retrieved_cases=[c.case_id for c in cases][:RETRIEVAL_K],
        retrieved_crops=retrieved_crops[:RETRIEVAL_K],
        recommended_crops=rec_crops,
        hit_at_k=any(c.lower() in expected for c in retrieved_crops[:RETRIEVAL_K]),
        recommendation_hit=any(c.lower() in expected for c in rec_crops),
        judge_passed=verdict.passed if verdict else None,
        degraded_mode=bool(state.get("degraded_mode")),
    )

    if metrics:
        from deepeval.test_case import LLMTestCase
        test_case = LLMTestCase(
            input=sample["query"],
            actual_output=state.get("explanation") or "",
            expected_output=expected_output_for(sample, resolved),
            retrieval_context=[c.narrative for c in cases[:RETRIEVAL_K]] or ["(no cases retrieved)"],
        )
        field_map = {
            "answer_relevancy": "relevancy_score",
            "faithfulness": "faithfulness_score",
            "contextual_relevancy": "contextual_relevancy_score",
            "contextual_precision": "contextual_precision_score",
            "crop_recommendation_quality": "geval_score",
        }
        for name, metric in metrics.items():
            try:
                call_with_timeout(lambda m=metric: m.measure(test_case), settings.EVAL_TIMEOUT_SECONDS,
                                  settings.EVAL_MAX_RETRIES)
                setattr(result, field_map[name], round(float(metric.score), 4))
                if getattr(metric, "reason", None):
                    result.metric_reasons[name] = str(metric.reason)[:500]
            except Exception as e:
                result.metric_reasons[name] = f"metric failed: {e}"
                logger.warning(f"DeepEval metric {name} failed for {sample.get('id')}: {e}")

    hits = [c for c in rec_crops if c.lower() in expected]
    result.reason = (f"Expected {resolved}; recommended {rec_crops[:5]}; "
                     f"{len(hits)} expected crop(s) recommended.")
    return result


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(results: List[EvalSampleResult]) -> Dict[str, Optional[float]]:
    ok = [r for r in results if not r.error]
    n = len(ok) or 1
    judged = [r for r in ok if r.judge_passed is not None]
    return {
        "hit_rate_at_k": round(sum(r.hit_at_k for r in ok) / n, 4) if ok else None,
        "recommendation_hit_rate": round(sum(r.recommendation_hit for r in ok) / n, 4) if ok else None,
        "judge_pass_rate": round(sum(bool(r.judge_passed) for r in judged) / len(judged), 4) if judged else None,
        "answer_relevancy": _mean([r.relevancy_score for r in ok]),
        "faithfulness": _mean([r.faithfulness_score for r in ok]),
        "contextual_relevancy": _mean([r.contextual_relevancy_score for r in ok]),
        "contextual_precision": _mean([r.contextual_precision_score for r in ok]),
        "crop_recommendation_quality_geval": _mean([r.geval_score for r in ok]),
        "degraded_runs": float(sum(r.degraded_mode for r in ok)),
        "failed_samples": float(len(results) - len(ok)),
        "total_evaluated": float(len(results)),
        "k": float(RETRIEVAL_K),
    }


def run_evaluation_sync(run_id: str, limit: Optional[int] = None, provider: Optional[str] = None,
                        model: Optional[str] = None, resume: bool = True) -> Dict[str, Any]:
    """
    Runs the evaluation and persists data/eval/results_{timestamp}.json (+ latest.json).

    When *resume* is True (the default), completed sample results are loaded from a
    checkpoint file so an interrupted run picks up where it left off instead of starting
    over.  The checkpoint is keyed on the golden set IDs + provider + model + limit so a
    different configuration always starts fresh.
    """
    from backend.app.data.ingest import is_synthetic_dataset

    golden_set = load_golden_set(limit=limit)
    fingerprint = _checkpoint_fingerprint(golden_set, limit, provider, model)

    # --- Resume: load previously completed samples ----------------------------
    completed_map: Dict[str, Dict[str, Any]] = {}  # sample id -> result dict
    resumed_count = 0
    if resume:
        cp = _load_checkpoint(fingerprint)
        if cp and isinstance(cp.get("completed"), dict):
            completed_map = cp["completed"]
            resumed_count = len(completed_map)
            logger.info(f"Evaluation {run_id}: resuming — {resumed_count}/{len(golden_set)} samples already done")

    job = _EVAL_JOBS.setdefault(run_id, {})
    job.update({
        "run_id": run_id, "status": "running", "progress": 0.0, "total_samples": len(golden_set),
        "completed_samples": resumed_count, "results": [], "aggregate_metrics": {},
        "created_at": _now(), "completed_at": None, "error": None,
        "is_synthetic_data": is_synthetic_dataset(), "resumed_samples": resumed_count,
    })
    logger.info(f"Evaluation {run_id}: {len(golden_set)} samples ({resumed_count} resumed)")

    try:
        judge, deepeval_status = resolve_judge(provider, model)
        job["deepeval_status"] = deepeval_status
        job["judge_model"] = judge.get_model_name() if judge else None

        metric_names = _configured_metric_names()
        metrics = build_metrics(judge, metric_names) if judge else None

        graph = get_advisory_graph()
        results: List[EvalSampleResult] = []
        done_count = 0

        # Separate resumed samples from samples that need running
        pending_samples: List[Tuple[int, Dict[str, Any]]] = []
        for idx, sample in enumerate(golden_set):
            sample_id = sample.get("id") or f"idx_{idx}"
            if sample_id in completed_map:
                try:
                    res = EvalSampleResult(**completed_map[sample_id])
                except Exception:
                    completed_map.pop(sample_id, None)
                    pending_samples.append((idx, sample))
                else:
                    results.append(res)
                    done_count += 1
                    continue
            pending_samples.append((idx, sample))

        job["results"] = [r.model_dump() for r in results]
        job["completed_samples"] = done_count
        job["progress"] = round(done_count / max(1, len(golden_set)), 4)

        # Run pending samples concurrently
        result_lock = threading.Lock()

        def _run_one(idx_sample: Tuple[int, Dict[str, Any]]) -> EvalSampleResult:
            idx, sample = idx_sample
            sample_id = sample.get("id") or f"idx_{idx}"
            t0 = time.time()
            try:
                state = call_with_timeout(
                    lambda s=sample: graph.invoke({
                        "query": s["query"], "top_k": 8, "mode": "hybrid",
                        "preferred_llm_provider": provider, "preferred_llm_model": model, "trace": [],
                    }),
                    settings.EVAL_TIMEOUT_SECONDS * 3, settings.EVAL_MAX_RETRIES,
                )
                res = evaluate_sample(sample, state, metrics)
            except Exception as e:
                logger.warning(f"Eval sample {sample_id} failed: {e}")
                res = EvalSampleResult(id=sample_id, query=sample["query"],
                                       scenario=sample.get("scenario", ""), expected_crops=sample.get("expected_crops", []),
                                       retrieved_cases=[], hit_at_k=False, error=str(e))
            res.latency_ms = int((time.time() - t0) * 1000)
            return res

        if pending_samples:
            with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as pool:
                for res in pool.map(_run_one, pending_samples):
                    sample_id = res.id or f"idx_{done_count}"
                    results.append(res)
                    done_count += 1

                    with result_lock:
                        completed_map[sample_id] = res.model_dump()
                        _save_checkpoint(fingerprint, completed_map, {
                            "run_id": run_id, "provider": provider, "model": model,
                            "limit": limit, "updated_at": _now(),
                        })
                        job["results"] = [r.model_dump() for r in results]
                        job["completed_samples"] = done_count
                        job["progress"] = round(done_count / max(1, len(golden_set)), 4)

        job["aggregate_metrics"] = aggregate(results)
        job["status"] = "completed"
    except Exception as e:
        logger.exception(f"Evaluation run {run_id} failed: {e}")
        job["status"], job["error"] = "failed", str(e)
    finally:
        job["completed_at"] = _now()

    eval_dir = Path(settings.EVAL_RESULTS_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(job, indent=2))
    if job["status"] == "completed":
        (eval_dir / "latest.json").write_text(json.dumps(job, indent=2))
        # Clean up checkpoint on successful completion
        _delete_checkpoint(fingerprint)
    logger.info(f"Evaluation {run_id} {job['status']}; saved to {out_file}")
    return job


def start_eval_task(limit: Optional[int] = None, provider: Optional[str] = None,
                    model: Optional[str] = None, resume: bool = True) -> Tuple[str, int]:
    """Starts an evaluation in a background thread; returns (run_id, number of samples)."""
    run_id = f"eval_{uuid.uuid4().hex[:8]}"
    total = len(load_golden_set(limit=limit))
    _EVAL_JOBS[run_id] = {"run_id": run_id, "status": "pending", "progress": 0.0, "total_samples": total,
                          "completed_samples": 0, "created_at": _now()}
    threading.Thread(target=run_evaluation_sync, args=(run_id, limit, provider, model, resume), daemon=True).start()
    return run_id, total


def get_eval_status(run_id: str) -> Optional[EvalStatusResponse]:
    data = _EVAL_JOBS.get(run_id)
    return EvalStatusResponse(**data) if data else None


def get_latest_eval_results() -> Optional[EvalStatusResponse]:
    for data in reversed(list(_EVAL_JOBS.values())):
        if data.get("status") == "completed":
            return EvalStatusResponse(**data)
    latest_file = Path(settings.EVAL_RESULTS_DIR) / "latest.json"
    if latest_file.exists():
        try:
            return EvalStatusResponse(**json.loads(latest_file.read_text()))
        except Exception as e:
            logger.warning(f"Could not read {latest_file}: {e}")
    return None
