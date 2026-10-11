import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query as FastQuery, Request
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.schemas import (
    PipelineQueryRequest,
    PipelineQueryResponse,
    SearchRequest,
    SearchResponse,
    ClassifyRequest,
    ClassifyResponse,
    RecommendationRequest,
    RecommendationResponse,
    YieldPredictionRequest,
    YieldPredictionResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    ImpactAnalysisRequest,
    ImpactAnalysisResponse,
    CompareRequest,
    CompareResponse,
    FeedbackCreate,
    FeedbackResponse,
    FeedbackStats,
    EvalRunRequest,
    EvalRunResponse,
    EvalStatusResponse,
    ProviderStatus,
    TraceStep
)
from backend.app.schemas.classification import QueryClassification
from backend.app.llm.factory import list_providers, check_ollama_status, check_openai_status
from backend.app.guardrails.input_validation import (
    check_rate_limit,
    sanitize_and_validate_query_text,
    validate_numeric_bounds,
    KNOWN_CROPS,
    KNOWN_SEASONS,
    KNOWN_STATES
)
from backend.app.guardrails.output_validation import DISCLAIMER_TEXT
from backend.app.retrieval.hybrid import search_with_mode
from backend.app.agents.classifier import classify_query
from backend.app.agents.nodes import classification_warnings
from backend.app.agents.recommender import generate_crop_recommendation
from backend.app.agents.graph import get_advisory_graph, get_graph_mermaid_diagram
from backend.app.ml import yield_model, risk_model
from backend.app.ml.yield_model import get_yield_predictor, CropYieldPredictor
from backend.app.ml.risk_model import get_risk_predictor, CropRiskPredictor
from backend.app.ml.impact import perform_impact_analysis, compare_farm_profiles, clear_importance_cache
from backend.app.feedback.store import save_feedback, get_feedback_statistics
from backend.app.feedback.reranker import get_feedback_boost_map
from backend.app.evaluation.deepeval_runner import start_eval_task, get_eval_status, get_latest_eval_results
from backend.app.retrieval.vectorstore import (
    check_collection_exists,
    get_collection_name,
    resolve_embedding_target,
    start_index_job,
    get_index_job,
    get_running_index_job,
)
from backend.app.retrieval.bm25 import reset_bm25_cache, get_bm25_retriever
from backend.app.retrieval.ranking import reset_year_range_cache
from backend.app.data.ingest import (
    get_active_dataframe,
    df_to_documents,
    get_dataset_info,
    is_synthetic_dataset,
    clean_and_prepare_dataset,
)
from backend.app.data.features import load_thresholds

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/v1")


def _enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not check_rate_limit(client_ip, settings.RATE_LIMIT_PER_MINUTE):
        raise HTTPException(status_code=429, detail={"code": "RATE_LIMITED", "message": "Too many requests. Please slow down."})


def _cfg(cfg: Optional[Dict[str, Optional[str]]], key: str) -> Optional[str]:
    return cfg.get(key) if cfg else None


# 1. Health Check
@api_router.get("/health")
def get_health():
    """Liveness plus dataset provenance, index, model and provider status."""
    df = get_active_dataframe()
    yield_pred = get_yield_predictor()
    risk_pred = get_risk_predictor()
    prov, mdl = resolve_embedding_target(None, None)
    col_exists, col_count = check_collection_exists(prov, mdl)
    info = get_dataset_info()

    return {
        "status": "ok",
        "dataset_rows": len(df),
        "is_synthetic_data": bool(info.get("is_synthetic", True)),
        "dataset_source": info.get("raw_path"),
        "chroma_collection": {
            "name": get_collection_name(prov, mdl),
            "indexed": col_exists,
            "document_count": col_count,
            "expected_documents": len(df),
        },
        "models": {
            "yield_model": yield_pred.metrics,
            "risk_model": risk_pred.metrics,
        },
        "providers": list_providers()
    }


# 2. Providers List
@api_router.get("/providers", response_model=List[ProviderStatus])
def get_providers():
    return list_providers()


# 3. Main Full Pipeline (LangGraph)
@api_router.post("/query", response_model=PipelineQueryResponse)
def execute_query(req: PipelineQueryRequest, request: Request):
    """Runs the full multi-agent LangGraph pipeline."""
    _enforce_rate_limit(request)

    result = get_advisory_graph().invoke({
        "query": req.query,
        "filters": req.filters,
        "top_k": req.top_k,
        "mode": req.mode,
        "preferred_llm_provider": _cfg(req.llm, "provider"),
        "preferred_llm_model": _cfg(req.llm, "model"),
        "preferred_embed_provider": _cfg(req.embedding, "provider"),
        "preferred_embed_model": _cfg(req.embedding, "model"),
        "trace": []
    })

    out_of_scope = bool(result.get("out_of_scope"))
    if not result.get("is_valid", True) and not out_of_scope:
        raise HTTPException(
            status_code=result.get("rejection_status") or 422,
            detail={"code": result.get("rejection_code") or "VALIDATION_FAILED",
                    "message": result.get("rejection_reason", "Validation error")}
        )

    classification = result.get("classification") or QueryClassification(intent="out_of_scope", confidence=1.0)
    return PipelineQueryResponse(
        classification=classification,
        similar_cases=result.get("ranked_cases") or [],
        recommendations=result.get("recommendations") or [],
        impact_analysis=result.get("soil_weather_analysis"),
        yield_gap=result.get("yield_gap"),
        explanation=result.get("explanation", ""),
        judge=result.get("judge_verdict"),
        agent_trace=[TraceStep(node=t["node"], ms=t["ms"], summary=t["summary"]) for t in result.get("trace", [])],
        llm_used=result.get("llm_used"),
        fallback_triggered=result.get("fallback_triggered", False),
        degraded_mode=result.get("degraded_mode", False),
        warnings=result.get("warnings") or [],
        filters_applied=result.get("filters_applied") or {},
        search_mode_used=result.get("effective_search_mode"),
        out_of_scope=out_of_scope,
        is_synthetic_data=is_synthetic_dataset(),
        disclaimer=result.get("disclaimer", DISCLAIMER_TEXT),
    )


# 4. Search Endpoint
@api_router.post("/search", response_model=SearchResponse)
def search_cases(req: SearchRequest):
    """Keyword / semantic / hybrid search with metadata filters and prioritization."""
    sanitize_and_validate_query_text(req.query)
    prov, mdl = resolve_embedding_target(_cfg(req.embedding, "provider"), _cfg(req.embedding, "model"))
    results, mode_used, note = search_with_mode(
        query=req.query,
        top_k=req.top_k,
        mode=req.mode,
        filters=req.filters,
        user_conditions=req.user_conditions,
        feedback_boosts=get_feedback_boost_map(),
        provider=prov,
        model=mdl
    )
    return SearchResponse(
        results=results,
        total_found=len(results),
        mode_used=mode_used,
        mode_requested=req.mode,
        collection_used=get_collection_name(prov, mdl) if mode_used != "keyword" else None,
        note=note,
    )


# 5. Classify Endpoint
@api_router.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest):
    """Classification agent only (LLM structured output, regex fallback)."""
    query = sanitize_and_validate_query_text(req.query)
    clf, llm_used, fb_triggered, degraded = classify_query(
        query=query,
        preferred_provider=_cfg(req.llm, "provider"),
        preferred_model=_cfg(req.llm, "model")
    )
    return ClassifyResponse(
        classification=clf,
        warnings=classification_warnings(clf, query),
        llm_used=llm_used,
        fallback_triggered=fb_triggered,
        degraded_mode=degraded
    )


# 6. Basic Crop Recommendation
@api_router.post("/recommend", response_model=RecommendationResponse)
def recommend_endpoint(req: RecommendationRequest):
    """Basic recommendation: hybrid retrieval + per-crop aggregation + grounded explanation."""
    query = sanitize_and_validate_query_text(req.query)
    ranked, _, _ = search_with_mode(
        query=query,
        top_k=req.top_k * 2,
        mode="hybrid",
        filters=req.filters,
        feedback_boosts=get_feedback_boost_map(),
        provider=_cfg(req.embedding, "provider"),
        model=_cfg(req.embedding, "model")
    )
    recs, explanation, llm_used, fb_trig, degraded = generate_crop_recommendation(
        query=query,
        ranked_cases=ranked,
        preferred_provider=_cfg(req.llm, "provider"),
        preferred_model=_cfg(req.llm, "model")
    )
    return RecommendationResponse(
        recommendations=recs,
        explanation=explanation,
        supporting_cases_count=len(ranked),
        llm_used=llm_used,
        fallback_triggered=fb_trig,
        degraded_mode=degraded
    )


# 7. Predict Yield
@api_router.post("/predict/yield", response_model=YieldPredictionResponse)
def predict_yield_endpoint(req: YieldPredictionRequest):
    validate_numeric_bounds(
        n=req.n, p=req.p, k=req.k, ph=req.ph,
        temperature=req.temperature, rainfall=req.rainfall,
        humidity=req.humidity, crop_year=req.crop_year
    )
    predictor = get_yield_predictor()
    return YieldPredictionResponse(
        crop=req.crop,
        prediction=predictor.predict_one(req.model_dump()),
        model_type=predictor.model_type,
        trained_on=predictor.trained_on,
        metrics=predictor.metrics
    )


# 8. Predict Risk
@api_router.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk_endpoint(req: RiskPredictionRequest):
    validate_numeric_bounds(
        n=req.n, p=req.p, k=req.k, ph=req.ph,
        temperature=req.temperature, rainfall=req.rainfall,
        humidity=req.humidity, crop_year=req.crop_year
    )
    return RiskPredictionResponse(crop=req.crop, risk=get_risk_predictor().assess_risk(req.model_dump()))


# 9. Analyze Impact
@api_router.post("/analyze/impact", response_model=ImpactAnalysisResponse)
def impact_endpoint(req: ImpactAnalysisRequest):
    validate_numeric_bounds(
        n=req.n, p=req.p, k=req.k, ph=req.ph,
        temperature=req.temperature, rainfall=req.rainfall,
        humidity=req.humidity, crop_year=req.crop_year
    )
    return perform_impact_analysis(req)


# 10. Compare Profiles
@api_router.post("/compare", response_model=CompareResponse)
def compare_endpoint(req: CompareRequest):
    crops = {p.crop for p in req.profiles}
    if len(crops) > 1:
        raise HTTPException(status_code=422, detail={"code": "MIXED_CROPS",
                                                     "message": "All farm profiles must be for the same crop."})
    for p in req.profiles:
        validate_numeric_bounds(
            n=p.n, p=p.p, k=p.k, ph=p.ph,
            temperature=p.temperature, rainfall=p.rainfall,
            humidity=p.humidity
        )
    return compare_farm_profiles(req)


# 11. Metadata Options
@api_router.get("/metadata/options")
def get_metadata_options():
    get_active_dataframe()  # ensures the known-value lists reflect the active dataset
    return {
        "crops": KNOWN_CROPS,
        "seasons": KNOWN_SEASONS,
        "states": KNOWN_STATES,
        "ph_classes": ["acidic", "neutral", "alkaline"],
        "nutrient_levels": ["low", "medium", "high"],
        "temp_classes": ["cool", "moderate", "hot"],
        "rainfall_classes": ["low", "moderate", "high"],
        "humidity_classes": ["low", "moderate", "high"],
        "thresholds": load_thresholds(settings.THRESHOLDS_PATH),
        "is_synthetic_data": is_synthetic_dataset(),
    }


# 12. Feedback Endpoints
@api_router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(fb: FeedbackCreate):
    fid = save_feedback(fb)
    return FeedbackResponse(id=fid, status="success", message="Feedback recorded; it will adjust case/crop re-ranking.")


@api_router.get("/feedback/stats", response_model=FeedbackStats)
def get_feedback_stats():
    return get_feedback_statistics()


# 13. DeepEval Evaluation
@api_router.post("/eval/run", response_model=EvalRunResponse)
def trigger_eval(req: EvalRunRequest):
    """Starts the evaluation suite as a background job; poll /eval/status/{run_id}."""
    run_id, total = start_eval_task(limit=req.limit, provider=_cfg(req.llm, "provider"), model=_cfg(req.llm, "model"),
                                    resume=req.resume)
    return EvalRunResponse(
        run_id=run_id,
        status="running",
        total_samples=total,
        message="Evaluation job started in the background."
    )


@api_router.get("/eval/status/{run_id}", response_model=EvalStatusResponse)
def eval_status(run_id: str):
    res = get_eval_status(run_id)
    if not res:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Eval run '{run_id}' not found"})
    return res


@api_router.get("/eval/latest", response_model=EvalStatusResponse)
def eval_latest():
    res = get_latest_eval_results()
    if not res:
        raise HTTPException(status_code=404, detail={"code": "NO_EVAL_RESULTS",
                                                     "message": "No evaluation has been run yet. Start one with POST /api/v1/eval/run."})
    return res


# 14. Admin: index management
class ReindexRequest(BaseModel):
    embedding_provider: Optional[str] = Field(None, description="ollama | openai (default from config)")
    embedding_model: Optional[str] = Field(None, description="Embedding model (default for the provider)")
    force: bool = Field(True, description="Drop and rebuild the collection")


def _embedding_provider_status(provider: str):
    available, reason, _, _ = check_openai_status() if provider == "openai" else check_ollama_status()
    return available, reason


@api_router.get("/admin/index/status")
def admin_index_status(
    embedding_provider: Optional[str] = FastQuery(None),
    embedding_model: Optional[str] = FastQuery(None)
):
    """Whether the provider/model collection exists, plus any running build job."""
    prov, mdl = resolve_embedding_target(embedding_provider, embedding_model)
    exists, count = check_collection_exists(prov, mdl)
    available, reason = _embedding_provider_status(prov)
    expected = len(get_active_dataframe())
    return {
        "collection": get_collection_name(prov, mdl),
        "provider": prov,
        "model": mdl,
        "exists": exists,
        "document_count": count,
        "expected_documents": expected,
        "up_to_date": exists and count == expected,
        "embeddings_available": available,
        "reason": reason,
        "job": get_running_index_job(prov, mdl),
    }


@api_router.post("/admin/reindex")
def admin_reindex(
    body: Optional[ReindexRequest] = Body(None),
    embedding_provider: Optional[str] = FastQuery(None),
    embedding_model: Optional[str] = FastQuery(None),
    force: Optional[bool] = FastQuery(None),
):
    """
    Builds the `cases__{provider}__{model}` collection in the background (batches of 200).
    Accepts a JSON body or query parameters. Poll /admin/reindex/status/{job_id}.
    """
    body = body or ReindexRequest()
    prov, mdl = resolve_embedding_target(body.embedding_provider or embedding_provider,
                                         body.embedding_model or embedding_model)
    available, reason = _embedding_provider_status(prov)
    if not available:
        raise HTTPException(status_code=409, detail={"code": "EMBEDDINGS_UNAVAILABLE",
                                                     "message": f"Cannot build '{get_collection_name(prov, mdl)}': {reason}"})
    docs = df_to_documents(get_active_dataframe())
    job = start_index_job(docs, prov, mdl, force_reindex=body.force if force is None else force)
    return {"status": job["status"], **job}


@api_router.get("/admin/reindex/status/{job_id}")
def admin_reindex_status(job_id: str):
    job = get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": f"Index job '{job_id}' not found"})
    return job


class RetrainRequest(BaseModel):
    reprocess_data: bool = Field(False, description="Re-run dataset resolution and cleaning before training")


@api_router.post("/admin/retrain")
def admin_retrain(body: Optional[RetrainRequest] = Body(None)):
    """Retrains the yield and risk models (optionally after re-processing the raw data)."""
    body = body or RetrainRequest()
    if body.reprocess_data:
        from scripts.download_data import ensure_dataset
        clean_and_prepare_dataset(ensure_dataset())
        reset_bm25_cache()
        reset_year_range_cache()
        get_bm25_retriever()

    df = get_active_dataframe()
    y_metrics = CropYieldPredictor().train(df, settings.MODELS_DIR)
    r_metrics = CropRiskPredictor().train(df, settings.MODELS_DIR)
    yield_model.reset_predictor_cache()
    risk_model.reset_predictor_cache()
    clear_importance_cache()
    return {"status": "success", "dataset_rows": len(df), "yield_metrics": y_metrics, "risk_metrics": r_metrics}


# 15. LangGraph Mermaid Export
@api_router.get("/graph")
def get_graph_mermaid():
    return {"mermaid": get_graph_mermaid_diagram()}
