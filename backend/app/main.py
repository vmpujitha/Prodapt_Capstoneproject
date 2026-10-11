import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.config import settings
from backend.app.api.v1.router import api_router
from backend.app.data.ingest import get_active_dataframe, df_to_documents, is_synthetic_dataset
from backend.app.llm.factory import get_embeddings, EmbeddingProviderUnavailable
from backend.app.retrieval.vectorstore import (
    check_collection_exists,
    get_collection_name,
    resolve_embedding_target,
    start_index_job,
)
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.ml.yield_model import get_yield_predictor
from backend.app.ml.risk_model import get_risk_predictor
from backend.app.feedback.store import init_db

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger("agri_assistant")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Agricultural Crop Advisory Assistant Backend...")
    init_db()

    df = get_active_dataframe()
    logger.info(f"Active dataset loaded: {len(df)} records.")
    if is_synthetic_dataset():
        logger.warning("!" * 70)
        logger.warning("SYNTHETIC DATA IN USE - results are for demonstration only, not real agronomy.")
        logger.warning("!" * 70)

    docs = df_to_documents(df)
    get_bm25_retriever(docs)

    prov, mdl = resolve_embedding_target(None, None)
    exists, count = check_collection_exists(prov, mdl)
    if count != len(docs):
        try:
            get_embeddings(prov, mdl)
            logger.info(f"Collection {get_collection_name(prov, mdl)} has {count}/{len(docs)} documents; "
                        f"building it in the background (keyword search is used until it is ready).")
            start_index_job(docs, prov, mdl, force_reindex=False)  # resume, never wipe a partial index
        except EmbeddingProviderUnavailable as e:
            logger.warning(f"{e}. Semantic/hybrid search will fall back to BM25 keyword search "
                           f"until the collection is built (POST /api/v1/admin/reindex).")

    get_yield_predictor()
    get_risk_predictor()

    logger.info("Backend initialization complete. Server ready.")
    yield
    logger.info("Shutting down backend services...")

app = FastAPI(
    title="AI-Powered Agriculture Crop Advisory Assistant API",
    version="1.0.0",
    description="Multi-agent decision support system for crop recommendation, yield prediction, risk assessment, and condition impact analysis.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID & Structured Logging Middleware
@app.middleware("http")
async def add_request_logging_and_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.time()
    try:
        response = await call_next(request)
        process_time_ms = int((time.time() - t0) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)
        logger.info(f'{{"req_id": "{request_id}", "method": "{request.method}", "path": "{request.url.path}", "status": {response.status_code}, "ms": {process_time_ms}}}')
        return response
    except Exception as exc:
        process_time_ms = int((time.time() - t0) * 1000)
        logger.error(f'{{"req_id": "{request_id}", "method": "{request.method}", "path": "{request.url.path}", "error": "{str(exc)}", "ms": {process_time_ms}}}')
        raise exc

# Consistent Error Handling Format
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code", "HTTP_ERROR")
        msg = exc.detail.get("message", "An HTTP error occurred")
        details = exc.detail.get("details", None)
    else:
        code = "HTTP_ERROR"
        msg = str(exc.detail)
        details = None

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": msg, "details": details}}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    field = errors[0]["loc"][-1] if errors and "loc" in errors[0] and errors[0]["loc"] else "field"
    msg = errors[0]["msg"] if errors else "Invalid request payload"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"Validation error on {field}: {msg}",
                "details": errors
            }
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred while processing your request.",
                "details": str(exc)
            }
        }
    )

# Include API v1 Router
app.include_router(api_router)

@app.get("/")
def root():
    return {
        "service": "AI-Powered Agriculture Crop Advisory Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
