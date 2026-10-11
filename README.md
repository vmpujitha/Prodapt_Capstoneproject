# AI-Powered Agriculture Crop Advisory Assistant

Ask farming questions in plain language ("Which crops do well in Punjab during Kharif with N around 90, pH 6.5 and 1200 mm rain?") and get:

- the most similar **historical farming cases** (hybrid BM25 + vector search, metadata filters, explainable ranking),
- **crop recommendations** aggregated from those cases,
- **yield predictions with intervals** and **low-yield risk** per crop (gradient boosting; XGBoost when its native library loads),
- **soil / weather / input impact analysis** (permutation importance + what-if sensitivity) and a **farm comparison** tool,
- a **grounded explanation** that passes input and output guardrails and an **LLM-as-judge** check.

Everything runs locally with **Ollama**; **OpenAI is optional**. With no LLM at all the service still answers in a clearly flagged **degraded mode** (deterministic classifier, template explanation, rule-based judge).

![Architecture](docs/architecture.png)

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| LLM orchestration | LangChain (LCEL, structured output) + LangGraph `StateGraph` |
| LLM providers | Ollama (default) → OpenAI → degraded mode |
| Retrieval | ChromaDB (one collection per embedding provider/model) + BM25 (`rank_bm25`), Reciprocal Rank Fusion |
| ML | scikit-learn GradientBoosting (XGBoost if available), isotonic calibration |
| Evaluation | DeepEval (custom judge wrapper) + retrieval hit-rate |
| Feedback | SQLite via SQLAlchemy |
| Frontend | React + Vite + TypeScript + Tailwind + Recharts |

Detailed docs: [Architecture](docs/ARCHITECTURE.md) · [Design decisions](docs/DESIGN.md) · [Assumptions](docs/ASSUMPTIONS.md) · [Data profile](docs/DATA_PROFILE.md) · [Demo script](docs/DEMO_SCRIPT.md)

---

## Quick start (3 commands)

```bash
make install   # .venv + pip install + npm ci
make setup     # .env, dataset (real Kaggle files or synthetic), models, indexes
make dev       # backend on :8000, frontend on :5173
```

Open http://localhost:5173 (API docs: http://localhost:8000/docs).

## Prerequisites

- Python 3.11+ (verified with 3.14 locally and 3.12 in Docker) and Node 18+ (verified with Node 26 locally and 22 in Docker)
- [Ollama](https://ollama.com) for local LLMs (optional but recommended):

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

- Optional: an `OPENAI_API_KEY` in `.env` to enable OpenAI as a selectable provider and as a fallback.

## Step-by-step setup (local)

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cp .env.example .env                       # edit if needed; never commit .env

.venv/bin/python scripts/download_data.py  # resolve + merge + clean dataset, writes docs/DATA_PROFILE.md
.venv/bin/python scripts/train_models.py   # yield + risk models -> models/
.venv/bin/python scripts/build_index.py    # BM25 + Chroma collection (needs the embedding provider running)

.venv/bin/python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm ci && npm run dev       # http://localhost:5173
```

Models and the BM25 index are also built automatically on first start if they are missing. If the configured embedding provider is reachable and its collection is missing or incomplete, the backend builds it **in the background** at startup. Until then, semantic and hybrid search fall back to keyword search, and the response says so.

### Data setup options

`scripts/download_data.py` resolves the dataset in this order:

1. **Merged CSV** already at `data/raw/crop_yield.csv`.
2. **Kaggle files**: the three files of *Crop Yield Data with Soil and Weather* (anshumish), namely `crop_yield.csv`, `state_soil_data.csv` and `state_weather_data_*.csv`, found in `data_set/` (`KAGGLE_SOURCE_DIR`), `data/raw/kaggle/` or `data/raw/`. They are joined (soil by state, weather by state + year), and fertilizer/pesticide totals are converted to kg/ha.
3. **Kaggle API** download when `KAGGLE_USERNAME` / `KAGGLE_KEY` are set.
4. **Synthetic fallback**: `data/raw/SYNTHETIC_crop_yield.csv` (6,000 rows), with a loud startup warning, a `SYNTHETIC DATA` banner in the UI, and `is_synthetic_data: true` in responses and model metadata.

This repository currently runs on the **real Kaggle data**: 19,689 raw rows, 19,595 after cleaning, covering 55 crops, 30 states and 1997–2020.

### Sharing the project without the data

When you share the project, leave out `data_set/` and the generated folders (`data/raw`, `data/processed`, `data/chroma`, `models/*.joblib`; all are git-ignored). The recipient places the three Kaggle files in `data_set/`, with these exact names:

```
data_set/
├── crop_yield.csv
├── state_soil_data.csv
└── state_weather_data_1997_2020.csv   (any state_weather_data*.csv name works)
```

Then they run `make install && make setup && make dev`. If they start the app before adding `data_set/`, it runs on clearly labelled synthetic data. On the next start after the files are added, it re-processes the data and retrains the models on the real data automatically.

## Docker

```bash
cp .env.example .env
docker compose up --build        # backend :8000, frontend :3000
```

- The backend container reaches Ollama on the host via `host.docker.internal:11434`. To run Ollama in a container instead, use `docker compose --profile ollama up` with `OLLAMA_BASE_URL_DOCKER=http://ollama:11434`, then `docker compose exec ollama ollama pull llama3.1:8b` (and `nomic-embed-text`).
- `data/`, `models/` and `data_set/` are mounted as volumes, so indexes and models persist.
- Behind a TLS-intercepting proxy, put your root CA (`*.crt`) in `docker/certs/`; see [docker/certs/README.md](docker/certs/README.md).

## Switching providers

- **UI**: the header has an LLM provider + model dropdown, populated from `GET /api/v1/providers`. Unavailable providers are disabled, with the reason as a tooltip. The gear menu holds the embedding provider, search mode (hybrid / semantic / keyword), top-k and **Build index**.
- **API**: pass `"llm": {"provider": "openai", "model": "gpt-4o-mini"}` and/or `"embedding": {"provider": "openai", "model": "text-embedding-3-small"}`.
- **Fallback chain** (every LLM call): selected provider → the other provider → degraded mode. Responses report `llm_used`, `fallback_triggered` and `degraded_mode`.
- **Embeddings are never substituted**: each provider/model has its own Chroma collection (`cases__ollama__nomic_embed_text`, `cases__openai__text_embedding_3_small`). After switching, build the collection with **Build index** in the UI or `POST /api/v1/admin/reindex`.
- **Judge model**: set `JUDGE_PROVIDER` / `JUDGE_MODEL` to judge with a different (e.g. larger) model. Leave them blank to use the chat model selected for the request.

## Sample query (real output)

Request (`POST /api/v1/query`):

```json
{
  "query": "Which crops do well in Punjab during Kharif with N around 90, P 40, K 40, pH 6.5 and about 1200 mm rainfall?",
  "filters": {},
  "top_k": 8,
  "mode": "hybrid",
  "llm": {"provider": "ollama", "model": "llama3.1:8b"},
  "embedding": {"provider": "ollama", "model": "nomic-embed-text"}
}
```

This response was captured from the running server on 2026-10-06. **Ollama was not running and no OpenAI key was set in the build environment**, so it is the **degraded-mode** output: regex classifier, BM25-only retrieval (reported in `warnings` and `search_mode_used`), template explanation and rule-based judge. Lists and the explanation are truncated for length; nothing else is edited.

```json
{
  "classification": {
    "intent": "recommend_crop",
    "crop": null,
    "season": "Kharif",
    "state": "Punjab",
    "confidence": 0.85,
    "soil": {
      "n": 90.0,
      "p": 40.0,
      "k": 40.0,
      "ph": 6.5,
      "n_level": "high",
      "p_level": "medium",
      "k_level": "high",
      "ph_class": "neutral"
    },
    "weather": {
      "rainfall": 1200.0,
      "rainfall_class": "high"
    }
  },
  "similar_cases": [
    {
      "case_id": "CASE-012806",
      "narrative": "Rice grown in Punjab during Kharif 2012. Soil: N 150.0, P 50.0, K 40.0 (high nitrogen), pH 8.0 (alkaline). Weather: 24.1°C, 852 mm rainfall (low), 45% humidity. Inputs: fertilizer 150.8 kg/ha, pesticide 0.31 kg/ha. Area 2849000.0 ha, production 11390000.0, yield 3.92 t/ha (top 2% for Rice).",
      "metadata": {
        "crop": "Rice",
        "state": "Punjab",
        "season": "Kharif",
        "crop_year": 2012,
        "yield": 3.923636364,
        "n": 150.0,
        "p": 50.0,
        "k": 40.0,
        "ph": 8.0,
        "rainfall": 851.78,
        "temperature": 24.1,
        "yield_percentile_within_crop": 97.7
      },
      "scores": {
        "retrieval": 1.0,
        "proximity": 0.823,
        "yield_pct": 0.977,
        "recency": 0.687,
        "feedback": 0.5,
        "final": 0.87
      }
    }
  ],
  "recommendations": [
    {
      "crop": "Rice",
      "score": 0.887,
      "median_yield": 3.83,
      "mean_yield": 3.84,
      "yield_percentile": 97.1,
      "similarity": 1.0,
      "case_count": 3,
      "supporting_cases": [
        "CASE-012806",
        "CASE-012297",
        "CASE-011737"
      ],
      "risk": {
        "level": "medium",
        "probability": 0.098,
        "factors": [
          {
            "factor": "Nitrogen (N)",
            "impact": "medium",
            "description": "Nitrogen (N) (90kg/ha) is above the typical Rice range (54-88kg/ha); about 2% over the 90th percentile."
          }
        ]
      },
      "predicted_yield": {
        "value": 2.48,
        "low": 2.22,
        "high": 3.46,
        "unit": "t/ha"
      }
    },
    {
      "crop": "Maize",
      "score": 0.873,
      "median_yield": 3.71,
      "mean_yield": 3.86,
      "yield_percentile": 90.0,
      "similarity": 1.0,
      "case_count": 3,
      "supporting_cases": [
        "CASE-012292",
        "CASE-011732",
        "CASE-012801"
      ],
      "risk": {
        "level": "low",
        "probability": 0.117,
        "factors": []
      },
      "predicted_yield": {
        "value": 2.47,
        "low": 2.12,
        "high": 4.2,
        "unit": "t/ha"
      }
    }
  ],
  "impact_analysis": {
    "crop": "Rice",
    "global_importance": [
      {
        "feature": "Season",
        "importance": 22.7
      },
      {
        "feature": "Phosphorus (P kg/ha)",
        "importance": 16.7
      },
      {
        "feature": "State (location)",
        "importance": 12.7
      }
    ],
    "what_if_sensitivities": [
      {
        "factor": "Nitrogen (N kg/ha)",
        "baseline_value": 90.0,
        "current_predicted_yield": 2.48,
        "plus_20_predicted_yield": 2.48,
        "minus_20_predicted_yield": 2.07,
        "sensitivity_delta": 0.41
      }
    ],
    "summary": "For Rice, the model's predicted yield is most sensitive to Nitrogen (N kg/ha): moving it from -20% to +20% changes the prediction by +0.41 t/ha. Across historical Rice records, the most important input is Season. These are model associations from historical data, not causal effects."
  },
  "explanation": "[Deterministic Rule-Based Advisory]\n\n### Crop Suitability Analysis (Historical Evidence)\n\n1. **Rice** (suitability score 0.89)\n   - Historical median yield of matched cases: 3.83 t/ha (mean 3.84 t/ha).\n   - Mean yield percentile within Rice records: 97.1.\n   - Supporting records: [CASE-012806], [CASE-012297], [CASE-011737] (3 matched cases).\n   - Model-predicted yield: 2.48 t/ha (10th-90th percentile range 2.22-3.46) ...",
  "judge": {
    "grounded": 5,
    "consistent_with_data": 5,
    "actionable": 5,
    "safe": 5,
    "passed": true,
    "critique": "Rule-based validation: all cited case IDs are in the retrieved set; historical yields agree with model intervals; 3/3 recommendations include yield and risk estimates."
  },
  "agent_trace": [
    {
      "node": "guardrail_node",
      "ms": 4,
      "summary": "Input validation & security screening passed"
    },
    {
      "node": "classifier_node",
      "ms": 6,
      "summary": "Intent: recommend_crop, Crop: None, State: Punjab, Season: Kharif"
    },
    {
      "node": "retriever_node",
      "ms": 58,
      "summary": "Retrieved and ranked 8 cases (keyword search); filters: {'season': 'Kharif', 'state': 'Punjab'} (relaxed 1 step(s) to find enough cases)"
    },
    {
      "node": "analyst_node",
      "ms": 265,
      "summary": "Computed feature importance & what-if sensitivity for Rice"
    },
    {
      "node": "risk_node",
      "ms": 49,
      "summary": "Evaluated suitability risk for 3 crop(s)"
    },
    {
      "node": "yield_node",
      "ms": 49,
      "summary": "Predicted yield intervals for 3 crop(s)"
    },
    {
      "node": "recommender_node",
      "ms": 4,
      "summary": "Synthesized 3 crop recommendations "
    },
    {
      "node": "judge_node",
      "ms": 3,
      "summary": "Judge verdict: passed=True, avg_score=5.0"
    },
    {
      "node": "finalize_node",
      "ms": 0,
      "summary": "Finalized advisory after 1 judge pass(es)"
    }
  ],
  "warnings": [
    "Semantic search unavailable (Collection 'cases__ollama__nomic_embed_text' has not been built. Use POST /api/v1/admin/reindex (or 'Build index' in the UI).); used keyword (BM25) search only."
  ],
  "filters_applied": {
    "season": "Kharif",
    "state": "Punjab"
  },
  "search_mode_used": "keyword",
  "llm_used": null,
  "fallback_triggered": true,
  "degraded_mode": true,
  "is_synthetic_data": false,
  "disclaimer": "Disclaimer: This advisory is generated from historical crop, soil, and weather data patterns for decision support. Field outcomes may vary based on localized micro-climate, pests, and seed variety. Consult local agronomy extension services for critical inputs."
}
```

With Ollama running, the same request uses the LLM classifier, hybrid retrieval, an LLM-written explanation (checked by the output guardrails) and the LLM judge. That path is covered by tests with a fake chat model (`backend/tests/test_hardening.py`), but **it was not run against a live model** here; see *Verification status*.

## API endpoints (`/api/v1`, OpenAPI at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness, dataset provenance, index, model and provider status |
| GET | `/providers` | provider/model availability for the dropdowns |
| POST | `/query` | **full LangGraph pipeline** |
| POST | `/search` | keyword / semantic / hybrid search + filters + ranking (reports the mode actually used) |
| POST | `/classify` | classification agent only (+ "did you mean" warnings) |
| POST | `/recommend` | basic recommendation (retrieval + aggregation + explanation) |
| POST | `/predict/yield` | yield prediction with 10th–90th percentile interval |
| POST | `/predict/risk` | P(yield < crop's 25th percentile), risk level, top factors |
| POST | `/analyze/impact` | permutation importance + ±20% what-if sensitivity |
| POST | `/compare` | explain the yield gap between 2–5 farm profiles of one crop |
| GET | `/metadata/options` | crops, seasons, states, classes, thresholds |
| POST / GET | `/feedback`, `/feedback/stats` | feedback loop |
| POST / GET | `/eval/run`, `/eval/status/{id}`, `/eval/latest` | evaluation jobs (`/eval/latest` returns 404 until a run exists) |
| GET | `/admin/index/status` | whether the provider/model collection exists and is complete |
| POST / GET | `/admin/reindex`, `/admin/reindex/status/{job_id}` | build a collection in the background (JSON body `{embedding_provider, embedding_model, force}`) |
| POST | `/admin/retrain` | retrain yield + risk models (`{"reprocess_data": true}` re-runs cleaning first) |
| GET | `/graph` | LangGraph Mermaid diagram |

Errors always look like `{"error": {"code", "message", "details"}}`. Guardrail rejections:
- `400 SECURITY_VIOLATION`: prompt injection.
- `422 INVALID_PH`, `INVALID_RAINFALL`, etc.: out-of-range numbers, including numbers written in the question text.
- `429 RATE_LIMITED`.

Off-topic questions return `200` with `out_of_scope: true` and skip retrieval.

## Tests and evaluation

```bash
make test                     # pytest suite, no LLM or network needed
make eval                     # DeepEval golden set (21 queries)
.venv/bin/python scripts/run_eval.py --provider ollama --model llama3.1:8b --limit 5
```

- Tests isolate the feedback DB and eval outputs in a temp directory and patch LLM providers with a fake chat model.
- The evaluation always computes retrieval **hit-rate@5**, **recommendation hit-rate** and the in-pipeline **judge pass rate**.
- The DeepEval metrics (AnswerRelevancy, Faithfulness, ContextualRelevancy, ContextualPrecision, and the G-Eval metric "CropRecommendationQuality") need a judge LLM. Without one they are reported as `null`, with `deepeval_status` explaining why. They are never filled with placeholder numbers.
- Results go to `data/eval/results_<timestamp>.json` and `data/eval/latest.json`. Runs are sequential, with per-call timeouts (`EVAL_TIMEOUT_SECONDS`) and retries (`EVAL_MAX_RETRIES`) to tolerate slow local models.

Latest run in this environment (no LLM available, real data, BM25-only retrieval):

| Metric | Value |
|---|---|
| Retrieval hit-rate@5 | 0.62 |
| Recommendation hit-rate | 0.81 |
| Judge pass rate (rule-based judge) | 1.00 |
| DeepEval metrics | not run |

## Model metrics

Real data, time-aware split (train < 2016, test 2016–2020):

| Model | Metric |
|---|---|
| Yield (GradientBoosting on log1p yield) | R² (log scale) 0.94, median absolute % error 22.6%, 10–90% interval coverage 0.80 |
| Risk (calibrated GradientBoosting) | ROC-AUC 0.83, Brier 0.107, accuracy 0.86 (test base rate 0.16) |

`models/metrics.json` holds the full set, including `trained_on: real_kaggle_dataset`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Header shows "Ollama not reachable" | Start Ollama (`ollama serve`) and pull the two models above. The app keeps working in degraded mode meanwhile. |
| `warnings` says "Semantic search unavailable … has not been built" | Build the collection for the selected embedding provider: gear menu → **Build index**, or `POST /api/v1/admin/reindex`. Indexing ~19.6k cases with `nomic-embed-text` takes several minutes. |
| Switched embedding provider and search fell back to keyword | Each provider/model has its own collection; build it once. |
| `409 EMBEDDINGS_UNAVAILABLE` from reindex | The chosen embedding provider is down or has no API key. |
| Very slow answers / eval timeouts | Small local models are slow on CPU. Use a smaller model (`qwen2.5:3b`), raise `EVAL_TIMEOUT_SECONDS`, or evaluate with `--limit`. |
| `XGBoost Library … libomp.dylib could not be loaded` printed at startup (macOS) | Harmless: the models fall back to scikit-learn GradientBoosting. `brew install libomp` enables XGBoost; then retrain. |
| `pip` SSL errors during `docker compose build` | TLS-intercepting proxy: add its root CA to `docker/certs/`. |
| Changed the dataset | Run `python scripts/download_data.py && python scripts/train_models.py`, then rebuild the vector index (the BM25 cache rebuilds automatically). Existing feedback case IDs refer to the old dataset. |

## Verification status

What was actually run in the build environment (macOS, Python 3.14, Node 26, no Ollama, no OpenAI key):

| Check | Result |
|---|---|
| `pytest backend/tests` | **54 passed** |
| Backend start (`uvicorn`), `/health` | OK, real dataset (19,595 cases) detected |
| Every endpoint called over HTTP | OK (including the expected 400 / 404 / 409 / 422 cases) |
| Dataset merge + cleaning, model training | OK on the real Kaggle files |
| `npm run build` | OK, no TypeScript errors |
| Frontend dev server in headless Chrome | loads and calls `/health`, `/providers`, `/metadata/options`, `/admin/index/status` successfully (CORS OK) |
| Evaluation harness (`scripts/run_eval.py`) | completed 21/21 queries; DeepEval metrics skipped (no judge LLM) |
| Architecture diagram | regenerated (`docs/architecture.{png,jpg,pdf}`) |
| Docker | both images build; `docker compose up` serves the backend (healthy, real data, `/query` answering) and the frontend on :3000. The backend image is ~2.6 GB, mostly XGBoost's CUDA (NCCL) dependency on Linux. Building behind this network's TLS proxy needed the `docker/certs/` CA hook |

**Not verified live**, because no model server was available:
- Ollama/OpenAI chat calls
- LLM structured classification, LLM explanations and the LLM judge
- DeepEval metric scores
- Chroma semantic search with real embeddings (no collection could be built)

These paths are implemented and covered by mocked tests (fake chat model, patched providers). When they are unavailable, the system degrades to the verified deterministic paths.

## Project layout

```
backend/app/        FastAPI app: api/v1, agents (LangGraph), retrieval, ml, guardrails, llm, data, evaluation, feedback
backend/tests/      pytest suite
frontend/src/       React app: api, components, pages, types, utils
scripts/            download_data, generate_synthetic_data, build_index, train_models, run_eval, make_architecture_diagram
docs/               ARCHITECTURE, DESIGN, ASSUMPTIONS, DATA_PROFILE, DEMO_SCRIPT, architecture.{png,jpg,pdf}
data/               raw / processed / chroma / eval (generated, git-ignored)
models/             trained models + metrics.json
docker/certs/       optional proxy root CAs for Docker builds
```

> Decision support only: recommendations come from historical, state-level data and statistical models, not field trials. See the disclaimer in every response.
