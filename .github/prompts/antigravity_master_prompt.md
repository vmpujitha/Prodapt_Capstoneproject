# MASTER BUILD PROMPT: AI-Powered Agriculture Crop Advisory Assistant

> Paste everything below this line into Antigravity as one task. Do NOT ask clarifying questions. Where something is ambiguous, choose the most reasonable option, record it in `docs/ASSUMPTIONS.md`, and keep building.

---

## 0. ROLE AND MISSION

You are a senior ML/AI engineer. Build, run, and verify a complete, working, containerizable microservice plus a React front end called **AI-Powered Agriculture Crop Advisory Assistant**.

Users type natural-language farming questions. The system retrieves the most similar historical farming cases from a crop/soil/weather CSV dataset and returns:
- ranked similar cases,
- crop recommendations,
- yield and risk predictions,
- soil/weather/input impact analysis,
- a validated, grounded explanation.

You must deliver ALL of: working code, README, architecture diagram, design document, tests, evaluation harness, and front end. "Working" means you actually started the backend and front end, called the endpoints, and confirmed correct output before finishing.

---

## 1. FIXED TECH STACK (DO NOT SUBSTITUTE)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (backend), TypeScript (frontend) |
| API | FastAPI + Uvicorn, Pydantic v2, async endpoints |
| LLM orchestration | **LangChain** (LCEL chains, structured output, retrievers, tools) |
| Multi-agent workflow | **LangGraph** (`StateGraph`) |
| LLM providers | **Ollama (local, default)** and **OpenAI (fallback)**, selectable at runtime via dropdown |
| Embeddings | Selectable: **Ollama `nomic-embed-text` (default when no OpenAI key)** or **OpenAI `text-embedding-3-small`** |
| Vector store | **ChromaDB** (persistent, local directory) |
| Keyword search | BM25 (`rank_bm25`) |
| Classical ML | scikit-learn + XGBoost (yield and risk prediction) |
| Evaluation | **DeepEval** (with a custom judge model wrapper) |
| Feedback storage | SQLite via SQLAlchemy |
| Frontend | **React + Vite + TypeScript + Tailwind CSS**, Recharts for charts |
| Packaging | Docker + docker-compose, `.env` configuration, Makefile |
| Tests | pytest (backend), Vitest optional (frontend) |

### Hard rules about the stack
1. Use LangChain and LangGraph primitives for all LLM logic. **Do NOT write custom agent frameworks, custom ReAct loops, or hand-rolled orchestration classes.** No CrewAI, AutoGen, LlamaIndex, or Haystack.
2. Agents are LangGraph nodes. Each node is a plain function or LCEL runnable that reads and writes a typed shared state.
3. All LLM outputs that feed code must use `.with_structured_output(PydanticModel)` (or a JSON-parser fallback if the model lacks tool calling). Never regex-parse free text for critical fields.
4. Pin dependency versions in `requirements.txt` and `package.json` after verifying that they install and run together.
5. Do not add technologies outside this table unless a listed one cannot work. If so, document the reason in `docs/ASSUMPTIONS.md`.

---

## 2. THE USER HAS NO OPENAI KEY (CRITICAL DESIGN CONSTRAINT)

The system must work fully offline with Ollama, and OpenAI must be optional.

### 2.1 Provider abstraction
Create `app/llm/factory.py` exposing:
- `get_chat_model(provider: str | None, model: str | None, temperature: float = 0.0) -> BaseChatModel`
- `get_embeddings(provider: str | None, model: str | None) -> Embeddings`
- `list_providers() -> list[ProviderStatus]`

Use `langchain-ollama` (`ChatOllama`, `OllamaEmbeddings`) and `langchain-openai` (`ChatOpenAI`, `OpenAIEmbeddings`).

### 2.2 Provider status and the dropdown
`GET /api/v1/providers` returns each provider with:
```json
{
  "provider": "ollama",
  "available": true,
  "reason": null,
  "chat_models": ["llama3.1:8b", "qwen2.5:7b"],
  "embedding_models": ["nomic-embed-text"],
  "default_chat_model": "llama3.1:8b"
}
```
- Ollama availability: ping `OLLAMA_BASE_URL/api/tags` with a 2-second timeout and list the locally pulled models.
- OpenAI availability: `available=false`, `reason="OPENAI_API_KEY not set"` when the key is missing. The UI must still show it in the dropdown, but disabled with that reason as a tooltip.

### 2.3 Automatic fallback chain
Every LLM call goes through one helper, `invoke_with_fallback`, with this order:
1. The provider/model the user selected.
2. The other provider, if available.
3. **Degraded mode**: no LLM. Return deterministic, template-based output built only from retrieved cases and ML models. The response must set `"degraded_mode": true` and `"llm_used": null`.

Every API response includes `llm_used: {provider, model}` and `fallback_triggered: bool`. The service must never crash because a provider is down or unconfigured.

### 2.4 Embedding and ChromaDB compatibility (IMPORTANT)
Embedding dimensions differ between providers (nomic-embed-text = 768, OpenAI text-embedding-3-small = 1536), and a Chroma collection cannot mix them.
- Name collections `cases__{provider}__{model_slug}`.
- `POST /api/v1/admin/reindex` takes `{embedding_provider, embedding_model}` and builds that collection if it does not exist.
- Default embedding provider: OpenAI if `OPENAI_API_KEY` is set AND the user chose it; otherwise Ollama.
- The UI "Advanced settings" panel has an embedding provider dropdown. A missing index shows a "Build index" button with progress.
- Ingestion must be idempotent and batched (e.g., 200 docs per batch) and must show progress logs.

### 2.5 Defaults
`.env.example` must contain: `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_CHAT_MODEL=llama3.1:8b`, `OLLAMA_EMBED_MODEL=nomic-embed-text`, `OPENAI_API_KEY=`, `OPENAI_CHAT_MODEL=gpt-4o-mini`, `OPENAI_EMBED_MODEL=text-embedding-3-small`, `DEFAULT_LLM_PROVIDER=ollama`, `DEFAULT_EMBEDDING_PROVIDER=ollama`, `CHROMA_DIR=./data/chroma`, `DATA_CSV_PATH=./data/raw/crop_yield.csv`.

The README must include the commands `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`.

---

## 3. DATASET HANDLING

Source: Kaggle "Crop Yield Data with Soil and Weather Dataset" (anshumish). Expected fields: `Crop, Crop_Year, Season, State, Area, Production, Yield, Fertilizer, Pesticide, N, P, K, pH, Temperature, Rainfall, Humidity`.

### 3.1 Rules
1. **Do not assume the exact column names.** First inspect the CSV (`head`, `dtypes`, `describe`, null counts) and write the findings to `docs/DATA_PROFILE.md`.
2. Implement a schema-mapping layer, `app/data/schema.py`, with a canonical-name alias dictionary (e.g., `crop_year`/`Year` → `crop_year`, `temp`/`Temperature` → `temperature`). All downstream code uses only canonical names.
3. Download: try the Kaggle API if `KAGGLE_USERNAME`/`KAGGLE_KEY` exist. Otherwise instruct the user to place the CSV at `data/raw/crop_yield.csv`.
4. **If no CSV is present**, generate a clearly labeled synthetic dataset (`data/raw/SYNTHETIC_crop_yield.csv`, ≥5,000 rows, realistic ranges, crop-specific soil/weather patterns) via `scripts/generate_synthetic_data.py` so the whole system can be demoed. Print a loud warning at startup and show a "SYNTHETIC DATA" banner in the UI whenever it is in use. Never silently pass synthetic data off as real.
5. Cleaning: strip and title-case categoricals, drop duplicates, handle nulls (impute numerics with the crop-wise median; drop rows missing Crop/State/Yield), clip impossible values, and write the cleaned file to `data/processed/cases.parquet` (or CSV). Log row counts before and after each step.
6. Assign a stable `case_id` (e.g., `CASE-000123`) to every row.

### 3.2 Derived categorical features (for filtering and for text)
- `ph_class`: acidic (<6.0), neutral (6.0–7.5), alkaline (>7.5)
- `n_level`, `p_level`, `k_level`: low/medium/high from dataset tertiles
- `temp_class`: cool/moderate/hot (tertiles)
- `rainfall_class`: low/moderate/high (tertiles)
- `humidity_class`: low/moderate/high
- `yield_percentile_within_crop`: 0–100
Persist the thresholds to `data/processed/thresholds.json` so queries are classified with the same cut-offs.

### 3.3 Document construction (each row becomes one LangChain `Document`)
`page_content` is a natural-language case narrative, for example:
> "Rice grown in Punjab during Kharif 2019. Soil: N 90, P 42, K 43 (medium nitrogen), pH 6.5 (neutral). Weather: 26.5°C, 1200 mm rainfall (high), 80% humidity. Inputs: fertilizer 120 kg/ha, pesticide 2.1 kg/ha. Area 1500 ha, production 4200 t, yield 2.8 t/ha (top 30% for rice)."

`metadata` holds ALL canonical fields plus the derived classes and `case_id`, stored as flat scalar types only (Chroma constraint).

---

## 4. REPOSITORY STRUCTURE (create exactly this layout)

```
crop-advisory/
├── README.md
├── Makefile
├── docker-compose.yml
├── .env.example
├── docs/
│   ├── ARCHITECTURE.md        # Mermaid + explanation
│   ├── architecture.pdf       # generated diagram (also .png/.jpg)
│   ├── DESIGN.md              # decisions + trade-offs
│   ├── ASSUMPTIONS.md
│   ├── DATA_PROFILE.md
│   └── DEMO_SCRIPT.md         # 8-minute demo flow + 2-minute Q&A prep
├── data/{raw,processed,chroma}/   # gitignored except .gitkeep
├── scripts/
│   ├── download_data.py
│   ├── generate_synthetic_data.py
│   ├── build_index.py
│   ├── train_models.py
│   ├── make_architecture_diagram.py
│   └── run_eval.py
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas/            # Pydantic request/response models
│   │   ├── api/v1/             # routers: query, search, classify, recommend, predict, analyze, compare, feedback, eval, admin, providers, health
│   │   ├── llm/factory.py, fallback.py
│   │   ├── data/schema.py, ingest.py, features.py
│   │   ├── retrieval/vectorstore.py, bm25.py, hybrid.py, filters.py, ranking.py
│   │   ├── guardrails/input_validation.py, output_validation.py
│   │   ├── agents/             # LangGraph: state.py, nodes.py, graph.py, prompts.py
│   │   ├── ml/yield_model.py, risk_model.py, impact.py
│   │   ├── evaluation/deepeval_runner.py, judge.py, golden_set.json
│   │   └── feedback/store.py, reranker.py
│   └── tests/
└── frontend/
    ├── Dockerfile
    ├── package.json, vite.config.ts, tailwind.config.js, tsconfig.json
    └── src/{api,components,pages,hooks,types}/
```

---

## 5. BACKEND SPECIFICATION

### 5.1 Basic tier (Requirement 1)

**A. Basic RAG for case similarity.** Chroma similarity search over case narratives. Returns the top-k cases with the similarity score, the full metadata, and the narrative.

**B. Hybrid search (keywords + semantic).**
- Run BM25 and Chroma vector search separately; merge with **Reciprocal Rank Fusion** (k=60). The weights are configurable (`HYBRID_ALPHA`, default 0.5).
- BM25 index is built from the same narratives and cached on disk (pickle) next to the index.
- **Both retrievers must respect the same metadata filters.** Pre-filter the BM25 corpus. Pass a `where` clause to Chroma.
- Expose `mode`: `semantic | keyword | hybrid` on the search endpoint, so the UI can demonstrate the difference.

**C. Metadata filtering.** Support filters on: `crop`, `season`, `state`, `crop_year` (range), `ph_class`/`ph` (range), `n_level`/`p_level`/`k_level`, `temp_class`, `rainfall_class`, `humidity_class`. Build Chroma `where` clauses with `$and`/`$in`/`$gte`/`$lte` correctly. Filters may come from the explicit request OR from the classification agent (explicit request values override).

**D. Crop and agricultural condition classification agent.** An LCEL chain with structured output → `QueryClassification`:
```
intent: recommend_crop | explain_yield_difference | alternative_crop | similar_cases | out_of_scope
crop, season, state (nullable, normalized against known values from the dataset via fuzzy matching)
soil: {n,p,k,ph, n_level,p_level,k_level, ph_class} (any nullable)
weather: {temperature, rainfall, humidity, temp_class, rainfall_class, humidity_class}
inputs: {fertilizer, pesticide}
confidence: 0-1
```
A deterministic regex/keyword fallback extractor must exist for degraded mode (e.g., "pH 6.5", "N=90", "rainfall 1200mm").

**E. Basic crop recommendation.** From the retrieved and ranked cases, aggregate by crop: count, mean/median yield, mean yield percentile, and mean similarity. Return the top-N crops with a score breakdown and supporting `case_id`s. The LLM writes a concise explanation grounded in those aggregates only.

**F. Input validation guardrails** (`guardrails/input_validation.py`), applied BEFORE any LLM call:
- Query length 5–1000 chars; strip control characters; reject empty or whitespace-only input.
- Numeric sanity ranges: pH 0–14, N/P/K 0–500, temperature −10–55 °C, rainfall 0–5000 mm, humidity 0–100 %, year 1950–current+1. Out-of-range values return a 422 with a clear, field-specific message.
- Prompt-injection screening (patterns like "ignore previous instructions", "system prompt", role-play overrides, code/SQL fragments). Block with a safe message.
- Domain check: queries unrelated to agriculture → `out_of_scope` response, no retrieval, no LLM answer.
- Unknown crop/state/season values: fuzzy-match to the nearest known value, or return "did you mean…" suggestions.
- Rate limiting (simple in-memory, per IP), configurable.

**Output guardrails** (`output_validation.py`): every case ID cited by the LLM must exist in the retrieved set; every number quoted must come from the retrieved cases or model outputs; otherwise strip the claim or fall back to the template answer. Always attach a short disclaimer that results are historical-data-based decision support, not a guarantee.

**G. Case prioritization** (`retrieval/ranking.py`). Final score for each retrieved case:
```
score = w1*retrieval_score_norm
      + w2*condition_proximity     # 1 - normalized weighted distance across N,P,K,pH,temp,rainfall,humidity (only over the fields the user supplied)
      + w3*yield_percentile_within_crop
      + w4*recency                 # newer crop_year scores higher
      + w5*feedback_boost          # from the feedback store (see 5.2)
```
Weights live in config. Return the per-component breakdown with each case so the ranking is explainable in the UI.

### 5.2 Advanced tier (Requirement 2)

**H. Multi-agent collaboration with LangGraph** (`agents/graph.py`). One `StateGraph` over a `TypedDict`/Pydantic `AdvisoryState` (query, validated_input, classification, filters, retrieved_cases, ranked_cases, soil_weather_analysis, yield_prediction, risk_assessment, recommendation, judge_verdict, revision_count, trace, llm_used, degraded_mode). Nodes:

1. `guardrail_node`: validate input; route to END on rejection.
2. `classifier_node`: classification agent (4D).
3. `retriever_node`: hybrid search + filters + prioritization.
4. Parallel fan-out after retrieval:
   - `analyst_node`: soil/weather/input impact analysis (5.2 J).
   - `yield_node`: yield prediction (5.2 I).
   - `risk_node`: crop suitability risk (5.2 K).
5. `recommender_node`: synthesizes everything into a final `Recommendation`.
6. `judge_node`: LLM-as-judge (5.2 L).
7. Conditional edge: if `judge_verdict.passed` is false and `revision_count < 2`, route back to `recommender_node` with the judge's critique; otherwise route to `finalize_node`.
8. `finalize_node`: attach the trace, the disclaimer and metadata.

Use LangGraph conditional edges and fan-out/fan-in. Record a per-node `trace` (node name, duration ms, short summary) and return it to the UI to display the agent workflow. Expose the compiled graph as a Mermaid diagram via `GET /api/v1/graph`.

**I. Crop yield prediction.** Train XGBoost (fall back to GradientBoosting) on the cleaned dataset with features: crop, season, state (encoded), N, P, K, pH, temperature, rainfall, humidity, fertilizer, pesticide, area, and year. Do a time-aware or grouped train/test split, and report MAE, RMSE and R² in `models/metrics.json`. Provide a prediction **interval** (quantile models or residual-based). Persist with joblib; train via `scripts/train_models.py` and auto-train on first startup if the model files are missing. Always state whether a model is trained on real or synthetic data.

**J. Soil, weather and farming-input impact analysis.** (a) Global: permutation importance (and SHAP if it installs cleanly) per crop. (b) Local: for the user's conditions, a what-if sensitivity: vary one factor at a time (N, P, K, pH, rainfall, temperature, fertilizer, pesticide) ±20% and report the predicted yield delta. (c) `POST /compare`: two or more farm profiles of the same crop → ranked list of the factors that best explain the yield gap, with magnitudes. Return chart-ready JSON.

**K. Crop suitability risk prediction.** For each candidate crop, estimate `P(yield < crop's 25th percentile)` using a classifier trained on the dataset (calibrated). Output a risk level (low/medium/high), the probability, and the top 3 contributing risk factors (e.g., "rainfall far below the crop's typical range"). Combine this with a rule-based range check against each crop's historical 10th–90th percentile soil/weather envelope.

**L. LLM-as-judge for recommendation validation.** The `judge_node` prompts a (preferably different or larger) model with the retrieved cases, the predictions, and the draft recommendation. It returns a structured verdict: `{grounded: 1-5, consistent_with_data: 1-5, actionable: 1-5, safe: 1-5, passed: bool, critique: str}`. `passed` = all scores ≥ 3 and the mean ≥ 3.5. Judge model is configurable (`JUDGE_PROVIDER`, `JUDGE_MODEL`); default is the same as the chat model. In degraded mode, run a deterministic rule-based validator in its place.

**M. DeepEval evaluation.** Implement `DeepEvalBaseLLM` wrappers around the LangChain chat model so DeepEval metrics run on Ollama or OpenAI. Metrics: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualRelevancyMetric`, `ContextualPrecisionMetric` and a custom `GEval` "CropRecommendationQuality" metric. Create `golden_set.json` with ≥20 realistic queries across all 3 scenarios from the brief (similar-conditions recommendation, same-crop yield-gap explanation, alternative-crop selection), each with expected crop(s)/criteria. `scripts/run_eval.py` and `POST /api/v1/eval/run` execute the suite (async job, with a status endpoint), save results to `data/eval/results_{timestamp}.json`, and return aggregate scores. Also compute simple retrieval metrics (hit-rate@k of the expected crop). Eval must tolerate slow local models (timeouts, retries, limited concurrency).

**N. Feedback loop for continuous improvement.** SQLite table `feedback(id, query, case_ids, recommended_crop, rating (+1/-1), comment, helpful_case_ids, created_at)`. `POST /feedback` stores it. `feedback/reranker.py` computes a bounded `feedback_boost` per case and per crop (Laplace-smoothed, clipped to ±0.1) that feeds into case prioritization. `GET /feedback/stats` returns aggregates. Document honestly that this is a lightweight re-ranking signal and not model retraining. Add a `POST /admin/retrain` hook to retrain the yield and risk models when the data changes.

### 5.3 API endpoints (all under `/api/v1`, OpenAPI docs at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + index/model/provider status |
| GET | `/providers` | provider/model availability for the dropdown |
| POST | `/query` | **full pipeline** (LangGraph): the main endpoint |
| POST | `/search` | hybrid/semantic/keyword search + filters + ranking |
| POST | `/classify` | classification agent only |
| POST | `/recommend` | basic crop recommendation (no advanced agents) |
| POST | `/predict/yield` | yield + interval |
| POST | `/predict/risk` | per-crop suitability risk |
| POST | `/analyze/impact` | feature importance + what-if sensitivity |
| POST | `/compare` | explain yield differences between farm profiles |
| GET | `/metadata/options` | distinct crops, seasons, states, classes (for filter dropdowns) |
| POST | `/feedback`, GET `/feedback/stats` | feedback loop |
| POST | `/eval/run`, GET `/eval/status/{id}`, GET `/eval/latest` | DeepEval |
| POST | `/admin/reindex`, `/admin/retrain` | maintenance |
| GET | `/graph` | LangGraph Mermaid |

All request and response bodies are Pydantic models with examples. Consistent error format: `{error: {code, message, details}}`. CORS enabled for the frontend origin. Structured JSON logging with request IDs.

### 5.4 `POST /query` sample (use this exact shape in the README)
Request:
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
Response (abbreviated):
```json
{
  "classification": {"intent": "recommend_crop", "state": "Punjab", "season": "Kharif", "soil": {...}, "weather": {...}, "confidence": 0.9},
  "similar_cases": [{"case_id": "CASE-001234", "narrative": "...", "metadata": {...}, "scores": {"retrieval": 0.82, "proximity": 0.91, "yield_pct": 0.77, "final": 0.84}}],
  "recommendations": [{"crop": "Rice", "score": 0.84, "median_yield": 2.9, "supporting_cases": ["CASE-001234"], "risk": {"level": "low", "probability": 0.12, "factors": []}, "predicted_yield": {"value": 2.8, "low": 2.2, "high": 3.4}}],
  "impact_analysis": {...},
  "explanation": "...",
  "judge": {"passed": true, "scores": {...}},
  "agent_trace": [{"node": "classifier", "ms": 640, "summary": "..."}],
  "llm_used": {"provider": "ollama", "model": "llama3.1:8b"},
  "fallback_triggered": false,
  "degraded_mode": false,
  "disclaimer": "..."
}
```

---

## 6. FRONTEND SPECIFICATION (React + Vite + TS + Tailwind)

Clean, responsive, professional agricultural theme (greens/earth tones). Components talk to the backend via a typed API client (`src/api/client.ts`) using `VITE_API_BASE_URL`. Handle loading, empty and error states everywhere.

**Global header:**
- **LLM provider dropdown + model dropdown** (populated from `/providers`; unavailable providers disabled, with the reason as a tooltip; selection persisted in localStorage).
- Status badge showing the provider that was actually used and a "fallback used" / "degraded mode" indicator after each query.
- "Advanced settings" popover: embedding provider/model dropdown, search mode (hybrid/semantic/keyword), top-k, "Build index" button when the selected collection is missing.
- "SYNTHETIC DATA" banner when applicable.

**Pages/tabs:**
1. **Advisor** (main): natural-language query box with 4–5 clickable example queries; collapsible filter panel (crop, season, state, year range, soil/weather classes, populated from `/metadata/options`); results area with: recommendation cards (crop, score, predicted yield with interval, risk badge), explanation text with clickable `case_id` citations, ranked similar-case table/cards (expandable, with score breakdown bars), impact charts (Recharts: bar chart of feature importance, tornado chart of what-if sensitivities), judge verdict panel, and the **agent workflow trace** (timeline of LangGraph nodes with durations).
2. **Compare Farms:** form for 2+ farm profiles → yield-gap factor explanation chart.
3. **Evaluation:** button to run DeepEval, a progress indicator, and a metrics dashboard from `/eval/latest`.
4. **Feedback:** thumbs up/down + optional comment + "which cases helped" checkboxes on each answer; a small stats view.

Show user-facing validation errors from the guardrails inline, next to the relevant input.

---

## 7. QUALITY, TESTING AND VERIFICATION (MANDATORY)

Write pytest tests for: schema mapping, cleaning, document construction, filter-to-where translation, RRF fusion, ranking math, input guardrails (valid, out-of-range, injection, off-topic), classifier fallback extractor, output guardrails, the LLM fallback chain (mock provider failures), yield/risk model loading and prediction shape, API contract tests via `TestClient` for every endpoint (mock the LLM using a fake chat model so the tests run without Ollama or OpenAI), and the LangGraph graph (happy path, rejection path, judge-revision loop).

**Before declaring completion you must actually:**
1. Install dependencies and start the backend (`uvicorn`). Confirm that `/health` is OK.
2. Run ingestion and build the index (use the synthetic data if the real CSV is absent). Train the models.
3. Call `/query` with the sample query above and paste the real output into the README. If Ollama is not running in your environment, demonstrate the degraded-mode output and a mocked-LLM run, and say so explicitly. Never fabricate output.
4. Run `pytest` and make sure all tests pass.
5. Build and start the frontend (`npm run build` must succeed with no TypeScript errors), then confirm it connects to the backend.
6. Generate `docs/architecture.pdf` (and `.png`/`.jpg`) from `scripts/make_architecture_diagram.py` (use graphviz or matplotlib, and keep the Mermaid source in `ARCHITECTURE.md` too).

---

## 8. DOCUMENTATION DELIVERABLES

- **README.md**: overview, architecture summary, prerequisites (Python, Node, Ollama + model pulls), step-by-step setup (local and Docker), the data setup options (Kaggle / manual / synthetic), how to build the index and train models, how to run backend + frontend, how to switch providers, a sample query with the real response, an endpoint table, how to run tests and evaluation, troubleshooting (Ollama not running, a missing Chroma collection after switching embeddings, slow local models).
- **ARCHITECTURE.md + architecture.pdf**: end-to-end data flow: CSV → cleaning → feature derivation → documents → embeddings → Chroma + BM25 → (query) → guardrails → classifier → hybrid retrieval + filters → ranking → parallel analysis agents → recommender → judge → response → feedback → re-ranking. Show the LLM provider abstraction and the fallback chain.
- **DESIGN.md**: every key decision with the alternatives and trade-offs. It must cover at least: hybrid vs. pure semantic search; the text-narrative embedding of numeric data and its limits (embeddings are weak on numbers, which is why numeric proximity re-ranking and metadata filters exist); ChromaDB vs. alternatives; Ollama-first with OpenAI fallback (quality vs. cost vs. privacy; small local models are weaker at structured output and judging); per-provider collections; LangGraph vs. a single chain; classical ML for yield instead of the LLM; RRF vs. weighted-score fusion; the limits of an LLM judge on a small model; the feedback loop as re-ranking only; dataset caveats (correlation is not causation, regional bias, synthetic fallback).
- **DEMO_SCRIPT.md**: an 8-minute demo plan (suggested order: architecture → three scenario queries from the brief → provider switch + fallback → impact analysis → eval dashboard → feedback), and a list of likely Q&A questions with answers.
- **ASSUMPTIONS.md**: every assumption you made.

---

## 9. ANTI-DEVIATION GUARDRAILS (READ CAREFULLY)

**You MUST:**
- Satisfy every item in Requirement 1 and Requirement 2 of the brief, plus the frontend and all deliverables.
- Keep the code modular: one responsibility per module, type hints and docstrings on public functions, config in one place, no hard-coded paths, keys or model names in business logic.
- Use LangChain/LangGraph for all LLM and agent logic.
- Make every feature degrade gracefully with no LLM available.
- Keep answers grounded in retrieved cases and model outputs. The LLM explains data; it must not invent facts, cases or numbers.
- Be honest in all docs about what was verified and what was not.

**You MUST NOT:**
- Leave `TODO`, `pass`-only stubs, placeholder functions, "implement later" comments, or mocked-up endpoints that return hard-coded data in the production path.
- Add features beyond this scope (no auth system, no payments, no maps, no extra microservices, no weather-API integrations).
- Hard-code or commit API keys. Never print keys in logs.
- Silently substitute the dataset, a framework, or a provider.
- Reimplement agent loops, tool-calling or orchestration by hand.
- Use `localStorage` for anything except UI preferences (provider selection etc.).
- Present synthetic data as real, or fabricate benchmark numbers, evaluation scores or sample outputs.
- Stop after scaffolding. Finish all phases.

**If something fails:** diagnose, fix, and retry. If it is truly blocked (e.g., Ollama is unavailable in your sandbox), implement the real code path anyway, test it with mocks, and document the limitation in the README under "Verification status".

---

## 10. EXECUTION ORDER (follow, and checkpoint after each phase)

1. **Phase 1**: Repo scaffold, config, `.env.example`, data profiling, schema mapping, cleaning, features, synthetic fallback.
2. **Phase 2**: LLM/embedding factory, provider status, fallback helper.
3. **Phase 3**: Documents, Chroma indexing (per-provider collections), BM25, hybrid fusion, filters, ranking.
4. **Phase 4**: Guardrails, classifier, basic recommendation, and the Requirement 1 endpoints + tests.
5. **Phase 5**: Yield model, risk model, impact analysis, compare.
6. **Phase 6**: LangGraph multi-agent graph, judge, `/query` endpoint.
7. **Phase 7**: Feedback store + re-ranker, DeepEval harness + golden set.
8. **Phase 8**: React frontend (all four tabs, the provider dropdown, and the fallback indicators).
9. **Phase 9**: Docker, Makefile, architecture diagram, all docs, final end-to-end verification (Section 7).

At the end, print a **completion report**: a checklist mapping every requirement from the brief to the file(s) that implement it, the verified test results, how to start everything with 3 commands or fewer, and the known limitations.
