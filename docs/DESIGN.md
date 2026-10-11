# Design Decisions and Trade-offs

Each section gives the decision, the alternatives considered, and the trade-off accepted.

## 1. Hybrid retrieval instead of pure semantic search

**Decision**: run BM25 and Chroma vector search on the same filtered corpus and merge them with Reciprocal Rank Fusion. The search endpoint exposes `semantic | keyword | hybrid`, so the difference can be demonstrated.

- Pure semantic search captures paraphrases ("water-scarce" ≈ "low rainfall") but is unreliable on exact tokens such as crop or state names and years.
- Pure keyword search is exact and fast, but it misses paraphrases and is brittle with free text.
- Hybrid search gets most of both. It also gives a graceful degradation path: when the embedding provider is down, the same pipeline runs keyword-only and says so in `warnings` / `search_mode_used`.

## 2. Embedding numeric data as text narratives, and its limits

Each row becomes a sentence ("Rice grown in Punjab during Kharif 2012. Soil: N 150 …") so it can be embedded and searched. **Embeddings and lexical matching are weak on numbers**: "1200 mm" is not close to "1150 mm" in either space. During development, BM25 even ranked an Urad case first because its *production* figure was "1200.0". The design therefore does not rely on text similarity for numbers:

- **Metadata filters** (classes such as `rainfall_class`, plus ranges) narrow candidates structurally.
- **Condition proximity** in ranking computes a normalised distance over the numeric fields the user actually supplied.
- **BM25 ignores bare numbers** (years excepted) and stopwords.
- Derived **tertile classes** (low/medium/high) are written into the narrative, so text search has categorical words to match.

## 3. ChromaDB

**Chosen** over FAISS (no metadata filtering or persistence model out of the box), pgvector/Qdrant/Weaviate (extra services, contrary to the single-microservice brief) and LanceDB (less mature LangChain integration at build time). Chroma is embedded, persistent, supports `where` filters with `$and/$in/$gte/$lte`, and has first-class LangChain support. Its limits are single-node scale and flat scalar metadata, so all metadata is flattened.

## 4. Ollama first, OpenAI as an optional fallback

- **Privacy and cost**: farm data and questions stay local and free by default.
- **Quality**: an 8B local model is noticeably weaker at structured output (JSON schemas), long-context grounding and judging. Mitigations:
  - `with_structured_output` with a JSON-parsing fallback
  - deterministic validation of everything the LLM returns (normalisation, number and citation guardrails)
  - LLM output used only for wording; numbers come from data and models
- **Availability**: `invoke_with_fallback` tries the selected provider, then the other one, then degraded mode. A missing key or a stopped server never crashes a request.

## 5. One Chroma collection per provider/model

Vectors from `nomic-embed-text` (768-d) and `text-embedding-3-small` (1536-d) are incompatible, and even same-dimension models are not comparable. Collections are named `cases__{provider}__{model_slug}`. Switching embeddings requires building that collection once (background job with progress in the UI). The earlier behaviour of silently embedding with a fake model when the provider was down was removed: it filled a collection named after one model with random vectors.

## 6. LangGraph instead of a single chain

The workflow has real control flow: early exits (guardrail rejection, out-of-scope), a fan-out/fan-in of three independent analyses, and a judge step whose verdict is reported with the answer (a revision loop exists in the design but is currently disabled in `check_judge_condition`). A single LCEL chain would hide this in imperative code. `StateGraph` makes it explicit, testable node by node, inspectable (`/graph`), and traceable (per-node timings in `agent_trace`). The cost is a little more boilerplate and state-merging rules (an `operator.add` reducer for the trace).

## 7. Classical ML for yield and risk, not the LLM

LLMs are not calibrated regressors and would hallucinate yields. Gradient boosting on tabular features is the right tool: fast, measurable (MAE, R², coverage, ROC-AUC, Brier), and explainable with permutation importance. Further choices:

- **Log target**: yields range from about 0.1 t/ha to about 20,000 nuts/ha (coconut), so training on `log1p(yield)` keeps crops on comparable terms.
- **Time-aware split**: train on earlier years and test on 2016–2020. This is closer to real use (forecasting a coming season) than a random split, which leaks year effects.
- **Residual-based per-crop intervals**: these are cheaper than quantile models, and their measured 10–90% coverage on the test set is 0.80, as intended.
- **XGBoost vs GradientBoosting**: XGBoost is used when it loads. On macOS without `libomp` the code falls back to scikit-learn GradientBoosting, and `model_type` in the metrics says which one ran.

## 8. RRF vs weighted-score fusion

BM25 scores are unbounded and corpus-dependent, while vector distances depend on the embedding model. Fusing raw scores needs fragile calibration per model. RRF only uses ranks, so it is robust across providers. The `HYBRID_ALPHA` weight still lets the balance be tuned. The trade-off is that RRF throws away score magnitudes; the separate prioritization stage adds back the domain signals that matter (proximity, yield percentile, recency, feedback).

## 9. Limits of an LLM judge on a small model

A small judge model tends to be lenient, inconsistent between runs, and easily convinced by fluent text. Mitigations:

- The pass rule is computed in code from the scores (all ≥ 3 and mean ≥ 3.5); the model's own `passed` flag is ignored.
- A different or larger judge can be configured (`JUDGE_PROVIDER`, `JUDGE_MODEL`).
- The revision loop (cap of 2) is disabled in the current graph, so each query makes exactly three LLM calls and latency stays predictable. The judge verdict is shown to the user instead of triggering a rewrite.
- The judge runs *after* deterministic guardrails that already remove unverifiable numbers and unknown case IDs. The judge is a second opinion, not the only safeguard.
- In degraded mode, a rule-based validator checks citations, consistency between historical medians and model intervals, and completeness of risk/yield data.

## 10. Feedback loop as re-ranking only

Ratings and "helpful case" ticks become Laplace-smoothed boosts, bounded to ±0.1, for cases and crops. They nudge ranking without letting a few votes dominate. This is **not** model retraining: the yield and risk models only change through `POST /admin/retrain` on data. Case IDs are stable only for a given dataset version, so feedback should be cleared when the dataset changes.

## 11. Dataset caveats

- **Correlation, not causation.** The models learn associations in observational data. "Most sensitive factor" means the model's prediction moves most with that input, not that changing it will change the field's yield.
- **Spatial granularity.** Soil N/P/K/pH comes from one value per state, and weather from one value per state-year. Soil effects are therefore confounded with location: the model cannot separate "Punjab" from "Punjab's soil". This is why soil features can rank highly in importance, and why changing N for a fixed state can produce odd what-if deltas. Responses label these as model associations.
- **Aggregation level.** Rows are state-season-year aggregates (areas up to millions of hectares), not farms. Advice is a regional prior, not a farm prescription.
- **Data quality.** 94 rows whose yield is more than 10× their crop's median (e.g. maize at 990 t/ha) were removed as recording errors. Coconut is recorded in nuts/ha and is labelled so in the UI and responses.
- **Coverage and bias.** 30 states and UTs are covered (e.g. Rajasthan is absent). Crops with few rows have wide intervals. Users are told when they mention a state the data does not cover.
- **Synthetic fallback.** If no real data is present, a synthetic dataset is generated for demos. It is labelled everywhere (startup log, UI banner, `is_synthetic_data`, model `trained_on`) and must not be used for real advice.

## 12. Other notable choices

- **Out-of-scope questions return 200** with `out_of_scope: true` instead of an error: the request was valid; the assistant declines it.
- **Startup never blocks on embeddings**: missing collections are built in a background job, and keyword search serves requests meanwhile.
- **Tests run without any LLM**: providers are patched with a fake chat model, and mutable state (feedback DB, eval outputs) goes to a temp directory.
