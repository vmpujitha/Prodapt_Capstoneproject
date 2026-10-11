# Assumptions

Decisions taken where the brief or the data was ambiguous. Each item says where it is implemented.

## Data

1. **Dataset shape.** The Kaggle dataset ships as three files rather than one flat CSV: crop yield (crop, year, season, state, area, production, fertilizer, pesticide, yield), state soil (N, P, K, pH) and state-year weather (temperature, rainfall, humidity). They are joined on `state` and `state + year` (`scripts/download_data.py::merge_kaggle_files`). All 19,689 rows found matching soil and weather records.
2. **Fertilizer and pesticide units.** The raw values are totals for the whole cultivated area in kg (e.g. 7,024,878 for 73,814 ha). They are divided by `area` to give kg/ha (resulting ranges: 95–194 kg/ha fertilizer, 0.09–0.38 kg/ha pesticide).
3. **Yield units.** Yield is t/ha for every crop except **Coconut**, which the source records in nuts/ha (median ~8,400). The unit is carried through responses (`unit`, `yield_unit`) and the UI.
4. **Implausible yields.** Rows with yield more than 10× their crop's median (94 rows, 0.5%) are treated as recording errors and dropped. Zero yields (112 rows, crop failures) are kept.
5. **Clipping bounds** follow the physical ranges in the brief for soil and weather. Area is allowed up to 1e8 ha, because rows are state-level aggregates (max ≈ 5e7 ha).
6. **Imputation**: numeric nulls are filled with the crop-wise median, then the overall median. Rows missing crop, state or yield are dropped. (The real data has no nulls.)
7. **Categorical normalisation**: strip, collapse inner whitespace, title-case (`"Kharif     "` → `Kharif`, `"Other  Rabi pulses"` → `Other Rabi Pulses`).
8. **Seasons** are whatever the data contains (Autumn, Kharif, Rabi, Summer, Whole Year, Winter). "Zaid" does not occur in the real data.
9. **Tertile thresholds** for N/P/K/temperature/rainfall/humidity are computed over the whole dataset (not per crop) and persisted to `data/processed/thresholds.json`. pH classes use the fixed agronomic cut-offs from the brief (6.0 / 7.5).
10. **Case IDs** (`CASE-000001` …) are stable for a given cleaned dataset. Re-processing a different dataset re-assigns them, so feedback stored against old IDs no longer applies.
11. **Synthetic data** is generated only when no real data can be found; the files in `data_set/` take precedence.

## Retrieval and ranking

12. **Default ranking weights**: retrieval 0.35, proximity 0.25, yield percentile 0.20, recency 0.10, feedback 0.10 (configurable in `.env`).
13. **Proximity scales** (the denominators of the normalised distance) are fixed agronomic ranges: N 250, P 120, K 120, pH 6, temperature 35 °C, rainfall 2000 mm, humidity 80%.
14. **Recency** is linear over the dataset's year range (earliest year 0.1, latest 1.0).
15. **Filter relaxation**: classifier-derived filters are dropped step by step when fewer than 3 cases match. Explicit filters from the request are never dropped.
16. **Crop diversity**: without a crop filter, at most ⌈top_k/3⌉ (min 2) cases per crop are returned.
17. **Alternative-crop questions** exclude the crop being replaced from the recommendations, and do not filter retrieval by it.
18. **Multi-state yield-gap questions** (e.g. "Punjab vs Bihar") retrieve a balanced number of cases per named state (up to 3 states).
19. **BM25 tokenization** drops stopwords and bare numbers (4-digit years are kept). Numeric similarity is handled by condition proximity.

## Models

20. **Train/test split**: time-aware. The latest years covering ~20% of rows form the test set (2016–2020 on the real data). A random split is used only if there are fewer than 5 distinct years.
21. **Prediction interval**: 10th–90th percentile of held-out log residuals, per crop when at least 15 test residuals exist, otherwise global.
22. **Risk target**: `yield < 25th percentile of that crop`. Risk level: low if p < 0.25 with no envelope violations; medium if p < 0.5 with at most one moderate violation; otherwise high.
23. **Envelope checks** in `/query` only consider conditions the user actually stated, so defaults never trigger warnings. The `/predict/risk` endpoint checks all supplied fields.
24. **Missing inputs** for predictions come first from the retrieved similar cases of that crop, then from the crop's historical medians and modes. There are no fixed defaults such as "Punjab" or 100 kg/ha.
25. **XGBoost** is preferred but optional. On macOS it needs `libomp`; without it, scikit-learn GradientBoosting is used and reported as such.
26. **SHAP** was not added: it is not in the fixed stack, and permutation importance meets the requirement without another dependency.

## LLM, guardrails and evaluation

27. **Judge model**: blank `JUDGE_PROVIDER` / `JUDGE_MODEL` means the chat model selected for the request. A configured judge provider takes precedence.
28. **Judge pass rule** is recomputed in code (all scores ≥ 3 and mean ≥ 3.5), regardless of the `passed` value the model returns.
29. **Output number guardrail**: numbers are verified with a tolerance of max(0.05, 2%). Integers ≤ 10 (list numbering, counts) are exempt. Probabilities and scores in [0, 1] may also be quoted as percentages.
30. **Off-topic questions** return HTTP 200 with `out_of_scope: true` (no retrieval, no LLM answer). Injection attempts return 400; out-of-range numbers return 422.
31. **Rate limit**: 60 requests/minute per IP on `/query` (in-memory, single process).
32. **Golden set expectations** use common crop names. They are resolved to dataset names at evaluation time ("Mustard" → `Rapeseed &Mustard`, "Chickpea" → `Gram`, "Millets" → Bajra / Jowar / Ragi / Small Millets). Two golden queries mention Rajasthan, which the dataset does not cover; they are kept to test behaviour for uncovered states and are annotated as such.
33. **DeepEval telemetry** is disabled by default (`DEEPEVAL_TELEMETRY_OPT_OUT=YES`) to keep evaluation local.
34. **CORS** allows only the origins in `CORS_ORIGINS` (Vite dev server and the Docker frontend by default).

## Environment

35. **Docker**: the backend image uses Python 3.12. Ollama is expected on the host (`host.docker.internal`), or in the optional `ollama` compose profile. Extra root CAs for TLS-intercepting proxies can be dropped into `docker/certs/`.
36. **Paths** in `.env` may be relative; they are resolved against the project root, so commands work from any working directory.
