# Model

Feature engineering, ST-DBSCAN candidate clustering, XGBoost ranker + SHAP, baselines,
and the rolling-origin / circularity evaluation protocol.

## Performance fix: graph-distance feature

`compute_graph_distance_feature` in `features.py` originally rebuilt a fresh NetworkX
graph from a 7-day lookback slice of transactions AND ran a full BFS for **every
distinct window_start** (~250 windows for config A). Consecutive 2h windows' 7-day
lookback slices overlap by ~84 windows, so this did ~250x redundant graph
construction and BFS work — the cause of an earlier multi-hour hang.

Fix: sweep `window_start` values in time order and maintain **one cumulative graph**,
adding only the new edges for transactions in `[previous_cursor, this window_start)`
at each step, then run a single multi-source BFS (a virtual super-source wired to all
known-mule nodes) from the graph's current as-of state. This also removes an earlier
artificial 7-day lookback cap, giving a true as-of cumulative graph — stricter and
more leak-safe than before, not just faster.

Verified timing (wall-clock, full real data, not a slice):
- config A: 1,150 terminals, 194k txns, 252 windows → **36s** end-to-end feature build
- config B: 1,150 terminals, 196k txns → **56s**
- config C: 1,150 terminals, 132k txns → **22s**

## Bug fix: rolling-origin split was starving "test" to zero

`make_rolling_origin_splits` originally applied train/val fractions (55%/15%) against
the **full** time span before subtracting embargo time. On config A's 20-day span, two
7-day embargoes (14 days) left only 6 days for train+val+test, and the fractions
(computed against the full 20 days) consumed all of it before reaching "test" — zero
test rows, silently. Fixed by applying fractions to the time **remaining after both
embargoes are reserved**, guaranteeing a real test slice whenever remaining time is
positive. Config A now has train=57,500 / embargo=193,200 / val=14,950 / test=24,150
rows. Configs B (31-day span) and C (13-day span) still can't fit a meaningful internal
test slice in every case (C's remaining time clamps to 0) — handled explicitly in
`evaluate.py` (see below), not papered over by shrinking the embargo.

## Bug fix: precision@10 was computed globally, not per-window

The original `precision_at_k` ranked across the **entire flattened test table**
(hundreds of windows x ~1,150 terminals/window). With positives spread 1-5 per window
across dozens of windows, "are any of the 10 single highest-scored rows in the whole
table positive" is a near-impossible target unrelated to real usage — the officer-view
API (`predict.py`) ranks terminals **within one as-of window** at a time, matching
the response shape in `docs/DATA_SCHEMA.md` (there is no cross-window rank concept).
Added `precision_at_k_per_window` (grouped by `window_start`, averaged) and made it
the headline `beats_baseline` comparison; the old global metric is kept alongside,
clearly labeled, for transparency.

## Pipeline status

- `data/processed/features_config_{A,B,C}.parquet` built from real simulator output
  (1,150 real Bengaluru OSM terminals), as-of-only features, `split_tag` populated,
  7-day embargo enforced exactly.
- `stdbscan.py` sanity-run against real terminals+transactions: 1,119/1,150 terminals
  clustered into 9 spatio-temporal groups at the latest as-of timestamp (18s).
- `baselines.py`: past-hotspot-frequency (the baseline to beat) and a per-config
  rule-only baseline (mirrors each config's documented `terminal_choice` heuristic —
  nearest_to_last_hop for A, hotspot_avoidant for B, uniform/random for C), both run.
- `train.py`: XGBoost trained on config A's train split (57,500 rows), tuned on val
  only via a 3-point grid (max_depth/learning_rate), never touching test. Saved to
  `artifacts/xgb_config_A.json` + `_meta.json`. Final val AP=0.0029, AUC=0.581.
- `evaluate.py` + `circularity.py`: full circularity protocol run (train A, test
  A/B/C). For cross-config checks (B, C), the model was never trained/tuned on any
  row of that config, so the full labeled table is used as the test set (not just its
  internal "test" tag) — more statistically robust and avoids silently skipping
  configs whose short time span can't fit an internal test slice.

## Headline results (ALL ON SIMULATED DATA — see `results/metrics_circularity.json`
for the full realism-gap statement)

| Trained on A, tested on | n_test_rows | n_positive | model P@10/window | baseline P@10/window | rule-only P@10/window | ROC AUC (model) | beats_baseline |
|---|---|---|---|---|---|---|---|
| A (in-sample test) | 24,150 | 48 | 0.000 | 0.000 | 0.000 | 0.524 | **false** |
| B (cross-config)   | 430,100 | 443 | 0.000 | 0.001 | — | 0.391 | **false** |
| C (cross-config)   | 193,200 | 537 | 0.002 | 0.004 | 0.003 | 0.503 | **false** |

**The model does not beat the past-hotspot or rule-only baselines on any config.**
ROC AUC hovers at 0.39-0.58 — essentially chance level, occasionally worse than chance
on B (0.39). This is reported plainly, not hidden. Likely causes, honestly assessed:
extremely low positive rate (0.1-0.3%) with only 1-5 positive terminals per
1,150-terminal window gives very little training signal; the as-of feature set
(rolling txn counts, past-hotspot frequency, complaint density, graph distance) may be
too coarse at 2h granularity to separate mule cash-outs from benign traffic at this
scale; and the thin 20-day time range limits how much the model can learn before the
embargo/test cutoff. All three — not just model capacity — are plausible contributors;
no single cause is claimed without more investigation time than this slice allows.

## Circularity protocol completeness

- Step 1 (train A, test A/B/C): **complete**, all three configs present.
- Step 2 (rule-only baseline per config): **complete**, included inline per config.
- Step 3 (label-noise robustness): **partial by construction** — the simulator bakes
  differing noise rates into each config (A: 5%/2% FN/FP, B: 8%/2%, C: 10%/4%) rather
  than exposing a standalone re-runnable knob, so topology/timing shift and noise-rate
  shift are conflated in the B/C degradation numbers above; documented as a known
  limitation in `results/metrics_circularity.json`, not glossed over.
- Step 4 (real-data separation): every number here is simulated-data-only and never
  blended with the real-data benchmarks in `results/benchmark_notes.md`.
- Step 5 (realism-gap statement): written plainly in `results/metrics_circularity.json`.

## Interface stability

`predict.py`'s `CassandraModel.load("A").score(df)` signature is stable. It loads
`artifacts/xgb_config_A.json` and produces exactly the response shape documented in
`docs/DATA_SCHEMA.md` (`terminal_id`, `lat`, `lon`, `window_start`, `risk_score`,
`rank`, `top_shap_features`, `baseline_score`), consumed by `api/app/data_source.py`.

## Files produced

`data/processed/features_config_{A,B,C}.parquet`, `artifacts/xgb_config_A.json` +
`_meta.json`, `results/metrics_config_{A,B,C}.json`, `results/metrics_circularity.json`,
`results/metrics_baselines.json`.
