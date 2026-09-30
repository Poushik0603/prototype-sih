# Cassandra AI — Thin Prototype Spec (SIH26184, Team NeferX)

Source of truth for all agents working on this repo. Read this before writing code.
Do not re-debate decisions marked (LOCKED). Flag conflicts back instead of silently deviating.

## Goal
End-to-end pipeline that produces real measured numbers + real screenshots for the deck.
Thin slice that RUNS beats a large slice that doesn't. One command (`make demo`) runs everything.

## Pipeline (LOCKED)
seeded simulator -> feature build -> ST-DBSCAN candidate terminals -> XGBoost ranker + SHAP
-> FastAPI -> React + MapLibre map view.

- Unit: (ATM/POS terminal, 2-hour window)
- Label: 1 if a complaint-linked mule account withdraws at that terminal in that window
- Split: rolling-origin, 7-day embargo, no future features (as-of graph per complaint)
- Baseline to beat: past-hotspot frequency per terminal (rule-only baseline mirrors generator)
- Metrics: lift vs baseline, precision@10, lead time, API latency (p50/p95)

## Stack (LOCKED — environment-adjusted)
- Python: use Anaconda3 Python 3.12 at `C:/Users/royal/anaconda3/python.exe` (NOT the system
  Python 3.14 — too new, ML wheels unavailable). Create a project venv from it:
  `C:/Users/royal/anaconda3/python.exe -m venv .venv`
- Storage: Parquet by day -> PostgreSQL/PostGIS via Docker (docker-compose.yml at repo root).
  Tables: complaints, transactions, terminals, labels + feature table (terminal x 2h window).
- Queue: pgmq (Postgres extension) for the pilot.
- Graph: NetworkX. Linkage: Splink (skip if install friction > 30 min, note it and use a
  simpler deterministic linkage instead — flag the deviation).
- Orchestration: plain Python scripts + a Makefile-equivalent (`make demo` — use a `justfile`
  or `scripts/run_demo.ps1` + `scripts/run_demo.sh` since this is Windows; no Prefect for the
  thin slice unless time allows).
- Experiment tracking: MLflow (local file store, no server needed — `mlruns/` dir already created).
- Model: XGBoost + SHAP. No GNN.
- Spatial index: H3 cells.
- API: FastAPI (api/app/).
- UI: React + MapLibre GL JS (ui/), plain Vite app, no heavy framework.
- SMS: MSG91 — MOCK ONLY, do not call real API, just log/stub.

## Simulator rules (LOCKED)
- Config-driven via `simulator/rules.yaml`: mule hop count, delay between hops, cash-out
  pattern, terminal choice logic.
- Every run logs seed, calibration hash, rules version (to MLflow + a local run manifest JSON,
  since MLflow server may not be running — log params/metrics via local file store).
- Calibrate background/benign traffic and terminal geography from PUBLIC data only:
  - NCRP state-level complaint totals (find a public summary table; if unavailable/unclear
    licence, use a documented plausible placeholder and say so clearly in results.md)
  - RBI ATM count statistics (public RBI publications)
  - OSM ATM coordinates for a chosen city/region (Overpass API or a pre-downloaded extract) —
    index with H3
  - Store raw calibration inputs unchanged under `data/calibration/` with SHA256 checksums
    recorded in a manifest.
- LLM is NEVER used to generate numeric/tabular data. Only optional scenario/complaint free
  text may be LLM-authored, and must be clearly labeled synthetic text.
- Produce 3 simulator configurations with different mule topology + timing for the
  circularity protocol (see below).
- Include label noise injection and delayed-label simulation (configurable rate).

## Circularity protocol (MANDATORY)
1. Train on simulator config A; test on configs B and C (different mule topology/timing).
2. Include a rule-only baseline that mirrors the generator's own heuristics.
3. Add label noise and delayed labels; report metric degradation.
4. Report real-data benchmark results SEPARATELY from simulator results — never blend them.
5. State the realism gap plainly in results.md. No overclaiming.

## Real-data benchmarks (best-effort, skip cleanly if blocked)
- Graph/linkage module sanity-checked against AMLworld or Elliptic (whichever has clear,
  obtainable licence/download in the time available).
- ST-DBSCAN sanity-checked against Chicago Crimes (public, clear licence, Socrata API).
- If a dataset's licence/download is unclear, DO NOT use it — log the skip and why in
  results.md instead of guessing.

## Reporting rules (MANDATORY)
- Only report numbers produced by actual runs, saved to disk (JSON/CSV under `results/`).
- Label every number "on simulated data" unless it came from a real dataset.
- If the model does not beat the baseline, say so plainly and show the numbers.
- Never tune hyperparameters on the test set (use validation fold only).
- Set random seeds everywhere; reproducible via one command.

## Directory layout (already scaffolded — fill in, don't restructure)
```
prototype/
  docs/                     # existing PDFs (deck + problem statement) — read-only
  data/
    raw/                    # frozen public calibration downloads
    calibration/            # calibration tables + checksums manifest
    simulated/              # simulator output (parquet by day)
    processed/              # feature tables
  simulator/                # rules.yaml, generator code
  model/                    # features, ST-DBSCAN, XGBoost, SHAP, evaluation
  api/app/                  # FastAPI app
  ui/                       # React + MapLibre + Vite
  notebooks/                # exploratory only, not load-bearing
  results/                  # results.md + metrics JSON/CSV + screenshots
  scripts/                  # run_demo.ps1 / .sh, data download scripts
  docker/                   # docker-compose, Dockerfiles
  mlruns/                   # MLflow local store
```

## Agent assignments (parallel)
1. **Data agent**: raw data freeze, calibration tables, H3 indexing, terminals table, checksums
   manifest. Owns `data/`, `scripts/fetch_*`.
2. **Simulator agent**: `simulator/rules.yaml`, seeded generator, 3 configs, label noise/delay.
   Owns `simulator/`. Depends on Data agent's terminals table (coordinate via stub/mock first,
   integrate after).
3. **Model agent**: features, ST-DBSCAN, XGBoost, SHAP, baseline + rule-only baseline,
   rolling-origin evaluation, circularity protocol runs. Owns `model/`. Depends on simulator
   output schema (agree on schema up front, build against a small fixture first).
4. **API/UI agent**: FastAPI endpoints + React/MapLibre map with ranked ATMs, officer alert
   view. Owns `api/`, `ui/`. Can build against a fixture/mock feature table first, wire to
   real model output after.
5. **Benchmark agent**: real-data module checks (AMLworld/Elliptic, Chicago Crimes). Owns
   `model/benchmarks/` or similar, writes to `results/`.
6. **Reviewer agent**: independent, reviews AFTER others finish — checks leakage, recomputes
   every reported metric from saved outputs, flags overclaims. Does not write pipeline code.

Schema contracts between agents (AGREE, then don't break silently):
- `terminals` table: terminal_id, lat, lon, h3_cell, type (ATM/POS), bank, install_date
- `transactions` table: txn_id, terminal_id, account_id, amount, timestamp, is_mule_hop
- `complaints` table: complaint_id, account_id, filed_at, state
- `labels`/feature table: terminal_id, window_start (2h bucket), label, features..., split_tag

## Deliverables
1. Working repo + README with run instructions and one-command demo.
2. `results/results.md`: metrics table (model vs baseline vs rule-only), circularity test
   results, real-data benchmark tables, limitations, honest realism-gap statement.
3. Screenshots (PNG, presentation quality) in `results/screenshots/`: ranked-ATM heatmap,
   officer alert view.
4. `results/deck_edits.md`: which slides to update with which numbers/screenshots, labeled
   "prototype on simulated data".
