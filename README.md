# Cassandra AI — Thin Prototype

SIH26184, Team NeferX. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full spec and
[SCHEMA_CONTRACT.md](SCHEMA_CONTRACT.md) for the data contracts between pipeline stages.

## Setup
```bash
# Python env (uses Anaconda Python 3.11, not system Python)
C:/Users/royal/anaconda3/python.exe -m venv .venv
./.venv/Scripts/pip install -r requirements.txt

# Postgres/PostGIS (via Docker inside WSL Ubuntu — see docker/README.md)
# from a WSL terminal:
#   sudo service docker start
#   cd "/mnt/c/Users/royal/OneDrive/Documents/SIH 2026/prototype/docker" && docker compose up -d

cp .env.example .env
```

## Run the full demo
```bash
./scripts/run_demo.sh
```
This regenerates calibration data, simulated transactions (configs A/B/C),
features, trains the model, runs evaluation, and starts the API (port 8000)
and UI (port 5173). Takes a few minutes end to end. Postgres is not required
for the demo path above (the API reads parquet/model artifacts directly) —
it's provisioned for the originally-specified Postgres/PostGIS storage layer
but the thin-slice pipeline currently runs off local Parquet files.

## Results
See [results/results.md](results/results.md) for the full metrics table,
circularity protocol results, real-data benchmarks, and honest limitations.
See [results/REVIEW.md](results/REVIEW.md) for an independent audit
(leakage check, metric recomputation, overclaim check).
See [results/deck_edits.md](results/deck_edits.md) for exactly which deck
slides to update with which numbers/screenshots.

## Status
Full pipeline built and run end-to-end on simulated data. Headline finding:
the XGBoost model does not currently beat the frequency baseline (reported
honestly in results.md, not hidden). See PROJECT_SPEC.md §Agent assignments
for which module was built by which stage of the process.
