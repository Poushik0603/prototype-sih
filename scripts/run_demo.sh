#!/usr/bin/env bash
# One-command demo: regenerate calibration/simulated data, train, evaluate, and
# start the API + UI. Assumes the venv (.venv) is already created and
# requirements.txt installed (see README.md Setup). Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f ./.venv/Scripts/python.exe ]; then
  PY=./.venv/Scripts/python.exe   # Windows venv layout
else
  PY=./.venv/bin/python            # macOS/Linux venv layout
fi

echo "[1/6] Fetching calibration data + terminals (skips if already cached)..."
$PY scripts/fetch_calibration.py
$PY scripts/build_terminals.py

echo "[2/6] Generating simulated data for configs A, B, C..."
$PY simulator/generate.py --all

echo "[3/6] Building feature tables..."
$PY model/features.py --config A
$PY model/features.py --config B
$PY model/features.py --config C

echo "[4/6] Training model and running baselines..."
$PY model/baselines.py
$PY model/train.py --config A

echo "[5/6] Evaluating (circularity protocol)..."
$PY model/evaluate.py
$PY model/circularity.py

echo "[6/6] Starting API (background) + UI dev server..."
USE_MOCK=false nohup $PY -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000 \
  > /tmp/cassandra_api.log 2>&1 &
echo "API started (pid $!), logs at /tmp/cassandra_api.log"

(cd ui && npm install && nohup npm run dev > /tmp/cassandra_ui.log 2>&1 &)
echo "UI starting at http://localhost:5173 (logs at /tmp/cassandra_ui.log)"

echo ""
echo "Done. Results: results/metrics_*.json, PROTOTYPE_REPORT.md"
echo "API:  http://127.0.0.1:8000/api/health"
echo "UI:   http://localhost:5173"
