# API

FastAPI backend serving ranked-terminal risk scores to the UI.

## Running

```
# Terminal 1 — API (from repo root)
USE_MOCK=false .venv/Scripts/python.exe -m uvicorn api.app.main:app --port 8000

# Terminal 2 — UI
cd ui
npm install   # first time only
npm run dev   # serves on http://localhost:5173
```

Open http://localhost:5173. `USE_MOCK` controls the data source:
- `USE_MOCK=false` (requires `model/artifacts/xgb_config_A.json` and
  `data/processed/features_config_A.parquet` to exist — run the model pipeline first,
  see `model/README.md`) — serves real trained-model output.
- `USE_MOCK=true` or unset — serves a seeded mock generator
  (`api/app/mock_data.py`) matching the same response shape, useful for UI
  development without the model pipeline running. The fallback is
  exception-safe: if the real backend fails to load for any reason, the API
  logs why and serves mock data instead of crashing.

## Endpoints

- `GET /api/health` — also reports `using_mock: true/false`
- `GET /api/terminals/ranked?window_start=&limit=50`
- `GET /api/terminals/{terminal_id}`
- `POST /api/alerts/send` (mock SMS stub — logs only, never calls a real API)

## Performance note

The real-data backend scores only the single latest `window_start` before running
SHAP (not the entire feature-table history) — scoring the full ~290k-row history
through SHAP on every request made the endpoint take minutes; filtering to one
window (~1,150 rows) brings it to ~6ms warm / ~8.5s cold (model + explainer load).
See `api/app/data_source.py`.

## Latency measurements

`results/api_latency.json` (mock data, 50 calls to `/api/terminals/ranked`):
p50 = 9.5ms, p95 = 11.6ms, mean = 9.7ms. Real-model warm latency (6.5ms) and cold
latency (~8.5s, one-time per process start) were measured separately during
integration — see `results/results.md` for the full comparison.

## Screenshots

Captured via Playwright (`ui/screenshot.mjs`) against the live, real, trained
model — real rendered pixels, not mocked:
- `results/screenshots/ranked_atm_heatmap.png` — map view, Bengaluru basemap
  (OpenFreeMap liberty style, no API key), risk-colored/sized markers, legend,
  top-10 sidebar.
- `results/screenshots/officer_alert_view.png` — same view with the detail panel
  open on the top-ranked terminal: risk score, baseline score, SHAP bar chart,
  Send Alert button.

Note: `ui/src/main.tsx` does not use `React.StrictMode` — its dev-mode
double-mount was aborting MapLibre's in-flight style fetch (mount, unmount,
remount), leaving the basemap blank during screenshot capture.
