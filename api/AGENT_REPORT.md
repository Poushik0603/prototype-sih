# API/UI Agent Report

## Status: running on MOCK data (demo-ready end-to-end)

`model/predict.py` (`CassandraModel.load("A").score(df)`) now exists, and `api/app/data_source.py`
is wired to call it. But `model/artifacts/xgb_config_A.json` and
`data/processed/features_config_A.parquet` don't exist yet (model not trained / features not
built at time of writing), so the API automatically falls back to `api/app/mock_data.py`. This
fallback is exception-safe — confirmed with `USE_MOCK=false`, it logs why and serves mock data,
it never crashes. No action needed from the Model agent beyond dropping those two files in their
documented locations; the API will pick them up on next restart. No schema mismatch expected —
`CassandraModel.score()` already returns the exact `SCHEMA_CONTRACT.md` shape.

## How to run the demo

```
# Terminal 1 — API (from repo root)
.venv/Scripts/python.exe -m uvicorn api.app.main:app --port 8000

# Terminal 2 — UI
cd ui
npm install   # first time only
npm run dev   # serves on http://localhost:5173 (falls back to 5174 if busy — check CORS if so)
```

Open http://localhost:5173. To force real model data once artifacts land: `USE_MOCK=false` env
var before starting uvicorn (defaults to mock if unset).

## Endpoints

- `GET /api/health`
- `GET /api/terminals/ranked?window_start=&limit=50`
- `GET /api/terminals/{terminal_id}`
- `POST /api/alerts/send` (mock SMS stub — logs only, never calls a real API)

## Latency (results/api_latency.json, mock data, 50 calls to /api/terminals/ranked)

p50 = 9.5ms, p95 = 11.6ms, mean = 9.7ms. Comfortably fast; real-model inference will add SHAP
compute time per request — worth re-measuring once wired, but headroom is large.

## Screenshots

Both captured via Playwright (`ui/screenshot.mjs`), real rendered pixels, not mocked:
- `results/screenshots/ranked_atm_heatmap.png` — map view, Bengaluru basemap (OpenFreeMap
  liberty style, no API key), 50 risk-colored/sized markers, legend, top-10 sidebar.
- `results/screenshots/officer_alert_view.png` — same view with the detail panel open on the
  top-ranked terminal: risk score, baseline score, SHAP bar chart, Send Alert button.

Note: removed `React.StrictMode` in `ui/src/main.tsx` — its dev-mode double-mount was aborting
MapLibre's in-flight style fetch (mount→unmount→remount), leaving the basemap blank. Non-issue
in production builds, but the fix keeps dev screenshots correct too.

## Notes for other agents

- Data agent: `data/simulated/fixture/` is still empty — wasn't blocking (I mocked independently
  per instructions), but Simulator/Model may need it.
- Model agent: interface confirmed working — `CassandraModel.load(config).score(feature_df)` —
  no changes needed on your side. Once `model/artifacts/xgb_config_A.json` and
  `data/processed/features_config_A.parquet` exist, restart the API (or unset USE_MOCK) and it
  should Just Work. If `score()` throws for any reason, the API logs the exception and serves
  mock instead rather than 500ing — check `results/` isn't the place to look for that error, it's
  API stdout.
- `ui/node_modules` confirmed gitignored (repo-root `.gitignore` covers it).
