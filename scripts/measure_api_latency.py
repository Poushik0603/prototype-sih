"""
Measures p50/p95 latency of GET /api/terminals/ranked across N calls.
Saves results to results/api_latency.json.

Run with the API already started, e.g.:
  .venv/Scripts/python.exe -m uvicorn api.app.main:app --port 8000
  .venv/Scripts/python.exe scripts/measure_api_latency.py
"""
import json
import statistics
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000/api/terminals/ranked"
N_CALLS = 50
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "api_latency.json"


def main():
    latencies_ms = []
    for i in range(N_CALLS):
        start = time.perf_counter()
        resp = requests.get(API_URL, params={"limit": 50})
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95_idx = min(len(latencies_ms) - 1, int(round(0.95 * (len(latencies_ms) - 1))))
    p95 = latencies_ms[p95_idx]
    mean = statistics.mean(latencies_ms)

    result = {
        "endpoint": "GET /api/terminals/ranked",
        "n_calls": N_CALLS,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(mean, 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "all_latencies_ms": [round(x, 3) for x in latencies_ms],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "all_latencies_ms"}, indent=2))
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
