"""
Cassandra AI -- calibration data reproducibility script.

Fetches real ATM terminal coordinates from OpenStreetMap (Overpass API) for the
chosen city (Bengaluru) and freezes the raw response under data/raw/.

Idempotent: if the raw file already exists, the fetch is skipped (Overpass has
rate limits and we don't want to hammer it on every `make demo` run).

Usage:
    .venv\\Scripts\\python.exe scripts\\fetch_calibration.py

This script ONLY fetches/freezes raw OSM data. Calibration numbers (NCRP, RBI)
were hand-transcribed from cited public sources -- see data/calibration/README.md
for exact URLs and access dates. No numeric data in this repo was invented by
an LLM; only real fetched/transcribed values are used.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CITY_NAME = "Bengaluru"
RAW_FILE = RAW_DIR / f"osm_atms_{CITY_NAME.lower()}.json"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box for Bengaluru urban area (south, west, north, east).
# Chosen instead of an admin-boundary `area` query because it is simpler,
# faster for Overpass to evaluate, and avoids ambiguity/failures when OSM's
# administrative boundary relation for the city is missing or differently
# tagged. Box covers greater Bengaluru (BBMP limits plus some margin).
BBOX = (12.83, 77.45, 13.14, 77.78)  # south, west, north, east

OVERPASS_QUERY = f"""
[out:json][timeout:180];
(
  node["amenity"="atm"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out body;
"""

HEADERS = {
    "User-Agent": "CassandraAI-SIH2026-Hackathon-Prototype/1.0 (calibration fetch script)",
    "Accept": "*/*",
}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_overpass(query: str, max_retries: int = 5) -> dict:
    """POST an Overpass QL query with exponential backoff retry."""
    backoff = 5
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[fetch_calibration] Overpass attempt {attempt}/{max_retries} ...")
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=HEADERS,
                timeout=180,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(
                    f"[fetch_calibration] HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            print(f"[fetch_calibration] request error: {exc}")
            last_exc = exc

        if attempt < max_retries:
            print(f"[fetch_calibration] retrying in {backoff}s ...")
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(
        f"Overpass API unreachable after {max_retries} attempts"
    ) from last_exc


def main():
    if RAW_FILE.exists():
        print(f"[fetch_calibration] {RAW_FILE} already exists -- skipping fetch "
              f"(idempotent; delete the file to force a re-fetch).")
        print(f"[fetch_calibration] SHA256: {sha256_of_file(RAW_FILE)}")
        return 0

    print(f"[fetch_calibration] Fetching OSM ATM nodes for {CITY_NAME} "
          f"(bbox={BBOX}) from Overpass API ...")
    try:
        data = fetch_overpass(OVERPASS_QUERY)
    except RuntimeError as exc:
        print(f"[fetch_calibration] FAILED: {exc}")
        print("[fetch_calibration] Overpass API appears blocked/unreachable. "
              "Not fabricating coordinates -- aborting. See README for manual "
              "fallback instructions (e.g. Geofabrik regional extract).")
        return 1

    n_elements = len(data.get("elements", []))
    print(f"[fetch_calibration] Got {n_elements} elements.")
    if n_elements == 0:
        print("[fetch_calibration] WARNING: 0 elements returned -- not writing "
              "an empty raw file. Check the query/bbox.")
        return 1

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[fetch_calibration] Wrote {RAW_FILE}")
    print(f"[fetch_calibration] SHA256: {sha256_of_file(RAW_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
