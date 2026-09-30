"""
Fetch a sample of Chicago Crimes data from the Socrata Open Data API (public, no auth
required for reasonable row limits). Source: City of Chicago Data Portal, dataset
"Crimes - 2001 to present" (ijzp-q8t2).

License: Public Domain / Open Data (Chicago Data Portal Terms of Use — public government
data, free to use). https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

This is REAL DATA (not simulated). Used only as a sanity check for the ST-DBSCAN
spatio-temporal clustering module, per PROJECT_SPEC.md "Real-data benchmarks" section.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOCRATA_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
LOOKBACK_DAYS = 120
LIMIT = 15000


def fetch():
    end = datetime(2026, 9, 29, 23, 59, 59, tzinfo=timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    where_clause = (
        f"date >= '{start.strftime('%Y-%m-%dT00:00:00')}' "
        f"AND date <= '{end.strftime('%Y-%m-%dT23:59:59')}' "
        f"AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )
    params = {
        "$where": where_clause,
        "$limit": LIMIT,
        "$order": "date DESC",
        "$select": (
            "id,case_number,date,primary_type,description,location_description,"
            "latitude,longitude,arrest,community_area,district,beat"
        ),
    }
    print(f"Fetching Chicago Crimes: {start.date()} to {end.date()}, limit={LIMIT}")
    resp = requests.get(SOCRATA_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"Fetched {len(data)} rows, {len(resp.content)} bytes")

    out_path = RAW_DIR / "chicago_crimes_sample_2026.json"
    out_path.write_bytes(resp.content)

    sha256 = hashlib.sha256(resp.content).hexdigest()
    manifest = {
        "dataset": "Chicago Crimes (2001 to present) — sample",
        "source_url": SOCRATA_URL,
        "query_params": params,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": (
            "City of Chicago Data Portal — Open Data, public government dataset. "
            "https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2 "
            "(Socrata Open Data Terms of Use, no auth required, free reuse)."
        ),
        "row_count": len(data),
        "file": str(out_path.relative_to(RAW_DIR.parents[1])),
        "sha256": sha256,
        "is_real_data": True,
    }
    manifest_path = RAW_DIR / "chicago_crimes_sample_2026.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved raw data to {out_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"SHA256: {sha256}")
    return out_path, manifest


if __name__ == "__main__":
    fetch()
