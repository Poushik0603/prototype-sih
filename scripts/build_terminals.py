"""
Cassandra AI -- builds the `terminals` table from the frozen OSM raw data, per
the exact schema in docs/DATA_SCHEMA.md:

    terminal_id, lat, lon, h3_cell (res 9), type, bank, install_date

OSM does not carry bank name or install date for ATM nodes, so those two
columns are synthesized (sampled from a fixed list of real Indian bank names /
random dates in the last 1-5 years). This is a synthetic AUGMENTATION of real
geographic coordinates, not fabricated core data -- documented in
data/calibration/README.md. Lat/lon/h3_cell come directly from real OSM data.

Output:
    data/simulated/fixture/terminals.parquet   (also serves as the "fixture"
        referenced in docs/DATA_SCHEMA.md for early Model/API integration --
        this IS the full terminal universe, just placed at the fixture path
        too, since the city-scale terminal count is already small enough to
        act as its own fixture)
    data/raw/terminals_full.parquet            (same content, raw copy)

Usage:
    .venv\\Scripts\\python.exe scripts\\build_terminals.py
"""

import json
import random
from pathlib import Path

import h3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_OSM_FILE = REPO_ROOT / "data" / "raw" / "osm_atms_bengaluru.json"
FIXTURE_OUT = REPO_ROOT / "data" / "simulated" / "fixture" / "terminals.parquet"
RAW_OUT = REPO_ROOT / "data" / "raw" / "terminals_full.parquet"

H3_RESOLUTION = 9

# Fixed list of real Indian bank names to sample from for the synthetic
# `bank` column (OSM ATM nodes do not reliably carry operator tags).
BANKS = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Punjab National Bank",
    "Canara Bank",
]

SEED = 42


def main():
    random.seed(SEED)

    with open(RAW_OSM_FILE, "r", encoding="utf-8") as f:
        osm = json.load(f)

    elements = [e for e in osm["elements"] if e.get("type") == "node"
                and "lat" in e and "lon" in e]
    print(f"[build_terminals] {len(elements)} ATM nodes with coordinates")

    # install_date: random date in the past 1-5 years from a fixed "today"
    # anchor (script run date) for reproducibility across re-runs on the same
    # day; seeded RNG makes the sampling itself deterministic.
    today = pd.Timestamp.today().normalize()
    min_days_ago = 365 * 1   # 1 year
    max_days_ago = 365 * 5   # 5 years

    rows = []
    for i, el in enumerate(elements, start=1):
        lat = float(el["lat"])
        lon = float(el["lon"])
        h3_cell = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        bank = random.choice(BANKS)
        days_ago = random.randint(min_days_ago, max_days_ago)
        install_date = (today - pd.Timedelta(days=days_ago)).date()

        rows.append({
            "terminal_id": f"T{i:05d}",
            "lat": lat,
            "lon": lon,
            "h3_cell": h3_cell,
            "type": "ATM",
            "bank": bank,
            "install_date": install_date,
        })

    df = pd.DataFrame(rows)
    df["install_date"] = pd.to_datetime(df["install_date"])

    # dtype enforcement per schema contract
    df["terminal_id"] = df["terminal_id"].astype(str)
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df["h3_cell"] = df["h3_cell"].astype(str)
    df["type"] = df["type"].astype(str)
    df["bank"] = df["bank"].astype(str)

    FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(FIXTURE_OUT, index=False)
    df.to_parquet(RAW_OUT, index=False)

    print(f"[build_terminals] Wrote {len(df)} terminals to:")
    print(f"  {FIXTURE_OUT}")
    print(f"  {RAW_OUT}")
    print(df.head())
    print(df.dtypes)


if __name__ == "__main__":
    main()
