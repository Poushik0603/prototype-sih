"""Terminal table loading.

Prefers the real terminals output at data/simulated/fixture/terminals.parquet
(schema: terminal_id, lat, lon, h3_cell, type, bank, install_date — see
docs/DATA_SCHEMA.md). If that file does not exist yet, generates a small
synthetic placeholder terminals table over plausible India coordinates so the
simulator can be developed and run end-to-end without waiting.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_TERMINALS_PATH = REPO_ROOT / "data" / "simulated" / "fixture" / "terminals.parquet"
PLACEHOLDER_TERMINALS_PATH = REPO_ROOT / "simulator" / "lib" / "_placeholder_terminals.parquet"

# Rough bounding box around central Bengaluru, used only for the placeholder.
# (Matches the real OSM extract's city — see data/calibration/README.md. The real
# terminals.parquet simply supersedes this placeholder once it exists on disk.)
_LAT_RANGE = (12.90, 13.05)
_LON_RANGE = (77.55, 77.70)
_BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "BOB", "Canara", "Kotak"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_placeholder_terminals(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    lats = np_rng.uniform(_LAT_RANGE[0], _LAT_RANGE[1], n)
    lons = np_rng.uniform(_LON_RANGE[0], _LON_RANGE[1], n)
    rows = []
    base_install = date(2015, 1, 1)
    for i in range(n):
        lat, lon = float(lats[i]), float(lons[i])
        cell = h3.latlng_to_cell(lat, lon, 9)
        rows.append(
            {
                "terminal_id": f"T{i+1:05d}",
                "lat": lat,
                "lon": lon,
                "h3_cell": cell,
                "type": rng.choices(["ATM", "POS"], weights=[0.55, 0.45])[0],
                "bank": rng.choice(_BANKS),
                "install_date": base_install + timedelta(days=rng.randint(0, 3650)),
            }
        )
    df = pd.DataFrame(rows)
    df["install_date"] = pd.to_datetime(df["install_date"])
    return df


def load_terminals() -> tuple[pd.DataFrame, str, str]:
    """Returns (terminals_df, source_label, calibration_hash).

    source_label is "real" or "placeholder". calibration_hash is the sha256 of
    whichever terminals file was used (hashing the DataFrame's parquet bytes
    for the placeholder case, since it's generated in-memory then cached).
    """
    if REAL_TERMINALS_PATH.exists():
        df = pd.read_parquet(REAL_TERMINALS_PATH)
        return df, "real", _sha256_file(REAL_TERMINALS_PATH)

    # Cache the placeholder to disk so repeated runs hash identically.
    if not PLACEHOLDER_TERMINALS_PATH.exists():
        df = generate_placeholder_terminals()
        PLACEHOLDER_TERMINALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PLACEHOLDER_TERMINALS_PATH, index=False)
    df = pd.read_parquet(PLACEHOLDER_TERMINALS_PATH)
    return df, "placeholder", _sha256_file(PLACEHOLDER_TERMINALS_PATH)
