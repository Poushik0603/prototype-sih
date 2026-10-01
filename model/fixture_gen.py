"""Generate a small synthetic dev fixture matching SCHEMA_CONTRACT.md exactly.

This is NOT the real simulator output. It exists so the Model agent can build
and test the feature pipeline, ST-DBSCAN, and XGBoost training code end-to-end
before data/simulated/config_A (etc.) exist. Saved under
data/processed/dev_fixture/ (kept separate from data/simulated/ to avoid
colliding with the Simulator agent's own output directory).

Produces three "mini configs" (fx_A, fx_B, fx_C) that loosely mirror the real
simulator's three terminal_choice heuristics (nearest_to_last_hop /
hotspot_avoidant / random_in_city) so the circularity-protocol code path can
also be exercised on the fixture before real config_B/C land.

Random seeds are fixed. This is clearly a toy: few dozen terminals, a few
thousand transactions, a handful of complaint-linked mule chains.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, date
from pathlib import Path

import h3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "processed" / "dev_fixture"

# Same bounding box convention used by simulator/lib/terminals.py's placeholder
# (central Bengaluru) so fixture terminals look like plausible ATM/POS geography.
LAT_RANGE = (12.90, 13.05)
LON_RANGE = (77.55, 77.70)
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "BOB", "Canara", "Kotak"]
H3_RES = 9

N_TERMINALS = 60
N_BENIGN_ACCOUNTS = 400
DAYS = 14  # keep small: a couple weeks is enough windows for train/val/test + embargo


def _gen_terminals(seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    lats = np_rng.uniform(*LAT_RANGE, N_TERMINALS)
    lons = np_rng.uniform(*LON_RANGE, N_TERMINALS)
    rows = []
    base_install = date(2015, 1, 1)
    for i in range(N_TERMINALS):
        lat, lon = float(lats[i]), float(lons[i])
        rows.append(
            {
                "terminal_id": f"T{i+1:05d}",
                "lat": lat,
                "lon": lon,
                "h3_cell": h3.latlng_to_cell(lat, lon, H3_RES),
                "type": rng.choices(["ATM", "POS"], weights=[0.55, 0.45])[0],
                "bank": rng.choice(BANKS),
                "install_date": base_install + timedelta(days=rng.randint(0, 3650)),
            }
        )
    df = pd.DataFrame(rows)
    df["install_date"] = pd.to_datetime(df["install_date"])
    return df


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _pick_terminal(terminals: pd.DataFrame, rng: np.random.Generator, mode: str,
                    last_lat: float | None, last_lon: float | None,
                    hotspot_counts: dict[str, int]) -> pd.Series:
    if mode == "nearest_to_last_hop" and last_lat is not None:
        d = _haversine(last_lat, last_lon, terminals["lat"].values, terminals["lon"].values)
        idx = np.argsort(d)[: max(3, len(terminals) // 10)]
        choice = rng.choice(idx)
        return terminals.iloc[choice]
    if mode == "hotspot_avoidant":
        counts = np.array([hotspot_counts.get(tid, 0) for tid in terminals["terminal_id"]])
        weights = 1.0 / (1.0 + counts)
        weights = weights / weights.sum()
        choice = rng.choice(len(terminals), p=weights)
        return terminals.iloc[choice]
    # random_in_city / fallback
    choice = rng.integers(0, len(terminals))
    return terminals.iloc[choice]


def _gen_config(terminals: pd.DataFrame, config_name: str, terminal_choice: str,
                 n_complaints: int, fan_out_range: tuple[int, int],
                 hop_lambda: float, start_date: datetime, days: int,
                 seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    accounts_rows = []
    for i in range(N_BENIGN_ACCOUNTS):
        accounts_rows.append(
            {
                "account_id": f"ACC{config_name}{i:06d}",
                "is_mule": False,
                "opened_date": date(2020, 1, 1) + timedelta(days=pyrng.randint(0, 2000)),
            }
        )

    txn_rows = []
    complaint_rows = []
    txn_counter = 0
    complaint_counter = 0
    hotspot_counts: dict[str, int] = {}

    end_date = start_date + timedelta(days=days)

    # --- benign background traffic: poisson(~6/day) per terminal, spread over the window ---
    for _, term in terminals.iterrows():
        n_benign_txn = rng.poisson(6 * days)
        for _ in range(n_benign_txn):
            ts = start_date + timedelta(seconds=int(rng.uniform(0, days * 86400)))
            acc = accounts_rows[pyrng.randrange(N_BENIGN_ACCOUNTS)]["account_id"]
            amt = float(rng.lognormal(6.0, 1.0))
            txn_rows.append(
                {
                    "txn_id": f"TXN{config_name}{txn_counter:08d}",
                    "terminal_id": term["terminal_id"],
                    "account_id": acc,
                    "amount": round(amt, 2),
                    "timestamp": ts,
                    "is_mule_hop": False,
                }
            )
            txn_counter += 1
            hotspot_counts[term["terminal_id"]] = hotspot_counts.get(term["terminal_id"], 0) + 1

    # --- mule rings: each complaint spawns a fan-out of mule accounts hopping + cashing out ---
    for c in range(n_complaints):
        complaint_id = f"CMP{config_name}{complaint_counter:05d}"
        victim_account = f"VIC{config_name}{complaint_counter:05d}"
        filed_offset_days = pyrng.uniform(1, days - 1)
        ring_start = start_date + timedelta(days=pyrng.uniform(0, days - 2))

        fan_out = pyrng.randint(*fan_out_range)
        for m in range(fan_out):
            mule_account = f"MUL{config_name}{complaint_counter:05d}_{m}"
            hop_count = max(1, rng.poisson(hop_lambda))
            last_lat, last_lon = None, None
            cur_time = ring_start + timedelta(hours=pyrng.uniform(0, 3))
            last_terminal = None
            for h in range(hop_count):
                term = _pick_terminal(terminals, rng, terminal_choice, last_lat, last_lon, hotspot_counts)
                amt = float(rng.lognormal(7.5, 0.6))
                txn_rows.append(
                    {
                        "txn_id": f"TXN{config_name}{txn_counter:08d}",
                        "terminal_id": term["terminal_id"],
                        "account_id": mule_account,
                        "amount": round(amt, 2),
                        "timestamp": cur_time,
                        "is_mule_hop": True,
                    }
                )
                txn_counter += 1
                hotspot_counts[term["terminal_id"]] = hotspot_counts.get(term["terminal_id"], 0) + 1
                last_lat, last_lon = term["lat"], term["lon"]
                last_terminal = term["terminal_id"]
                cur_time = cur_time + timedelta(hours=float(rng.lognormal(1.5, 0.8)))
                if cur_time >= end_date:
                    break

            accounts_rows.append(
                {"account_id": mule_account, "is_mule": True,
                 "opened_date": date(2024, 1, 1) + timedelta(days=pyrng.randint(0, 600))}
            )

        # label noise: false_negative_rate ~ complaint sometimes not filed / filed but late
        false_negative = pyrng.random() < 0.05
        label_delay = timedelta(days=pyrng.uniform(0, 10))
        if not false_negative:
            complaint_rows.append(
                {
                    "complaint_id": complaint_id,
                    "account_id": victim_account,
                    "filed_at": ring_start + timedelta(days=filed_offset_days) + label_delay,
                    "state": "Karnataka",
                }
            )
        complaint_counter += 1

    accounts_df = pd.DataFrame(accounts_rows)
    accounts_df["opened_date"] = pd.to_datetime(accounts_df["opened_date"])
    txn_df = pd.DataFrame(txn_rows)
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    complaints_df = pd.DataFrame(complaint_rows)
    if len(complaints_df):
        complaints_df["filed_at"] = pd.to_datetime(complaints_df["filed_at"])

    return {"accounts": accounts_df, "transactions": txn_df, "complaints": complaints_df}


def build_fixture():
    terminals = _gen_terminals(seed=7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    terminals.to_parquet(OUT_DIR / "terminals.parquet", index=False)

    specs = {
        "fx_A": dict(terminal_choice="nearest_to_last_hop", n_complaints=18,
                     fan_out_range=(2, 5), hop_lambda=2,
                     start_date=datetime(2026, 1, 5), days=DAYS, seed=101),
        "fx_B": dict(terminal_choice="hotspot_avoidant", n_complaints=14,
                     fan_out_range=(3, 7), hop_lambda=4,
                     start_date=datetime(2026, 1, 8), days=DAYS, seed=102),
        "fx_C": dict(terminal_choice="random_in_city", n_complaints=12,
                     fan_out_range=(6, 10), hop_lambda=1,
                     start_date=datetime(2026, 1, 5), days=DAYS, seed=103),
    }

    for cfg_name, kwargs in specs.items():
        data = _gen_config(terminals, cfg_name, **kwargs)
        cfg_dir = OUT_DIR / cfg_name
        (cfg_dir / "transactions").mkdir(parents=True, exist_ok=True)
        txn = data["transactions"]
        txn["date"] = txn["timestamp"].dt.date.astype(str)
        for d, grp in txn.groupby("date"):
            day_dir = cfg_dir / "transactions" / f"date={d}"
            day_dir.mkdir(parents=True, exist_ok=True)
            grp.drop(columns=["date"]).to_parquet(day_dir / "part.parquet", index=False)
        data["accounts"].to_parquet(cfg_dir / "accounts.parquet", index=False)
        data["complaints"].to_parquet(cfg_dir / "complaints.parquet", index=False)
        print(f"{cfg_name}: {len(txn)} txns, {len(data['accounts'])} accounts, "
              f"{len(data['complaints'])} complaints, mule_hops={txn['is_mule_hop'].sum()}")

    print(f"Fixture written to {OUT_DIR}")


if __name__ == "__main__":
    build_fixture()
