#!/usr/bin/env python
"""Simulator entry point.

Usage:
    .venv/Scripts/python.exe simulator/generate.py --config A
    .venv/Scripts/python.exe simulator/generate.py --config B
    .venv/Scripts/python.exe simulator/generate.py --config C
    .venv/Scripts/python.exe simulator/generate.py --all

For a given config (A/B/C), loads simulator/configs/<X>.yaml, loads the
terminals table (real data/simulated/fixture/terminals.parquet if present,
else an auto-generated placeholder), generates accounts/transactions/
complaints, writes them to data/simulated/config_<X>/ per docs/DATA_SCHEMA.md,
and writes a run manifest (+ best-effort MLflow local-store logging) with
seed, calibration hash, config name, and rules version for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.config import load_config  # noqa: E402
from lib.generate_core import generate  # noqa: E402
from lib.terminals import load_terminals  # noqa: E402

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
OUT_ROOT = REPO_ROOT / "data" / "simulated"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_one(config_name: str) -> dict:
    cfg_path = CONFIGS_DIR / f"{config_name}.yaml"
    cfg = load_config(cfg_path, name=config_name)

    # seed everything for exact reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    terminals, terminals_source, calibration_hash = load_terminals()

    result = generate(cfg, terminals)

    out_dir = OUT_ROOT / f"config_{config_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # accounts.parquet (unpartitioned)
    result.accounts.to_parquet(out_dir / "accounts.parquet", index=False)

    # complaints.parquet (unpartitioned)
    result.complaints.to_parquet(out_dir / "complaints.parquet", index=False)

    # terminals.parquet passthrough (shared terminal universe for this config)
    terminals.to_parquet(out_dir / "terminals.parquet", index=False)

    # transactions partitioned by day
    txn_dir = out_dir / "transactions"
    txn_dir.mkdir(parents=True, exist_ok=True)
    transactions = result.transactions.copy()
    transactions["date"] = transactions["timestamp"].dt.strftime("%Y-%m-%d")
    for day, group in transactions.groupby("date"):
        day_dir = txn_dir / f"date={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["date"]).to_parquet(day_dir / "part.parquet", index=False)

    n_days = transactions["date"].nunique()

    cfg_bytes = (CONFIGS_DIR / f"{config_name}.yaml").read_bytes()
    manifest = {
        "config_name": config_name,
        "config_file_sha256": _sha256_bytes(cfg_bytes),
        "seed": cfg.seed,
        "rules_version": cfg.rules_version,
        "terminals_source": terminals_source,
        "terminals_calibration_sha256": calibration_hash,
        "n_terminals": len(terminals),
        "time_range": {"start": cfg.time_range[0], "end": cfg.time_range[1]},
        "n_days_with_transactions": int(n_days),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats": result.stats,
    }

    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # best-effort MLflow local-store logging; never block/fail the run on this
    try:
        import os

        # MLflow is used with a local file store here, no server. Newer MLflow
        # (3.x) puts the plain filesystem backend in "maintenance mode" and refuses to
        # write unless this is explicitly opted into.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(f"file:{(REPO_ROOT / 'mlruns').as_posix()}")
        mlflow.set_experiment("simulator")
        with mlflow.start_run(run_name=f"config_{config_name}"):
            mlflow.log_param("config_name", config_name)
            mlflow.log_param("seed", cfg.seed)
            mlflow.log_param("rules_version", cfg.rules_version)
            mlflow.log_param("terminals_source", terminals_source)
            mlflow.log_param("terminals_calibration_sha256", calibration_hash)
            mlflow.log_param("time_range_start", cfg.time_range[0])
            mlflow.log_param("time_range_end", cfg.time_range[1])
            for k, v in result.stats.items():
                mlflow.log_metric(k, v)
    except ImportError:
        print(f"[generate] mlflow not installed yet - skipping MLflow logging for config {config_name} "
              f"(manifest still written to {out_dir / 'run_manifest.json'})")
    except Exception as e:  # never let mlflow issues fail the generator run
        print(f"[generate] mlflow logging failed non-fatally for config {config_name}: {e}")

    print(f"[generate] config {config_name}: wrote {len(result.transactions)} transactions "
          f"({n_days} days), {len(result.accounts)} accounts, {len(result.complaints)} complaints "
          f"to {out_dir}")
    print(f"[generate] config {config_name}: stats = {json.dumps(result.stats)}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C"], help="single config to run")
    parser.add_argument("--all", action="store_true", help="run all configs A, B, C")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("specify --config A|B|C or --all")

    names = ["A", "B", "C"] if args.all else [args.config]
    for name in names:
        run_one(name)


if __name__ == "__main__":
    main()
