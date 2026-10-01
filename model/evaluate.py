"""Evaluate a trained model (config A) on its own test split and, once
available, on other configs' test splits (circularity protocol step 1).
Also computes API-latency-style p50/p95 for model.predict() batches.

Saves results/metrics_config_<X>.json for each config evaluated (model vs
past-hotspot baseline vs rule-only baseline), honestly reporting
beats_baseline: true/false.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines import past_hotspot_baseline, rule_only_baseline, precision_at_k, lift_at_k
from features import load_raw_tables
from train import get_feature_cols, load_features

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"


def load_trained_model(train_config: str) -> tuple[xgb.Booster, list[str]]:
    model_path = ARTIFACTS_DIR / f"xgb_config_{train_config}.json"
    meta_path = ARTIFACTS_DIR / f"xgb_config_{train_config}_meta.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    meta = json.loads(meta_path.read_text())
    return booster, meta["feature_cols"]


def latency_benchmark(booster: xgb.Booster, feature_cols: list[str], df: pd.DataFrame,
                       batch_size: int = 100, n_batches: int = 30) -> dict:
    """Benchmarks model.predict() call time on batches of terminals (own
    model only -- NOT full API server latency, that's the API agent's job)."""
    rng = np.random.default_rng(42)
    times_ms = []
    n = len(df)
    if n == 0:
        return {"p50_ms": None, "p95_ms": None, "batch_size": batch_size, "n_batches": 0}
    for _ in range(n_batches):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = df.iloc[idx]
        dmat = xgb.DMatrix(batch[feature_cols])
        t0 = time.perf_counter()
        booster.predict(dmat)
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms = np.array(times_ms)
    return {
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "batch_size": batch_size,
        "n_batches": n_batches,
        "note": "model.predict() only (XGBoost Booster.predict on a DMatrix batch); "
                "excludes HTTP/serialization/API overhead -- that is the API agent's benchmark.",
    }


def lead_time_stats(df_test: pd.DataFrame, config: str) -> dict:
    """Lead time = time between the START of the predicted window and the
    actual mule withdrawal timestamp at that (terminal, window), for
    label=1 rows. Since our unit is a 2h window and the label is "a mule
    withdrawal occurred somewhere in this window", we can only bound lead
    time by the window width (a genuine limitation of the 2h-bucket design,
    noted honestly here rather than computing a precise timestamp-level lead
    time we don't have without re-joining raw transactions)."""
    n_pos = int(df_test["label"].sum())
    return {
        "computable": False,
        "reason": (
            "Feature table unit is (terminal, 2h window); exact withdrawal timestamp "
            "is not retained in the feature table, only window_start. Lead time is "
            "therefore bounded by the window width (<=2h from window_start to the "
            "actual mule withdrawal) but not point-estimable from this table alone. "
            "Would require re-joining raw transactions by (terminal_id, window) to "
            "get exact timestamps -- left as a follow-up, not computed here to avoid "
            "fabricating a precise number."
        ),
        "n_positive_test_windows": n_pos,
        "window_width_hours": 2,
    }


def evaluate_config(train_config: str, test_config: str, label_noise_note: str | None = None) -> dict:
    """Evaluate the model trained on `train_config` against `test_config`'s
    test split. If train_config == test_config, this is the in-config
    holdout test. Otherwise this is circularity-protocol cross-config eval."""
    booster, feature_cols = load_trained_model(train_config)

    df = load_features(test_config)
    if train_config == test_config:
        # in-config holdout: must use the held-out "test" split only (never
        # train/val rows -- those were used for fitting/tuning this model).
        test = df[df["split_tag"] == "test"].copy()
        if len(test) == 0:
            return {"error": f"no test split rows for config {test_config} "
                              f"(span too short to carve out a non-embargo test slice "
                              f"after two 7-day embargoes -- see make_rolling_origin_splits)"}
    else:
        # cross-config circularity check: the model was never trained or
        # tuned on ANY row of test_config (it's a different simulator run
        # entirely), so there is no leakage risk in using the full labeled
        # table here rather than just its internal "test" tag -- doing so
        # also gives more statistical power than config B/C's often-tiny
        # internal test slices (B/C's short time spans can leave "test"
        # empty under the embargo scheme; using the whole config avoids
        # silently dropping the circularity check in that case).
        test = df.copy()

    dmat = xgb.DMatrix(test[feature_cols])
    model_scores = booster.predict(dmat)
    baseline_scores = past_hotspot_baseline(test).values

    raw = load_raw_tables(test_config)
    rule_scores = rule_only_baseline(test, raw["terminals"], raw["transactions"], test_config).values

    y = test["label"].values

    def metric_block(scores):
        block = {
            "precision_at_10": precision_at_k(y, scores, k=10),
            "lift_at_10": lift_at_k(y, scores, k=10),
        }
        if y.sum() > 0:
            block["average_precision"] = float(average_precision_score(y, scores))
            block["roc_auc"] = float(roc_auc_score(y, scores)) if len(set(y)) > 1 else None
        else:
            block["average_precision"] = None
            block["roc_auc"] = None
        return block

    model_metrics = metric_block(model_scores)
    baseline_metrics = metric_block(baseline_scores)
    rule_metrics = metric_block(rule_scores)

    beats_baseline = (
        model_metrics["precision_at_10"] > baseline_metrics["precision_at_10"]
        if model_metrics["precision_at_10"] is not None else False
    )

    result = {
        "trained_on_config": train_config,
        "tested_on_config": test_config,
        "label": "SIMULATED DATA (synthetic simulator output, NOT real-world transactions)",
        "n_test_rows": int(len(test)),
        "n_positive_test_rows": int(y.sum()),
        "positive_rate_test": float(y.mean()),
        "model": model_metrics,
        "baseline_past_hotspot_freq": baseline_metrics,
        "rule_only": rule_metrics,
        "beats_baseline": bool(beats_baseline),
        "beats_baseline_note": (
            "model precision@10 > past-hotspot-frequency baseline precision@10 on this test set"
            if beats_baseline else
            "MODEL DID NOT BEAT THE BASELINE on this test set -- reported honestly, not hidden."
        ),
        "lead_time": lead_time_stats(test, test_config),
        "latency": latency_benchmark(booster, feature_cols, test),
    }
    if label_noise_note:
        result["label_noise_note"] = label_noise_note
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 of circularity protocol: train on A, test on A/B/C (whichever exist).
    available_configs = []
    for c in ["A", "B", "C"]:
        feat_path = REPO_ROOT / "data" / "processed" / f"features_config_{c}.parquet"
        if feat_path.exists():
            available_configs.append(c)
    print(f"[evaluate] available feature tables: {available_configs}")

    if "A" not in available_configs:
        print("[evaluate] config A features not found -- run features.py and train.py first.")
        return

    for test_cfg in available_configs:
        print(f"[evaluate] evaluating model trained on A against config {test_cfg} test split...")
        result = evaluate_config("A", test_cfg)
        out_path = RESULTS_DIR / f"metrics_config_{test_cfg}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[evaluate] wrote {out_path}")
        print(f"  model precision@10={result['model']['precision_at_10']:.3f} "
              f"baseline precision@10={result['baseline_past_hotspot_freq']['precision_at_10']:.3f} "
              f"beats_baseline={result['beats_baseline']}")


if __name__ == "__main__":
    main()
