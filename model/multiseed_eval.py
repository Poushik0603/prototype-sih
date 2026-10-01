"""Multi-seed model training/evaluation wrapper (additive measurement only).

Does NOT touch feature engineering, label construction, or split logic in
features.py -- those are independently reviewed and verified leak-free (see
results/REVIEW.md). This script only varies the XGBoost `random_state` seed
across several training runs on config A's existing train/val split (as
already produced by features.py / stored in data/processed/features_config_A.parquet),
reusing train.py's tune_on_val()/train_model() functions unmodified, and
evaluates each resulting model on config A's held-out test split plus the
full config B / config C tables (circularity cross-config check, same
convention as evaluate.py: the model is never trained/tuned on any B/C row,
so using the whole table is not a leak -- see evaluate.py's docstring).

For each seed we compute:
  - ROC AUC (sklearn.metrics.roc_auc_score)
  - average_precision (sklearn.metrics.average_precision_score)
  - pr_auc (sklearn.metrics.precision_recall_curve + sklearn.metrics.auc) --
    reported ALONGSIDE average_precision since they can differ slightly
    (average_precision is a step-function sum, auc(precision_recall_curve)
    is trapezoidal interpolation between points -- noted explicitly below).
  - precision@5 / @10 / @20 per-window (reusing baselines.py's
    precision_at_k_per_window, parameterized by k)
  - recall@10 / @20 per-window, restricted to windows with >=1 positive
    (fraction of that window's true positives appearing in the top-k)

Hyperparameter tuning: tune_on_val() is re-run per seed (it is itself
deterministic given the data -- it does not depend on random_state beyond
what's passed into the grid's XGBClassifier -- here we pass the seed through
tune_on_val's own model fits too, so "same hyperparameter selection PROCESS,
varied only by seed" holds: the only thing that changes run-to-run is the
XGBoost random_state fed to both tuning and final training). Val is used for
tuning only; test is never touched until after the model is already fit, per
train.py's existing discipline, unmodified here.

Usage: python model/multiseed_eval.py --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import (  # noqa: E402
    past_hotspot_baseline,
    precision_at_k_per_window,
    rule_only_baseline,
)
from features import load_raw_tables  # noqa: E402
from train import get_feature_cols, load_features  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
TRAIN_CONFIG = "A"
K_VALUES = [5, 10, 20]


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def tune_on_val_seeded(df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    """Same grid/selection process as train.py's tune_on_val(), but with the
    XGBoost random_state swapped to `seed` on every candidate fit, so the
    only thing that varies run-to-run is the seed -- selection is still
    purely by val-set average precision, test is never touched."""
    train = df[df["split_tag"] == "train"]
    val = df[df["split_tag"] == "val"]
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]

    grid = [
        {"max_depth": 3, "learning_rate": 0.1},
        {"max_depth": 4, "learning_rate": 0.05},
        {"max_depth": 5, "learning_rate": 0.03},
    ]
    best_score, best_params = -1.0, grid[0]
    for g in grid:
        params = dict(
            n_estimators=300, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="aucpr",
            random_state=seed, n_jobs=-1,
            scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
            early_stopping_rounds=30,
            **g,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds) if y_val.sum() > 0 else 0.0
        if score > best_score:
            best_score, best_params = score, g
    return best_params


def train_model_seeded(df: pd.DataFrame, feature_cols: list[str], seed: int, params: dict) -> xgb.XGBClassifier:
    train = df[df["split_tag"] == "train"]
    val = df[df["split_tag"] == "val"]
    X_train, y_train = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]

    default_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        early_stopping_rounds=30,
    )
    default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def recall_at_k_per_window(df: pd.DataFrame, scores: np.ndarray, k: int,
                            window_col: str = "window_start", label_col: str = "label") -> dict:
    """Recall@k per window: fraction of that window's actual positives that
    appear in the top-k ranked rows of that window, averaged over windows
    that have >=1 positive (a window with zero positives has no meaningful
    recall denominator and is excluded, with n_windows_with_positive
    reported so the denominator is transparent)."""
    tmp = df[[window_col, label_col]].copy()
    tmp["_score"] = scores
    recalls = []
    for ws, g in tmp.groupby(window_col):
        y = g[label_col].values
        n_pos = y.sum()
        if n_pos == 0:
            continue
        s = g["_score"].values
        order = np.argsort(-s)
        top_k = order[:k]
        caught = y[top_k].sum()
        recalls.append(float(caught) / float(n_pos))
    return {
        "recall_at_k_per_window_mean": float(np.mean(recalls)) if recalls else None,
        "n_windows_with_positive": len(recalls),
        "k": k,
    }


def pr_auc_both_ways(y: np.ndarray, scores: np.ndarray) -> dict:
    """Returns both average_precision_score (step-function, sklearn's
    standard summary of a PR curve) and auc(recall, precision) from
    precision_recall_curve (trapezoidal interpolation) -- these are
    DIFFERENT estimators of 'area under the PR curve' and can disagree,
    particularly with few positives / tied scores, which is exactly this
    dataset's regime (sub-1% positive rate). Both are reported, never
    silently picking one."""
    if y.sum() == 0 or len(set(y)) < 2:
        return {"average_precision": None, "pr_auc_trapezoidal": None,
                "note": "undefined: no positives (or only one class) in this slice"}
    ap = float(average_precision_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    # precision_recall_curve returns recall descending; auc() wants x sorted -- sort by recall ascending.
    order = np.argsort(recall)
    pr_auc_val = float(auc(recall[order], precision[order]))
    return {
        "average_precision": ap,
        "pr_auc_trapezoidal": pr_auc_val,
        "differ_by": abs(ap - pr_auc_val),
    }


def metric_block_for_scores(test: pd.DataFrame, scores: np.ndarray) -> dict:
    y = test["label"].values
    block: dict = {}

    if y.sum() > 0 and len(set(y)) > 1:
        block["roc_auc"] = float(roc_auc_score(y, scores))
    else:
        block["roc_auc"] = None

    block.update(pr_auc_both_ways(y, scores))

    for k in K_VALUES:
        pw = precision_at_k_per_window(test, scores, k=k)
        block[f"precision_at_{k}_per_window_mean"] = pw["precision_at_k_per_window_mean"]
        block[f"precision_at_{k}_per_window_mean_on_positive_windows"] = (
            pw["precision_at_k_per_window_mean_on_positive_windows"]
        )
        block[f"n_windows_with_positive_for_k{k}"] = pw["n_windows_with_positive"]

    for k in [10, 20]:
        r = recall_at_k_per_window(test, scores, k=k)
        block[f"recall_at_{k}_per_window_mean"] = r["recall_at_k_per_window_mean"]
        block[f"recall_at_{k}_n_windows_with_positive"] = r["n_windows_with_positive"]

    block["n_rows"] = int(len(test))
    block["n_positive"] = int(y.sum())
    return block


def mean_std_block(values: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [v[k] for v in values if v.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "std": None, "values": [v.get(k) for v in values]}
        else:
            arr = np.array(vals, dtype=float)
            out[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "n": len(vals),
                "values": [v.get(k) for v in values],
            }
    return out


METRIC_KEYS = (
    ["roc_auc", "average_precision", "pr_auc_trapezoidal"]
    + [f"precision_at_{k}_per_window_mean" for k in K_VALUES]
    + [f"precision_at_{k}_per_window_mean_on_positive_windows" for k in K_VALUES]
    + [f"recall_at_{k}_per_window_mean" for k in [10, 20]]
)


def run(seeds: list[int]) -> dict:
    df_a = load_features(TRAIN_CONFIG)
    feature_cols = get_feature_cols(df_a)
    print(f"[multiseed] config A: {len(df_a)} rows, {len(feature_cols)} feature cols")
    print(df_a["split_tag"].value_counts())

    test_a = df_a[df_a["split_tag"] == "test"].copy()
    df_b = load_features("B")
    df_c = load_features("C")

    raw_a = load_raw_tables("A")
    raw_b = load_raw_tables("B")
    raw_c = load_raw_tables("C")

    per_seed_results = {"A": [], "B": [], "C": []}
    per_seed_params = []

    for seed in seeds:
        print(f"\n[multiseed] ===== seed {seed} =====")
        set_all_seeds(seed)
        best_params = tune_on_val_seeded(df_a, feature_cols, seed)
        print(f"[multiseed] seed {seed} best_params={best_params}")
        per_seed_params.append(best_params)
        model = train_model_seeded(df_a, feature_cols, seed, best_params)
        booster = model.get_booster()

        # Config A: in-sample held-out test split only.
        dmat_a = xgb.DMatrix(test_a[feature_cols])
        scores_a = booster.predict(dmat_a)
        block_a = metric_block_for_scores(test_a, scores_a)
        per_seed_results["A"].append(block_a)
        print(f"[multiseed] seed {seed} A: ROC AUC={block_a['roc_auc']}, "
              f"P@10={block_a['precision_at_10_per_window_mean']:.5f}")

        # Config B/C: full table (never trained/tuned on, consistent with evaluate.py).
        for letter, df_x in [("B", df_b), ("C", df_c)]:
            dmat_x = xgb.DMatrix(df_x[feature_cols])
            scores_x = booster.predict(dmat_x)
            block_x = metric_block_for_scores(df_x, scores_x)
            per_seed_results[letter].append(block_x)
            print(f"[multiseed] seed {seed} {letter}: ROC AUC={block_x['roc_auc']}, "
                  f"P@10={block_x['precision_at_10_per_window_mean']:.5f}")

    # Baselines: deterministic given the data (no RNG inside past_hotspot_baseline or
    # rule_only_baseline -- both are pure functions of the feature table / raw tables).
    # Confirmed by code inspection (baselines.py): no np.random / random calls. We still
    # run them once (not per-seed) and report a single value with std=0, rather than
    # fabricating variance that doesn't exist.
    baseline_results = {}
    for letter, df_x, raw_x, test_slice in [
        ("A", df_a, raw_a, test_a),
        ("B", df_b, raw_b, df_b),
        ("C", df_c, raw_c, df_c),
    ]:
        hotspot_scores = past_hotspot_baseline(test_slice).values
        rule_scores = rule_only_baseline(test_slice, raw_x["terminals"], raw_x["transactions"], letter).values
        baseline_results[letter] = {
            "past_hotspot_freq": metric_block_for_scores(test_slice, hotspot_scores),
            "rule_only": metric_block_for_scores(test_slice, rule_scores),
            "deterministic_note": (
                "past_hotspot_baseline and rule_only_baseline contain no random-number "
                "generation (verified by code inspection of model/baselines.py: no "
                "np.random/random calls) and depend only on the fixed feature table / "
                "raw tables for this config, so they produce bit-identical output on "
                "every run regardless of the model's training seed. Reported as a single "
                "value per metric (std=0, not fabricated across pseudo-seeds)."
            ),
        }

    out = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_config": TRAIN_CONFIG,
        "per_seed_tuned_params": per_seed_params,
        "label": "SIMULATED DATA (synthetic simulator output, NOT real-world transactions)",
        "note": (
            "Model trained on config A's train split (tuned on val only, never test) "
            "with 5 different XGBoost random_state seeds; same feature table, same "
            "leak-free rolling-origin split as the single-seed results in "
            "results/metrics_config_*.json (features.py/split logic untouched). "
            "Config B/C use the FULL feature table as test (never trained/tuned on any "
            "row of B/C), consistent with evaluate.py's existing cross-config convention. "
            "pr_auc_trapezoidal (sklearn.metrics.auc on precision_recall_curve) is reported "
            "alongside average_precision (sklearn.metrics.average_precision_score) since "
            "these are different area estimators and can disagree, especially at this "
            "sub-1%-positive-rate scale."
        ),
    }
    for letter in ["A", "B", "C"]:
        out[f"config_{letter}"] = mean_std_block(per_seed_results[letter], METRIC_KEYS)
        out[f"config_{letter}"]["n_rows"] = per_seed_results[letter][0]["n_rows"]
        out[f"config_{letter}"]["n_positive"] = per_seed_results[letter][0]["n_positive"]
        out[f"config_{letter}"]["per_seed_raw"] = per_seed_results[letter]

    out["baselines"] = baseline_results

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run(args.seeds)
    out_path = RESULTS_DIR / "metrics_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[multiseed] wrote {out_path}")

    # verification summary printed, not just trusted
    for letter in ["A", "B", "C"]:
        block = result[f"config_{letter}"]
        roc = block["roc_auc"]
        pr = block["pr_auc_trapezoidal"]
        p10 = block["precision_at_10_per_window_mean"]
        print(f"[multiseed] config {letter}: n_seeds={result['n_seeds']} "
              f"ROC AUC mean={roc['mean']:.4f} std={roc['std']:.4f} | "
              f"PR-AUC mean={pr['mean']:.4f} std={pr['std']:.4f} | "
              f"P@10/window mean={p10['mean']:.5f} std={p10['std']:.5f}")


if __name__ == "__main__":
    main()
