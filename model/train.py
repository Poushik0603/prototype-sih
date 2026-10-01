"""Train an XGBoost binary classifier (risk_score = predicted probability) on
config A's train split, tuned only on val, and save the model + a fixed
feature-column list for inference.

Usage: python model/train.py --config A
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Feature columns used by the model -- everything except identifiers/label/meta.
NON_FEATURE_COLS = {"terminal_id", "window_start", "label", "split_tag", "sim_config",
                     "lat", "lon", "type", "bank"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_features(config: str) -> pd.DataFrame:
    path = REPO_ROOT / "data" / "processed" / f"features_config_{config}.parquet"
    return pd.read_parquet(path)


def train_model(df: pd.DataFrame, feature_cols: list[str], params: dict | None = None) -> xgb.XGBClassifier:
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
        random_state=SEED,
        n_jobs=-1,
        # handle class imbalance (label=1 is rare -- mule windows are a small minority)
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        early_stopping_rounds=30,
    )
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def tune_on_val(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Minimal hyperparameter search over a small grid, selecting purely by
    val-set average precision (never touches test). Kept deliberately small
    for a thin slice -- a few configs, not a full sweep."""
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
            random_state=SEED, n_jobs=-1,
            scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
            early_stopping_rounds=30,
            **g,
        )
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, preds) if y_val.sum() > 0 else 0.0
        print(f"[train] grid {g} -> val AP={score:.4f}")
        if score > best_score:
            best_score, best_params = score, g
    print(f"[train] best val params: {best_params} (val AP={best_score:.4f})")
    return best_params


def main():
    set_all_seeds(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A")
    ap.add_argument("--skip-tune", action="store_true")
    args = ap.parse_args()

    df = load_features(args.config)
    feature_cols = get_feature_cols(df)
    print(f"[train] {len(df)} rows, {len(feature_cols)} feature cols: {feature_cols}")
    print(df["split_tag"].value_counts())

    best_params = {} if args.skip_tune else tune_on_val(df, feature_cols)
    model = train_model(df, feature_cols, best_params)

    artifacts_dir = REPO_ROOT / "model" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"xgb_config_{args.config}.json"
    model.get_booster().save_model(str(model_path))

    meta = {
        "config": args.config,
        "feature_cols": feature_cols,
        "seed": SEED,
        "best_params": best_params,
        "n_train": int((df["split_tag"] == "train").sum()),
        "n_val": int((df["split_tag"] == "val").sum()),
        "n_test": int((df["split_tag"] == "test").sum()),
        "train_label_rate": float(df.loc[df["split_tag"] == "train", "label"].mean()),
    }
    with open(artifacts_dir / f"xgb_config_{args.config}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    val = df[df["split_tag"] == "val"]
    val_preds = model.predict_proba(val[feature_cols])[:, 1]
    if val["label"].sum() > 0:
        print(f"[train] final val AP={average_precision_score(val['label'], val_preds):.4f} "
              f"AUC={roc_auc_score(val['label'], val_preds):.4f}")
    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved meta -> {artifacts_dir / f'xgb_config_{args.config}_meta.json'}")


if __name__ == "__main__":
    main()
