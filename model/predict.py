"""Inference helper consumed by the API (see api/app/data_source.py).

Usage:

    from model.predict import CassandraModel

    model = CassandraModel.load("A")  # loads model/artifacts/xgb_config_A.json
    rows = model.score(feature_df)    # feature_df: rows from features_config_*.parquet
                                       # (or any DataFrame with the same feature columns)
    # rows is a list of dicts matching docs/DATA_SCHEMA.md's
    # "model output / API response" shape:
    # {terminal_id, lat, lon, window_start, risk_score, rank,
    #  top_shap_features, baseline_score}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "model" / "artifacts"


class CassandraModel:
    def __init__(self, booster: xgb.Booster, feature_cols: list[str], config: str):
        self.booster = booster
        self.feature_cols = feature_cols
        self.config = config
        self._explainer = shap.TreeExplainer(booster)

    @classmethod
    def load(cls, config: str = "A") -> "CassandraModel":
        model_path = ARTIFACTS_DIR / f"xgb_config_{config}.json"
        meta_path = ARTIFACTS_DIR / f"xgb_config_{config}_meta.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python model/train.py --config {config}` first."
            )
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        meta = json.loads(meta_path.read_text())
        return cls(booster, meta["feature_cols"], config)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_df missing required columns: {missing}")
        dmat = xgb.DMatrix(df[self.feature_cols])
        return self.booster.predict(dmat)

    def top_shap_features(self, df: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
        """Returns, per row, the top_n |SHAP value|-ranked features as
        [{"feature": name, "value": shap_value}, ...] -- matches
        docs/DATA_SCHEMA.md's top_shap_features field."""
        shap_values = self._explainer.shap_values(df[self.feature_cols])
        out = []
        for i in range(len(df)):
            row_vals = shap_values[i]
            order = np.argsort(-np.abs(row_vals))[:top_n]
            out.append([
                {"feature": self.feature_cols[j], "value": float(row_vals[j])}
                for j in order
            ])
        return out

    def score(self, df: pd.DataFrame, baseline_col: str = "past_hotspot_freq",
              top_n_shap: int = 3) -> list[dict]:
        """Scores a feature-table DataFrame and returns the API response shape
        from docs/DATA_SCHEMA.md, ranked descending by risk_score (rank=1 is
        highest risk)."""
        risk = self.predict_proba(df)
        shap_top = self.top_shap_features(df, top_n=top_n_shap)
        baseline = df[baseline_col].values if baseline_col in df.columns else np.zeros(len(df))

        out_rows = []
        for i in range(len(df)):
            row = df.iloc[i]
            out_rows.append({
                "terminal_id": row["terminal_id"],
                "lat": float(row["lat"]) if "lat" in df.columns else None,
                "lon": float(row["lon"]) if "lon" in df.columns else None,
                "window_start": pd.Timestamp(row["window_start"]).isoformat(),
                "risk_score": float(risk[i]),
                "rank": None,  # filled in below after sorting
                "top_shap_features": shap_top[i],
                "baseline_score": float(baseline[i]),
            })

        out_rows.sort(key=lambda r: -r["risk_score"])
        for rank, r in enumerate(out_rows, start=1):
            r["rank"] = rank
        return out_rows


if __name__ == "__main__":
    # smoke test
    import time
    model = CassandraModel.load("A")
    feat_path = REPO_ROOT / "data" / "processed" / "features_config_A.parquet"
    df = pd.read_parquet(feat_path)
    sample = df[df["split_tag"] == "test"].head(20)
    t0 = time.time()
    result = model.score(sample)
    print(f"scored {len(sample)} rows in {time.time()-t0:.3f}s")
    print(json.dumps(result[0], indent=2, default=str))
