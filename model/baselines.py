"""Baselines to beat / compare against.

1. Past-hotspot frequency baseline: rank terminals by their as-of historical
   label rate (the same `past_hotspot_freq` feature computed in features.py,
   reused here unmodified so the baseline and the feature are guaranteed
   consistent). THIS IS THE BASELINE TO BEAT per PROJECT_SPEC.md.

2. Rule-only baseline: mirrors the simulator's own generator heuristic for
   terminal_choice. Per simulator/configs/{A,B,C}.yaml:
     - config A: "nearest_to_last_hop"    -> score terminals by proximity to
       recently (last 24h) active accounts' most recent terminal.
     - config B: "hotspot_avoidant"       -> score terminals INVERSELY by
       recent txn volume (mules deliberately avoid hotspots in B).
     - config C: "random_in_city"         -> no informative spatial signal by
       construction; rule-only baseline degenerates to uniform/random score.
   We implement all three variants and pick the one whose config we're
   scoring, since a "rule-only baseline that mirrors the generator" is
   inherently config-specific (that's the whole point of testing it under
   the circularity protocol -- a rule tuned to A's heuristic should do
   noticeably worse on B and C).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def past_hotspot_baseline(features: pd.DataFrame) -> pd.Series:
    """Returns baseline_score = past_hotspot_freq column (already as-of safe,
    computed in features.py). This IS the baseline -- no extra computation,
    just surfaced as its own named function for clarity/reuse from evaluate.py."""
    return features["past_hotspot_freq"].fillna(0.0)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def rule_only_baseline(features: pd.DataFrame, terminals: pd.DataFrame,
                        txns: pd.DataFrame, config_letter: str) -> pd.Series:
    """Rule-only score per row of `features` (must have terminal_id,
    window_start, lat, lon). Mirrors the generator's terminal_choice rule for
    the given config letter ('A'/'B'/'C', or 'fx_A' style -> normalized to
    the leading letter). As-of only: uses txns strictly before window_start.
    """
    letter = config_letter[-1] if config_letter.startswith("fx_") else config_letter
    term_meta = terminals.set_index("terminal_id")[["lat", "lon"]]

    txns_sorted = txns.sort_values("timestamp")
    scores = np.zeros(len(features))

    if letter == "C":
        # random_in_city: no spatial signal by construction -> uniform score.
        return pd.Series(np.full(len(features), 1.0 / max(len(terminals), 1)), index=features.index)

    # Precompute recent txn counts per terminal in trailing 6h, per unique window_start,
    # and (for A) the nearest "recently active account's last terminal" proximity.
    unique_ws = sorted(features["window_start"].unique())
    term_ids = terminals["terminal_id"].values
    lat_arr = terminals.set_index("terminal_id").loc[term_ids, "lat"].values
    lon_arr = terminals.set_index("terminal_id").loc[term_ids, "lon"].values

    row_score = {}
    for ws in unique_ws:
        lo6 = ws - pd.Timedelta(hours=6)
        recent = txns_sorted[(txns_sorted["timestamp"] >= lo6) & (txns_sorted["timestamp"] < ws)]
        counts = recent.groupby("terminal_id").size()
        count_arr = counts.reindex(term_ids).fillna(0).values

        if letter == "B":
            # hotspot_avoidant: mules prefer LOW recent activity -> score = inverse activity.
            s = 1.0 / (1.0 + count_arr)
        else:
            # A (default "nearest_to_last_hop"): score by proximity to the terminal(s)
            # with the most recent activity (proxy for "near last hop"), i.e. terminals
            # spatially close to wherever recent activity concentrated.
            if count_arr.sum() == 0:
                s = np.ones(len(term_ids))
            else:
                hot_idx = np.argsort(count_arr)[-max(1, len(term_ids) // 10):]
                hot_lat, hot_lon = lat_arr[hot_idx].mean(), lon_arr[hot_idx].mean()
                d = _haversine(hot_lat, hot_lon, lat_arr, lon_arr)
                s = 1.0 / (1.0 + d)

        s = s / (s.sum() if s.sum() > 0 else 1.0)
        for tid, sc in zip(term_ids, s):
            row_score[(tid, ws)] = sc

    out = features.apply(lambda r: row_score.get((r["terminal_id"], r["window_start"]), 0.0), axis=1)
    return out


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    top_k = order[:k]
    if len(top_k) == 0:
        return 0.0
    return float(y_true[top_k].sum()) / len(top_k)


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    base_rate = y_true.mean()
    if base_rate == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base_rate


def assemble_baseline_summary() -> dict:
    """Pulls the baseline (past-hotspot-freq and rule-only) blocks already
    computed per-config by evaluate.py's metrics_config_<X>.json files into
    one standalone results/metrics_baselines.json, so "what does the
    baseline alone score" is answerable without re-reading three separate
    model-comparison files. Does NOT re-run anything -- purely an
    aggregation of numbers already produced by actual evaluate.py runs."""
    results_dir = REPO_ROOT / "results"
    out = {
        "label": "SIMULATED DATA (synthetic simulator output, NOT real-world transactions)",
        "note": (
            "Baseline-only summary assembled from results/metrics_config_{A,B,C}.json "
            "(produced by model/evaluate.py). 'past_hotspot_freq' = as-of historical "
            "label rate per terminal (THE baseline to beat per PROJECT_SPEC.md). "
            "'rule_only' = heuristic mirroring the simulator's own terminal_choice "
            "rule per config (see model/baselines.py:rule_only_baseline / "
            "simulator/README.md). Headline metric is precision_at_10_per_window_mean "
            "(see evaluate.py for why the per-window version, not the global-flattened "
            "one, is the metric that matches actual deployment usage)."
        ),
        "by_config": {},
    }
    import json
    for c in ["A", "B", "C"]:
        p = results_dir / f"metrics_config_{c}.json"
        if not p.exists():
            out["by_config"][c] = {"error": "metrics_config file not found -- run evaluate.py first"}
            continue
        data = json.loads(p.read_text())
        out["by_config"][c] = {
            "tested_on_config": data.get("tested_on_config"),
            "n_test_rows": data.get("n_test_rows"),
            "n_positive_test_rows": data.get("n_positive_test_rows"),
            "positive_rate_test": data.get("positive_rate_test"),
            "past_hotspot_freq_baseline": data.get("baseline_past_hotspot_freq"),
            "rule_only_baseline": data.get("rule_only"),
            "model_for_comparison": data.get("model"),
            "model_beats_baseline": data.get("beats_baseline"),
        }
    out_path = results_dir / "metrics_baselines.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[baselines] wrote {out_path}")
    return out


def precision_at_k_per_window(df: pd.DataFrame, scores: np.ndarray, k: int = 10,
                               window_col: str = "window_start", label_col: str = "label") -> dict:
    """Precision@k computed PER WINDOW and averaged, not globally over the
    whole flattened test table.

    This matters a lot at this problem's scale: the real deployment ranks
    terminals WITHIN one 2h window at a time (the officer-view API scores
    one as-of window's ~1150 terminal candidates and returns a top-N ranked
    list -- see model/predict.py:CassandraModel.score and
    SCHEMA_CONTRACT.md's response shape, which has no window-spanning
    notion of "top 10 overall"). A single flattened precision@10 across
    hundreds of windows x 1150 terminals/window asks "are any of the 10
    single highest-scored rows in the ENTIRE test table positive", which is
    a near-impossible target when positives are spread across ~20 distinct
    windows (one every window, not concentrated in 10 rows total) --
    the metric that was actually requested is "when the officer asks ranks
    terminals for a window, are the true mule-hit terminals near the top
    of THAT window's list", which is precision@k averaged per window.

    Returns {"precision_at_k_per_window": mean over windows with >=1 row,
    "n_windows": count, "n_windows_with_positive": count of windows that
    had >=1 true positive label at all (a window with zero positives can
    only ever score precision=0, so we also report the metric restricted
    to windows that actually had a positive, which is the more informative
    "did we find the needle" number)}.
    """
    tmp = df[[window_col, label_col]].copy()
    tmp["_score"] = scores
    per_window = []
    per_window_only_positive = []
    for ws, g in tmp.groupby(window_col):
        y = g[label_col].values
        s = g["_score"].values
        p = precision_at_k(y, s, k=k)
        per_window.append(p)
        if y.sum() > 0:
            per_window_only_positive.append(p)
    return {
        "precision_at_k_per_window_mean": float(np.mean(per_window)) if per_window else 0.0,
        "precision_at_k_per_window_mean_on_positive_windows": (
            float(np.mean(per_window_only_positive)) if per_window_only_positive else 0.0
        ),
        "n_windows": len(per_window),
        "n_windows_with_positive": len(per_window_only_positive),
        "k": k,
    }


if __name__ == "__main__":
    assemble_baseline_summary()
