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
