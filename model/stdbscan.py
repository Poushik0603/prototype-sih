"""ST-DBSCAN candidate terminal shortlisting.

Clusters terminals in space + time using their recent transaction activity,
to narrow the search space before XGBoost ranking (per PROJECT_SPEC.md
pipeline: "ST-DBSCAN candidate terminals -> XGBoost ranker").

Implementation: plain scikit-learn DBSCAN over a precomputed distance matrix
that combines:
  - spatial distance: haversine distance (km) between terminal lat/lon
  - temporal distance: difference (hours) between each terminal's most recent
    transaction timestamp inside the lookback window ending at `as_of`

This runs per as-of timestamp (e.g. once per 2h window during scoring) over
the small terminal universe (tens to low hundreds of terminals), so an O(n^2)
precomputed distance matrix is fine at this scale -- no need for a custom
spatio-temporal index library.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _last_activity_hours_before(txns: pd.DataFrame, terminals: pd.DataFrame,
                                 as_of: pd.Timestamp, lookback_hours: float = 24.0) -> np.ndarray:
    """Hours since each terminal's most recent txn before as_of, within lookback.
    Terminals with no recent activity get lookback_hours (max distance)."""
    lo = as_of - pd.Timedelta(hours=lookback_hours)
    recent = txns[(txns["timestamp"] >= lo) & (txns["timestamp"] < as_of)]
    last_seen = recent.groupby("terminal_id")["timestamp"].max()
    out = []
    for tid in terminals["terminal_id"]:
        if tid in last_seen.index:
            hrs = (as_of - last_seen[tid]).total_seconds() / 3600.0
            out.append(hrs)
        else:
            out.append(lookback_hours)
    return np.array(out)


def candidate_terminals(terminals: pd.DataFrame, txns: pd.DataFrame, as_of: pd.Timestamp,
                         spatial_eps_km: float = 1.5, temporal_eps_hours: float = 6.0,
                         min_samples: int = 3, lookback_hours: float = 24.0) -> pd.DataFrame:
    """Returns terminals with an added `st_cluster` column (-1 = noise, not a
    candidate). Terminals assigned to any non-noise cluster are the ST-DBSCAN
    candidate shortlist for ranking at this as_of time.

    Combined distance: each dimension normalized by its own eps so a single
    scalar eps=1.0 in the combined metric means "within spatial_eps_km AND
    within temporal_eps_hours" (Chebyshev-style combination avoids having to
    invent an arbitrary km-per-hour conversion factor).
    """
    lat = terminals["lat"].values
    lon = terminals["lon"].values
    spatial_d = _haversine_matrix(lat, lon)
    temporal_hours = _last_activity_hours_before(txns, terminals, as_of, lookback_hours)
    temporal_d = np.abs(temporal_hours[:, None] - temporal_hours[None, :])

    combined = np.maximum(spatial_d / spatial_eps_km, temporal_d / temporal_eps_hours)

    db = DBSCAN(eps=1.0, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(combined)

    out = terminals.copy()
    out["st_cluster"] = labels
    out["last_activity_hours_before"] = temporal_hours
    return out


def shortlist(terminals: pd.DataFrame, txns: pd.DataFrame, as_of: pd.Timestamp,
              **kwargs) -> list[str]:
    """Convenience wrapper: returns just the terminal_id list of candidates
    (non-noise cluster members). If ST-DBSCAN finds zero clusters (e.g. very
    sparse activity), falls back to returning ALL terminals so the ranker
    still has something to score (fail-open, never fail-closed on a
    prototype's sparse dev fixture)."""
    clustered = candidate_terminals(terminals, txns, as_of, **kwargs)
    cands = clustered.loc[clustered["st_cluster"] != -1, "terminal_id"].tolist()
    if not cands:
        return terminals["terminal_id"].tolist()
    return cands


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import load_raw_tables

    raw = load_raw_tables("A")
    as_of = raw["transactions"]["timestamp"].max()
    result = candidate_terminals(raw["terminals"], raw["transactions"], as_of)
    n_clusters = (result["st_cluster"] != -1).sum()
    print(f"as_of={as_of}: {n_clusters}/{len(result)} terminals in a cluster "
          f"({result['st_cluster'].nunique() - (1 if -1 in result['st_cluster'].values else 0)} clusters)")
