"""
ST-DBSCAN sanity check on real Chicago Crimes data (REAL DATA, not simulated).

Purpose: validate that the spatio-temporal clustering approach the Model agent uses for
candidate-terminal detection (ST-DBSCAN over ATM/POS terminal x 2h window events) behaves
sensibly on a real spatio-temporal point process — crime incidents in space and time, which
share the same "clustered in space AND time" structure as mule cash-out bursts.

Method: scikit-learn DBSCAN with a combined space+time distance. We build a 3D feature
space (x_km, y_km, t_hours) using a local equirectangular projection for lat/lon -> km, and
scale time so that `eps` trades off a spatial radius against a temporal window in one DBSCAN
call. This is a standard, lightweight way to implement ST-DBSCAN without a bespoke library,
per the benchmark agent's task note ("don't over-engineer a full custom library").

Two points are considered neighbors if:
  spatial_distance_km <= spatial_eps_km   AND   temporal_distance_hours <= temporal_eps_hours
We approximate this conjunction by rescaling both axes to share one Euclidean eps (a common
ST-DBSCAN approximation): x' = x_km / spatial_eps_km, t' = t_hours / temporal_eps_hours, then
run DBSCAN with eps=1 (unit hypersphere in rescaled space) — this is equivalent to an
ellipsoidal (space, time) neighborhood, a reasonable stand-in for the strict
space-AND-time-independently version for a sanity check at this scale.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "chicago_crimes_sample_2026.json"

# ST-DBSCAN parameters (chosen to mirror plausible terminal-cash-out clustering scales:
# a few hundred meters, a few hours — same order of magnitude as the project's 2h window unit)
SPATIAL_EPS_KM = 0.4    # ~400m radius
TEMPORAL_EPS_HOURS = 6.0  # 6-hour window
MIN_SAMPLES = 5

# Chicago reference point for equirectangular projection (approx city center)
REF_LAT = 41.8781
REF_LON = -87.6298
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * np.cos(np.radians(REF_LAT))


def load_data():
    with open(RAW_FILE, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["latitude", "longitude", "date"])
    return df


def project_xy(df):
    x_km = (df["longitude"] - REF_LON) * KM_PER_DEG_LON
    y_km = (df["latitude"] - REF_LAT) * KM_PER_DEG_LAT
    return x_km.values, y_km.values


def run_stdbscan(df):
    x_km, y_km = project_xy(df)
    t0 = df["date"].min()
    t_hours = (df["date"] - t0).dt.total_seconds().values / 3600.0

    # rescale so a unit Euclidean distance in rescaled space == being within
    # SPATIAL_EPS_KM spatially AND within TEMPORAL_EPS_HOURS temporally
    X = np.column_stack([
        x_km / SPATIAL_EPS_KM,
        y_km / SPATIAL_EPS_KM,
        t_hours / TEMPORAL_EPS_HOURS,
    ])

    db = DBSCAN(eps=1.0, min_samples=MIN_SAMPLES, metric="euclidean")
    labels = db.fit_predict(X)
    return labels, t0


def summarize(df, labels, t0):
    df = df.copy()
    df["cluster"] = labels
    n_noise = int((labels == -1).sum())
    n_points = len(labels)
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)

    sizes = df[df["cluster"] != -1].groupby("cluster").size().sort_values(ascending=False)

    # example hotspots: top 5 largest clusters with representative location/time/crime type
    examples = []
    for cid in sizes.index[:5]:
        sub = df[df["cluster"] == cid]
        examples.append({
            "cluster_id": int(cid),
            "size": int(len(sub)),
            "centroid_lat": round(float(sub["latitude"].mean()), 5),
            "centroid_lon": round(float(sub["longitude"].mean()), 5),
            "time_span_start": sub["date"].min().isoformat(),
            "time_span_end": sub["date"].max().isoformat(),
            "top_crime_types": sub["primary_type"].value_counts().head(3).to_dict(),
            "example_block_description": sub["location_description"].mode().iloc[0]
            if not sub["location_description"].mode().empty else None,
        })

    size_dist = sizes.describe().to_dict()

    result = {
        "benchmark": "st_dbscan_sanity_check",
        "data_source": "REAL DATA — Chicago Data Portal, Crimes 2001-present (Socrata API)",
        "is_real_data": True,
        "label_note": "All metrics below are from REAL Chicago crime data, NOT simulated.",
        "method": "DBSCAN with rescaled (space_km/eps_km, space_km/eps_km, time_hr/eps_hr) "
                  "features, eps=1.0 -- approximates ST-DBSCAN space-AND-time neighborhoods.",
        "parameters": {
            "spatial_eps_km": SPATIAL_EPS_KM,
            "temporal_eps_hours": TEMPORAL_EPS_HOURS,
            "min_samples": MIN_SAMPLES,
        },
        "input": {
            "n_points": n_points,
            "date_range_start": df["date"].min().isoformat(),
            "date_range_end": df["date"].max().isoformat(),
        },
        "results": {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_fraction": round(n_noise / n_points, 4),
            "n_clustered_points": n_points - n_noise,
            "clustered_fraction": round((n_points - n_noise) / n_points, 4),
            "cluster_size_distribution": {
                "min": float(size_dist.get("min", 0)),
                "25%": float(size_dist.get("25%", 0)),
                "50%_median": float(size_dist.get("50%", 0)),
                "75%": float(size_dist.get("75%", 0)),
                "max": float(size_dist.get("max", 0)),
                "mean": round(float(size_dist.get("mean", 0)), 2),
            },
            "top_5_example_hotspots": examples,
        },
    }
    return result


def main():
    df = load_data()
    print(f"Loaded {len(df)} real Chicago crime records")
    labels, t0 = run_stdbscan(df)
    result = summarize(df, labels, t0)

    out_path = RESULTS_DIR / "benchmark_stdbscan_chicago.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved results to {out_path}")
    print(json.dumps(result["results"], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
