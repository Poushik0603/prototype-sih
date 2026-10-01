"""
Mock ranked-terminal data generator.

Fabricates plausible ranked-terminal JSON objects matching the docs/DATA_SCHEMA.md
"model output / API response" shape, so the API + UI can be built and tested
end-to-end before model/predict.py exists.

Deterministic (seeded) so repeated calls within a process are stable per window.
"""
import hashlib
import random
from datetime import datetime, timedelta, timezone

# Bengaluru, India — plausible ATM/POS terminal cluster used for the demo city.
CENTER_LAT = 12.9716
CENTER_LON = 77.5946

BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Bank of Baroda", "Canara Bank", "Union Bank"]
TERMINAL_TYPES = ["ATM", "POS"]

SHAP_FEATURE_POOL = [
    "past_hotspot_freq",
    "txn_count_24h",
    "complaint_density",
    "mule_hop_count_7d",
    "avg_txn_amount",
    "distinct_accounts_24h",
    "night_txn_ratio",
    "cash_out_velocity",
    "terminal_age_days",
    "h3_neighbor_risk",
    "time_since_last_complaint",
    "account_age_min",
]

N_TERMINALS = 80

# Fixed window used when caller doesn't request a specific one — "latest available".
LATEST_WINDOW = datetime(2026, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
AVAILABLE_WINDOWS = [LATEST_WINDOW - timedelta(hours=2 * i) for i in range(12)]


def _seed_for_window(window_start: str) -> int:
    h = hashlib.sha256(window_start.encode()).hexdigest()
    return int(h[:8], 16)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_ranked_terminals(window_start: str | None = None):
    """
    Returns a list of ~N_TERMINALS dicts matching the schema contract, sorted
    by risk_score descending, with rank assigned 1..N.
    """
    if window_start is None:
        window_dt = LATEST_WINDOW
    else:
        try:
            window_dt = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
        except ValueError:
            window_dt = LATEST_WINDOW

    seed = _seed_for_window(_iso(window_dt))
    rng = random.Random(seed)

    terminals = []
    for i in range(1, N_TERMINALS + 1):
        terminal_id = f"T{i:05d}"
        # scatter around Bengaluru within ~0.15 deg (~15km)
        lat = CENTER_LAT + rng.uniform(-0.15, 0.15)
        lon = CENTER_LON + rng.uniform(-0.15, 0.15)

        # skew risk scores so a handful are clearly "hot"
        risk_score = round(rng.betavariate(1.5, 4.0), 4)
        baseline_score = round(max(0.0, min(1.0, risk_score + rng.uniform(-0.2, 0.05))), 4)

        n_feats = rng.randint(3, 5)
        feats = rng.sample(SHAP_FEATURE_POOL, n_feats)
        shap_values = sorted(
            (round(rng.uniform(0.02, 0.35), 4) for _ in feats), reverse=True
        )
        top_shap_features = [
            {"feature": f, "value": v} for f, v in zip(feats, shap_values)
        ]

        terminals.append(
            {
                "terminal_id": terminal_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "window_start": _iso(window_dt),
                "risk_score": risk_score,
                "rank": None,  # assigned after sort
                "top_shap_features": top_shap_features,
                "baseline_score": baseline_score,
                # extra fields useful for UI, not in the strict contract but harmless
                "bank": rng.choice(BANKS),
                "type": rng.choice(TERMINAL_TYPES),
            }
        )

    terminals.sort(key=lambda t: t["risk_score"], reverse=True)
    for idx, t in enumerate(terminals, start=1):
        t["rank"] = idx

    return terminals


def get_latest_window() -> str:
    return _iso(LATEST_WINDOW)


def get_available_windows():
    return [_iso(w) for w in AVAILABLE_WINDOWS]
