"""
Data source adapter: switches between mock data and the real model pipeline.

Controlled by the USE_MOCK env var:
  - USE_MOCK=true  (default if unset, or if real artifacts aren't available yet)
  - USE_MOCK=false -> attempts to import model/predict.py and score the real
    feature table at data/processed/features_config_A.parquet. Falls back to
    mock automatically (and logs why) if anything is missing, so the demo
    always works end-to-end.
"""
import logging
import os
import sys
from pathlib import Path

from . import mock_data

logger = logging.getLogger("data_source")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [DATA_SOURCE] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features_config_A.parquet"
PREDICT_MODULE_PATH = REPO_ROOT / "model" / "predict.py"

_USE_MOCK_ENV = os.environ.get("USE_MOCK", "true").strip().lower()
FORCE_MOCK = _USE_MOCK_ENV in ("1", "true", "yes")

_cached_real_terminals = None
_real_backend_available = None  # tri-state: None=unchecked, True/False after check


def _try_load_real_backend():
    """
    Attempts to import model.predict and score the real feature table.
    Returns a list of ranked terminal dicts, or None if unavailable for any reason.
    Never raises — always falls back cleanly.
    """
    global _real_backend_available

    if not PREDICT_MODULE_PATH.exists():
        logger.info("model/predict.py not found yet — using mock data.")
        _real_backend_available = False
        return None

    if not FEATURES_PATH.exists():
        logger.info(
            "data/processed/features_config_A.parquet not found yet — using mock data."
        )
        _real_backend_available = False
        return None

    try:
        sys.path.insert(0, str(REPO_ROOT))
        from model import predict as predict_module  # type: ignore

        import pandas as pd

        features_df = pd.read_parquet(FEATURES_PATH)

        if hasattr(predict_module, "score_ranked_terminals"):
            results = predict_module.score_ranked_terminals(features_df)
        elif hasattr(predict_module, "predict"):
            results = predict_module.predict(features_df)
        else:
            logger.warning(
                "model/predict.py exists but exposes no score_ranked_terminals()/predict() "
                "function — using mock data."
            )
            _real_backend_available = False
            return None

        # normalize to list of dicts
        if hasattr(results, "to_dict"):
            results = results.to_dict(orient="records")

        logger.info("Loaded real model output: %d terminals.", len(results))
        _real_backend_available = True
        return results

    except Exception as exc:  # noqa: BLE001 - deliberately broad, must never crash the API
        logger.warning("Real model backend failed (%s) — falling back to mock data.", exc)
        _real_backend_available = False
        return None


def is_using_mock() -> bool:
    if FORCE_MOCK:
        return True
    return not bool(_real_backend_available)


def get_ranked_terminals(window_start: str | None = None):
    """
    Returns the full ranked terminal list (unfiltered by limit) for the given window.
    Tries the real backend first (unless USE_MOCK forces mock), falls back to mock_data.
    """
    global _cached_real_terminals

    if not FORCE_MOCK:
        if _cached_real_terminals is None:
            _cached_real_terminals = _try_load_real_backend()
        if _cached_real_terminals:
            return _cached_real_terminals

    return mock_data.generate_ranked_terminals(window_start)


def get_latest_window() -> str:
    # Real backend doesn't have a "latest window" concept wired yet; mock provides one.
    return mock_data.get_latest_window()
