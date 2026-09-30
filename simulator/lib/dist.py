"""Sample from the small set of distribution specs used in rules.yaml configs.

Each spec is a dict like {"type": "poisson", "lambda": 3} or
{"type": "lognormal", "mu": 1.5, "sigma": 0.8} or {"type": "uniform", "low": 2, "high": 8}.
All sampling goes through a single numpy Generator for full reproducibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def sample(spec: dict[str, Any], rng: np.random.Generator, size: int | None = None):
    t = spec["type"]
    if t == "poisson":
        return rng.poisson(spec["lambda"], size=size)
    if t == "lognormal":
        return rng.lognormal(spec["mu"], spec["sigma"], size=size)
    if t == "uniform":
        low, high = spec["low"], spec["high"]
        if isinstance(low, int) and isinstance(high, int):
            return rng.integers(low, high + 1, size=size)
        return rng.uniform(low, high, size=size)
    raise ValueError(f"Unknown distribution type: {t}")


def sample_one(spec: dict[str, Any], rng: np.random.Generator):
    v = sample(spec, rng, size=None)
    return v.item() if hasattr(v, "item") else v
