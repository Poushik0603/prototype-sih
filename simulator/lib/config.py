"""Load and validate a simulator rules.yaml config file."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimConfig:
    name: str
    seed: int
    raw: dict[str, Any]

    @property
    def region(self) -> str:
        return self.raw["calibration"].get("region", "unknown")

    @property
    def mule_network(self) -> dict[str, Any]:
        return self.raw["mule_network"]

    @property
    def benign_background(self) -> dict[str, Any]:
        return self.raw["benign_background"]

    @property
    def label_noise(self) -> dict[str, Any]:
        return self.raw["label_noise"]

    @property
    def time_range(self) -> tuple[str, str]:
        tr = self.raw["time_range"]
        return tr["start"], tr["end"]

    @property
    def scale(self) -> dict[str, Any]:
        return self.raw.get("scale", {})

    @property
    def rules_version(self) -> str:
        return self.scale.get("rules_version", "sim-rules-v1.0-unknown")


def load_config(path: str | Path, name: str) -> SimConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SimConfig(name=name, seed=int(raw["seed"]), raw=raw)
