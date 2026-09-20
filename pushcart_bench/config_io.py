"""benchmark.yaml loading with defaults."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import DEFAULT_CONFIG


def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("defaults", {})
    cfg.setdefault("follower", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("mirror", [])
    cfg.setdefault("steps", [])
    cfg.setdefault("step_payloads", [0.0])
    cfg.setdefault("robust", {"path": "s_curve", "factors": {}})
    cfg.setdefault("overload_thresholds", {})
    cfg.setdefault("gain_sweep", {})
    return cfg
