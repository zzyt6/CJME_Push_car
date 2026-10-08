"""Immutable V3.2 training defaults used by V7, independent of the logs directory.

The snapshot is copied from the actual V3.2 run, not today's V3/V4 defaults.
Every public accessor returns an independent copy so Hydra or a caller cannot
change the baseline used by a later environment.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path


BASELINE_PATH = Path(__file__).resolve().parent / "config" / "baseline_v32.json"
BASELINE_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _snapshot() -> dict:
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported V3.2 baseline snapshot schema: {BASELINE_PATH}")
    for name in ("source", "task", "homie", "agent", "env"):
        if not isinstance(data.get(name), dict) or not data[name]:
            raise ValueError(f"V3.2 baseline snapshot has no {name} mapping: {BASELINE_PATH}")
    if data["source"].get("version") != "V3.2":
        raise ValueError("V7 requires the pinned V3.2 baseline")
    return data


def load_baseline() -> dict:
    """Return the complete pinned snapshot (task, HOMIE, runner, env, provenance)."""
    return deepcopy(_snapshot())


def task_baseline() -> dict:
    return deepcopy(_snapshot()["task"])


def homie_baseline() -> dict:
    return deepcopy(_snapshot()["homie"])


def agent_baseline() -> dict:
    return deepcopy(_snapshot()["agent"])


def env_baseline() -> dict:
    """Return the complete original env.yaml mapping, with tuples normalized to lists."""
    result = deepcopy(_snapshot()["env"])
    result["task"] = task_baseline()
    result["homie"] = homie_baseline()
    return result


def baseline_source() -> dict:
    """Source files/checkpoints and exact hashes, embedded in each V7 saved task."""
    result = deepcopy(_snapshot()["source"])
    # Source hashes above preserve the original files. Hash the bundled data in
    # canonical form so Windows/Linux line endings cannot break V7 deployment.
    canonical = json.dumps(_snapshot(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["snapshot_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    result["snapshot_hash_format"] = "canonical_json_v1"
    return result


def apply_config(target, values: dict, path: str = "") -> None:
    """Apply known fields only; an unknown key must not silently become a default.

    Nested configuration objects keep their attribute interface. Dictionary
    values (such as provenance) remain dictionaries. All values are copied.
    """
    if not isinstance(values, dict):
        raise TypeError(f"Expected a configuration mapping at {path or '<root>'}")
    for key, value in values.items():
        name = f"{path}.{key}" if path else key
        if not hasattr(target, key):
            raise ValueError(f"Unknown V7 configuration field: {name}")
        current = getattr(target, key)
        if isinstance(value, dict) and not isinstance(current, dict):
            if not hasattr(current, "__dict__"):
                raise ValueError(f"V7 configuration field is not a section: {name}")
            apply_config(current, value, name)
        elif hasattr(current, "__dict__") and not isinstance(current, dict):
            raise ValueError(f"V7 configuration section requires a mapping: {name}")
        else:
            setattr(target, key, deepcopy(value))


def config_dict(value):
    """Serialize plain HOMIE-style configuration without importing a simulator."""
    if isinstance(value, dict):
        return {key: config_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: config_dict(getattr(value, key)) for key in dir(value)
                if not key.startswith("_") and not callable(getattr(value, key))}
    return deepcopy(value)
