"""V6: pinned V3.2 task with only the authorized estimator/observation additions.

The Actor retains five directly stacked frames and loses only the three handle
position entries. The Critic retains all five original V3.2 frames and its
original privileges, then adds the independent hand labels and validity bit.
The cart HIM estimator has its own six-frame input history. The force estimator
keeps the V5.3 MLP and consumes the five Actor frames plus predicted cart speed.
"""
from __future__ import annotations

from pathlib import Path

from homie_pushcart.task_config import PushCartTaskCfg as V3TaskCfg

from .baseline import apply_config, baseline_source, config_dict, task_baseline


VERSION = 6
IMPLEMENTATION_REVISION = 1
OBSERVATION_SCHEMA = "v6_v32_121x5_him77x6_critic665_v1"
HOMIE_FRAME = 76
LEGACY_HL_FRAME = 124
HL_FRAME = LEGACY_HL_FRAME - 3
HL_EXTRA_FRAME = HL_FRAME - HOMIE_FRAME
ACTOR_HISTORY = 5
ESTIMATOR_FRAME = HOMIE_FRAME + 1
ESTIMATOR_HISTORY = 6
LEGACY_PRIVILEGED_FRAME = 32
ADDITIONAL_PRIVILEGED_FRAME = 12 + 1
PRIVILEGED_FRAME = LEGACY_PRIVILEGED_FRAME + ADDITIONAL_PRIVILEGED_FRAME


class PushCartTaskCfg(V3TaskCfg):
    version = VERSION
    implementation_revision = IMPLEMENTATION_REVISION
    observation_schema = OBSERVATION_SCHEMA
    baseline_source = {}

    class env(V3TaskCfg.env):
        estimator_history = ESTIMATOR_HISTORY

    def __init__(self):
        super().__init__()
        apply_config(self, task_baseline())
        self.baseline_source = baseline_source()
        # These options did not exist in V3.2; do not inherit V4 behavior.
        self.commands.sequence_enabled = False
        self.idle_correction.enabled = False
        self.rewards.force_rate_seed_on_reset = False


def num_actor_obs(cfg: PushCartTaskCfg | None = None) -> int:
    """Only the policy observation group; the estimator group is independent."""
    return HL_FRAME * (cfg.env.actor_history if cfg is not None else ACTOR_HISTORY)


def num_critic_obs(cfg: PushCartTaskCfg | None = None) -> int:
    history = cfg.env.actor_history if cfg is not None else ACTOR_HISTORY
    return LEGACY_HL_FRAME * history + PRIVILEGED_FRAME


def num_estimator_obs(cfg: PushCartTaskCfg | None = None) -> int:
    history = cfg.env.estimator_history if cfg is not None else ESTIMATOR_HISTORY
    return ESTIMATOR_FRAME * history


def run_config_path(checkpoint_or_run_dir) -> Path | None:
    path = Path(checkpoint_or_run_dir)
    run = path.parent if path.suffix == ".pt" else path
    for candidate in (run / "params" / "env.yaml", run.parent / "params" / "env.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _require_complete(values: dict, required: dict, path: str = "task") -> None:
    for key, expected in required.items():
        name = f"{path}.{key}"
        if key not in values:
            raise ValueError(f"Incomplete saved V6 configuration: missing {name}")
        if isinstance(expected, dict):
            if not isinstance(values[key], dict):
                raise ValueError(f"Saved V6 configuration requires a mapping at {name}")
            _require_complete(values[key], expected, name)


def validate_task_schema(cfg: PushCartTaskCfg) -> None:
    """Reject accidental cross-version loading or a different observation contract."""
    expected = {
        "version": VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "observation_schema": OBSERVATION_SCHEMA,
        "baseline_source": baseline_source(),
    }
    for name, value in expected.items():
        if getattr(cfg, name, None) != value:
            raise ValueError(f"V6 requires the pinned {name}; checkpoint migration is not implicit")
    if (cfg.env.actor_history != ACTOR_HISTORY
            or cfg.env.estimator_history != ESTIMATOR_HISTORY
            or cfg.env.num_hl_actions != 18):
        raise ValueError("V6 requires Actor history 5, estimator history 6, and 18 actions")
    if (cfg.commands.sequence_enabled or cfg.idle_correction.enabled
            or cfg.rewards.force_rate_seed_on_reset):
        raise ValueError("V6 retains V3.2 behavior; V4 command/idle/force-reset switches must stay off")


def task_cfg_from_run(checkpoint_or_run_dir=None, verbose: bool = True) -> PushCartTaskCfg:
    """New V6 defaults, or a complete saved V6 task; never fall back for a checkpoint."""
    cfg = PushCartTaskCfg()
    if checkpoint_or_run_dir is None:
        return cfg
    source = run_config_path(checkpoint_or_run_dir)
    if source is None:
        raise FileNotFoundError(f"Saved V6 params/env.yaml not found for {checkpoint_or_run_dir}")
    import yaml
    try:
        saved = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    except Exception as exc:
        raise ValueError(f"Cannot read saved V6 configuration {source}: {exc}") from exc
    task = saved.get("task") if isinstance(saved, dict) else None
    if not isinstance(task, dict):
        raise ValueError(f"Saved V6 configuration has no task mapping: {source}")
    if task.get("version") != VERSION:
        raise ValueError("V6 requires task.version=6; use the original entrypoint for older checkpoints")
    # A restored run may have saved the previous restore path as diagnostic metadata.
    task = {key: value for key, value in task.items() if key != "source_yaml"}
    _require_complete(task, config_dict(cfg))
    apply_config(cfg, task)
    validate_task_schema(cfg)
    cfg.source_yaml = str(source)
    if verbose:
        print(f"[pushcart-v6-cfg] restored pinned V6 task from {source}", flush=True)
    return cfg
