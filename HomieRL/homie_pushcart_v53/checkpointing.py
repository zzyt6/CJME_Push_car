"""V5.3 curriculum checkpoint adapters without editing the installed RSL-RL package.

Only environments exposing the V5.3 methods participate. The state is inserted
into RSL-RL's public ``infos`` checkpoint payload before its normal save.
Physics, in-flight episodes and PPO rollout buffers restart on resume; finished
episode windows and curriculum hysteresis survive exactly.
"""
from __future__ import annotations

import copy
import math
import os

import torch

from homie_pushcart_v52.curriculum import CurriculumTracker

STATE_KEY = "homie_pushcart_v53_curriculum"
LEVEL_OVERRIDE_ENV = "HOMIE_PUSHCART_V53_EXPLICIT_LEVEL"
WINDOW_TENSORS = ("_track_sum", "_fail_sum", "_count", "_gain_vr", "_gain_rr", "_gain_steps")


def _schedule_signature(env):
    from homie_isaaclab.math_utils import class_to_dict
    values = class_to_dict(env.task_cfg.curriculum)
    values.pop("init_level", None)
    values.pop("enabled", None)
    # Reward/command changes alter episode score meaning as well as the schedule.
    return {"curriculum": values, "commands": class_to_dict(env.task_cfg.commands),
            "randomization": class_to_dict(env.task_cfg.randomization),
            "observations": class_to_dict(env.task_cfg.observations),
            "tracking_sigma": env.task_cfg.rewards.tracking_sigma}


def curriculum_state(env) -> dict:
    return {"schema_version": 1, "tracker": env.curriculum.state_dict(),
            "completed_window": {name: getattr(env, name).detach().cpu().clone() for name in WINDOW_TENSORS},
            "next_eval_remaining_s": max(0, env._next_curriculum_eval - env.common_step_counter) * env.dt,
            "episodes_finished": int(env.episodes_finished), "term_counts": env.term_counts.detach().cpu().clone(),
            "schedule_signature": _schedule_signature(env),
            "resume_scope": "completed episode windows and curriculum; active physical episodes restart"}


def restore_curriculum(env, state: dict, explicit_level=None) -> dict:
    """Validate first, reset at the selected level, then restore aggregate statistics."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported V5.3 curriculum checkpoint state")
    probe = CurriculumTracker(env.task_cfg)
    probe.load_state_dict(state["tracker"])
    remaining = float(state["next_eval_remaining_s"])
    episodes = int(state["episodes_finished"])
    if not math.isfinite(remaining) or remaining < 0 or episodes < 0:
        raise ValueError("Invalid curriculum timer/episode totals in checkpoint")
    restored_tensors = {}
    for name in (*WINDOW_TENSORS, "term_counts"):
        value = state["term_counts"] if name == "term_counts" else state["completed_window"][name]
        target = getattr(env, name)
        tensor = torch.as_tensor(value, device=target.device, dtype=target.dtype)
        if tensor.shape != target.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid V5.3 curriculum tensor {name}: shape or finite-value mismatch")
        if name not in ("_gain_vr",) and (tensor < 0).any():
            raise ValueError(f"Invalid negative V5.3 curriculum tensor {name}")
        restored_tensors[name] = tensor.clone()
    if restored_tensors["_fail_sum"] > restored_tensors["_count"]:
        raise ValueError("Curriculum failed episodes exceed completed count")
    changed_schedule = state.get("schedule_signature") != _schedule_signature(env)
    discard_window = explicit_level is not None or changed_schedule
    if explicit_level is not None:
        level = int(explicit_level)
        if not 0 <= level < env.task_cfg.curriculum.num_levels:
            raise ValueError("Explicit resume level is outside the V5.3 curriculum")
        probe = CurriculumTracker(env.task_cfg, level)
    elif changed_schedule:
        previous_level, changes = probe.level, probe.changes
        probe = CurriculumTracker(env.task_cfg, previous_level)
        probe.changes = changes
    env.curriculum = probe
    env._apply_level(probe.params)
    # A runner has already constructed a fresh environment. Restart it again at
    # the restored level, then copy only completed-episode aggregates afterward.
    env.episode_length_buf.zero_()
    env.reset()
    for name in WINDOW_TENSORS:
        getattr(env, name).copy_(torch.zeros_like(restored_tensors[name]) if discard_window else restored_tensors[name])
    env.term_counts.copy_(restored_tensors["term_counts"])
    env.episodes_finished = episodes
    env._next_curriculum_eval = env.common_step_counter + (0 if discard_window else math.ceil(remaining / env.dt))
    result = {"level": probe.level, "completed_window_restored": not discard_window,
              "explicit_level": explicit_level, "schedule_changed": changed_schedule,
              "episodes_finished": episodes, "active_episodes_restarted": True}
    print(f"[pushcart-v53] restored curriculum: {result}", flush=True)
    return result


def _v53_env(runner):
    env = getattr(runner.env, "unwrapped", runner.env)
    return env if callable(getattr(env, "v53_curriculum_state_dict", None)) else None


def install_runner_checkpoint_hooks(runner_class=None):
    """Install once on the already-imported class used by NVIDIA's entrypoint."""
    if runner_class is None:
        from rsl_rl.runners import OnPolicyRunner
        runner_class = OnPolicyRunner
    if getattr(runner_class, "_homie_v53_checkpoint_hooks", False):
        return
    original_save, original_load = runner_class.save, runner_class.load

    def save(runner, path, infos=None):
        env = _v53_env(runner)
        if env is None:
            return original_save(runner, path, infos)
        if infos is not None and not isinstance(infos, dict):
            raise TypeError("V5.3 runner checkpoint infos must be a mapping or None")
        values = copy.deepcopy(infos) if infos is not None else {}
        values[STATE_KEY] = env.v53_curriculum_state_dict()
        return original_save(runner, path, values)

    def load(runner, path, *args, **kwargs):
        infos = original_load(runner, path, *args, **kwargs)
        env = _v53_env(runner)
        if env is not None:
            state = infos.get(STATE_KEY) if isinstance(infos, dict) else None
            if state is None:
                print("[pushcart-v53] checkpoint has no V5.3 curriculum state; using configured initial level. "
                      "Use --level to choose the migration level.", flush=True)
            else:
                override = os.environ.get(LEVEL_OVERRIDE_ENV)
                env.v53_load_curriculum_state_dict(state, explicit_level=int(override) if override is not None else None)
        return infos

    runner_class.save, runner_class.load = save, load
    runner_class._homie_v53_checkpoint_hooks = True
