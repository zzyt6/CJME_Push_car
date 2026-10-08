"""V7-only forwarding into NVIDIA's standard RSL-RL train/play entrypoints.

New training uses the pinned V3.2 defaults. Resume restores a complete V7
environment/runner config; it never treats a V3/V5 checkpoint as V7 weights.
The original V3.2 curriculum restart semantics are retained.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

from homie_pushcart.training import _lab_scripts, _checkpoint_args, _selection_patterns
from homie_pushcart.training import _resolve_checkpoint as _resolve_existing_checkpoint

from .baseline import config_dict
from .paths import ROOT
from .task_config import run_config_path, task_cfg_from_run

TASK_ID = "Homie-G1-PushCart-V7-Direct-v0"
EXPERIMENT = "homie_pushcart_v7"
_RESTORE_RUN_ENV = "HOMIE_PUSHCART_V7_RESTORE_CHECKPOINT"
_ACTOR_CLASS = "homie_pushcart_v7.interaction_policy:InteractionActor"
_PPO_CLASS = "homie_pushcart_v7.interaction_ppo:InteractionPPO"


def register_tasks():
    """Pre-Kit callback: register string factories without simulator imports."""
    from . import register_tasks as register_base_tasks
    args = register_base_tasks()
    if os.environ.get(_RESTORE_RUN_ENV):
        import gymnasium as gym
        spec = gym.spec(TASK_ID)
        spec.kwargs["env_cfg_entry_point"] = "homie_pushcart_v7.training:restored_env_cfg"
        spec.kwargs["rsl_rl_cfg_entry_point"] = "homie_pushcart_v7.training:restored_runner_cfg"
    return args


def _saved_configs(checkpoint):
    import yaml
    task = task_cfg_from_run(checkpoint, verbose=False)
    source = run_config_path(checkpoint)
    agent_source = source.with_name("agent.yaml")
    if not agent_source.is_file():
        raise FileNotFoundError(f"Saved V7 params/agent.yaml not found for {checkpoint}")
    env = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    agent = yaml.load(agent_source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(agent, dict):
        raise ValueError(f"Saved V7 agent config must be a mapping: {agent_source}")
    if (agent.get("actor", {}).get("class_name") != _ACTOR_CLASS
            or agent.get("algorithm", {}).get("class_name") != _PPO_CLASS
            or agent.get("obs_groups") != {"actor": ["policy", "estimator"], "critic": ["critic"]}):
        raise ValueError("Saved checkpoint is not the V7 Actor/PPO observation contract")
    for name in ("actor", "critic"):
        model = agent.get(name, {})
        if (model.get("hidden_dims") != [512, 256, 128]
                or model.get("activation") != "elu" or model.get("obs_normalization") is not True):
            raise ValueError(f"V7 retains the V3.2 {name} MLP and observation normalization")
    if not isinstance(env.get("homie"), dict):
        raise ValueError(f"Saved V7 environment has no complete HOMIE profile: {source}")
    return env, agent, task


def restored_env_cfg():
    from .lab_config import PushCartEnvCfg
    from .task_config import num_actor_obs, num_critic_obs
    saved, _, task = _saved_configs(os.environ[_RESTORE_RUN_ENV])
    cfg = PushCartEnvCfg()
    cfg.task = config_dict(task)
    cfg.task.pop("source_yaml", None)
    cfg.homie = deepcopy(saved["homie"])
    cfg.observation_space = num_actor_obs(task)
    cfg.state_space = num_critic_obs(task)
    cfg.action_space = task.env.num_hl_actions
    for name in ("checkpoint", "decimation", "episode_length_s", "curriculum_level", "seed",
                 "enable_randomization", "allow_robot_cart_contact", "cart_solver_iterations"):
        if name not in saved:
            raise ValueError(f"Incomplete saved V7 environment: missing {name}")
        setattr(cfg, name, deepcopy(saved[name]))
    for name in ("dt", "gravity"):
        if name not in saved.get("sim", {}):
            raise ValueError(f"Incomplete saved V7 environment: missing sim.{name}")
        setattr(cfg.sim, name, deepcopy(saved["sim"][name]))
    return cfg


def restored_runner_cfg():
    from .lab_config import PushCartRunnerCfg
    _, saved, _ = _saved_configs(os.environ[_RESTORE_RUN_ENV])
    cfg = PushCartRunnerCfg()
    fields = ("class_name", "actor", "critic", "algorithm", "num_steps_per_env", "empirical_normalization",
              "obs_groups", "clip_actions", "seed", "save_interval", "check_for_nan")
    missing = [name for name in fields if name not in saved]
    if missing:
        raise ValueError(f"Incomplete saved V7 runner config: missing {', '.join(missing)}")
    cfg.from_dict({name: deepcopy(saved[name]) for name in fields})
    return cfg


def _run(entrypoint: Path, args: list[str], *, restore_checkpoint: Path | None = None) -> int:
    command = [sys.executable, "-u", str(entrypoint), "--rl_library", "rsl_rl",
               "--external_callback", "homie_pushcart_v7.training.register_tasks", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(ROOT / "HomieRL"), env.get("PYTHONPATH", "")) if part)
    env.pop(_RESTORE_RUN_ENV, None)
    if restore_checkpoint is not None:
        env[_RESTORE_RUN_ENV] = str(restore_checkpoint)
    print("[pushcart-v7] NVIDIA entrypoint:", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def _validate_experiment(name):
    if (not name.startswith(EXPERIMENT) or Path(name).name != name
            or any(separator in name for separator in ("/", "\\"))):
        raise ValueError("V7 writes only to homie_pushcart_v7 or a homie_pushcart_v7_* experiment")


def _reject_alternate_profiles(argv):
    forbidden = {"--quality_profile", "--domain_randomization_profile", "--scene_version", "--scene_profile",
                 "--physics_backend", "--use_pretrained_checkpoint", "--homie_checkpoint"}
    for arg in argv:
        if arg.split("=", 1)[0] in forbidden:
            raise ValueError("V7 has one V3.2 physical/reward baseline and a fixed frozen HOMIE; "
                             f"alternate profile option {arg.split('=', 1)[0]} is not supported")


def _resolve_checkpoint(checkpoint, load_run, experiment):
    if not (ROOT / "logs/rsl_rl" / experiment).is_dir() and not (checkpoint and Path(checkpoint).is_file()):
        raise FileNotFoundError(f"No V7 training run under logs/rsl_rl/{experiment}; train V7 first")
    resolved = _resolve_existing_checkpoint(checkpoint, load_run, experiment)
    _saved_configs(resolved)
    return resolved


def _parser(resume):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--level", type=int, choices=range(18))
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--checkpoint", help="V7 high-level checkpoint; older schemas are rejected")
    parser.add_argument("--load_run", help="V7 run name or regex")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    if resume:
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--init_weights", action="store_true",
                            help="with --checkpoint: a new run with the CURRENT V7 configuration, initialised from that "
                                 "checkpoint's network / optimizer (no saved-configuration restore)")
    return parser


def _environment_args(options, args):
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    if options.no_randomization:
        args.append("env.enable_randomization=false")
    args.extend(["--task", TASK_ID, "--experiment_name", options.experiment_name])


def train(argv: list[str]) -> int:
    _reject_alternate_profiles(argv)
    parser = _parser(resume=True)
    options, args = parser.parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    _environment_args(options, args)
    checkpoint = None
    if options.checkpoint or options.resume:
        model_pattern, run_pattern = _selection_patterns(options, args)
        checkpoint = _resolve_checkpoint(model_pattern, run_pattern, options.experiment_name)
        args.extend(_checkpoint_args(checkpoint, resume_flag=True, experiment=options.experiment_name))
    elif options.load_run:
        parser.error("--load_run requires --resume or --checkpoint")
    if options.init_weights and checkpoint is None:
        parser.error("--init_weights requires --checkpoint")
    restore = None if options.init_weights else checkpoint
    if options.init_weights:
        print(f"[pushcart-v7] new run with the current configuration, weights from {checkpoint}", flush=True)
    return _run(_lab_scripts() / "train.py", args, restore_checkpoint=restore)


def play(argv: list[str]) -> int:
    _reject_alternate_profiles(argv)
    options, args = _parser(resume=False).parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    _environment_args(options, args)
    model_pattern, run_pattern = _selection_patterns(options, args)
    checkpoint = _resolve_checkpoint(model_pattern, run_pattern, options.experiment_name)
    args.extend(_checkpoint_args(checkpoint, resume_flag=False, experiment=options.experiment_name))
    return _run(_lab_scripts() / "play.py", args, restore_checkpoint=checkpoint)
