"""Forward `pushcart_v5.py train / play` into NVIDIA's unmodified RSL-RL entrypoints.

Same approach as ``homie_isaaclab.training``: a clean child interpreter runs
``scripts/reinforcement_learning/{train,play}.py --rl_library rsl_rl`` with our task registered through
``--external_callback homie_pushcart_v51.training.register_tasks``. The callback restores the selected run's
configuration before Hydra applies user overrides. Everything after the recognised options is forwarded
verbatim (``--num_envs``, ``--max_iterations``, ``--headless``, Hydra overrides such as
``env.task.curriculum.init_level=4`` or ``agent.max_iterations=...``).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .paths import ROOT
from .cart_quality import QUALITY_PROFILES, quality_profile_overrides

TASK_ID = "Homie-G1-PushCart-V51-Direct-v0"
EXPERIMENT = "homie_pushcart_v51"
_RESTORE_RUN_ENV = "HOMIE_PUSHCART_V51_RESTORE_CHECKPOINT"


def register_tasks():
    """Register saved configuration factories before Hydra applies explicit user overrides."""
    from . import register_tasks as register_base_tasks
    args = register_base_tasks()
    from .checkpointing import install_runner_checkpoint_hooks
    install_runner_checkpoint_hooks()
    if os.environ.get(_RESTORE_RUN_ENV):
        import gymnasium as gym
        gym.spec(TASK_ID).kwargs["env_cfg_entry_point"] = "homie_pushcart_v51.training:restored_env_cfg"
        gym.spec(TASK_ID).kwargs["rsl_rl_cfg_entry_point"] = "homie_pushcart_v51.training:restored_runner_cfg"
    return args


def restored_env_cfg():
    """Restore task/control settings, retaining today's launcher/device/logging defaults."""
    import yaml
    from homie_isaaclab.math_utils import class_to_dict
    from .lab_config import PushCartEnvCfg
    from .task_config import num_actor_obs, num_critic_obs, run_config_path, task_cfg_from_run

    checkpoint = Path(os.environ[_RESTORE_RUN_ENV])
    source = run_config_path(checkpoint)
    cfg = PushCartEnvCfg()
    if source is None:
        print(f"[pushcart] No saved env.yaml for {checkpoint}; using current defaults", flush=True)
        return cfg
    # Isaac Lab writes tuples in env.yaml. FullLoader supports those without constructing arbitrary objects.
    saved = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(saved, dict) or not isinstance(saved.get("task"), dict):
        raise ValueError(f"Saved environment config has no task mapping: {source}")
    task = task_cfg_from_run(checkpoint)
    cfg.task = class_to_dict(task)
    cfg.task.pop("source_yaml", None)
    cfg.observation_space = num_actor_obs(task)
    cfg.state_space = num_critic_obs(task)
    cfg.action_space = task.env.num_hl_actions

    def merge_known(defaults, values):
        for key, value in values.items():
            if key in defaults:
                if isinstance(defaults[key], dict) and isinstance(value, dict):
                    merge_known(defaults[key], value)
                else:
                    defaults[key] = copy.deepcopy(value)

    if isinstance(saved.get("homie"), dict):
        merge_known(cfg.homie, saved["homie"])
    for name in ("checkpoint", "decimation", "episode_length_s", "curriculum_level",
                 "enable_randomization", "allow_robot_cart_contact", "cart_solver_iterations"):
        if name in saved:
            setattr(cfg, name, copy.deepcopy(saved[name]))
    # A run may have used the default frozen policy, leaving env.yaml's checkpoint null.
    metadata_path = source.parent.parent / "pushcart_task.json"
    if not cfg.checkpoint and metadata_path.is_file():
        cfg.checkpoint = json.loads(metadata_path.read_text(encoding="utf-8")).get("homie_checkpoint")
    for name in ("dt", "gravity"):
        if name in saved.get("sim", {}):
            setattr(cfg.sim, name, copy.deepcopy(saved["sim"][name]))
    from .checkpointing import LEVEL_OVERRIDE_ENV
    if os.environ.get(LEVEL_OVERRIDE_ENV) is not None:
        cfg.curriculum_level = int(os.environ[LEVEL_OVERRIDE_ENV])
    return cfg


def restored_runner_cfg():
    """Restore network/PPO settings without reusing the previous run's name or device."""
    import yaml
    from .lab_config import PushCartRunnerCfg
    from .task_config import run_config_path

    cfg = PushCartRunnerCfg()
    source = run_config_path(os.environ[_RESTORE_RUN_ENV])
    if source is None or not source.with_name("agent.yaml").is_file():
        return cfg
    saved = yaml.load(source.with_name("agent.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(saved, dict):
        raise ValueError(f"Saved agent config must be a mapping: {source.with_name('agent.yaml')}")
    fields = ("class_name", "actor", "critic", "algorithm", "num_steps_per_env", "empirical_normalization",
              "obs_groups", "clip_actions", "seed", "save_interval")
    cfg.from_dict({key: value for key, value in saved.items() if key in fields})
    return cfg


def _lab_scripts() -> Path:
    import isaaclab
    lab_root = Path(os.environ.get("ISAACLAB_PATH", Path(isaaclab.__file__).resolve().parents[3]))
    scripts = lab_root / "scripts/reinforcement_learning"
    if not (scripts / "train.py").is_file():
        raise FileNotFoundError(f"Official Isaac Lab entrypoints not found under {scripts}")
    return scripts


def _run(entrypoint: Path, args: list[str], *, restore_checkpoint: Path | None = None) -> int:
    command = [sys.executable, "-u", str(entrypoint), "--rl_library", "rsl_rl",
               "--external_callback", "homie_pushcart_v51.training.register_tasks", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT / "HomieRL"), env.get("PYTHONPATH", "")) if p)
    env.pop(_RESTORE_RUN_ENV, None)
    from .checkpointing import LEVEL_OVERRIDE_ENV
    env.pop(LEVEL_OVERRIDE_ENV, None)
    if restore_checkpoint is not None:
        env[_RESTORE_RUN_ENV] = str(restore_checkpoint)
        explicit = _explicit_level(args)
        if explicit is not None:
            env[LEVEL_OVERRIDE_ENV] = str(explicit)
    print("[pushcart] NVIDIA entrypoint:", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def _explicit_level(args):
    """Track CLI intent independently of the saved config's initial level."""
    selected = None
    for arg in args:
        key, separator, value = arg.lstrip("+").partition("=")
        if separator and key in ("env.curriculum_level", "env.task.curriculum.init_level"):
            selected = int(value)
    return selected


def _resolve_checkpoint(checkpoint: str | None, load_run: str | None, experiment: str) -> Path:
    """Resolve before launch so the policy and its saved environment always select the same run."""
    if checkpoint:
        source = Path(checkpoint).expanduser()
        if source.is_file():
            return source.resolve()
        if source.is_absolute() or "/" in checkpoint:
            raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    try:
        model_pattern = re.compile(checkpoint or r"model_.*\.pt$")
    except re.error as exc:
        raise ValueError(f"Checkpoint is neither an existing file nor a valid filename regex: {checkpoint}") from exc
    log_root = ROOT / "logs/rsl_rl" / experiment
    if not log_root.is_dir():
        raise FileNotFoundError(f"No V5 training runs under {log_root}; train V5 before selecting a checkpoint")
    runs = sorted(p for p in log_root.iterdir() if p.is_dir() and re.match(load_run or ".*", p.name))
    if not runs:
        raise FileNotFoundError(f"No training runs under {log_root} matching {load_run or '.*'}")
    models = [p for p in runs[-1].iterdir() if p.is_file() and model_pattern.match(p.name)]
    if not models:
        raise FileNotFoundError(f"No checkpoints in {runs[-1]} matching {checkpoint or 'model_*.pt'}")
    return max(models, key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)]).resolve()


def _checkpoint_args(checkpoint: Path, resume_flag: bool, experiment: str = EXPERIMENT) -> list[str]:
    if not resume_flag:
        # Official play treats --checkpoint as a file path, unlike train's filename regex.
        return ["--checkpoint", str(checkpoint)]
    log_root = (ROOT / "logs/rsl_rl" / experiment).resolve()
    if checkpoint.parent.parent != log_root:
        raise FileNotFoundError(f"Resume checkpoint must live in a run directory under {log_root}: {checkpoint}")
    return ["--load_run", re.escape(checkpoint.parent.name) + "$", "--checkpoint", re.escape(checkpoint.name) + "$", "--resume"]


def _selection_patterns(options, args: list[str]) -> tuple[str | None, str | None]:
    """Apply filename/run Hydra selectors, with official CLI options taking precedence."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    selected = {}
    for arg in args:
        key = arg.lstrip("+").split("=", 1)[0]
        if key in ("agent.load_run", "agent.load_checkpoint") and "=" in arg:
            selected[key] = OverridesParser.create().parse_override(arg).value()
    return (options.checkpoint or selected.get("agent.load_checkpoint"),
            options.load_run or selected.get("agent.load_run"))


def train(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--homie_checkpoint", help="frozen HOMIE model_*.pt (default: latest in the HOMIE run dir)")
    parser.add_argument("--level", type=int, help="curriculum start level")
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--checkpoint", help="resume from this high-level checkpoint (inside logs/rsl_rl/homie_pushcart_v51)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", help="run directory name or regex for resuming")
    parser.add_argument("--quality_profile", choices=QUALITY_PROFILES,
                        help="scratch default: v51 interaction learning; resume keeps saved settings unless explicitly overridden")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    profile = options.quality_profile or (None if options.checkpoint or options.resume else "v51")
    if profile:
        overrides = quality_profile_overrides(profile)
        args = [f"env.task.{key}={str(value).lower() if isinstance(value, bool) else value}"
                for key, value in overrides.items()] + args
    if options.homie_checkpoint:
        args.append(f"env.checkpoint={Path(options.homie_checkpoint).resolve().as_posix()}")
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    if options.no_randomization:
        args.append("env.enable_randomization=false")
    args.extend(["--task", TASK_ID, "--experiment_name", options.experiment_name])
    checkpoint = None
    if options.checkpoint or options.resume:
        model_pattern, run_pattern = _selection_patterns(options, args)
        checkpoint = _resolve_checkpoint(model_pattern, run_pattern, options.experiment_name)
        args.extend(_checkpoint_args(checkpoint, resume_flag=True, experiment=options.experiment_name))
    elif options.load_run:
        parser.error("--load_run requires --resume or --checkpoint")
    return _run(_lab_scripts() / "train.py", args, **({"restore_checkpoint": checkpoint} if checkpoint else {}))


def play(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--homie_checkpoint")
    parser.add_argument("--level", type=int)
    parser.add_argument("--checkpoint", help="high-level checkpoint to play (default: latest run / latest model)")
    parser.add_argument("--load_run", help="run directory name or regex (default: latest)")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    if options.homie_checkpoint:
        args.append(f"env.checkpoint={Path(options.homie_checkpoint).resolve().as_posix()}")
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    args.extend(["--task", TASK_ID, "--experiment_name", options.experiment_name])
    if options.use_pretrained_checkpoint:
        args.append("--use_pretrained_checkpoint")
        return _run(_lab_scripts() / "play.py", args)
    model_pattern, run_pattern = _selection_patterns(options, args)
    checkpoint = _resolve_checkpoint(model_pattern, run_pattern, options.experiment_name)
    args.extend(_checkpoint_args(checkpoint, resume_flag=False))
    return _run(_lab_scripts() / "play.py", args, restore_checkpoint=checkpoint)


def _validate_experiment(name: str) -> None:
    """Keep V5 launchers from accidentally writing into the established V4 logs."""
    if name in ("homie_pushcart", "homie_pushcart_v5") or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("V5 needs a separate experiment name; use homie_pushcart_v51 or a new V5 experiment")
