"""Forward `pushcart.py train / play` into NVIDIA's unmodified RSL-RL entrypoints.

Same approach as ``homie_isaaclab.training``: a clean child interpreter runs
``scripts/reinforcement_learning/{train,play}.py --rl_library rsl_rl`` with our task registered through
``--external_callback homie_pushcart.register_tasks``. Everything after the recognised options is forwarded
verbatim (``--num_envs``, ``--max_iterations``, ``--headless``, Hydra overrides such as
``env.task.curriculum.init_level=4`` or ``agent.max_iterations=...``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from .paths import ROOT

TASK_ID = "Homie-G1-PushCart-Direct-v0"
EXPERIMENT = "homie_pushcart"


def _lab_scripts() -> Path:
    import isaaclab
    lab_root = Path(os.environ.get("ISAACLAB_PATH", Path(isaaclab.__file__).resolve().parents[3]))
    scripts = lab_root / "scripts/reinforcement_learning"
    if not (scripts / "train.py").is_file():
        raise FileNotFoundError(f"Official Isaac Lab entrypoints not found under {scripts}")
    return scripts


def _run(entrypoint: Path, args: list[str]) -> int:
    command = [sys.executable, "-u", str(entrypoint), "--rl_library", "rsl_rl",
               "--external_callback", "homie_pushcart.register_tasks", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT / "HomieRL"), env.get("PYTHONPATH", "")) if p)
    print("[pushcart] NVIDIA entrypoint:", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def _checkpoint_args(checkpoint: str | None, resume_flag: bool) -> list[str]:
    """Translate a checkpoint path into the official --load_run / --checkpoint pair (run dirs only)."""
    if not checkpoint:
        return []
    source = Path(checkpoint).expanduser()
    log_root = ROOT / "logs/rsl_rl" / EXPERIMENT
    args = []
    if source.is_file():
        source = source.resolve()
        if source.parent.parent != log_root.resolve():
            raise FileNotFoundError(f"checkpoint must live in a run directory under {log_root}: {source}")
        args += ["--load_run", re.escape(source.parent.name), "--checkpoint", source.name]
    else:
        args += ["--checkpoint", str(checkpoint)]
    if resume_flag:
        args.append("--resume")
    return args


def train(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--homie_checkpoint", help="frozen HOMIE model_*.pt (default: latest in the HOMIE run dir)")
    parser.add_argument("--level", type=int, help="curriculum start level")
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--checkpoint", help="resume from this high-level checkpoint (inside logs/rsl_rl/homie_pushcart)")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    if options.homie_checkpoint:
        args.append(f"env.checkpoint={Path(options.homie_checkpoint).resolve().as_posix()}")
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    if options.no_randomization:
        args.append("env.enable_randomization=false")
    args.extend(["--task", TASK_ID, "--experiment_name", options.experiment_name])
    args.extend(_checkpoint_args(options.checkpoint, resume_flag=True))
    return _run(_lab_scripts() / "train.py", args)


def play(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--homie_checkpoint")
    parser.add_argument("--level", type=int)
    parser.add_argument("--checkpoint", help="high-level checkpoint to play (default: latest run / latest model)")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    if options.homie_checkpoint:
        args.append(f"env.checkpoint={Path(options.homie_checkpoint).resolve().as_posix()}")
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    args.extend(["--task", TASK_ID, "--experiment_name", options.experiment_name])
    args.extend(_checkpoint_args(options.checkpoint, resume_flag=False))
    return _run(_lab_scripts() / "play.py", args)
