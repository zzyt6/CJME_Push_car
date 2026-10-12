"""V8 training commands, forwarded into NVIDIA's standard RSL-RL train entrypoint (rma_policy.py, final design):

    pushcart_v8.py train [--roa_from <roa model.pt>] [env.task.v8.arm_drive.enabled=true] ...   ROA teacher / continuation
    pushcart_v8.py adapt --checkpoint <roa model.pt>                                           RMA phase 2 (phi only)
    pushcart_v8.py adapt --student_rl --checkpoint <phase-2 model.pt>                         student RL (pi only)

New training uses the pinned V3.2 defaults (environment, runner, PPO values).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from homie_pushcart.training import _lab_scripts

from homie_pushcart_v7.paths import ROOT
from .task_config import task_cfg_from_run

TEACHER_ROA_TASK_ID = "Homie-G1-PushCart-V8-TeacherROA-Direct-v0"
EXPERIMENT = "homie_pushcart_v8"


def register_tasks():
    """Pre-Kit callback: register string factories without simulator imports."""
    from . import register_tasks as register_base_tasks
    return register_base_tasks()


def homie_velocity_file(checkpoint) -> Path | None:
    """The fine-tuned HOMIE velocity estimator paired with a checkpoint: homie_velocity_estimator_<N>.pt with N the
    global iteration (model number + the run's start_update); else the run's latest file; else (phase 2 / student RL,
    which do not train it) the estimator the run was started with."""
    checkpoint = Path(checkpoint)
    run = checkpoint.parent if checkpoint.suffix == ".pt" else checkpoint
    number = checkpoint.stem.split("_")[-1] if checkpoint.suffix == ".pt" else ""
    if number.isdigit():
        try:
            import yaml
            agent = yaml.load((run / "params/agent.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
            number = str(int(number) + int((agent.get("algorithm") or {}).get("start_update", 0) or 0))
        except (OSError, ValueError, TypeError):
            pass
    paired = run / f"homie_velocity_estimator_{number}.pt"
    if number.isdigit() and paired.is_file():
        return paired
    latest = run / "homie_velocity_estimator.pt"
    if latest.is_file():
        print(f"[pushcart-v8] WARNING: no {paired.name}; using the latest estimator {latest}", flush=True)
        return latest
    try:
        init = task_cfg_from_run(run, verbose=False).v8.homie_velocity.init_checkpoint
        if init and Path(init).is_file():
            return Path(init)
    except Exception:
        pass
    return None


def _hv_override(checkpoint, *, train: bool | None = None) -> list[str]:
    path = homie_velocity_file(checkpoint)
    out = []
    if path is not None:
        out.append(f"env.task.v8.homie_velocity.init_checkpoint='{path.resolve().as_posix()}'")
        print(f"[pushcart-v8] HOMIE velocity estimator: {path}", flush=True)
    if train is not None:
        out.append(f"env.task.v8.homie_velocity.train={'true' if train else 'false'}")
    return out


def _roa_source_overrides(checkpoint) -> list[str]:
    """Environment settings a ROA run was trained with that continuations, phase 2 and student RL must keep: the
    upper-body drive (V8.0 runs: implicit; the ROA task default is explicit since V8.1) and arm-drive randomisation."""
    import yaml
    v8 = task_cfg_from_run(checkpoint, verbose=False).v8
    agent = yaml.load((Path(checkpoint).parent / "params/agent.yaml").read_text(encoding="utf-8"),
                      Loader=yaml.FullLoader) or {}
    head = bool((agent.get("actor") or {}).get("velocity_head", False))          # V8.0 runs: no velocity head
    return [f"env.task.v8.upper_drive.mode={v8.upper_drive.mode}",
            f"env.task.v8.arm_drive.enabled={'true' if v8.arm_drive.enabled else 'false'}",
            f"agent.actor.velocity_head={'true' if head else 'false'}"]


def _run(entrypoint: Path, args: list[str]) -> int:
    command = [sys.executable, "-u", str(entrypoint), "--rl_library", "rsl_rl",
               "--external_callback", "homie_pushcart_v8.training.register_tasks", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(ROOT / "HomieRL"), env.get("PYTHONPATH", "")) if part)
    print("[pushcart-v8] NVIDIA entrypoint:", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def _validate_experiment(name):
    if (not name.startswith(EXPERIMENT) or Path(name).name != name
            or any(separator in name for separator in ("/", "\\"))):
        raise ValueError("V8 writes only to homie_pushcart_v8 or a homie_pushcart_v8_* experiment")


def _reject_alternate_profiles(argv):
    forbidden = {"--quality_profile", "--domain_randomization_profile", "--scene_version", "--scene_profile",
                 "--physics_backend", "--use_pretrained_checkpoint", "--homie_checkpoint"}
    for arg in argv:
        if arg.split("=", 1)[0] in forbidden:
            raise ValueError("V8 has one V3.2 physical/reward baseline and a fixed frozen HOMIE; "
                             f"alternate profile option {arg.split('=', 1)[0]} is not supported")


def _rma_checkpoint(path) -> Path:
    from .rma_policy import is_rma_run
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not is_rma_run(checkpoint):
        raise ValueError(f"{checkpoint} is not a V8 ROA run checkpoint")
    return checkpoint


def train(argv: list[str]) -> int:
    """ROA teacher from scratch, or (--roa_from) a continuation of a ROA run in a new run: weights, critic and the
    paired HOMIE velocity estimator from that checkpoint, phi fitted every update, lambda / H schedules from its global
    iteration."""
    _reject_alternate_profiles(argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--level", type=int, choices=range(18))
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--roa", action="store_true", help=argparse.SUPPRESS)     # accepted: the only V8 teacher
    parser.add_argument("--roa_from", help="continue a ROA run from this checkpoint in a new run")
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    if options.level is not None:
        args.append(f"env.curriculum_level={options.level}")
    if options.no_randomization:
        args.append("env.enable_randomization=false")
    args.extend(["--task", TEACHER_ROA_TASK_ID, "--experiment_name", options.experiment_name])
    if options.roa_from:
        import torch
        import yaml
        src = _rma_checkpoint(options.roa_from)
        it = int(torch.load(str(src), map_location="cpu", weights_only=False).get("iter", 0))
        src_agent = yaml.load((src.parent / "params/agent.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
        it += int((src_agent.get("algorithm") or {}).get("start_update", 0) or 0)      # global iteration
        args.extend([f"agent.algorithm.init_checkpoint='{src.as_posix()}'", f"agent.algorithm.start_update={it}",
                     "agent.algorithm.adapt_every_update=true"])
        args.extend(_hv_override(src))                # its fine-tuned HOMIE velocity estimator (still training)
        args.extend(o for o in _roa_source_overrides(src) if o not in args)
        print(f"[pushcart-v8] ROA continuation from {src} (iteration {it})", flush=True)
    return _run(_lab_scripts() / "train.py", args)


def adapt(argv: list[str]) -> int:
    """RMA phase 2 (default): train only the adaptation module on the student's own rollouts (every rollout acts on
    z_phi; base policy, encoder and critic frozen). --student_rl: PPO fine-tuning of the deployable network (phi and
    the input statistics frozen, privileged critic). Both run on the source run's environment (arm randomisation kept)
    and its HOMIE velocity estimator (frozen)."""
    _reject_alternate_profiles(argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True, help="ROA teacher (phase 2) or phase-2 (student RL) checkpoint")
    parser.add_argument("--student_rl", action="store_true",
                        help="PPO fine-tuning of the deployable student instead of phase 2")
    parser.add_argument("--level", type=int, choices=range(18), default=17)
    parser.add_argument("--experiment_name", default=EXPERIMENT)
    options, args = parser.parse_known_args(argv)
    _validate_experiment(options.experiment_name)
    checkpoint = _rma_checkpoint(options.checkpoint)
    args.extend([f"env.curriculum_level={options.level}", "--task", TEACHER_ROA_TASK_ID,
                 "--experiment_name", options.experiment_name])
    args.extend(_hv_override(checkpoint, train=False))
    args.extend(o for o in _roa_source_overrides(checkpoint) if o not in args)
    lr = task_cfg_from_run(checkpoint, verbose=False).v8.leg_residual
    args.extend([f"env.task.v8.leg_residual.mode={lr.mode}", f"env.task.v8.leg_residual.range={lr.range}",
                 f"env.task.v8.leg_residual.clip={lr.clip}", f"env.task.v8.leg_residual.filter_tau_s={lr.filter_tau_s}",
                 f"agent.algorithm.init_checkpoint='{checkpoint.as_posix()}'"])
    if options.student_rl:
        args.extend(["agent.algorithm.student_rl=true", "agent.algorithm.learning_rate=0.0001",
                     "agent.algorithm.entropy_coef=0.0"])
        print(f"[pushcart-v8] student RL from {checkpoint}", flush=True)
    else:
        args.append("agent.algorithm.adapt_only=true")
        print(f"[pushcart-v8] RMA phase 2 (adaptation module) from {checkpoint}", flush=True)
    return _run(_lab_scripts() / "train.py", args)
