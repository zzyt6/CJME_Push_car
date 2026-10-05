"""Forward the project train command into NVIDIA's unmodified train.py."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def train(argv):
    import isaaclab

    lab_root = Path(os.environ.get("ISAACLAB_PATH", Path(isaaclab.__file__).resolve().parents[3]))
    entrypoint = lab_root / "scripts/reinforcement_learning/train.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Official Isaac Lab entrypoint not found: {entrypoint}")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task", default="Homie-G1-HandForce-Direct-v0")
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--rl_device")
    parser.add_argument("--sim_device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--experiment_name", default="homie_g1_hand_force")
    options, args = parser.parse_known_args(argv)
    if options.no_randomization:
        args.append("env.enable_randomization=false")
    if options.sim_device:
        args.extend(["--device", options.sim_device])
    if options.rl_device:
        args.append(f"agent.device={options.rl_device}")
    task = "Homie-G1-HandForce-Direct-v0" if options.task == "g1" else options.task
    args.extend(["--task", task, "--experiment_name", options.experiment_name])
    if options.checkpoint:
        checkpoint = options.checkpoint
        source = Path(checkpoint).expanduser()
        if source.is_file():
            source = source.resolve()
            log_root = root / "logs/rsl_rl" / options.experiment_name
            if source.parent.parent == log_root.resolve():
                run = source.parent.name
                checkpoint = source.name
            else:
                # This Lab checkout's official resolver only searches run
                # directories. Import older/elsewhere checkpoints as an explicit
                # run so its unmodified --load_run/--checkpoint API can find it.
                digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
                run = f"0000_imported_{digest}"
                dest_dir = log_root / run
                dest_dir.mkdir(parents=True, exist_ok=True)
                destination = dest_dir / "model_imported.pt"
                if not destination.exists():
                    shutil.copy2(source, destination)
                checkpoint = destination.name
            args.extend(["--load_run", re.escape(run)])
        elif "/" in checkpoint or "\\" in checkpoint:
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        args.extend(["--checkpoint", checkpoint])
        if "--resume" not in args:
            args.append("--resume")
    command = [sys.executable, "-u", str(entrypoint), "--rl_library", "rsl_rl",
               "--external_callback", "homie_hand_force.register_tasks", *args]
    print("[HOMIE] NVIDIA training entrypoint:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=root)
    return result.returncode
