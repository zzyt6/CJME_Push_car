"""Isolated V5.3 training/play entrypoint. Existing assets are read-only inputs."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "mujoco":
        from eval_pushcart_v53_mujoco import main as run_mujoco
        return run_mujoco(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("train", "play", "mujoco"))
    args, extra = parser.parse_known_args()
    from homie_pushcart_v53 import training
    return getattr(training, args.command)(extra)


if __name__ == "__main__":
    raise SystemExit(main())
