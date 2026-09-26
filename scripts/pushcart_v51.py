"""Isolated V5.1 training/play entrypoint. Existing assets are read-only inputs."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("train", "play"))
    args, extra = parser.parse_known_args()
    from homie_pushcart_v51 import training
    return getattr(training, args.command)(extra)


if __name__ == "__main__":
    raise SystemExit(main())
