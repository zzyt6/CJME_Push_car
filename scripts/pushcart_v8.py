"""V8: high-level teacher-student on the V7.1 plant with a gated HOMIE leg residual.

    pushcart_v8.py train|play [...]                 teacher phase (PPO), NVIDIA RSL-RL entrypoints (assets: the V7 build)
    pushcart_v8.py distill --teacher <model.pt>     student phase (DAgger distillation)
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "play", "distill"))
    args, extra = parser.parse_known_args()
    from homie_pushcart_v8 import training
    return getattr(training, args.command)(extra)


if __name__ == "__main__":
    raise SystemExit(main())
