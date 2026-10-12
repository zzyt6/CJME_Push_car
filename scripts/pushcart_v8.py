"""V8: ROA teacher / RMA student on the V7.1 plant with a gated HOMIE leg residual (NVIDIA RSL-RL entrypoint).

    pushcart_v8.py train [...]                                 ROA teacher from scratch
    pushcart_v8.py train --roa_from <roa model.pt> [...]      continue a ROA run (new run, global schedules)
    pushcart_v8.py adapt --checkpoint <roa model.pt>           RMA phase 2 (adaptation module only)
    pushcart_v8.py adapt --student_rl --checkpoint <p2 .pt>    student RL (base policy, phi frozen)
The final V8 sequence is in docs/push_cart_v8_teacher_student_plan.md (section 11).
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "adapt"))
    args, extra = parser.parse_known_args()
    from homie_pushcart_v8 import training
    return getattr(training, args.command)(extra)


if __name__ == "__main__":
    raise SystemExit(main())
