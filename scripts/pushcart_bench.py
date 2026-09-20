"""Push-cart control benchmark entry point (MuJoCo + viser, CPU): compare controllers on the same paths / steps / conditions.

    pushcart_bench.py path   --policies passthrough two_stage:latest --view   # large S, live payload + commanded vx
    pushcart_bench.py step   --policies passthrough two_stage:latest [--steps vx_0_to_03 ...] [--payloads 0 12.5 25]
    pushcart_bench.py robust --policies passthrough two_stage:latest [--factors payload_kg kick ...]
    pushcart_bench.py report --out_dir logs/bench/<dir>

Policy specs: passthrough | robot_only | two_stage[:<model_N.pt>|:latest] | plugin:<module>:<Class>[:<arg>]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL" / "rsl_rl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _common(sp):
    sp.add_argument("--policies", nargs="+", default=["passthrough", "two_stage:latest"],
                    help="passthrough | robot_only | two_stage[:ckpt|latest] | plugin:module:Class")
    sp.add_argument("--level", type=int, help="curriculum level of the two-stage policy (correction scale); default from yaml")
    sp.add_argument("--homie_checkpoint", type=Path, help="HOMIE model_*.pt (default: the frozen model_60000)")
    sp.add_argument("--config", type=Path, help="benchmark.yaml (default: pushcart_bench/config/benchmark.yaml)")
    sp.add_argument("--out_dir", type=Path, help="output directory (default: logs/bench/<timestamp>_<kind>)")
    sp.add_argument("--view", action="store_true", help="viser live view (http://localhost:<port>)")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--speed", type=float, default=1.0, help="with --view: real-time multiplier (0 = as fast as possible)")
    sp.add_argument("--caster_deg", type=float, help="initial caster angle [deg]; default from yaml")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="push-cart control benchmark (MuJoCo + viser)")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("path", help="experiment 1: follow the reference paths to the goal with the shared PID")
    _common(sp)
    sp.add_argument("--paths", nargs="*", help="subset of paths from the yaml (default: all)")
    sp.add_argument("--mirror", action="store_true", help="also run the mirrored (right-turning) copies listed in yaml `mirror`")
    sp.add_argument("--seeds", nargs="*", type=int)
    sp.add_argument("--payload", type=float, help="payload [kg]; default from yaml")
    sp.add_argument("--v_nom", type=float, help="commanded path-follow speed [m/s]; default from yaml (0.5)")
    sp.add_argument("--cruise", action="store_true",
                    help="hold v_nom through curves (no curvature / remaining slowdown). --view turns this on by default")
    sp.add_argument("--mode", choices=["car", "holonomic"], help="follower mode; default from yaml")
    sp.add_argument("--gain_sweep", action="store_true", help="run the yaml `gain_sweep` grid for every policy")

    sp = sub.add_parser("step", help="experiment 2: open-loop velocity steps, rise / settling / overshoot / steady state")
    _common(sp)
    sp.add_argument("--steps", nargs="*", help="subset of step names from the yaml (default: all)")
    sp.add_argument("--payloads", nargs="*", type=float)

    sp = sub.add_parser("robust", help="experiment 3: robustness sweep (payload, casters, friction, COM, kicks, wheel stall)")
    _common(sp)
    sp.add_argument("--path", help="path used for the sweep; default from yaml `robust.path`")
    sp.add_argument("--factors", nargs="*", help="subset of factors (default: all in yaml)")
    sp.add_argument("--seeds", nargs="*", type=int)

    sp = sub.add_parser("report", help="regenerate report.md + figures from an output directory")
    sp.add_argument("--out_dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "report":
        from pushcart_bench.report import write_report
        write_report(args.out_dir)
        return 0
    from pushcart_bench.config_io import load_config
    from pushcart_bench import experiments
    cfg = load_config(args.config)
    out = {"path": experiments.exp_path, "step": experiments.exp_step, "robust": experiments.exp_robust}[args.command](args, cfg)
    print(f"[bench] done -> {out}", flush=True)
    if getattr(args, "view", False) and args.command != "path":
        print("[bench] viewer stays open; Ctrl+C to exit", flush=True)
        import time
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
