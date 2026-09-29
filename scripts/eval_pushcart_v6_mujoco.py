"""Evaluate a V6 checkpoint on its only supported legacy four-sphere plant."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"HomieRL")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--payload", type=float, default=5.)
    parser.add_argument("--level", type=int, default=17)
    parser.add_argument("--command", type=float, nargs=3, default=[.3, 0., 0.])
    parser.add_argument("--output", type=Path, default=ROOT/"logs/bench/v6_mujoco")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    import numpy as np
    import torch
    from pushcart_bench.policies_v6 import V6Policy
    torch.set_num_threads(1)
    policy = V6Policy(args.checkpoint, args.level)
    policy.reset(payload_kg=args.payload, seed=0)
    samples = {key:[] for key in ("t","pose","velocity","command","pred_velocity","pred_hands","action","homie_command")}
    start = time.perf_counter()
    for k in range(args.steps):
        command = np.asarray(args.command)
        policy.step(command)
        r = policy.runner
        values = [("t",(k+1)*policy.dt),("pose",policy.pose()),("velocity",policy.velocity()),
                  ("command",command),("pred_velocity",r.pred_cart_velocity),("pred_hands",r.pred_hands),
                  ("action",r.hl_actions),("homie_command",r.cmd_homie)]
        for name,value in values:
            samples[name].append(np.asarray(value).copy())
        if policy.failure():
            break
    arrays = {key:np.asarray(value) for key,value in samples.items()}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("Non-finite V6 inference or physical sample")
    report = {"policy":policy.describe(), "steps":len(arrays["t"]), "requested_steps":args.steps,
              "payload_kg":args.payload,"command":args.command,"failure":policy.failure(),
              "velocity_rmse":np.sqrt(np.mean((arrays["velocity"]-arrays["command"])**2,axis=0)).tolist(),
              "wall_s":time.perf_counter()-start}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"samples.npz", **arrays)
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="policy"},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
