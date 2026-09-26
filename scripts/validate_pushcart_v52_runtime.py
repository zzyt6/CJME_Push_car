"""Bounded V5.2 runtime check: per-hand interval labels, static prior and estimator targets on real Isaac physics.

Zero actor, fixed command. Reports label/prior statistics and the prior-vs-label correlation per hand axis
(a sanity check of the (J^T)^+ tau sign/axes, not an accuracy claim).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--level", type=int, default=12)
parser.add_argument("--output", type=Path, default=ROOT / "logs/bench/v52_runtime/runtime.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import torch
    from homie_pushcart_v52.env_smoke import make_configs
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.observation import FRAME_DIM, frame_layout

    torch.manual_seed(52)
    cfg, task = make_configs(args.num_envs, True, args.level, 20.0)
    task.curriculum.enabled = False
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level)
    env.fixed_command = [.3, 0., .1]
    env.reset()
    prior_slice = next(f for f in frame_layout() if f["name"] == "static_hand_wrench")
    lo, hi = prior_slice["start"], prior_slice["start"] + prior_slice["dim"]
    labels, priors, counts = [], [], []
    for step in range(args.steps):
        if step == 100:
            env.inject_cart_wrench([20., 0., 0.], [0., 0., 1.], .2)
        obs, critic, reward, done, extra = env.step(torch.zeros(args.num_envs, 18, device=args.device))
        assert obs.shape == (args.num_envs, FRAME_DIM * 20) and env.interaction_target.shape == (args.num_envs, 13)
        assert torch.isfinite(obs).all() and torch.isfinite(env.interaction_target).all()
        valid = env.interaction_target[:, 12] > .5
        labels.append(env.interaction_target[valid, :12].cpu())
        priors.append(obs[valid, -FRAME_DIM:][:, lo:hi].cpu())
        counts.append(env.hand_wrench_count[valid].cpu())
    y, p, c = torch.cat(labels), torch.cat(priors), torch.cat(counts)
    names = [f"{s}_{a}" for s in "LR" for a in ("fx", "fy", "fz", "mx", "my", "mz")]
    corr = [float(torch.corrcoef(torch.stack((p[:, i], y[:, i])))[0, 1]) for i in range(12)]
    report = dict(result="PASS_EXECUTION", num_envs=args.num_envs, steps=args.steps, level=args.level,
                  valid_samples=int(len(y)), valid_frac=float(len(y) / (args.num_envs * args.steps)),
                  substeps_per_label_mean=float(c.mean()),
                  label_std={n: float(v) for n, v in zip(names, y.std(0))},
                  prior_std={n: float(v) for n, v in zip(names, p.std(0))},
                  prior_label_corr={n: v for n, v in zip(names, corr)},
                  note="Runtime execution check with a zero actor; correlations only sanity-check prior axes/sign.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[v52-runtime]", json.dumps(report), flush=True)
finally:
    launcher.app.close()
