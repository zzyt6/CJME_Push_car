"""Fixed-scenario evaluation of a high-level checkpoint (or the passthrough baseline) in ``PushCartEnv``.

Scenarios drive the task command, payload and caster heading directly (curriculum frozen at ``--level``)
and report cart-velocity tracking RMSE, grasp-force percentiles per direction, wrist saturation,
termination fractions and robot-handle distance. This is the acceptance gate of phases B4 / B5 and the
Isaac-side reference for the MuJoCo sim2sim comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .paths import ROOT

# name, command (vx, vy, wz) in the cart frame, payload kg, caster heading (None = level default), seconds,
# optional command switch (t_switch_s, new command)
SCENARIOS = {
    "hold": ((0.0, 0.0, 0.0), 0.0, 0.0, 8.0, None),
    "push_03": ((0.3, 0.0, 0.0), 0.0, 0.0, 8.0, None),
    "push_06": ((0.6, 0.0, 0.0), 0.0, 0.0, 8.0, None),
    "backward_03": ((-0.3, 0.0, 0.0), 0.0, 0.0, 8.0, None),
    "turn_left": ((0.3, 0.0, 0.3), 0.0, 0.0, 8.0, None),
    "turn_right": ((0.3, 0.0, -0.3), 0.0, 0.0, 8.0, None),
    "stop": ((0.5, 0.0, 0.0), 0.0, 0.0, 8.0, (4.0, (0.0, 0.0, 0.0))),
    "heavy_25": ((0.3, 0.0, 0.0), 25.0, 0.0, 8.0, None),
    "caster_reverse": ((0.3, 0.0, 0.0), 5.0, math.pi, 8.0, None),
}


def load_actor(checkpoint: Path, env, device: str):
    """Rebuild the RSL-RL 5 actor from a training checkpoint for deterministic inference."""
    from .hl_policy import HighLevelPolicy
    hl = HighLevelPolicy(checkpoint, env.num_hl_obs, env.num_hl_critic_obs, env.num_hl_actions, device)
    return hl, hl.iteration


def run_eval(args, launcher, root: Path = ROOT) -> dict:
    from .env_smoke import make_configs
    from .task_env import PushCartEnv, TERMINATION_REASONS

    device = args.device
    headless = bool(getattr(launcher, "_headless", True))
    cfg, task_cfg = make_configs(args.num_envs, not args.no_randomization, args.level, episode_length_s=60.0)
    task_cfg.curriculum.enabled = False
    env = PushCartEnv(cfg, task_cfg, root, device, headless, checkpoint=args.homie_checkpoint)
    env.reset()
    if args.checkpoint:
        policy, iteration = load_actor(Path(args.checkpoint), env, device)
        tag = Path(args.checkpoint).stem
    else:
        def policy(hl_obs, critic_obs):
            return torch.zeros(hl_obs.shape[0], env.num_hl_actions, device=device)
        iteration, tag = None, "passthrough"
    names = args.scenarios or list(SCENARIOS)
    results, start = {}, time.perf_counter()
    for name in names:
        cmd, payload, caster, seconds, switch = SCENARIOS[name]
        env.fixed_command, env.fixed_payload_kg, env.fixed_caster_angle = list(cmd), payload, caster
        env.term_counts.zero_()
        env.episodes_finished = 0
        obs, critic = env.reset()
        steps = int(seconds / env.dt)
        skip = int(args.transient_s / env.dt)
        vx_err, wz_err, vy, dist, pull, up, tang, push, wrist_sat, base_speed, cart_speed, ep_len = ([] for _ in range(12))
        done_any = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        for step in range(steps):
            if not launcher.app.is_running():
                raise RuntimeError("Simulation closed during evaluation")
            if switch is not None and step == int(switch[0] / env.dt):
                env.fixed_command = list(switch[1])
                env.cmd_task[:] = torch.tensor(switch[1], device=device)
            obs, critic, rew, done, extras = env.step(policy(obs, critic))
            done_any |= extras["terminated"]
            if step < skip or (switch is not None and abs(step - int(switch[0] / env.dt)) < skip):
                continue
            ts = env.ts
            vx_err.append((env.cmd_task[:, 0] - ts.cart_v[:, 0]).cpu().numpy())
            wz_err.append((env.cmd_task[:, 2] - ts.cart_wz).cpu().numpy())
            vy.append(ts.cart_v[:, 1].abs().cpu().numpy())
            dist.append(ts.dist.cpu().numpy())
            f = ts.force_c
            pull.append((-f[..., 0]).clamp_min(0).max(dim=1).values.cpu().numpy())
            push.append(f[..., 0].clamp_min(0).max(dim=1).values.cpu().numpy())
            up.append(f[..., 2].clamp_min(0).max(dim=1).values.cpu().numpy())
            tang.append(f[..., 1].abs().max(dim=1).values.cpu().numpy())
            tau = ts.upper_tau[:, env.wrist_ids].abs()
            wrist_sat.append((tau > 0.8 * task_cfg.rewards.wrist_tau_max_nm).float().mean(dim=1).cpu().numpy())
            base_speed.append(env.base_lin_vel[:, :2].norm(dim=-1).cpu().numpy())
            cart_speed.append(ts.cart_speed.cpu().numpy())
        summary = env.termination_summary()
        counts = summary["counts"]
        vx_err, wz_err = np.concatenate(vx_err), np.concatenate(wz_err)
        res = {
            "command": list(cmd), "payload_kg": payload, "seconds": seconds,
            "cart_vx_rmse": float(np.sqrt(np.mean(vx_err ** 2))), "cart_vx_bias": float(np.mean(vx_err)),
            "cart_wz_rmse": float(np.sqrt(np.mean(wz_err ** 2))),
            "cart_vy_abs_mean": float(np.mean(np.concatenate(vy))),
            "cart_speed_mean": float(np.mean(np.concatenate(cart_speed))),
            "base_speed_mean": float(np.mean(np.concatenate(base_speed))),
            "robot_bar_distance_mean": float(np.mean(np.concatenate(dist))),
            "grasp_p95_N": {"push": float(np.percentile(np.concatenate(push), 95)), "pull": float(np.percentile(np.concatenate(pull), 95)),
                            "up": float(np.percentile(np.concatenate(up), 95)), "tangential": float(np.percentile(np.concatenate(tang), 95))},
            "wrist_saturation_frac": float(np.mean(np.concatenate(wrist_sat))),
            "terminated_frac": float(done_any.float().mean().item()),
            "termination_counts": {k: int(v) for k, v in counts.items() if k != "time_out"},
        }
        results[name] = res
        print(f"[eval] {name:15s} vx rmse {res['cart_vx_rmse']:.3f} (bias {res['cart_vx_bias']:+.3f}) wz rmse {res['cart_wz_rmse']:.3f} "
              f"| dist {res['robot_bar_distance_mean']:.2f} | p95 push/pull/up/tan {res['grasp_p95_N']['push']:.0f}/{res['grasp_p95_N']['pull']:.0f}/"
              f"{res['grasp_p95_N']['up']:.0f}/{res['grasp_p95_N']['tangential']:.0f} N | wrist sat {res['wrist_saturation_frac']:.2f} "
              f"| terminated {res['terminated_frac']:.2f} {res['termination_counts']}", flush=True)
    out = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None, "iteration": iteration, "tag": tag,
        "homie_checkpoint": str(env.policy.checkpoint), "level": env.curriculum.level, "num_envs": args.num_envs,
        "randomization": not args.no_randomization, "transient_s": args.transient_s,
        "scenarios": results, "elapsed_s": time.perf_counter() - start,
    }
    path = root / f"logs/pushcart_eval_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {path}", flush=True)
    return out
