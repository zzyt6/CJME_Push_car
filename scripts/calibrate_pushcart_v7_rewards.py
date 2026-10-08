"""Measure the per-second contribution of every V7 reward term at a fixed difficulty (for weight calibration).

The policy is HOMIE alone (zero high-level action) or small random corrections; the numbers show the scale each
term reaches under realistic loads, not a trained policy's values.

    E:\\envs\\RLisaac601\\python.exe scripts/calibrate_pushcart_v7_rewards.py --headless --difficulty 0.7
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=750)
    p.add_argument("--difficulty", type=float, default=0.7)
    p.add_argument("--action_std", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.curriculum import LEVELS
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = a.num_envs
        task = PushCartTaskCfg()
        task.curriculum_v7.enabled = False
        env = PushCartEnv(cfg, task, ROOT, a.device, True, level=0)
        env.cur.d[:] = a.difficulty
        env._refresh_global_level()
        env.reset_idx(torch.arange(a.num_envs, device=a.device))
        env.reset()
        sums = {n: 0. for n in env.reward_names}
        funcs = dict(zip(env.reward_names, env.reward_functions))
        extra = {"internal_frac": 0., "pivot_frac_max": 0., "terminated": 0., "grasp_lost": 0.}
        for _ in range(a.steps):
            env.step(a.action_std * torch.randn(a.num_envs, 18, device=a.device))
            for n, fn in funcs.items():                      # recompute on the post-step state (same as the reward)
                sums[n] += float((fn() * env.reward_scales[n]).mean()) / env.dt
            extra["internal_frac"] += float(env.ts.internal_frac.mean())
            extra["pivot_frac_max"] = max(extra["pivot_frac_max"], float(env.ts.pivot.max()))
            extra["terminated"] += float(env.terminated_buf.float().mean())
            extra["grasp_lost"] += float(env.term_reason_buf["grasp_overload"].float().mean())
        per_s = {n: v / a.steps for n, v in sums.items()}
        groups = {"tracking": ("track_cart_vx", "track_cart_vy", "track_cart_wz", "track_progress", "cart_yaw_lead"),
                  "interaction": ("grasp_internal", "grasp_margin", "grasp_slip", "grasp_drift", "force_rate",
                                  "wrist_saturation", "upper_torque"),
                  "smoothness": ("cart_lin_ripple", "cart_yaw_ripple", "cart_response_error", "cart_wrench_osc")}
        summary = {g: sum(per_s.get(n, 0.) for n in names) for g, names in groups.items()}
        summary["other"] = sum(v for n, v in per_s.items() if not any(n in names for names in groups.values()))
        rec = {"difficulty": a.difficulty, "level": a.difficulty * LEVELS, "action_std": a.action_std,
               "per_second": per_s, "groups": summary,
               "internal_frac_mean": extra["internal_frac"] / a.steps, "pivot_frac_max": extra["pivot_frac_max"],
               "termination_rate_per_s": extra["terminated"] / a.steps / env.dt,
               "grasp_loss_rate_per_s": extra["grasp_lost"] / a.steps / env.dt}
        for n, v in sorted(per_s.items(), key=lambda kv: kv[1]):
            print(f"[calib] d={a.difficulty:.2f} {n:24s} {v:+.4f} /s", flush=True)
        print("[calib] groups", json.dumps({k: round(v, 4) for k, v in summary.items()}), flush=True)
        print("[calib] extra", json.dumps({k: round(v, 4) for k, v in rec.items() if k not in ("per_second", "groups")}), flush=True)
        out = ROOT / "logs/validation/v7_reward_calibration"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"d{a.difficulty:.2f}_a{a.action_std:.2f}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
