"""What does the V7 reward think of V6's driving? (2026-10-08)

One Isaac launch on the V7 training plant at full difficulty, training command distribution; half the environments
run V6, half V7 (deterministic). Every V7 reward term is accumulated per environment (episode long enough that no
time-out reset happens) and reported per second, with tracking RMSE and hand-load / grasp statistics per policy.
If V6's more precise driving pays clearly more in force-related terms than it gains in tracking, the reward is what
pushes V7 to drive "softer".

    E:\\envs\\RLisaac601\\python.exe scripts/reward_audit_v6_v7.py --v6 <model.pt> --v7 <model.pt> --headless [--payload 25]
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser()
    p.add_argument("--v6", required=True)
    p.add_argument("--v7", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--payload", type=float, default=None, help="fixed payload (default: training distribution)")
    p.add_argument("--difficulty", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv
        from homie_pushcart_v6.hl_policy import HighLevelPolicy as P6
        from homie_pushcart_v7.hl_policy import HighLevelPolicy as P7
        dev, N = a.device, a.num_envs
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = N
        task = PushCartTaskCfg(); task.curriculum_v7.enabled = False
        task.env.episode_length_s = a.seconds + 30.0
        env = PushCartEnv(cfg, task, ROOT, dev, True, level=0)
        env.cur.d[:] = a.difficulty
        env._refresh_global_level()
        env.fixed_payload_kg = a.payload
        env.reset_idx(torch.arange(N, device=dev)); env.reset()
        pols = [P6(Path(a.v6), device=dev), P7(Path(a.v7), device=dev)]
        owner = torch.arange(N, device=dev) % 2
        for k in env.episode_sums:
            env.episode_sums[k][:] = 0.
        steps = int(a.seconds / env.dt)
        err2 = torch.zeros(N, 3, device=dev); cnt = torch.zeros(N, device=dev)
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        hand = {k: torch.zeros(N, device=dev) for k in ("abs_fx", "abs_fy", "abs_mz", "sliding", "arm_action", "sat")}
        for t in range(steps):
            obs = env.observation_dict()
            with torch.inference_mode():
                act = torch.zeros(N, 18, device=dev)
                for k, pol in enumerate(pols):
                    act[owner == k] = pol(obs["policy"], obs["estimator"])[owner == k]
            env.step(act)
            alive &= ~env.reset_buf
            if t < 50:
                continue
            e = torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1) - env.cmd_ref
            err2 += e.square() * alive[:, None]; cnt += alive.float()
            f = env._hand_cart                                   # (N, 2, 6) hand loads, cart axes
            hand["abs_fx"] += f[:, :, 0].abs().sum(1) * alive
            hand["abs_fy"] += f[:, :, 1].abs().sum(1) * alive
            hand["abs_mz"] += f[:, :, 5].abs().sum(1) * alive
            hand["sliding"] += env.slide_sliding.float().mean(1) * alive
            hand["arm_action"] += act[:, 3:].abs().mean(1) * alive
            hand["sat"] += (act.abs() >= 0.99).float().mean(1) * alive
        dt = env.dt
        out = {"payload": a.payload, "difficulty": a.difficulty, "seconds": a.seconds}
        for k, name in enumerate(("V6", "V7")):
            m = (owner == k) & alive
            per_s = {key: float((v[m] / steps / dt).mean()) for key, v in env.episode_sums.items()}
            out[name] = {
                "envs_kept": int(m.sum()), "envs_terminated": int(((owner == k) & ~alive).sum()),
                "tracking_rmse_vx_vy_wz": (err2[m].sum(0) / cnt[m].sum()).sqrt().tolist(),
                "reward_per_s_total": sum(per_s.values()), "reward_per_s": per_s,
                "hand_mean": {key: float((v[m] / cnt[m]).mean()) for key, v in hand.items()},
            }
        print("[audit]", json.dumps(out), flush=True)
        o = ROOT / "logs/validation/reward_audit"; o.mkdir(parents=True, exist_ok=True)
        tag = f"payload{a.payload:g}" if a.payload is not None else "train_dist"
        (o / f"audit_{tag}_d{a.difficulty:g}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush()
        raise
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
