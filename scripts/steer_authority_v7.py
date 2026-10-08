"""Coupled robot + cart yaw authority, MuJoCo vs Isaac/PhysX, on the V7 plant (2026-10-08).

Passthrough HOMIE (zero high-level action: arms held at the grasp pose) receives a constant task command
(vx, 0, wz); after a 4 s settle the cart's mean yaw rate / forward speed over 6 s are reported. The same command in
both backends must give the same cart motion if the plants agree; a policy trained in one backend relies on that.

    E:\\envs\\RLisaac601\\python.exe scripts/steer_authority_v7.py mujoco
    E:\\envs\\RLisaac601\\python.exe scripts/steer_authority_v7.py physx --headless
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
OUT = ROOT / "logs/validation/v7_steer_authority"
CASES = [(0.5, 0.0), (0.5, 0.4), (0.3, 0.0)]
PAYLOADS = (0.0, 12.5)
SETTLE_S, MEASURE_S = 4.0, 6.0


def run_mujoco() -> dict:
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.policies_cross import CrossPolicy
    pol = CrossPolicy(None, "v7")
    out = {}
    for payload in PAYLOADS:
        for vx, wz in CASES:
            pol.reset(payload_kg=payload)
            rows = []
            for k in range(int((SETTLE_S + MEASURE_S) / pol.dt)):
                pol.step(np.array([vx, 0.0, wz]))
                if k * pol.dt >= SETTLE_S:
                    v = pol.velocity()
                    e = pol.runner.vel_est - np.asarray(pol.env.base_lin_vel)
                    rows.append((v[0], v[1], v[2], pol.env.base_ang_vel[2], *e))
            a = np.array(rows)
            key = f"{payload:g}kg_vx{vx}_wz{wz}"
            out[key] = {"cart_vx": float(a[:, 0].mean()), "cart_vy": float(a[:, 1].mean()),
                        "cart_wz": float(a[:, 2].mean()), "pelvis_wz": float(a[:, 3].mean()), "fallen": pol.failure(),
                        "him_err_rms": np.sqrt((a[:, 4:7] ** 2).mean(0)).round(4).tolist(),
                        "him_err_bias": a[:, 4:7].mean(0).round(4).tolist()}
            print("[mujoco]", key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items()}, flush=True)
    return out


def run_physx(device: str, num_envs: int) -> dict:
    import torch
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v7.baseline import apply_config, homie_baseline
    from homie_pushcart_v7.task_config import PushCartTaskCfg
    from homie_pushcart_v7.task_env import PushCartEnv
    cfg = G1RoughCfg()
    apply_config(cfg, homie_baseline())
    cfg.env.num_envs = num_envs
    task = PushCartTaskCfg()
    task.curriculum_v7.enabled = False
    task.env.episode_length_s = SETTLE_S + MEASURE_S + 5.0
    env = PushCartEnv(cfg, task, ROOT, device, True, level=0)
    env.cur.d[:] = 0.0                          # physics within 25 % of nominal, no noise / disturbances
    env._refresh_global_level()
    env.fixed_caster_angle = 0.0
    zero = torch.zeros(num_envs, 18, device=device)
    out = {}
    for payload in PAYLOADS:
        for vx, wz in CASES:
            env.fixed_payload_kg = payload
            env.fixed_command = [vx, 0.0, wz]
            env.reset_idx(torch.arange(num_envs, device=device))
            env.reset()
            rows, fell = [], torch.zeros(num_envs, dtype=torch.bool, device=device)
            for k in range(int((SETTLE_S + MEASURE_S) / env.dt)):
                env.step(zero)
                fell |= env.reset_buf
                if k * env.dt >= SETTLE_S:
                    rows.append(torch.cat((torch.stack((env.ts.cart_v[:, 0], env.ts.cart_v[:, 1], env.ts.cart_wz,
                                                        env.base_ang_vel[:, 2]), -1), env.vel_est - env.base_lin_vel), -1))
            a = torch.stack(rows)[:, ~fell]            # (T, envs, 4)
            m = a.mean(0)
            key = f"{payload:g}kg_vx{vx}_wz{wz}"
            out[key] = {"cart_vx": float(m[:, 0].mean()), "cart_vy": float(m[:, 1].mean()),
                        "cart_wz": float(m[:, 2].mean()), "cart_wz_env_p10_p90": torch.quantile(m[:, 2], torch.tensor([.1, .9], device=device)).tolist(),
                        "pelvis_wz": float(m[:, 3].mean()), "envs_reset": int(fell.sum()),
                        "him_err_rms": a[..., 4:7].square().mean((0, 1)).sqrt().tolist(),
                        "him_err_bias": a[..., 4:7].mean((0, 1)).tolist()}
            print("[physx]", key, out[key], flush=True)
    env.sim.clear_instance()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backend", choices=("mujoco", "physx"))
    p.add_argument("--num_envs", type=int, default=32)
    if "physx" in (argv or sys.argv[1:]):
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.backend == "mujoco":
        res = run_mujoco()
    else:
        from isaaclab.app import AppLauncher
        app = AppLauncher(a)
        try:
            res = run_physx(a.device, a.num_envs)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            if "res" in locals():
                (OUT / "physx.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
            app.app.close()
    (OUT / f"{a.backend}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
