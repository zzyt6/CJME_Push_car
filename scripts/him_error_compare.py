"""HOMIE HIM velocity-estimate error, MuJoCo vs Isaac, robot holding the V7 cart, with / without a high-level policy.

Same conditions in both backends: V7 plant, payload, constant task commands (after a 4 s settle, 6 s measured),
driver = none (passthrough HOMIE) / V6 / V7. Error = HIM estimate - pelvis body-frame velocity; reported as RMS / bias,
plus the pelvis lateral-velocity RMS (body sway) and the high-level arm-action RMS.

    E:\\envs\\RLisaac601\\python.exe scripts/him_error_compare.py mujoco
    E:\\envs\\RLisaac601\\python.exe scripts/him_error_compare.py physx --headless
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
OUT = ROOT / "logs/validation/him_error_compare"
CK = {"V6": ROOT / "logs/rsl_rl/homie_pushcart_v6/2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt",
      "V7": ROOT / "logs/rsl_rl/homie_pushcart_v7/2026-10-08_02-16-36_v7r9_rollscrub_v6track_hwgrip_20k/model_10000.pt"}
CMDS = [(0.5, 0.0, 0.0), (0.5, 0.0, 0.3), (0.3, 0.0, 0.0)]
PAYLOAD, SETTLE_S, MEASURE_S = 12.5, 4.0, 6.0


def summary(a):
    """a: (T, k) rows of [err_x, err_y, err_z, pelvis_vy, arm_action_abs_mean]."""
    a = np.asarray(a)
    return {"him_err_rms": np.sqrt((a[:, :3] ** 2).mean(0)).round(4).tolist(), "him_err_bias": a[:, :3].mean(0).round(4).tolist(),
            "pelvis_vy_rms": float(np.sqrt((a[:, 3] ** 2).mean())), "arm_action_abs": float(a[:, 4].mean())}


def run_mujoco():
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.policies_cross import CrossPolicy
    out = {}
    for driver in ("none", "V6", "V7"):
        pol = CrossPolicy(None if driver == "none" else int(driver[1]), "v7", None if driver == "none" else CK[driver])
        for cmd in CMDS:
            pol.reset(payload_kg=PAYLOAD)
            rows = []
            for k in range(int((SETTLE_S + MEASURE_S) / pol.dt)):
                pol.step(np.array(cmd))
                if k * pol.dt >= SETTLE_S:
                    r = pol.runner
                    e = r.vel_est - np.asarray(pol.env.base_lin_vel)
                    rows.append([*e, pol.env.base_lin_vel[1], float(np.abs(r.hl_actions[3:]).mean())])
            key = f"{driver}_vx{cmd[0]}_wz{cmd[2]}"
            out[key] = summary(rows)
            print("[mujoco]", key, out[key], flush=True)
    return out


def run_physx(device):
    import torch
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v7.baseline import apply_config, homie_baseline
    from homie_pushcart_v7.task_config import PushCartTaskCfg
    from homie_pushcart_v7.task_env import PushCartEnv
    from homie_pushcart_v6.hl_policy import HighLevelPolicy as P6
    from homie_pushcart_v7.hl_policy import HighLevelPolicy as P7
    N = 48                                              # 16 envs per driver: none / V6 / V7
    cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = N
    task = PushCartTaskCfg(); task.curriculum_v7.enabled = False
    task.env.episode_length_s = SETTLE_S + MEASURE_S + 5.0
    env = PushCartEnv(cfg, task, ROOT, device, True, level=0)
    env.cur.d[:] = 0.0
    env._refresh_global_level()
    env.fixed_caster_angle = 0.0
    env.fixed_payload_kg = PAYLOAD
    pols = {1: P6(CK["V6"], device=device), 2: P7(CK["V7"], device=device)}
    owner = torch.arange(N, device=device) % 3
    out = {}
    for cmd in CMDS:
        env.fixed_command = list(cmd)
        env.reset_idx(torch.arange(N, device=device)); env.reset()
        rows = []
        for k in range(int((SETTLE_S + MEASURE_S) / env.dt)):
            obs = env.observation_dict()
            act = torch.zeros(N, 18, device=device)
            with torch.inference_mode():
                for i, pol in pols.items():
                    act[owner == i] = pol(obs["policy"], obs["estimator"])[owner == i]
            env.step(act)
            if k * env.dt >= SETTLE_S:
                e = env.vel_est - env.base_lin_vel
                rows.append(torch.cat((e, env.base_lin_vel[:, 1:2], env.hl_actions[:, 3:].abs().mean(1, keepdim=True)), -1))
        R = torch.stack(rows).cpu().numpy()             # (T, N, 5)
        for i, name in enumerate(("none", "V6", "V7")):
            key = f"{name}_vx{cmd[0]}_wz{cmd[2]}"
            out[key] = summary(R[:, (owner == i).cpu().numpy()].reshape(-1, 5))
            print("[physx]", key, out[key], flush=True)
    env.sim.clear_instance()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backend", choices=("mujoco", "physx"))
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
            res = run_physx(a.device)
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
