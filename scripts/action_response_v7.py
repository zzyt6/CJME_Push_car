"""High-level action -> robot + cart response, MuJoCo vs Isaac (V7 plant), with the policy taken out of the loop.

Constant high-level actions (18: HOMIE command residual vx vy wz, waist yaw, 7 left-arm, 7 right-arm joints, all in
the policy's [-1, 1] action space) are applied while the task command is a steady forward walk. If the same action
moves the cart differently in the two engines, a policy trained in one cannot transfer to the other.

    E:\\envs\\RLisaac601\\python.exe scripts/action_response_v7.py mujoco
    E:\\envs\\RLisaac601\\python.exe scripts/action_response_v7.py physx --headless
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
OUT = ROOT / "logs/validation/action_response_v7"
UPPER = ["waist_yaw", "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wr_roll", "L_wr_pitch", "L_wr_yaw",
         "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wr_roll", "R_wr_pitch", "R_wr_yaw"]


def pattern(name):
    a = np.zeros(18, np.float32)
    idx = {n: 3 + i for i, n in enumerate(UPPER)}
    if name == "cmd_wz":
        a[2] = 0.5
    elif name == "cmd_vy":
        a[1] = 0.5
    elif name == "waist_yaw":
        a[idx["waist_yaw"]] = 0.5
    elif name == "diff_sh_pitch":
        a[idx["L_sh_pitch"]], a[idx["R_sh_pitch"]] = 0.5, -0.5
    elif name == "wrist_yaw_both":
        a[idx["L_wr_yaw"]], a[idx["R_wr_yaw"]] = 0.5, 0.5
    elif name == "sh_roll_both":
        a[idx["L_sh_roll"]], a[idx["R_sh_roll"]] = 0.5, 0.5
    elif name == "elbow_both":
        a[idx["L_elbow"]], a[idx["R_elbow"]] = 0.5, 0.5
    return a


PATTERNS = ["none", "cmd_wz", "cmd_vy", "waist_yaw", "diff_sh_pitch", "wrist_yaw_both", "sh_roll_both", "elbow_both"]
CMD, PAYLOAD, SETTLE_S, MEASURE_S = (0.4, 0.0, 0.0), 12.5, 4.0, 6.0
KEYS = ("cart_vx", "cart_vy", "cart_wz", "pelvis_wz", "rel_yaw_deg", "upper_dq_max")


def run_mujoco():
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.policies_cross import CrossPolicy
    pol = CrossPolicy(None, "v7")
    r = pol.runner
    out = {}
    for name in PATTERNS:
        act = torch.from_numpy(pattern(name)).unsqueeze(0)
        r.hl = (lambda obs, act=act: act) if name != "none" else None
        pol.reset(payload_kg=PAYLOAD)
        rows = []
        for k in range(int((SETTLE_S + MEASURE_S) / pol.dt)):
            pol.step(np.array(CMD))
            if k * pol.dt >= SETTLE_S:
                v = pol.velocity()
                rows.append([v[0], v[1], v[2], pol.env.base_ang_vel[2], np.degrees(r.ts.get("rel_yaw", np.nan)),
                             float(np.abs(r.upper_delta_q).max())])
        a = np.array(rows)
        out[name] = dict(zip(KEYS, np.nanmean(a, 0).round(4).tolist()))
        print("[mujoco]", name, out[name], flush=True)
    return out


def run_physx(device):
    import torch
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v7.baseline import apply_config, homie_baseline
    from homie_pushcart_v7.task_config import PushCartTaskCfg
    from homie_pushcart_v7.task_env import PushCartEnv
    per = 6
    N = per * len(PATTERNS)
    cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = N
    cfg.noise.add_noise = False                          # as MuJoCo
    task = PushCartTaskCfg(); task.curriculum_v7.enabled = False
    task.env.episode_length_s = SETTLE_S + MEASURE_S + 5.0
    env = PushCartEnv(cfg, task, ROOT, device, True, level=0)
    env.cur.d[:] = 0.0
    env._refresh_global_level()
    env.fixed_caster_angle = 0.0
    env.fixed_payload_kg = PAYLOAD
    env.fixed_command = list(CMD)
    acts = torch.tensor(np.stack([pattern(p) for p in PATTERNS]), device=device)
    owner = torch.arange(N, device=device) % len(PATTERNS)
    A = acts[owner]
    env.reset_idx(torch.arange(N, device=device)); env.reset()
    rows = []
    for k in range(int((SETTLE_S + MEASURE_S) / env.dt)):
        env.step(A)
        if k * env.dt >= SETTLE_S:
            rel = getattr(env.ts, "rel_yaw", torch.full((N,), float("nan"), device=device))
            rows.append(torch.stack((env.ts.cart_v[:, 0], env.ts.cart_v[:, 1], env.ts.cart_wz, env.base_ang_vel[:, 2],
                                     torch.rad2deg(rel), env.upper_delta_q.abs().max(1).values), -1))
    R = torch.stack(rows).mean(0)                       # (N, 6)
    out = {}
    for i, name in enumerate(PATTERNS):
        m = R[owner == i]
        out[name] = dict(zip(KEYS, [round(float(x), 4) for x in m.mean(0)]))
        out[name]["cart_wz_env_min_max"] = [round(float(m[:, 2].min()), 3), round(float(m[:, 2].max()), 3)]
        print("[physx]", name, out[name], flush=True)
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
