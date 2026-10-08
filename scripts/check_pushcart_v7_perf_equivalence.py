"""The V7 performance caches must not change any value: compare each cache with a fresh read after real steps.

    E:\envs\RLisaac601\python.exe scripts/check_pushcart_v7_perf_equivalence.py --headless
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=1024)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    ok = True
    try:
        import torch
        import warp as wp
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart.sim_env import PushCartSimEnv
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.hand_labels import compensate_subtree_wrench, rotate_inertia, rotation_matrix_xyzw
        from homie_pushcart_v7.observation import world_to_cart_wrenches
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = a.num_envs
        task = PushCartTaskCfg(); task.env.episode_length_s = 3.0          # many resets
        task.curriculum.init_level = 12
        env = PushCartEnv(cfg, task, ROOT, a.device, True, level=12)
        env.reset()
        for _ in range(400):
            env.step(0.3 * torch.randn(a.num_envs, 18, device=a.device))

        # 1. hand labels: cached masses / inertias vs. a fresh PhysX read (the original V6 code path)
        lab, robot, data = env._labels, env.robot, env.robot.data
        palms = env.palm_body_ids
        pose = data.body_link_pose_w.torch[:, palms]
        R = rotation_matrix_xyzw(pose[..., 3:7])
        inc = wp.to_torch(robot.root_view.get_link_incoming_joint_force()).reshape(env.num_envs, robot.num_bodies, 6)[:, palms]
        inc = torch.cat(((R @ inc[..., :3, None]).squeeze(-1), (R @ inc[..., 3:, None]).squeeze(-1)), -1)
        ref = []
        for side, ids in enumerate(lab.subtrees):
            inertia = rotate_inertia(data.body_inertia.torch[:, ids].reshape(env.num_envs, -1, 3, 3).transpose(-1, -2),
                                     data.body_link_pose_w.torch[:, ids, 3:7])
            acc = data.body_com_acc_w.torch[:, ids]
            ref.append(compensate_subtree_wrench(inc[:, side], pose[:, side, :3], data.body_mass.torch[:, ids],
                                                 data.body_com_pose_w.torch[:, ids, :3], data.body_com_vel_w.torch[:, ids, 3:],
                                                 acc[..., :3], acc[..., 3:], inertia, lab.gravity))
        valid = (env.episode_length_buf > 0) & torch.isfinite(torch.stack(ref, 1)).all(dim=(1, 2))
        ref = world_to_cart_wrenches(torch.where(valid[:, None, None], torch.stack(ref, 1), 0.), env.ts.cart_yaw)
        new, _ = lab.sample()
        d1 = float((new - ref).abs().max())
        print(f"[equiv] hand labels: cached vs fresh-read max |diff| = {d1:.3e}", flush=True)
        ok &= d1 < 1e-4
        # 2. cart COM: GPU mass mirror vs V3.2's body_mass read
        dm = float((env._cart_mass - env.cart.data.body_mass.torch).abs().max())
        d2 = float((env.cart_com_world() - PushCartSimEnv.cart_com_world(env)).abs().max())
        print(f"[equiv] cart masses: GPU mirror vs PhysX max |diff| = {dm:.3e} kg", flush=True)
        print(f"[equiv] cart COM: mirror vs fresh-read max |diff| = {d2:.3e} m (float32 at env origins ~1e2 m)", flush=True)
        ok &= dm < 1e-4 and d2 < 1e-4      # PhysX stores masses in float32: ~2e-6 kg round trip
        # 2b. COM positions from link pose + mirrored offsets vs Isaac Lab's body_com_pose_w
        d5 = float((env._cart_com_positions() - env.cart.data.body_com_pose_w.torch[..., :3]).abs().max())
        d6 = max(float((lab.com_b[k] - data.body_com_pose_b.torch[:, ids, :3]).abs().max()) for k, ids in enumerate(lab.subtrees))
        d7 = float((env._cart_com_b - env.cart.data.body_com_pose_b.torch[..., :3]).abs().max())
        print(f"[equiv] cart body COMs: computed vs body_com_pose_w max |diff| = {d5:.3e} m", flush=True)
        print(f"[equiv] COM offset mirrors vs PhysX: hands {d6:.3e} m, cart {d7:.3e} m", flush=True)
        ok &= d5 < 1e-4 and d6 < 1e-6 and d7 < 1e-6
        # 3. max-force CPU mirror vs PhysX (the per-step slide-limit writer)
        phys = wp.to_torch(env.cart.root_view.get_dof_max_forces())
        d3 = float((env._maxf_cpu - phys).abs().max())
        print(f"[equiv] dof max forces: CPU mirror vs PhysX max |diff| = {d3:.3e}", flush=True)
        ok &= d3 < 1e-4
        # 4. the slide limit in PhysX is the load-dependent value the task computed; the slide target is the anchor
        d4 = float((phys[:, env.slide_ids.cpu()] - env.slide_limit_now.cpu()).abs().max())
        d8 = float((env.cart.data.joint_pos_target.torch[:, env.slide_ids] - env.slide_anchor).abs().max())
        print(f"[equiv] slide max force in PhysX vs task limit max |diff| = {d4:.3e} N; target vs anchor {d8:.3e} m", flush=True)
        ok &= d4 < 1e-4 and d8 < 1e-7
        print("PERF EQUIVALENCE", "PASS" if ok else "FAIL", flush=True)
    finally:
        app.app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
