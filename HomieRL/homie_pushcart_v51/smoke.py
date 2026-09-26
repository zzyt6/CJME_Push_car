"""Headless smoke test of PushCartSimEnv driven by the frozen HOMIE policy (phase C, also used before teleop)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .paths import ASSET_REPORT, HOMIE_RUN_DIR, PALM_LINKS, ROOT


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from homie_isaaclab.math_utils import quat_apply
    return quat_apply(q, v)


def grasp_errors(env, pose: dict, report: dict, env_ids=None) -> dict:
    """Distance between each palm's grasp point and the cart-side joint frame (should stay ~0)."""
    if env.cart is None:
        return {}
    cart_pos, cart_quat, _, _ = env.cart_state()
    out = {}
    for side, palm in PALM_LINKS.items():
        idx = env.body_names.index(palm)
        palm_state = env.rigid_body_states[:, idx]
        gp_palm = torch.tensor(pose["grasp"]["point_in_palm"][side], device=env.device, dtype=torch.float32)
        p0 = palm_state[:, :3] + _quat_apply_xyzw(palm_state[:, 3:7], gp_palm.expand(env.num_envs, -1))
        gp_cart = torch.tensor(report["grasp_joints"][side]["local_pos1_cart"], device=env.device, dtype=torch.float32)
        p1 = cart_pos + _quat_apply_xyzw(cart_quat, gp_cart.expand(env.num_envs, -1))
        err = (p0 - p1).norm(dim=-1)
        if env_ids is not None:
            err = err[env_ids]
        out[side] = float(err.max().item())
    return out


def run_smoke(args, launcher, root: Path = ROOT) -> dict:
    from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from .sim_env import PushCartSimEnv
    from .teleop import play_config

    device = args.device
    with_cart = args.mode == "cart"
    headless = bool(getattr(launcher, "_headless", True))
    checkpoint = Path(args.checkpoint) if args.checkpoint else frozen_checkpoint()
    cfg = play_config(num_envs=args.num_envs, episode_length_s=60.0)
    cart_params = None
    if getattr(args, "wheel_damping", None) is not None:
        from .assets.cart_urdf import load_cart_params
        cart_params = load_cart_params()
        cart_params["wheels"]["wheel_damping"] = float(args.wheel_damping)
    env = PushCartSimEnv(cfg, root, device, headless, with_cart=with_cart, cart_params=cart_params)
    if getattr(args, "arms", "grasp") == "default":
        env.upper_hold_actions.zero_()      # HOMIE's own default arm pose (hanging), for comparison only
    if env.cart is not None and getattr(args, "cart_mass", None) is not None:
        masses = env.cart.data.body_mass.torch.clone()
        inertias = env.cart.data.body_inertia.torch.clone()
        base = env.cart.body_names.index("base_link")
        scale = float(args.cart_mass) / float(masses[0, base])
        masses[:, base] *= scale
        inertias[:, base] *= scale
        env.cart.set_masses_index(masses=masses)
        env.cart.set_inertias_index(inertias=inertias)
        env.refresh_dynamics_properties(env.cart, masses=masses, inertias=inertias)
        print(f"[smoke] cart base_link mass set to {float(args.cart_mass):.1f} kg", flush=True)
    policy = HimLowLevelPolicy(checkpoint, device)
    report = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
    pose = env.pose

    obs, _ = env.reset()
    env.set_command(0., 0., 0.)
    static = {
        "root_height": env.root_states[:, 2].tolist(),
        "feet_z": env.feet_pos[:, :, 2].flatten().tolist(),
        "grasp_error_m": grasp_errors(env, pose, report),
    }
    if env.cart is not None:
        cart_pos, cart_quat, _, _ = env.cart_state()
        wheel_ids = [i for i, n in enumerate(env.cart.body_names) if n.startswith("wheel_")]
        wheel_z = env.cart.data.body_link_pose_w.torch[:, wheel_ids, 2]
        static["cart_pos"] = cart_pos[0].tolist()
        static["wheel_bottom_z"] = (wheel_z - env.cart_params["wheels"]["radius"]).flatten().tolist()
    print("[smoke] static:", json.dumps(static), flush=True)

    falls, max_grasp_err, start = 0, 0.0, time.perf_counter()
    base_vel, cart_speed, yaw_rate, windows, wheel_lift = [], [], [], [], []
    grasp_force, grasp_vec, root_height, leg_pos = [], [], [], []
    camera_offset = np.array([2.6, -2.8, 1.7])
    for step in range(args.steps):
        if not launcher.app.is_running():
            raise RuntimeError("Simulation closed before the smoke test completed")
        if not headless and step % 5 == 0:
            # keep the viewport on the first robot while it walks away with the cart
            target = env.root_states[0, :3].cpu().numpy()
            target = np.array([target[0] + 0.4, target[1], 0.7])
            env.sim.set_camera_view((target + camera_offset).tolist(), target.tolist())
        vx = args.vx if step >= args.settle_steps else 0.0
        env.set_command(vx, args.vy if step >= args.settle_steps else 0., args.wz if step >= args.settle_steps else 0.)
        obs, _, _, done, info, ids, _ = env.step(policy(obs))
        for name, tensor in (("obs", obs), ("joints", env.dof_pos)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} has non-finite values at step {step}")
        falls += int(info["terminated"].sum().item())
        base_vel.append(env.base_lin_vel[:, :2].mean(0).tolist())
        yaw_rate.append(float(env.base_ang_vel[:, 2].mean().item()))
        force_w, _, _ = env.grasp_wrench()
        grasp_force.append(force_w.norm(dim=-1).mean(0).tolist())       # [left, right] mean over envs
        grasp_vec.append(force_w.mean(0).flatten().tolist())            # [Lx, Ly, Lz, Rx, Ry, Rz] world
        root_height.append(float(env.root_states[:, 2].mean().item()))
        leg_pos.append(env.dof_pos[:, :12].mean(0).tolist())
        if env.cart is not None:
            err = grasp_errors(env, pose, report)
            max_grasp_err = max(max_grasp_err, *err.values())
            _, _, lin, _ = env.cart_state()
            cart_speed.append(float(lin[:, :2].norm(dim=-1).mean().item()))
        if env.cart is not None:
            wheel_ids = [i for i, n in enumerate(env.cart.body_names) if n.startswith("wheel_")]
            lift = (env.cart.data.body_link_pose_w.torch[:, wheel_ids, 2] - env.cart_params["wheels"]["radius"]).clamp_min(0.)
            wheel_lift.append(float(lift.max().item()))
        if (step + 1) % 100 == 0:
            seg = slice(step - 99, step + 1)
            speeds = np.linalg.norm(np.array(base_vel)[seg], axis=1)
            windows.append({
                "t_s": round((step + 1) * env.dt, 2),
                "cmd_vx": vx,
                "base_speed_mean": float(speeds.mean()), "base_speed_max": float(speeds.max()),
                "yaw_rate_abs_mean": float(np.abs(yaw_rate[seg]).mean()),
                "cart_speed_mean": float(np.mean(cart_speed[seg])) if cart_speed else None,
                "wheel_lift_max_mm": float(np.max(wheel_lift[seg]) * 1000) if wheel_lift else None,
                "wheel_lift_gt2mm_pct": float(np.mean(np.array(wheel_lift[seg]) > 0.002) * 100) if wheel_lift else None,
                "feet_contact": env.contact_forces[:, env.feet_indices, 2].gt(1.).float().mean().item(),
                "grasp_force_mean_N": np.mean(np.array(grasp_force)[seg], axis=0).round(2).tolist(),
                "grasp_force_max_N": np.max(np.array(grasp_force)[seg], axis=0).round(2).tolist(),
                "root_height_mean": float(np.mean(root_height[seg])),
            })
            print("[smoke] window", json.dumps(windows[-1]), flush=True)
    elapsed = time.perf_counter() - start
    tail = slice(max(0, args.steps - 100), None)
    result = {
        "result": "PASS" if falls == 0 else "FALLS",
        "mode": args.mode, "num_envs": args.num_envs, "steps": args.steps, "checkpoint": str(checkpoint),
        "command": [args.vx, args.vy, args.wz], "falls": falls,
        "base_vel_xy_last100_mean": np.mean(np.array(base_vel)[tail], axis=0).tolist(),
        "final_root_height": env.root_states[:, 2].tolist(),
        "root_height_last100_mean": float(np.mean(root_height[tail])),
        "preset_root_height": float(pose["robot"]["base_pos"][2]),
        "grasp_force_last100_mean_N": np.mean(np.array(grasp_force)[tail], axis=0).round(2).tolist(),
        "grasp_force_last100_max_N": np.max(np.array(grasp_force)[tail], axis=0).round(2).tolist(),
        "grasp_force_last100_mean_vec_w": np.mean(np.array(grasp_vec)[tail], axis=0).round(2).tolist(),
        "hand_subtree_mass_kg": env.hand_subtree_mass.tolist(),
        "leg_joint_last100_mean": dict(zip(env.dof_names[:12], np.mean(np.array(leg_pos)[tail], axis=0).round(4).tolist())),
        "realtime_factor": (args.steps * env.dt) / elapsed,
        "static": static,
        "windows": windows,
    }
    if env.cart is not None:
        cart_pos, _, _, _ = env.cart_state()
        result.update({
            "cart_speed_last100_mean": float(np.mean(cart_speed[tail])),
            "cart_travel_x_m": float(cart_pos[0, 0].item() - static["cart_pos"][0]),
            "max_grasp_error_m": max_grasp_err,
        })
    path = root / "logs/pushcart_v5_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "logs/pushcart_v5_physics_config.json").write_text(json.dumps(env.physics_report, indent=2), encoding="utf-8")
    result["physics_vs_homie_training"] = compare_physics_with_homie(env.physics_report, root)
    print("[smoke] physics vs HOMIE training run: " + json.dumps(result["physics_vs_homie_training"]), flush=True)
    print("[smoke] " + json.dumps({k: v for k, v in result.items() if k != "windows"}), flush=True)
    return result


def compare_physics_with_homie(report: dict, root: Path) -> dict:
    """Diff the effective robot physics against the physics_config.json written by the HOMIE training run."""
    ref_path = HOMIE_RUN_DIR / "physics_config.json"
    if not ref_path.is_file():
        return {"reference": str(ref_path), "status": "missing"}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    out = {"reference": str(ref_path)}
    out["joint_order_equal"] = ref["joint_order"] == report["joint_order"]
    for key in ("armature", "stiffness", "damping"):
        a, b = np.array(ref[key]), np.array(report[key])
        out[f"{key}_max_abs_diff"] = float(np.abs(a - b).max()) if a.shape == b.shape else "shape_mismatch"
    ref_mass = dict(zip(ref["body_names"], ref["body_masses"]))
    new_mass = dict(zip(report["body_names"], report["body_masses"]))
    common = sorted(set(ref_mass) & set(new_mass))
    diffs = {n: float(new_mass[n] - ref_mass[n]) for n in common}
    out["body_count_ref_vs_new"] = [len(ref_mass), len(new_mass)]
    # The training run's report was written with mass randomisation active, so compare against the URDF too.
    out["body_mass_max_abs_diff_vs_training_run_kg"] = float(max(abs(v) for v in diffs.values())) if diffs else None
    import xml.etree.ElementTree as ET
    from .paths import G1_URDF
    urdf_mass = {l.get("name"): float(l.find("inertial/mass").get("value"))
                 for l in ET.parse(G1_URDF).getroot().findall("link") if l.find("inertial/mass") is not None}
    out["body_mass_max_abs_diff_vs_urdf_kg"] = float(max(abs(new_mass[n] - urdf_mass[n]) for n in new_mass if n in urdf_mass))
    out["training_run_had_randomization"] = bool(ref.get("randomization"))
    out["bodies_only_in_ref"] = sorted(set(ref_mass) - set(new_mass))
    out["bodies_only_in_new"] = sorted(set(new_mass) - set(ref_mass))
    out["physics_dt_equal"] = abs(ref["physics_dt_s"] - report["physics_dt_s"]) < 1e-9
    out["decimation_equal"] = ref["decimation"] == report["decimation"]
    rb_ref = {b["path"].split("/")[-1]: b for b in ref["rigid_body_properties"]}
    rb_new = {b["path"].split("/")[-1]: b for b in report["rigid_body_properties"]}
    damp_diff = [abs((rb_new[n]["linear_damping"] or 0) - (rb_ref[n]["linear_damping"] or 0)) for n in rb_ref if n in rb_new]
    out["rigid_body_linear_damping_max_abs_diff"] = float(max(damp_diff)) if damp_diff else None
    cs_ref = {s["path"].split("/")[-1]: s for s in ref["collision_shapes"]}
    cs_new = {s["path"].split("/")[-1]: s for s in report["collision_shapes"] if "Robot" in s["path"] or "Scene" not in s["path"]}
    out["robot_collision_shapes_ref_vs_new"] = [len(cs_ref), len([s for s in report["collision_shapes"] if "/Robot/" in s["path"]])]
    return out
