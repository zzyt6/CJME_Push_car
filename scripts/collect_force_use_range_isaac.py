"""Independent Isaac route rollouts for hand-on-cart resultant-force envelopes.

The production controller/physics are unchanged. Measurements are captured before
automatic reset. Capability snapshots describe posture-conditioned actuator-only
quasi-static wrench sets; they are not a whole-body maximum-force certification.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
import types
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yaw_xyzw(q):
    x, y, z, w = q
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def child(args):
    from isaaclab.app import AppLauncher
    launcher = None
    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        launcher = AppLauncher(headless=True, device=args.device)
        import torch
        import warp as wp
        from collect_motion_ability_isaac import configure, capture_physics
        from homie_pushcart_v6.task_env import PushCartEnv, TERMINATION_REASONS
        from homie_pushcart_v6.hl_policy import HighLevelPolicy
        from homie_pushcart_v6.hand_labels import ReadOnlyHandLabels
        from pushcart_bench.config_io import load_config
        from pushcart_bench.paths import build_path
        from pushcart_bench.follower import PathFollower, FollowerConfig
        torch.set_num_threads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg, task = configure(args.checkpoint, args.timeout)
        env = PushCartEnv(cfg, task, ROOT, args.device, True, checkpoint=None, level=17)
        env.fixed_payload_kg = args.payload
        env.fixed_caster_angle = 0.
        env.fixed_command = [0., 0., 0.]
        env.reset()
        controller = HighLevelPolicy(args.checkpoint, device=args.device)
        assert not env.add_noise and not cfg.domain_rand.delay
        assert torch.count_nonzero(env.hl_delay) == 0 and torch.count_nonzero(env.cart_pose_delay) == 0
        initial = capture_physics(env)
        np.savez_compressed(raw_dir / f"{args.method}_initial_physics.npz", **initial)
        bench = load_config()
        fcfg = FollowerConfig.from_dict(bench["follower"])
        fcfg.v_nom, fcfg.cruise, fcfg.k_curv, fcfg.v_min_frac = args.speed, True, 0., 1.
        fcfg.timeout_s = args.timeout
        local = build_path("composite_course", bench["paths"]["composite_course"])
        pose0 = env.cart.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        initial_yaw = yaw_xyzw(pose0[3:])
        path = local.transformed(pose0[:2], initial_yaw)
        follower = PathFollower(path, fcfg, float(env.dt))
        np.savez_compressed(raw_dir / f"{args.method}_reference_path.npz", xy=path.xy, yaw=path.yaw, s=path.s)
        pending = {}
        original_check = env.check_termination
        def capture(this):
            original_check()
            fields = {
                "cart_pose": this.cart.data.root_link_pose_w.torch,
                "cart_velocity": torch.cat((this.ts.cart_v, this.ts.cart_wz[:, None]), -1),
                "reference": this.cmd_ref, "homie_command": this.cmd_homie,
                "hand_wrench_cart": this._pre_reset_hands[:, :12] * this.hand_scale,
                "hand_label_valid": this._pre_reset_hands[:, 12:13],
                "legacy_hand_force_world": this.ts.force_w_raw.flatten(1),
                "high_action": this.hl_actions, "upper_delta_q": this.upper_delta_q,
                "joint_position": this.dof_pos, "joint_velocity": this.dof_vel,
                "joint_torque_proxy": this.torques,
                "terminated": this.terminated_buf[:, None], "env_timeout": this.time_out_buf[:, None],
                "termination_flags": torch.stack([this.term_reason_buf[n] for n in TERMINATION_REASONS], -1),
            }
            for key, value in fields.items():
                pending[key] = value[0].detach().cpu().numpy().copy()
        env.check_termination = types.MethodType(capture, env)

        def capability():
            # Floating-base Jacobian includes six root columns; use actual simulator joint order.
            data = env.robot.data
            j = data.body_link_jacobian_w.torch[0]
            palms = env.palm_body_ids.long()
            link_offset = 0 if j.shape[0] == env.robot.num_bodies else 1
            nroot = j.shape[-1] - env.robot.num_joints
            ids = env.joint_ids.long()[12:]
            hand_j = j[palms - link_offset][:, :, ids + nroot]
            g = data.gravity_compensation_forces.torch[0].to(env.device)
            g_offset = g.shape[-1] - env.robot.num_joints
            limits = wp.to_torch(env.robot.root_view.get_dof_max_forces()).to(env.device)[0, ids]
            np.testing.assert_allclose(limits.cpu().numpy(), env.torque_limits[12:].cpu().numpy(), rtol=0, atol=1e-5)
            return {"jacobian_world": hand_j.cpu().numpy().copy(),
                    "gravity_torque": g[ids + g_offset].cpu().numpy().copy(),
                    "torque_limits": limits.cpu().numpy().copy(),
                    "joint_position": env.dof_pos[0].cpu().numpy().copy(),
                    "cart_yaw": yaw_xyzw(data_pose())}

        def data_pose():
            return env.cart.data.root_link_pose_w.torch[0, 3:].cpu().numpy()

        capacity_rows = [{"t": 0., "s": 0., **capability()}]
        print("[force-range] initialized", args.method, "capacity shape", capacity_rows[0]["jacobian_world"].shape, flush=True)
        records = {}
        failure = None
        dt = float(env.dt)
        max_ticks = int(math.ceil((args.timeout + fcfg.post_reach_s + dt) / dt))
        with torch.inference_mode():
            for tick in range(max_ticks):
                st = follower.state
                if st.timed_out or (st.reached and st.t >= st.t_reached + fcfg.post_reach_s):
                    break
                pose = env.cart.data.root_link_pose_w.torch[0].cpu().numpy()
                yaw = yaw_xyzw(pose[3:])
                vel_world = env.cart.data.root_com_vel_w.torch[0].cpu().numpy()
                c, s = math.cos(yaw), math.sin(yaw)
                velocity = np.array([c*vel_world[0]+s*vel_world[1], -s*vel_world[0]+c*vel_world[1], vel_world[5]])
                command = follower.update((pose[0], pose[1], yaw), velocity)
                info = follower.info()
                env.cmd_task.copy_(torch.as_tensor(command, device=args.device, dtype=torch.float32)[None])
                env.cmd_timer.fill_(10**9)
                action = controller(env.hl_obs, env.estimator_obs) if args.method == "CWC" else torch.zeros(1, 18, device=args.device)
                env.step(action)
                sample = {"t": (tick+1)*dt, "interval_start": tick*dt,
                          "command": command.copy(), "progress_pre": info["s"],
                          "cross_track_pre": info["cross_track"], "heading_error_pre": info["heading_err"], **pending}
                if not (pending["terminated"][0] or pending["env_timeout"][0]):
                    speed, hands = controller.physical_estimates(env.hl_obs, env.estimator_obs)
                    sample["pred_hand_wrench"] = hands[0].cpu().numpy().copy()
                    sample["pred_cart_velocity"] = speed[0].cpu().numpy().copy()
                else:
                    sample["pred_hand_wrench"] = np.full(12, np.nan)
                    sample["pred_cart_velocity"] = np.full(3, np.nan)
                for key, value in sample.items():
                    records.setdefault(key, []).append(value)
                if tick % 25 == 24 and not pending["terminated"][0]:
                    capacity_rows.append({"t": (tick+1)*dt, "s": info["s"], **capability()})
                if tick % 250 == 249:
                    print(f"[force-range] {args.method} t={(tick+1)*dt:.1f}s s={info['s']:.2f}m ct={info['cross_track']:.3f}m", flush=True)
                if pending["terminated"][0] or pending["env_timeout"][0]:
                    failure = [n for n, flag in zip(TERMINATION_REASONS, pending["termination_flags"]) if flag]
                    if pending["env_timeout"][0]:
                        failure.append("env_timeout")
                    break
        arrays = {k: np.asarray(v) for k, v in records.items()}
        for key in ("terminated", "env_timeout", "hand_label_valid"):
            arrays[key] = arrays[key].reshape(-1).astype(bool)
        assert np.isfinite(arrays["hand_wrench_cart"]).all()
        assert arrays["hand_label_valid"].all()
        if args.method == "baseline":
            assert np.count_nonzero(arrays["high_action"]) == 0
        raw_path = raw_dir / f"{args.method}.npz"
        np.savez_compressed(raw_path, **arrays)
        np.savez_compressed(raw_dir / f"{args.method}_capacity.npz", **{k: np.asarray([r[k] for r in capacity_rows]) for k in capacity_rows[0]})
        final = capture_physics(env)
        np.savez_compressed(raw_dir / f"{args.method}_final_physics.npz", **final)
        for label in ("robot", "cart"):
            for key in ("body_mass", "body_inertia", "body_com_pose_b", "joint_stiffness", "joint_damping", "joint_armature", "materials"):
                np.testing.assert_array_equal(initial[f"{label}.{key}"], final[f"{label}.{key}"])
        meta = {"run_id": os.environ.get("CWC_FORCE_RANGE_RUN"), "engine": "Isaac Sim / PhysX",
                "method": args.method, "payload_kg": args.payload, "seed": args.seed,
                "speed_m_s": args.speed, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
                "iteration": controller.iteration, "homie_checkpoint": str(env.policy.checkpoint), "homie_sha256": sha(env.policy.checkpoint),
                "path_name": "composite_course", "path_params": local.params, "path_length_m": path.length,
                "follower": fcfg.to_dict(), "dt": dt, "samples": len(arrays["t"]),
                "success": bool(follower.state.reached and not failure), "failure": failure,
                "route_timed_out": follower.state.timed_out, "time_to_goal": follower.state.t_reached,
                "final_progress_m": follower.state.s, "final_follower_info": follower.info(),
                "force_measurement": ReadOnlyHandLabels.metadata(),
                "force_sum": "Add left and right hand forces in cart-heading axes; N. Moments are kept separate.",
                "sampling": "50 Hz, last physics substep, pre-reset; progress/errors are pre-step, forces and poses post-step",
                "capability_measurement": "Read-only link-origin Jacobians, gravity torques, simulator effort limits, sampled every 0.5 s plus reset",
                "upper_joint_names": env.dof_names[12:], "robot_joint_names": list(env.robot.joint_names),
                "npz_sha256": sha(raw_path), "physics_unchanged_within_run": True,
                "cart_mass_total_kg": float(initial["cart.body_mass"].sum())}
        (raw_dir / f"{args.method}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[force-range] DONE", json.dumps({k: meta[k] for k in ("method", "success", "failure", "time_to_goal", "final_progress_m")}), flush=True)
    except BaseException:
        (args.output / f"{args.method}.failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "logs/Picture/force_use_range")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--payload", type=float, default=15.)
    p.add_argument("--speed", type=float, default=.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=90.)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--method", choices=("baseline", "CWC"))
    args = p.parse_args()
    if args.method:
        return child(args)
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    args.checkpoint = resolve_v6_checkpoint(args.checkpoint or "latest").resolve()
    run_id = uuid.uuid4().hex
    attempt = args.output / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{run_id[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "engine": "Isaac Sim / PhysX", "payload_kg": args.payload,
                "speed_m_s": args.speed, "seed": args.seed, "timeout_s": args.timeout,
                "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
                "collector_sha256": sha(__file__), "attempt_dir": str(attempt), "runs": []}
    for method in ("baseline", "CWC"):
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--method", method,
               "--output", str(attempt), "--checkpoint", str(args.checkpoint), "--payload", str(args.payload),
               "--speed", str(args.speed), "--seed", str(args.seed), "--timeout", str(args.timeout), "--device", args.device]
        print(f"[force-range] START {method}, log={attempt / (method+'.log')}", flush=True)
        with (attempt / f"{method}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CWC_FORCE_RANGE_RUN=run_id), stdout=log, stderr=subprocess.STDOUT)
        meta_path = attempt / "raw" / f"{method}.json"
        if result.returncode or not meta_path.is_file():
            raise RuntimeError(f"{method} failed to collect: {attempt / (method+'.log')}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["npz_sha256"] == sha(meta_path.with_suffix(".npz"))
        manifest["runs"].append(meta)
        (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[force-range] FINISH {method}: success={meta['success']} progress={meta['final_progress_m']:.2f}", flush=True)
    with np.load(attempt / "raw/baseline_initial_physics.npz") as a, np.load(attempt / "raw/CWC_initial_physics.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    manifest["initial_physics_and_state_exact_equal"] = True
    destination = args.output / "raw"
    destination.mkdir(exist_ok=True)
    for source in (attempt / "raw").iterdir():
        if (destination / source.name).exists():
            raise FileExistsError(f"Refusing to replace previous raw data: {destination / source.name}")
        shutil.copy2(source, destination / source.name)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance = args.output / "provenance"
    provenance.mkdir(exist_ok=True)
    shutil.copy2(__file__, provenance / Path(__file__).name)
    print("[force-range] VERIFIED", args.output, flush=True)


if __name__ == "__main__":
    main()
