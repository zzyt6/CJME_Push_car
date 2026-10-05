"""Independently replay vy/wz via the production viewer factory and audit signals.

No server is opened, no physics parameters are changed, and paper data are read
only. A transparent hook records the actual tensor passed into frozen HOMIE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yaw(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def replay(args):
    import mujoco
    import torch
    from homie_pushcart.mujoco_viewer import make_viewer_runner

    torch.set_num_threads(1)
    paper_root = ROOT / "logs/Picture/motion_ability/mujoco"
    metadata = json.loads((paper_root / "manifest.json").read_text(encoding="utf-8"))
    key = f"{args.axis}_{args.method}"
    runner, homie_checkpoint, family = make_viewer_runner(Path(metadata["checkpoint"]), level=17)
    assert family == "v6"
    env, m, d = runner.env, runner.env.model, runner.env.data
    np.random.seed(metadata["seed"])
    torch.manual_seed(metadata["seed"])
    runner.set_payload(metadata["payload_kg"])
    env.set_command(0., 0., 0.)
    runner.reset(caster_angle=0.)
    if args.method == "baseline":
        runner.hl = None  # Production viewer's high-level checkbox off.
    assert not d.time and not np.any(d.qvel)
    actual_low = runner.lowlevel.act_and_estimate
    latest = {}

    def record_low_input(obs):
        latest["homie_input_frame"] = obs[0, -env.one_step_obs:].detach().numpy().copy()
        actions, velocity = actual_low(obs)
        latest["homie_output_legs"] = actions[0].detach().numpy().copy()
        return actions, velocity

    runner.lowlevel.act_and_estimate = record_low_input
    dt = env.dt
    before = dict(cart_xpos=d.xpos[runner.cart_body].copy(), cart_xipos=d.xipos[runner.cart_body].copy(),
                  cart_xquat=d.xquat[runner.cart_body].copy(), robot_xpos=d.xpos[env.pelvis_id].copy())
    trace = {}
    failure = None
    axis_index = 1 if args.axis == "vy" else 2
    adr = int(m.joint("cart_free").qposadr[0])
    va = int(m.joint("cart_free").dofadr[0])
    with torch.inference_mode():
        for tick in range(round(20. / dt)):
            # Independent schedule implementation, not the paper collector's helper.
            command = np.zeros(3)
            if tick >= round(12. / dt):
                command[axis_index] = .3
            elif tick >= round(4. / dt):
                command[axis_index] = .15
            _, fallen = runner.step(command)
            failure = "fall" if fallen else runner.geometry_failure()
            raw_world = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, runner.cart_body, raw_world, 0)
            q = d.xquat[runner.cart_body]
            heading = float(yaw(q))
            c, s = np.cos(heading), np.sin(heading)
            recomputed = np.array([c * raw_world[3] + s * raw_world[4],
                                   -s * raw_world[3] + c * raw_world[4], raw_world[2]])
            expected_low_cmd = runner.cmd_homie * env.commands_scale
            np.testing.assert_allclose(latest["homie_input_frame"][:3], expected_low_cmd, rtol=0., atol=1e-7)
            np.testing.assert_array_equal(env.actions, np.clip(latest["homie_output_legs"], -env.clip_actions, env.clip_actions))
            eq = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            values = dict(t=(tick + 1) * dt, interval_start=tick * dt,
                          command=command, reference=runner.cmd_ref, homie_command=runner.cmd_homie,
                          cart_velocity=[*runner.ts["cart_v"], runner.ts["cart_wz"]],
                          independent_object_velocity=recomputed,
                          cart_world_velocity=raw_world,
                          cart_pose=np.r_[d.xpos[runner.cart_body], q[1:], q[0]],
                          cart_com_position=d.xipos[runner.cart_body], cart_qpos=d.qpos[adr:adr + 7],
                          cart_qvel=d.qvel[va:va + 6], robot_position=d.xpos[env.pelvis_id],
                          homie_input_frame=latest["homie_input_frame"], homie_output_legs=latest["homie_output_legs"],
                          lower_action=env.actions, joint_position=env.dof_pos, joint_velocity=env.dof_vel,
                          joint_target=env.joint_pos_target, joint_torque=d.ctrl[env.act_idx],
                          high_action=runner.hl_actions, upper_delta_q=runner.upper_delta_q,
                          hand_wrenches=np.stack([runner.ts["hand_cart_wrenches"][side] for side in ("left", "right")]),
                          grasp_constraint_max_abs_residual=np.max(np.abs(d.efc_pos[eq])),
                          terminated=failure is not None)
            for name, value in values.items():
                trace.setdefault(name, []).append(np.asarray(value).copy())
            if failure:
                break
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{key}.npz", **arrays)
    with np.load(paper_root / "raw" / f"{key}.npz") as saved:
        comparisons = {}
        for name in ("cart_velocity", "command", "reference", "homie_command", "cart_pose", "lower_action",
                     "high_action", "upper_delta_q", "joint_position", "joint_velocity"):
            comparisons[name] = dict(exactly_equal=bool(np.array_equal(arrays[name], saved[name])),
                                     max_abs_difference=float(np.abs(arrays[name] - saved[name]).max()))
    np.testing.assert_allclose(arrays["cart_velocity"], arrays["independent_object_velocity"], rtol=0., atol=1e-12)
    # Independently differentiate measured COM position and measured yaw. These
    # are finite-difference estimates, with the expected 50 Hz discretization error.
    yaws = np.unwrap(yaw(np.c_[arrays["cart_pose"][:, 6], arrays["cart_pose"][:, 3:6]]))
    world_fd = np.gradient(arrays["cart_com_position"], dt, axis=0, edge_order=2)
    c, s = np.cos(yaws), np.sin(yaws)
    fd = np.c_[c * world_fd[:, 0] + s * world_fd[:, 1],
               -s * world_fd[:, 0] + c * world_fd[:, 1], np.gradient(yaws, dt, edge_order=2)]
    np.savez_compressed(args.output / f"{key}_kinematic_check.npz", t=arrays["t"],
                        pose_derived_cart_velocity=fd, raw_cart_velocity=arrays["cart_velocity"],
                        measured_cart_com_position=arrays["cart_com_position"], measured_cart_yaw=yaws)
    periods = []
    for start, end, desired in [(0., 4., 0.), (4., 12., .15), (12., 20., .3)]:
        selected = (arrays["interval_start"] >= start) & (arrays["interval_start"] < end)
        tail = selected & (arrays["interval_start"] >= start + 2.)
        positions = np.flatnonzero(selected)
        first, last = positions[0], positions[-1]
        previous = max(first - 1, 0)
        com0 = before["cart_xipos"] if first == 0 else arrays["cart_com_position"][previous]
        yaw0 = float(yaw(before["cart_xquat"])) if first == 0 else yaws[previous]
        delta_world = arrays["cart_com_position"][last] - com0
        heading0 = yaw0
        delta_body_start = np.array([np.cos(heading0) * delta_world[0] + np.sin(heading0) * delta_world[1],
                                    -np.sin(heading0) * delta_world[0] + np.cos(heading0) * delta_world[1]])
        cmds = arrays["homie_command"][tail]
        actions = arrays["lower_action"][tail]
        leg_q = arrays["joint_position"][tail, :12]
        periods.append(dict(start_s=start, end_s=end, requested_velocity=desired,
                            homie_command_tail_mean=cmds.mean(0).tolist(),
                            actual_homie_input_command_tail_mean=arrays["homie_input_frame"][tail, :3].mean(0).tolist(),
                            actual_velocity_tail_mean=arrays["cart_velocity"][tail].mean(0).tolist(),
                            pose_derived_velocity_tail_mean=fd[tail].mean(0).tolist(),
                            finite_difference_vs_raw_rms=np.sqrt(((fd[tail] - arrays["cart_velocity"][tail]) ** 2).mean(0)).tolist(),
                            world_com_displacement_m=delta_world.tolist(), local_start_frame_displacement_m=delta_body_start.tolist(),
                            yaw_displacement_deg=float(np.rad2deg(yaws[last] - yaw0)),
                            lower_action_peak_to_peak=actions.ptp(0).tolist() if hasattr(actions,"ptp") else np.ptp(actions,axis=0).tolist(),
                            leg_joint_motion_peak_to_peak_rad=np.ptp(leg_q, axis=0).tolist(),
                            leg_action_clip_fraction=float((np.abs(actions) >= env.clip_actions).mean()),
                            leg_torque_at_limit_fraction=float((np.abs(arrays["joint_torque"][tail, :12]) >= env.torque_limits[:12] - 1e-7).mean()),
                            high_action_clip_fraction=float((np.abs(arrays["high_action"][tail]) >= runner.task_cfg.actions.clip - 1e-7).mean()),
                            grasp_constraint_max_abs_residual=float(arrays["grasp_constraint_max_abs_residual"][tail].max())))
    report = dict(axis=args.axis, method=args.method, engine="MuJoCo", samples=len(arrays["t"]), failure=failure,
                  independent_entrypoint="homie_pushcart.mujoco_viewer.make_viewer_runner; same runner.step as TwoStageViewer.run",
                  headless_only=True, no_running_viewer_touched=True, official_physics_unchanged=True,
                  checkpoint=metadata["checkpoint"], checkpoint_sha256=sha(metadata["checkpoint"]),
                  homie_checkpoint=str(homie_checkpoint), homie_sha256=sha(homie_checkpoint),
                  payload_kg=metadata["payload_kg"], dt=dt, seed=metadata["seed"],
                  commands_scale=env.commands_scale.tolist(), command_axes="[vx, vy, wz] in task/cart-heading convention",
                  applied_command_max_abs_error=float(np.abs(arrays["homie_input_frame"][:, :3] - arrays["homie_command"] * env.commands_scale).max()),
                  action_forwarding_verified=True, original_raw_comparisons=comparisons, periods=periods,
                  pose_check_semantics="Central difference of measured body COM xipos in world frame, transformed by measured cart yaw; yaw unwrapped before differencing. Finite differences have sampling error and are not substituted into paper raw data.",
                  observation_hook="Copies actual latest 76-dim HOMIE frame before original act_and_estimate, returns original tensors unchanged",
                  no_new_failure_or_response_cause_assumed=True)
    (args.output / f"{key}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(key=key, samples=len(arrays["t"]), failure=failure,
                          exact_replay=all(value["exactly_equal"] for value in comparisons.values()),
                          actual_input_command_max_abs_error=report["applied_command_max_abs_error"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["vy", "wz"])
    parser.add_argument("--method", choices=["baseline", "CWC"])
    parser.add_argument("--output", type=Path, default=ROOT / "logs/Picture/motion_ability/diagnostics/mujoco_axis_audit")
    args = parser.parse_args()
    if args.axis:
        assert args.method
        return replay(args)
    args.output.mkdir(parents=True, exist_ok=True)
    for axis in ["vy", "wz"]:
        for method in ["baseline", "CWC"]:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--axis", axis, "--method", method,
                       "--output", str(args.output)]
            with (args.output / f"{axis}_{method}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Audit child failed: {axis}/{method}; see log")
            report = json.loads((args.output / f"{axis}_{method}.json").read_text(encoding="utf-8"))
            print(json.dumps(dict(axis=axis, method=method, samples=report["samples"],
                                  exact_replay=all(value["exactly_equal"] for value in report["original_raw_comparisons"].values()))), flush=True)


if __name__ == "__main__":
    main()
