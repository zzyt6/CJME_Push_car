"""Collect six independent official-MuJoCo paper motion trials, without a viewer.

Each axis/method starts in a fresh process with a full reset and no warmup. The
raw output schema matches collect_motion_ability_isaac.py and the shared plotter.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
NONCE = "CWC_MUJOCO_MOTION_RUN"
AXES = ("vx", "vy", "wz")
METHODS = ("baseline", "CWC")


def protocol(axis, idle_s=4., plateau_s=8.):
    values = {"vx": [0., .3, .5, .8], "vy": [0., .15, .3], "wz": [0., .15, .3]}[axis]
    stages, start = [], 0.
    for index, value in enumerate(values):
        end = start + (idle_s if index == 0 else plateau_s)
        stages.append(dict(start_s=start, end_s=end, value=value))
        start = end
    return stages


def schedule(axis, time_s, stages):
    stage = next((stage for stage in stages if time_s < stage["end_s"] - 1e-9), stages[-1])
    command = np.zeros(3, dtype=float)
    command[AXES.index(axis)] = stage["value"]
    return command


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_snapshot(policy):
    env, runner = policy.env, policy.runner
    m, d = env.model, env.data
    result = {"model." + name: getattr(m, name).copy() for name in
              ("body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_friction", "geom_pos",
               "geom_size", "dof_damping", "dof_armature", "dof_frictionloss", "eq_solref", "eq_solimp")}
    result.update({"state." + name: getattr(d, name).copy() for name in ("qpos", "qvel", "ctrl")})
    result.update({"controller.p_gains": env.p_gains.copy(), "controller.d_gains": env.d_gains.copy(),
                   "controller.homie_history": env.obs_hist.copy(), "controller.actor_history": runner.actor_hist.copy(),
                   "controller.estimator_history": runner.estimator_obs.numpy().copy(),
                   "controller.command": runner.cmd_task.copy(), "controller.reference": runner.cmd_ref.copy(),
                   "controller.homie_command": runner.cmd_homie.copy()})
    return result


def run_child(args):
    run_id = os.environ.get(NONCE)
    if not run_id:
        raise RuntimeError("A child requires the parent process nonce")
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    key = f"{args.axis}_{args.method}"
    try:
        import mujoco
        import torch
        from homie_pushcart_v6.baseline import config_dict
        from pushcart_bench.policies_v6 import V6Policy

        torch.set_num_threads(1)
        policy = V6Policy(args.checkpoint, level=17)
        runner, env = policy.runner, policy.env
        if args.method == "baseline":
            # This is the runner's established all-18-zero residual path. The
            # plant, frozen HOMIE, filters, history updates and grasp are shared.
            runner.hl = None
        policy.reset(payload_kg=args.payload, caster_angle=0., seed=args.seed)
        assert runner.v6_policy.iteration == args.iteration
        m, d = env.model, env.data
        assert not np.any(d.qvel) and d.time == 0.
        assert not np.any(runner.cmd_task) and not np.any(runner.cmd_ref) and not np.any(runner.cmd_homie)
        assert runner.idle_correction is None
        initial = physics_snapshot(policy)
        np.savez_compressed(raw / f"{key}_initial_physics.npz", **initial)
        stages = protocol(args.axis, args.idle_s, args.plateau_s)
        duration, dt = stages[-1]["end_s"], policy.dt
        steps = round(duration / dt)
        records, failure = {}, None
        with torch.inference_mode():
            for tick in range(steps):
                interval_start = tick * dt
                command = schedule(args.axis, interval_start, stages)
                policy.step(command)
                failure = policy.failure()
                cart_p, cart_q, _, _ = env.cart_state()
                robot_q = d.xquat[env.pelvis_id].copy()
                sample = dict(t=(tick + 1) * dt, interval_start=interval_start,
                              command=runner.cmd_task, reference=runner.cmd_ref, homie_command=runner.cmd_homie,
                              cart_velocity=policy.velocity(),
                              cart_pose=np.r_[cart_p, cart_q[1:], cart_q[0]],
                              robot_pose=np.r_[d.xpos[env.pelvis_id], robot_q[1:], robot_q[0]],
                              high_action=runner.hl_actions, upper_delta_q=runner.upper_delta_q,
                              lower_action=env.actions, joint_position=env.dof_pos, joint_velocity=env.dof_vel,
                              joint_torque_proxy=d.ctrl[env.act_idx],
                              hand_force_moment_world=np.stack([runner.ts["hand_cart_wrenches"][side] for side in ("left", "right")]),
                              pred_cart_velocity=runner.pred_cart_velocity, pred_hand_wrench=runner.pred_hands,
                              terminated=failure is not None, timeout=False)
                for name, value in sample.items():
                    records.setdefault(name, []).append(np.asarray(value).copy())
                if failure:
                    break
        arrays = {name: np.asarray(value) for name, value in records.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        if args.method == "baseline":
            assert not np.any(arrays["high_action"])
        zero = arrays["interval_start"] < args.idle_s - 1e-9
        assert not np.any(arrays["homie_command"][zero])
        assert not np.any(arrays["upper_delta_q"][zero])
        np.savez_compressed(raw / f"{key}.npz", **arrays)
        with (raw / f"{key}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", "interval_start_s"] +
                            [f"{prefix}_{axis}" for prefix in ("command", "reference", "actual", "homie") for axis in AXES] +
                            ["terminated", "timeout"])
            for index, time_s in enumerate(arrays["t"]):
                writer.writerow([time_s, arrays["interval_start"][index], *arrays["command"][index],
                                 *arrays["reference"][index], *arrays["cart_velocity"][index],
                                 *arrays["homie_command"][index], int(arrays["terminated"][index]), 0])
        final_physics = physics_snapshot(policy)
        np.savez_compressed(raw / f"{key}_final_physics.npz", **final_physics)
        for name in initial:
            if name.startswith("model.") or name in ("controller.p_gains", "controller.d_gains"):
                np.testing.assert_array_equal(initial[name], final_physics[name], err_msg=name)
        # All descendants of the cart root, including wheels and swivel bodies.
        descendants = []
        for body in range(m.nbody):
            ancestor = body
            while ancestor and ancestor != runner.cart_body:
                ancestor = int(m.body_parentid[ancestor])
            if ancestor == runner.cart_body:
                descendants.append(body)
        report = dict(result="PASS_EXECUTION", run_id=run_id, engine="MuJoCo", mujoco_version=mujoco.__version__,
                      axis=args.axis, component=AXES.index(args.axis), method=args.method, level=17,
                      payload_kg=args.payload, cart_assembly_mass_kg=float(m.body_mass[descendants].sum()),
                      checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint), iteration=args.iteration,
                      homie_checkpoint=str(policy.homie_checkpoint), homie_sha256=sha(policy.homie_checkpoint),
                      dt=dt, physics_dt=float(m.opt.timestep), stages=stages, duration_s=duration, samples=len(arrays["t"]),
                      completed=len(arrays["t"]) == steps and failure is None, failure=failure,
                      seed=args.seed, device="cpu", num_envs=1, independent_simulator_process=True,
                      baseline_definition="Zero all 18 high-level residuals; same frozen HOMIE, fixed grasp, filters and official plant",
                      CWC_definition="V6 deterministic Actor mean with learned estimators; no training updates",
                      velocity_source="Raw MuJoCo cart root COM velocity via mj_objectVelocity; translation in cart-heading axes, angular velocity world-z",
                      velocity_is_estimate=False, velocity_lowpass_applied_to_saved_actual=False,
                      sampling="Post-physics, pre-reset; t is interval end; interval_start is command application time",
                      pose_quaternion_order="xyzw", reference_tau_s=runner.task_cfg.commands.ref_tau_s,
                      homie_command_tau_s=runner.task_cfg.actions.cmd_filter_tau_s,
                      wz_schedule_basis="User-selected independent angular-axis steps 0, .15, .3 rad/s",
                      vy_requested_extrapolation=args.axis == "vy" and .3 > runner.task_cfg.commands.vy_max,
                      reset="Fresh process, mj_resetData, preset pose, zero velocities, zero commands and fresh histories",
                      warmup_s=0., automatic_resets=0, initial_velocity_exactly_zero=True,
                      zero_stage_homie_command_and_upper_residual_exactly_zero=True,
                      termination_checks="Official MuJoCo policy.failure: robot fall, cart yaw, cart distance and cart tilt; stop without reset",
                      npz_sha256=sha(raw / f"{key}.npz"), physics_contract=policy.describe()["physics"],
                      nominal_parameters_verified_unchanged=True, live_model_ablation=False,
                      randomization_effective=False, observation_noise_effective=False, action_sampling=False,
                      arrays={name: list(value.shape) for name, value in arrays.items()},
                      cfg=config_dict(env.cfg), task_cfg=config_dict(runner.task_cfg))
        (raw / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({name: report[name] for name in ("axis", "method", "completed", "samples", "failure", "cart_assembly_mass_kg")}), flush=True)
    except BaseException:
        error = traceback.format_exc()
        (args.output / f"{key}.failure.json").write_text(json.dumps(dict(run_id=run_id, error=error), indent=2), encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--payload", type=float, default=15.)
    parser.add_argument("--idle_s", type=float, default=4.)
    parser.add_argument("--plateau_s", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/Picture/motion_ability/mujoco")
    parser.add_argument("--axis", choices=AXES, help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--iteration", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if os.environ.get(NONCE):
        return run_child(args)
    if args.idle_s <= 0 or args.plateau_s <= 0:
        parser.error("Stage durations must be positive")
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    import torch
    args.checkpoint = resolve_v6_checkpoint(args.checkpoint or "latest").resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.iteration = int(checkpoint["iter"])
    del checkpoint
    checkpoint_hash = sha(args.checkpoint)
    run_id = uuid.uuid4().hex
    attempt = args.output / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{run_id[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_id=run_id, engine="MuJoCo", checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                    iteration=args.iteration, payload_kg=args.payload, seed=args.seed, device="cpu",
                    protocol={axis: protocol(axis, args.idle_s, args.plateau_s) for axis in AXES},
                    independence="6 separate simulator processes, one fresh reset per axis/method; no warmup",
                    attempt_dir=str(attempt.resolve()), runs=[])
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for axis in AXES:
        for method in METHODS:
            key = f"{axis}_{method}"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--axis", axis, "--method", method,
                       "--checkpoint", str(args.checkpoint), "--iteration", str(args.iteration),
                       "--payload", str(args.payload), "--idle_s", str(args.idle_s), "--plateau_s", str(args.plateau_s),
                       "--seed", str(args.seed), "--output", str(attempt)]
            print(f"[motion-mujoco] starting {key}", flush=True)
            with (attempt / f"{key}.log").open("w", encoding="utf-8") as handle:
                child = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **{NONCE: run_id}),
                                       stdout=handle, stderr=subprocess.STDOUT)
            report_path = attempt / "raw" / f"{key}.json"
            if child.returncode or not report_path.is_file():
                raise RuntimeError(f"{key} failed, exit {child.returncode}; see {attempt / (key + '.log')}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert report["run_id"] == run_id and report["result"] == "PASS_EXECUTION"
            assert report["checkpoint_sha256"] == checkpoint_hash
            assert report["npz_sha256"] == sha(report_path.with_suffix(".npz"))
            report.update(parent_verified=True, child_exit_code=child.returncode)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["runs"].append({name: report[name] for name in ("axis", "method", "completed", "samples", "failure", "npz_sha256")})
            print(f"[motion-mujoco] {key}: completed={report['completed']}, samples={report['samples']}", flush=True)
    reference, equivalence = None, []
    zero_reference = None
    for axis in AXES:
        for method in METHODS:
            with np.load(attempt / "raw" / f"{axis}_{method}_initial_physics.npz") as archive:
                current = {name: archive[name] for name in archive.files}
            if reference is None:
                reference = current
            for name in reference:
                np.testing.assert_array_equal(reference[name], current[name], err_msg=f"{axis}/{method}/{name}")
            with np.load(attempt / "raw" / f"{axis}_{method}.npz") as archive:
                zero = archive["interval_start"] < args.idle_s - 1e-9
                current_zero = {name: archive[name][zero] for name in ("cart_velocity", "cart_pose", "robot_pose", "lower_action")}
            if zero_reference is None:
                zero_reference = current_zero
            for name in zero_reference:
                np.testing.assert_array_equal(zero_reference[name], current_zero[name], err_msg=f"zero-stage/{axis}/{method}/{name}")
            equivalence.append(dict(axis=axis, method=method, all_initial_physics_and_state_equal=True,
                                    zero_stage_physical_trajectory_equal=True, arrays=len(current)))
    assert sha(args.checkpoint) == checkpoint_hash
    manifest["validation"] = dict(initial_physics=equivalence, checkpoint_unchanged=True,
                                   all_programs_completed=all(run["completed"] for run in manifest["runs"]))
    (attempt / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    destination = args.output / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    for path in (attempt / "raw").iterdir():
        shutil.copy2(path, destination / path.name)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[motion-mujoco] verified data: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
