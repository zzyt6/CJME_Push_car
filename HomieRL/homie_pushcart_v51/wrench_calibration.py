"""Bounded no-cart checks of V5's model-compensated palm joint-wrench estimate.

This diagnostic does not modify training, checkpoints, or compensation parameters.
V5 subtracts hand-subtree Newton-Euler inertial and gravity loads. HOMIE's standing
motion is retained: the no-contact residual measures remaining model, solver and
sampling error. This source check does not certify dynamic cart-contact accuracy.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from .paths import ROOT


PHASES = ("static", "arms", "force_positive", "force_negative",
          "torque_positive", "torque_negative", "recovery")

INTERPRETATION = [
    "No cart or grasp constraint is present. Static/dynamic residuals are model/solver errors, not cart contact force.",
    "V5 subtracts the hand-subtree Newton-Euler inertial and gravity wrench from the incoming palm joint wrench.",
    "The known external palm loads test sign and frame through signed positive/negative differences.",
    "HOMIE balances during this diagnostic; motion, contact changes and sampling errors contribute to probe errors.",
    "Passing this bounded no-cart test does not establish dynamic loaded-grasp accuracy or two-sided momentum closure.",
    "The diagnostic changes no checkpoint or compensation parameter; V5's production wrench formula differs from V4.",
]


def summarize_wrench_trace(trace: dict[str, np.ndarray], *, resets: int = 0) -> dict:
    """Evaluate each hand/environment independently; opposite errors cannot cancel.

    force/torque are the production proxy. Applied loads are external loads ON
    the hand, whereas the proxy is the reaction the hand exerts on its support.
    Thus a quasi-static signed load difference should be negative applied load.
    Torque probes use pure couples, avoiding a force application-point ambiguity.
    """
    phase = np.asarray(trace["phase"])
    selected = np.asarray(trace["selected"], dtype=bool)
    force = np.asarray(trace["force_w"], dtype=float)
    torque = np.asarray(trace["torque_w"], dtype=float)
    applied_force = np.asarray(trace["applied_force_w"], dtype=float)
    applied_torque = np.asarray(trace["applied_torque_w"], dtype=float)
    if force.ndim != 4 or force.shape[-2:] != (2, 3) or force.shape != torque.shape:
        raise ValueError("Expected wrench traces [time, environment, 2 hands, 3 axes]")
    if (len(phase) != len(force) or selected.shape != phase.shape
            or applied_force.shape != force.shape or applied_torque.shape != force.shape):
        raise ValueError("Trace lengths/shapes differ")
    finite = (np.isfinite(force).all(axis=(1, 2, 3)) & np.isfinite(torque).all(axis=(1, 2, 3))
              & np.isfinite(applied_force).all(axis=(1, 2, 3))
              & np.isfinite(applied_torque).all(axis=(1, 2, 3)))
    phase_stats = {}
    means = {}
    for index, name in enumerate(PHASES):
        mask = (phase == index) & selected & finite
        if not mask.any():
            raise ValueError(f"No finite analysis samples for phase {name}")
        f, m = force[mask], torque[mask]
        means[name] = (f.mean(0), m.mean(0), applied_force[mask].mean(0), applied_torque[mask].mean(0))
        phase_stats[name] = {
            "samples": int(mask.sum()),
            "force_bias_N": f.mean(0).tolist(),
            "torque_bias_Nm": m.mean(0).tolist(),
            "force_vector_rms_N": np.sqrt(np.mean(np.sum(f * f, axis=-1), axis=0)).tolist(),
            "force_ac_vector_rms_N": np.sqrt(np.mean(np.sum((f - f.mean(0)) ** 2, axis=-1), axis=0)).tolist(),
            "torque_vector_rms_Nm": np.sqrt(np.mean(np.sum(m * m, axis=-1), axis=0)).tolist(),
            "torque_ac_vector_rms_Nm": np.sqrt(np.mean(np.sum((m - m.mean(0)) ** 2, axis=-1), axis=0)).tolist(),
        }
    probes = {}
    for quantity, offset, units, tolerance in (("force", 0, "N", 1.0), ("torque", 1, "Nm", .05)):
        positive = means[quantity + "_positive"]
        negative = means[quantity + "_negative"]
        measured = .5 * (positive[offset] - negative[offset])
        expected = -.5 * (positive[offset + 2] - negative[offset + 2])
        error = np.linalg.norm(measured - expected, axis=-1)
        # A missing/zero injected probe cannot be accepted as a correct source.
        expected_norm = np.linalg.norm(expected, axis=-1)
        passed = (error <= tolerance) & (expected_norm > tolerance)
        probes[quantity] = {
            "expected_signed_response_" + units: expected.tolist(),
            "measured_signed_response_" + units: measured.tolist(),
            "error_norm_" + units: error.tolist(),
            "tolerance_" + units: tolerance,
            "passed_by_environment_hand": passed.tolist(),
            "passed": bool(passed.all()),
        }
    return {
        "known_load_source_validation_passed": bool(all(p["passed"] for p in probes.values())
                                                     and finite.all() and resets == 0),
        "dynamic_contact_wrench_calibrated": False,
        "nonfinite_steps": int((~finite).sum()),
        "robot_resets": int(resets),
        "phases": phase_stats,
        "known_load_probes": probes,
        "version": 5,
        "wrench_source": "incoming_joint_minus_hand_subtree_newton_euler_inertial_and_gravity_wrench",
        "interpretation": list(INTERPRETATION),
    }


def run_wrench_calibration(args, launcher, root: Path = ROOT) -> dict:
    """Run seven no-cart phases, writing report.json, report.md and raw traces.npz."""
    import torch
    from homie_isaaclab.math_utils import quat_mul
    from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from .sim_env import PushCartSimEnv
    from .teleop import play_config

    num_envs = int(getattr(args, "num_envs", 4))
    steps = int(getattr(args, "steps", 180))
    if not 1 <= num_envs <= 16 or steps < 120:
        raise ValueError("Use 1..16 environments and at least 120 control steps per phase")
    device = args.device
    checkpoint_arg = getattr(args, "checkpoint", None)
    checkpoint = Path(checkpoint_arg) if checkpoint_arg else frozen_checkpoint()
    output_arg = getattr(args, "output", None)
    output = Path(output_arg) if output_arg else root / "logs/bench" / ("v5_wrench_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    headless = bool(getattr(launcher, "_headless", True))
    cfg = play_config(num_envs=num_envs, episode_length_s=steps * len(PHASES) * .02 + 20.)
    env = PushCartSimEnv(cfg, root, device, headless, with_cart=False)
    policy = HimLowLevelPolicy(checkpoint, device)
    obs, _ = env.reset()
    env.set_command(0., 0., 0.)
    # Rotate the entire no-cart articulation, including its arms; world load
    # directions stay constant to expose an erroneous body/world conversion.
    yaw = torch.arange(num_envs, device=device, dtype=torch.float32) * (2. * math.pi / num_envs)
    yaw_q = torch.zeros(num_envs, 4, device=device)
    yaw_q[:, 2], yaw_q[:, 3] = torch.sin(yaw / 2.), torch.cos(yaw / 2.)
    root_pose = env.robot.data.root_link_pose_w.torch.clone()
    root_pose[:, 3:] = quat_mul(yaw_q, root_pose[:, 3:])
    env.robot.write_root_pose_to_sim_index(root_pose=root_pose)
    env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_envs, 6, device=device))
    force_probe = torch.tensor([[5., 2., 1.], [4., -2., 1.]], device=device).expand(num_envs, -1, -1)
    torque_probe = torch.tensor([[.10, -.08, .15], [-.10, .08, .15]], device=device).expand(num_envs, -1, -1)
    zeros = torch.zeros(num_envs, 2, 3, device=device)
    records = {key: [] for key in ("time_s", "phase", "selected", "force_w", "torque_w", "applied_force_w",
                                   "applied_torque_w", "palm_velocity_w", "root_yaw_rad", "upper_joint_pos_rad")}
    resets = 0
    try:
        with torch.inference_mode():
            for phase_id, phase in enumerate(PHASES):
                print(f"[wrenchcal] phase={phase} steps={steps}", flush=True)
                for step in range(steps):
                    if not launcher.app.is_running():
                        raise RuntimeError("Simulation closed before wrench calibration completed")
                    delta = torch.zeros_like(env.upper_delta)
                    if phase == "arms":
                        t = step * env.dt
                        ramp = min(1., t / .5)
                        for side, phase_shift in (("left", 0.), ("right", math.pi / 2.)):
                            for joint, amplitude, frequency in (("shoulder_pitch", .12, .8),
                                                                ("shoulder_roll", .08, 1.0), ("elbow", .12, 1.2)):
                                index = env.dof_names.index(f"{side}_{joint}_joint") - env.num_lower_dof
                                delta[:, index] = ramp * amplitude * math.sin(2. * math.pi * frequency * t + phase_shift) / cfg.control.action_scale
                    env.set_upper_delta(delta)
                    f = force_probe if phase == "force_positive" else -force_probe if phase == "force_negative" else zeros
                    m = torque_probe if phase == "torque_positive" else -torque_probe if phase == "torque_negative" else zeros
                    # Loads act at the palm CoM. Pure-torque phases have no
                    # force moment; force phases are checked only for force.
                    env.robot.permanent_wrench_composer.set_forces_and_torques_index(
                        forces=f.contiguous(), torques=m.contiguous(), body_ids=env.palm_body_ids.to(torch.int32), is_global=True)
                    obs, _, _, _, info, reset_ids, _ = env.step(policy(obs))
                    resets += int(len(reset_ids))
                    force, torque, _ = env.grasp_wrench()
                    records["time_s"].append((phase_id * steps + step + 1) * env.dt)
                    records["phase"].append(phase_id)
                    # Load probes use settled tails; dynamic phase samples
                    # the oscillation after its ramp, not just its final half.
                    records["selected"].append(step >= (int(.2 * steps) if phase == "arms" else int(.6 * steps)))
                    for key, value in (("force_w", force), ("torque_w", torque), ("applied_force_w", f),
                                       ("applied_torque_w", m), ("palm_velocity_w", env.rigid_body_states[:, env.palm_body_ids, 7:13]),
                                       ("root_yaw_rad", env.yaw), ("upper_joint_pos_rad", env.dof_pos[:, env.num_lower_dof:])):
                        records[key].append(value.cpu().numpy().copy())
        trace = {key: np.asarray(value) for key, value in records.items()}
        np.savez_compressed(output / "traces.npz", **trace)
        report = summarize_wrench_trace(trace, resets=resets)
        report["result"] = "PASS" if report["known_load_source_validation_passed"] else "FAIL"
        report.update({"checkpoint": str(checkpoint), "num_envs": num_envs, "steps_per_phase": steps,
                       "control_dt_s": env.dt, "initial_yaw_rad": yaw.cpu().tolist(), "phase_names": PHASES,
                       "arm_target_amplitudes_rad": {"shoulder_pitch": .12, "shoulder_roll": .08, "elbow": .12},
                       "arm_target_frequencies_hz": {"shoulder_pitch": .8, "shoulder_roll": 1., "elbow": 1.2},
                       "output": str(output)})
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        rows = []
        for name in ("static", "arms", "recovery"):
            p = report["phases"][name]
            rows.append(f"| {name} | {np.max(p['force_vector_rms_N']):.4f} | {np.max(p['force_ac_vector_rms_N']):.4f} | {np.max(p['torque_vector_rms_Nm']):.4f} |")
        (output / "report.md").write_text(
            "# V5 model-compensated no-cart palm-wrench validation\n\n"
            f"Known-load source validation passed: **{report['known_load_source_validation_passed']}**. "
            "Dynamic contact wrench calibrated: **False**.\n\n"
            f"{num_envs} yaw variants, {steps} steps per phase; no cart; frozen HOMIE standing policy. "
            "Values below are maximum across environment/hand pairs.\n\n"
            "| Phase | Force vector RMS (N) | Force AC RMS (N) | Torque vector RMS (Nm) |\n"
            "|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
            f"Maximum signed force-probe error: {np.max(report['known_load_probes']['force']['error_norm_N']):.4f} N; "
            f"torque-probe error: {np.max(report['known_load_probes']['torque']['error_norm_Nm']):.4f} Nm.\n\n"
            + "\n".join("- " + line for line in report["interpretation"]) + "\n\n"
            "Per-environment/hand/axis statistics: `report.json`. Raw time series: `traces.npz`.\n",
            encoding="utf-8")
        print(f"[wrenchcal] source_pass={report['known_load_source_validation_passed']} dynamic_calibrated=False output={output}", flush=True)
        return report
    finally:
        env.robot.permanent_wrench_composer.reset()
        env.set_upper_delta(torch.zeros_like(env.upper_delta))
