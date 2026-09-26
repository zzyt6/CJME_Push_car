"""Deterministic, physical MuJoCo V5 reaction/recovery experiments.

Every disturbance is paired with an undisturbed rollout of the same controller,
seed and initial state. Simulator truth is recorded only for evaluation; the
runner retains its causal sensor/observer interface. A paired trajectory is a
reference, not a perfect gait-phase-aligned counterfactual after perturbation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .paths import ROOT


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    onset_s: float
    release_s: float
    force_cart_n: tuple[float, float, float] = (0., 0., 0.)
    torque_cart_nm: tuple[float, float, float] = (0., 0., 0.)
    wheel_damping_factor: float = 1.


SCENARIOS = (
    Scenario("longitudinal_pulse", "pulse", 4., 4.2, (15., 0., 0.)),
    Scenario("lateral_pulse", "pulse", 4., 4.2, (0., 15., 0.)),
    Scenario("yaw_pulse", "pulse", 4., 4.2, torque_cart_nm=(0., 0., 2.)),
    Scenario("wheel_resistance_release", "wheel_damping", 3., 4., wheel_damping_factor=40.),
)

RECOVERY_BANDS = {
    "cart_twist": [.08, .08, .15],         # m/s, m/s, rad/s vs paired no-disturb rollout
    "normalized_robot_momentum": [.12, .12, .12, .10, .10, .10],  # p/M and L/(M*0.74)
    "robot_tilt_rad": .08,
    "hand_force_n": 20.,
    "hand_moment_nm": 4.,
    "dwell_s": .50,
    "momentum_height_m": .74,
    "task_twist_error": [.10, .08, .15],   # actual reference tracking, not paired error
    "absolute_robot_tilt_rad": .35,
    "wheel_lift_m": .03,
}

TRACKING_WINDOW_START_S = 2.0


def _tracking_window(times):
    """Use the same post-ramp tracking window for reference and disturbed runs."""
    times = np.asarray(times)
    steady = times >= TRACKING_WINDOW_START_S
    fallback = not bool(steady.any())
    if fallback:
        steady = np.ones(len(times), dtype=bool)
    return steady, {
        "tracking_window_requested_start_s": TRACKING_WINDOW_START_S,
        "tracking_window_start_s": float(times[steady][0]) if len(times) else None,
        "tracking_window_end_s": float(times[steady][-1]) if len(times) else None,
        "tracking_window_samples": int(steady.sum()),
        "tracking_window_fallback_to_available": fallback,
    }


def task_command(t: float) -> np.ndarray:
    """One second at rest, then a one-second ramp to 0.3 m/s."""
    return np.array([.3 * np.clip(t - 1., 0., 1.), 0., 0.])


def _first_recovery(times, in_band, release_s, dwell_s):
    """Require observed time coverage for the complete dwell, with no gap/rebound."""
    start = None
    differences = np.diff(times)
    nominal_dt = float(np.median(differences)) if len(differences) else math.inf
    previous = None
    for t, calm in zip(times, in_band):
        if previous is not None and t - previous > 1.5 * nominal_dt + 1e-9:
            start = None
        if t < release_s or not calm:
            start = None
        elif start is None:
            start = float(t)
        if start is not None and float(t) - start >= dwell_s - 1e-9:
            return {"time_s": max(0., start - release_s), "confirmed_at_s": float(t)}
        previous = t
    return None


def _disturbance_evidence(trace, reference, scenario, dt):
    """Verify delivery independently of whether the controller visibly reacts."""
    if scenario.kind == "pulse":
        samples = trace.get("physics_cart_applied_wrench", trace.get("applied_wrench"))
        sample_dt = float(trace.get("physics_dt_s", dt)) if "physics_cart_applied_wrench" in trace else dt
        if samples is None:
            return {"verified": False, "reason": "missing applied-wrench trace"}
        impulse = np.asarray(samples).sum(axis=0) * sample_dt
        target = np.r_[scenario.force_cart_n, scenario.torque_cart_nm] * (scenario.release_s - scenario.onset_s)
        measured = np.array([np.linalg.norm(impulse[:3]), np.linalg.norm(impulse[3:])])
        expected = np.array([np.linalg.norm(target[:3]), np.linalg.norm(target[3:])])
        complete = bool(np.isfinite(impulse).all() and np.all(np.abs(measured - expected) <= .10 * expected + 1e-5))
        return {"verified": complete and bool(expected.max() > 0), "actual_impulse_world_ns_nms": impulse.tolist(),
                "expected_force_moment_impulse_norms": expected.tolist(), "relative_tolerance": .10}
    n = min(len(trace["time_s"]), len(reference["time_s"]))
    if "wheel_damping" not in trace or "wheel_damping" not in reference:
        return {"verified": False, "reason": "missing wheel-damping trace"}
    before = np.asarray(reference["wheel_damping"][:n])
    after = np.asarray(trace["wheel_damping"][:n])
    active = after > before + 1e-12
    duration = float(active.sum() * dt)
    expected = scenario.release_s - scenario.onset_s
    return {"verified": bool(active.any() and abs(duration - expected) <= dt + 1e-8 and
                             np.allclose(after[active], before[active] * scenario.wheel_damping_factor)),
            "active_duration_s": duration, "expected_duration_s": expected}


def summarize_trial(trace: dict, reference: dict, scenario: Scenario, *, failure=None,
                    reference_failure=None, bands=None) -> dict:
    """Pure NumPy metrics: never substitute the horizon for an unsuccessful recovery."""
    bands = {**RECOVERY_BANDS, **(bands or {})}
    times = np.asarray(trace["time_s"])
    n = min(len(times), len(reference["time_s"]))
    result = {"scenario": asdict(scenario), "samples": len(times), "paired_samples": n,
              "failure": failure, "reference_failure": reference_failure,
              "recovery_bands": bands, "recovery_time_s": None, "recovery_confirmed_at_s": None,
              "recovered": False, "disturbance_rejected": False, "task_restored": False,
              "paired_return_time_s": None}
    if not len(times) or not n:
        result["recovery_status"] = "no_paired_samples"
        return result
    if not np.allclose(times[:n], reference["time_s"][:n], rtol=0, atol=1e-8):
        raise ValueError("paired traces must share the same sample times")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("sample times must be finite and strictly increasing")
    finite_keys = ("cart_twist", "command_reference", "robot_momentum", "robot_mass", "robot_tilt_rad", "hand_wrench")
    if any(not np.isfinite(np.asarray(sample[key])).all() for sample in (trace, reference) for key in finite_keys):
        result["recovery_status"] = "nonfinite_trace"
        return result
    dt = float(np.median(np.diff(times))) if len(times) > 1 else float(trace.get("dt_s", .02))
    steady, tracking_window = _tracking_window(times)
    error = trace["cart_twist"] - trace["command_reference"]
    hand = np.asarray(trace.get("physics_hand_wrench", trace["hand_wrench"]))
    if hand.ndim == 3:
        hand = hand.sum(axis=1)
    if not np.isfinite(hand).all() or np.any(np.asarray(trace["robot_mass"]) <= 0) or np.any(np.asarray(reference["robot_mass"]) <= 0):
        result["recovery_status"] = "nonfinite_or_invalid_physics"
        return result
    hdt = float(trace.get("physics_dt_s", dt)) if "physics_hand_wrench" in trace else dt
    result.update({
        **tracking_window,
        "cart_twist_tracking_rmse": np.sqrt(np.mean(error[steady] ** 2, axis=0)).tolist(),
        "raw_hand_force_peak_n": float(np.linalg.norm(hand[:, :3], axis=1).max()),
        "raw_hand_moment_peak_nm": float(np.linalg.norm(hand[:, 3:], axis=1).max()),
        "raw_hand_signed_impulse_ns_nms": (hand.sum(axis=0) * hdt).tolist(),
        "raw_hand_force_norm_integral_ns": float(np.linalg.norm(hand[:, :3], axis=1).sum() * hdt),
        "raw_hand_moment_norm_integral_nms": float(np.linalg.norm(hand[:, 3:], axis=1).sum() * hdt),
        "raw_hand_metric_rate_hz": 1. / hdt,
    })
    mass = np.asarray(trace["robot_mass"][:n]).reshape(n, 1)
    mass_ref = np.asarray(reference["robot_mass"][:n]).reshape(n, 1)
    height = bands["momentum_height_m"]
    scale = np.array([1., 1., 1., height, height, height])
    momentum = trace["robot_momentum"][:n]
    momentum_ref = reference["robot_momentum"][:n]
    deviation = momentum - momentum_ref
    normalized = momentum / (mass * scale) - momentum_ref / (mass_ref * scale)
    cart_deviation = trace["cart_twist"][:n] - reference["cart_twist"][:n]
    hand_deviation = trace["hand_wrench"][:n] - reference["hand_wrench"][:n]
    tilt_deviation = np.abs(trace["robot_tilt_rad"][:n] - reference["robot_tilt_rad"][:n])
    post = times[:n] > scenario.onset_s + 1e-9  # post-step samples: t=onset precedes injection
    pre = ~post
    prefix_keys = ["cart_twist", "robot_momentum", "hand_wrench"]
    prefix_keys.extend(key for key in ("sensor_wrench", "command_correction") if key in trace and key in reference)
    prefix_errors = {key: float(np.max(np.abs(trace[key][:n][pre] - reference[key][:n][pre])))
                     for key in prefix_keys} if pre.any() else {}
    result["paired_pre_event_max_abs_difference"] = prefix_errors
    result["paired_pre_event_match"] = bool(prefix_errors and max(prefix_errors.values()) <= 1e-9)
    result["event_reached"] = bool(post.any())
    if "feet_wrench" in trace and post.any():
        feet = np.asarray(trace["feet_wrench"][:n])[post]
        tangential = np.linalg.norm(feet[..., :2], axis=-1)
        normal = np.maximum(feet[..., 2], 0.)
        loaded = normal > 5.
        result["foot_friction_ratio_peak_loaded"] = float((tangential / np.maximum(normal, 5.))[loaded].max()) if loaded.any() else None
        result["total_foot_normal_load_min_n"] = float(normal.sum(axis=-1).min())
        result["post_event_no_loaded_foot_fraction"] = float((~loaded.any(axis=-1)).mean())
    if post.any():
        result.update({
            "paired_robot_linear_momentum_deviation_peak_ns": float(np.linalg.norm(deviation[post, :3], axis=1).max()),
            "paired_robot_angular_momentum_deviation_peak_nms": float(np.linalg.norm(deviation[post, 3:], axis=1).max()),
            "paired_cart_twist_rmse": np.sqrt(np.mean(cart_deviation[post] ** 2, axis=0)).tolist(),
            "paired_normalized_momentum_rmse": np.sqrt(np.mean(normalized[post] ** 2, axis=0)).tolist(),
            "paired_excess_hand_signed_impulse_ns_nms": (hand_deviation[post].sum(axis=0) * dt).tolist(),
            "paired_excess_hand_force_norm_integral_ns": float(np.linalg.norm(hand_deviation[post, :3], axis=1).sum() * dt),
            "paired_excess_hand_metric_rate_hz": 1. / dt,
        })
    paired_calm = (np.all(np.abs(cart_deviation) <= bands["cart_twist"], axis=1)
            & np.all(np.abs(normalized) <= bands["normalized_robot_momentum"], axis=1)
            & (tilt_deviation <= bands["robot_tilt_rad"])
            & (np.linalg.norm(hand_deviation[:, :3], axis=1) <= bands["hand_force_n"])
            & (np.linalg.norm(hand_deviation[:, 3:], axis=1) <= bands["hand_moment_nm"]))
    active = np.asarray(trace.get("disturbance_active", np.zeros(len(times), dtype=bool)))[:n].astype(bool)
    # Absolute task criteria prevent a stationary/failing nominal policy from
    # appearing successful simply because perturbed and unperturbed traces agree.
    task_calm = (np.all(np.abs(error[:n]) <= bands["task_twist_error"], axis=1)
                 & (np.abs(trace["robot_tilt_rad"][:n]) <= bands["absolute_robot_tilt_rad"]))
    if "wheel_lift_m" in trace:
        task_calm &= trace["wheel_lift_m"][:n] <= bands["wheel_lift_m"]
    result["left_paired_recovery_band"] = bool(np.any(post & ~paired_calm))
    result["post_event_task_in_band_fraction"] = float(task_calm[post].mean()) if post.any() else None
    delivery = _disturbance_evidence(trace, reference, scenario, dt)
    result["disturbance_delivery"] = delivery
    paired_return = _first_recovery(times[:n], paired_calm & ~active, scenario.release_s, bands["dwell_s"])
    result["paired_return_time_s"] = paired_return["time_s"] if paired_return else None
    eligible = np.ones(n, dtype=bool)
    departure = np.flatnonzero(post & ~paired_calm)
    if len(departure):
        eligible[:departure[0]] = False
    recovery = _first_recovery(times[:n], paired_calm & task_calm & ~active & eligible,
                               scenario.release_s, bands["dwell_s"])
    if prefix_errors and not result["paired_pre_event_match"]:
        result["recovery_status"] = "reference_prefix_mismatch"
    elif reference_failure:
        result["recovery_status"] = "reference_terminated"
    elif failure:
        result["recovery_status"] = "terminated"
    elif not post.any():
        result["recovery_status"] = "event_not_reached"
    elif not delivery["verified"]:
        result["recovery_status"] = "disturbance_not_verified"
    elif recovery is None:
        result["recovery_status"] = "task_not_restored" if paired_return else "not_recovered_within_horizon"
    elif not result["left_paired_recovery_band"]:
        result.update(disturbance_rejected=True, task_restored=True, recovery_status="disturbance_rejected_within_band")
    else:
        result.update(recovered=True, task_restored=True, recovery_status="recovered", recovery_time_s=recovery["time_s"],
                      recovery_confirmed_at_s=recovery["confirmed_at_s"])
    return result


def _centroidal(env, body):
    state = env.centroidal_state(body)
    state["momentum"] = np.concatenate((state["linear_momentum"], state["angular_momentum"]))
    return state


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _run_trial(runner, scenario, args, baseline_damping):
    """Run until first failure; never hide a failure with automatic episode resets."""
    import torch
    env = runner.env
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.model.dof_damping[:] = baseline_damping
    runner.set_payload(args.payload)
    runner.reset(caster_angle=0.)
    wheel_dof = int(env.model.joint("wheel_0_joint").dofadr[0])
    pulse_applied = False
    records, physics_records = [], []
    failure = None
    wheel_lift_steps = 0
    for step in range(args.steps):
        t = step * runner.dt
        if scenario is not None:
            if scenario.kind == "pulse" and not pulse_applied and t >= scenario.onset_s - 1e-9:
                yaw = runner.ts["cart_yaw"]
                c, s = math.cos(yaw), math.sin(yaw)
                rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                env.inject_cart_wrench(rotation @ scenario.force_cart_n, rotation @ scenario.torque_cart_nm,
                                      scenario.release_s - scenario.onset_s)
                pulse_applied = True
            elif scenario.kind == "wheel_damping":
                active = scenario.onset_s <= t < scenario.release_s
                env.model.dof_damping[wheel_dof] = baseline_damping[wheel_dof] * (scenario.wheel_damping_factor if active else 1.)
        _, fallen = runner.step(task_command(t))
        ts = runner.ts
        robot = _centroidal(env, env.pelvis_id)
        cart = _centroidal(env, env.cart_id)
        ground = env.ground_contact_wrenches()
        reaction = runner.reaction
        records.append({
            "time_s": (step + 1) * runner.dt,
            "command_reference": runner.cmd_ref.copy(),
            "cart_twist": np.array([*ts["cart_v"], ts["cart_wz"]]),
            "robot_momentum": _array(robot["momentum"]).copy(),
            "robot_mass": float(robot["mass"]),
            "robot_com": _array(robot["com"]).copy(),
            "cart_momentum": _array(cart["momentum"]).copy(),
            "robot_tilt_rad": ts["robot_tilt_rad"],
            "wheel_lift_m": ts["wheel_lift_m"],
            "cart_yaw": ts["cart_yaw"],
            "hand_wrench": np.r_[ts["cart_force_w"], ts["cart_moment_w"]],
            "applied_wrench": _array(env.cart_applied_wrench).copy(),
            "disturbance_active": bool(scenario is not None and
                (np.linalg.norm(_array(env.cart_applied_wrench)) > 1e-10 if scenario.kind == "pulse"
                 else scenario.onset_s <= t < scenario.release_s)),
            "required_wrench": _array(reaction.required_wrench).reshape(3).copy(),
            "aligned_required_wrench": _array(reaction.aligned_required_wrench).reshape(3).copy(),
            "sensor_wrench": _array(reaction.filtered_wrench).reshape(6).copy(),
            "excess_wrench": _array(reaction.excess_wrench).reshape(3).copy(),
            "command_correction": _array(reaction.command_correction).reshape(3).copy(),
            "recovery_gate": float(_array(reaction.recovery_gate).reshape(-1)[0]),
            "sensor_valid": float(_array(reaction.sensor_valid).reshape(-1)[0]),
            "feet_wrench": _array(ground["feet"]).copy(),
            "wheel_wrench": _array(ground["wheels"]).copy(),
            "wheel_damping": float(env.model.dof_damping[wheel_dof]),
        })
        for sample in env.physics_samples:
            physics_records.append({name: _array(value).copy() for name, value in sample.items()})
        wheel_lift_steps = wheel_lift_steps + 1 if ts["wheel_lift_m"] > runner.task_cfg.termination.wheel_lift_m else 0
        finite = all(np.isfinite(value).all() for value in records[-1].values())
        hand_peak = max((np.linalg.norm(value[:3]) for value in ts["hand_cart_wrenches"].values()), default=0.)
        failure_name = ("nonfinite" if not finite else "fall" if fallen else
                        "blowup" if (ts["cart_speed"] > runner.task_cfg.termination.blowup_cart_speed or
                                      hand_peak > runner.task_cfg.termination.blowup_force_n) else
                        "wheel_lift" if wheel_lift_steps >= runner.task_cfg.termination.wheel_lift_persist_steps else
                        runner.geometry_failure())
        if failure_name:
            failure = {"reason": failure_name, "time_s": (step + 1) * runner.dt}
            break
    trace = {key: np.asarray([record[key] for record in records]) for key in records[0]}
    if physics_records:
        for key in physics_records[0]:
            trace["physics_" + key] = np.asarray([record[key] for record in physics_records])
    trace["dt_s"] = np.asarray(runner.dt)
    trace["physics_dt_s"] = np.asarray(env.model.opt.timestep)
    trace["applied_impulse_ns_nms"] = _array(env.cart_pulse_impulse).copy()
    env.model.dof_damping[:] = baseline_damping
    return trace, failure


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reference_metrics(reference):
    error = reference["cart_twist"] - reference["command_reference"]
    tilt = reference["robot_tilt_rad"]
    steady, window = _tracking_window(reference["time_s"])
    return {**{"reference_" + key: value for key, value in window.items()},
            "reference_cart_twist_tracking_rmse": np.sqrt(np.mean(error[steady] ** 2, axis=0)).tolist()
            if steady.any() and np.isfinite(error).all() else None,
            "reference_robot_tilt_peak_rad": float(np.max(tilt)) if np.isfinite(tilt).all() else None}


def run_reaction_bench(args) -> dict:
    from homie_isaaclab.math_utils import class_to_dict
    from .mujoco_hl import make_runner
    if args.steps <= 0 or args.payload < 0 or not math.isfinite(args.payload):
        raise ValueError("steps must be positive and payload must be finite and nonnegative")
    if args.no_compensation and args.compare_compensation:
        raise ValueError("choose either --no-compensation or --compare-compensation")
    output = Path(args.output) if args.output else ROOT / "logs/bench" / ("v5_reaction_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    modes = [True, False] if args.compare_compensation else [not args.no_compensation]
    report = {"version": 5, "benchmark": "physical_reaction_recovery", "seed": args.seed,
              "metrics_revision": 3,
              "payload_kg": args.payload, "requested_steps": args.steps,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "policy": "trained_V5_actor" if args.checkpoint else "zero_high_level_action_with_frozen_HOMIE",
              "command": "rest until 1 s, linear ramp to 0.3 m/s by 2 s, then hold",
              "recovery_reference": "same-seed undisturbed rollout of the same controller; gait phases may diverge after disturbance",
              "signal_contract": "raw world-axis robot-on-cart hand wrench about whole-cart COM; robot momentum about robot COM; contact truth is evaluation-only",
              "comparison_contract": "each mode starts from the same reset and seed; its dynamics may differ before onset. Pulse direction follows that mode's cart heading at onset and is then fixed in world axes. Comparison is closed-loop performance, not a matched-state causal effect",
              "limitations": ["single-seed diagnostic, not a statistical performance claim",
                              "friction/contact model and wrist sensing differ between MuJoCo, Isaac and hardware",
                              "no-disturb reference is time aligned, not gait-phase aligned",
                              "recovery requires absolute task bands in addition to paired return; within-band disturbance rejection is separate from recovery",
                              "simulator geometry/fall/wheel-lift/blowup checks do not reproduce every Isaac grasp-overload termination"],
              "modes": {}, "output": str(output)}
    started = time.perf_counter()
    for enabled in modes:
        label = "compensation_on" if enabled else "compensation_off"
        runner, lowlevel = make_runner(SimpleNamespace(checkpoint=args.checkpoint, homie_checkpoint=getattr(args, "homie_checkpoint", None),
                                                      level=getattr(args, "level", 17)))
        runner.task_cfg.reaction.compensation_enabled = enabled
        runner.reaction.compensation_enabled = enabled
        baseline_damping = runner.env.model.dof_damping.copy()
        reference, reference_failure = _run_trial(runner, None, args, baseline_damping)
        np.savez_compressed(output / f"{label}_reference.npz", **reference)
        results = {}
        for scenario in SCENARIOS:
            trace, failure = _run_trial(runner, scenario, args, baseline_damping)
            np.savez_compressed(output / f"{label}_{scenario.name}.npz", **trace)
            result = summarize_trial(trace, reference, scenario, failure=failure, reference_failure=reference_failure)
            result["applied_impulse_ns_nms"] = trace["applied_impulse_ns_nms"].tolist()
            results[scenario.name] = result
            print(f"[v5-bench] {label} {scenario.name}: {result['recovery_status']}; recovery={result['recovery_time_s']}", flush=True)
        report["modes"][label] = {"compensation_enabled": enabled, "reference_failure": reference_failure,
                                  "config": class_to_dict(runner.task_cfg), "scenarios": results,
                                  **_reference_metrics(reference),
                                  "control_rate_hz": 1. / runner.dt,
                                  "physics_rate_hz": 1. / float(runner.env.model.opt.timestep)}
        report["homie_checkpoint"] = str(lowlevel)
        report["homie_checkpoint_sha256"] = _hash_file(lowlevel)
        report["scene"] = str(runner.env.xml_path)
        report["scene_sha256"] = _hash_file(runner.env.xml_path)
    if args.checkpoint:
        report["checkpoint_sha256"] = _hash_file(args.checkpoint)
    report["elapsed_s"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[v5-bench] wrote {output / 'summary.json'}", flush=True)
    return report
