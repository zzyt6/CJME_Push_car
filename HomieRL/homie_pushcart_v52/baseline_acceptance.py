"""Predeclared undisturbed task gate; raw and bandwidth-limited errors stay separate.

This module never changes a recovery verdict or a simulation parameter. A task
gate is evaluated per uninterrupted trial; a terminated trial cannot pass by
averaging its post-reset trajectory with its initial one.
"""
from __future__ import annotations

import numpy as np

TASKS = ("hold", "forward", "backward", "turn_left", "turn_right", "stop", "turn_reversal")
PROTOCOL = {
    "revision": 2, "duration_s": 12., "start_s": 1., "switch_s": 6.,
    "scoring_start_s": 3., "post_switch_scoring_start_s": 8.,
    "twist_rmse_limits": [.10, .08, .15], "commanded_gain_min": .7,
    "command_rms_min": .10, "diagnostic_filter_tau_s": .20,
    "hold_displacement_limit_m": .10, "stop_dwell_s": .5,
    "stop_deadline_after_switch_s": 2., "stop_twist_bands": [.08, .08, .15],
    "gate": "raw RMSE and commanded-axis progress in EACH steady segment, no termination; hold drift/stop dwell also required",
    "filtered_role": "separate fixed-bandwidth diagnostic; never substitutes for raw gate",
}


def task_command(name, time_s):
    if name not in TASKS:
        raise ValueError(f"Unknown task {name}")
    if time_s < PROTOCOL["start_s"] or name == "hold":
        return [0., 0., 0.]
    if name == "backward":
        return [-.3, 0., 0.]
    if name == "stop" and time_s >= PROTOCOL["switch_s"]:
        return [0., 0., 0.]
    yaw = (.2 if name == "turn_left" else -.2 if name == "turn_right" else
           (.2 if time_s < PROTOCOL["switch_s"] else -.2) if name == "turn_reversal" else 0.)
    return [.3, 0., yaw]


def rotate_xy(vectors, angle):
    values = np.array(vectors, dtype=np.float64, copy=True)
    c, s = np.cos(angle), np.sin(angle)
    values[..., 0] = c * vectors[..., 0] - s * vectors[..., 1]
    values[..., 1] = s * vectors[..., 0] + c * vectors[..., 1]
    return values


def _first_dwell(times, inside, start, duration):
    begin = None
    for t, good in zip(times, inside):
        if t < start:
            continue
        if not good:
            begin = None
        elif begin is None:
            begin = t
        if begin is not None and t - begin >= duration - 1e-8:
            return float(begin - start), float(t - start)
    return None, None


def _final_dwell(times, inside, start, duration):
    selected = np.flatnonzero(times >= start)
    if not len(selected) or not inside[-1]:
        return None, None
    outside = selected[~inside[selected]]
    first = int(outside[-1]+1) if len(outside) else int(selected[0])
    if first >= len(times) or times[-1] - times[first] < duration - 1e-8:
        return None, None
    return float(times[first]-start), float(times[first]+duration-start)


def summarize_trial(trace, task, failed=False):
    """Summarize [time,3] cart-COM twist in current cart axes, about one episode."""
    time = np.asarray(trace["time_s"], dtype=float)
    actual, ref = (np.asarray(trace[k], dtype=float) for k in ("actual", "reference"))
    yaw = np.asarray(trace["cart_yaw"], dtype=float)
    position = np.asarray(trace["cart_com"], dtype=float)
    finite = (np.isfinite(time).all() and np.isfinite(actual).all() and np.isfinite(ref).all()
              and np.isfinite(yaw).all() and np.isfinite(position).all())
    if (len(time) < 2 or not finite or not np.all(np.diff(time) > 0)
            or actual.shape != (len(time), 3) or ref.shape != actual.shape):
        return {"pass": False, "status": "invalid_trace"}
    dt = np.diff(time)
    if np.max(dt) > 1.5 * np.median(dt):
        return {"pass": False, "status": "gapped_trace"}
    complete = time[-1] >= PROTOCOL["duration_s"] - 1e-6
    scored = time >= PROTOCOL["scoring_start_s"]
    # Score both steady portions, retaining transients in saved raw traces.
    if task in ("stop", "turn_reversal"):
        scored &= ((time < PROTOCOL["switch_s"]) | (time >= PROTOCOL["post_switch_scoring_start_s"]))
    if not scored.any():
        return {"pass": False, "status": "no_scored_samples"}
    error = actual - ref
    world_error = rotate_xy(error, yaw)
    filtered = np.zeros_like(error)
    state = world_error[0].copy()
    for i in range(len(time)):
        if i:
            state += -np.expm1(-dt[i-1] / PROTOCOL["diagnostic_filter_tau_s"]) * (world_error[i] - state)
        filtered[i] = rotate_xy(state, -yaw[i])
    rmse = np.sqrt(np.mean(error[scored] ** 2, axis=0))
    filtered_rmse = np.sqrt(np.mean(filtered[scored] ** 2, axis=0))
    ref_energy = np.sum(ref[scored] ** 2, axis=0)
    commanded = np.sqrt(ref_energy / scored.sum()) >= PROTOCOL["command_rms_min"]
    gains = np.sum(actual[scored] * ref[scored], axis=0) / np.maximum(ref_energy, 1e-12)
    gain_ok = bool(np.all(gains[commanded] >= PROTOCOL["commanded_gain_min"]))
    raw_ok = bool(np.all(rmse <= PROTOCOL["twist_rmse_limits"]))
    filtered_ok = bool(np.all(filtered_rmse <= PROTOCOL["twist_rmse_limits"]))
    segments = {}
    segment_masks = {"steady": scored}
    if task in ("stop", "turn_reversal"):
        segment_masks = {"before_switch": scored & (time < PROTOCOL["switch_s"]),
                         "after_switch": scored & (time >= PROTOCOL["post_switch_scoring_start_s"])}
    for name, mask in segment_masks.items():
        if not mask.any():
            segments[name] = {"pass": False, "samples": 0}
            raw_ok = gain_ok = False
            continue
        segment_rmse = np.sqrt(np.mean(error[mask] ** 2, axis=0))
        energy = np.sum(ref[mask] ** 2, axis=0)
        commanded_axis = np.sqrt(energy / mask.sum()) >= PROTOCOL["command_rms_min"]
        gain = np.sum(actual[mask] * ref[mask], axis=0) / np.maximum(energy, 1e-12)
        tracking_pass = bool(np.all(segment_rmse <= PROTOCOL["twist_rmse_limits"]))
        progress_pass = bool(np.all(gain[commanded_axis] >= PROTOCOL["commanded_gain_min"]))
        raw_ok &= tracking_pass
        gain_ok &= progress_pass
        segments[name] = {"pass": tracking_pass and progress_pass, "samples": int(mask.sum()),
                          "raw_twist_rmse": segment_rmse.tolist(),
                          "gains": [float(gain[i]) if commanded_axis[i] else None for i in range(3)]}
    displacement = np.linalg.norm(position[:, :2] - position[0, :2], axis=-1)
    extra_ok = True
    extra = {}
    if task == "hold":
        extra["hold_max_displacement_m"] = float(displacement.max())
        extra_ok = extra["hold_max_displacement_m"] <= PROTOCOL["hold_displacement_limit_m"]
    if task == "stop":
        inside = np.all(np.abs(actual) <= PROTOCOL["stop_twist_bands"], axis=1)
        entry, confirmation = _first_dwell(time, inside, PROTOCOL["switch_s"], PROTOCOL["stop_dwell_s"])
        final_entry, final_confirmation = _final_dwell(time, inside, PROTOCOL["switch_s"], PROTOCOL["stop_dwell_s"])
        extra.update(stop_first_entry_after_switch_s=entry, stop_first_confirmed_after_switch_s=confirmation,
                     stop_entry_after_switch_s=final_entry, stop_confirmed_after_switch_s=final_confirmation,
                     stop_rebounded_after_first_dwell=bool(confirmation is not None and
                        np.any(~inside[time > PROTOCOL["switch_s"]+confirmation])))
        extra_ok = final_confirmation is not None and final_confirmation <= PROTOCOL["stop_deadline_after_switch_s"]
    passed = complete and not failed and raw_ok and gain_ok and extra_ok
    return {"pass": bool(passed), "status": "pass" if passed else "task_gate_failed",
            "terminated": bool(failed), "complete": bool(complete), "scored_samples": int(scored.sum()),
            "raw_twist_rmse": rmse.tolist(), "raw_twist_bias": error[scored].mean(0).tolist(),
            "filtered_twist_rmse": filtered_rmse.tolist(), "raw_tracking_pass": raw_ok,
            "filtered_tracking_diagnostic_pass": filtered_ok, "progress_pass": gain_ok,
            "segments": segments,
            "gains": [float(gains[i]) if commanded[i] else None for i in range(3)], **extra}
