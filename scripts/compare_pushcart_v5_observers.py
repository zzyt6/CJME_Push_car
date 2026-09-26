"""Offline candidate-vs-recorded-V5 observer comparison; never changes control.

Usage: python scripts/compare_pushcart_v5_observers.py --input <trace-directory>
       --output <new-output-directory>
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v5.motion_observer import compare_motion_estimators


def rotate_xy(vector, yaw):
    value = np.asarray(vector, dtype=np.float64).copy()
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = value[..., 0].copy(), value[..., 1].copy()
    value[..., 0], value[..., 1] = c * x - s * y, s * x + c * y
    return value


def kernel_matched_truth(data, valid, dt, window_s):
    """Privileged evaluation only: regress exact cart COM/yaw at pose captures.

    Uses the candidate's acceptance/window convention, then rotates the fitted
    world velocity into current level-base axes. This is a smoothing/latency
    diagnostic, not an independent instantaneous accuracy or control metric.
    """
    positions = np.asarray(data["cart_com"], dtype=np.float64)
    yaw = np.asarray(data["cart_yaw"], dtype=np.float64)
    base_yaw = np.asarray(data["base_yaw"], dtype=np.float64)
    delay = np.asarray(data["pose_delay_steps"], dtype=int)
    steps, n = valid.shape
    result = np.zeros((steps, n, 3))
    ready = np.zeros((steps, n), dtype=bool)
    window_steps = int(np.ceil(window_s / dt)) + 1
    for env in range(n):
        history = deque(maxlen=window_steps)
        episode_start = 0
        for index in range(steps):
            if not valid[index, env]:
                history.clear()
                episode_start = index + 1
                continue
            capture = index - delay[index, env]
            if capture < episode_start:
                continue
            if not history or capture > history[-1]:
                history.append(capture)
            latest = history[-1]
            ids = np.array([i for i in history if (latest - i) * dt <= window_s + dt * 1e-4])
            if len(ids) < 3 or (ids[-1] - ids[0]) * dt < window_s - dt * .5:
                continue
            t = ids.astype(np.float64) * dt
            centered = t - t.mean()
            values = np.c_[positions[ids, env, :2], np.unwrap(yaw[ids, env])]
            slope = ((values - values[-1]) * centered[:, None]).sum(0) / np.dot(centered, centered)
            result[index, env] = rotate_xy(slope, -base_yaw[index, env])
            ready[index, env] = True
    return result, ready


def _errors(estimate, truth, mask):
    error = (estimate - truth)[mask]
    if not len(error):
        return {"sample_count": 0, "rmse": None, "bias": None, "xy_vector_rmse": None}
    return {"sample_count": len(error), "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "bias": error.mean(0).tolist(), "xy_vector_rmse": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=-1))))}


def _comparison(candidate, recorded, truth, mask):
    a, b = _errors(candidate, truth, mask), _errors(recorded, truth, mask)
    improvement = None
    if a["sample_count"]:
        improvement = (np.asarray(b["rmse"]) - np.asarray(a["rmse"])).tolist()
    return {"candidate": a, "recorded_v5": b, "rmse_improvement_positive_is_better": improvement}


def compare_recorded_trace(data, metadata, *, window_s=None):
    """Pure replay entrypoint used by CLI/tests; returns report and saved arrays."""
    dt = float(metadata["dt"])
    window_s = float(window_s if window_s is not None else metadata.get("task_config", {}).get("reaction", {}).get("motion_window_s", .4))
    pose = np.asarray(data["pose_base"])
    if pose.ndim != 3 or pose.shape[-1] != 4:
        raise ValueError("pose_base must be [T,N,4]")
    steps, n, _ = pose.shape
    required = {"base_velocity_est": 3, "imu_gyro_b": 3, "gravity_b": 3,
                "actual": 3, "estimated_cart_twist": 3}
    for name, width in required.items():
        if np.shape(data[name]) != (steps, n, width):
            raise ValueError(f"{name} must share [T,N,{width}]")
    for name in ("valid", "pose_delay_steps", "cart_yaw", "base_yaw", "failed", "observer_ready", "time_s"):
        if np.shape(data[name]) != (steps, n):
            raise ValueError(f"{name} must share [T,N]")
    times = np.asarray(data["time_s"], dtype=np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times, axis=0), dt, atol=1e-6, rtol=0):
        raise ValueError("trace must contain regular control timestamps matching metadata dt")
    failed = np.maximum.accumulate(np.asarray(data["failed"], dtype=bool), axis=0)
    valid = np.asarray(data["valid"], dtype=bool) & ~failed
    for name in ("pose_base", "base_velocity_est", "imu_gyro_b", "gravity_b"):
        valid &= np.isfinite(data[name]).all(-1)
    delay = np.asarray(data["pose_delay_steps"])
    if not np.isfinite(delay).all() or np.any(delay < 0) or not np.equal(delay, np.round(delay)).all():
        raise ValueError("known pose latency must be nonnegative integer samples")
    geometry = metadata["observer_geometry"]
    replay = compare_motion_estimators({
        "handle_pose_b": pose, "base_velocity_b": data["base_velocity_est"],
        "gyro_b": data["imu_gyro_b"], "gravity_b": data["gravity_b"],
        "valid": valid, "pose_delay_steps": delay}, dt=dt, window_s=window_s,
        max_pose_delay_steps=max(4, int(delay.max())), handle_offset_cart=geometry)
    candidate = replay["candidate_twist_base"].numpy()
    # The recorded estimate uses the observed (possibly delayed/noisy) heading.
    # Rotating it by true cart-minus-base yaw would give it privileged alignment.
    recorded = rotate_xy(data["estimated_cart_twist"], pose[..., 3])
    truth = rotate_xy(data["actual"], np.asarray(data["cart_yaw"]) - np.asarray(data["base_yaw"]))
    recorded_ready = np.asarray(data["observer_ready"]) > .5
    common = valid & recorded_ready & replay["candidate_ready"].numpy()
    common &= np.isfinite(candidate).all(-1) & np.isfinite(recorded).all(-1) & np.isfinite(truth).all(-1)
    # Current baseline replay is a sanity check only. Primary comparator is the
    # EXACT estimate recorded in the live V5 controller, including float32/state.
    baseline_replay = replay["baseline_twist_base"].numpy()
    replay_mask = valid & recorded_ready & replay["baseline_ready"].numpy()
    baseline_difference = np.abs(baseline_replay - recorded)[replay_mask]
    report = {
        "comparison_revision": 1, "candidate_promoted_to_default": False,
        "dt_s": dt, "candidate_window_s": window_s, "num_cases": n,
        "input_samples": steps * n, "common_samples": int(common.sum()),
        "coordinate_contract": "All scored estimates/truth use current upright base-heading axes. Recorded V5 estimate is rotated with its observed relative yaw; truth with true cart-minus-base yaw. Candidate receives no true yaw or cart motion.",
        "mask_contract": "Identical mask per comparison: candidate ready AND recorded sensor ready AND all observed inputs valid, excluding first termination and every subsequent sample of that case.",
        "primary_target": "current raw simulator cart COM twist; includes estimator smoothing and latency",
        "overall_instantaneous": _comparison(candidate, recorded, truth, common),
        "replayed_baseline_vs_recorded_max_abs": baseline_difference.max(0).tolist() if len(baseline_difference) else None,
        "candidate_mean_fit_center_age_s": float(replay["candidate_fit_center_age_s"].numpy()[common].mean()) if common.any() else None,
        "candidate_mean_pose_age_s": float(replay["candidate_sample_age_s"].numpy()[common].mean()) if common.any() else None,
        "source_policy": metadata.get("policy"), "source_noisy": metadata.get("noisy"),
        "source_level": metadata.get("level"), "cases": [],
        "limitations": [
            "Offline replay only; estimates cannot influence the recorded robot trajectory, so this is not a control-performance comparison.",
            "Candidate needs known pose latency; unknown sensor timing and HIM/IMU bias are not identified.",
            "Both estimators use nominal handle-to-COM geometry; payload COM shift/cart tilt remains model error.",
            "Candidate regression averages motion; raw-current RMSE includes latency. Kernel-matched truth is supplementary and follows the candidate's own kernel.",
            "Recorded V5 uses a yaw-only body/heading approximation; candidate gravity-levels body vectors. This is a combined frame/timing/geometry candidate, not a single-change ablation.",
            "Results on one captured policy/noise setting do not establish robustness or justify default promotion.",
        ],
    }
    arrays = {"candidate_twist_base": candidate, "recorded_v5_twist_base": recorded,
              "replayed_baseline_twist_base": baseline_replay, "truth_current_twist_base": truth,
              "common_mask": common, "valid_before_first_failure": valid,
              "candidate_ready": replay["candidate_ready"].numpy(), "recorded_ready": recorded_ready,
              "candidate_fit_center_age_s": replay["candidate_fit_center_age_s"].numpy(),
              "candidate_sample_age_s": replay["candidate_sample_age_s"].numpy(), "time_s": times}
    if "cart_com" in data:
        matched, matched_ready = kernel_matched_truth(data, valid, dt, window_s)
        matched_mask = common & matched_ready & np.isfinite(matched).all(-1)
        arrays.update(truth_kernel_matched_twist_base=matched, kernel_matched_mask=matched_mask)
        report["kernel_matched_target_contract"] = "Regression of true world COM XY and unwrapped cart yaw at the candidate's accepted pose-capture samples, then rotate fitted world velocity to current base heading. Same target for both estimators; diagnostic only."
        report["overall_kernel_matched"] = _comparison(candidate, recorded, matched, matched_mask)
    cases = metadata.get("cases", [])
    for index in range(n):
        info = cases[index] if index < len(cases) else {}
        result = {"index": index, "task": info.get("task", str(index)), "payload_kg": info.get("payload_kg"),
                  "first_failure_time_s": info.get("failure_time_s"),
                  "instantaneous": _comparison(candidate[:, index], recorded[:, index], truth[:, index], common[:, index])}
        if "cart_com" in data:
            result["kernel_matched"] = _comparison(candidate[:, index], recorded[:, index], matched[:, index], matched_mask[:, index])
        report["cases"].append(result)
    improved_xy = sum(case["instantaneous"]["candidate"]["xy_vector_rmse"] < case["instantaneous"]["recorded_v5"]["xy_vector_rmse"]
                      for case in report["cases"] if case["instantaneous"]["candidate"]["sample_count"])
    comparable = sum(bool(case["instantaneous"]["candidate"]["sample_count"]) for case in report["cases"])
    report["case_count_xy_improved"] = improved_xy
    report["case_count_xy_not_improved"] = comparable - improved_xy
    return report, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory containing traces.npz and report.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing output is not overwritten")
    parser.add_argument("--window-s", type=float, help="candidate fit window; defaults to recorded V5 setting")
    args = parser.parse_args(argv)
    trace_path, metadata_path = args.input / "traces.npz", args.input / "report.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(trace_path) as data:
        report, arrays = compare_recorded_trace(data, metadata, window_s=args.window_s)
    report["sources"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in (("trace", trace_path), ("metadata", metadata_path))}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "comparison_arrays.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "common_samples": report["common_samples"],
                      "overall_instantaneous": report["overall_instantaneous"],
                      "overall_kernel_matched": report.get("overall_kernel_matched"),
                      "case_count_xy_improved": report["case_count_xy_improved"],
                      "case_count_xy_not_improved": report["case_count_xy_not_improved"],
                      "candidate_promoted_to_default": False}, indent=2))
    return report


if __name__ == "__main__":
    main()
