"""Audit recorded per-hand loads before choosing stage-one disturbance ranges.

Reads original saved samples, never plotted/trimmed/smoothed force arrays. The
older multi-scenario export is explicitly labeled because its parent physics
trace no longer exists. Does not pool different simulators or coordinate frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(vectors):
    norms = np.linalg.norm(vectors, axis=-1)
    return {
        "count": len(vectors),
        "component_min": vectors.min(axis=0).tolist(),
        "component_max": vectors.max(axis=0).tolist(),
        "component_q01": np.quantile(vectors, .01, axis=0).tolist(),
        "component_q99": np.quantile(vectors, .99, axis=0).tolist(),
        "norm_p50": float(np.quantile(norms, .5)),
        "norm_p95": float(np.quantile(norms, .95)),
        "norm_p99": float(np.quantile(norms, .99)),
        "norm_max": float(norms.max()),
    }


def longest_run(mask, dt):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return float((ends-starts).max()*dt) if len(starts) else 0.


def summarize(force, moment, time, segment, dt, mask, commands=None):
    finite = np.isfinite(force).all(axis=(1, 2))
    if moment is not None:
        finite &= np.isfinite(moment).all(axis=(1, 2))
    valid = mask & finite
    if not valid.any():
        raise ValueError("No valid force samples")
    indices = np.flatnonzero(valid)
    out = {"samples": int(valid.sum()), "invalid_or_excluded": int((~valid).sum()), "hands": {}}
    for i, side in enumerate(("left", "right")):
        f = force[:, i]
        value = stats(f[valid])
        value["units"] = "N"
        norm = np.linalg.norm(f, axis=-1)
        peak = indices[np.argmax(norm[valid])]
        value["peak"] = {"index": int(peak), "time_s": float(time[peak]),
                         "segment": int(segment[peak]), "vector_N": f[peak].tolist()}
        if commands is not None:
            value["peak"]["command"] = commands[peak].tolist()
        value["longest_contiguous_above_threshold_s"] = {
            str(threshold): max(longest_run(valid & (segment == sid) & (norm > threshold), dt)
                                for sid in np.unique(segment))
            for threshold in (50, 75, 100)
        }
        # Average vector first, then take its norm; windows never cross a reset
        # boundary or an invalid sample. This is a separate sustained-load metric.
        sustained = {}
        for window in (.1, .5, 1.):
            count = max(1, round(window/dt))
            maxima = []
            for sid in np.unique(segment):
                ids = np.flatnonzero(segment == sid)
                if len(ids) < count:
                    continue
                keep = np.convolve(valid[ids].astype(int), np.ones(count, dtype=int), "valid") == count
                x = np.where(valid[ids, None], f[ids], 0.)
                cs = np.vstack((np.zeros((1, 3)), np.cumsum(x, axis=0)))
                means = (cs[count:]-cs[:-count])/count
                if keep.any():
                    maxima.append(float(np.linalg.norm(means[keep], axis=-1).max()))
            sustained[str(window)] = max(maxima) if maxima else None
        value["max_norm_of_window_mean_force_N"] = sustained
        if moment is not None:
            value["moment"] = {**stats(moment[valid, i]), "units": "N m"}
        out["hands"][side] = value
    pair = force[valid]
    net = pair.sum(axis=1)
    norms = np.linalg.norm(pair, axis=-1)
    opposite = (pair[:, 0]*pair[:, 1]).sum(axis=-1) < 0.
    out["pair"] = {"resultant": stats(net),
                   "opposing_direction_fraction": float(opposite.mean()),
                   "mean_sum_norm_minus_resultant_N": float((norms.sum(axis=1)-np.linalg.norm(net, axis=-1)).mean()),
                   "left_right_norm_correlation": float(np.corrcoef(norms.T)[0, 1])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"logs/research/hand_force_20261004")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    historical = ROOT/"assets/cjme_play_visuals/whole_hand_force_envelopes_multiscenario_20260923"
    path = historical/"force_points_and_hulls.npz"
    meta_path = historical/"envelope_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf8"))
    with np.load(path) as d:
        assert d["hand_order"].tolist() == ["left", "right"]
        np.testing.assert_allclose(d["force_on_hand_world_N"], -d["force_on_cart_world_N"], rtol=0, atol=0)
        force = d["force_on_hand_world_N"]
        segment, time = d["scenario_id"], d["time_s"]
        windows = {"all_samples": np.ones(len(time), dtype=bool),
                   "each_segment_after_1s": d["segment_time_s"] >= 1.}
        scenarios = {int(x["id"]): x for x in meta["scenario_metadata"]}
        for sid in np.unique(segment):
            windows[f"scenario_{sid}_{scenarios[int(sid)]['name']}"] = segment == sid
        records.append({"name": "historical_mujoco_multiscenario", "engine": "MuJoCo", "sampling_hz": 200,
                        "source": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                        "metadata_source": meta_path.relative_to(ROOT).as_posix(), "metadata_sha256": digest(meta_path),
                        "frame": "world", "direction": "cart on hand", "moment_reference": None,
                        "label": "retained unsmoothed per-hand export from MuJoCo weld wrench",
                        "parent_trace_exists": Path(meta["source"]).is_file(),
                        "parent_trace_sha256_recorded": meta["source_sha256"],
                        "scenario_metadata": meta["scenario_metadata"],
                        "windows": {k: summarize(force, None, time, segment, .005, mask, d["task_command"])
                                    for k, mask in windows.items()}})
    paths = [(ROOT/f"logs/Picture/force_use_range/raw/{name}.npz", f"route_{name}", "hand_wrench_cart")
             for name in ("baseline", "CWC")]
    paths += [(ROOT/f"logs/Picture/force_velocity_predict/isaac/raw/{axis}_CWC.npz", f"axis_{axis}_CWC", "true_hand_wrench")
              for axis in ("vx", "vy", "wz")]
    for path, name, key in paths:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        with np.load(path) as d:
            # Saved labels are HAND ON CART. Disturbance training requires the
            # reaction CART ON HAND, hence negate both force and moment.
            wrench = -d[key].reshape(-1, 2, 6).astype(np.float64)
            force, moment, time = wrench[:, :, :3], wrench[:, :, 3:], d["t"]
            segment = np.zeros(len(time), dtype=int)
            valid = d["hand_label_valid"].astype(bool) & ~d["terminated"].astype(bool)
            windows = {"all_valid_nonterminated": valid,
                       "after_1s": valid & (time >= 1.)}
            if "progress_pre" in d.files:
                windows["route_progress_lt19m"] = valid & (d["progress_pre"] < 19.)
            records.append({"name": name, "engine": "Isaac Sim / PhysX", "sampling_hz": 50,
                            "source": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                            "metadata_source": meta_path.relative_to(ROOT).as_posix(), "metadata_sha256": digest(meta_path),
                            "frame": "cart heading, world vertical", "direction": "cart on hand", "moment_reference": "each palm origin",
                            "label": "incoming joint wrench minus hand-subtree Newton-Euler load; model-based, not independently calibrated contact truth",
                            "payload_kg": meta.get("payload_kg"), "checkpoint_sha256": meta.get("checkpoint_sha256"),
                            "failure": meta.get("failure"), "completed": meta.get("completed", meta.get("success")),
                            "windows": {k: summarize(force, moment, time, segment, .02, mask, d["command"])
                                        for k, mask in windows.items()}})
    # Current MuJoCo axis rollouts use the actual equality-constraint wrench,
    # not the older gravity-compensated proxy or the Isaac dynamic label.
    for path in sorted((ROOT/"logs/Picture/motion_ability/mujoco/raw").glob("*.npz")):
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf8"))
        with np.load(path) as d:
            if "hand_force_moment_world" not in d.files:
                continue
            wrench = -d["hand_force_moment_world"].astype(np.float64)
            force, moment, time = wrench[:, :, :3], wrench[:, :, 3:], d["t"]
            segment = np.zeros(len(time), dtype=int)
            valid = ~d["terminated"].astype(bool)
            windows = {"all_valid_nonterminated": valid, "after_1s": valid & (time >= 1.)}
            records.append({"name": "mujoco_axis_"+path.stem, "engine": "MuJoCo", "sampling_hz": 50,
                            "source": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                            "metadata_source": meta_path.relative_to(ROOT).as_posix(), "metadata_sha256": digest(meta_path),
                            "frame": "world", "direction": "cart on hand",
                            "moment_reference": "saved cart-side grip reference; do not pool with palm moments",
                            "label": "MuJoCo equality/weld constraint wrench",
                            "payload_kg": meta.get("payload_kg"), "checkpoint_sha256": meta.get("checkpoint_sha256"),
                            "failure": meta.get("failure"), "completed": meta.get("completed"),
                            "windows": {k: summarize(force, moment, time, segment, .02, mask, d["command"])
                                        for k, mask in windows.items()}})
    output = {"date": "2026-10-04", "script_sha256": digest(Path(__file__)), "records": records,
              "limitations": ["Observed extrema are not theoretical load capacity or force limits for every pose.",
                              "Historical world axes cannot be relabeled robot/cart heading without orientation records.",
                              "50 Hz logs can miss 200 Hz impulse peaks; no pooling across measurement definitions.",
                              "No sample clipping or smoothing before raw maxima. Sustained metrics are separate.",
                              "Per-axis extrema generally occur at different times; their box corners are unobserved loads."]}
    (args.output/"ranges.json").write_text(json.dumps(output, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    fields = ["dataset", "window", "frame", "side", "samples", "p50_N", "p95_N", "p99_N", "max_N", "max_mean_0.1s_N", "max_mean_0.5s_N", "max_mean_1s_N", "min_x_N", "max_x_N", "min_y_N", "max_y_N", "min_z_N", "max_z_N", "peak_t_s", "peak_segment"]
    with (args.output/"per_hand_ranges.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in records:
            for window, summary in record["windows"].items():
                for side, values in summary["hands"].items():
                    row = {"dataset": record["name"], "window": window, "frame": record["frame"], "side": side,
                           "samples": values["count"], "peak_t_s": values["peak"]["time_s"], "peak_segment": values["peak"]["segment"]}
                    row.update({f"{q}_N": values[f"norm_{q}"] for q in ("p50", "p95", "p99", "max")})
                    row.update({f"max_mean_{w}s_N": values["max_norm_of_window_mean_force_N"][w] for w in ("0.1", "0.5", "1.0")})
                    row["max_mean_1s_N"] = row.pop("max_mean_1.0s_N")
                    for i, axis in enumerate("xyz"):
                        row[f"min_{axis}_N"] = values["component_min"][i]
                        row[f"max_{axis}_N"] = values["component_max"][i]
                    writer.writerow(row)
            first = next(iter(record["windows"].values()))
            print(record["name"], {side: {key: round(value[key], 2) for key in ("norm_p95", "norm_p99", "norm_max")}
                                    for side, value in first["hands"].items()})
    print("Saved", args.output)


if __name__ == "__main__":
    main()
