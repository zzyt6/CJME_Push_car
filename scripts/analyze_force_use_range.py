"""Analyze measured Isaac force envelopes and actuator-only capability bounds."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

ROOT = Path(__file__).resolve().parents[1]
PLANES = ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(root):
    result = {}
    for method in ("baseline", "CWC"):
        source = root / "raw" / f"{method}.npz"
        meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        assert sha(source) == meta["npz_sha256"]
        with np.load(source) as z:
            data = {k: z[k].copy() for k in z.files}
        force = data["hand_wrench_cart"].reshape(-1, 2, 6)[:, :, :3].astype(float)
        # Geometric progress is sampled before a step; force is sampled after it.
        # Use the next pre-step progress for the same post-physics position.
        progress = np.r_[data["progress_pre"][1:], meta["final_progress_m"]]
        assert max(progress) >= 19., "Both trials must cover the declared first-19-m window"
        valid = data["hand_label_valid"] & ~data["terminated"] & ~data["env_timeout"]
        selected = valid & (progress < 19.)
        result[method] = {"meta": meta, "data": data, "force_hands": force, "force": force.sum(axis=1),
                          "progress": progress, "valid": valid, "selected": selected}
    a, b = [result[m]["meta"] for m in ("baseline", "CWC")]
    for key in ("payload_kg", "seed", "speed_m_s", "checkpoint_sha256", "homie_sha256", "path_params", "follower"):
        assert a[key] == b[key], key
    return result


def kde_pair(points, plane, grid_n=220):
    """Same physical grid and isotropic bandwidth for both methods; no sample deletion."""
    i, j, tag = plane
    pooled = np.concatenate([points[m][:, [i, j]] for m in ("baseline", "CWC")])
    # Scott-like common bandwidth, chosen from both methods together in physical N.
    bandwidth = max(.8, float(np.mean(np.std(pooled, axis=0))) * len(pooled) ** (-1/6))
    lo, hi = pooled.min(axis=0) - 4*bandwidth, pooled.max(axis=0) + 4*bandwidth
    edges = [np.linspace(lo[k], hi[k], grid_n+1) for k in range(2)]
    spacing = (hi-lo)/grid_n
    area = float(np.prod(spacing))
    output = {"x": (edges[0][:-1]+edges[0][1:])/2, "y": (edges[1][:-1]+edges[1][1:])/2,
              "bandwidth_N": bandwidth, "cell_area_N2": area}
    for method in ("baseline", "CWC"):
        f = points[method][:, [i, j]]
        hist, _, _ = np.histogram2d(f[:, 0], f[:, 1], bins=edges)
        density = gaussian_filter(hist.astype(float), bandwidth/spacing, mode="constant")
        density /= density.sum()*area
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered)*area
        thresholds = [float(ordered[min(np.searchsorted(cumulative, q), len(ordered)-1)]) for q in (.5, .8, .95)]
        output[method+"_density"] = density
        output[method+"_thresholds"] = thresholds
        output[method+"_hdr95_area_N2"] = float((density >= thresholds[-1]).sum()*area)
    return output


def capacity_bound(root, directions=72):
    """Support-function projection of |gravity + J^T w| <= effort limits.

    All 12 palm wrench coordinates are free. The 15 upper-joint inequalities
    include shared waist effort. No ground balance/contact/friction constraints
    are imposed, so this is an optimistic actuator-only model, NOT verified force
    attainability. Projection halfplanes and the circle are conservative outers.
    """
    theta = np.arange(directions)*2*np.pi/directions
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    objectives = []
    for i, j, _ in PLANES:
        for direction in normals:
            objective = np.zeros(12)
            objective[[i, i+6]] = direction[0]
            objective[[j, j+6]] = direction[1]
            objectives.append(objective)
    objectives = np.asarray(objectives)
    supports = np.full(len(objectives), -np.inf)
    witness_points = np.zeros((len(objectives), 3))
    witnesses = np.full(len(objectives), -1, dtype=int)
    snapshot_records = []
    finite_snapshots = 0
    for method in ("baseline", "CWC"):
        with np.load(root / "raw" / f"{method}_capacity.npz") as z:
            data = dict(z)
        for index in range(len(data["t"])):
            if data["s"][index] >= 19.:
                continue
            J = data["jacobian_world"][index].astype(float)
            g, limits = data["gravity_torque"][index].astype(float), data["torque_limits"][index].astype(float)
            yaw = float(data["cart_yaw"][index])
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            R6 = np.zeros((6, 6)); R6[:3, :3] = R; R6[3:, 3:] = R
            A = np.concatenate([J[k].T @ R6 for k in (0, 1)], axis=1)
            matrix, rhs = np.r_[A, -A], np.r_[limits-g, limits+g]
            record = {"method": method, "sample_index": index, "t": float(data["t"][index]),
                      "progress_m": float(data["s"][index]), "matrix_rank": int(np.linalg.matrix_rank(A))}
            for k, objective in enumerate(objectives):
                sol = linprog(-objective, A_ub=matrix, b_ub=rhs, bounds=[(None, None)]*12, method="highs")
                if not sol.success:
                    raise RuntimeError(f"Cannot certify finite actuator bound: {record}, status={sol.status}, {sol.message}")
                support = float(-sol.fun)
                assert np.max(matrix @ sol.x - rhs) < 1e-5
                if support > supports[k]:
                    supports[k] = support
                    witness_points[k] = sol.x[:3] + sol.x[6:9]
                    witnesses[k] = len(snapshot_records)
            snapshot_records.append(record)
            finite_snapshots += 1
            if finite_snapshots % 20 == 0:
                print(f"[force-analysis] capacity {finite_snapshots} postures", flush=True)
    output = {"directions": directions, "snapshots": snapshot_records, "planes": {}}
    for k, (i, j, tag) in enumerate(PLANES):
        h = supports[k*directions:(k+1)*directions]
        assert np.all(h > 0), "Zero must be inside union projection for this representation"
        vertices = HalfspaceIntersection(np.column_stack([normals, -h]), np.zeros(2)).intersections
        vertices = vertices[ConvexHull(vertices).vertices]
        output["planes"][tag] = {"normals": normals.tolist(), "support_N": h.tolist(),
                                  "outer_polygon_N": vertices.tolist(),
                                  "outer_circle_radius_N": float(np.linalg.norm(vertices, axis=1).max()),
                                  "support_witness_force_N": witness_points[k*directions:(k+1)*directions].tolist()}
    output["definition"] = "Union across recorded postures s<19m of actuator-only quasistatic wrench sets; projected via 5-degree support halfplanes, circumscribed by a zero-centered circle. Not a certified whole-body attainable force set."
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "logs/Picture/force_use_range")
    p.add_argument("--skip-capacity", action="store_true")
    args = p.parse_args()
    root = args.root.resolve(); derived = root / "derived"; derived.mkdir(exist_ok=True)
    trials = load(root)
    stats = {"window": "post-physics route progress 0<=s<19m; all valid 50Hz force samples; no transient trimming", "methods": {}, "planes": {}}
    points = {}
    for method, trial in trials.items():
        mask, d = trial["selected"], trial["data"]
        force = trial["force"][mask]; points[method] = force
        hands = trial["force_hands"][mask]
        trend = uniform_filter1d(force, size=25, axis=0, mode="nearest")
        fluct = force-trend
        ct_mask = d["progress_pre"] < 19.
        all_hands_norm = np.linalg.norm(hands, axis=-1).sum(axis=1)
        net_norm = np.linalg.norm(force, axis=1)
        opposing = np.maximum(0, all_hands_norm-net_norm)
        prediction = d["pred_hand_wrench"][mask].reshape(-1, 2, 6)[:, :, :3].sum(axis=1)
        metrics = {"samples": int(mask.sum()), "duration_s": float(mask.sum()*trial["meta"]["dt"]),
                   "route_success": trial["meta"]["success"], "route_timeout": trial["meta"]["route_timed_out"],
                   "failure": trial["meta"]["failure"],
                   "cross_track_rmse_m": float(np.sqrt(np.mean(d["cross_track_pre"][ct_mask]**2))),
                   "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(d["heading_error_pre"][ct_mask]**2)))),
                   "time_to_19m_s": float(d["interval_start"][np.flatnonzero(d["progress_pre"] >= 19.)[0]]),
                   "force_mean_N": force.mean(axis=0).tolist(), "force_std_N": force.std(axis=0).tolist(),
                   "force_norm_mean_N": float(net_norm.mean()), "force_norm_p95_N": float(np.quantile(net_norm, .95)),
                   "force_norm_max_N": float(net_norm.max()),
                   "local_force_fluctuation_rms_N": float(np.sqrt(np.mean(np.sum(fluct**2, axis=1)))),
                   "opposing_force_mean_N": float(opposing.mean()),
                   "opposing_force_fraction": float(opposing.sum()/all_hands_norm.sum()),
                   "estimator_net_force_rmse_N": np.sqrt(np.mean((prediction-force)**2, axis=0)).tolist(),
                   "force_3d_convex_hull_volume_N3": float(ConvexHull(force).volume),
                   "all_valid_force_max_N": float(np.linalg.norm(trial["force"][trial["valid"]], axis=1).max())}
        stats["methods"][method] = metrics
        csv_path = derived / f"{method}_resultant_force.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["t_s", "route_progress_m", "in_main_window", "Fx_N", "Fy_N", "Fz_N", "left_Fx_N", "left_Fy_N", "left_Fz_N", "right_Fx_N", "right_Fy_N", "right_Fz_N"])
            for k, t in enumerate(d["t"]):
                writer.writerow([t, trial["progress"][k], int(mask[k]), *trial["force"][k], *trial["force_hands"][k].reshape(-1)])
    kdes = {}
    for plane in PLANES:
        data = kde_pair(points, plane); tag = plane[-1]
        kdes.update({tag+"_"+k: v for k, v in data.items()})
        stats["planes"][tag] = {"bandwidth_N": data["bandwidth_N"]}
        for method in ("baseline", "CWC"):
            i, j, _ = plane
            stats["planes"][tag][method] = {"hull_area_N2": float(ConvexHull(points[method][:, [i, j]]).volume),
                                            "hdr95_area_N2": data[method+"_hdr95_area_N2"]}
    np.savez_compressed(derived / "force_envelopes.npz", **kdes, **{m+"_force": f for m, f in points.items()})
    (derived / "metrics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    if not args.skip_capacity:
        result = capacity_bound(root)
        (derived / "capability_bound.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("[force-analysis] capability radii", {k: v["outer_circle_radius_N"] for k, v in result["planes"].items()}, flush=True)


if __name__ == "__main__":
    main()
