"""Independently verify command tracking RMSE and reference-path cross-track RMS."""
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pushcart_bench.paths import build_path, wrap_angle

BASE = ROOT / "logs/Picture/motion_ability"
OUT = BASE / "tables"
axes = ("vx", "vy", "wz")
motions = dict(vx="Forward", vy="Lateral", wz="Turning")
old_rows = json.loads((OUT / "table1_metrics.json").read_text(encoding="utf-8"))
motion_rows = []
examples = []
for backend, directory in (("Isaac Sim / PhysX", BASE), ("MuJoCo", BASE / "mujoco")):
    for motion_axis in axes:
        for method in ("baseline", "CWC"):
            source = directory / "raw" / f"{motion_axis}_{method}.npz"
            meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(source, allow_pickle=False) as raw:
                actual = np.asarray(raw["cart_velocity"], dtype=np.float64)
                command = np.asarray(raw["command"], dtype=np.float64)
                start = np.asarray(raw["interval_start"], dtype=np.float64)
            error = actual - command
            full_mask = np.any(np.abs(command) > 1e-12, axis=1)
            steady_mask = np.zeros(len(start), dtype=bool)
            centered = np.zeros_like(actual)
            stage_bias = np.zeros_like(actual)
            for stage in meta["stages"]:
                if abs(stage["value"]) < 1e-12:
                    continue
                mask = (start >= stage["start_s"] + 2.0 - 1e-7) & (start < stage["end_s"] - 1e-7)
                steady_mask |= mask
                centered[mask] = actual[mask] - actual[mask].mean(axis=0)
                stage_bias[mask] = error[mask].mean(axis=0)
                if backend == "MuJoCo" and motion_axis == "vy" and method == "baseline" and stage["value"] == 0.3:
                    examples.append({"backend": backend, "motion": "Lateral", "method": method,
                                     "command_m_s": 0.3, "actual_mean_m_s": float(actual[mask, 1].mean()),
                                     "command_rmse_m_s": float(np.sqrt(np.mean(error[mask, 1] ** 2))),
                                     "within_stage_std_m_s": float(np.std(actual[mask, 1]))})
            old = next(r for r in old_rows if r["backend"] == backend and r["motion"] == motions[motion_axis]
                       and r["method"].lower() == method.lower())
            for i, axis in enumerate(axes):
                rmse = float(np.sqrt(np.mean(error[steady_mask, i] ** 2)))
                ripple = float(np.sqrt(np.mean(centered[steady_mask, i] ** 2)))
                bias_rms = float(np.sqrt(np.mean(stage_bias[steady_mask, i] ** 2)))
                assert abs(rmse - old[f"rmse_{axis}"]) < 1e-12
                assert abs(ripple - old[f"ripple_{axis}"]) < 1e-12
                assert abs(rmse ** 2 - ripple ** 2 - bias_rms ** 2) < 1e-12
                motion_rows.append({"backend": backend, "motion": motions[motion_axis], "method": old["method"],
                                    "component": axis, "primary_axis": axis == motion_axis,
                                    "plateau_tail_rmse": rmse,
                                    "full_motion_rmse": float(np.sqrt(np.mean(error[full_mask, i] ** 2))),
                                    "plateau_tail_mae": float(np.mean(np.abs(error[steady_mask, i]))),
                                    "within_plateau_std": ripple, "plateau_bias_rms": bias_rms,
                                    "tail_samples": int(steady_mask.sum()), "full_samples": int(full_mask.sum())})

route_dir = ROOT / "logs/bench/table4_composite_a07_v05_payloads_20260930"
old_route = json.loads((route_dir / "table4_rmse_exclude_final_1m.json").read_text(encoding="utf-8"))["rows"]
route_rows = []
for tag in ("0", "12p5", "25"):
    for run in json.loads((route_dir / f"payload_{tag}kg/runs.json").read_text(encoding="utf-8"))["runs"]:
        tr = run["trace"]
        s = np.asarray(tr["s"])
        sel = s < 19.0
        ct = np.asarray(tr["cross_track"])[sel]
        local = build_path(run["path"], run["path_params"])
        # Recover the exact start transform using the unrounded saved goal.
        yaw0 = float(wrap_angle(run["goal"][2] - local.goal[2]))
        rot = np.array([[math.cos(yaw0), -math.sin(yaw0)], [math.sin(yaw0), math.cos(yaw0)]])
        origin = np.asarray(run["goal"][:2]) - rot @ np.asarray(local.goal[:2])
        path = local.transformed(origin, yaw0)
        # Stored geometry errors are computed pre-step; recorded positions are post-step.
        recorded_xy = np.column_stack([tr["x"], tr["y"]])
        pre_xy = np.vstack([origin, recorded_xy[:-1]])[sel]
        indices = np.rint(s[sel] / path.ds).astype(int)
        reference = path.xy[indices]
        tangent_yaw = path.yaw[indices]
        deviation = pre_xy - reference
        reconstructed_ct = np.cos(tangent_yaw) * deviation[:, 1] - np.sin(tangent_yaw) * deviation[:, 0]
        max_delta = float(np.max(np.abs(ct - reconstructed_ct)))
        assert max_delta < 0.00013  # Saved positions and cross-track errors each have four decimals.
        rmse = float(np.sqrt(np.mean(ct ** 2)))
        mean, std = float(ct.mean()), float(ct.std())
        assert abs(rmse ** 2 - mean ** 2 - std ** 2) < 1e-12
        method = "V6" if run["policy"]["kind"] == "v6" else "Pass-through"
        old = next(r for r in old_route if r["payload_kg"] == run["condition"]["payload_kg"] and r["method"] == method)
        assert abs(rmse - old["cross_track_rmse_m"]) < 1e-12
        route_rows.append({"payload_kg": run["condition"]["payload_kg"], "method": method,
                           "cross_track_rmse_m": rmse, "cross_track_mae_m": float(np.mean(np.abs(ct))),
                           "cross_track_abs_p95_m": float(np.quantile(np.abs(ct), .95)),
                           "cross_track_abs_max_m": float(np.max(np.abs(ct))),
                           "signed_mean_m": mean, "demeaned_std_m": std,
                           "reconstructed_rmse_m": float(np.sqrt(np.mean(reconstructed_ct ** 2))),
                           "reconstruction_max_abs_delta_m": max_delta})

for filename, rows in (("metric_audit_motion.csv", motion_rows), ("metric_audit_route.csv", route_rows)):
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(OUT / "metric_audit.json").write_text(json.dumps({"examples": examples, "motion": motion_rows, "route": route_rows}, indent=2), encoding="utf-8")
report = ["# RMSE 与 Ripple 核查", "",
          "结论：此前表1速度 RMSE 是实际速度与原始外部命令之差的均方根；表4横向 RMSE 是参考路线横向距离误差的均方根。两者都没有对误差去均值，保留了系统性偏差。原有数值不需要因本次质疑而替换。", "",
          "表1的 Ripple 是各平台实际速度相对于本平台均值的波动标准差，经样本数加权合并。它与 RMSE 不同。每个平台上 RMSE² = Bias² + Std²；合并后 RMSE² = RMS(Bias_k)² + pooled-Std²。此恒等式已对全部工况和三轴核对。", "",
          "表4的横向误差为 e_y = n(s)^T[p - p_ref(s)]，参考为 bench 单调前向搜索所选路径点的法向。统计是 sqrt(mean(e_y²))，没有减 mean(e_y)，正负误差不会抵消。它测量空间偏离；速度命令不是位置误差的参照。按保存位置与精确参考路径变换独立复算，所有采样差异均在保存四位小数的精度界内。", "",
          "例如恒定偏离路径 0.10 m：横向 RMSE = 0.10 m，横向 MAE = 0.10 m，去均值标准差 = 0。RMSE 是误差量级，不是平均绝对误差，也不是最大误差。", "",
          "## 需要明确的表头与窗口", "",
          "建议保留跟踪 RMSE，并把表1写为 Command-tracking RMSE (plateau-tail windows)，把 Ripple 写为 Within-plateau velocity std.（或在图注中明确 Ripple 的该定义）。每档剔除前2 s后，仍可能有残余漂移，不能保证已进入严格稳态。", "",
          "如果表1要衡量完整启动、加速和换档跟踪表现，应使用下方 Full-motion RMSE。它包含所有非零指令期间的真实物理采样，不包含 Isaac 显示初始化或 MuJoCo 零指令段。此修改属于统计窗口改变，不是纠正误差参照。三轴完整结果见 metric_audit_motion.csv。", "",
          "| 后端 | 运动 | 方法 | 原平台后6s主轴 RMSE | 含瞬态全程主轴 RMSE |",
          "|---|---|---|---:|---:|"]
for row in motion_rows:
    if row["primary_axis"]:
        report.append(f"| {row['backend']} | {row['motion']} | {row['method']} | {row['plateau_tail_rmse']:.4f} | {row['full_motion_rmse']:.4f} |")
report += ["", "主轴单位：Forward/Lateral 为 m/s，Turning 为 rad/s。", "",
           "## 路线误差大小（0 ≤ s < 19 m）", "",
           "建议表4保留 Cross-track RMSE (m)，若要说明平均偏离距离则补充 MAE，若关注严重偏离则补充绝对误差 P95 或最大值，不应替换成去均值波动量。以下都取此前相同采样窗口。", "",
           "| 负载 kg | 方法 | RMSE m | MAE m | 绝对误差 P95 m | 绝对误差最大值 m |",
           "|---:|---|---:|---:|---:|---:|"]
for row in route_rows:
    report.append(f"| {row['payload_kg']:g} | {row['method']} | {row['cross_track_rmse_m']:.4f} | {row['cross_track_mae_m']:.4f} | {row['cross_track_abs_p95_m']:.4f} | {row['cross_track_abs_max_m']:.4f} |")
report += ["", "## 原始数据示例", "", json.dumps(examples[0], ensure_ascii=False, indent=2), "",
           "MuJoCo Baseline 的侧向指令为 0.3 m/s 时，实际均值约 -0.000134 m/s，跟踪 RMSE 约 0.300134 m/s，但波动标准差约 0.000229 m/s。实际不运动不会得到小的命令跟踪 RMSE，却可能得到小的 Ripple。", ""]
(OUT / "metric_audit.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
