"""Export Table 1 three-axis RMSE/ripple from existing, validated raw motion trials.

No simulation or display-filtered samples are used. Nonzero command plateaus are
pooled by sample count, with their first 2 s excluded. Ripple is demeaned within
each plateau so command changes and between-plateau biases are not ripple.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from plot_motion_ability import AXES, load_trial, sha256, validate_pair


MOTIONS = {"vx": "Forward", "vy": "Lateral", "wz": "Turning"}
METRIC_KEYS = [f"{metric}_{axis}" for metric in ("rmse", "ripple") for axis in AXES]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, transient_s: float = 2.0):
    all_rows, all_stages, checks, sources = [], [], [], {}
    for backend, trial_root in (("Isaac Sim / PhysX", root), ("MuJoCo", root / "mujoco")):
        existing_file = trial_root / "derived/stage_metrics.csv"
        with existing_file.open(encoding="utf-8-sig", newline="") as fp:
            existing = {(r["axis"], r["method"], int(r["stage"])): r for r in csv.DictReader(fp)}
        sources[str(existing_file)] = sha256(existing_file)
        for motion_axis, motion in MOTIONS.items():
            pair = {method: load_trial(trial_root, motion_axis, method) for method in ("baseline", "CWC")}
            validate_pair(pair["CWC"], pair["baseline"])
            for key in ("t", "interval_start", "command", "reference"):
                np.testing.assert_array_equal(getattr(pair["CWC"], key), getattr(pair["baseline"], key))
            for key in ("checkpoint_sha256", "homie_sha256", "payload_kg", "seed"):
                assert pair["CWC"].metadata[key] == pair["baseline"].metadata[key], key
            for method, trial in pair.items():
                meta = trial.metadata
                assert meta["engine"] == backend and meta["payload_kg"] == 15.0
                assert trial.completed and not trial.terminated.any()
                assert not meta["velocity_is_estimate"] and not meta["velocity_lowpass_applied_to_saved_actual"]
                assert sha256(trial.source_paths[0]) == meta["npz_sha256"]
                for path in trial.source_paths:
                    sources[str(path)] = sha256(path)
                errors, demeaned, stage_values = [], [], []
                for idx, stage in enumerate(trial.stages):
                    if abs(stage["value"]) < 1e-12:
                        continue
                    mask = ((trial.stage_index == idx)
                            & (trial.interval_start >= stage["start_s"] + transient_s - 1e-7))
                    n = int(mask.sum())
                    assert n > 0
                    actual = trial.velocity[mask].astype(np.float64)
                    command = trial.command[mask].astype(np.float64)
                    error = actual - command
                    centered = actual - actual.mean(axis=0)
                    rmse = np.sqrt(np.mean(error ** 2, axis=0))
                    ripple = np.sqrt(np.mean(centered ** 2, axis=0))
                    bias = error.mean(axis=0)
                    np.testing.assert_allclose(rmse ** 2, ripple ** 2 + bias ** 2, rtol=1e-10, atol=1e-15)
                    old = existing[(motion_axis, method, idx)]
                    assert n == int(old["samples"])
                    assert float(old["discard_after_change_s"]) == transient_s
                    component = AXES[motion_axis][0]
                    rmse_delta = abs(float(rmse[component]) - float(old["raw_rmse"]))
                    ripple_delta = abs(float(ripple[component]) - float(old["raw_ripple_rms_demeaned"]))
                    # Existing Isaac statistics use float32; compute new tables in float64.
                    assert rmse_delta < 2e-7 and ripple_delta < 2e-7
                    checks.append({"backend": backend, "motion": motion, "method": method, "stage": idx,
                                   "existing_main_axis_rmse_delta": rmse_delta,
                                   "existing_main_axis_ripple_delta": ripple_delta})
                    for axis, (component, _) in AXES.items():
                        all_stages.append({
                            "backend": backend, "motion": motion, "method": method, "stage": idx,
                            "stage_command": stage["value"], "component": axis,
                            "unit": "rad/s" if axis == "wz" else "m/s", "samples": n,
                            "first_sample_end_s": float(trial.t[mask][0]),
                            "last_sample_end_s": float(trial.t[mask][-1]),
                            "command": float(command[0, component]),
                            "mean": float(actual[:, component].mean()), "bias": float(bias[component]),
                            "rmse": float(rmse[component]), "ripple": float(ripple[component]),
                        })
                    errors.append(error)
                    demeaned.append(centered)
                    stage_values.append(stage["value"])
                e, d = np.concatenate(errors), np.concatenate(demeaned)
                pooled_rmse = np.sqrt(np.mean(e ** 2, axis=0))
                pooled_ripple = np.sqrt(np.mean(d ** 2, axis=0))
                row = {"backend": backend, "motion": motion, "method": "Baseline" if method == "baseline" else "CWC"}
                for metric, values in (("rmse", pooled_rmse), ("ripple", pooled_ripple)):
                    row.update({f"{metric}_{axis}": float(values[i]) for i, axis in enumerate(AXES)})
                row.update(payload_kg=meta["payload_kg"], samples=len(e),
                           retained_duration_s=len(e) * meta["dt"],
                           command_plateaus=";".join(f"{v:g}" for v in stage_values),
                           discard_after_change_s=transient_s, seed=meta["seed"],
                           checkpoint=meta["checkpoint"], source=str(trial.source_paths[0]))
                all_rows.append(row)
    return all_rows, all_stages, checks, sources


def fmt(value):
    return f"{value:.6f}" if 0 < abs(value) < 0.00005 else f"{value:.4f}"


def table(rows):
    lines = ["| Motion | Method | RMSE Fwd. | RMSE Lat. | RMSE Yaw | Ripple Fwd. | Ripple Lat. | Ripple Yaw |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row["motion"], row["method"]] + [fmt(row[k]) for k in METRIC_KEYS]) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/Picture/motion_ability"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows, stages, checks, sources = summarize(root)
    out = root / "tables"
    out.mkdir(exist_ok=True)
    write_csv(out / "table1_motion_metrics.csv", rows)
    write_csv(out / "table1_stage_component_metrics.csv", stages)
    report = ["# Table 1: Cart-motion RMSE and ripple under matched conditions", "",
              "两套后端独立列出。单位：Fwd./Lat. 为 m/s，Yaw 为 rad/s。负载 15 kg，seed=1；每个运动/方法各一次标称轨迹。CWC 为 V6 model_19999；Baseline 为相同 HOMIE 的零高层残差控制。", "",
              "指标取 raw/*.npz 的原始实际小车 COM 三轴速度；vx/vy 在车体航向坐标系，wz 为世界 z 轴角速度。RMSE 相对于外部原始三轴 command（未测试轴的目标为零），不使用内部低通 reference 或绘图平滑曲线。", "",
              "每个非零平台剔除前 2 s，使用剩余 6 s。Forward 汇总 0.3/0.5/0.8 m/s 三档（900 点、18 s）；Lateral 汇总 0.15/0.3 m/s，Turning 汇总 0.15/0.3 rad/s（各 600 点、12 s）。不纳入 Isaac 的显示静止段或 MuJoCo 的初始零指令阶段。", "",
              "合并方法：RMSE_j = sqrt(sum_k sum_i (v_kij - command_kj)^2 / sum_k N_k)；Ripple_j = sqrt(sum_k sum_i (v_kij - mean_i(v_kij))^2 / sum_k N_k)。Ripple 在每个平台内去均值，再按样本数合并平方量；不是把不同档位拼接后求标准差，也不是各平台 RMS 的算术平均。这里的波动包含平台内剩余慢漂移，没有频段分离。", ""]
    for backend, tag in (("Isaac Sim / PhysX", "isaac"), ("MuJoCo", "mujoco")):
        selected = [r for r in rows if r["backend"] == backend]
        write_csv(out / f"table1_{tag}.csv", selected)
        section = [f"## {backend}", "", *table(selected), ""]
        report.extend(section)
        (out / f"table1_{tag}.md").write_text("\n".join(section), encoding="utf-8")
        latex = ["% Fwd./Lat.: m/s; Yaw: rad/s. Requires booktabs.",
                 r"\begin{tabular}{llrrrrrr}", r"\toprule",
                 r"Motion & Method & \multicolumn{3}{c}{RMSE $\downarrow$} & \multicolumn{3}{c}{Ripple $\downarrow$} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r" & & Fwd. & Lat. & Yaw & Fwd. & Lat. & Yaw \\", r"\midrule"]
        for row in selected:
            latex.append(" & ".join([row["motion"], row["method"]] + [fmt(row[k]) for k in METRIC_KEYS]) + r" \\")
        latex.extend([r"\bottomrule", r"\end{tabular}", ""])
        (out / f"table1_{tag}.tex").write_text("\n".join(latex), encoding="utf-8")
    report.extend([
        "MuJoCo 的 Baseline 在纯侧移和转向指令下几乎不运动，因此 Ripple 很小但主轴 RMSE 很大。低波动不等于完成了运动跟踪，需结合 RMSE 判断。表中小于 0.00005 的非零量保留六位小数；CSV/JSON 保留完整数值。", "",
        "验证：两种方法的三轴 command、reference 和时间数组逐项一致；模型、载荷、seed 匹配；原始 NPZ 哈希符合采集元数据；各档主轴 RMSE/Ripple 与已有 derived/stage_metrics.csv 一致（浮点精度容差 2e-7）。三轴均核对 RMSE² = Ripple² + Bias²。", "",
        "table1_stage_component_metrics.csv 保存每个档位、每个输出轴的细分统计。table1_manifest.json 保存输入 SHA-256 和验证差异。复现命令：`python scripts/summarize_motion_ability_table.py --root logs/Picture/motion_ability`。", "",
    ])
    (out / "table1_report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "table1_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "definition": "Nonzero plateaus, discard first 2 s; raw-command RMSE; within-plateau demeaned RMS ripple; pool squared values by sample count.",
                "sources_sha256": sources, "script_sha256": sha256(Path(__file__)), "checks": checks,
                "outputs_sha256": {p.name: sha256(p) for p in out.glob("table1_*") if p.name != "table1_manifest.json"}}
    (out / "table1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
