"""runs.json -> report.md (tables) + PNG figures (trajectory overlays, error traces, step responses, degradation)."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .metrics import aggregate, fmt_ms

PATH_COLUMNS = [("success", "成功率", None), ("time_to_goal", "到点时间 s", 1), ("time_enter", "首次进圈 s", 1),
                ("final_pos_err", "到点误差 m", 3), ("settled_pos_err", "停稳后误差 m", 3), ("min_goal_dist", "最近距离 m", 3),
                ("cross_track_rms", "横向 RMS m", 3), ("cross_track_max", "横向 max m", 3), ("heading_rms", "航向 RMS rad", 3),
                ("cmd_vx_rmse", "vx 跟踪 RMSE", 3), ("cmd_wz_rmse", "wz 跟踪 RMSE", 3), ("grasp_p95", "抓握 p95 N", 0),
                ("lateral_p95", "侧向力 p95 N", 0), ("overload_frac", "过载占比", 3),
                ("cmd_clip_frac", "命令残差饱和占比", 2), ("arm_clip_frac", "臂动作饱和占比", 2), ("arm_target_std", "臂修正波动 rad", 3)]
STEP_COLUMNS = [("rise_time", "上升 s", 2), ("settling_time", "整定 s", 2), ("overshoot", "超调", 2), ("ss_error", "稳态误差", 3),
                ("post_rmse", "稳态 RMSE", 3), ("ripple_rms", "纹波 RMS", 3), ("ss_std", "慢振荡 std (0.4 s 滤波)", 3),
                ("stopping_distance", "停止距离 m", 2), ("drift_distance", "漂移 m", 2), ("grasp_p95", "抓握 p95 N", 0)]


def load_runs(out_dir: Path) -> tuple[dict, list]:
    data = json.loads((Path(out_dir) / "runs.json").read_text(encoding="utf-8"))
    return data.get("meta", {}), data.get("runs", [])


def _group(runs, key):
    groups = defaultdict(list)
    for r in runs:
        groups[key(r)].append(r)
    return groups


ALL_RUN_METRICS = {"success", "min_goal_dist", "time_enter", "grasp_p95", "lateral_p95", "overload_frac",
                   "cmd_clip_frac", "arm_clip_frac", "arm_target_std",
                   "rise_time", "settling_time", "overshoot", "ss_error", "post_rmse", "ripple_rms", "ss_std", "stopping_distance", "drift_distance"}


def _cell(runs, metric, digits):
    """Success rate over all runs; goal / tracking metrics over the successful runs only (a run that drifted away for
    40 s would otherwise dominate the mean); interaction / step metrics over all runs."""
    if metric == "success":
        vals = [1.0 if r["metrics"].get("success") else 0.0 for r in runs]
        return f"{100 * np.mean(vals):.0f} % ({int(sum(vals))}/{len(vals)})" if vals else "-"
    if metric not in ALL_RUN_METRICS and runs and runs[0]["kind"] != "step":
        runs = [r for r in runs if r["metrics"].get("success")]
    agg = aggregate([r["metrics"].get(metric) for r in runs])
    return fmt_ms(agg, digits=digits if digits is not None else 2)


def _table(runs, row_key, row_title, columns):
    policies = sorted({r["policy"]["name"] for r in runs})
    lines = ["| " + row_title + " | 策略 | " + " | ".join(c[1] for c in columns) + " |",
             "|" + "---|" * (2 + len(columns))]
    for row, rows in _group(runs, row_key).items():
        by_policy = _group(rows, lambda r: r["policy"]["name"])
        for pol in policies:
            if pol not in by_policy:
                continue
            cells = [_cell(by_policy[pol], m, d) for m, _, d in columns]
            lines.append(f"| {row} | {pol} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _policy_table(runs):
    seen, lines = {}, ["| 策略 | 类型 | 检查点 | 迭代 | HOMIE |", "|---|---|---|---|---|"]
    for r in runs:
        p = r["policy"]
        if p["name"] not in seen:
            seen[p["name"]] = p
            lines.append(f"| {p['name']} | {p.get('kind')} | {Path(p['hl_checkpoint']).name if p.get('hl_checkpoint') else '-'} | "
                         f"{p.get('hl_iteration') or '-'} | {Path(p.get('homie_checkpoint', '-')).name} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _short(name: str) -> str:
    """two_stage:<run dir>/model_N -> two_stage/model_N for legends."""
    return name.split("/")[-1].join(["two_stage/", ""]) if name.startswith("two_stage:") and "/" in name else name


def _fig_paths(runs, out_dir: Path) -> list[str]:
    plt = _plt()
    files = []
    for pname, rows in _group(runs, lambda r: r["path"] + (f" {r.get('gains')}" if r.get("gains") else "")).items():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
        drawn = set()
        for r in rows:
            pol = _short(r["policy"]["name"])
            if r["condition"].get("seed", 0) != min(x["condition"].get("seed", 0) for x in rows if x["policy"]["name"] == r["policy"]["name"]):
                continue
            xy = np.asarray(r["path_xy"])
            if "ref" not in drawn:
                ax.plot(xy[:, 0], xy[:, 1], "k--", lw=1.2, label="参考轨迹")
                gx, gy = r["goal"][:2]
                ax.plot(gx, gy, "r*", ms=12, label="目标")
                drawn.add("ref")
            tr = r["trace"]
            ax.plot(tr["x"], tr["y"], lw=1.5, label=f"{pol}" + ("" if r["metrics"].get("success") else f" ({r['metrics'].get('failure') or 'timeout'})"))
            if tr.get("cross_track"):
                ax2.plot(tr["t"], tr["cross_track"], lw=1.2, label=pol)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(f"{pname}: 轨迹叠加")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        ax2.axhline(0, color="k", lw=0.8); ax2.grid(alpha=0.3); ax2.set_xlabel("t [s]"); ax2.set_ylabel("横向误差 [m]"); ax2.set_title("横向误差")
        ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        name = f"path_{pname.replace(' ', '_').replace('{', '').replace('}', '').replace(':', '').replace(',', '_').replace(chr(39), '')}.png"
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=130); plt.close(fig)
        files.append(name)
    return files


def _fig_steps(runs, out_dir: Path) -> list[str]:
    plt = _plt()
    files = []
    for sname, rows in _group(runs, lambda r: r["step"]["name"]).items():
        payloads = sorted({r["condition"]["payload_kg"] for r in rows})
        fig, axes = plt.subplots(1, len(payloads), figsize=(5 * len(payloads), 4), squeeze=False)
        for ax, payload in zip(axes[0], payloads):
            for r in rows:
                if r["condition"]["payload_kg"] != payload:
                    continue
                tr, sig = r["trace"], r["metrics"]["signal"]
                ax.plot(tr["t"], tr[sig], lw=1.4, label=_short(r["policy"]["name"]))
                cmd = np.asarray(tr["cmd"])
                ax.plot(tr["t"], cmd[:, {"vx": 0, "vy": 1, "wz": 2}[sig]], "k--", lw=0.9)
            ax.set_title(f"{sname} · {payload:g} kg"); ax.grid(alpha=0.3); ax.set_xlabel("t [s]"); ax.set_ylabel(r["metrics"]["signal"])
            ax.legend(fontsize=8)
        name = f"step_{sname}.png"
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=130); plt.close(fig)
        files.append(name)
    return files


def _fig_robust(runs, out_dir: Path) -> list[str]:
    plt = _plt()
    files = []
    for factor, rows in _group(runs, lambda r: r.get("factor")).items():
        if factor is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for pol, prow in _group(rows, lambda r: _short(r["policy"]["name"])).items():
            xs = sorted({r["factor_value"] for r in prow})
            def series(metric, success_only=True):
                ys, es = [], []
                for x in xs:
                    sel = [r for r in prow if r["factor_value"] == x and (r["metrics"].get("success") or not success_only)]
                    agg = aggregate([r["metrics"].get(metric) for r in sel])
                    ys.append(agg["mean"] if agg["mean"] is not None else np.nan); es.append(agg["std"] or 0.0)
                return np.array(ys, float), np.array(es, float)
            for ax, (metric, title) in zip(axes, (("time_to_goal", "到点时间 [s]"), ("cross_track_rms", "横向 RMS [m]"))):
                y, e = series(metric)
                ax.errorbar(xs, y, yerr=e, marker="o", capsize=3, label=pol); ax.set_title(title); ax.set_xlabel(factor); ax.grid(alpha=0.3)
            fail = [100 * np.mean([0.0 if r["metrics"].get("success") else 1.0 for r in prow if r["factor_value"] == x]) for x in xs]
            axes[2].plot(xs, fail, marker="s", label=pol); axes[2].set_title("失败率 [%]"); axes[2].set_xlabel(factor); axes[2].grid(alpha=0.3)
        for ax in axes:
            ax.legend(fontsize=8)
        name = f"robust_{factor}.png"
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=130); plt.close(fig)
        files.append(name)
    return files


# ---------------------------------------------------------------------------- report

def write_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    meta, runs = load_runs(out_dir)
    kind = meta.get("kind") or (runs[0]["kind"] if runs else "path")
    lines = [f"# 推车控制对比实验报告 · {kind}", "", f"生成时间 {meta.get('created', '')} · 运行数 {len(runs)} · 目录 `{out_dir}`", "",
             "## 公平性约定", "", meta.get("fairness", ""), ""]
    if meta.get("follower"):
        f = meta["follower"]
        lines += ["巡线 PID（所有策略相同）：" + ", ".join(f"{k}={v}" for k, v in f.items() if k not in ("cmd_limits", "rate_limits")), ""]
    lines += ["## 被测策略", "", _policy_table(runs), ""]
    figures = []
    try:
        if kind == "path":
            if any(r.get("gains") for r in runs):
                lines += ["## 增益扫描：每个策略的最佳组合（按到点时间，仅成功运行）", ""]
                best = {}
                for r in runs:
                    if not r["metrics"].get("success"):
                        continue
                    key = (r["policy"]["name"], r["path"])
                    t = r["metrics"]["time_to_goal"]
                    if key not in best or t < best[key][0]:
                        best[key] = (t, r.get("gains"), r["metrics"]["cross_track_rms"])
                lines += ["| 策略 | 轨迹 | 最佳增益 | 到点时间 s | 横向 RMS m |", "|---|---|---|---|---|"]
                for (pol, path), (t, g, ct) in sorted(best.items()):
                    lines.append(f"| {pol} | {path} | {g} | {t:.1f} | {ct:.3f} |")
                lines.append("")
            lines += ["## 实验 1：巡线到点（策略 × 轨迹，均值 ± 标准差；成功率、首次进圈、最近距离、力统计取全部运行，其余只取成功运行）", "",
                      _table(runs, lambda r: r["path"] + (f" {r['gains']}" if r.get("gains") else ""), "轨迹", PATH_COLUMNS), ""]
            figures = _fig_paths(runs, out_dir)
        elif kind == "step":
            lines += ["## 实验 2：速度阶跃响应（策略 × 阶跃 × 载荷）", "",
                      _table(runs, lambda r: f"{r['step']['name']} @ {r['condition']['payload_kg']:g} kg", "阶跃 @ 载荷", STEP_COLUMNS), ""]
            figures = _fig_steps(runs, out_dir)
        elif kind == "robust":
            lines += ["## 实验 3：鲁棒性扫描（单因子，其余取默认）", ""]
            for factor, rows in _group(runs, lambda r: r.get("factor")).items():
                lines += [f"### 因子 `{factor}`", "", _table(rows, lambda r: f"{factor}={r['factor_value']:g}", "取值", PATH_COLUMNS), ""]
            figures = _fig_robust(runs, out_dir)
    except Exception as exc:  # figures must never block the tables
        lines += [f"_图表生成失败: {exc!r}_", ""]
    if figures:
        lines += ["## 图", ""] + [f"![{f}]({f})" for f in figures] + [""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[bench] report -> {path}", flush=True)
    return path
