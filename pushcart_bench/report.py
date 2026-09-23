"""runs.json -> report.md (tables) + PNG figures (trajectory overlays, error traces, step responses, degradation)."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from .metrics import aggregate, fmt_ms

PATH_COLUMNS = [("success", "成功率", None), ("time_to_goal", "到点时间 s", 1), ("time_enter", "首次进圈 s", 1),
                ("final_pos_err", "到点误差 m", 3), ("settled_pos_err", "停稳后误差 m", 3), ("min_goal_dist", "最近距离 m", 3),
                ("cross_track_rms", "横向 RMS m", 3), ("cross_track_max", "横向 max m", 3), ("heading_rms", "航向 RMS rad", 3),
                ("cmd_vx_rmse", "vx 跟踪 RMSE", 3), ("cmd_wz_rmse", "wz 跟踪 RMSE", 3), ("grasp_p95", "抓握 p95 N", 0),
                ("lateral_p95", "侧向力 p95 N", 0), ("overload_frac", "过载占比", 3),
                ("cmd_clip_frac", "命令残差饱和占比", 2), ("arm_clip_frac", "臂动作饱和占比", 2), ("arm_target_std", "臂修正波动 rad", 3)]
STEP_COLUMNS = [("failure", "失败数/运行数", None),
                ("rise_time", "上升 s", 2), ("reference_rise_time", "参考自身上升 s", 2),
                ("reference_settling_time", "参考自身整定 s", 2), ("settling_time", "整定 s", 2),
                ("settling_confirmed", "整定确认率", None), ("overshoot", "超调", 2), ("ss_error", "稳态误差", 3),
                ("post_rmse", "稳态 RMSE", 3), ("ripple_rms", "纹波 RMS", 3), ("ss_std", "慢振荡 std (0.4 s 滤波)", 3),
                ("drift_distance", "漂移 m", 2), ("grasp_p95", "抓握 p95 N", 0)]
RESPONSE_COLUMNS = [("reference_tracking_transient_rmse", "瞬态实际−参考 RMSE", 3),
                    ("reference_tracking_full_rmse", "全程实际−参考 RMSE", 3),
                    ("reference_tracking_full_iae", "全程实际−参考 IAE", 3),
                    ("reference_tracking_full_coverage", "参考跟踪有效覆盖率", 3),
                    ("cart_response_error_cost_mean_transient", "瞬态响应成本均值", 3),
                    ("cart_response_error_cost_integral_transient", "瞬态响应成本积分 s", 3),
                    ("cart_response_error_cost_integral_full", "全程响应成本积分 s", 3),
                    ("cart_response_full_coverage", "响应成本有效覆盖率", 3)]
STOP_COLUMNS = [("stopping_confirmed", "最终停稳确认率", None), ("stopping_distance", "最终确认距离 m", 3),
                ("stopping_time", "最终确认时间 s", 2), ("first_low_distance", "首次低速距离 m", 3),
                ("first_stop_confirmed_time", "首次确认时间 s", 2),
                ("post_first_stop_travel", "首次确认后移动 m", 3),
                ("post_stop_travel", "最终确认后移动 m", 3), ("stop_rebound_events", "回弹事件数", 0),
                ("stop_rebound_count", "超阈值样本数", 0), ("stop_invalid_samples", "无效样本数", 0)]


QUALITY_COLUMNS = [("cart_vx_band_rms", "vx 带限 RMS m/s", 3), ("cart_wz_band_rms", "wz 带限 RMS rad/s", 3),
                   ("cart_fx_mean", "净推力均值 N", 1), ("cart_mz_abs_p95", "偏航力矩 p95 N·m", 2),
                   ("cart_mz_band_rms", "力矩带限 RMS N·m", 2), ("cart_tilt_rad_max", "车倾角 max rad", 3),
                   ("wheel_lift_m_max", "轮离地 max m", 3), ("robot_tilt_rad_max", "机器人倾角 max rad", 3)]
RECOVERY_COLUMNS = [("disturbance_recovery_events", "扰动次数", 0), ("disturbance_recovered_frac", "恢复比例", 2),
                    ("disturbance_recovery_mean_s", "已恢复事件平均恢复 s", 2),
                    ("disturbance_recovery_max_s", "已恢复事件最大恢复 s", 2)]


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
                   "rise_time", "reference_rise_time", "reference_settling_time", "settling_time", "overshoot", "ss_error", "cart_response_error_cost_mean_transient", "post_rmse", "ripple_rms", "ss_std",
                   "stopping_confirmed", "stopping_distance", "stopping_time", "first_low_distance", "post_stop_travel",
                   "stop_rebound_count", "drift_distance"}


ALL_RUN_METRICS.update(column[0] for column in QUALITY_COLUMNS)
ALL_RUN_METRICS.update(column[0] for column in RECOVERY_COLUMNS)
ALL_RUN_METRICS.update(column[0] for column in RESPONSE_COLUMNS + STOP_COLUMNS)


def _cell(runs, metric, digits):
    """Success rate over all runs; goal / tracking metrics over the successful runs only (a run that drifted away for
    40 s would otherwise dominate the mean); interaction / step metrics over all runs."""
    if metric == "success":
        vals = [1.0 if r["metrics"].get("success") else 0.0 for r in runs]
        return f"{100 * np.mean(vals):.0f} % ({int(sum(vals))}/{len(vals)})" if vals else "-"
    if metric == "failure":
        return f"{sum(bool(r['metrics'].get('failure')) for r in runs)}/{len(runs)}"
    if metric in ("stopping_confirmed", "settling_confirmed"):
        vals = [bool(r["metrics"][metric]) for r in runs if r["metrics"].get(metric) is not None]
        return f"{100 * np.mean(vals):.0f} % ({sum(vals)}/{len(vals)})" if vals else "-"
    if metric == "stopping_distance":
        # A pre-V4 distance means first crossing, not final confirmation; never relabel it.
        runs = [r for r in runs if "stopping_confirmed" in r["metrics"]]
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
    """Keep the run identity: different versions often share the same model_N filename."""
    if name.startswith("two_stage:") and "/" in name:
        run, checkpoint = name.split(":", 1)[1].rsplit("/", 1)
        run = run.rsplit("/", 1)[-1]
        run = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", run)
        return f"{run}/{checkpoint}"
    return name


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
                ref = tr.get(f"cart_ref_{sig}")
                if ref is not None:
                    ax.plot(tr["t"], ref, ":", lw=1.0, label=f"{_short(r['policy']['name'])} 参考")
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
            if meta.get("recomputation"):
                rec = meta["recomputation"]
                lines += [f"重算来源：`{rec['source']}`；{rec.get('note', '')}", ""]
            if any("settling_confirmed" not in r["metrics"] for r in runs):
                lines += ["注意：部分运行使用旧指标，缺少持续整定/停稳确认字段；重新生成报告不会自动重算原始指标。", ""]
            lines += ["## 实验 2：速度阶跃响应（策略 × 阶跃 × 载荷）", "",
                      "上升/整定/超调相对原始命令计算；参考自身上升/整定只描述任务参考曲线，不能直接解释为控制器延迟。"
                      "整定必须在最后的连续入带区间满足保持时长。未确认的时间为缺失，均值只包含已确认运行，须结合确认率阅读。", "",
                      _table(runs, lambda r: f"{r['step']['name']} @ {r['condition']['payload_kg']:g} kg", "阶跃 @ 载荷", STEP_COLUMNS), ""]
            if any("reference_tracking_full_rmse" in r["metrics"] for r in runs):
                lines += ["## 实际车速对动态参考的跟踪", "",
                          "全程指命令切换后到测试结束，瞬态时长取 defaults.transient_s；原始跟踪误差未经包络平滑。"
                          "RMSE 单位同被测速度，IAE 为绝对误差对时间积分；响应成本是训练用短时滤波误差的无量纲成本。"
                          "积分不跨无效样本或采样缺口，覆盖率不足时不能将较小积分直接当成改善。", "",
                          _table(runs, lambda r: f"{r['step']['name']} @ {r['condition']['payload_kg']:g} kg", "阶跃 @ 载荷", RESPONSE_COLUMNS), ""]
            stops = [r for r in runs if r.get("step", {}).get("stopping")]
            if stops:
                lines += ["## 停车与回弹", "",
                          "最终停稳要求记录末尾连续低于阈值并达到 stop_dwell_s；中间短暂停住后再次运动不会算作最终停稳。"
                          "首次低速、首次确认与最终确认分别记录。回弹事件数统计首次低速后重新超阈值的连续段；样本数随采样率改变。"
                          "无效位姿跨越的距离无法确认，显示缺失。旧报告的首次低速距离不会自动改名为最终停止距离。", "",
                          _table(stops, lambda r: f"{r['step']['name']} @ {r['condition']['payload_kg']:g} kg", "停车 @ 载荷", STOP_COLUMNS), ""]
            figures = _fig_steps(runs, out_dir)
        elif kind == "robust":
            lines += ["## 实验 3：鲁棒性扫描（单因子，其余取默认）", ""]
            for factor, rows in _group(runs, lambda r: r.get("factor")).items():
                lines += [f"### 因子 `{factor}`", "", _table(rows, lambda r: f"{factor}={r['factor_value']:g}", "取值", PATH_COLUMNS), ""]
            figures = _fig_robust(runs, out_dir)
    except Exception as exc:  # figures must never block the tables
        lines += [f"_图表生成失败: {exc!r}_", ""]
    if any("quality_samples" in r["metrics"] for r in runs):
        lines += ["## 车侧施力与运动质量", "",
                  "带限量采用各策略描述中的滤波配置；它不替代上方原始纹波 RMS。"
                  "力矩参考点为整车质心。几何峰值包含初始阶段与失败运行；其余量排除滤波预热，阶跃另排除命令瞬态。", "",
                  _table(runs, lambda r: f"{r.get('path', r.get('step', {}).get('name', ''))} @ {r['condition']['payload_kg']:g} kg"
                         + (f" {r['factor']}={r['factor_value']}" if r.get("factor") else ""), "工况", QUALITY_COLUMNS), ""]
    if any(r.get("recovery") for r in runs):
        lines += ["## 车侧扰动后的任务恢复", "",
                  "恢复时间从扰动结束到首次进入并持续满足阈值的区间起点计算；"
                  "within_tolerance 表示观察窗口开始即保持在容差内，不表示没有运动波动。"
                  "平均时间只统计满足恢复条件的事件，未恢复和被后续扰动打断的事件保留在 runs.json，不能按 0 秒处理。", "",
                  _table(runs, lambda r: f"{r.get('factor', '')}={r.get('factor_value', '')}", "扰动", RECOVERY_COLUMNS), ""]
    if figures:
        lines += ["## 图", ""] + [f"![{f}]({f})" for f in figures] + [""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[bench] report -> {path}", flush=True)
    return path
