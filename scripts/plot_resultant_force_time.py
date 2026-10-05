"""Time-series force plot: central 95% raw magnitudes, local mean +/- std.

Uses the same previously selected active-route records as the source force plot.
The horizontal coordinate is each rollout's true elapsed time, never rescaled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from analyze_force_use_range import load

ROOT = Path(__file__).resolve().parents[1]
STEM = "resultant_force_time_central95"
COLORS = {"baseline": "#147FAF", "CWC": "#CF283E"}
LABELS = {"baseline": "Locomotion-only", "CWC": "CWC"}
# elsarticle final/5p: (textwidth 522 TeX pt - columnsep 18 TeX pt) / 2.
FIGURE_WIDTH_MM = 252 / 72.27 * 25.4
FIGURE_HEIGHT_MM = 55.
DISPLAY_END_S = 30.
FONT_SIZE_PT = 8.


def summarize(t, force, window_s=.5):
    lower, upper = np.quantile(force, [.025, .975], method="linear")
    retained = (force >= lower) & (force <= upper)
    mean = np.full(len(t), np.nan)
    std = np.full(len(t), np.nan)
    count = np.zeros(len(t), dtype=int)
    kept_t, kept_force = t[retained], force[retained]
    # A physical-time window: removing outliers must not compress time.
    left = np.searchsorted(kept_t, t-window_s/2-1e-9, side="left")
    right = np.searchsorted(kept_t, t+window_s/2+1e-9, side="right")
    for i, (a, b) in enumerate(zip(left, right)):
        values = kept_force[a:b]
        count[i] = len(values)
        if len(values):
            mean[i] = values.mean()
            std[i] = values.std(ddof=0)
    valid = count > 0
    assert np.all(mean[valid] >= lower-1e-10) and np.all(mean[valid] <= upper+1e-10)
    assert retained.sum() + (~retained).sum() == len(t)
    return dict(q025_N=float(lower), q975_N=float(upper), retained=retained,
                mean=mean, std=std, lower=mean-std, upper=mean+std, count=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "logs/Picture/force_use_range")
    args = parser.parse_args(); root = args.root.resolve()
    figures, derived = root / "figures", root / "derived"
    figures.mkdir(exist_ok=True); derived.mkdir(exist_ok=True)
    trials = load(root)
    manifest = {
        "figure": STEM,
        "legend_labels": LABELS,
        "baseline_definition": "Frozen robot locomotion policy with zero task-level residuals; no learned task-level force coordination. The underlying locomotion policy still responds to contact loads through proprioceptive feedback.",
        "ordinate": "norm(F_left + F_right) in N; add vectors before taking the norm",
        "abscissa": "Actual elapsed simulation time in seconds, unchanged from raw timestamps",
        "presentation": {"display_time_s": [0., DISPLAY_END_S],
                         "width_mm": FIGURE_WIDTH_MM, "height_mm": FIGURE_HEIGHT_MM,
                         "font_family": "Arial", "all_text_size_pt": FONT_SIZE_PT,
                         "width_source": "CJME/main.tex uses elsarticle final,5p,twocolumn; elsarticle.cls textwidth=522 TeX pt and columnsep=18 TeX pt, giving columnwidth=252 TeX pt.",
                         "crop_policy": "Only the display interval is cropped. Existing central-95% thresholds and smoothed values are preserved; no re-estimation at the 30 s display edge."},
        "window_selection": "Same existing active-route mask as the user's source figure: valid post-physics samples with progress < 19.0. The figure uses time, not progress, on its horizontal axis.",
        "trim": "Per method, calculate raw force-magnitude Q2.5 and Q97.5 over the selected records; exclude samples outside this interval. No winsorization and no density/HDR selection.",
        "smoothing": "Centered 0.5 s physical-time window over retained raw force magnitudes; truncate edge windows; no time compression or extrapolation; blank where no samples are available",
        "band": "Local retained-sample mean +/- one population standard deviation (ddof=0); this is temporal variation, not a 95% confidence interval",
        "force_source": trials["CWC"]["meta"]["force_measurement"],
        "runs": {},
    }
    curves = {}
    for method, trial in trials.items():
        selected = trial["selected"]
        t = trial["data"]["t"][selected]
        force = np.linalg.norm(trial["force"][selected], axis=1)
        result = summarize(t, force)
        curves[method] = (t, result)
        fields = {"time_s": t, "raw_resultant_force_N": force,
                  "retained_central95": result["retained"].astype(int),
                  "smooth_mean_N": result["mean"], "local_std_N": result["std"],
                  "band_lower_N": result["lower"], "band_upper_N": result["upper"],
                  "window_retained_samples": result["count"]}
        with (derived / f"{STEM}_{method}.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp); writer.writerow(fields)
            for row in zip(*fields.values()):
                writer.writerow(["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row])
        np.savez_compressed(derived / f"{STEM}_{method}.npz", **fields)
        source = root / "raw" / f"{method}.npz"
        manifest["runs"][method] = {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source": source.relative_to(root).as_posix(),
            "samples": len(t), "retained": int(result["retained"].sum()),
            "discarded_low": int((force < result["q025_N"]).sum()),
            "discarded_high": int((force > result["q975_N"]).sum()),
            "retained_fraction": float(result["retained"].mean()),
            "q025_N": result["q025_N"], "q975_N": result["q975_N"],
            "first_time_s": float(t[0]), "last_time_s": float(t[-1]),
            "min_window_count": int(result["count"].min()),
            "max_window_count": int(result["count"].max()),
            "time_windows_without_samples": int((result["count"] == 0).sum()),
        }
    plt.rcParams.update({"font.family": "Arial", "font.size": FONT_SIZE_PT,
                         "axes.labelsize": FONT_SIZE_PT, "axes.titlesize": FONT_SIZE_PT,
                         "xtick.labelsize": FONT_SIZE_PT, "ytick.labelsize": FONT_SIZE_PT,
                         "legend.fontsize": FONT_SIZE_PT,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM/25.4, FIGURE_HEIGHT_MM/25.4))
    ax = fig.add_axes([13.5/FIGURE_WIDTH_MM, 10.5/FIGURE_HEIGHT_MM,
                       (FIGURE_WIDTH_MM-16.)/FIGURE_WIDTH_MM, 40.5/FIGURE_HEIGHT_MM])
    for method, label in LABELS.items():
        t, result = curves[method]
        display = t <= DISPLAY_END_S + 1e-9
        ax.fill_between(t[display], result["lower"][display], result["upper"][display],
                        color=COLORS[method], alpha=.19, linewidth=0, zorder=2)
        ax.plot(t[display], result["mean"][display], color=COLORS[method], linewidth=.95, label=label, zorder=3)
    ymax = max(float(np.nanmax(result["upper"][t <= DISPLAY_END_S+1e-9])) for t, result in curves.values())
    ax.set_xlim(0, DISPLAY_END_S)
    ax.set_ylim(0, np.ceil(ymax*1.09/10)*10)
    ax.set_xlabel("Task time (s)", labelpad=3)
    ax.set_ylabel("Resultant force (N)", labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, color="#DDE2E6", linewidth=.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=.5, pad=2)
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=.94,
              edgecolor="#D6DCE0", handlelength=1.6, handletextpad=.45,
              columnspacing=.9, borderpad=.35)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(figures / f"{STEM}.{suffix}", dpi=450)
    plt.close(fig)
    manifest["outputs_sha256"] = {f"figures/{STEM}.{suffix}": hashlib.sha256((figures / f"{STEM}.{suffix}").read_bytes()).hexdigest() for suffix in ("pdf", "svg", "png")}
    manifest["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# 双手合力—任务时间：中间95%采样点", "",
             "图：`figures/resultant_force_time_central95.pdf`，另有同名PNG和SVG。横轴为真实任务时间(s)，仅显示0–30 s；纵轴为作用于小车的双手合力大小(N)。蓝色图例为Locomotion-only，红色图例为CWC。", "",
             "Locomotion-only表示仅使用冻结的本体运控策略，高层残差置零，不含学习得到的任务层施力协调。它不是完全不感知接触载荷：底层策略仍会通过本体反馈响应载荷。红线CWC对应既有V6模型。", "",
             f"单栏排版尺寸为{FIGURE_WIDTH_MM:.4f} × {FIGURE_HEIGHT_MM:g} mm，坐标标题、刻度和图例均为8 pt Arial。宽度匹配论文elsarticle 5p双栏格式的252 TeX pt单栏宽度。按该物理宽度插入时，无需缩放，字体保持8 pt；建议使用 `\\includegraphics[width=\\linewidth]{{...}}`。", "",
             "合力先作向量相加，再取模：F = ||F_left + F_right||。对原始合力大小筛选，而不是先平滑三个力分量再取模；这两种运算不等价。",
             "", "按照第二张示例图的形式：实线为0.5 s中心时间窗口的均值，浅色带为同窗口保留点的均值±1总体标准差。先按各方法的原始力大小Q2.5～Q97.5保留中间95%数据，再计算窗口统计。去掉的点不压缩时间、不以阈值替换。",
             "", "这里95%指幅值两端各去除2.5%，与上一版二维95%密度区域不同。浅色带也不是95%置信区间，而是一次运行中的局部波动。没有独立的目标合力命令，所以不添加示例中的黑色目标阶跃线。",
             "", "底层统计范围沿用此前图的有效任务区间，95%阈值与平滑数值保持一致；本次只将显示轴限制到前30 s。CSV/NPZ保留完整派生序列，图中仅显示0–30 s，两个方法均使用原始时间戳。完整筛选条件保存在manifest中。", "",
             "| 方法 | 原始点数 | 保留点数 | 保留比例 | 下限(N) | 上限(N) | 最后时间(s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, label in LABELS.items():
        r = manifest["runs"][method]
        lines.append(f"| {label} | {r['samples']} | {r['retained']} | {r['retained_fraction']:.2%} | {r['q025_N']:.4f} | {r['q975_N']:.4f} | {r['last_time_s']:.2f} |")
    lines += ["", "`derived/resultant_force_time_central95_<method>.csv/.npz`保存每个时间点的原始合力、是否保留、平滑曲线、上下带和有效窗口点数。原始试验文件未修改。", "",
              "Suggested caption: Time histories of the dual-hand resultant force applied to the cart in matched Isaac Sim trials. Locomotion-only uses the frozen locomotion policy with zero task-level residuals, whereas CWC adds learned task-conditioned coordination. For each method, samples outside the 2.5th–97.5th percentile range of raw force magnitude are excluded. Solid curves show 0.5 s centered local means of retained samples, and shaded bands show ±1 local population standard deviation. Original timestamps are preserved.", "",
              "复现：`python scripts/plot_resultant_force_time.py`。", ""]
    (root / f"{STEM}_README.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(__file__, root / "provenance" / Path(__file__).name)
    print(json.dumps(manifest["runs"], indent=2))
    print(figures / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
