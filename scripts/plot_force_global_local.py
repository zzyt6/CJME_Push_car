"""Force time series with compact per-hand RMSE panels beside the local view."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

from plot_force_velocity_predict import (
    ROOT, FORCES, FORCE_LABELS, force_pair, load_trial, save, sha, style, write_csv,
)

TRUE_BLUE = "#355FAA"
PRED_RED = "#F00000"
FIGURE_MM = (180., 100.)
GLOBAL_MM = (17., 61.4, 158., 31.6)
LOCAL_MM = (17., 13., 85.32, 31.6)
RMSE_MM = ((115., 13., 25., 31.6), (150., 13., 25., 31.6))
WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
PANEL_LABEL_OFFSET_PT = (-10., 6.)


def label_panel(ax, letter):
    # Use one physical offset from each axes corner, independent of its width.
    # Right alignment gives every label the same gap to its plotting rectangle.
    ax.annotate(f"({letter})", xy=(0., 1.), xycoords="axes fraction",
                xytext=PANEL_LABEL_OFFSET_PT, textcoords="offset points",
                ha="right", va="bottom", fontsize=9, fontfamily="Arial",
                fontstyle="normal", fontweight="normal", annotation_clip=False)


def compute_hand_rmse(data):
    # Physical predictions and the matching training labels are synchronous.
    # Accumulate in float64 over the complete run, including all transients.
    error = (data["pred_hand_wrench"].astype(np.float64)
             - data["true_hand_wrench"].astype(np.float64))
    return np.sqrt(np.mean(error**2, axis=0)).reshape(2, 6)


def preserve_previous(output, stem):
    old_manifest = output / f"{stem}_manifest.json"
    if not old_manifest.is_file():
        return
    destination = output / "archive" / f"{stem}_{sha(old_manifest)[:12]}"
    if destination.exists():
        return
    destination.mkdir(parents=True)
    for suffix in ("pdf", "png", "svg"):
        source = output / "figures" / f"{stem}.{suffix}"
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    shutil.copy2(old_manifest, destination / old_manifest.name)


def draw_rmse_panels(fig, hand_rmse):
    colors = ("#86B2E2", "#8FC9D9", "#FFAC65")
    # Shared physical scale for the six force components (all in newtons).
    forces = hand_rmse[:, :3]
    ymax = np.ceil(forces.max()*1.10/.5)*.5
    panels = []
    for hand, (x, y, width, height) in enumerate(RMSE_MM):
        ax = fig.add_axes([x/FIGURE_MM[0], y/FIGURE_MM[1],
                           width/FIGURE_MM[0], height/FIGURE_MM[1]])
        panels.append(ax)
        values = forces[hand]
        bars = ax.bar(np.arange(3), values, width=.64, color=colors,
                      edgecolor="#303030", linewidth=.55, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=2, fontsize=6.5)
        ax.set_xticks(np.arange(3), [r"$F_x$", r"$F_y$", r"$F_z$"])
        ax.set_xlim(-.6, 2.6)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.grid(axis="y", color="#D7D7D7", linestyle="--", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=7, pad=2.5)
        if hand == 0:
            ax.set_ylabel("RMSE (N)", fontsize=8, labelpad=2)
        ax.set_xlabel(("Left hand", "Right hand")[hand], fontsize=8, labelpad=5)
        for spine in ax.spines.values():
            spine.set_color("#B8B8B8")
            spine.set_linewidth(.6)
        label_panel(ax, chr(99+hand))
    return panels


def create_figure(t, truth, prediction, ylabel, zoom, hand_rmse):
    style()
    plt.rcParams.update({"font.family": "Arial", "font.style": "normal", "font.weight": "normal",
                         "axes.labelweight": "normal", "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
                         "mathtext.default": "regular", "font.size": 8, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=np.asarray(FIGURE_MM) / 25.4)
    axes = []
    for index, (x, y, width, height) in enumerate((GLOBAL_MM, LOCAL_MM)):
        ax = fig.add_axes([x/FIGURE_MM[0], y/FIGURE_MM[1],
                           width/FIGURE_MM[0], height/FIGURE_MM[1]])
        axes.append(ax)
        local = index == 1
        bounds = zoom if local else (0., float(t[-1]))
        mask = (t >= bounds[0]-1e-8) & (t <= bounds[1]+1e-8)
        # True has no markers. Prediction points use their own artist so their
        # opacity stays at 1 even though the connecting red line is translucent.
        true_line, = ax.plot(t[mask], truth[mask], color=TRUE_BLUE,
                             linestyle=(0, (4, 2.4)), linewidth=.65, zorder=3)
        pred_line, = ax.plot(t[mask], prediction[mask], color=PRED_RED,
                             linewidth=.6, alpha=.5, zorder=4)
        pred_points, = ax.plot(t[mask], prediction[mask], linestyle="none",
                               marker="o", markersize=1.45 if local else .65,
                               markeredgewidth=0, color=PRED_RED, alpha=1., zorder=5)
        ax.set_xlim(*bounds)
        ax.set_xlabel("Time (s)", labelpad=3)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MultipleLocator(1 if local else 4))
        ax.tick_params(length=2.5, width=.65, pad=2.5)
        ax.grid(True, color="#E5E5E5", linewidth=.4)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
            spine.set_color("#333333")
        values = np.r_[truth[mask], prediction[mask]]
        span = max(float(np.ptp(values)), 1.)
        # Keep every extreme and leave a little breathing room for the legend.
        ax.set_ylim(float(values.min())-.10*span, float(values.max())+.38*span)
        ax.legend([(pred_line, pred_points), true_line], ["Prediction", "True"],
                  handler_map={tuple: HandlerTuple(ndivide=1)}, loc="upper right",
                  fontsize=7, frameon=True, framealpha=.92, facecolor="white",
                  edgecolor="#A0A0A0", borderpad=.3, labelspacing=.2,
                  handlelength=2.2, handletextpad=.55, borderaxespad=.35)
        label_panel(ax, chr(97+index))
    rmse_axes = draw_rmse_panels(fig, hand_rmse)
    fig.canvas.draw()
    geometry = []
    for ax in axes:
        rectangle = ax.get_position().bounds
        measured = [rectangle[0]*FIGURE_MM[0], rectangle[1]*FIGURE_MM[1],
                    rectangle[2]*FIGURE_MM[0], rectangle[3]*FIGURE_MM[1]]
        geometry.append(measured)
    np.testing.assert_allclose([geometry[0][2]/geometry[0][3], geometry[1][2]/geometry[1][3]], [5., 2.7])
    np.testing.assert_allclose(geometry[0][0], geometry[1][0])
    np.testing.assert_allclose(geometry[0][3], geometry[1][3])
    np.testing.assert_allclose(geometry[1][2]/geometry[0][2], .54)
    for ax in rmse_axes:
        position = ax.get_position().bounds
        geometry.append([position[0]*FIGURE_MM[0], position[1]*FIGURE_MM[1],
                         position[2]*FIGURE_MM[0], position[3]*FIGURE_MM[1]])
    return fig, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "logs/Picture/force_velocity_predict/isaac")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/Picture/force_velocity_predict")
    parser.add_argument("--channel", choices=FORCES, default="left_fx")
    parser.add_argument("--trial-axis", choices=("vx", "vy", "wz"), default="vx")
    parser.add_argument("--zoom-start", type=float, default=6.)
    parser.add_argument("--zoom-duration", type=float, default=4.)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    data, metadata, source = load_trial(args.source.resolve(), args.trial_axis)
    if metadata["engine"] != "Isaac Sim / PhysX":
        raise ValueError("Use the user-selected Isaac reference data")
    t = data["t"]
    zoom = (args.zoom_start, args.zoom_start+args.zoom_duration)
    if not 0 <= zoom[0] < zoom[1] <= t[-1]:
        raise ValueError("Local window must lie within the recorded trajectory")
    column = FORCES.index(args.channel)
    all_truth, all_prediction = force_pair(data)
    truth, prediction = all_truth[:, column], all_prediction[:, column]
    output = args.output.resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "derived").mkdir(parents=True, exist_ok=True)
    stem = f"force_{args.channel}_global_local"
    preserve_previous(output, stem)
    hand_rmse = compute_hand_rmse(data)
    fig, geometry = create_figure(t, truth, prediction, FORCE_LABELS[column], zoom, hand_rmse)
    paths = save(fig, output / "figures" / stem, args.dpi)
    mask = (t >= zoom[0]-1e-8) & (t <= zoom[1]+1e-8)
    csv_path = output / "derived" / f"{stem}.csv"
    write_csv(csv_path, ["time_s", "true_N", "prediction_N", "in_local_window"],
              [t, truth, prediction, mask])
    rmse_csv = output / "derived" / f"{stem}_rmse.csv"
    with rmse_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hand", "component", "rmse", "unit", "samples", "plotted"])
        for hand, side in enumerate(("left", "right")):
            for component, name in enumerate(WRENCH_COMPONENTS):
                writer.writerow([side, name, float(hand_rmse[hand, component]),
                                 "N" if component < 3 else "N m", len(t), component < 3])
    manifest = dict(engine=metadata["engine"], channel=args.channel, trial_axis=args.trial_axis,
                    checkpoint=metadata["checkpoint"], payload_kg=metadata["payload_kg"],
                    source=str(source), source_sha256=sha(source), metadata_sha256=sha(source.with_suffix(".json")),
                    source_samples=len(t), local_window_s=zoom, local_samples=int(mask.sum()),
                    sampling_hz=50, data_processing="None; exact synchronous Isaac true/predicted forces",
                    figure_size_mm=FIGURE_MM, axes_rectangles_mm=geometry,
                    axes_width_height_ratios=[5., 2.7], local_global_width_ratio=.54,
                    panel_labels=["(a)", "(b)", "(c)", "(d)"], equal_panel_height=True,
                    typography=dict(family="Arial", style="normal", weight="normal",
                                    mathtext="upright Arial with subscripts",
                                    panel_label_anchor="axes upper-left", panel_label_align="right/bottom",
                                    panel_label_offset_pt=PANEL_LABEL_OFFSET_PT),
                    time_series_left_aligned=True,
                    rmse_panels=dict(left="(c)", right="(d)", components=list(WRENCH_COMPONENTS[:3]),
                                     unit="N", samples=len(t), interval_s=[0., float(t[-1])],
                                     sample_timestamps_s=[float(t[0]), float(t[-1])],
                                     left_values=hand_rmse[0, :3].tolist(), right_values=hand_rmse[1, :3].tolist(),
                                     aggregation="sqrt(mean((prediction - true)**2)), float64, full run",
                                     common_y_axis=True, numeric_labels="two decimal places",
                                     grid="horizontal dashed", size_mm=[25., 31.6]),
                    true_style=dict(color=TRUE_BLUE, linestyle="dashed", linewidth_pt=.65, markers=False),
                    prediction_style=dict(color=PRED_RED, linewidth_pt=.6, alpha=.5,
                                          marker_color=PRED_RED, marker_alpha=1., marker_every_sample=True,
                                          global_marker_size_pt=.65, local_marker_size_pt=1.45),
                    script_sha256=sha(__file__), csv_sha256=sha(csv_path), rmse_csv_sha256=sha(rmse_csv),
                    figures={str(p.relative_to(output)): sha(p) for p in paths})
    (output / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
