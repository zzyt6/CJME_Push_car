"""Render measured dual-hand force envelopes, with conditional capability context."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

from analyze_force_use_range import load, kde_pair, PLANES

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED = "#397DAF", "#C61F36"
LIGHT_BLUE = ["#EBF3F9", "#B8D3E8", "#679BC3"]
LIGHT_RED = ["#FDE7E3", "#F7A495", "#DF3C48"]
NAMES = [r"$F_x$ (N)", r"$F_y$ (N)", r"$F_z$ (N)"]


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "mathtext.fontset": "dejavusans", "axes.linewidth": .55,
                         "savefig.facecolor": "white"})


def save(fig, directory, stem, pdf=True):
    for extension in (["pdf"] if pdf else []) + ["svg", "png"]:
        fig.savefig(directory / f"{stem}.{extension}", dpi=420)


def hull(ax, force, i, j, color, alpha=.85):
    points = force[:, [i, j]]
    vertices = points[ConvexHull(points).vertices]
    ax.add_patch(Polygon(vertices, closed=True, fill=False, edgecolor=color,
                         linewidth=.65, linestyle=(0, (3, 2)), alpha=alpha, zorder=5))


def density(ax, kde, i, j, points, *, small=False):
    x, y = kde["x"], kde["y"]
    for method, colors, line in (("baseline", LIGHT_BLUE, BLUE), ("CWC", LIGHT_RED, RED)):
        p = np.asarray(kde[method+"_density"]).T
        thresholds = sorted(kde[method+"_thresholds"])
        levels = thresholds + [float(p.max())*1.00001]
        ax.contourf(x, y, p, levels=levels, colors=colors, alpha=.78 if method == "CWC" else .85, zorder=2 if method == "baseline" else 3)
        ax.contour(x, y, p, levels=thresholds, colors=[line], linewidths=[.75, .6, .6], alpha=.85, zorder=4)
        mean = points[method].mean(axis=0)
        ax.plot(mean[i], mean[j], marker="+", color=line, markersize=5 if not small else 3, mew=1, zorder=6)


def circle_grid(ax, radius, label=False):
    for frac in (.25, .5, .75):
        ax.add_patch(Circle((0, 0), radius*frac, fill=False, edgecolor="#DDE1E5", lw=.5, zorder=0))
    ax.axhline(0, color="#BFC5CC", lw=.5, zorder=0)
    ax.axvline(0, color="#BFC5CC", lw=.5, zorder=0)
    for angle in (np.pi/4, 3*np.pi/4):
        vec = radius*np.array([np.cos(angle), np.sin(angle)])
        ax.plot([-vec[0], vec[0]], [-vec[1], vec[1]], color="#E4E7EA", lw=.5, ls="--", zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(-radius*1.07, radius*1.07)
    ax.set_ylim(-radius*1.07, radius*1.07)
    ax.spines[:].set_visible(False)
    if not label:
        ax.set_xticks([]); ax.set_yticks([])


def legend_handles():
    return [Line2D([], [], color="black", lw=1.2, label="Actuator-only reference bound"),
            Line2D([], [], color=BLUE, lw=2.5, label="Pass-through"),
            Line2D([], [], color=RED, lw=2.5, label="V6 / CWC")]


def zoom_panel(ax, points, plane, label, *, full=False):
    i, j, tag = plane
    kde = kde_pair(points, plane)
    X, Y = np.meshgrid(kde["x"], kde["y"], indexing="ij")
    body = np.zeros(X.shape, dtype=bool)
    for method in ("baseline", "CWC"):
        body |= kde[method+"_density"] >= kde[method+"_thresholds"][-1]
    extent = max(float(np.abs(X[body]).max()), float(np.abs(Y[body]).max()))
    radius = max(10., float(np.ceil(extent*1.08/5)*5))
    # This faint frame is only a plotting grid, never the black capability circle.
    circle_grid(ax, radius, label=True)
    ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#B8C0C8", lw=.55, zorder=0))
    density(ax, kde, i, j, points)
    ax.set_xlabel(NAMES[i], labelpad=1)
    ax.set_ylabel(NAMES[j], labelpad=1)
    ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
    ax.tick_params(length=2, width=.5, pad=1)
    title = {"xy": "Planar resultant", "xz": "Forward / vertical", "yz": "Lateral / vertical"}[tag]
    ax.set_title(f"({label}) {title}", pad=7)
    return radius


def smooth_force(force):
    # 25 control samples at 50 Hz; truncate and renormalize at window boundaries.
    denominator = uniform_filter1d(np.ones((len(force), 1)), 25, axis=0, mode="constant")
    return uniform_filter1d(force, 25, axis=0, mode="constant") / denominator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "logs/Picture/force_use_range")
    args = p.parse_args(); root = args.root.resolve()
    figures = root / "figures"; figures.mkdir(exist_ok=True)
    trials = load(root)
    raw_points = {m: r["force"][r["selected"]] for m, r in trials.items()}
    points = {m: smooth_force(f) for m, f in raw_points.items()}
    bounds = json.loads((root / "derived/capability_bound.json").read_text())
    metrics = json.loads((root / "derived/metrics.json").read_text())
    smooth_metrics = {"smoothing": "Centered 25-sample (0.5 s nominal) moving mean; edge windows truncated and renormalized; same filter for both methods.",
                      "envelope": "Only highest-density regions containing 50/80/95 percent of the 2-D KDE probability are drawn; the 95% HDR is the outer boundary. No all-sample convex hull is plotted.",
                      "density_caveat": "95% probability mass under the smoothed KDE is not exactly 95% of discrete samples; actual sample inclusion is recorded per projection.",
                      "window": metrics["window"], "methods": {}, "planes": {}}
    smooth_arrays = {}
    for method, f in points.items():
        smooth_metrics["methods"][method] = {"samples": len(f), "force_mean_N": f.mean(0).tolist(),
                                               "force_std_N": f.std(0).tolist(),
                                               "smoothed_force_norm_p95_N": float(np.quantile(np.linalg.norm(f, axis=1), .95))}
        smooth_arrays[method+"_force"] = f
    for plane in PLANES:
        i, j, tag = plane
        kde = kde_pair(points, plane)
        smooth_arrays.update({tag+"_"+k: v for k, v in kde.items()})
        smooth_metrics["planes"][tag] = {"bandwidth_N": kde["bandwidth_N"]}
        for method, f in points.items():
            density_values = RegularGridInterpolator((kde["x"], kde["y"]), kde[method+"_density"], bounds_error=False, fill_value=0.)(f[:, [i, j]])
            keep = density_values >= kde[method+"_thresholds"][-1]
            smooth_arrays[f"{method}_{tag}_inside_hdr95"] = keep
            smooth_metrics["planes"][tag][method] = {"hdr95_area_N2": kde[method+"_hdr95_area_N2"],
                                                     "sample_inclusion_fraction": float(keep.mean()),
                                                     "outside_samples": int((~keep).sum())}
    np.savez_compressed(root / "derived/force_envelopes_smoothed95.npz", **smooth_arrays)
    (root / "derived/smoothed95_metrics.json").write_text(json.dumps(smooth_metrics, indent=2), encoding="utf-8")
    import csv
    for method, f in points.items():
        trial = trials[method]
        with (root / f"derived/{method}_smoothed95_force.csv").open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["t_s", "route_progress_m", "smoothed_Fx_N", "smoothed_Fy_N", "smoothed_Fz_N", "inside_xy_95pct_density", "inside_xz_95pct_density", "inside_yz_95pct_density"])
            for idx, (t, s, force) in enumerate(zip(trial["data"]["t"][trial["selected"]], trial["progress"][trial["selected"]], f)):
                writer.writerow([t, s, *force, *[int(smooth_arrays[f"{method}_{tag}_inside_hdr95"][idx]) for _, _, tag in PLANES]])
    style()

    fig = plt.figure(figsize=(180/25.4, 145/25.4))
    fig.text(.5, .972, "Dual-hand force usage: smoothed 95% density envelopes", ha="center", va="top", fontsize=10.5, weight="bold")
    fig.text(.5, .934, "Isaac Sim  |  15 kg payload  |  0.5 m/s nominal speed  |  first 19 m of the composite route", ha="center", fontsize=7.2, color="#4A5159")
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.5, .916), ncol=3,
               frameon=False, handlelength=2, columnspacing=1.5)
    for k, plane in enumerate(PLANES):
        i, j, tag = plane
        left = .079 + k*.312
        ax = fig.add_axes([left, .544, .258, .305])
        bound = bounds["planes"][tag]; radius = bound["outer_circle_radius_N"]
        circle_grid(ax, radius)
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#171717", lw=1.25, zorder=6))
        polygon = np.asarray(bound["outer_polygon_N"])
        ax.add_patch(Polygon(polygon, closed=True, facecolor="#E8EBEE", alpha=.38, edgecolor="#757D85", lw=.7))
        density(ax, kde_pair(points, plane), i, j, points, small=True)
        ax.set_title(f"({chr(97+k)}) {NAMES[i].replace(' (N)', '')} - {NAMES[j].replace(' (N)', '')}", pad=7)
        ax.text(.5, -.08, f"Reference-circle radius: {radius:.0f} N", transform=ax.transAxes,
                ha="center", fontsize=7)
        zoom = fig.add_axes([left, .142, .258, .323])
        zr = zoom_panel(zoom, points, plane, chr(100+k))
        ax.add_patch(Rectangle((-zr, -zr), 2*zr, 2*zr, fill=False, ec="#777777", lw=.65, zorder=7))
    fig.text(.5, .055, "0.5 s moving mean; nested 50 / 80 / 95% density regions; outer 5% density tails omitted; +: means.",
             ha="center", fontsize=7)
    fig.text(.5, .025, "Black circles bound a posture-conditioned, quasi-static actuator model; they do not certify whole-body attainable forces.",
             ha="center", fontsize=6.7, color="#4A5159")
    save(fig, figures, "force_use_range")
    plt.close(fig)

    fig = plt.figure(figsize=(180/25.4, 77/25.4))
    fig.text(.5, .972, "Smoothed dual-hand force: 95% density envelopes", ha="center", va="top", fontsize=10.5, weight="bold")
    fig.legend(handles=legend_handles()[1:], loc="upper center", bbox_to_anchor=(.5, .927), ncol=2, frameon=False)
    for k, plane in enumerate(PLANES):
        ax = fig.add_axes([.079+k*.312, .19, .258, .59])
        zoom_panel(ax, points, plane, chr(97+k))
    fig.text(.5, .042, "15 kg, 0.5 m/s, first 19 m. Same 0.5 s moving mean; 50 / 80 / 95% density regions; low-density tails omitted.", ha="center", fontsize=6.7)
    save(fig, figures, "force_use_range_zoom")
    plt.close(fig)

    fig = plt.figure(figsize=(180/25.4, 142/25.4))
    fig.text(.5, .977, "Raw-data audit: route performance and force diagnostics", ha="center", va="top", fontsize=10, weight="bold")
    ax = fig.add_axes([.08, .58, .43, .29])
    with np.load(root / "raw/baseline_reference_path.npz") as reference:
        xy = reference["xy"]
    origin = xy[0]
    ax.plot(*(xy-origin).T, color="#8A9299", ls="--", lw=1.1, label="Reference")
    for method, color, name in (("baseline", BLUE, "Pass-through"), ("CWC", RED, "V6 / CWC")):
        trial = trials[method]; pose = trial["data"]["cart_pose"][trial["selected"], :2]-origin
        ax.plot(*pose.T, color=color, lw=.8, label=name)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) Matched route interval", loc="left"); ax.legend(loc="upper left", fontsize=6.4)
    ax.grid(color="#E3E6E9", lw=.5)
    ax = fig.add_axes([.59, .58, .35, .29])
    for method, color in (("baseline", BLUE), ("CWC", RED)):
        trial = trials[method]; selected = trial["selected"]
        f = np.linalg.norm(trial["force"][selected], axis=1); progress = trial["progress"][selected]
        ax.scatter(progress, f, s=.4, alpha=.18, color=color, rasterized=True)
        ax.plot(progress, uniform_filter1d(f, 25, mode="nearest"), lw=.8, color=color)
    ax.set_title("(b) Resultant-force magnitude", loc="left"); ax.set_xlabel("Route progress (m)"); ax.set_ylabel("|F| (N)")
    ax.set_xlim(0, 19); ax.set_ylim(bottom=0); ax.grid(color="#E3E6E9", lw=.5)
    ax = fig.add_axes([.08, .16, .40, .29])
    values, positions, colors = [], [], []
    for i in range(3):
        for offset, method, color in ((-.18, "baseline", BLUE), (.18, "CWC", RED)):
            values.append(raw_points[method][:, i]); positions.append(i+offset); colors.append(color)
    artists = ax.boxplot(values, positions=positions, widths=.28, whis=(5, 95), patch_artist=True,
                        showfliers=True, flierprops={"markersize": 1, "alpha": .22},
                        medianprops={"color": "white", "linewidth": 1.})
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(.78)
    ax.set_xticks([0, 1, 2], [r"$F_x$", r"$F_y$", r"$F_z$"]); ax.set_ylabel("Force (N)")
    ax.set_title("(c) Component distributions", loc="left"); ax.axhline(0, color="#999999", ls="--", lw=.6)
    ax.grid(axis="y", color="#E3E6E9", lw=.5)
    ax = fig.add_axes([.54, .12, .43, .355]); ax.axis("off")
    a, b = [metrics["methods"][m] for m in ("baseline", "CWC")]
    cell = [
        ["Cross-track RMSE (m)", f"{a['cross_track_rmse_m']:.3f}", f"{b['cross_track_rmse_m']:.3f}"],
        ["Heading RMSE (deg)", f"{a['heading_rmse_deg']:.2f}", f"{b['heading_rmse_deg']:.2f}"],
        ["Time to 19 m (s)", f"{a['time_to_19m_s']:.2f}", f"{b['time_to_19m_s']:.2f}"],
        ["Force norm P95 (N)", f"{a['force_norm_p95_N']:.2f}", f"{b['force_norm_p95_N']:.2f}"],
        ["Peak force norm (N)", f"{a['force_norm_max_N']:.2f}", f"{b['force_norm_max_N']:.2f}"],
        ["Mean vertical force (N)", f"{a['force_mean_N'][2]:.2f}", f"{b['force_mean_N'][2]:.2f}"],
        ["Local fluctuation RMS (N)", f"{a['local_force_fluctuation_rms_N']:.2f}", f"{b['local_force_fluctuation_rms_N']:.2f}"],
        ["Opposing-force fraction", f"{a['opposing_force_fraction']:.3f}", f"{b['opposing_force_fraction']:.3f}"],
    ]
    table = ax.table(cellText=cell, colLabels=["Metric", "Pass-through", "V6"], colWidths=[.56, .27, .17],
                     loc="center", cellLoc="right")
    table.auto_set_font_size(False); table.set_fontsize(6.3); table.scale(1., 1.45)
    for (r, c), cellobj in table.get_celld().items():
        cellobj.set_edgecolor("#D3D8DD"); cellobj.set_linewidth(.4)
        if r == 0: cellobj.set_facecolor("#EDF1F5")
        if c == 0: cellobj.set_text_props(ha="left")
    ax.set_title("(d) Statistics over the first 19 m", loc="left", fontsize=8.2)
    fig.text(.5, .052, "Force = reconstructed hand-on-cart load (PhysX incoming wrench minus hand-subtree Newton-Euler load).", ha="center", fontsize=6.5)
    fig.text(.5, .025, "Full route: V6 settles at 41.16 s; pass-through does not satisfy the stop criterion within 90 s. One nominal trial per method.", ha="center", fontsize=6.5)
    save(fig, figures, "force_use_range_diagnostics")
    plt.close(fig)

    # All-run diagnostic explicitly includes different stop/timeout exposure durations.
    all_points = {m: smooth_force(r["force"][r["valid"]]) for m, r in trials.items()}
    fig = plt.figure(figsize=(180/25.4, 77/25.4))
    fig.text(.5, .971, "Full-rollout smoothed 95% envelopes - unequal terminal dwell", ha="center", va="top", fontsize=9.5)
    fig.legend(handles=legend_handles()[1:], loc="upper center", bbox_to_anchor=(.5, .927), ncol=2, frameon=False)
    for k, plane in enumerate(PLANES):
        zoom_panel(fig.add_axes([.079+k*.312, .19, .258, .59]), all_points, plane, chr(97+k))
    fig.text(.5, .042, "0.5 s smoothing, 95% density. Pass-through: 90.02 s; V6: 42.16 s. Unequal full-rollout exposure durations.", ha="center", fontsize=6.8)
    save(fig, figures, "full_rollout_envelopes", pdf=False)
    plt.close(fig)
    manifest = {"style": {"baseline": BLUE, "V6": RED, "density_probabilities": [.5, .8, .95]},
                "smoothing": smooth_metrics["smoothing"], "main_figure_envelope": smooth_metrics["envelope"],
                "density": "Common pooled-data bandwidth and grid per projection; each method integrates to one; color represents within-rollout density, not across-trial confidence",
                "force_source": trials["CWC"]["meta"]["force_measurement"],
                "capability_definition": bounds["definition"],
                "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in figures.iterdir() if f.is_file()}}
    (root / "plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[force-plots]", figures)


if __name__ == "__main__":
    main()
