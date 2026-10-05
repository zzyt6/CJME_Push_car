"""Plot recorded V6 estimator time series, preserving all 50 Hz samples.

Force estimates are synchronous; cart velocity estimates predict t + 20 ms.
No smoothing, fitted time shifts, interpolation, or synthetic data are used.
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
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AXES = ("vx", "vy", "wz")
FORCES = ("left_fx", "left_fy", "right_fx", "right_fy")
FORCE_LABELS = (r"Left $F_x$ (N)", r"Left $F_y$ (N)", r"Right $F_x$ (N)", r"Right $F_y$ (N)")
VELOCITY_LABELS = (r"$v_x$ (m/s)", r"$v_y$ (m/s)", r"$\omega_z$ (rad/s)")
TRUE_COLOR, PRED_COLOR = "#526F9E", "#D8292F"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trial(source, axis):
    path = source / "raw" / f"{axis}_CWC.npz"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("npz_sha256") != sha(path):
        raise ValueError(f"Source hash mismatch: {path}")
    engine = meta.get("engine")
    if engine not in ("MuJoCo", "Isaac Sim / PhysX") or not meta.get("completed"):
        raise ValueError("Expected a complete recorded Isaac or MuJoCo trial")
    if meta.get("axis") != axis or meta.get("method") != "CWC":
        raise ValueError("Source trial identity mismatch")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    t = data["t"]
    if not np.isclose(meta["dt"], .02) or not np.allclose(np.diff(t), .02, atol=1e-8, rtol=0):
        raise ValueError("Expected contiguous 50 Hz observations")
    shapes = {"cart_velocity": (len(t), 3), "pred_cart_velocity": (len(t), 3),
              "pred_hand_wrench": (len(t), 12)}
    if engine == "Isaac Sim / PhysX":
        shapes.update(true_hand_wrench=(len(t), 12), actor_observation=(len(t), 605),
                      estimator_observation=(len(t), 462))
        if not meta.get("estimator_recording") or not np.isclose(meta["cart_prediction_horizon_s"], .02):
            raise ValueError("Expected recorded Isaac estimators with a 20 ms cart prediction horizon")
        if not data["hand_label_valid"].all() or not data["estimator_sample_valid"].all():
            raise ValueError("Invalid/reset estimator samples must not be plotted as valid predictions")
    else:
        shapes.update(cart_pose=(len(t), 7), hand_force_moment_world=(len(t), 2, 6))
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} in {path}")
    if data["terminated"].any() or data["timeout"].any():
        raise ValueError("Do not join reset episodes")
    if engine == "MuJoCo" and meta.get("pose_quaternion_order") != "xyzw":
        raise ValueError("Expected cart pose quaternion in xyzw order")
    return data, meta, path


def force_pair(data):
    if "true_hand_wrench" in data:
        # The exact Isaac training label, already in physical units and in the
        # same cart-heading axes / palm-origin convention as the force head.
        return data["true_hand_wrench"][:, [0, 1, 6, 7]], data["pred_hand_wrench"][:, [0, 1, 6, 7]]
    # MuJoCo records hand-on-cart forces in world axes. The V6 head uses
    # cart-heading axes. Forces are independent of the wrench moment origin;
    # these plots deliberately do not compare COM moments to palm moments.
    x, y, z, w = data["cart_pose"][:, 3:7].T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    force = data["hand_force_moment_world"]
    reference = np.stack((c*force[:, :, 0] + s*force[:, :, 1],
                          -s*force[:, :, 0] + c*force[:, :, 1]), -1).reshape(-1, 4)
    return reference, data["pred_hand_wrench"][:, [0, 1, 6, 7]]


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})


def axis_style(ax, ylabel, bounds):
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_xlim(*bounds)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, color="#E5E5E5", linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.margins(y=.12)


def draw_pair(ax, t, reference, prediction, *, local=False, velocity=False):
    # Lines always connect every raw sample. Sparse global markers only reduce
    # visual clutter; local figures mark each 20 ms sample on both curves.
    every = 1 if local else 25
    width = .85 if local else .65 if velocity else .48
    ref, = ax.plot(t, reference, color=TRUE_COLOR, linestyle=(0, (4, 2.6)),
                   linewidth=width, alpha=.95, marker="o", markevery=every,
                   markersize=1.8 if local else 1.45, markerfacecolor="white",
                   markeredgewidth=.4, label="True", zorder=3)
    pred, = ax.plot(t, prediction, color=PRED_COLOR, linewidth=width,
                    alpha=.65, marker=".", markevery=every,
                    markersize=2.1 if local else 1.6, label="Prediction", zorder=4)
    return ref, pred


def save(fig, stem, dpi):
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=dpi, bbox_inches=None)
        paths.append(path)
    plt.close(fig)
    return paths


def force_figure(t, truth, prediction, bounds, zoom, stem, dpi, local):
    fig, axes = plt.subplots(4, 1, figsize=(180/25.4, 143/25.4), sharex=True)
    fig.subplots_adjust(left=.112, right=.985, bottom=.085, top=.925, hspace=.22)
    mask = (t >= bounds[0] - 1e-8) & (t <= bounds[1] + 1e-8)
    for col, ax in enumerate(axes):
        if not local:
            ax.axvspan(*zoom, color="#7E8AA0", alpha=.09, linewidth=0)
            for edge in zoom:
                ax.axvline(edge, color="#7E8AA0", linestyle=":", linewidth=.6)
        handles = draw_pair(ax, t[mask], truth[mask, col], prediction[mask, col], local=local)
        axis_style(ax, FORCE_LABELS[col], bounds)
        ax.text(-.083, 1.02, f"({chr(97+col)})", transform=ax.transAxes, fontsize=9, va="top")
    axes[-1].set_xlabel("Time (s)", labelpad=4)
    axes[-1].xaxis.set_major_locator(MultipleLocator(.5 if local else 4))
    fig.align_ylabels(axes)
    fig.legend(handles, ["True", "Prediction"], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.53, .992), handlelength=2.9, columnspacing=2.5)
    return save(fig, stem, dpi)


def velocity_figure(trials, stem, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(180/25.4, 65/25.4))
    fig.subplots_adjust(left=.077, right=.985, bottom=.22, top=.87, wspace=.38)
    for col, (axis, ax) in enumerate(zip(AXES, axes)):
        data, meta, _ = trials[axis]
        # At recorded time t[k], the head predicts the state at t[k+1].
        # The last forecast has no recorded target and is omitted, not padded.
        handles = draw_pair(ax, data["t"][1:], data["cart_velocity"][1:, col],
                            data["pred_cart_velocity"][:-1, col], velocity=True)
        axis_style(ax, VELOCITY_LABELS[col], (0, float(meta["duration_s"])))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel("Time (s)", labelpad=4)
        ax.text(0, 1.035, f"({chr(97+col)})", transform=ax.transAxes, fontsize=9)
    fig.legend(handles, ["True", "Prediction"], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.53, 1.01), handlelength=2.9, columnspacing=2.5)
    return save(fig, stem, dpi)


def metrics(reference, prediction):
    error = prediction - reference
    return {"samples": len(error), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(error.mean())}


def write_csv(path, names, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        # Promote float32 before formatting, so parsing as float64 reproduces
        # the exact recorded value rather than numpy's shortened display text.
        writer.writerows(zip(*(np.asarray(column, dtype=np.float64) for column in columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "logs/Picture/force_velocity_predict/isaac")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/Picture/force_velocity_predict")
    parser.add_argument("--force-axis", choices=AXES, default="vx")
    parser.add_argument("--zoom-start", type=float, default=6.)
    parser.add_argument("--zoom-duration", type=float, default=4.)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    trials = {axis: load_trial(source, axis) for axis in AXES}
    if len({meta["checkpoint_sha256"] for _, meta, _ in trials.values()}) != 1:
        raise ValueError("Trials must use the same checkpoint")
    if len({meta["payload_kg"] for _, meta, _ in trials.values()}) != 1:
        raise ValueError("Trials must use the same payload")
    if len({meta["engine"] for _, meta, _ in trials.values()}) != 1:
        raise ValueError("All figures must use the same simulation engine")
    data, meta, _ = trials[args.force_axis]
    engine = meta["engine"]
    isaac = engine == "Isaac Sim / PhysX"
    zoom = (args.zoom_start, args.zoom_start + args.zoom_duration)
    if not 0 <= zoom[0] < zoom[1] <= data["t"][-1]:
        raise ValueError("Zoom must be a positive interval inside the recorded trajectory")
    previous = output / "plots_manifest.json"
    if previous.is_file():
        archive = output / "archive" / ("plots_" + sha(previous)[:16])
        if not archive.exists():
            archive.mkdir(parents=True)
            for name in ("figures", "derived"):
                if (output / name).is_dir():
                    shutil.copytree(output / name, archive / name)
            for name in ("README.md", "plots_manifest.json"):
                if (output / name).is_file():
                    shutil.copy2(output / name, archive / name)
    for name in ("figures", "derived"):
        (output / name).mkdir(parents=True, exist_ok=True)
    truth, prediction = force_pair(data)
    zoom_mask = (data["t"] >= zoom[0] - 1e-8) & (data["t"] <= zoom[1] + 1e-8)
    style()
    files = force_figure(data["t"], truth, prediction, (0., data["t"][-1]), zoom,
                         output / "figures/force_global", args.dpi, False)
    files += force_figure(data["t"], truth, prediction, zoom, zoom,
                          output / "figures/force_local_4s", args.dpi, True)
    files += velocity_figure(trials, output / "figures/velocity_estimation", args.dpi)
    names = ["time_s"] + [f"{channel}_{kind}_N" for channel in FORCES for kind in ("true", "prediction")]
    columns = [data["t"]] + [value[:, col] for col in range(4) for value in (truth, prediction)]
    write_csv(output / "derived/force_global.csv", names, columns)
    write_csv(output / "derived/force_local_4s.csv", names, [value[zoom_mask] for value in columns])
    report = {"force_global": {}, "force_local": {}, "velocity": {}}
    for col, channel in enumerate(FORCES):
        report["force_global"][channel] = metrics(truth[:, col], prediction[:, col])
        report["force_local"][channel] = metrics(truth[zoom_mask, col], prediction[zoom_mask, col])
    for col, axis in enumerate(AXES):
        trial, _, _ = trials[axis]
        write_csv(output / f"derived/velocity_{axis}.csv",
                  ["target_time_s", "prediction_origin_time_s", "true", "prediction"],
                  [trial["t"][1:], trial["t"][:-1], trial["cart_velocity"][1:, col],
                   trial["pred_cart_velocity"][:-1, col]])
        report["velocity"][axis] = metrics(trial["cart_velocity"][1:, col], trial["pred_cart_velocity"][:-1, col])
    (output / "derived/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    reference_description = ("Isaac dynamic hand-subtree Newton-Euler label, physical units, cart-heading axes, palm origins"
                             if isaac else "MuJoCo weld hand-on-cart force, rotated from world to cart-heading axes")
    manifest = {"engine": engine, "checkpoint": meta["checkpoint"],
                "checkpoint_sha256": meta["checkpoint_sha256"], "payload_kg": meta["payload_kg"],
                "sampling_hz": 50, "force_trial_axis": args.force_axis, "force_zoom_s": zoom,
                "force_zoom_selection": "Fixed user-scale window; not selected by prediction error",
                "force_zoom_samples": int(zoom_mask.sum()), "velocity_horizon_s": .02,
                "force_time_alignment": "Same sample", "velocity_time_alignment": "prediction[k] versus truth[k+1]",
                "force_reference": reference_description,
                "force_prediction": "Frozen V6 head; consumes recorded observable histories and its own predicted cart velocity",
                "data_processing": ("Declared one-step velocity prediction alignment; force physical-unit conversion; no smoothing or resampling"
                                    if isaac else "Coordinate rotation and declared one-step prediction alignment; no smoothing or resampling"),
                "line_samples": "All recorded samples; no decimation", "marker_stride_global": 25,
                "marker_stride_local": 1, "prediction_alpha": .65,
                "sources": {axis: {"path": str(path), "sha256": sha(path),
                                   "metadata_sha256": sha(path.with_suffix('.json'))} for axis, (_, _, path) in trials.items()},
                "script_sha256": sha(__file__),
                "figures": {str(p.relative_to(output)): sha(p) for p in files}}
    (output / "plots_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = f"""# V6 force and velocity estimation figures

Simulation engine: **{engine}**. All three figures use this same engine.

Three figures, each exported as {args.dpi} dpi PNG, vector PDF and editable SVG:

- `figures/force_global`: four wide time-series panels (left/right Fx/Fy), 0-{data['t'][-1]:g} s.
- `figures/force_local_4s`: the same four channels, {zoom[0]:g}-{zoom[1]:g} s ({int(zoom_mask.sum())} samples including both endpoints).
- `figures/velocity_estimation`: three independent longitudinal, lateral and yaw trials, showing measured and predicted cart velocity.

Blue dashed = True (simulation reference). Translucent red = Prediction (alpha 0.65).
Hollow reference markers and red dots mark every local 50 Hz sample; global markers
occur every 25 samples (0.5 s) while the lines retain ALL samples. The pale gray
window in the global force figure identifies the local view. No smoothing,
downsampling, interpolation, fitted lag, or display-only initialization is used.

Source: `{source}`; V6 model_{meta['iteration']}, {meta['payload_kg']:g} kg payload, completed {engine} trials.
The force plot uses the independent {args.force_axis} trial. The local window is fixed at
{zoom[0]:g}-{zoom[1]:g} s, not chosen to minimize error. Force reference:
{reference_description}.
{'The reference is the exact dynamic label definition used during V6 training. The model is frozen, in eval mode; no fitting or training is done on the plotted traces. Both true forces and predictions come from the same 50 Hz state. Actor605 and estimator462 inputs are saved for offline inference replay.' if isaac else 'V6 was trained against Isaac dynamic hand labels; this is a cross-simulator comparison. Only forces are compared, not incompatible COM/palm moments.'}
Here 'True' means simulation reference, not a physical force-sensor measurement.

Velocity timing follows the saved V6 training contract: a prediction produced at
t is compared with the measured velocity at t+0.02 s. The CSV includes both
timestamps. The final forecast has no recorded target and is omitted. Hand
predictions are compared at the same timestamp. Time is actual simulation time;
there is no display-only initialization hold or artificial shift.

`derived/` contains every plotted pair and raw-sample RMSE/MAE/bias. All source
records remain unchanged; source/output hashes are in `plots_manifest.json`.
Every measured prediction error is retained. No samples or channels were selected
for better agreement. Previous figure versions are preserved in `archive/`.

Reproduce:

```powershell
E:\\envs\\RLisaac601\\python.exe scripts/collect_estimator_predictions_isaac.py
E:\\envs\\RLisaac601\\python.exe scripts/plot_force_velocity_predict.py
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": len(files), "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
