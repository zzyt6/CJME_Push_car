"""Render paper figures from six separately collected simulation motion trials.

No simulation, interpolation, resampling, or source-data modification is performed.
The band is the local mean +/- population standard deviation, not a confidence interval.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AXES = {"vx": (0, r"$v_x$ (m/s)"), "vy": (1, r"$v_y$ (m/s)"), "wz": (2, r"$\omega_z$ (rad/s)")}
METHODS = ("CWC", "baseline")
COLORS = {"CWC": "#0072B2", "baseline": "#D55E00"}
PLOT_VERSION = 2
PANEL_WIDTH_MM = 60.0
FIGURE_HEIGHT_MM = 58.0
# Each panel has the same physical axes rectangle, including standalone exports.
AXES_RECT_MM = (12.0, 14.0, 45.0, 38.0)  # left, bottom, width, height


@dataclass
class Trial:
    axis: str
    method: str
    metadata: dict
    t: np.ndarray
    interval_start: np.ndarray
    command: np.ndarray
    reference: np.ndarray
    velocity: np.ndarray
    terminated: np.ndarray
    stages: list[dict]
    stage_index: np.ndarray
    source_paths: tuple[Path, ...]
    initial_t: float | None = None
    initial_velocity: np.ndarray | None = None

    @property
    def signal(self) -> np.ndarray:
        return self.velocity[:, AXES[self.axis][0]]

    @property
    def completed(self) -> bool:
        return bool(self.metadata["completed"])


def canonical_stages(stages: list[dict]) -> list[dict]:
    if not stages:
        raise ValueError("A trial must contain at least one stage")
    result = []
    previous_end = 0.0
    for stage in stages:
        start, end, value = (float(stage[key]) for key in ("start_s", "end_s", "value"))
        if not np.isfinite([start, end, value]).all() or end <= start:
            raise ValueError("Stage boundaries/values must be finite and increasing")
        if not np.isclose(start, previous_end, atol=1e-8, rtol=0):
            raise ValueError("Stages must be contiguous and start at time zero")
        result.append({"start_s": start, "end_s": end, "value": value})
        previous_end = end
    return result


def load_trial(root: Path, axis: str, method: str) -> Trial:
    npz_path = root / "raw" / f"{axis}_{method}.npz"
    json_path = npz_path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata["axis"] != axis or metadata["method"] != method:
        raise ValueError(f"{json_path}: axis/method do not match the filename")
    if not isinstance(metadata["completed"], bool):
        raise ValueError(f"{json_path}: completed must be a boolean")
    stages = canonical_stages(metadata["stages"])
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in
                  ("t", "interval_start", "command", "reference", "cart_velocity", "terminated")}
    t, starts = arrays["t"], arrays["interval_start"]
    if t.ndim != 1 or not len(t) or starts.shape != t.shape:
        raise ValueError(f"{npz_path}: expected non-empty t[T] and interval_start[T]")
    n = len(t)
    for key in ("command", "reference", "cart_velocity"):
        if arrays[key].shape != (n, 3) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{npz_path}: {key} must be finite [T, 3]")
    term = arrays["terminated"]
    if term.shape != (n,) or not np.isin(term, [0, 1]).all():
        raise ValueError(f"{npz_path}: terminated must be boolean [T]")
    term = term.astype(bool)
    if np.any(term[:-1]):
        raise ValueError(f"{npz_path}: data continue after termination; reset trajectories cannot be spliced")
    dt = float(metadata["dt"])
    if not np.isfinite([*t, *starts, dt]).all() or dt <= 0:
        raise ValueError(f"{npz_path}: invalid timing")
    if (not np.isclose(starts[0], 0, atol=1e-7, rtol=0)
            or not np.allclose(t - starts, dt, atol=2e-6, rtol=0)
            or not np.allclose(starts[1:], t[:-1], atol=2e-6, rtol=0)):
        raise ValueError(f"{npz_path}: samples must be contiguous post-physics intervals")
    duration = stages[-1]["end_s"]
    if t[-1] > duration + 2e-6:
        raise ValueError(f"{npz_path}: samples exceed the protocol duration")
    if metadata["completed"] and (term.any() or not np.isclose(t[-1], duration, atol=2e-6, rtol=0)):
        raise ValueError(f"{npz_path}: completed trial must reach its planned end without termination")
    stage_index = np.searchsorted([stage["end_s"] for stage in stages], starts + 1e-7, side="right")
    if np.any(stage_index >= len(stages)):
        raise ValueError(f"{npz_path}: invalid stage assignment")
    desired = np.zeros((n, 3), dtype=np.float64)
    desired[:, AXES[axis][0]] = np.array([stage["value"] for stage in stages])[stage_index]
    if not np.allclose(arrays["command"], desired, atol=1e-6, rtol=0):
        raise ValueError(f"{npz_path}: recorded applied commands differ from the staged protocol")
    source_paths = (npz_path, json_path)
    initial_t, initial_velocity = None, None
    if metadata.get("initial_state_file"):
        initial_path = (root / metadata["initial_state_file"]).resolve()
        if not initial_path.is_relative_to(root.resolve()):
            raise ValueError(f"{json_path}: initial_state_file must remain inside the trial root")
        if sha256(initial_path) != metadata.get("initial_state_sha256"):
            raise ValueError(f"{json_path}: initial state hash does not match")
        with np.load(initial_path, allow_pickle=False) as initial:
            initial_time = np.asarray(initial["t"])
            initial_velocity = np.asarray(initial["cart_velocity"], dtype=np.float64)
        if initial_time.size != 1 or initial_velocity.shape != (3,) or not np.isfinite(initial_velocity).all():
            raise ValueError(f"{initial_path}: expected t scalar and finite cart_velocity[3]")
        initial_t = float(initial_time.item())
        if not np.isfinite(initial_t) or not np.isclose(initial_t, 0.0, atol=1e-12, rtol=0):
            raise ValueError(f"{initial_path}: reset initial state must have t=0")
        source_paths += (initial_path,)
    elif metadata.get("initial_state_sha256"):
        raise ValueError(f"{json_path}: initial state hash provided without initial_state_file")
    return Trial(axis, method, metadata, t, starts, arrays["command"], arrays["reference"],
                 arrays["cart_velocity"], term, stages, stage_index, source_paths, initial_t, initial_velocity)


def validate_pair(cwc: Trial, baseline: Trial) -> None:
    if cwc.axis != baseline.axis or cwc.stages != baseline.stages:
        raise ValueError("Compared methods must have exactly the same axis and command stages")
    if not np.isclose(float(cwc.metadata["dt"]), float(baseline.metadata["dt"]), atol=1e-12, rtol=0):
        raise ValueError("Compared methods must have the same sampling period")
    # Allow a method to terminate early, but every shared physical interval must agree.
    n = min(len(cwc.t), len(baseline.t))
    if not np.allclose(cwc.t[:n], baseline.t[:n], atol=2e-6, rtol=0):
        raise ValueError("Compared methods have inconsistent post-physics timestamps")
    if (cwc.initial_t is None) != (baseline.initial_t is None):
        raise ValueError("Compared methods must both provide a reset initial state or neither")
    for key in ("payload_kg", "payload", "seed", "level"):
        if key in cwc.metadata and key in baseline.metadata and cwc.metadata[key] != baseline.metadata[key]:
            raise ValueError(f"Compared methods have different {key}")


def local_display(trial: Trial, window_s: float) -> dict[str, np.ndarray]:
    """Centered time-window statistics, truncated at each command-stage boundary."""
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    mean, low, high, std = (np.empty_like(trial.signal, dtype=np.float64) for _ in range(4))
    count = np.empty(len(trial.t), dtype=np.int64)
    for stage in range(len(trial.stages)):
        positions = np.flatnonzero(trial.stage_index == stage)
        if not len(positions):
            continue
        times, signal = trial.t[positions], trial.signal[positions]
        left = np.searchsorted(times, times - window_s / 2 - 1e-8, side="left")
        right = np.searchsorted(times, times + window_s / 2 + 1e-8, side="right")
        for j, pos in enumerate(positions):
            local = signal[left[j]:right[j]]
            mean[pos], low[pos], high[pos] = local.mean(), local.min(), local.max()
            std[pos] = local.std(ddof=0)
            count[pos] = len(local)
    return {"mean": mean, "min": low, "max": high, "std": std,
            "lower": mean - std, "upper": mean + std, "n": count}


def stage_metrics(trial: Trial, transient_s: float = 2.0) -> list[dict]:
    """Physical-unit metrics from raw samples; no display filter enters this path."""
    if transient_s < 0:
        raise ValueError("transient_s must be nonnegative")
    rows = []
    for i, stage in enumerate(trial.stages):
        mask = ((trial.stage_index == i) & (trial.interval_start >= stage["start_s"] + transient_s - 1e-7))
        signal = trial.signal[mask]
        row = {"axis": trial.axis, "method": trial.method, "stage": i,
               "start_s": stage["start_s"], "end_s": stage["end_s"],
               "desired": stage["value"], "discard_after_change_s": transient_s,
               "samples": int(mask.sum()),
               "stage_completed": bool(trial.t[-1] >= stage["end_s"] - 2e-6 and not
                                       (trial.terminated[-1] and trial.t[-1] <= stage["end_s"] + 2e-6)),
               "unit": "rad/s" if trial.axis == "wz" else "m/s"}
        if len(signal):
            error = signal - stage["value"]
            row.update({"sample_first_end_s": float(trial.t[mask][0]),
                        "sample_last_end_s": float(trial.t[mask][-1]),
                        "raw_mean": float(signal.mean()), "raw_bias": float(error.mean()),
                        "raw_rmse": float(np.sqrt(np.mean(error ** 2))),
                        "raw_mae": float(np.mean(np.abs(error))),
                        "raw_ripple_rms_demeaned": float(np.std(signal, ddof=0)),
                        "raw_peak_to_peak": float(np.ptp(signal)),
                        "raw_min": float(signal.min()), "raw_max": float(signal.max())})
        else:
            row.update({key: None for key in ("sample_first_end_s", "sample_last_end_s", "raw_mean",
                                              "raw_bias", "raw_rmse", "raw_mae", "raw_ripple_rms_demeaned",
                                              "raw_peak_to_peak", "raw_min", "raw_max")})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax, axis: str, duration: float) -> None:
    ax.set_xlabel("Time (s)", labelpad=3)
    ax.set_ylabel(AXES[axis][1], labelpad=1)
    ax.set_xlim(0, duration)
    ax.set_xticks(np.arange(0, duration + 1e-7, 5.0))
    ax.grid(True, color="#DEDEDE", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(width=0.65, length=2.5, pad=2, color="#555555")
    ax.margins(y=0.08)


def plot_panel(ax, axis: str, trials: dict, display: dict, *, raw: bool, show_legend: bool = True,
               initialization_hold_s: float = 0.0) -> list:
    pair = [trials[(axis, method)] for method in METHODS]
    stages = pair[0].stages
    # A genuine command step at its applied start; actual samples occur at interval ends.
    edges = [stage["start_s"] + initialization_hold_s for stage in stages]
    edges += [stages[-1]["end_s"] + initialization_hold_s]
    values = [stage["value"] for stage in stages]
    if initialization_hold_s > 0:
        edges, values = [0.0] + edges, [0.0] + values
    desired = ax.stairs(values, edges, baseline=None, color="#222222", linestyle="--",
                        linewidth=1.1, label="vel cmd", zorder=6)
    if initialization_hold_s == 0 and stages[0]["value"] != 0 and all(trial.initial_t is not None for trial in pair):
        # The measured reset point precedes the first command at the same t=0;
        # no zero-duration interval or invented zero-command plateau is added.
        ax.plot([0.0, 0.0], [0.0, stages[0]["value"]], color="#222222", linestyle="--",
                linewidth=1.1, zorder=6)
    handles = [desired]
    for trial in pair:
        values = trial.signal if raw else display[(axis, trial.method)]["mean"]
        plot_t = trial.t + initialization_hold_s
        initial_value = None if trial.initial_velocity is None else float(trial.initial_velocity[AXES[axis][0]])
        if initial_value is not None:
            initial_plot_times = [0.0, initialization_hold_s] if initialization_hold_s > 0 else [trial.initial_t]
            plot_t = np.r_[initial_plot_times, plot_t]
            initial_plot_values = [initial_value] * len(initial_plot_times)
            values = np.r_[initial_plot_values, values]
        color = COLORS[trial.method]
        if not raw:
            summary = display[(axis, trial.method)]
            lower, upper = summary["lower"], summary["upper"]
            if initial_value is not None:
                lower, upper = np.r_[initial_plot_values, lower], np.r_[initial_plot_values, upper]
            ax.fill_between(plot_t, lower, upper, color=color,
                            alpha=0.18, linewidth=0, zorder=2)
        line, = ax.plot(plot_t, values, color=color, linewidth=0.45 if raw else 1.2,
                        label=trial.method, zorder=4 if trial.method == "CWC" else 3)
        handles.append(line)
        if not trial.completed:
            stop_time = trial.t[-1] + initialization_hold_s
            ax.axvline(stop_time, color=color, linewidth=0.7, linestyle=":", alpha=0.8)
            ax.plot(stop_time, values[-1], marker="x", markersize=5, color=color, zorder=7)
            ax.annotate(f"{trial.method} stopped", (stop_time, values[-1]),
                        xytext=(-4, 10 if trial.method == "CWC" else -14),
                        textcoords="offset points", ha="right", fontsize=7, color=color)
    style_axis(ax, axis, stages[-1]["end_s"] + initialization_hold_s)
    # Explicit bounds include all recorded extrema, even in the presentation variant.
    limit_parts = [trial.signal for trial in pair] + [np.asarray([stage["value"] for stage in stages])]
    limit_parts += [np.asarray([0.0, trial.initial_velocity[AXES[axis][0]]])
                    for trial in pair if trial.initial_velocity is not None]
    if not raw:
        limit_parts += [display[(axis, trial.method)][bound] for trial in pair for bound in ("lower", "upper")]
    limits = np.concatenate(limit_parts)
    lower, upper = float(limits.min()), float(limits.max())
    padding = max(0.01, (upper - lower) * 0.08)
    ax.set_ylim(lower - padding, upper + padding)
    if show_legend:
        panel_legend(ax, handles)
    return handles


def panel_figure(count: int):
    """Create one or three equal physical panels, with no content-dependent layout."""
    if count not in (1, 3):
        raise ValueError("Use either a standalone panel or a three-panel horizontal figure")
    width_mm = count * PANEL_WIDTH_MM
    fig = plt.figure(figsize=(width_mm / 25.4, FIGURE_HEIGHT_MM / 25.4))
    left, bottom, width, height = AXES_RECT_MM
    axes = [fig.add_axes([(i * PANEL_WIDTH_MM + left) / width_mm,
                         bottom / FIGURE_HEIGHT_MM, width / width_mm,
                         height / FIGURE_HEIGHT_MM]) for i in range(count)]
    return fig, axes


def panel_legend(ax, handles: list) -> None:
    ax.legend(handles=handles, labels=["vel cmd", "CWC", "baseline"], loc="upper left",
              frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", ncol=1,
              borderaxespad=0.35, borderpad=0.25, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.25, fontsize=7)


def save_figure(fig, stem: Path, dpi: int) -> list[Path]:
    paths = []
    for extension in ("png", "pdf", "svg"):
        target = stem.with_suffix(f".{extension}")
        kwargs = {"dpi": dpi} if extension == "png" else {}
        # Keep the exact physical canvas dimensions. Tight bounding boxes make
        # different axis tick labels produce different standalone image widths.
        fig.savefig(target, bbox_inches=None, facecolor="white", **kwargs)
        paths.append(target)
    plt.close(fig)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_existing_artifacts(root: Path) -> Path | None:
    """Snapshot the previous publication once per provenance manifest before replacement."""
    previous_manifest = root / "plots_manifest.json"
    if not previous_manifest.is_file():
        return None
    previous_hash = sha256(previous_manifest)
    archive = root / "archive" / f"plots_{previous_hash[:16]}"
    saved_manifest = archive / "plots_manifest.json"
    if archive.exists():
        if saved_manifest.is_file() and sha256(saved_manifest) == previous_hash:
            return archive
        raise RuntimeError(f"Refusing to overwrite an incomplete/conflicting plot archive: {archive}")
    archive.mkdir(parents=True, exist_ok=False)
    # Retain the old plotting code snapshot where the collector stored one.
    for name in ("figures", "derived", "provenance"):
        source = root / name
        if source.is_dir():
            shutil.copytree(source, archive / name)
    for name in ("FIGURES_README.md", "plots_manifest.json"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    return archive


def render(root: Path, window_s: float = 0.5, transient_s: float = 2.0, dpi: int = 600,
           initialization_hold_s: float = 0.0) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = manifest.get("engine")
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("The collection manifest must identify its simulation engine")
    engine = engine.strip()
    payload_kg = float(manifest["payload_kg"])
    if not np.isfinite(payload_kg) or payload_kg < 0:
        raise ValueError("The collection manifest must specify a nonnegative payload_kg")
    trials = {(axis, method): load_trial(root, axis, method) for axis in AXES for method in METHODS}
    for axis in AXES:
        validate_pair(trials[(axis, "CWC")], trials[(axis, "baseline")])
    if any(not np.isclose(float(trial.metadata["payload_kg"]), payload_kg, atol=1e-8, rtol=0)
           for trial in trials.values()):
        raise ValueError("The collection manifest and raw trials have different payload_kg")
    if not np.isfinite(initialization_hold_s) or initialization_hold_s < 0:
        raise ValueError("initialization_hold_s must be finite and nonnegative")
    if initialization_hold_s > 0 and any(trial.initial_t is None for trial in trials.values()):
        raise ValueError("A display-only initialization hold requires a verified reset record for every trial")
    display = {key: local_display(trial, window_s) for key, trial in trials.items()}
    previous_archive = archive_existing_artifacts(root)
    figures_dir, derived_dir = root / "figures", root / "derived"
    figures_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metric_rows = []
    for key, trial in trials.items():
        component = AXES[trial.axis][0]
        summary = display[key]
        rows = [{"time_s": float(trial.t[i]), "simulation_time_s": float(trial.t[i]),
                 "plot_time_s": float(trial.t[i] + initialization_hold_s),
                 "interval_start_s": float(trial.interval_start[i]),
                 "sample_kind": "control_end",
                 "stage": int(trial.stage_index[i]), "desired_raw": float(trial.command[i, component]),
                 "internal_reference": float(trial.reference[i, component]), "actual_raw": float(trial.signal[i]),
                 "display_mean": float(summary["mean"][i]), "display_min": float(summary["min"][i]),
                 "display_max": float(summary["max"][i]), "display_std": float(summary["std"][i]),
                 "display_lower": float(summary["lower"][i]), "display_upper": float(summary["upper"][i]),
                 "window_samples": int(summary["n"][i]),
                 "terminated": bool(trial.terminated[i])} for i in range(len(trial.t))]
        if trial.initial_velocity is not None:
            initial_value = float(trial.initial_velocity[component])
            # Display-only endpoint, never included in time-window statistics or
            # physical stage metrics. Blank interval/reference fields are intentional.
            initial_plot_times = [0.0, initialization_hold_s] if initialization_hold_s > 0 else [trial.initial_t]
            initial_rows = [{"time_s": trial.initial_t, "simulation_time_s": trial.initial_t,
                             "plot_time_s": plot_time, "interval_start_s": None,
                             "sample_kind": "initialization_hold" if initialization_hold_s > 0 else "reset_initial",
                             "stage": None, "desired_raw": 0.0, "internal_reference": None,
                             "actual_raw": initial_value, "display_mean": initial_value,
                             "display_min": initial_value, "display_max": initial_value, "display_std": 0.0,
                             "display_lower": initial_value, "display_upper": initial_value,
                             "window_samples": 0, "terminated": None} for plot_time in initial_plot_times]
            rows = initial_rows + rows
        csv_path = derived_dir / f"display_{trial.axis}_{trial.method}.csv"
        write_csv(csv_path, rows)
        outputs.append(csv_path)
        metric_rows.extend(stage_metrics(trial, transient_s))
    metrics_path = derived_dir / "stage_metrics.csv"
    write_csv(metrics_path, metric_rows)
    outputs.append(metrics_path)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for raw in (False, True):
            suffix = "_raw" if raw else ""
            for axis in AXES:
                fig, (ax,) = panel_figure(1)
                plot_panel(ax, axis, trials, display, raw=raw, show_legend=axis == "vx",
                           initialization_hold_s=initialization_hold_s)
                outputs.extend(save_figure(fig, figures_dir / f"{axis}{suffix}", dpi))
            fig, axes = panel_figure(3)
            for i, (ax, axis) in enumerate(zip(axes, AXES)):
                plot_panel(ax, axis, trials, display, raw=raw, show_legend=i == 0,
                           initialization_hold_s=initialization_hold_s)
            outputs.extend(save_figure(fig, figures_dir / f"motion_ability{suffix}", dpi))
    protocol_table = ("| Axis | Raw command plateaus | Simulation stage ends (s) | Display stage ends (s) |\n"
                      "|---|---|---|---|\n")
    for axis in AXES:
        stages = trials[(axis, "CWC")].stages
        protocol_table += (f"| {axis} | " + " → ".join(f"{stage['value']:g}" for stage in stages)
                           + " | " + ", ".join(f"{stage['end_s']:g}" for stage in stages)
                           + " | " + ", ".join(f"{stage['end_s'] + initialization_hold_s:g}" for stage in stages) + " |\n")
    readme_path = root / "FIGURES_README.md"
    initial_state_count = sum(trial.initial_t is not None for trial in trials.values())
    if initialization_hold_s > 0:
        initial_description = (
            f"The **0–{initialization_hold_s:g} s flat segment is a display-only initialization hold** "
            "of each trial's verified, measured reset snapshot. **The simulation is not advanced and "
            "the controller is not run during this displayed interval.** It is not a recorded "
            "stationary-control experiment. The actual rollout starts immediately with the first "
            f"nonzero command at simulation t=0, shown at plot t={initialization_hold_s:g} s. "
            f"Every recorded post-physics sample and command-stage boundary is shifted by +{initialization_hold_s:g} s "
            "on the plot clock; raw timestamps, values, sample counts and metrics are unchanged. "
            "The held reset value is excluded from smoothing and metrics, with a zero-width shaded band. "
            "CSV rows at both hold endpoints have `sample_kind=initialization_hold`, simulation time 0, "
            "`window_samples=0`, and blank interval/stage/reference/termination fields. These two display "
            "endpoints are not extra physical measurements.\n\n")
    else:
        initial_description = (
        "A separate measured **t=0 reset state** is prepended to each actual curve when supplied "
        "by `initial_state_file` and verified by `initial_state_sha256`. It is not a zero-duration "
        "physics interval. For immediate nonzero command protocols, the command line jumps from "
        "zero to the first plateau at t=0; no initial waiting plateau is introduced. The measured "
        "reset value is **excluded** from centered-window means, standard deviations, and all "
        "stage metrics. Its shaded band has zero width. In display CSVs, this row has "
        "`sample_kind=reset_initial`, `window_samples=0`, and blank interval/stage/reference/termination "
        "fields; the display value is exactly the measured value, not a smoothing estimate. "
        f"Reset records are present in {initial_state_count} of the six trials.\n\n"
        if initial_state_count else
        "No separate reset-state records are supplied for this collection. No synthetic sample "
        "is inserted at time zero; the actual curves begin at the first recorded interval end.\n\n")
    initialization_caption = (
        f"The 0–{initialization_hold_s:g} s flat segment holds the measured reset snapshot for display only; "
        "the simulation is not advanced and the controller is not run during that interval. "
        f"The measured rollouts and command schedules are plotted with a +{initialization_hold_s:g} s "
        "time offset. " if initialization_hold_s > 0 else "")
    repo_root = Path(__file__).resolve().parents[1]
    reproduction_root = root.relative_to(repo_root).as_posix() if root.is_relative_to(repo_root) else root.as_posix()
    readme_path.write_text(
        "# Motion ability figures\n\n"
        f"Simulation backend: **{engine}** (read from the collection manifest). "
        f"Each axis and controller is an independent {engine} rollout. The black dashed step is the applied "
        "unfiltered task command. Blue is **CWC** and vermillion is **baseline**. "
        "The actual signal is the raw cart COM velocity in the cart frame; angular velocity is yaw rate. "
        "Control samples are recorded at the end of each physics/control interval.\n\n"
        + initial_description +
        "The combined figure uses a **1 × 3 horizontal layout, 180 × 58 mm**. Only the leftmost "
        "vx panel has a compact three-row legend in its **upper-left corner** (`vel cmd`, `CWC`, "
        "`baseline`); the vy/wz panels and area above the figure have no legend. Standalone figures "
        "are each **60 × 58 mm** and follow the same rule: vx retains the upper-left legend, "
        "while vy/wz omit it so the three files can be placed side by side without duplicate legends. "
        "Every axes rectangle is exactly **45 × 38 mm** (width/height 1.184), regardless of tick-label "
        "length or physical units. Exports keep their fixed canvas dimensions without tight cropping. "
        "The three panels therefore align with three equally spaced illustration columns.\n\n"
        "The yaw-rate plateaus use the user-specified angular-velocity protocol independently of the "
        "longitudinal-speed schedule. Command levels and simulation stage ends below are read from "
        "the saved trial metadata; displayed stage ends include the declared initialization offset. "
        "A training command limit is not a tested command level.\n\n"
        + protocol_table + "\n"
        "`figures/vx`, `vy`, `wz` and `motion_ability` show a centered "
        f"**{window_s:.2f} s** local mean with **mean ± one population standard deviation** shaded "
        "(`ddof=0`), computed from the raw samples in that same window. Every axis and method "
        "uses the identical display window. "
        "Windows are truncated at each command-stage boundary and at the start/end of a trial, "
        "so they do not mix samples from different command levels. The exact sample count is recorded "
        "for every window. The band describes within-rollout temporal variation; "
        "it is **not** a confidence interval or variation across repeated experiments. It does not "
        "bound every raw sample and can extend beyond local observed extrema. This centered display "
        f"filter uses up to {window_s / 2:.2f} s of future samples and is not a controller or estimator "
        "output. It attenuates visible high-frequency motion; response timing, overshoot and ripple "
        "must be measured from raw data, not from this presentation curve.\n\n"
        "The corresponding `_raw` figures show every recorded velocity sample at 50 Hz. "
        "All figures include the full range of both raw signals; spikes are not clipped. "
        "The original `raw/*.npz` and `raw/*.json` are unchanged. Missing post-start trajectories are "
        "never interpolated or extended, and incomplete trials are marked with a stop indicator.\n\n"
        "`derived/display_*.csv` contains the raw command, internal filtered reference, raw velocity, "
        "display mean, population standard deviation, lower/upper shaded bounds, observed local "
        "min/max for traceability, and exact number of samples in every window. `simulation_time_s` "
        "and its compatibility alias `time_s` retain the physical simulation timestamp; `plot_time_s` "
        "is the separate figure clock and identifies any display-only initialization offset. "
        "`derived/stage_metrics.csv` uses **only raw recorded velocity against the raw command**, "
        f"after discarding the first {transient_s:g} s of each plateau. The de-meaned raw RMS is "
        "the population standard deviation in that window, including any remaining slow drift; "
        "it is not a frequency-band-specific ripple metric. Stages with no eligible samples "
        "have blank metrics. An incomplete stage is explicitly flagged.\n\n"
        "## Suggested caption\n\n"
        f"Cart velocity tracking in {engine} with a {payload_kg:g} kg payload under independent longitudinal, lateral "
        "and yaw-rate step commands. CWC and the HOMIE pass-through baseline use identical command "
        "schedules. Dashed black lines denote desired velocities. Solid colored lines show "
        f"{window_s:.2f} s centered local means; shaded bands show ±1 population standard deviation "
        "of raw samples within the same time window, truncated at command changes. This is temporal "
        "variation in one rollout per method, not uncertainty across repeated trials. The accompanying raw plots retain "
        "all 50 Hz samples. " + initialization_caption + "\n\n"
        "A previous figure publication, including its derived data, figure README, plotting manifest "
        "and available provenance snapshots, is retained once under `archive/plots_<manifest-hash>/` "
        "before replacement. Raw simulation data are never moved or changed.\n\n"
        f"Reproduce with `python scripts/plot_motion_ability.py --root \"{reproduction_root}\" "
        f"--initialization-hold-s {initialization_hold_s:g}`. "
        "`plots_manifest.json` records source and output hashes, plotting settings and collection metadata.\n",
        encoding="utf-8")
    outputs.append(readme_path)
    source_paths = [manifest_path] + [path for trial in trials.values() for path in trial.source_paths]
    plot_manifest = {
        "schema_version": PLOT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "collection_manifest_sha256": sha256(manifest_path),
        "collection_metadata": manifest,
        "simulation_engine": engine,
        "layout": {"combined": "1x3", "combined_mm": [180.0, FIGURE_HEIGHT_MM],
                   "standalone_mm": [PANEL_WIDTH_MM, FIGURE_HEIGHT_MM],
                   "axes_rectangle_mm": list(AXES_RECT_MM), "tight_crop": False,
                   "legend": ["vel cmd", "CWC", "baseline"],
                   "legend_location": "upper left in vx panel only; combined and standalone",
                   "combined_legend_panels": ["vx"], "standalone_legend_panels": ["vx"], "legend_columns": 1,
                   "shared_legend": False},
        "previous_publication_archive": str(previous_archive.relative_to(root)) if previous_archive else None,
        "display": {"window_s": window_s, "window_alignment": "centered; inclusive endpoints",
                    "stage_boundary_handling": "truncate; never mix command plateaus",
                    "line": "local arithmetic mean", "band": "local mean +/- 1 population standard deviation",
                    "standard_deviation_ddof": 0, "band_is_confidence_interval": False,
                    "same_window_all_axes_and_methods": True, "local_observed_extrema_in_csv": True,
                    "reset_initial_records": initial_state_count,
                    "reset_initial_in_smoothing_or_metrics": False,
                    "reset_initial_band_width": 0,
                    "initialization_hold_s": initialization_hold_s,
                    "initialization_hold_is_display_only": True,
                    "simulation_advanced_during_initialization_hold": False,
                    "controller_run_during_initialization_hold": False,
                    "recorded_sample_plot_time": f"simulation_time_s + {initialization_hold_s:g}",
                    "raw_figures": True, "resampling": False, "dpi_png": dpi},
        "metrics": {"source": "raw cart_velocity versus raw command", "discard_after_stage_change_s": transient_s,
                    "initialization_hold_in_metrics": False,
                    "ripple": "demeaned raw RMS over each eligible stage (no frequency filtering)"},
        "trial_status": [{"axis": trial.axis, "method": trial.method, "completed": trial.completed,
                          "samples": len(trial.t), "last_time_s": float(trial.t[-1]),
                          "last_plot_time_s": float(trial.t[-1] + initialization_hold_s)} for trial in trials.values()],
        "sources": [{"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in source_paths],
        "outputs": [{"path": str(path.relative_to(root)), "sha256": sha256(path)} for path in outputs],
    }
    (root / "plots_manifest.json").write_text(json.dumps(plot_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plot_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/Picture/motion_ability"))
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--transient-s", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--initialization-hold-s", type=float, default=0.0,
                        help="Display-only hold of verified reset states before the measured rollout; raw times and metrics stay unchanged")
    args = parser.parse_args()
    result = render(args.root, args.window_s, args.transient_s, args.dpi, args.initialization_hold_s)
    print(f"Wrote {len(result['outputs'])} figure/data artifacts and plots_manifest.json to {args.root.resolve()}")


if __name__ == "__main__":
    main()
