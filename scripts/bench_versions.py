"""Cross-version MuJoCo benchmark (V3 ... V5.2) with interaction, balance and force-estimation metrics.

One worker process per policy (``--worker``); the parent launches them in parallel and then summarizes.
Every policy uses the same MuJoCo scene, payload model, HOMIE model_60000, follower gains and seeds.

    python scripts/bench_versions.py --out logs/bench/versions_20k            # run all + summarize
    python scripts/bench_versions.py --out logs/bench/versions_20k --summarize_only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL" / "rsl_rl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUNS = ROOT / "logs/rsl_rl"
VERSIONS = {
    "V3": ("v4pkg", RUNS / "homie_pushcart/2026-09-18_06-37-12_v3e_full/model_12600.pt"),
    "V3.1": ("v4pkg", RUNS / "homie_pushcart/2026-09-20_21-22-19_v3_1_combined_scratch/model_19999.pt"),
    "V3.2": ("v4pkg", RUNS / "homie_pushcart/2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt"),
    "V4": ("v4pkg", RUNS / "homie_pushcart/2026-09-22_18-22-27_v4_speed_quality_scratch/model_19999.pt"),
    "V5.1": ("homie_pushcart_v51", RUNS / "homie_pushcart_v51/2026-09-25_04-06-26_v51_interaction_privileged_20k/model_14600.pt"),
    "V5.2": ("homie_pushcart_v52", RUNS / "homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt"),
}
ORDER = ["passthrough", "V3", "V3.1", "V3.2", "V4", "V5.1", "V5.2"]
PAYLOADS = [0.0, 12.5, 25.0]
PATHS = ["s_large", "s_large_mirror", "s_curve"]
STEPS = ["vx_0_to_03", "vx_0_to_06", "stop_06", "wz_p03", "wz_m03", "hold"]
# Known cart-side pulses (cart-heading axes): name, (Fx, Fy, Mz), duration s, response axis index.
PULSES = [("+Fx25", (25., 0., 0.), .2, 0), ("-Fx25", (-25., 0., 0.), .2, 0), ("+Fy25", (0., 25., 0.), .2, 1),
          ("+Mz3", (0., 0., 3.), .2, 2), ("-Fx20_hold3s", (-20., 0., 0.), 3., 0)]
PULSE_PAYLOAD = 12.5
PULSE_CMDS = {"move03": (0.3, 0., 0.), "stand": (0., 0., 0.)}


# ============================================================================ worker
def make(label: str):
    from pushcart_bench.policies_v5x import V5xPolicy, install_instrumented_env
    from pushcart_bench.extra_metrics import Recorder
    install_instrumented_env()
    from pushcart_bench.policies import PassthroughPolicy, TwoStagePolicy
    if label == "passthrough":
        policy = PassthroughPolicy(17)
    else:
        kind, ckpt = VERSIONS[label]
        policy = (TwoStagePolicy(ckpt, 17, name=label) if kind == "v4pkg"
                  else V5xPolicy(kind, ckpt, 17, name=label, scene_profile="legacy"))
    return policy, Recorder(policy)


def worker(label: str, out: Path, trace_stride: int, quick: bool = False):
    import numpy as np
    from pushcart_bench.config_io import load_config
    from pushcart_bench.follower import FollowerConfig
    from pushcart_bench.paths import build_path
    from pushcart_bench.runner import Condition, run_path, run_step
    from pushcart_bench import extra_metrics as xm

    global PAYLOADS, PATHS, STEPS, PULSES, PULSE_CMDS
    if quick:
        PAYLOADS, PATHS, STEPS = [12.5], ["s_large"], ["vx_0_to_03", "stop_06"]
        PULSES, PULSE_CMDS = PULSES[:1] + PULSES[-1:], {"move03": PULSE_CMDS["move03"]}
    cfg = load_config(None)
    cfg["trace_stride"] = trace_stride
    policy, rec = make(label)
    dt = policy.dt
    env = policy.env
    robot_mass = env.robot_mass
    results = {"label": label, "describe": policy.describe(), "paths": [], "steps": [], "pulses": []}
    est_natural = []
    t0 = time.perf_counter()

    fcfg = FollowerConfig.from_dict(cfg["follower"])
    for payload in PAYLOADS:
        for pname in PATHS:
            base = pname.replace("_mirror", "")
            path = build_path(base, cfg["paths"][base], mirror=pname.endswith("_mirror"))
            res = run_path(policy, path, fcfg, Condition(payload_kg=payload, seed=0), cfg, label=pname)
            a = rec.arrays()
            T = len(a["yaw"])
            skip = int(2.0 / dt)
            extra = xm.window_metrics(a, skip, T, dt, policy.cart_mass_total(), robot_mass)
            est_natural.append(xm.estimator_series(a, skip, T))
            results["paths"].append({"path": pname, "payload_kg": payload, "metrics": res["metrics"], "extra": extra,
                                     "trace": res["trace"]})
            m = res["metrics"]
            print(f"[{label}] path {pname:15s} {payload:4.1f}kg ok={m.get('success')} ttg={m.get('time_to_goal')} "
                  f"ct={m.get('cross_track_rms', float('nan')):.3f} hd={m.get('heading_rms', float('nan')):.3f} "
                  f"Fx_band={extra.get('handle_Fx_band_rms')} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if pname == "s_large" and payload == 12.5:
                np.savez_compressed(out / f"trace_{label}_s_large_12.5.npz",
                                    **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})

    for step in [s for s in cfg["steps"] if s["name"] in STEPS]:
        for payload in PAYLOADS:
            res = run_step(policy, step, Condition(payload_kg=payload, seed=0), cfg, label=step["name"])
            a = rec.arrays()
            T = len(a["yaw"])
            start = int((float(step.get("pre_s", 2.0)) + 1.5) / dt)
            extra = xm.window_metrics(a, start, T, dt, policy.cart_mass_total(), robot_mass)
            est_natural.append(xm.estimator_series(a, int(1.0 / dt), T))
            results["steps"].append({"step": step["name"], "payload_kg": payload, "metrics": res["metrics"],
                                     "extra": extra, "trace": res["trace"]})

    # Known cart-side pulses, open-loop command, 12.5 kg.
    for cname, cmd in PULSE_CMDS.items():
        for name, (fx, fy, mz), dur, axis in PULSES:
            Condition(payload_kg=PULSE_PAYLOAD, seed=0).apply(policy)
            onset, n = int(4.0 / dt), int((4.0 + dur + 4.5) / dt)
            end = onset + max(1, int(round(dur / dt)))
            failure = None
            for k in range(n):
                if k == onset:
                    _, _, yaw = policy.pose()
                    c, s = math.cos(yaw), math.sin(yaw)
                    env.inject_cart_wrench([c * fx - s * fy, s * fx + c * fy, 0.], [0., 0., mz], dur)
                policy.step(np.asarray(cmd, float))
                failure = policy.failure()
                if failure:
                    break
            a = rec.arrays()
            pm = xm.pulse_metrics(a, onset, end, dt, axis=axis) if not failure and len(a["yaw"]) > end else {}
            pm.update(failure=failure)
            results["pulses"].append({"cmd": cname, "pulse": name, "metrics": pm})
            if cname == "move03" and name in ("+Fx25", "-Fx20_hold3s"):
                sl = slice(onset - int(1.0 / dt), min(len(a["yaw"]), end + int(4.0 / dt)))
                np.savez_compressed(out / f"pulse_{label}_{cname}_{name}.npz", t0=onset - sl.start,
                                    F=a["F"][sl], F_sub=a["F_sub"][sl], pred=a["pred_resultant"][sl],
                                    vel=a["vel"][sl], ref=a["cmd_ref"][sl], zmp=a["zmp"][sl], tilt=a["tilt"][sl])
    results["estimator"] = xm.estimator_accuracy(est_natural)
    results["wall_s"] = time.perf_counter() - t0
    (out / f"result_{label}.json").write_text(json.dumps(results, default=float), encoding="utf-8")
    print(f"[{label}] done in {results['wall_s']:.0f}s", flush=True)


# ============================================================================ summary
def summarize(out: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    data = {}
    for label in ORDER:
        f = out / f"result_{label}.json"
        if f.is_file():
            data[label] = json.loads(f.read_text(encoding="utf-8"))
    labels = [l for l in ORDER if l in data]
    colors = dict(zip(ORDER, ["#9ca3af", "#a3a3a3", "#60a5fa", "#2563eb", "#f59e0b", "#10b981", "#dc2626"]))

    def path_mean(label, key, payload, src="metrics", paths=("s_large", "s_large_mirror")):
        vals = [r[src].get(key) for r in data[label]["paths"] if r["payload_kg"] == payload and r["path"] in paths]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def step_val(label, step, payload, key, src="metrics"):
        r = next((r for r in data[label]["steps"] if r["step"] == step and r["payload_kg"] == payload), None)
        v = None if r is None else r[src].get(key)
        return v if v is None or np.isfinite(v) else None

    def pulse_val(label, cmd, pulse, key):
        r = next((r for r in data[label]["pulses"] if r["cmd"] == cmd and r["pulse"] == pulse), None)
        return None if r is None else r["metrics"].get(key)

    summary = {"labels": labels, "paths": {}, "steps": {}, "pulses": {}, "estimator": {},
               "success": {l: [sum(bool(r["metrics"].get("success")) for r in data[l]["paths"]), len(data[l]["paths"])]
                           for l in labels},
               "step_failures": {l: sum(r["metrics"].get("failure") is not None for r in data[l]["steps"]) for l in labels}}
    path_keys = [("cross_track_rms", "metrics", 100, "Lateral RMS (cm)"), ("heading_rms", "metrics", 180 / math.pi, "Heading RMS (deg)"),
                 ("time_to_goal", "metrics", 1, "Time to goal (s)"), ("cart_vx_band_rms", "metrics", 1, "Speed ripple vx 0.5-5 Hz (m/s)"),
                 ("cart_wz_band_rms", "metrics", 1, "Yaw-rate ripple 0.5-5 Hz (rad/s)"),
                 ("handle_Fx_band_rms", "extra", 1, "Push-force ripple Fx 0.5-5 Hz (N)"),
                 ("handle_Mz_band_rms", "extra", 1, "Yaw-moment ripple Mz 0.5-5 Hz (N·m)"),
                 ("handle_Fx_hf_rms", "extra", 1, "Force HF content Fx >5 Hz (N)"),
                 ("squeeze_y_std", "extra", 1, "Internal squeeze force std (N)"),
                 ("hand_force_p95", "extra", 1, "Hand force p95 (N)"),
                 ("zmp_offset_band_rms", "extra", 100, "Hand-reaction ZMP ripple (cm)"),
                 ("mech_power_mean_W", "extra", 1, "Joint mech. power (W)"),
                 ("cost_of_transport", "extra", 1, "Cost of transport (-)"),
                 ("robot_tilt_p95", "extra", 180 / math.pi, "Robot tilt p95 (deg)")]
    for key, src, scale, title in path_keys:
        summary["paths"][key] = {l: [None if (v := path_mean(l, key, p, src)) is None else v * scale for p in PAYLOADS]
                                 for l in labels}

    fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
    for ax, (key, src, scale, title) in zip(axes.flat, path_keys):
        w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            vals = [v if v is not None else np.nan for v in summary["paths"][key][l]]
            ax.bar(np.arange(3) + (i - (len(labels) - 1) / 2) * w, vals, width=w, color=colors[l], label=l)
        ax.set_xticks(range(3), [f"{p:g} kg" for p in PAYLOADS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(path_keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Wide S + mirror (mean of 2 deterministic runs) | lower is better", fontsize=14)
    fig.savefig(out / "fig1_path_metrics.png", dpi=150)
    fig.savefig(out / "fig1_path_metrics.pdf")
    plt.close(fig)

    step_keys = [("vx_0_to_03", "rise_time"), ("vx_0_to_03", "ss_error"), ("vx_0_to_06", "ss_error"),
                 ("stop_06", "stopping_distance"), ("stop_06", "stopping_time"), ("stop_06", "stop_rebound_events"),
                 ("wz_p03", "settling_time"), ("wz_m03", "settling_time"), ("wz_p03", "ripple_rms"), ("hold", "drift_distance")]
    for step, key in step_keys:
        summary["steps"][f"{step}:{key}"] = {l: [step_val(l, step, p, key) for p in PAYLOADS] for l in labels}
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), constrained_layout=True)
    for ax, (step, key) in zip(axes.flat, step_keys):
        w = 0.8 / len(labels)
        for i, l in enumerate(labels):
            vals = [v if v is not None else np.nan for v in summary["steps"][f"{step}:{key}"][l]]
            if key == "ss_error":
                vals = [abs(v) for v in vals]
            ax.bar(np.arange(3) + (i - (len(labels) - 1) / 2) * w, vals, width=w, color=colors[l], label=l)
        ax.set_xticks(range(3), [f"{p:g} kg" for p in PAYLOADS])
        ax.set_title(f"{step}: {'|ss_error|' if key == 'ss_error' else key}")
        ax.grid(axis="y", alpha=.25)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Step / stop / hold tests (open-loop command) | missing bar = not confirmed", fontsize=14)
    fig.savefig(out / "fig2_step_metrics.png", dpi=150)
    fig.savefig(out / "fig2_step_metrics.pdf")
    plt.close(fig)

    pulse_keys = [("peak_dev_vx", 1, "Peak |Δvx| (m/s)"), ("peak_dev_wz", 1, "Peak |Δwz| (rad/s)"),
                  ("recovery_s", 1, "Recovery time (s)"), ("hand_force_peak_200hz", 1, "Hand force peak 200 Hz (N)"),
                  ("zmp_offset_peak_dev", 100, "ZMP offset peak dev (cm)"), ("robot_tilt_peak", 180 / math.pi, "Robot tilt peak (deg)")]
    for cmd in PULSE_CMDS:
        for name, *_ in PULSES:
            summary["pulses"][f"{cmd}:{name}"] = {l: {k: pulse_val(l, cmd, name, k) for k, *_ in pulse_keys +
                                                     [("recovered",), ("failure",), ("est_peak_ratio",), ("est_sign_correct",),
                                                      ("est_half_rise_lag_ms",), ("left_tolerance",)]} for l in labels}
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)
    names = [n for n, *_ in PULSES]
    for row, cmd in enumerate(PULSE_CMDS):
        for ax, (key, scale, title) in zip(axes[row], pulse_keys):
            w = 0.8 / len(labels)
            for i, l in enumerate(labels):
                vals = [pulse_val(l, cmd, n, key) for n in names]
                vals = [np.nan if v is None else v * scale for v in vals]
                ax.bar(np.arange(len(names)) + (i - (len(labels) - 1) / 2) * w, vals, width=w, color=colors[l], label=l)
            ax.set_xticks(range(len(names)), names, rotation=30, fontsize=7)
            ax.set_title(f"{cmd}: {title}")
            ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Known cart-side force pulses at {PULSE_PAYLOAD:g} kg | recovery: filtered error within tolerance for 0.5 s "
                 "(missing = not recovered)", fontsize=13)
    fig.savefig(out / "fig3_pulse_metrics.png", dpi=150)
    fig.savefig(out / "fig3_pulse_metrics.pdf")
    plt.close(fig)

    # Estimator accuracy (V5.1 resultant head vs V5.2 per-hand estimator + composition).
    for l in labels:
        summary["estimator"][l] = data[l].get("estimator", {})
    est_labels = [l for l in labels if summary["estimator"][l]]
    if est_labels:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        ax = axes[0]
        for i, l in enumerate(est_labels):
            e = summary["estimator"][l]
            vals = [e.get(f"est_handle_vs_mean_{n}_r2") for n in ("Fx", "Fy", "Mz")]
            ax.bar(np.arange(3) + (i - .5) * .38, [np.nan if v is None else v for v in vals], width=.38, color=colors[l], label=l)
        ax.set_xticks(range(3), ["Fx", "Fy", "Mz"])
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
        ax.set_title("Handle resultant R² (MuJoCo, vs 4-substep mean)")
        ax.legend()
        ax.grid(axis="y", alpha=.25)
        ax = axes[1]
        e = summary["estimator"].get("V5.2", {})
        hnames = [f"{s}_{a}" for s in ("L", "R") for a in ("fx", "fy", "fz", "mx", "my", "mz")]
        if e:
            ax.bar(np.arange(12) - .2, [np.nan if e.get(f"est_hand_{n}_r2") is None else e[f"est_hand_{n}_r2"] for n in hnames],
                   width=.4, color=colors["V5.2"], label="V5.2 estimator")
            ax.bar(np.arange(12) + .2, [np.nan if e.get(f"prior_hand_{n}_r2") is None else e[f"prior_hand_{n}_r2"] for n in hnames],
                   width=.4, color="#6b7280", label="(J^T)^+ tau prior alone")
            ax.set_xticks(range(12), hnames, rotation=45, fontsize=7)
            ax.set_ylim(min(-0.5, ax.get_ylim()[0]), 1)
            ax.legend(fontsize=8)
        ax.set_title("V5.2 per-hand R² (MuJoCo)")
        ax.grid(axis="y", alpha=.25)
        ax = axes[2]
        for l in est_labels:
            f = out / f"pulse_{l}_move03_-Fx20_hold3s.npz"
            if f.is_file():
                d = np.load(f)
                t = (np.arange(len(d["F"])) - int(d["t0"])) * 0.02
                ax.plot(t, d["F"][:, 0], color=colors[l], alpha=.35, lw=.8)
                ax.plot(t, d["pred"][:, 0], color=colors[l], lw=1.4, label=f"{l} pred (thin = truth)")
        ax.set_title("Sustained -20 N resistance: handle Fx")
        ax.set_xlabel("s from onset")
        ax.legend(fontsize=8)
        fig.savefig(out / "fig4_estimator.png", dpi=150)
        fig.savefig(out / "fig4_estimator.pdf")
        plt.close(fig)

    # Force spectra on the same path (wide S, 12.5 kg): gait-coupled ripple.
    from scipy import signal as sg
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    for l in labels:
        f = out / f"trace_{l}_s_large_12.5.npz"
        if not f.is_file():
            continue
        d = np.load(f)
        for j, (ax, n) in enumerate(zip(axes, ("Fx (N)", "Fy (N)", "Mz (N·m)"))):
            x = d["F_sub"][int(2 / .02):].reshape(-1, 3)[:, j]
            fr, p = sg.welch(x - x.mean(), fs=200., nperseg=1024)
            ax.semilogy(fr, p, color=colors[l], lw=1, label=l)
            ax.set_xlim(0, 30)
            ax.set_title(f"Handle {n} PSD, wide S 12.5 kg (200 Hz)")
            ax.set_xlabel("Hz")
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "fig5_force_psd.png", dpi=150)
    fig.savefig(out / "fig5_force_psd.pdf")
    plt.close(fig)

    # Pulse traces.
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), constrained_layout=True)
    for col, pulse in enumerate(("+Fx25", "-Fx20_hold3s")):
        for l in labels:
            f = out / f"pulse_{l}_move03_{pulse}.npz"
            if not f.is_file():
                continue
            d = np.load(f)
            t = (np.arange(len(d["vel"])) - int(d["t0"])) * 0.02
            axes[0, col].plot(t, d["vel"][:, 0], color=colors[l], lw=1, label=l)
            axes[1, col].plot(t, np.linalg.norm(d["zmp"], axis=-1) * 100, color=colors[l], lw=1, label=l)
        axes[0, col].axvline(0, color="#999", lw=.8)
        axes[1, col].axvline(0, color="#999", lw=.8)
        axes[0, col].set_title(f"{pulse} while pushing at 0.3 m/s: cart vx (m/s)")
        axes[1, col].set_title(f"{pulse}: hand-reaction ZMP offset |Δ| (cm)")
        axes[1, col].set_xlabel("s from onset")
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.savefig(out / "fig6_pulse_traces.png", dpi=150)
    fig.savefig(out / "fig6_pulse_traces.pdf")
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"[bench-versions] summary + figures -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "logs/bench/versions_20k")
    ap.add_argument("--worker")
    ap.add_argument("--labels", nargs="*", default=ORDER)
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--trace_stride", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="smoke test: one path/payload, two steps, two pulses")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import torch
        torch.set_num_threads(1)
        worker(args.worker, args.out, args.trace_stride, args.quick)
        return
    if not args.summarize_only:
        procs = []
        for label in args.labels:
            log = open(args.out / f"log_{label}.txt", "w", encoding="utf-8")
            procs.append((label, log, subprocess.Popen([sys.executable, "-u", __file__, "--out", str(args.out),
                                                        "--worker", label, "--trace_stride", str(args.trace_stride)]
                                                       + (["--quick"] if args.quick else []),
                                                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)))
        for label, log, p in procs:
            p.wait()
            log.close()
            print(f"[bench-versions] {label}: exit {p.returncode}", flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
