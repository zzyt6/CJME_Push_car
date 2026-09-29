"""Per-run traces and the metrics computed from them (path following, step response, aggregation)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


CART_QUALITY_SIGNALS = (
    "idle_correction_gate", "idle_correction_ready", "idle_est_vx", "idle_est_vy", "idle_est_wz",
    "cart_fx", "cart_fy", "cart_fz", "cart_mx", "cart_my", "cart_mz", "cart_mz_left", "cart_mz_right",
    "cart_vx_band", "cart_vy_band", "cart_wz_band", "cart_fx_band", "cart_fy_band", "cart_mz_band",
    "cart_ref_vx", "cart_ref_vy", "cart_ref_wz", "cart_tilt_rad", "robot_tilt_rad", "wheel_lift_m",
    "quality_ready", "cart_ripple_cost", "cart_wrench_osc_cost",
    "cart_lin_ripple_cost", "cart_yaw_ripple_cost", "cart_response_ready", "cart_response_error_cost",
    "cart_response_error_vx", "cart_response_error_vy", "cart_response_error_wz",
    "cart_est_vx", "cart_est_vy", "cart_est_wz",
    "est_left_fx", "est_left_fy", "est_left_fz", "est_left_mx", "est_left_my", "est_left_mz",
    "est_right_fx", "est_right_fy", "est_right_fz", "est_right_mx", "est_right_my", "est_right_mz",
)


@dataclass
class Trace:
    """Per-control-step record of one run (lists, appended by the runner)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    yaw: list = field(default_factory=list)
    vx: list = field(default_factory=list)
    vy: list = field(default_factory=list)
    wz: list = field(default_factory=list)
    cmd: list = field(default_factory=list)               # (3,) command given to the policy
    s: list = field(default_factory=list)                 # path progress
    cross_track: list = field(default_factory=list)
    heading_err: list = field(default_factory=list)
    dist_goal: list = field(default_factory=list)
    telemetry: list = field(default_factory=list)         # dict per step (policy specific, may be empty)
    failure: str | None = None
    failure_t: float | None = None

    def append(self, t, pose, vel, cmd, follower_info=None, telemetry=None):
        self.t.append(float(t))
        self.x.append(float(pose[0])); self.y.append(float(pose[1])); self.yaw.append(float(pose[2]))
        self.vx.append(float(vel[0])); self.vy.append(float(vel[1])); self.wz.append(float(vel[2]))
        self.cmd.append([float(v) for v in cmd])
        if follower_info is not None:
            self.s.append(float(follower_info["s"]))
            self.cross_track.append(float(follower_info["cross_track"]))
            self.heading_err.append(float(follower_info["heading_err"]))
            self.dist_goal.append(float(follower_info["dist_goal"]))
        self.telemetry.append(dict(telemetry or {}))

    def __len__(self):
        return len(self.t)

    def telemetry_series(self, key: str) -> np.ndarray:
        vals = [d.get(key, np.nan) for d in self.telemetry]
        return np.asarray(vals, dtype=float)

    def decimated(self, stride: int = 5) -> dict:
        """Compact copy for the results JSON (position, velocity, command, errors)."""
        sl = slice(None, None, max(1, stride))
        out = {"t": self.t[sl], "x": self.x[sl], "y": self.y[sl], "yaw": self.yaw[sl], "vx": self.vx[sl], "vy": self.vy[sl],
               "wz": self.wz[sl], "cmd": self.cmd[sl]}
        if self.cross_track:
            out.update(s=self.s[sl], cross_track=self.cross_track[sl], heading_err=self.heading_err[sl], dist_goal=self.dist_goal[sl])
        out = {k: np.round(np.asarray(v, dtype=float), 4).tolist() for k, v in out.items()}
        for key in ("grasp_left", "grasp_right", "push_left", "push_right", "lat_left", "lat_right", "up_left", "up_right"):
            series = self.telemetry_series(key)
            if np.isfinite(series).any():
                out[key] = [None if not np.isfinite(v) else round(float(v), 2) for v in series[sl]]
        for key in CART_QUALITY_SIGNALS:
            series = self.telemetry_series(key)
            if np.isfinite(series).any():
                out[key] = [None if not np.isfinite(v) else round(float(v), 6) for v in series[sl]]
        return out


def _rms(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def _p95(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 95)) if len(a) else float("nan")


def _time_series(t, *values):
    """Validate the sampling clock without silently sorting or removing missing data."""
    t = np.asarray(t, dtype=float)
    arrays = [np.asarray(value, dtype=float) for value in values]
    if t.ndim != 1 or any(a.ndim != 1 or len(a) != len(t) for a in arrays):
        raise ValueError("time and signal arrays must be one-dimensional with equal length")
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("timestamps must be finite and strictly increasing")
    return (t, *arrays)


def _continuous_intervals(t, nominal_dt=None):
    """Reject gaps exceeding 1.5 nominal sample periods in full-rate bench traces."""
    dt = np.diff(t)
    if not len(dt):
        return np.zeros(0, dtype=bool)
    if nominal_dt is None:
        nominal_dt = float(np.median(dt))
    return dt <= 1.5 * nominal_dt + 1e-9


def _window_error_metrics(t, error, valid, start, end, *, discrete=False):
    """Statistics on (start, end], with integrals that never bridge invalid samples.

    Continuous errors use trapezoidal quadrature, clipped at window boundaries. Discrete
    reward costs use their right endpoint value for each control interval; an invalid prior
    cost does not discard the next valid transition. Coverage excludes invalid intervals and
    sampling gaps; unobserved duration is not zero error.
    """
    valid = np.asarray(valid, dtype=bool) & np.isfinite(error)
    sample = valid & (t > start + 1e-9) & (t <= end + 1e-9)
    out = {"samples": int(sample.sum()), "rmse": _rms(error[sample]) if sample.any() else None,
           "mean": float(np.mean(error[sample])) if sample.any() else None,
           "iae": None, "ise": None, "integral": None, "valid_duration_s": 0.0, "coverage": 0.0}
    requested = max(0.0, float(end - start))
    if len(t) < 2 or requested == 0:
        return out
    left = np.maximum(t[:-1], start)
    right = np.minimum(t[1:], end)
    interval = valid[1:] & _continuous_intervals(t) & (right > left)
    if not discrete:
        interval &= valid[:-1]
    if not interval.any():
        return out
    dt = (right - left)[interval]
    duration = float(dt.sum())
    out.update(valid_duration_s=duration, coverage=min(1.0, duration / requested))
    if discrete:
        value = error[1:][interval]
        out.update(iae=float(np.sum(dt * np.abs(value))), ise=float(np.sum(dt * value ** 2)),
                   integral=float(np.sum(dt * value)))
    else:
        base = t[:-1][interval]
        spacing = np.diff(t)[interval]
        y0, y1 = error[:-1][interval], error[1:][interval]
        a = y0 + (y1 - y0) * (left[interval] - base) / spacing
        b = y0 + (y1 - y0) * (right[interval] - base) / spacing
        out.update(iae=float(np.sum(dt * (np.abs(a) + np.abs(b)) * .5)),
                   ise=float(np.sum(dt * (a ** 2 + b ** 2) * .5)),
                   integral=float(np.sum(dt * (a + b) * .5)))
    return out


def reference_tracking_metrics(t, actual, reference, t_step: float, transient_s: float = 1.5) -> dict:
    """Tracking of the *varying* reference, separate from its own rise/settling times.

    These are raw velocity errors, without envelope smoothing. Each integral is restricted to
    valid adjacent observations; coverage reports the fraction of the requested window observed.
    """
    t, actual, reference = _time_series(t, actual, reference)
    if not math.isfinite(t_step) or not math.isfinite(transient_s) or transient_s < 0:
        raise ValueError("t_step must be finite and transient_s finite and nonnegative")
    end = max(t_step, float(t[-1])) if len(t) else t_step
    valid = np.isfinite(actual) & np.isfinite(reference)
    error = actual - reference
    out = {}
    for name, start, stop in (("full", t_step, end), ("transient", t_step, t_step + transient_s),
                              ("post", t_step + transient_s, end)):
        metrics = _window_error_metrics(t, error, valid, start, stop)
        for key in ("rmse", "iae", "ise", "valid_duration_s", "coverage"):
            out[f"reference_tracking_{name}_{key}"] = metrics[key]
    return out


# ---------------------------------------------------------------------------- path following

def path_metrics(trace: Trace, path, follower_info: dict, overload_thresholds: dict | None = None) -> dict:
    """Success / time-to-goal / final error / tracking quality / interaction statistics of one path run."""
    reached = bool(follower_info.get("reached"))
    failed = trace.failure is not None
    gx, gy, gyaw = path.goal
    if len(trace) == 0:
        return {"success": False, "failure": trace.failure}
    end = -1
    if reached and follower_info.get("t_reached") is not None:
        end = int(np.searchsorted(np.asarray(trace.t), follower_info["t_reached"])) - 1
        end = max(0, min(end, len(trace) - 1))
    final_pos_err = float(math.hypot(trace.x[end] - gx, trace.y[end] - gy))
    final_yaw_err = float(abs(math.atan2(math.sin(trace.yaw[end] - gyaw), math.cos(trace.yaw[end] - gyaw))))
    settled_pos_err = float(math.hypot(trace.x[-1] - gx, trace.y[-1] - gy))       # after the post-reach hold (drift included)
    min_goal_dist = float(np.min(np.hypot(np.asarray(trace.x) - gx, np.asarray(trace.y) - gy)))
    xy = np.stack([trace.x, trace.y], axis=1)
    travelled = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0
    ct = np.asarray(trace.cross_track[: end + 1] if end >= 0 else trace.cross_track)
    he = np.asarray(trace.heading_err[: end + 1] if end >= 0 else trace.heading_err)
    cmd = np.asarray(trace.cmd)
    vx, wz = np.asarray(trace.vx), np.asarray(trace.wz)
    duration = float(trace.t[-1] - trace.t[0]) if len(trace) > 1 else 0.0
    res = {
        "success": bool(reached and not failed),
        "reached": reached,
        "failure": trace.failure,
        "failure_t": trace.failure_t,
        "timed_out": bool(follower_info.get("timed_out")),
        "time_to_goal": follower_info.get("t_reached"),
        "time_enter": follower_info.get("t_enter"),
        "final_pos_err": final_pos_err,
        "settled_pos_err": settled_pos_err,
        "min_goal_dist": min_goal_dist,
        "final_yaw_err": final_yaw_err,
        "progress": float(follower_info.get("s", 0.0)),
        "progress_frac": float(follower_info.get("s", 0.0)) / max(path.length, 1e-9),
        "cross_track_mean": float(np.mean(np.abs(ct))) if len(ct) else float("nan"),
        "cross_track_rms": _rms(ct),
        "cross_track_max": float(np.max(np.abs(ct))) if len(ct) else float("nan"),
        "heading_rms": _rms(he),
        "path_length": path.length,
        "travelled": travelled,
        "path_ratio": travelled / max(path.length, 1e-9),
        "mean_speed": float(follower_info.get("s", 0.0)) / max(duration, 1e-9),
        "cmd_vx_rmse": _rms(cmd[:, 0] - vx) if len(cmd) else float("nan"),
        "cmd_wz_rmse": _rms(cmd[:, 2] - wz) if len(cmd) else float("nan"),
        "cmd_rate": float(np.mean(np.linalg.norm(np.diff(cmd, axis=0), axis=1))) if len(cmd) > 1 else 0.0,
        "duration": duration,
    }
    res.update(interaction_metrics(trace, overload_thresholds))
    res.update(cart_quality_metrics(trace))
    return res


def cart_quality_metrics(trace: Trace, start_s: float = 0.0,
                        response_window: tuple[float, float] | None = None) -> dict:
    """Full-rate diagnostics. Missing telemetry stays missing, never zero.

    Bands exclude warmup and the caller's transient window. Geometric peaks cover the entire run,
    including warmup and failures. Force peaks mean sampled peaks, not continuous-time maxima.
    """
    t, = _time_series(trace.t)
    ready = trace.telemetry_series("quality_ready")
    selected = np.isfinite(ready) & (ready > 0.5) & (t >= start_s)
    out = {"quality_samples": int(selected.sum())} if np.isfinite(ready).any() else {}
    for name in ("cart_tilt_rad", "robot_tilt_rad", "wheel_lift_m"):
        values = trace.telemetry_series(name)
        values = values[np.isfinite(values)]
        if len(values):
            out[f"{name}_p95"] = float(np.percentile(values, 95))
            out[f"{name}_max"] = float(np.max(values))
    for name in ("cart_fx", "cart_fy", "cart_fz", "cart_mx", "cart_my", "cart_mz"):
        values = trace.telemetry_series(name)
        full = values[np.isfinite(values)]
        if len(full):
            out[f"{name}_abs_peak"] = float(np.max(np.abs(full)))
        values = values[selected & np.isfinite(values)]
        if len(values):
            out[f"{name}_mean"] = float(np.mean(values))
            out[f"{name}_std"] = float(np.std(values))
            out[f"{name}_abs_p95"] = float(np.percentile(np.abs(values), 95))
    for name in ("cart_vx_band", "cart_vy_band", "cart_wz_band", "cart_fx_band", "cart_fy_band", "cart_mz_band"):
        values = trace.telemetry_series(name)
        values = values[selected & np.isfinite(values)]
        if len(values):
            out[f"{name}_rms"] = _rms(values)
    for name in ("cart_ripple_cost", "cart_lin_ripple_cost", "cart_yaw_ripple_cost", "cart_wrench_osc_cost"):
        values = trace.telemetry_series(name)
        values = values[selected & np.isfinite(values)]
        if len(values):
            out[f"{name}_mean"] = float(np.mean(values))
    response_ready = trace.telemetry_series("cart_response_ready")
    response_valid = np.isfinite(response_ready) & (response_ready > .5)
    response_selected = response_valid & (t >= start_s)
    for axis in ("vx", "vy", "wz"):
        values = trace.telemetry_series(f"cart_response_error_{axis}")
        values = values[response_selected & np.isfinite(values)]
        if len(values):
            out[f"cart_response_error_{axis}_rmse"] = _rms(values)
    response_cost = trace.telemetry_series("cart_response_error_cost")
    response_cost = response_cost[response_selected & np.isfinite(response_cost)]
    if len(response_cost):
        out["cart_response_error_cost_mean"] = float(np.mean(response_cost))
    if response_window is not None:
        window_start, window_end = map(float, response_window)
        if not math.isfinite(window_start) or not math.isfinite(window_end) or window_end < window_start:
            raise ValueError("response_window boundaries must be finite and end >= start")
        windows = (("transient", window_end), ("full", max(window_start, float(t[-1])) if len(t) else window_start))
        for name, end in windows:
            for axis in ("vx", "vy", "wz"):
                values = trace.telemetry_series(f"cart_response_error_{axis}")
                metrics = _window_error_metrics(t, values, response_valid, window_start, end)
                for key in ("rmse", "iae", "ise", "valid_duration_s", "coverage"):
                    out[f"cart_response_error_{axis}_{key}_{name}"] = metrics[key]
            values = trace.telemetry_series("cart_response_error_cost")
            metrics = _window_error_metrics(t, values, response_valid, window_start, end, discrete=True)
            out[f"cart_response_error_cost_mean_{name}"] = metrics["mean"]
            out[f"cart_response_error_cost_integral_{name}"] = metrics["integral"]
            out[f"cart_response_{name}_valid_duration_s"] = metrics["valid_duration_s"]
            out[f"cart_response_{name}_coverage"] = metrics["coverage"]
    for axis in ("vx", "vy", "wz"):
        values = np.asarray(getattr(trace, axis), dtype=float)
        ref = trace.telemetry_series(f"cart_ref_{axis}")
        mask = selected & np.isfinite(ref) & np.isfinite(values)
        if mask.any():
            error = values[mask] - ref[mask]
            out[f"cart_{axis}_raw_ref_rmse"] = _rms(error)
            out[f"cart_{axis}_raw_ref_error_std"] = float(np.std(error))
            out[f"cart_{axis}_mean"] = float(np.mean(values[mask]))
    return out


def disturbance_recovery_metrics(trace: Trace, events: list, cfg: dict) -> dict:
    """Task recovery after each known disturbance ends, with causal filtering and a dwell window.

    Report unrecovered/interrupted events explicitly; never count missing recovery as zero seconds.
    Thresholds define an evaluation convention, not a physical safety guarantee.
    """
    settings = {"velocity_tolerance": [0.08, 0.06, 0.10], "cross_track_m": 0.10,
                "hold_s": 0.5, "filter_tau_s": 0.2, **cfg}
    tol = np.asarray(settings["velocity_tolerance"], dtype=float)
    if tol.shape != (3,) or not np.isfinite(tol).all() or (tol <= 0).any():
        raise ValueError("recovery.velocity_tolerance needs three positive finite values")
    if any(not math.isfinite(float(settings[k])) or settings[k] < 0 for k in ("cross_track_m", "hold_s", "filter_tau_s")):
        raise ValueError("recovery thresholds must be finite and nonnegative")
    t = np.asarray(trace.t)
    if len(t) < 2 or not events:
        return {"settings": settings, "events": [], "metrics": {"disturbance_recovery_events": len(events)}}
    velocity = np.column_stack((trace.vx, trace.vy, trace.wz))
    ref = np.column_stack([trace.telemetry_series(f"cart_ref_{axis}") for axis in ("vx", "vy", "wz")])
    ref = np.where(np.isfinite(ref), ref, np.asarray(trace.cmd))
    error = velocity - ref
    filtered = error.copy()
    tau = float(settings["filter_tau_s"])
    for i in range(1, len(t)):
        alpha = -math.expm1(-(t[i] - t[i-1]) / tau) if tau > 0 else 1.
        filtered[i] = filtered[i-1] + alpha * (error[i] - filtered[i-1])
    within = np.isfinite(filtered).all(axis=1) & (np.abs(filtered) <= tol).all(axis=1)
    if len(trace.cross_track) == len(t):
        within &= np.abs(trace.cross_track) <= settings["cross_track_m"]
    rows = []
    for i, event in enumerate(events):
        end = float(event["end_t"])
        next_t = float(events[i+1]["t"]) if i+1 < len(events) else math.inf
        row = {**event, "recovery_s": None, "confirmed_at_s": None, "status": "not_recovered"}
        if next_t <= end:
            row["status"] = "interrupted"
        else:
            begin = None
            left_tolerance = False
            for j in np.flatnonzero((t >= end) & (t < next_t)):
                if not within[j] or (trace.failure_t is not None and t[j] >= trace.failure_t):
                    begin = None
                    left_tolerance = True
                    continue
                if begin is None:
                    begin = t[j]
                if t[j] - begin + 1e-9 >= settings["hold_s"]:
                    row.update(recovery_s=float(begin - end), confirmed_at_s=float(t[j]),
                               status="recovered" if left_tolerance else "within_tolerance")
                    break
            if row["recovery_s"] is None and math.isfinite(next_t):
                row["status"] = "interrupted"
        rows.append(row)
    times = [r["recovery_s"] for r in rows if r["recovery_s"] is not None]
    return {"settings": settings, "events": rows, "metrics": {
        "disturbance_recovery_events": len(rows), "disturbance_recovered_frac": len(times) / len(rows),
        "disturbance_recovery_mean_s": float(np.mean(times)) if times else None,
        "disturbance_recovery_max_s": max(times) if times else None}}


def interaction_metrics(trace: Trace, overload_thresholds: dict | None = None) -> dict:
    """Grasp-force statistics from the telemetry (empty dict when the policy has no cart)."""
    fl, fr = trace.telemetry_series("grasp_left"), trace.telemetry_series("grasp_right")
    if not np.isfinite(fl).any():
        return {}
    push = np.maximum(trace.telemetry_series("push_left"), trace.telemetry_series("push_right"))
    pull = np.maximum(-trace.telemetry_series("push_left"), -trace.telemetry_series("push_right"))
    lat = np.maximum(np.abs(trace.telemetry_series("lat_left")), np.abs(trace.telemetry_series("lat_right")))
    up = np.maximum(trace.telemetry_series("up_left"), trace.telemetry_series("up_right"))
    out = {
        "grasp_p95": _p95(np.maximum(fl, fr)), "grasp_mean": float(np.nanmean((fl + fr) / 2)),
        "push_p95": _p95(push), "pull_p95": _p95(pull), "lateral_p95": _p95(lat), "up_p95": _p95(up),
    }
    if overload_thresholds:
        thr = overload_thresholds
        extra = thr.get("per_kg", 0.0) * max(0.0, float(thr.get("payload_kg", 0.0)))
        over = (pull > thr["pull"] + extra) | (up > thr["up"] + extra) | (lat > thr["tangential"] + extra)
        out["overload_frac"] = float(np.mean(over[np.isfinite(pull)])) if np.isfinite(pull).any() else float("nan")
    dq = trace.telemetry_series("upper_dq_max")
    if np.isfinite(dq).any():
        out["upper_dq_mean"] = float(np.nanmean(dq))
    dq_all = [d.get("upper_dq") for d in trace.telemetry if d.get("upper_dq") is not None]
    if len(dq_all) > 1:
        dq_all = np.asarray(dq_all, dtype=float)
        out["arm_target_std"] = float(np.mean(np.std(dq_all, axis=0)))        # how much the corrections actually vary
        out["arm_target_abs_mean"] = float(np.mean(np.abs(dq_all)))            # how far from the grasp pose they sit
    for key in ("arm_clip_frac", "cmd_clip_frac"):
        series = trace.telemetry_series(key)
        if np.isfinite(series).any():
            out[key] = float(np.nanmean(series))
    hc = [d.get("homie_cmd") for d in trace.telemetry if d.get("homie_cmd") is not None]
    if len(hc) > 1:
        hc = np.asarray(hc, dtype=float)
        out["homie_cmd_rate"] = float(np.mean(np.linalg.norm(np.diff(hc, axis=0), axis=1)))
        cmd = np.asarray(trace.cmd)
        out["residual_mean_abs"] = [float(v) for v in np.mean(np.abs(hc - cmd[: len(hc)]), axis=0)]
    return out


# ---------------------------------------------------------------------------- step response

def moving_average(y, window: int, t=None):
    """Centred moving average over full windows only (the half-window at each end is dropped, so a periodic ripple
    is cancelled everywhere). Returns the filtered signal, or ``(t_valid, y_filtered)`` when ``t`` is given."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y if t is None else (np.asarray(t, dtype=float), y)
    kernel = np.ones(window) / window
    yf = np.convolve(y, kernel, mode="valid")
    if t is None:
        return yf
    pad = window // 2
    return np.asarray(t, dtype=float)[pad:pad + len(yf)], yf


def step_response_metrics(t, y, t_step: float, y0: float, y1: float, band: float | None = None,
                          ss_window_s: float = 2.0, smooth_s: float = 0.0,
                          settling_dwell_s: float = 0.5) -> dict:
    """Classic step metrics of a signal ``y(t)`` that should move from ``y0`` to ``y1`` at ``t_step``.

    rise_time: 10 % -> 90 % of the step; settling_time: from t_step until |y - y1| stays inside ``band``
    (default 10 % of the step, at least 0.05); overshoot: max excursion past y1 in the step direction / |step|;
    ss_error: mean(y) - y1 over the last ``ss_window_s`` seconds. ``smooth_s`` > 0 applies a centred moving
    average of that length first (a legged robot's velocity oscillates with the gait; the metrics describe the
    envelope, not the steps). Settling must be observed for ``settling_dwell_s`` and persist
    through the last observed sample; missing data or a sampling gap interrupts confirmation.
    """
    t, y = _time_series(t, y)
    if not all(math.isfinite(float(v)) for v in (t_step, y0, y1, ss_window_s, smooth_s, settling_dwell_s)):
        raise ValueError("step-response parameters must be finite")
    if ss_window_s <= 0 or smooth_s < 0 or settling_dwell_s < 0:
        raise ValueError("ss_window_s must be positive; smooth_s and settling_dwell_s nonnegative")
    if band is not None and (not math.isfinite(float(band)) or band < 0):
        raise ValueError("band must be finite and nonnegative")
    nominal_dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    if smooth_s > 0 and len(t) > 2:
        t, y = moving_average(y, int(round(smooth_s / max(nominal_dt, 1e-9))), t)
    m = t >= t_step - 1e-9
    t, y = t[m], y[m]
    continuous = _continuous_intervals(t, nominal_dt=nominal_dt)
    step = y1 - y0
    out = {"rise_time": None, "settling_time": None, "overshoot": None, "ss_error": None, "ss_std": None,
           "settling_confirmed": False, "settling_confirmed_time": None,
           "settling_dwell_s": float(settling_dwell_s)}
    if len(t) == 0:
        return out
    tail = t >= t[-1] - ss_window_s
    if np.isfinite(y[tail]).all():
        out["ss_error"] = float(np.mean(y[tail]) - y1)
        out["ss_std"] = float(np.std(y[tail]))
    band = max(0.1 * abs(step), 0.05) if band is None else band
    inside = np.isfinite(y) & (np.abs(y - y1) <= band)
    # The final in-band run needs an observed dwell; a lone endpoint cannot prove settling.
    if inside[-1]:
        last_out = np.where(~inside)[0]
        idx = 0 if len(last_out) == 0 else last_out[-1] + 1
        gaps = np.flatnonzero(~continuous)
        if len(gaps):
            idx = max(idx, int(gaps[-1]) + 1)
        confirmed = np.flatnonzero(t[idx:] - t[idx] + 1e-9 >= settling_dwell_s)
        if len(confirmed):
            out["settling_time"] = max(0.0, float(t[idx] - t_step))
            out["settling_confirmed"] = True
            out["settling_confirmed_time"] = max(0.0, float(t[idx + confirmed[0]] - t_step))
    if abs(step) < 1e-9:
        return out
    prog = (y - y0) / step                                   # 0 at y0, 1 at y1
    finite = np.isfinite(prog)
    i10 = np.argmax(finite & (prog >= 0.1)) if np.any(finite & (prog >= 0.1)) else None
    i90 = np.argmax(finite & (prog >= 0.9)) if np.any(finite & (prog >= 0.9)) else None
    if i10 is not None and i90 is not None and i90 >= i10:
        if finite[i10:i90 + 1].all() and continuous[i10:i90].all():
            out["rise_time"] = float(t[i90] - t[i10])
    if finite.any():
        out["overshoot"] = float(max(0.0, np.max(prog[finite]) - 1.0))
    return out


def stopping_distance(t, x, y, speed, t_step: float, v_stop: float = 0.05) -> float | None:
    """Legacy first-low-speed distance; missing crossings/positions return ``None``."""
    return stopping_metrics(t, x, y, speed, t_step, v_stop, dwell_s=0.0)["first_low_distance"]


def stopping_metrics(t, x, y, speed, t_step: float, v_stop: float = 0.05,
                     dwell_s: float = 0.5) -> dict:
    """Confirm the final uninterrupted low-speed run after an observed ``dwell_s``.

    A prior dwell followed by a rebound does not establish the task-level stopping result.
    ``first_confirmed_*`` preserves that earlier event as a diagnostic. Invalid samples and
    sampling gaps break the dwell; missing positions make the associated path length unknown.
    ``rebound_count`` remains a sample count, while ``rebound_events`` counts low-to-high edges.
    """
    if dwell_s < 0 or not math.isfinite(float(dwell_s)):
        raise ValueError("dwell_s must be finite and nonnegative")
    if not math.isfinite(float(t_step)) or not math.isfinite(float(v_stop)) or v_stop <= 0:
        raise ValueError("t_step must be finite and v_stop positive and finite")
    t, x, y, speed = _time_series(t, x, y, speed)
    if (speed[np.isfinite(speed)] < 0).any():
        raise ValueError("speed must be a nonnegative magnitude")
    selected = t >= t_step - 1e-9
    out = {"first_low_time": None, "first_low_distance": None, "confirmed": False,
           "confirmed_time": None, "confirmed_distance": None, "post_confirm_travel": None,
           "first_confirmed_time": None, "first_confirmed_distance": None, "post_first_confirm_travel": None,
           "rebound_count": 0, "rebound_events": 0, "invalid_samples": 0,
           "dwell_s": float(dwell_s), "v_stop": float(v_stop)}
    if not selected.any():
        return out
    continuous = _continuous_intervals(t)
    # Preserve the original clock's nominal cadence when the post-step segment is short.
    first = int(np.flatnonzero(selected)[0])
    continuous = continuous[first:]
    ts, xs, ys, sp = t[selected], x[selected], y[selected], speed[selected]
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(sp)
    low = valid & (sp < v_stop)
    out["invalid_samples"] = int((~valid).sum())
    xy = np.stack((xs, ys), axis=1)

    def distance(begin, end):
        points = xy[begin:end + 1]
        if not np.isfinite(points).all() or not continuous[begin:end].all():
            return None
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    low_idx = np.flatnonzero(low)
    if not len(low_idx):
        return out
    first_low = int(low_idx[0])
    out["first_low_time"] = max(0.0, float(ts[first_low] - t_step))
    out["first_low_distance"] = distance(0, first_low)
    high = valid & (sp >= v_stop)
    out["rebound_count"] = int(high[first_low + 1:].sum())
    out["rebound_events"] = int((low[:-1] & high[1:] & continuous).sum())
    run_start, run_confirmed = None, None
    for j in range(len(ts)):
        if not low[j]:
            run_start, run_confirmed = None, None
            continue
        if run_start is None or (j and not continuous[j - 1]):
            run_start, run_confirmed = j, None
        if run_confirmed is None and ts[j] - ts[run_start] + 1e-9 >= dwell_s:
            run_confirmed = j
            if out["first_confirmed_time"] is None:
                out["first_confirmed_time"] = max(0.0, float(ts[j] - t_step))
                out["first_confirmed_distance"] = distance(0, j)
                out["post_first_confirm_travel"] = distance(j, len(ts) - 1)
    if run_confirmed is not None:
        out["confirmed"] = True
        out["confirmed_time"] = max(0.0, float(ts[run_confirmed] - t_step))
        out["confirmed_distance"] = distance(0, run_confirmed)
        out["post_confirm_travel"] = distance(run_confirmed, len(ts) - 1)
    return out


# ---------------------------------------------------------------------------- aggregation

def aggregate(values) -> dict:
    """mean / std / n of a list that may contain None (dropped)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "n": int(len(v))}


def fmt_ms(agg: dict, unit: str = "", digits: int = 2) -> str:
    if agg is None or agg.get("mean") is None:
        return "-"
    if agg.get("n", 0) > 1 and agg.get("std") is not None:
        return f"{agg['mean']:.{digits}f} ± {agg['std']:.{digits}f}{unit}"
    return f"{agg['mean']:.{digits}f}{unit}"
