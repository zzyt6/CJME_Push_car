"""Per-run traces and the metrics computed from them (path following, step response, aggregation)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


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
        return out


def _rms(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def _p95(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 95)) if len(a) else float("nan")


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
    return res


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
                          ss_window_s: float = 2.0, smooth_s: float = 0.0) -> dict:
    """Classic step metrics of a signal ``y(t)`` that should move from ``y0`` to ``y1`` at ``t_step``.

    rise_time: 10 % -> 90 % of the step; settling_time: from t_step until |y - y1| stays inside ``band``
    (default 10 % of the step, at least 0.05); overshoot: max excursion past y1 in the step direction / |step|;
    ss_error: mean(y) - y1 over the last ``ss_window_s`` seconds. ``smooth_s`` > 0 applies a centred moving
    average of that length first (a legged robot's velocity oscillates with the gait; the metrics describe the
    envelope, not the steps).
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if smooth_s > 0 and len(t) > 2:
        t, y = moving_average(y, int(round(smooth_s / max(float(np.median(np.diff(t))), 1e-9))), t)
    m = t >= t_step
    t, y = t[m], y[m]
    step = y1 - y0
    out = {"rise_time": None, "settling_time": None, "overshoot": None, "ss_error": None, "ss_std": None}
    if len(t) < 2:
        return out
    tail = t >= t[-1] - ss_window_s
    out["ss_error"] = float(np.mean(y[tail]) - y1)
    out["ss_std"] = float(np.std(y[tail]))
    if abs(step) < 1e-9:
        return out
    prog = (y - y0) / step                                   # 0 at y0, 1 at y1
    i10 = np.argmax(prog >= 0.1) if np.any(prog >= 0.1) else None
    i90 = np.argmax(prog >= 0.9) if np.any(prog >= 0.9) else None
    if i10 is not None and i90 is not None and i90 >= i10:
        out["rise_time"] = float(t[i90] - t[i10])
    band = max(0.1 * abs(step), 0.05) if band is None else band
    inside = np.abs(y - y1) <= band
    # settling time: first index after which every later sample is inside the band
    if inside.any():
        last_out = np.where(~inside)[0]
        idx = 0 if len(last_out) == 0 else last_out[-1] + 1
        if idx < len(t):
            out["settling_time"] = float(t[idx] - t_step)
    out["overshoot"] = float(max(0.0, np.max(prog) - 1.0))
    return out


def stopping_distance(t, x, y, speed, t_step: float, v_stop: float = 0.05) -> float | None:
    """Distance travelled after ``t_step`` until the speed first drops below ``v_stop``."""
    t = np.asarray(t, dtype=float)
    m = t >= t_step
    if not m.any():
        return None
    xs, ys, sp = np.asarray(x)[m], np.asarray(y)[m], np.asarray(speed)[m]
    below = np.where(sp < v_stop)[0]
    end = int(below[0]) if len(below) else len(xs) - 1
    xy = np.stack([xs[: end + 1], ys[: end + 1]], axis=1)
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


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
