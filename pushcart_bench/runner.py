"""Single-run loops shared by the experiments: path following, velocity steps, disturbances, optional viser view."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .follower import FollowerConfig, PathFollower
from .metrics import Trace, path_metrics, step_response_metrics, stopping_metrics, reference_tracking_metrics
from .paths import Path


# ---------------------------------------------------------------------------- disturbances

@dataclass
class Disturbance:
    """Cart-side disturbances applied during a run (robustness sweep); zero magnitudes = off."""
    kick_dv: float = 0.0                # planar velocity impulse [m/s], random direction
    kick_interval_s: float = 5.0
    stall_factor: float = 0.0           # wheel damping multiplier (0 / 1 = off)
    stall_prob_per_s: float = 0.2
    stall_duration_s: float = 0.3
    seed: int = 0

    def active(self) -> bool:
        return self.kick_dv > 0 or self.stall_factor > 1.0

    def start(self):
        self.rng = np.random.default_rng(self.seed)
        self._next_kick = self.kick_interval_s * (0.5 + 0.5 * self.rng.random()) if self.kick_dv > 0 else math.inf
        self._stall_until, self._stall_wheel = -1.0, None
        self.events = {"kicks": 0, "stalls": 0}
        self.event_log = []

    def apply(self, policy, t: float, dt: float):
        if t >= self._next_kick:
            ang = self.rng.uniform(-math.pi, math.pi)
            policy.kick(self.kick_dv * np.array([math.cos(ang), math.sin(ang)]))
            self._next_kick = t + self.kick_interval_s * (0.8 + 0.4 * self.rng.random())
            self.events["kicks"] += 1
            self.event_log.append({"kind": "kick", "t": t, "end_t": t, "dv": self.kick_dv,
                                   "direction_rad": float(ang)})
        if self.stall_factor > 1.0:
            if self._stall_wheel is not None and t >= self._stall_until:
                policy.set_wheel_damping_factor(self._stall_wheel, 1.0)
                self._stall_wheel = None
            elif self._stall_wheel is None and self.rng.random() < self.stall_prob_per_s * dt:
                self._stall_wheel = int(self.rng.integers(0, 4))
                policy.set_wheel_damping_factor(self._stall_wheel, self.stall_factor)
                self._stall_until = t + self.stall_duration_s
                self.events["stalls"] += 1
                self.event_log.append({"kind": "wheel_stall", "t": t, "end_t": self._stall_until,
                                       "wheel": self._stall_wheel, "factor": self.stall_factor})

    def to_dict(self) -> dict:
        return {"kick_dv": self.kick_dv, "kick_interval_s": self.kick_interval_s, "stall_factor": self.stall_factor,
                "stall_prob_per_s": self.stall_prob_per_s, "stall_duration_s": self.stall_duration_s}


@dataclass
class Condition:
    """Physical conditions of one run."""
    payload_kg: float = 5.0
    caster_angle_deg: float = 0.0
    wheel_friction: float | None = None
    payload_com_offset_m: float = 0.0
    seed: int = 0

    def apply(self, policy):
        policy.reset(payload_kg=self.payload_kg, caster_angle=math.radians(self.caster_angle_deg), seed=self.seed,
                     wheel_friction=self.wheel_friction, com_offset_m=self.payload_com_offset_m)

    def to_dict(self) -> dict:
        return {"payload_kg": self.payload_kg, "caster_angle_deg": self.caster_angle_deg, "wheel_friction": self.wheel_friction,
                "payload_com_offset_m": self.payload_com_offset_m, "seed": self.seed}


# ---------------------------------------------------------------------------- runs

def _overload_thresholds(cfg: dict, payload_kg: float) -> dict | None:
    thr = cfg.get("overload_thresholds") or {}
    if not thr:
        return None
    return {**thr, "payload_kg": payload_kg}


def run_path(policy, path_local: Path, follower_cfg: FollowerConfig, cond: Condition, cfg: dict,
             disturbance: Disturbance | None = None, viewer=None, label: str = "") -> dict:
    """Drive ``policy`` along ``path_local`` (local frame; placed at the tracked body's pose after reset)."""
    cond.apply(policy)
    x0, y0, yaw0 = policy.pose()
    path = path_local.transformed((x0, y0), yaw0)
    follower = PathFollower(path, follower_cfg, policy.dt)
    dist = disturbance or Disturbance()
    dist.start()
    trace = Trace()
    if viewer is not None:
        viewer.begin_run(path, policy, label)
    wall = time.perf_counter()
    while True:
        st = follower.state
        if st.timed_out or (st.reached and st.t >= st.t_reached + follower_cfg.post_reach_s):
            break
        pose, vel = policy.pose(), policy.velocity()
        cmd = follower.update(pose, vel)
        dist.apply(policy, follower.state.t, policy.dt)
        policy.step(cmd)
        info = follower.info()
        trace.append(info["t"], policy.pose(), policy.velocity(), cmd, info, policy.telemetry())
        failure = policy.failure()
        if failure:
            trace.failure, trace.failure_t = failure, info["t"]
            break
        if viewer is not None:
            if hasattr(viewer, "sync_follower"):
                viewer.sync_follower(follower.cfg, path_length=follower.path.length)
            viewer.on_step(policy, follower, cmd, trace)
            if getattr(viewer, "want_rerun", False):
                break
    info = follower.info()
    metrics = path_metrics(trace, path, info, _overload_thresholds(cfg, cond.payload_kg))
    recovery = None
    if dist.active():
        from .metrics import disturbance_recovery_metrics
        recovery = disturbance_recovery_metrics(trace, dist.event_log, cfg.get("recovery", {}))
        metrics.update(recovery["metrics"])
    result = {
        "kind": "path", "label": label, "policy": policy.describe(), "path": path_local.name, "path_params": path_local.params,
        "path_length": path.length, "path_min_radius": path.min_radius, "condition": cond.to_dict(),
        "disturbance": dist.to_dict() if dist.active() else None, "disturbance_events": dist.events if dist.active() else None,
        "disturbance_event_log": dist.event_log, "recovery": recovery,
        "follower": follower_cfg.to_dict(), "metrics": metrics,
        "trace_dt_s": policy.dt * int(cfg.get("trace_stride", 5)),
        "trace": trace.decimated(int(cfg.get("trace_stride", 5))),
        "path_xy": path.xy[::10].round(4).tolist(), "goal": list(path.goal), "wall_s": time.perf_counter() - wall,
    }
    if viewer is not None:
        viewer.end_run(result)
    return result


def run_step(policy, step: dict, cond: Condition, cfg: dict, viewer=None, label: str = "") -> dict:
    """Open-loop velocity step ``cmd0 -> cmd1`` at ``pre_s``; step metrics on the tracked body's ``signal``."""
    cond.apply(policy)
    cmd0, cmd1 = np.asarray(step["cmd0"], float), np.asarray(step["cmd1"], float)
    pre_s, seconds = float(step.get("pre_s", 2.0)), float(step["seconds"])
    signal = step.get("signal", "vx")
    idx = {"vx": 0, "vy": 1, "wz": 2}[signal]
    trace = Trace()
    if viewer is not None:
        viewer.begin_run(None, policy, label)
    n = int(round(seconds / policy.dt))
    # Use integer control ticks: repeated floating-point addition delayed a 5 s step by one tick.
    switch_tick = max(0, int(math.ceil(pre_s / policy.dt - 1e-9)))
    switch_s = switch_tick * policy.dt
    for k in range(n):
        cmd = cmd0 if k < switch_tick else cmd1
        policy.step(cmd)
        t = (k + 1) * policy.dt
        trace.append(t, policy.pose(), policy.velocity(), cmd, None, policy.telemetry())
        failure = policy.failure()
        if failure:
            trace.failure, trace.failure_t = failure, t
            break
        if viewer is not None:
            viewer.on_step(policy, None, cmd, trace)
    m = summarize_step(trace, step, cond, cfg, switch_s=switch_s)
    result = {"kind": "step", "label": label, "policy": policy.describe(), "step": step, "condition": cond.to_dict(),
              "metrics_version": "v4", "metrics": m, "trace_dt_s": policy.dt * int(cfg.get("trace_stride", 2)),
              "trace": trace.decimated(int(cfg.get("trace_stride", 2)))}
    if viewer is not None:
        viewer.end_run(result)
    return result


def summarize_step(trace: Trace, step: dict, cond: Condition, cfg: dict, *, switch_s: float | None = None) -> dict:
    """Shared online/offline step evaluation; timestamps are post-physics sample times."""
    pre_s = float(step.get("pre_s", 2.0)) if switch_s is None else float(switch_s)
    signal = step.get("signal", "vx")
    idx = {"vx": 0, "vy": 1, "wz": 2}[signal]
    cmd0, cmd1 = np.asarray(step["cmd0"], float), np.asarray(step["cmd1"], float)
    y = [trace.vx, trace.vy, trace.wz][idx]
    smooth = float(cfg.get("defaults", {}).get("step_filter_s", 0.0))
    dwell = float(cfg.get("defaults", {}).get("settling_dwell_s", 0.5))
    tr = float(cfg.get("defaults", {}).get("transient_s", 1.5))
    m = step_response_metrics(trace.t, y, pre_s, float(cmd0[idx]), float(cmd1[idx]),
                              smooth_s=smooth, settling_dwell_s=dwell)
    m["command_switch_s"] = pre_s
    # The task intentionally tracks a 0.4 s first-order reference. Report its own envelope
    # separately so the raw-command response is not mistaken for high-level actuator latency.
    ref_series = trace.telemetry_series(f"cart_ref_{signal}")
    if np.isfinite(ref_series).any():
        ref_m = step_response_metrics(trace.t, ref_series, pre_s, float(cmd0[idx]), float(cmd1[idx]),
                                     smooth_s=0.0, settling_dwell_s=dwell)
        for key in ("rise_time", "settling_time", "settling_confirmed", "overshoot", "ss_error", "ss_std"):
            m[f"reference_{key}"] = ref_m.get(key)
        m.update(reference_tracking_metrics(trace.t, y, ref_series, pre_s, transient_s=tr))
    m["signal"] = signal
    m["filter_s"] = smooth
    m["failure"] = trace.failure
    speed = np.hypot(trace.vx, trace.vy)
    if step.get("stopping"):
        stop = stopping_metrics(trace.t, trace.x, trace.y, speed, pre_s,
                                dwell_s=float(cfg.get("defaults", {}).get("stop_dwell_s", 0.5)))
        m["stopping_distance"] = stop["confirmed_distance"]
        m["stopping_confirmed"] = stop["confirmed"]
        m["stopping_time"] = stop["confirmed_time"]
        m["first_low_distance"] = stop["first_low_distance"]
        m["post_stop_travel"] = stop["post_confirm_travel"]
        m["stop_rebound_count"] = stop["rebound_count"]
        m["stop_rebound_events"] = stop["rebound_events"]
        m["stop_invalid_samples"] = stop["invalid_samples"]
        m["first_low_time"] = stop["first_low_time"]
        m["stop_dwell_s"] = stop["dwell_s"]
        m["stop_v_threshold"] = stop["v_stop"]
        m["first_stop_confirmed_time"] = stop["first_confirmed_time"]
        m["first_stop_confirmed_distance"] = stop["first_confirmed_distance"]
        m["post_first_stop_travel"] = stop["post_first_confirm_travel"]
    if step.get("drift"):
        xy = np.stack([trace.x, trace.y], axis=1)
        m["drift_distance"] = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else None
        m["drift_speed_mean"] = float(np.mean(speed)) if len(speed) else None
        m["drift_yaw"] = float(abs(math.atan2(math.sin(trace.yaw[-1] - trace.yaw[0]), math.cos(trace.yaw[-1] - trace.yaw[0])))) if trace.yaw else None
    # command tracking after the transient
    tt = np.asarray(trace.t)
    sel = tt >= pre_s + tr
    if sel.any():
        err = np.asarray(y)[sel] - float(cmd1[idx])
        m["post_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        # cart smoothness after the transient: the ripple of the raw velocity with the bias removed (gait sway +
        # contact jitter + any slow oscillation); ``ss_std`` (metrics) is the slow part alone (0.4 s filtered)
        m["ripple_rms"] = float(np.std(err))
    from .metrics import interaction_metrics
    m.update(interaction_metrics(trace, _overload_thresholds(cfg, cond.payload_kg)))
    from .metrics import cart_quality_metrics
    m.update(cart_quality_metrics(trace, start_s=pre_s + tr,
                                  response_window=(pre_s, pre_s + tr)))
    return m
