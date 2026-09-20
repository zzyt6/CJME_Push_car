"""The shared high-level path-following controller: same gains for every policy under test.

Pure pursuit with a speed-dependent lookahead, a PID on the bearing to the lookahead point for the yaw rate,
a speed profile that (unless ``cruise``) slows on curvature / misalignment and approaches the goal
proportionally, output clipping to the task-command ranges and a rate limit. Two modes:

    car         vy = 0; the yaw-rate PID acts on the bearing to the lookahead point (a pushed cart is car-like)
    holonomic   vy from the lateral offset of the lookahead point in the body frame; the yaw-rate PID aligns the
                heading with the path tangent (uses the lateral command channel of the policy)

The controller sees only the tracked body's pose and velocity, so it is identical for a cart policy (cart base
frame) and the robot-only reference (pelvis frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .paths import Path, wrap_angle


@dataclass
class FollowerConfig:
    mode: str = "car"                                 # car | holonomic
    v_nom: float = 0.5                                # commanded cruise speed [m/s]
    cruise: bool = False                              # hold v_nom through curves (no curvature / remaining slowdown)
    lookahead_base: float = 0.4                       # L_d = clip(base + gain * |v|, min, max) [m]
    lookahead_gain: float = 0.6
    lookahead_min: float = 0.4
    lookahead_max: float = 1.0
    kp_yaw: float = 1.5                               # PID on the bearing to the lookahead point [rad/s per rad]
    ki_yaw: float = 0.1
    kd_yaw: float = 0.05
    i_limit: float = 0.3
    k_curv: float = 0.6                               # v = v_nom * max(v_min_frac, 1 - k_curv * |kappa(lookahead)|)
    v_min_frac: float = 0.35
    kp_goal: float = 1.0                              # approach speed = kp_goal * distance to goal ...
    v_near: float = 0.25                              # ... capped here (absorbs the policy's lag) ...
    v_min_cmd: float = 0.1                            # ... and never below this (legged policies dither on tiny commands)
    t_lag: float = 0.6                                # response lag assumed for the deceleration (distance - speed * t_lag)
    near_dist: float = 0.6                            # within this arc length of the goal: home on the goal point directly
    kp_lat: float = 1.0                               # holonomic: vy = kp_lat * lateral offset of the lookahead point
    goal_radius: float = 0.10                         # reached when inside and slower than goal_speed for goal_hold_s
    goal_speed: float = 0.06
    goal_hold_s: float = 0.5
    post_reach_s: float = 1.0                         # zero command held after reaching (measures the settled error / drift)
    timeout_s: float = 60.0
    nearest_window_m: float = 3.0                     # forward search window for the progress point
    cmd_limits: tuple = ((-0.6, 0.9), (-0.35, 0.35), (-0.6, 0.6))    # task-command ranges (75 % of HOMIE's)
    rate_limits: tuple = (1.0, 1.0, 2.0)              # max |d cmd / dt| per component

    @classmethod
    def from_dict(cls, d: dict) -> "FollowerConfig":
        d = dict(d)
        if "cmd_limits" in d:
            d["cmd_limits"] = tuple(tuple(float(v) for v in pair) for pair in d["cmd_limits"])
        if "rate_limits" in d:
            d["rate_limits"] = tuple(float(v) for v in d["rate_limits"])
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: (list(list(p) for p in v) if k == "cmd_limits" else (list(v) if k == "rate_limits" else v))
                for k, v in self.__dict__.items()}


@dataclass
class FollowerState:
    t: float = 0.0
    index: int = 0
    s: float = 0.0
    remaining: float = 0.0
    cross_track: float = 0.0
    heading_err: float = 0.0
    bearing: float = 0.0
    lookahead_xy: tuple = (0.0, 0.0)
    lookahead_dist: float = 0.0
    dist_goal: float = 0.0
    curvature: float = 0.0
    inside: bool = False
    t_enter: float | None = None
    t_reached: float | None = None
    reached: bool = False
    timed_out: bool = False
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def done(self) -> bool:
        return self.reached or self.timed_out


class PathFollower:
    def __init__(self, path: Path, cfg: FollowerConfig, dt: float):
        if cfg.mode not in ("car", "holonomic"):
            raise ValueError(f"unknown follower mode {cfg.mode!r}")
        self.path, self.cfg, self.dt = path, cfg, dt
        # progress / lookahead are evaluated on the path continued straight past the goal, so the lookahead point
        # never collapses onto the body (pure-pursuit singularity) and "past the goal" is a signed arc length
        self.ext = path.extended(cfg.lookahead_max + 1.0)
        self.goal_s = path.length
        self.reset()

    def reset(self):
        self.state = FollowerState()
        self._i_term = 0.0
        self._prev_err = None
        self._prev_cmd = np.zeros(3)
        self._recent = []                                                     # (t, x, y, inside) over the hold window
        self._homing_dir = None                                               # +1 forward / -1 reverse during the final approach

    # ------------------------------------------------------------------ one control step
    def update(self, pose, vel) -> np.ndarray:
        """``pose`` = (x, y, yaw) of the tracked body in the world; ``vel`` = (vx, vy, wz) in its own frame."""
        cfg, path, st = self.cfg, self.path, self.state
        x, y, yaw = (float(v) for v in pose)
        vx, vy, wz = (float(v) for v in vel)
        speed = math.hypot(vx, vy)
        pos = np.array([x, y])

        # progress along the (extended) path with a monotonic search window, and the tracking errors
        ext = self.ext
        st.index = ext.nearest_index(pos, start=st.index, window_m=cfg.nearest_window_m)
        st.s = float(min(ext.s[st.index], self.goal_s))
        st.remaining = self.goal_s - float(ext.s[st.index])                 # signed: negative once past the goal
        st.cross_track = ext.cross_track(pos, st.index)
        st.heading_err = float(wrap_angle(yaw - ext.yaw[st.index]))
        gx, gy, gyaw = path.goal
        st.dist_goal = float(math.hypot(gx - x, gy - y))

        # lookahead point and its bearing in the body frame
        st.lookahead_dist = float(np.clip(cfg.lookahead_base + cfg.lookahead_gain * speed, cfg.lookahead_min, cfg.lookahead_max))
        la_xy, la_yaw, st.curvature = ext.point_at(float(ext.s[st.index]) + st.lookahead_dist)
        st.lookahead_xy = (float(la_xy[0]), float(la_xy[1]))
        d = la_xy - pos
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx_b, dy_b = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
        st.bearing = math.atan2(dy_b, dx_b)

        # goal logic: reached = inside the radius for the whole hold window with a mean speed (from the position
        # change over the window, robust to the standing sway of a legged robot) below goal_speed
        st.inside = st.dist_goal < cfg.goal_radius
        if st.inside and st.t_enter is None:
            st.t_enter = st.t
        self._recent.append((st.t, x, y, st.inside))
        while self._recent and st.t - self._recent[0][0] > cfg.goal_hold_s + 0.5 * self.dt:
            self._recent.pop(0)
        window_ok = st.t - self._recent[0][0] >= cfg.goal_hold_s - 0.5 * self.dt and all(r[3] for r in self._recent)
        mean_speed = math.hypot(x - self._recent[0][1], y - self._recent[0][2]) / max(st.t - self._recent[0][0], 1e-9)
        if window_ok and mean_speed < cfg.goal_speed and not st.reached:
            st.reached, st.t_reached = True, st.t
        if st.t >= cfg.timeout_s and not st.reached:
            st.timed_out = True

        dock = 0.35 if cfg.cruise else cfg.near_dist
        if st.done or st.inside:
            target = np.zeros(3)                                                 # settle inside the radius / hold after
        elif st.remaining < dock:
            # final approach and recovery: home on the goal point itself (handles overshoot and lateral drift;
            # pure pursuit along the path cannot correct a lateral error near its end). Forward if the goal is in
            # front, otherwise back up with the rear axis aligned to the goal (unicycle: wz = k * angle error).
            gdx, gdy = gx - x, gy - y
            gdx_b, gdy_b = c * gdx + s_ * gdy, -s_ * gdx + c * gdy
            beta = math.atan2(gdy_b, gdx_b)
            speed_mag = float(np.clip(cfg.kp_goal * max(st.dist_goal - speed * cfg.t_lag, 0.0), cfg.v_min_cmd, cfg.v_near))
            # car-like docking with hysteresis (no chatter when the goal sits beside the body):
            #   forward  - goal ahead (|beta| small): drive at it
            #   back up  - goal beside / behind: reverse to gain a runway while turning the nose towards it; if the
            #              goal is nearly straight behind, align the rear axis instead and dock in reverse
            if self._homing_dir is None:
                self._homing_dir = 1 if abs(beta) <= math.radians(75) else -1
            elif self._homing_dir > 0 and abs(beta) > math.radians(75):
                self._homing_dir = -1
            elif self._homing_dir < 0 and abs(beta) < math.radians(30):
                self._homing_dir = 1
            if self._homing_dir > 0:
                v_cmd, err = speed_mag, beta
            else:
                v_cmd = -max(cfg.v_min_cmd, min(speed_mag, 0.15))
                err = float(wrap_angle(beta - math.pi)) if abs(beta) >= math.radians(120) else beta
            wz_cmd = self._pid(err)
            vy_cmd = float(np.clip(cfg.kp_lat * gdy_b, -cfg.v_near, cfg.v_near)) if cfg.mode == "holonomic" else 0.0
            target = np.array([v_cmd, vy_cmd, wz_cmd])
        else:
            err = st.bearing if cfg.mode == "car" else -st.heading_err
            wz_cmd = self._pid(err)
            heading_scale = max(0.0, math.cos(st.bearing))
            if cfg.cruise:
                v_cmd = cfg.v_nom * heading_scale
            else:
                slow = max(cfg.v_min_frac, 1.0 - cfg.k_curv * abs(st.curvature))
                v_track = cfg.v_nom * slow * heading_scale
                v_cmd = min(v_track, max(cfg.kp_goal * (st.remaining - speed * cfg.t_lag), cfg.v_min_cmd))
            vy_cmd = cfg.kp_lat * dy_b if cfg.mode == "holonomic" else 0.0
            target = np.array([v_cmd, vy_cmd, wz_cmd])

        # rate limit then clip
        lim = np.asarray(cfg.rate_limits) * self.dt
        cmd = self._prev_cmd + np.clip(target - self._prev_cmd, -lim, lim)
        lo = np.array([p[0] for p in cfg.cmd_limits])
        hi = np.array([p[1] for p in cfg.cmd_limits])
        cmd = np.clip(cmd, lo, hi)
        self._prev_cmd = cmd
        st.cmd = cmd.copy()
        st.t += self.dt
        return cmd

    def _pid(self, err: float) -> float:
        cfg = self.cfg
        self._i_term = float(np.clip(self._i_term + err * self.dt, -cfg.i_limit, cfg.i_limit))
        d_err = 0.0 if self._prev_err is None else (err - self._prev_err) / self.dt
        self._prev_err = err
        return cfg.kp_yaw * err + cfg.ki_yaw * self._i_term + cfg.kd_yaw * d_err

    def info(self) -> dict:
        st = self.state
        return {"t": st.t, "s": st.s, "remaining": st.remaining, "cross_track": st.cross_track, "heading_err": st.heading_err,
                "bearing": st.bearing, "lookahead_xy": list(st.lookahead_xy), "lookahead_dist": st.lookahead_dist,
                "dist_goal": st.dist_goal, "curvature": st.curvature, "inside": st.inside, "reached": st.reached,
                "timed_out": st.timed_out, "t_enter": st.t_enter, "t_reached": st.t_reached}
