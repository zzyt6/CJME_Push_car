"""Fixed, representative reference paths for the path-following experiment.

Every path is generated in a local frame (start at the origin, heading +x), resampled at a uniform arc-length
step with tangent heading and signed curvature (left turn positive), then transformed onto the tracked body's
initial pose. The last sample is the goal.

    straight      straight line
    s_curve       windowed sine  y = A sin(2 pi x / lambda) sin^2(pi x / L): zero lateral offset and zero heading
                  at both ends, alternating left / right turns in between
    right_angle   straight -> 90 deg circular arc -> straight (left turn; mirror for right)
    u_turn        straight -> 180 deg semicircle -> straight back (left; mirror for right)
    composite_course  wide sinusoidal S -> 2 m connector -> raised rectangular detour with rounded corners;
                      total arc length 20 m, start and finish on the same forward axis and heading
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


@dataclass
class Path:
    name: str
    xy: np.ndarray                 # (N, 2) world frame
    yaw: np.ndarray                # (N,) tangent heading [rad]
    s: np.ndarray                  # (N,) arc length [m], s[0] = 0
    curvature: np.ndarray          # (N,) signed [1/m], left positive
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ properties
    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def goal(self) -> tuple[float, float, float]:
        return float(self.xy[-1, 0]), float(self.xy[-1, 1]), float(self.yaw[-1])

    @property
    def min_radius(self) -> float:
        k = float(np.max(np.abs(self.curvature))) if len(self.curvature) else 0.0
        return float("inf") if k < 1e-9 else 1.0 / k

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0]) if len(self.s) > 1 else 0.0

    # ------------------------------------------------------------------ transforms
    def transformed(self, origin_xy, origin_yaw: float) -> "Path":
        """The same geometry starting at ``origin_xy`` with initial heading ``origin_yaw``."""
        c, s = math.cos(origin_yaw), math.sin(origin_yaw)
        rot = np.array([[c, -s], [s, c]])
        xy = self.xy @ rot.T + np.asarray(origin_xy, dtype=float)[None, :]
        return Path(self.name, xy, wrap_angle(self.yaw + origin_yaw), self.s.copy(), self.curvature.copy(),
                    {**self.params, "origin_xy": [float(v) for v in origin_xy], "origin_yaw": float(origin_yaw)})

    def mirrored(self) -> "Path":
        """Flip the local y axis (left turns become right turns). Only meaningful before ``transformed``."""
        xy = self.xy.copy()
        xy[:, 1] *= -1.0
        return Path(self.name + "_mirror", xy, wrap_angle(-self.yaw), self.s.copy(), -self.curvature, {**self.params, "mirror": True})

    # ------------------------------------------------------------------ queries
    def point_at(self, s: float) -> tuple[np.ndarray, float, float]:
        """Interpolated (xy, yaw, curvature) at arc length ``s`` (clamped to the path)."""
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.s, s, side="right")) - 1
        i = max(0, min(i, len(self.s) - 2))
        w = 0.0 if self.s[i + 1] == self.s[i] else (s - self.s[i]) / (self.s[i + 1] - self.s[i])
        xy = (1 - w) * self.xy[i] + w * self.xy[i + 1]
        dyaw = wrap_angle(self.yaw[i + 1] - self.yaw[i])
        yaw = float(wrap_angle(self.yaw[i] + w * dyaw))
        curv = float((1 - w) * self.curvature[i] + w * self.curvature[i + 1])
        return xy, yaw, curv

    def nearest_index(self, xy, start: int = 0, window_m: float | None = None) -> int:
        """Index of the closest sample, searched forward from ``start`` over ``window_m`` of arc length.

        The forward window keeps the progress monotonic on paths that come back near themselves (u_turn).
        """
        start = max(0, min(int(start), len(self.s) - 1))
        if window_m is None:
            stop = len(self.s)
        else:
            stop = min(len(self.s), start + int(math.ceil(window_m / max(self.ds, 1e-9))) + 1)
        seg = self.xy[start:stop]
        d = np.linalg.norm(seg - np.asarray(xy, dtype=float)[None, :], axis=1)
        return start + int(np.argmin(d))

    def cross_track(self, xy, index: int) -> float:
        """Signed lateral offset of ``xy`` from the path at ``index`` (left of the path positive)."""
        t = np.array([math.cos(self.yaw[index]), math.sin(self.yaw[index])])
        d = np.asarray(xy, dtype=float) - self.xy[index]
        return float(t[0] * d[1] - t[1] * d[0])

    def extended(self, extra_m: float) -> "Path":
        """The path continued straight past the goal by ``extra_m`` (keeps a lookahead point ahead near the goal)."""
        if extra_m <= 0:
            return self
        gx, gy, gyaw = self.goal
        n = int(math.ceil(extra_m / max(self.ds, 1e-9)))
        k = np.arange(1, n + 1) * self.ds
        ext_xy = np.stack([gx + k * math.cos(gyaw), gy + k * math.sin(gyaw)], axis=1)
        return Path(self.name, np.concatenate([self.xy, ext_xy]), np.concatenate([self.yaw, np.full(n, gyaw)]),
                    np.concatenate([self.s, self.s[-1] + k]), np.concatenate([self.curvature, np.zeros(n)]), dict(self.params))

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self, stride: int = 1) -> dict:
        return {"name": self.name, "params": self.params, "length": self.length, "min_radius": self.min_radius,
                "goal": list(self.goal), "xy": self.xy[::stride].round(4).tolist(), "yaw": self.yaw[::stride].round(4).tolist(),
                "s": self.s[::stride].round(4).tolist()}


# ---------------------------------------------------------------------------- construction helpers

def resample(points: np.ndarray, ds: float, name: str, params: dict | None = None) -> Path:
    """Uniform arc-length resampling of a dense polyline; heading from central differences, curvature = dyaw/ds."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    pts, seg = pts[keep], seg[seg > 1e-12]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(cum[-1])
    n = max(2, int(round(length / ds)) + 1)
    s = np.linspace(0.0, length, n)
    xy = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)
    d = np.gradient(xy, s, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    dyaw = wrap_angle(np.diff(yaw))
    unwrapped = np.concatenate([[yaw[0]], yaw[0] + np.cumsum(dyaw)])
    curvature = np.gradient(unwrapped, s)
    return Path(name, xy, wrap_angle(unwrapped), s, curvature, dict(params or {}))


def _arc(center, radius, phi0, phi1, n):
    phi = np.linspace(phi0, phi1, n)
    return np.stack([center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)], axis=1)


def _dense_line(p0, p1, n):
    return np.linspace(np.asarray(p0, float), np.asarray(p1, float), n)


def straight(length: float = 6.0, ds: float = 0.02) -> Path:
    pts = _dense_line((0.0, 0.0), (length, 0.0), 200)
    return resample(pts, ds, "straight", {"type": "straight", "length": length})


def s_curve(length: float = 12.0, amplitude: float = 1.0, periods: float = 2.0, ds: float = 0.02) -> Path:
    x = np.linspace(0.0, length, 4000)
    lam = length / periods
    y = amplitude * np.sin(2 * math.pi * x / lam) * np.sin(math.pi * x / length) ** 2
    return resample(np.stack([x, y], axis=1), ds, "s_curve",
                    {"type": "s_curve", "length": length, "amplitude": amplitude, "periods": periods})


def right_angle(l1: float = 4.0, l2: float = 4.0, radius: float = 0.6, ds: float = 0.02) -> Path:
    """Straight l1 along +x, quarter circle of ``radius`` turning left, straight l2 along +y."""
    a = _dense_line((0.0, 0.0), (l1, 0.0), 400)
    arc = _arc((l1, radius), radius, -math.pi / 2, 0.0, 400)[1:]
    b = _dense_line((l1 + radius, radius), (l1 + radius, radius + l2), 400)[1:]
    return resample(np.concatenate([a, arc, b]), ds, "right_angle",
                    {"type": "right_angle", "l1": l1, "l2": l2, "radius": radius})


def u_turn(l_out: float = 3.0, radius: float = 1.0, l_back: float = 3.0, ds: float = 0.02) -> Path:
    """Straight l_out along +x, semicircle turning left, straight l_back along -x (parallel, offset 2 radius)."""
    a = _dense_line((0.0, 0.0), (l_out, 0.0), 400)
    arc = _arc((l_out, radius), radius, -math.pi / 2, math.pi / 2, 800)[1:]
    b = _dense_line((l_out, 2 * radius), (l_out - l_back, 2 * radius), 400)[1:]
    return resample(np.concatenate([a, arc, b]), ds, "u_turn",
                    {"type": "u_turn", "l_out": l_out, "radius": radius, "l_back": l_back})


def s_bend(radius: float = 0.6, lead: float = 1.0, ds: float = 0.02) -> Path:
    """Driving-test S: straight lead -> 90 deg left arc -> 180 deg right arc -> 90 deg left arc -> straight lead.

    Constant curvature +-1/radius through the bends (no easing), lateral bulge 2 * radius, ends aligned with the
    start heading. With radius 0.6 m the cart (1.2 m long) has to be steered at its geometric limit.
    """
    r = radius
    a = _dense_line((0.0, 0.0), (lead, 0.0), 200)
    arc1 = _arc((lead, r), r, -math.pi / 2, 0.0, 300)[1:]                                   # left, heading +x -> +y
    arc2 = _arc((lead + 2 * r, r), r, math.pi, 0.0, 600)[1:]                                # right (clockwise), +y -> -y
    arc3 = _arc((lead + 4 * r, r), r, math.pi, 3 * math.pi / 2, 300)[1:]                    # left, -y -> +x
    b = _dense_line((lead + 4 * r, 0.0), (2 * lead + 4 * r, 0.0), 200)[1:]
    return resample(np.concatenate([a, arc1, arc2, arc3, b]), ds, "s_bend", {"type": "s_bend", "radius": radius, "lead": lead})


def composite_course(total_length: float = 20.0, lead: float = 0.3, s_length: float = 8.0,
                     amplitude: float = 0.7, middle: float = 2.0, corner_radius: float = 1.25,
                     height: float = 2.7, finish: float = 0.3, ds: float = 0.02) -> Path:
    """A full sinusoidal positive/negative S followed by a raised rectangular detour.

    A cosine ramp of the sine phase makes both ends horizontal with zero curvature;
    the two sine extrema retain the requested +/- amplitude. Four rounded 90-degree
    corners raise the path, continue forward, then return it to y=0, as in the sketch.
    The top straight absorbs the length budget, without shrinking either turning
    radius or the separate ``middle`` connector. ``total_length`` is route arc length,
    not the horizontal start-to-finish distance.
    """
    values = (total_length, lead, s_length, amplitude, middle, corner_radius, height, finish, ds)
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError("Composite-course dimensions and sample spacing must be finite and positive")
    if height <= 2.0 * corner_radius:
        raise ValueError("height must exceed twice corner_radius to retain the vertical straight sections")
    r = corner_radius
    u = np.linspace(0.0, 1.0, 8001)
    ramp = 0.2
    phase_speed = 1.0 / (1.0 - ramp)
    phase = np.where(u < ramp, phase_speed / 2 * (u - ramp / math.pi * np.sin(math.pi * u / ramp)),
                     np.where(u > 1.0 - ramp,
                              1.0 - phase_speed / 2 * ((1.0 - u) - ramp / math.pi * np.sin(math.pi * (1.0 - u) / ramp)),
                              phase_speed * (u - ramp / 2)))
    sine_y = amplitude * np.sin(2 * math.pi * phase)
    sine_y[[0, -1]] = 0.0
    sine = np.stack((lead + s_length * u, sine_y), axis=1)
    corner_x = lead + s_length + middle

    def points(top):
        return np.concatenate([
            _dense_line((0., 0.), (lead, 0.), 200),
            sine[1:],
            _dense_line((lead + s_length, 0.), (corner_x, 0.), 300)[1:],
            _arc((corner_x, r), r, -math.pi / 2, 0., 1000)[1:],
            _dense_line((corner_x + r, r), (corner_x + r, height - r), 200)[1:],
            _arc((corner_x + 2 * r, height - r), r, math.pi, math.pi / 2, 1000)[1:],
            _dense_line((corner_x + 2 * r, height), (corner_x + 2 * r + top, height), 200)[1:],
            _arc((corner_x + 2 * r + top, height - r), r, math.pi / 2, 0., 1000)[1:],
            _dense_line((corner_x + 3 * r + top, height - r), (corner_x + 3 * r + top, r), 200)[1:],
            _arc((corner_x + 4 * r + top, r), r, math.pi, 3 * math.pi / 2, 1000)[1:],
            _dense_line((corner_x + 4 * r + top, 0.), (corner_x + 4 * r + top + finish, 0.), 200)[1:],
        ])

    fixed = points(0.)
    fixed_length = float(np.linalg.norm(np.diff(fixed, axis=0), axis=1).sum())
    top = total_length - fixed_length
    if top <= 0.0:
        raise ValueError(f"total_length is too short for these bends and connector; must exceed {fixed_length:.3f} m")
    return resample(points(top), ds, "composite_course", {
        "type": "composite_course", "total_length": total_length, "lead": lead, "s_length": s_length,
        "amplitude": amplitude, "middle": middle, "corner_radius": r, "height": height,
        "finish": finish, "ds": ds,
    })


GENERATORS = {"straight": straight, "s_curve": s_curve, "s_bend": s_bend, "right_angle": right_angle,
              "u_turn": u_turn, "composite_course": composite_course}


def build_path(name: str, spec: dict, mirror: bool = False) -> Path:
    """Path from a config entry ``{type: ..., <params>}``; ``mirror`` flips left / right turns."""
    spec = dict(spec)
    kind = spec.pop("type", name)
    if kind not in GENERATORS:
        raise KeyError(f"unknown path type {kind!r}; available: {sorted(GENERATORS)}")
    path = GENERATORS[kind](**spec)
    path.name = name
    path.params = {"type": kind, **spec}
    return path.mirrored() if mirror else path
