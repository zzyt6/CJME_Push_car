"""Fit the baked Dex3 grasp to the real handle tube: fingers closed onto the tube until contact.

The tube is a ``Profile``: a round tube (radius) or a square tube (side, faces along the cart x / z axes, as a
tube welded to the cart frame). Distances are signed distances to the infinite prism around the bar axis.

V3.2 baked the planner's finger angles around a bar of 7.5 mm radius (the largest that did not overlap them).
The real tube is 22 mm across. Here the bar axis keeps the cart's direction (world y) and stays as close to the
V3.2 grasp point as the palm allows; each finger then closes (GraspIt!-style auto-grasp: joints move along their
closing direction until the link surface reaches the tube) so that palm and fingers rest on the tube surface with
a small clearance. Hand geometry: the Dex3 visual meshes of g1.urdf (the URDF has no hand colliders).

    fit_grasp() -> {"finger_pos": {...14...}, "point_in_palm": {left, right}, "diagnostics": {...}}
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from homie_pushcart.paths import FINGER_JOINTS, G1_DESCRIPTION_DIR, G1_URDF, PALM_LINKS
from homie_pushcart.pose import Kinematics, load_initial_pose

SIDES = ("left", "right")
CHAINS = {"thumb": ("thumb_0", "thumb_1", "thumb_2"), "index": ("index_0", "index_1"), "middle": ("middle_0", "middle_1")}
STEP_RAD = np.deg2rad(0.25)


def _origin(el):
    o = el.find("origin")
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
    rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


class HandModel:
    """Palm-frame forward kinematics and surface samples of one Dex3 hand from the original g1.urdf."""

    def __init__(self, side: str, samples_per_mesh: int = 30000):
        import trimesh
        root = ET.parse(G1_URDF).getroot()
        self.side = side
        self.joints = {}
        for j in root.findall("joint"):
            name = j.get("name")
            if name.startswith(f"{side}_hand_") and name != f"{side}_hand_palm_joint":
                lim = j.find("limit")
                self.joints[name] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                         T=_origin(j), axis=np.array([float(v) for v in j.find("axis").get("xyz").split()]),
                                         limit=(float(lim.get("lower")), float(lim.get("upper"))))
        self.points = {}
        rng = np.random.default_rng(0)
        for link in root.findall("link"):
            name = link.get("name")
            if not name.startswith(f"{side}_hand"):
                continue
            visual = link.find("visual")
            mesh_el = visual.find("geometry/mesh") if visual is not None else None
            if mesh_el is None:
                continue
            mesh = trimesh.load(G1_DESCRIPTION_DIR / mesh_el.get("filename"), force="mesh")
            surface, _ = trimesh.sample.sample_surface(mesh, samples_per_mesh, seed=int(rng.integers(1 << 31)))
            local = np.concatenate([np.asarray(mesh.vertices), np.asarray(surface)])
            self.points[name] = (_origin(visual) @ np.c_[local, np.ones(len(local))].T).T[:, :3]
        self.palm = PALM_LINKS[side]

    def link_T(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        """Palm-frame transform of every hand link for the given finger angles."""
        out = {self.palm: np.eye(4)}
        pending = dict(self.joints)
        while pending:
            for name, j in list(pending.items()):
                if j["parent"] in out:
                    rot = np.eye(4)
                    rot[:3, :3] = R.from_rotvec(j["axis"] / np.linalg.norm(j["axis"]) * angles.get(name, 0.)).as_matrix()
                    out[j["child"]] = out[j["parent"]] @ j["T"] @ rot
                    del pending[name]
        return out

    def link_points(self, angles, links=None):
        T = self.link_T(angles)
        return {l: (T[l] @ np.c_[p, np.ones(len(p))].T).T[:, :3] for l, p in self.points.items() if links is None or l in links}


def axis_distance(points: np.ndarray, center: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = points - center
    return np.linalg.norm(d - np.outer(d @ u, u), axis=1)


class Profile:
    """Tube cross-section in the plane normal to the bar. ``e1``/``e2``: cart x / z in the same frame as u."""

    def __init__(self, kind: str, size: float, u, e1=None, e2=None):
        self.kind, self.size = kind, float(size)                  # radius (round) or side (square)
        self.u = np.asarray(u, float) / np.linalg.norm(u)
        if kind == "square":
            e1 = np.asarray(e1, float) - (np.asarray(e1, float) @ self.u) * self.u
            self.e1 = e1 / np.linalg.norm(e1)
            self.e2 = np.cross(self.u, self.e1)
            if e2 is not None and np.dot(self.e2, e2) < 0:
                self.e2 = -self.e2
        elif kind != "round":
            raise ValueError(kind)

    def gap(self, points, center):
        """Signed distance from the tube surface (positive outside)."""
        if self.kind == "round":
            return axis_distance(points, center, self.u) - self.size
        d = points - center
        q = np.abs(np.c_[d @ self.e1, d @ self.e2]) - self.size / 2
        outside = np.linalg.norm(np.maximum(q, 0.), axis=1)
        return outside + np.minimum(q.max(axis=1), 0.)

    def gap_gradient_dir(self, point, center):
        """Unit direction (normal to u) that increases the gap at ``point`` when the centre moves along it."""
        if self.kind == "round":
            d = center - point
            n = d - (d @ self.u) * self.u
            return n / np.linalg.norm(n)
        eps = 1e-5
        g0 = self.gap(point[None], center)[0]
        basis = [self.e1, self.e2]
        grad = np.array([(self.gap(point[None], center + eps * b)[0] - g0) / eps for b in basis])
        v = grad[0] * basis[0] + grad[1] * basis[1]
        return v / np.linalg.norm(v)

    def outline(self):
        if self.kind == "round":
            t = np.linspace(0, 2 * np.pi, 200)
            return self.size * np.cos(t), self.size * np.sin(t)
        h = self.size / 2
        return np.array([-h, h, h, -h, -h]), np.array([-h, -h, h, h, -h])


def place_axis(palm_points, start, profile: Profile, clearance, iters=400):
    """Move the axis (normal to the bar only) the least distance that clears the palm by ``clearance``."""
    p = np.array(start, dtype=float)
    for _ in range(iters):
        gap = profile.gap(palm_points, p)
        i = int(np.argmin(gap))
        if gap[i] >= clearance - 1e-6:
            return p, float(gap[i])
        p = p + profile.gap_gradient_dir(palm_points[i], p) * (clearance - gap[i] + 1e-5)
    raise RuntimeError("axis placement did not converge")


def close_finger(hand: HandModel, angles: dict, joints: list[str], links: list[str], center, profile: Profile,
                 target, scaled: bool):
    """Move ``joints`` along their closing direction until any of ``links`` reaches the tube (or a limit)."""
    def clear(a):
        pts = np.concatenate(list(hand.link_points(a, links).values()))
        return profile.gap(pts, center).min()

    start = {j: angles[j] for j in joints}
    sign = {j: np.sign(start[j]) or (np.sign(hand.joints[j]["limit"][1]) or -1.) for j in joints}
    if scaled:                                        # common scale s on the planner angles: keeps the posture
        def at(s):
            return {**angles, **{j: float(np.clip(s * start[j], *hand.joints[j]["limit"])) for j in joints}}
        s = 1.0
        while clear(at(s)) < target and s > 0:        # opens until clear ...
            s -= 0.002
        while clear(at(s + 0.002)) >= target and s < 2.0 and at(s + 0.002) != at(s):   # ... then closes to contact
            s += 0.002
        return at(s)
    j = joints[0]
    a = dict(angles)
    lo, hi = hand.joints[j]["limit"]
    while clear(a) < target and abs(a[j]) > 0:
        a[j] = float(np.clip(a[j] - sign[j] * STEP_RAD, lo, hi))
    while True:
        nxt = dict(a)
        nxt[j] = float(np.clip(a[j] + sign[j] * STEP_RAD, lo, hi))
        if nxt[j] == a[j] or clear(nxt) < target:
            return a
        a = nxt


def fit_grasp(kind: str = "round", size: float = 0.011, clearance: float = 0.0005, pose: dict | None = None) -> dict:
    """``kind`` round: ``size`` = radius; square: ``size`` = side (faces along the cart x / z axes)."""
    pose = pose or load_initial_pose()
    kin = Kinematics(load_meshes=False)
    kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])
    bar_world = np.asarray(pose["grasp"]["bar_axis_world"], dtype=float)
    out = {"kind": kind, "size": float(size), "finger_pos": dict(pose["robot"]["finger_pos"]), "point_in_palm": {},
           "diagnostics": {}}
    for side in SIDES:
        hand = HandModel(side)
        Rp = kin.world_T(PALM_LINKS[side])[:3, :3].T
        profile = Profile(kind, size, Rp @ bar_world, Rp @ np.array([1., 0., 0.]), Rp @ np.array([0., 0., 1.]))
        angles = {j: float(v) for j, v in pose["robot"]["finger_pos"].items() if j.startswith(side)}
        p0 = np.asarray(pose["grasp"]["point_in_palm"][side], dtype=float)
        before = {l: float(profile.gap(p, p0).min()) for l, p in hand.link_points(angles).items()}
        static = [hand.palm, f"{side}_hand_thumb_0_link"]          # thumb_0 (abduction) stays at the planner value
        palm_pts = np.concatenate(list(hand.link_points(angles, static).values()))
        center, palm_gap = place_axis(palm_pts, p0, profile, clearance)
        for finger, names in CHAINS.items():
            joints = [f"{side}_hand_{n}_joint" for n in names if n != "thumb_0"]
            links = [f"{side}_hand_{n}_link" for n in names if n != "thumb_0"]
            angles = close_finger(hand, angles, joints, links, center, profile, clearance, scaled=True)
            angles = close_finger(hand, angles, joints[-1:], links[-1:], center, profile, clearance, scaled=False)
        after = {l: float(profile.gap(p, center).min()) for l, p in hand.link_points(angles).items()}
        change = {j: float(angles[j] - pose["robot"]["finger_pos"][j]) for j in angles}
        out["finger_pos"].update(angles)
        out["point_in_palm"][side] = center.tolist()
        out["diagnostics"][side] = {
            "bar_axis_in_palm": profile.u.tolist(), "e1_in_palm": getattr(profile, "e1", np.zeros(3)).tolist(),
            "axis_shift_m": float(np.linalg.norm(center - p0)), "palm_clearance_m": float(palm_gap),
            "surface_gap_before_m": before, "surface_gap_after_m": after,
            "links_in_contact": [l for l, g in after.items() if g < 0.0015],
            "finger_angle_change_rad": change,
            "finger_angle_change_abs_sum_rad": float(sum(abs(v) for v in change.values())),
        }
    missing = [j for j in FINGER_JOINTS if j not in out["finger_pos"]]
    if missing:
        raise KeyError(missing)
    return out


def sweep_square(sides_m=(0.016, 0.019, 0.020, 0.022, 0.025), clearance=0.0005) -> list[dict]:
    """Commercial square-tube sizes: how well does the closed hand wrap each one (contacts, finger change)."""
    rows = []
    for a in sides_m:
        fit = fit_grasp("square", a, clearance)
        d = fit["diagnostics"]
        rows.append({"side_m": a, "contacts": {s: len(d[s]["links_in_contact"]) for s in SIDES},
                     "axis_shift_m": {s: d[s]["axis_shift_m"] for s in SIDES},
                     "finger_change_abs_sum_rad": {s: d[s]["finger_angle_change_abs_sum_rad"] for s in SIDES},
                     "min_gap_m": {s: min(d[s]["surface_gap_after_m"].values()) for s in SIDES}, "fit": fit})
    return rows


def section_figure(fit: dict, path: Path, pose: dict | None = None):
    """Hand surface within +-4 cm of the grasp point, projected on the plane normal to the bar, plus the tube."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pose = pose or load_initial_pose()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for k, side in enumerate(SIDES):
        hand = HandModel(side, samples_per_mesh=8000)
        dg = fit["diagnostics"][side]
        profile = Profile(fit["kind"], fit["size"], dg["bar_axis_in_palm"],
                          dg["e1_in_palm"] if fit["kind"] == "square" else None)
        u = profile.u
        if fit["kind"] == "square":
            e1 = profile.e1
        else:
            e1 = np.cross(u, [0., 0., 1.])
            e1 /= np.linalg.norm(e1)
        e2 = np.cross(e1, u)          # cart forward x cart left (bar) = cart up
        for col, (label, angles, center) in enumerate((
                ("V3.2 baked angles", {j: v for j, v in pose["robot"]["finger_pos"].items() if j.startswith(side)},
                 np.asarray(pose["grasp"]["point_in_palm"][side])),
                ("V7 fitted", {j: v for j, v in fit["finger_pos"].items() if j.startswith(side)},
                 np.asarray(fit["point_in_palm"][side])))):
            ax = axes[2 * k + col]
            for link, pts in hand.link_points(angles).items():
                d = pts - center
                sel = np.abs(d @ u) < 0.04
                ax.scatter(d[sel] @ e1 * 1e3, d[sel] @ e2 * 1e3, s=0.3, label=link.replace(f"{side}_hand_", ""))
            x, y = profile.outline()
            ax.plot(x * 1e3, y * 1e3, "k-", lw=2)
            ax.set_aspect("equal")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            if fit["kind"] == "square":
                ax.set_title(f"{side}: {label} ({fit['size'] * 1e3:.0f} mm square)\nx = cart forward, y = cart up")
            else:
                ax.set_title(f"{side}: {label} (r={fit['size'] * 1e3:.0f} mm)")
            ax.set_xlabel("mm")
        axes[2 * k].legend(markerscale=15, fontsize=7, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
