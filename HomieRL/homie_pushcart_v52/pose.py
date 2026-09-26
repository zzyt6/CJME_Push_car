"""Initial grasp pose: import from the planner trajectory, store as YAML, evaluate with yourdfpy.

No custom kinematics is implemented here; link poses come from ``yourdfpy``,
the same library viser uses for rendering.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from .paths import (ARM_JOINTS, FINGER_JOINTS, G1_GRASP_URDF, HOMIE_JOINTS, INITIAL_POSE_YAML, LEG_JOINTS,
                    PALM_LINKS, ROOT, WAIST_JOINTS)

# HOMIE treats the sole as 2 cm below the ankle_roll link origin (cfg.asset.ankle_sole_distance).
ANKLE_SOLE_DISTANCE = 0.02
FEET_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def homie_default_joint_angles() -> dict[str, float]:
    """Read HOMIE's standing pose without importing any simulator module."""
    if str(ROOT / "HomieRL") not in sys.path:
        sys.path.insert(0, str(ROOT / "HomieRL"))
    from homie_isaaclab.config import G1RoughCfg
    return dict(G1RoughCfg.init_state.default_joint_angles)


def load_trajectory_grasp_frame(json_path: Path) -> dict:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    stage = data["solved_cartesian_sequence"][-1]
    frame = stage["frames"][-1]
    motors = frame["controlled_motor_positions"]
    arms = {name: float(motors[name]) for name in ARM_JOINTS}
    fingers = {name: float(motors[name]) for name in FINGER_JOINTS}
    reach = data["solved_cartesian_sequence"][0]["frames"][-1]["cartesian_targets"]
    tcp = {side: {"position": reach[side]["actual_position_world"], "wxyz": reach[side]["actual_wxyz_world"]}
           for side in ("left", "right") if side in reach}
    return {
        "arms": arms, "fingers": fingers, "planner_tcp_world": tcp,
        "source": {
            "json": str(Path(json_path).resolve().relative_to(ROOT)) if Path(json_path).resolve().is_relative_to(ROOT) else str(json_path),
            "sequence": "solved_cartesian_sequence", "stage": stage["stage"], "frame_index": frame["frame_index"],
            "experiment": data["metadata"].get("experiment", {}),
            "planner_joint_count": len(data["joint_orders"]["viser_urdf_order"]),
        },
    }


class Kinematics:
    """Link poses of g1_grasp.urdf at a given configuration, expressed in the world frame."""

    def __init__(self, urdf_path: Path = G1_GRASP_URDF, load_meshes: bool = True):
        import yourdfpy
        self.urdf = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=load_meshes,
                                       build_collision_scene_graph=False, load_collision_meshes=False)
        if list(self.urdf.actuated_joint_names) != HOMIE_JOINTS:
            raise ValueError("yourdfpy joint order differs from HOMIE's 27-joint order")
        self.world_T_base = np.eye(4)

    def set_configuration(self, joint_pos: dict[str, float], base_pos, base_quat_xyzw=(0., 0., 0., 1.)):
        cfg = np.array([joint_pos[name] for name in HOMIE_JOINTS], dtype=float)
        self.urdf.update_cfg(cfg)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(np.asarray(base_quat_xyzw, dtype=float)).as_matrix()
        T[:3, 3] = np.asarray(base_pos, dtype=float)
        self.world_T_base = T

    def base_T(self, link: str) -> np.ndarray:
        return self.urdf.get_transform(link, self.urdf.base_link)

    def world_T(self, link: str) -> np.ndarray:
        return self.world_T_base @ self.base_T(link)

    def pelvis_height_on_ground(self, joint_pos: dict[str, float], base_quat_xyzw=(0., 0., 0., 1.)) -> float:
        """Pelvis z such that the lowest foot point touches z = 0 for the given leg configuration and base
        orientation.

        Uses the foot STL vertices (the URDF and MJCF collision geometry) when meshes are loaded;
        otherwise falls back to HOMIE's ankle_sole_distance approximation.
        """
        self.set_configuration(joint_pos, (0., 0., 0.), base_quat_xyzw)
        graph = self.urdf.scene.graph
        lowest = None
        for node in graph.nodes_geometry:
            link = graph.transforms.parents[node]
            if link not in FEET_LINKS:
                continue
            T, geom = graph[node]
            verts = np.asarray(self.urdf.scene.geometry[geom].vertices)
            z = (self.world_T_base @ T @ np.c_[verts, np.ones(len(verts))].T).T[:, 2].min()
            lowest = z if lowest is None else min(lowest, z)
        if lowest is None:
            lowest = min(self.world_T(link)[2, 3] for link in FEET_LINKS) - ANKLE_SOLE_DISTANCE
        return float(-lowest)

    def grasp_point_world(self, side: str, point_in_palm) -> np.ndarray:
        T = self.world_T(PALM_LINKS[side])
        return (T @ np.append(np.asarray(point_in_palm, dtype=float), 1.0))[:3]

    def hand_surface_points_in_palm(self, side: str, samples_per_mesh: int = 20000) -> np.ndarray:
        """Dense surface samples (plus vertices) of the palm and finger meshes, in the palm frame."""
        import trimesh
        palm = PALM_LINKS[side]
        graph = self.urdf.scene.graph
        inv_palm = np.linalg.inv(self.base_T(palm))
        points = []
        rng = np.random.default_rng(0)
        for node in graph.nodes_geometry:
            link = graph.transforms.parents[node]
            if not link.startswith(f"{side}_hand"):
                continue
            T_node, geom = graph[node]
            mesh = self.urdf.scene.geometry[geom]
            surface, _ = trimesh.sample.sample_surface(mesh, samples_per_mesh, seed=int(rng.integers(1 << 31)))
            local = np.concatenate([np.asarray(mesh.vertices), np.asarray(surface)])
            points.append((inv_palm @ T_node @ np.c_[local, np.ones(len(local))].T).T[:, :3])
        return np.concatenate(points)

    def finger_cavity_center(self, side: str, cell: float = 0.001) -> tuple[np.ndarray, float, dict]:
        """Centre of the largest circle inscribed in the cavity enclosed by palm, thumb and fingers.

        The bar runs roughly along the palm z axis (finger joint axes), so the cavity is found in
        the palm x-y cross-section: rasterise the hand surface, flood-fill the free region around a
        seed between palm and fingers, and take the free cell farthest from any hand surface point.
        Returns (centre_in_palm [x, y, 0], inscribed_radius, diagnostics).
        """
        from collections import deque
        from scipy.spatial import cKDTree

        pts = self.hand_surface_points_in_palm(side)
        sign = -1.0 if side == "left" else 1.0          # fingers curl towards -y (left) / +y (right)
        x0, x1 = -0.01, 0.13
        y0, y1 = (-0.10, 0.03) if sign < 0 else (-0.03, 0.10)
        nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
        occ = np.zeros((ny, nx), dtype=bool)
        ix = ((pts[:, 0] - x0) / cell).astype(int)
        iy = ((pts[:, 1] - y0) / cell).astype(int)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ[iy[keep], ix[keep]] = True
        # close 1-cell gaps between surface samples
        dil = occ.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dil |= np.roll(np.roll(occ, dy, axis=0), dx, axis=1)
        seed = (int((0.050 - x0) / cell), int((sign * 0.030 - y0) / cell))
        if dil[seed[1], seed[0]]:
            raise RuntimeError(f"{side}: cavity seed lies inside hand geometry; adjust the seed")
        comp = np.zeros_like(dil)
        queue = deque([seed])
        comp[seed[1], seed[0]] = True
        touched_border = False
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                px, py = cx + dx, cy + dy
                if px < 0 or py < 0 or px >= nx or py >= ny:
                    touched_border = True
                    continue
                if not dil[py, px] and not comp[py, px]:
                    comp[py, px] = True
                    queue.append((px, py))
        cells = np.argwhere(comp)
        centers = np.c_[x0 + (cells[:, 1] + 0.5) * cell, y0 + (cells[:, 0] + 0.5) * cell]
        dist, _ = cKDTree(pts[:, :2]).query(centers)
        best = int(np.argmax(dist))
        center = np.array([centers[best, 0], centers[best, 1], 0.0])
        diag = {"cavity_cells": int(comp.sum()), "cavity_area_cm2": float(comp.sum() * cell * cell * 1e4),
                "flood_touched_border": touched_border, "inscribed_radius_m": float(dist[best])}
        return center, float(dist[best]), diag

    def point_in_palm(self, side: str, point_world) -> np.ndarray:
        T = np.linalg.inv(self.world_T(PALM_LINKS[side]))
        return (T @ np.append(np.asarray(point_world, dtype=float), 1.0))[:3]


def _round(values, digits=6):
    if isinstance(values, dict):
        return {k: _round(v, digits) for k, v in values.items()}
    if isinstance(values, (list, tuple, np.ndarray)):
        return [_round(v, digits) for v in values]
    if isinstance(values, (float, np.floating)):
        return round(float(values), digits)
    return values


def import_initial_pose(json_path: Path, destination: Path = INITIAL_POSE_YAML) -> dict:
    """A1 + A2: build g1_grasp.urdf from the trajectory's finger angles, then write initial_pose.yaml."""
    from .assets.g1_grasp_urdf import build_g1_grasp_urdf

    grasp = load_trajectory_grasp_frame(json_path)
    build_g1_grasp_urdf(grasp["fingers"])
    defaults = homie_default_joint_angles()
    joint_pos = {name: float(defaults[name]) for name in LEG_JOINTS + WAIST_JOINTS}
    joint_pos.update(grasp["arms"])

    kin = Kinematics(load_meshes=True)
    pelvis_z = kin.pelvis_height_on_ground(joint_pos)
    kin.set_configuration(joint_pos, (0., 0., pelvis_z))
    # Grasp centre = centre of the largest circle inscribed in the finger cavity. The arms are
    # mirrored, so use the symmetric mean of both hands and mirror it (x, -y, z) for a level bar.
    left_c, left_r, left_diag = kin.finger_cavity_center("left")
    right_c, right_r, right_diag = kin.finger_cavity_center("right")
    left = (left_c + right_c * np.array([1., -1., 1.])) / 2
    hollow = {"left": left, "right": left * np.array([1., -1., 1.])}
    cavity = {"left": {**left_diag, "center_in_palm": left_c.tolist()},
              "right": {**right_diag, "center_in_palm": right_c.tolist()},
              "inscribed_radius_m": float(min(left_r, right_r))}
    hypotheses = {}
    straight = kin.pelvis_height_on_ground({**joint_pos, **{n: 0. for n in LEG_JOINTS}})
    for label, h in (("homie_stance", pelvis_z), ("straight_legs", straight)):
        kin.set_configuration(joint_pos, (0., 0., h))
        hypotheses[label] = {"pelvis_z": h, **{side: kin.point_in_palm(side, grasp["planner_tcp_world"][side]["position"])
                                                for side in grasp["planner_tcp_world"]}}
    kin.set_configuration(joint_pos, (0., 0., pelvis_z))

    pose = {
        "source": grasp["source"],
        "robot": {
            "base_pos": [0., 0., pelvis_z],
            "base_quat_xyzw": [0., 0., 0., 1.],
            "joint_pos": joint_pos,
            "finger_pos": grasp["fingers"],
        },
        "grasp": {
            "point_in_palm": {side: hollow[side].tolist() for side in hollow},
            "bar_axis_world": [0., 1., 0.],
            "cavity": cavity,
        },
        "reference": {
            "planner_tcp_world": grasp["planner_tcp_world"],
            "planner_tcp_in_palm_hypotheses": hypotheses,
            "palm_world": {side: kin.world_T(PALM_LINKS[side])[:3, 3].tolist() for side in PALM_LINKS},
            "grasp_point_world": {side: kin.grasp_point_world(side, hollow[side]).tolist() for side in hollow},
        },
    }
    save_initial_pose(pose, destination)
    return pose


LEG_JOINT_KINDS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")


def symmetrize_leg_angles(measured: dict[str, float]) -> dict[str, float]:
    """Left/right average of a measured leg pose (roll and yaw mirrored) so both hands stay mirrored."""
    out = {}
    for kind in LEG_JOINT_KINDS:
        sign = -1.0 if kind.endswith(("roll", "yaw")) else 1.0
        mean = (float(measured[f"left_{kind}_joint"]) + sign * float(measured[f"right_{kind}_joint"])) / 2
        out[f"left_{kind}_joint"] = mean
        out[f"right_{kind}_joint"] = sign * mean
    return out


def foot_tilt(kin: "Kinematics", side: str) -> tuple[float, float]:
    """(pitch, roll) of the ankle_roll link in the world frame (includes the base orientation), radians."""
    Rm = kin.world_T(f"{side}_ankle_roll_link")[:3, :3]
    pitch = float(np.arctan2(-Rm[2, 0], np.hypot(Rm[0, 0], Rm[1, 0])))
    roll = float(np.arctan2(Rm[2, 1], Rm[2, 2]))
    return pitch, roll


def level_feet(kin: "Kinematics", joint_pos: dict[str, float], base_quat_xyzw=(0., 0., 0., 1.),
               iterations: int = 8, tol: float = 1e-7) -> dict[str, float]:
    """Adjust ankle pitch/roll so both soles are horizontal in the world for the given pelvis orientation
    (Newton on FK)."""
    jp = dict(joint_pos)
    for side in ("left", "right"):
        names = (f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint")
        for _ in range(iterations):
            kin.set_configuration(jp, (0., 0., 0.), base_quat_xyzw)
            residual = np.array(foot_tilt(kin, side))
            if np.abs(residual).max() < tol:
                break
            jac = np.zeros((2, 2))
            for j, name in enumerate(names):
                trial = dict(jp)
                trial[name] += 1e-5
                kin.set_configuration(trial, (0., 0., 0.), base_quat_xyzw)
                jac[:, j] = (np.array(foot_tilt(kin, side)) - residual) / 1e-5
            step = np.linalg.solve(jac, -residual)
            for j, name in enumerate(names):
                jp[name] += float(step[j])
    kin.set_configuration(jp, (0., 0., 0.), base_quat_xyzw)
    return jp


def apply_stance(stance: dict, destination: Path = INITIAL_POSE_YAML, source: Path | None = None,
                 bake_pitch: bool = True) -> dict:
    """Replace the preset leg angles by HOMIE's measured standing pose and re-derive the pelvis height.

    The arms, fingers, and grasp points are untouched; the whole robot simply stands where the policy
    actually stands, so the handle height derived from the pose (``cart build``) follows it and the static
    grasp preload vanishes. Feet: soles level in the world, lowest sole point at the (few mm) gap between the
    measured standing height and the FK-on-ground height.

    ``bake_pitch``: also give the pelvis the policy's measured standing pitch (the hands sit ~0.3 m ahead,
    so 1 deg moves the handle ~6 mm, about 10 N of preload). ``False`` keeps the pelvis upright at reset.
    """
    pose = load_initial_pose(source or destination)
    legs = symmetrize_leg_angles(stance["leg_joint_mean"])
    joint_pos = dict(pose["robot"]["joint_pos"])
    joint_pos.update(legs)
    pitch = float(stance.get("pelvis_pitch_mean") or 0.0) if bake_pitch else 0.0
    if abs(pitch) > math.radians(5.0):
        raise ValueError(f"measured pelvis pitch {math.degrees(pitch):.1f} deg is implausible; stance data is suspect")
    base_quat = R.from_euler("y", pitch).as_quat().tolist()          # xyzw
    kin = Kinematics(load_meshes=True)
    leveled = level_feet(kin, joint_pos, base_quat)
    leveling = {n: leveled[n] - joint_pos[n] for n in LEG_JOINTS if abs(leveled[n] - joint_pos[n]) > 1e-9}
    fk_pelvis_z = kin.pelvis_height_on_ground(leveled, base_quat)
    # The policy shuffles while standing (the averaged leg angles mix stance and lifted legs), so the FK
    # height of the mean pose is a few mm below where the pelvis actually rests. The handle height must
    # match the *measured* standing height; the feet then float by that gap at reset and settle in one
    # physics step. Anything beyond a few mm means the measurement is suspect.
    pelvis_z = float(stance.get("root_height_mean") or fk_pelvis_z)
    feet_float = pelvis_z - fk_pelvis_z
    if abs(feet_float) > 0.006:
        raise ValueError(f"measured standing height differs from FK-on-ground by {feet_float * 1000:.1f} mm; "
                         "stance data is suspect")
    kin.set_configuration(leveled, (0., 0., pelvis_z), base_quat)
    # Arm sag: the preset is the PD *target*; under gravity the held arms rest slightly off it. Raising the
    # target by the measured sag makes the hands rest exactly at the preset (where the handle is placed).
    sag = stance.get("upper_sag_rad") or {}
    hold_offset = {n: float(sag.get(n, 0.0)) for n in HOMIE_JOINTS[12:]}
    if max(abs(v) for v in hold_offset.values()) > 0.15:
        raise ValueError(f"upper-body sag above 0.15 rad: {hold_offset}; stance data is suspect")
    pose["robot"]["joint_pos"] = {n: float(leveled[n]) for n in HOMIE_JOINTS}
    pose["robot"]["base_pos"] = [0., 0., float(pelvis_z)]
    pose["robot"]["base_quat_xyzw"] = [float(v) for v in base_quat]
    pose["robot"]["upper_hold_offset"] = hold_offset
    pose["reference"]["palm_world"] = {side: kin.world_T(PALM_LINKS[side])[:3, 3].tolist() for side in PALM_LINKS}
    pose["reference"]["grasp_point_world"] = {
        side: kin.grasp_point_world(side, pose["grasp"]["point_in_palm"][side]).tolist() for side in PALM_LINKS}
    pose["stance"] = {
        "note": "legs = HOMIE's measured zero-command standing pose (symmetrised, soles levelled); "
                "re-run `stance --apply` after freezing a different HOMIE checkpoint",
        "checkpoint": stance.get("checkpoint"),
        "measured_root_height": stance.get("root_height_mean"),
        "measured_pelvis_pitch_rad": stance.get("pelvis_pitch_mean"),
        "base_pitch_baked_rad": pitch,
        "measured_leg_joint_mean": {n: float(stance["leg_joint_mean"][n]) for n in LEG_JOINTS},
        "measured_leg_joint_std": {n: float(stance["leg_joint_std"][n]) for n in LEG_JOINTS} if "leg_joint_std" in stance else None,
        "ankle_leveling_delta_rad": leveling,
        "measured_feet_link_z": stance.get("feet_link_z_stance"),
        "fk_pelvis_z_on_ground": float(fk_pelvis_z),
        "feet_float_at_reset_m": float(feet_float),
    }
    save_initial_pose(pose, destination)
    return pose


def load_initial_pose(path: Path = INITIAL_POSE_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_initial_pose(pose: dict, path: Path = INITIAL_POSE_YAML) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_round(pose), sort_keys=False, allow_unicode=True), encoding="utf-8")


def full_joint_dict(pose: dict) -> dict[str, float]:
    return dict(pose["robot"]["joint_pos"])
