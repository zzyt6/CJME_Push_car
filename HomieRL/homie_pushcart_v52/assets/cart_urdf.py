"""Cart geometry shared by the URDF writer and the viser check, plus the pose derivation.

The handle bar is *defined* to pass through the two grasp points of the preset
pose; the cart body is placed behind it. So the cart adapts to the hands, not
the other way round.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from ..paths import CART_OBJ, CART_PARAMS_YAML, CART_URDF, INITIAL_POSE_YAML, ROOT
from ..pose import Kinematics, load_initial_pose


def load_cart_params(path: Path = CART_PARAMS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_cart_params(params: dict, path: Path = CART_PARAMS_YAML) -> None:
    text = path.read_text(encoding="utf-8")
    head = text.split("derived:")[0].rstrip("\n") + "\n"
    derived = yaml.safe_dump({"derived": params["derived"]}, sort_keys=False)
    path.write_text(head + derived, encoding="utf-8")


@dataclass
class Primitive:
    name: str
    kind: str                 # box | cylinder | sphere
    size: tuple              # box: (x, y, z); cylinder: (radius, length) with axis along local z; sphere: (radius,)
    xyz: tuple
    rpy: tuple = (0., 0., 0.)
    collision: bool = True
    visual: bool = True
    link: str = "base_link"


def cart_primitives(params: dict, handle_height: float) -> list[Primitive]:
    """Collision/visual primitives of the base link in the base frame."""
    deck, frame, handle = params["deck"], params["frame"], params["handle"]
    # Deck, posts and shelves are collision-only: the OBJ mesh already provides their visuals.
    # The handle bars are visual-only: the OBJ lacks the horizontal bar and the hands have no colliders.
    prims = [Primitive("deck", "box", tuple(deck["size"]), (0., 0., deck["center_z"]), visual=False)]
    deck_top = deck["center_z"] + deck["size"][2] / 2
    px, py = frame["post_section"]
    for i, x in enumerate(frame["x_positions"]):
        post_h = handle_height - deck_top
        for j, y in enumerate((-frame["post_half_y"], frame["post_half_y"])):
            prims.append(Primitive(f"post_{i}{j}", "box", (px, py, post_h), (x, y, deck_top + post_h / 2), visual=False))
        prims.append(Primitive(f"bar_{i}", "cylinder", (handle["radius"], handle["length"]), (x, 0., handle_height),
                               rpy=(math.pi / 2, 0., 0.), collision=bool(handle["collision"]), visual=True))
    for k, z in enumerate(params["shelves"]["heights"]):
        prims.append(Primitive(f"shelf_{k}", "box", tuple(params["shelves"]["size"]), (0., 0., z), visual=False))
    if params["model"] == "rigid":
        for i, (x, y) in enumerate(params["wheels"]["positions_xy"]):
            r = params["wheels"]["radius"]
            prims.append(Primitive(f"pad_{i}", "sphere", (r,), (x, y, r), visual=False))
    return prims


def derive_cart_pose(params: dict, pose: dict, kin: Kinematics | None = None) -> dict:
    kin = kin or Kinematics(load_meshes=False)
    kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])
    p_left = kin.grasp_point_world("left", pose["grasp"]["point_in_palm"]["left"])
    p_right = kin.grasp_point_world("right", pose["grasp"]["point_in_palm"]["right"])
    axis = p_left - p_right
    axis /= np.linalg.norm(axis)
    center = (p_left + p_right) / 2
    yaw = math.atan2(axis[1], axis[0]) - math.pi / 2
    near_x = params["frame"]["x_positions"][0]          # handle end, negative
    base = center + np.array([-near_x * math.cos(yaw), -near_x * math.sin(yaw), 0.])
    base[2] = 0.
    return {
        "handle_height": float(center[2]),
        "bar_center_world": center.tolist(),
        "bar_axis_world": axis.tolist(),
        "bar_tilt_deg": float(math.degrees(math.acos(min(1., abs(axis[1]))))),
        "grasp_points_world": {"left": p_left.tolist(), "right": p_right.tolist()},
        "base_pos_world": base.tolist(),
        "base_yaw": float(yaw),
    }


# ----------------------------------------------------------------------------- URDF writing

def _box_inertia(m, sx, sy, sz):
    return (m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2))


def _cyl_inertia(m, r, h):
    return (m / 12 * (3 * r**2 + h**2), m / 12 * (3 * r**2 + h**2), m / 2 * r**2)


def _inertial(mass, ixx, iyy, izz, xyz=(0., 0., 0.)):
    return (f'    <inertial>\n      <origin xyz="{xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}" rpy="0 0 0"/>\n'
            f'      <mass value="{mass:.6g}"/>\n'
            f'      <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>\n    </inertial>\n')


def _geometry(prim: Primitive) -> str:
    if prim.kind == "box":
        return f'<box size="{prim.size[0]:.6g} {prim.size[1]:.6g} {prim.size[2]:.6g}"/>'
    if prim.kind == "cylinder":
        return f'<cylinder radius="{prim.size[0]:.6g}" length="{prim.size[1]:.6g}"/>'
    return f'<sphere radius="{prim.size[0]:.6g}"/>'


def _shape(tag: str, prim: Primitive, material: str | None = None) -> str:
    origin = f'<origin xyz="{prim.xyz[0]:.6g} {prim.xyz[1]:.6g} {prim.xyz[2]:.6g}" rpy="{prim.rpy[0]:.6g} {prim.rpy[1]:.6g} {prim.rpy[2]:.6g}"/>'
    mat = f"\n      <material name=\"{material}\"/>" if material else ""
    return f'    <{tag} name="{prim.name}">\n      {origin}\n      <geometry>{_geometry(prim)}</geometry>{mat}\n    </{tag}>\n'


def write_cart_urdf(params: dict, handle_height: float, destination: Path = CART_URDF) -> None:
    prims = cart_primitives(params, handle_height)
    m_base = params["mass"]["base_kg"]
    ixx, iyy, izz = _box_inertia(m_base, 1.0, 0.54, 0.6)
    vis = params["visual"]
    # Mesh paths are resolved relative to the URDF file; the shared OBJ lives one directory up.
    obj_path = Path("..") / Path(CART_OBJ).name
    out = ['<?xml version="1.0"?>',
           '<!-- Generated by homie_pushcart_v52.assets.cart_urdf from cart_params.yaml; handle height derived from initial_pose.yaml. -->',
           '<robot name="xiaotuiche">',
           '  <material name="steel"><color rgba="0.55 0.6 0.65 1"/></material>',
           '  <material name="rubber"><color rgba="0.15 0.15 0.15 1"/></material>',
           '  <link name="base_link">',
           _inertial(m_base, ixx, iyy, izz, (0., 0., 0.3))]
    out.append(f'    <visual name="obj_visual">\n      <origin xyz="{vis["offset"][0]:.6g} {vis["offset"][1]:.6g} {vis["offset"][2]:.6g}" rpy="0 0 0"/>\n'
               f'      <geometry><mesh filename="{obj_path.as_posix()}" scale="{vis["scale"]} {vis["scale"]} {vis["scale"]}"/></geometry>\n'
               f'      <material name="steel"/>\n    </visual>\n')
    for prim in prims:
        if prim.visual:
            out.append(_shape("visual", prim, "steel"))
        if prim.collision:
            out.append(_shape("collision", prim))
    out.append("  </link>")

    w = params["wheels"]
    r, width, trail = w["radius"], w["width"], w["caster_trail"]
    if params["model"] == "caster":
        for i, (x, y) in enumerate(w["positions_xy"]):
            sm, wm = w["swivel_mass_kg"], w["wheel_mass_kg"]
            out.append(f'  <link name="caster_{i}_swivel">\n{_inertial(sm, *_box_inertia(sm, 0.04, 0.04, 0.04))}  </link>')
            out.append(f'  <joint name="caster_{i}_swivel_joint" type="continuous">\n'
                       f'    <origin xyz="{x:.6g} {y:.6g} {r + 0.02:.6g}" rpy="0 0 0"/>\n'
                       f'    <parent link="base_link"/>\n    <child link="caster_{i}_swivel"/>\n'
                       f'    <axis xyz="0 0 1"/>\n    <dynamics damping="{w["swivel_damping"]}" friction="0"/>\n  </joint>')
            wheel = Primitive(f"wheel_{i}", "cylinder", (r, width), (0., 0., 0.), rpy=(math.pi / 2, 0., 0.), link=f"wheel_{i}")
            # PhysX under Isaac Lab approximates cylinders by convex hulls (custom cylinder geometry is disabled for
            # the GPU pipeline), which makes rolling wheels bump. A sphere collider of the same radius rolls exactly.
            if w.get("collider", "cylinder") == "sphere":
                wheel_col = Primitive(f"wheel_{i}", "sphere", (r,), (0., 0., 0.), link=f"wheel_{i}")
            else:
                wheel_col = wheel
            cx, cy, cz = _cyl_inertia(wm, r, width)
            out.append(f'  <link name="wheel_{i}">\n{_inertial(wm, cx, cz, cy)}{_shape("visual", wheel, "rubber")}{_shape("collision", wheel_col)}  </link>')
            out.append(f'  <joint name="wheel_{i}_joint" type="continuous">\n'
                       f'    <origin xyz="{-trail:.6g} 0 {-0.02:.6g}" rpy="0 0 0"/>\n'
                       f'    <parent link="caster_{i}_swivel"/>\n    <child link="wheel_{i}"/>\n'
                       f'    <axis xyz="0 1 0"/>\n    <dynamics damping="{w["wheel_damping"]}" friction="0"/>\n  </joint>')
    out.append("</robot>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_cart(params_path: Path = CART_PARAMS_YAML, pose_path: Path = INITIAL_POSE_YAML) -> dict:
    params = load_cart_params(params_path)
    pose = load_initial_pose(pose_path)
    derived = derive_cart_pose(params, pose)
    params["derived"] = {k: derived[k] for k in ("handle_height", "bar_center_world", "bar_axis_world", "base_pos_world", "base_yaw")}
    save_cart_params(params, params_path)
    write_cart_urdf(params, derived["handle_height"])
    return {"derived": derived, "urdf": str(CART_URDF)}
