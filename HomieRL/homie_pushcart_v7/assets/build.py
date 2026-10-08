"""V7 assets: the V3.2 cart plus a slide/roll grasp chain per hand, for Isaac (URDF -> USD) and MuJoCo.

    base_link --slide (prismatic, bar axis)--> slider --roll (bar axis)--> roll_link --tilt (cart x)--> tilt_link
              --yaw (cart z)--> grasp_<side>_link  <== palm weld / FixedJoint

The chain origin is the V7 grasp point (assets/grasp_fit.py: Dex3 fingers closed onto the square handle tube,
the tube axis on the slide/roll axis), with the cart's orientation. The robot uses g1_grasp_v7.urdf (fitted finger
angles baked in); the cart is placed from the fitted grasp points. Joint friction is not authored in the
URDF (importers map it inconsistently); the task writes it per environment at reset. MuJoCo gets nominal
values and stiff friction-constraint parameters (see FRICTION_SOLREF) so frictionloss acts as Coulomb friction
on the casters. The grasp slide is elasto-plastic in both simulators: a spring to an anchor, force-capped at the
friction limit (Isaac implicit drive maxForce, MuJoCo actuator forcerange), the anchor dragged past the limit.

    python scripts/pushcart_v7.py build grasp          (CPU: fit fingers to the tube, robot URDF, section figure)
    python scripts/pushcart_v7.py build urdf|mjcf      (CPU)
    python scripts/pushcart_v7.py build usd            (inside Kit)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from homie_pushcart.assets.cart_urdf import derive_cart_pose, load_cart_params as load_v32_cart_params, write_cart_urdf
from homie_pushcart.pose import Kinematics, load_initial_pose

from ..paths import (ASSET_DIR, ASSET_REPORT, BUILD_MANIFEST, CART_PARAMS_V7_YAML, CART_URDF, G1_GRASP_V7_URDF,
                     GRASP_LINKS, GRASP_POSE_V7_YAML, GRASP_SECTION_PNG, MJCF_REPORT, MJCF_SCENE, PACKED_USD,
                     ROLL_JOINTS, ROLL_LINKS, ROOT, SIDES, SLIDE_JOINTS, SLIDER_LINKS, TILT_JOINTS, TILT_LINKS,
                     YAW_JOINTS)

# MuJoCo friction-loss constraints are soft by default: with solref 0.02 / solimp 0.9 a 30 N frictionloss joint
# creeps from ~1 N (validated in scripts/validate_pushcart_v7_mujoco.py). This pair makes it stick until the
# limit and slide at it (onset within 1 % of the limit, creep < 1e-4 m).
FRICTION_SOLREF = "0.005 1"
FRICTION_SOLIMP = "0.999 0.9999 0.001"


def load_v7_pose() -> dict:
    """V3.2 initial pose (robot joints, base) with the V7 fitted finger angles and grasp points."""
    pose = load_initial_pose()
    if not GRASP_POSE_V7_YAML.is_file():
        raise FileNotFoundError(f"{GRASP_POSE_V7_YAML} missing; run `pushcart_v7.py build grasp`")
    fit = yaml.safe_load(GRASP_POSE_V7_YAML.read_text(encoding="utf-8"))
    pose["robot"]["finger_pos"] = {k: float(v) for k, v in fit["finger_pos"].items()}
    pose["grasp"]["point_in_palm"] = {s: [float(v) for v in fit["point_in_palm"][s]] for s in SIDES}
    pose["grasp"]["v7_fit"] = {"profile": fit["profile"], "size_m": fit["size_m"], "clearance_m": fit["clearance_m"]}
    return pose


def load_v7_params(path: Path = CART_PARAMS_V7_YAML, pose: dict | None = None) -> dict:
    """V3.2 cart parameters with the V7 sections, the real handle tube, and the cart placed from the V7 grasp."""
    v7 = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = load_v32_cart_params(ROOT / "HomieRL" / v7.pop("inherit"))
    overlap = set(base) & set(v7)
    if overlap:
        raise ValueError(f"V7 cart parameters must not redefine V3.2 sections: {sorted(overlap)}")
    params = {**base, **v7}
    h = params["handle_v7"]
    params["handle"] = {**params["handle"], "profile": h["profile"], "side": float(h["side"]),
                        "radius": float(h["side"]) / 2}
    if GRASP_POSE_V7_YAML.is_file() or pose is not None:
        derived = derive_cart_pose(params, pose or load_v7_pose(), Kinematics(load_meshes=False))
        params["derived"] = {k: derived[k] for k in ("handle_height", "bar_center_world", "bar_axis_world",
                                                     "base_pos_world", "base_yaw")}
    return params


def slide_static_force(params: dict, mu, normal, load_normal_to_bar=0.0):
    """Coulomb slide limit: mu * (grip normal force + load across the tube) + offset, floored."""
    g = params["grasp"]
    return np.maximum(np.asarray(mu) * (np.asarray(normal) + np.asarray(load_normal_to_bar))
                      + float(g["slide_offset_n"]), float(g["slide_min_n"]))


def fit_and_write_grasp() -> dict:
    """Fit the fingers to the handle tube, store the result, bake the robot URDF and draw the cross-sections."""
    from homie_pushcart.assets.g1_grasp_urdf import build_g1_grasp_urdf
    from .grasp_fit import fit_grasp, section_figure
    params = load_v7_params(pose=load_initial_pose())
    h, clearance = params["handle_v7"], float(params["grasp"]["fit_clearance_m"])
    size = float(h["side"]) if h["profile"] == "square" else float(h["side"]) / 2
    fit = fit_grasp(h["profile"], size, clearance)
    record = {"profile": h["profile"], "size_m": size, "clearance_m": clearance,
              "finger_pos": {k: float(v) for k, v in fit["finger_pos"].items()},
              "point_in_palm": fit["point_in_palm"], "diagnostics": fit["diagnostics"]}
    header = "# Generated by homie_pushcart_v7.assets.build.fit_and_write_grasp; do not edit.\n"
    GRASP_POSE_V7_YAML.write_text(header + yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    build_g1_grasp_urdf(record["finger_pos"], destination=G1_GRASP_V7_URDF)
    section_figure(fit, GRASP_SECTION_PNG)
    return record


def mean(r):
    return 0.5 * (float(r[0]) + float(r[1]))


def nominal_grasp(params: dict) -> dict:
    """Mid-range grasp parameters (MuJoCo defaults; Isaac samples them per environment)."""
    g = params["grasp"]
    mu, normal = mean(g["friction_coeff_range"]), mean(g["normal_force_range"])
    k = mean(g["rotation_stiffness_range"])
    kr = mean(g["roll_stiffness_range"])
    ks = mean(g["slide_stiffness_range"])
    return {"mu": mu, "normal": normal, "slide_static": float(slide_static_force(params, mu, normal)),
            "slide_stiffness": ks, "slide_damping": float(slide_damping(params, ks)),
            "stiffness": k, "damping": rotation_damping(params, k),
            "roll_stiffness": kr, "roll_damping": float(rotation_damping(params, kr)),
            "roll_capacity": mean(g["roll_capacity_range"]), "moment_capacity": mean(g["moment_capacity_range"])}


def slide_damping(params: dict, stiffness):
    g = params["grasp"]
    return 2.0 * float(g["slide_damping_ratio"]) * np.sqrt(np.asarray(stiffness) * float(g["slide_damping_mass_kg"]))


def rotation_damping(params: dict, stiffness):
    g = params["grasp"]
    return 2.0 * float(g["rotation_damping_ratio"]) * np.sqrt(np.asarray(stiffness) * float(g["rotation_armature"]))


def nominal_wheel_friction(params: dict) -> tuple[float, float]:
    """Mid-range wheel rolling-resistance torque and swivel friction torque [N.m] for the 12 kg + 5 kg cart."""
    w = params["wheels_v7"]
    load = (params["mass"]["base_kg"] + 5.0) * 9.81 / 4
    return mean(w["rolling_coeff_range"]) * load * params["wheels"]["radius"], mean(w["swivel_friction_range"])


def grasp_geometry(params: dict, pose: dict | None = None, kin: Kinematics | None = None) -> dict:
    """Grasp points in the cart base frame plus the world frames needed by the MJCF/USD welds."""
    pose = pose or load_v7_pose()
    kin = kin or Kinematics(load_meshes=False)
    derived = derive_cart_pose(params, pose, kin)
    yaw = float(derived["base_yaw"])
    cart_T = np.eye(4)
    cart_T[:3, :3] = R.from_euler("z", yaw).as_matrix()
    cart_T[:3, 3] = derived["base_pos_world"]
    local = {}
    for side in SIDES:
        p = np.asarray(derived["grasp_points_world"][side], dtype=float)
        local[side] = (np.linalg.inv(cart_T) @ np.append(p, 1.))[:3]
    return {"derived": derived, "cart_T": cart_T, "grasp_local": local}


def slide_limits(params: dict, side: str) -> tuple[float, float]:
    g = params["grasp"]
    out, inward = float(g["slide_limit_out_m"]), float(g["slide_limit_in_m"])
    return (-inward, out) if side == "left" else (-out, inward)          # left hand is at +y


def _box_inertia(m, s):
    return m * s * s / 6.0


def chain(side: str) -> list[tuple[str, str, str, str]]:
    """(joint, type, axis, child link) from base_link to the grasp link; all joint origins at the grasp point."""
    return [(SLIDE_JOINTS[side], "prismatic", "0 1 0", SLIDER_LINKS[side]),
            (ROLL_JOINTS[side], "revolute", "0 1 0", ROLL_LINKS[side]),
            (TILT_JOINTS[side], "revolute", "1 0 0", TILT_LINKS[side]),
            (YAW_JOINTS[side], "revolute", "0 0 1", GRASP_LINKS[side])]


# ----------------------------------------------------------------------------------------------- URDF

def write_cart_urdf_v7(params: dict | None = None, destination: Path = CART_URDF) -> dict:
    params = params or load_v7_params()
    geom = grasp_geometry(params)
    write_cart_urdf(params, geom["derived"]["handle_height"], destination)
    tree = ET.parse(destination)
    robot = tree.getroot()
    robot.insert(0, ET.Comment(" V7: V3.2 cart, square handle, per-hand slide/roll/tilt/yaw grasp chain "
                               "(homie_pushcart_v7.assets.build). "))
    g = params["grasp"]
    m, i = float(g["link_mass_kg"]), _box_inertia(float(g["link_mass_kg"]), 0.03)
    lim = float(g["rotation_limit_rad"])
    for side in SIDES:
        x, y, z = (float(v) for v in geom["grasp_local"][side])
        lo, hi = slide_limits(params, side)
        parent = "base_link"
        for k, (name, kind, axis, child) in enumerate(chain(side)):
            el = ET.SubElement(robot, "link", name=child)
            inertial = ET.SubElement(el, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{m:.6g}")
            ET.SubElement(inertial, "inertia", ixx=f"{i:.6g}", ixy="0", ixz="0", iyy=f"{i:.6g}", iyz="0", izz=f"{i:.6g}")
            joint = ET.SubElement(robot, "joint", name=name, type=kind)
            ET.SubElement(joint, "origin", xyz=f"{x:.8g} {y:.8g} {z:.8g}" if k == 0 else "0 0 0", rpy="0 0 0")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "axis", xyz=axis)
            if kind == "prismatic":
                ET.SubElement(joint, "limit", lower=f"{lo:.6g}", upper=f"{hi:.6g}", effort="1000", velocity="5")
            else:
                ET.SubElement(joint, "limit", lower=f"{-lim:.6g}", upper=f"{lim:.6g}", effort="1000", velocity="50")
            ET.SubElement(joint, "dynamics", damping="0", friction="0")
            parent = child
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)
    return {"urdf": str(destination), "grasp_local": {s: v.tolist() for s, v in geom["grasp_local"].items()}}


# ----------------------------------------------------------------------------------------------- MuJoCo

def build_mjcf_v7(params: dict | None = None, output: Path = MJCF_SCENE) -> dict:
    """V3.2 MJCF (same robot, cart; V7 fingers / tube / cart placement) with the grasp chain and welds on it.

    Slide: a ``general`` actuator force = k (anchor - q) - d qdot clamped to +-friction limit; physics.MujocoGrasp
    moves the anchor (ctrl) and the limit every control step (elasto-plastic friction, as the Isaac task).
    Roll / tilt / yaw: a ``general`` actuator per joint with force = -k q - d qdot, clamped to +-capacity
    (forcerange) -- the same elastic-then-yield law as the Isaac implicit drive with maxForce. Nominal (mid-range)
    values; runners may rewrite them per episode.
    """
    import mujoco
    from homie_pushcart.assets import mjcf_scene as v32

    params = params or load_v7_params()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("_v32_staging.xml")
    pose = load_v7_pose()
    v32.build_mjcf(True, output=staging, pose=pose, params=params)    # V7 fingers, square tube, cart placement
    tree = ET.parse(staging)
    staging.unlink()
    root = tree.getroot()
    root.set("model", "g1_pushcart_v7")
    kin = Kinematics(load_meshes=False)
    geom = grasp_geometry(params, pose, kin)
    kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])
    cart = next(b for b in root.iter("body") if b.get("name") == v32.CART_BODY)
    g = params["grasp"]
    nom = nominal_grasp(params)
    rolling_t, swivel_t = nominal_wheel_friction(params)
    fric = {"solreffriction": FRICTION_SOLREF, "solimpfriction": FRICTION_SOLIMP}
    # Casters: Coulomb resistance with the stiff friction constraint (V3.2 used the soft default).
    for joint in cart.iter("joint"):
        name = joint.get("name", "")
        if name.endswith("_swivel_joint"):
            joint.attrib.update(frictionloss=f"{swivel_t:.6g}", damping=f'{mean(params["wheels_v7"]["swivel_viscous_range"]):.6g}', **fric)
        elif name.startswith("wheel_"):
            joint.attrib.update(frictionloss=f"{rolling_t:.6g}", damping=f'{mean(params["wheels_v7"]["wheel_viscous_range"]):.6g}', **fric)
    m, i = float(g["link_mass_kg"]), _box_inertia(float(g["link_mass_kg"]), 0.03)
    lim = float(g["rotation_limit_rad"])
    actuator = root.find("actuator")
    for side in SIDES:
        lo, hi = slide_limits(params, side)
        parent = cart
        for k, (name, kind, axis, child) in enumerate(chain(side)):
            pos = " ".join(f"{v:.8g}" for v in geom["grasp_local"][side]) if k == 0 else "0 0 0"
            body = ET.SubElement(parent, "body", name=child, pos=pos)
            if kind == "prismatic":
                ET.SubElement(body, "joint", name=name, type="slide", axis=axis, range=f"{lo:.6g} {hi:.6g}",
                              armature=f'{g["slide_armature"]}', damping="0", frictionloss="0")
                fs = nom["slide_static"]
                ET.SubElement(actuator, "general", name=f"{name}_servo", joint=name, gaintype="fixed", biastype="affine",
                              gainprm=f'{nom["slide_stiffness"]:.6g}',
                              biasprm=f'0 {-nom["slide_stiffness"]:.6g} {-nom["slide_damping"]:.6g}',
                              ctrllimited="true", ctrlrange=f"{lo:.6g} {hi:.6g}", forcelimited="true",
                              forcerange=f"{-fs:.6g} {fs:.6g}")
            else:
                # frictionloss="0" explicitly: g1.xml's <default> gives every joint frictionloss 0.1, which would add
                # an unmodelled 0.1 N.m friction to the pivots (found by validate_pushcart_v7_mujoco demand check)
                ET.SubElement(body, "joint", name=name, type="hinge", axis=axis, range=f"{-lim:.6g} {lim:.6g}",
                              armature=f'{g["rotation_armature"]}', damping="0", frictionloss="0")
                roll = name == ROLL_JOINTS[side]
                cap = nom["roll_capacity"] if roll else nom["moment_capacity"]
                k, d = (nom["roll_stiffness"], nom["roll_damping"]) if roll else (nom["stiffness"], nom["damping"])
                ET.SubElement(actuator, "general", name=f"{name}_servo", joint=name, gaintype="fixed", biastype="affine",
                              gainprm="0", biasprm=f'0 {-k:.6g} {-d:.6g}',
                              ctrllimited="true", ctrlrange="-1 1", forcelimited="true", forcerange=f"{-cap:.6g} {cap:.6g}")
            ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{m}", diaginertia=f"{i:.6g} {i:.6g} {i:.6g}")
            parent = body
    # Welds: wrist (MuJoCo merged the fixed palm into it) <-> grasp link, relative pose at the preset.
    grasp_T = {}
    for side in SIDES:
        T = geom["cart_T"].copy()
        T[:3, 3] = (geom["cart_T"] @ np.append(geom["grasp_local"][side], 1.))[:3]
        grasp_T[side] = T
    contact = root.find("contact")
    for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
        for side in SIDES:
            for _, _, _, body in chain(side):
                ET.SubElement(contact, "exclude", body1=foot, body2=body)
    welds = {}
    for weld in root.find("equality").findall("weld"):
        side = weld.get("name").replace("grasp_", "")
        wrist = weld.get("body1")
        rel = np.linalg.inv(kin.world_T(wrist)) @ grasp_T[side]
        q = R.from_matrix(rel[:3, :3]).as_quat()
        relpose = np.concatenate([rel[:3, 3], [q[3], q[0], q[1], q[2]]])
        weld.set("body2", GRASP_LINKS[side])
        weld.set("relpose", " ".join(f"{v:.8g}" for v in relpose))
        welds[side] = {"body1": wrist, "body2": GRASP_LINKS[side], "relpose": relpose.tolist()}
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    report = v32.verify_mjcf(output, pose, geom["derived"], True)
    model = mujoco.MjModel.from_xml_path(str(output))
    report.update({
        "output": str(output), "welds": welds, "nominal_grasp": nom,
        "grasp_chain": {side: [n for n, *_ in chain(side)] for side in SIDES},
        "slide_range": {side: model.joint(SLIDE_JOINTS[side]).range.tolist() for side in SIDES},
        "caster_frictionloss": {"swivel": swivel_t, "wheel": rolling_t},
    })
    MJCF_REPORT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


# ----------------------------------------------------------------------------------------------- USD (Kit)

def pack_usd_v7(params: dict | None = None) -> dict:
    from homie_pushcart.assets.pack_usd import find_prim_named, pack

    params = params or load_v7_params()
    geom = grasp_geometry(params)

    def grasp_target(stage, side, cart_T, gp_world, palm_T):
        prim = find_prim_named(stage, "/Scene/Cart", GRASP_LINKS[side])
        expected = (cart_T @ np.append(geom["grasp_local"][side], 1.))[:3]
        if np.linalg.norm(expected - gp_world) > 1e-6:
            raise ValueError(f"{side}: grasp link origin {expected} differs from the palm grasp point {gp_world}")
        # The link frame has the cart orientation at zero joint positions; the joint sits on its origin.
        return prim, np.zeros(3), np.linalg.inv(cart_T)[:3, :3] @ palm_T[:3, :3]

    return pack(PACKED_USD, asset_dir=ASSET_DIR, cart_urdf=CART_URDF, report_path=ASSET_REPORT,
                params=params, grasp_target=grasp_target, robot_urdf=G1_GRASP_V7_URDF, pose=load_v7_pose())


def write_build_manifest() -> dict:
    """Hash every built V7 asset file; physics.validate_assets() rejects later drift."""
    files = [CART_URDF, CART_PARAMS_V7_YAML, GRASP_POSE_V7_YAML, G1_GRASP_V7_URDF, MJCF_SCENE]
    if PACKED_USD.is_file():
        files += sorted(p for p in ASSET_DIR.rglob("*") if p.suffix in (".usd", ".usda") and p.is_file())
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    BUILD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    BUILD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
