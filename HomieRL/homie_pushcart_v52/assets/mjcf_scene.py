"""Build the MuJoCo scene for sim2sim: HOMIE's g1.xml with the Dex3 grasp baked in, plus the cart and welds.

Outputs (assets/g1_pushcart/mujoco/):
    g1_grasp.xml      robot only (fingers at the recorded grasp), floor from g1.xml
    g1_pushcart.xml   robot + cart (free base, 4 passive casters) + two weld constraints palm <-> cart
    mjcf_report.json  joint order check, weld violation at the preset pose, contacts at rest

MuJoCo merged the fixed palm link into ``*_wrist_yaw_link``, so the welds attach there; the
relative pose is computed from the same yourdfpy kinematics used by the viser check.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..paths import (ASSET_DIR, CART_OBJ, FINGER_JOINTS, G1_DESCRIPTION_DIR, G1_URDF, HOMIE_JOINTS,
                     INITIAL_POSE_YAML)
from ..pose import Kinematics, load_initial_pose
from .cart_urdf import cart_primitives, derive_cart_pose, load_cart_params

G1_XML = G1_DESCRIPTION_DIR / "g1.xml"
MJCF_DIR = ASSET_DIR / "mujoco"
MJCF_ROBOT = MJCF_DIR / "g1_grasp.xml"
MJCF_SCENE = MJCF_DIR / "g1_pushcart.xml"
MJCF_REPORT = MJCF_DIR / "mjcf_report.json"
WRIST_LINKS = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
CART_BODY = "cart_base"


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    q = R.from_matrix(matrix).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def _finger_axes() -> dict[str, np.ndarray]:
    root = ET.parse(G1_URDF).getroot()
    axes = {}
    for joint in root.findall("joint"):
        if joint.get("name") in FINGER_JOINTS:
            axes[joint.get("name")] = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
    return axes


def bake_fingers(root: ET.Element, finger_pos: dict[str, float]) -> int:
    """Rotate each fixed finger body by the recorded joint angle and disable its collisions."""
    axes = _finger_axes()
    count = 0
    for body in root.iter("body"):
        name = body.get("name", "")
        joint = name.replace("_link", "_joint")
        if joint not in finger_pos:
            continue
        rot = R.from_rotvec(axes[joint] / np.linalg.norm(axes[joint]) * float(finger_pos[joint]))
        if body.get("quat"):
            w, x, y, z = (float(v) for v in body.get("quat").split())
            base = R.from_quat([x, y, z, w])
        else:
            base = R.identity()
        body.set("quat", _fmt(_quat_wxyz((base * rot).as_matrix())))
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        count += 1
    return count


def _box_inertia(m, sx, sy, sz):
    return (m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2))


def cart_body_xml(params: dict, derived: dict) -> ET.Element:
    yaw = float(derived["base_yaw"])
    body = ET.Element("body", name=CART_BODY, pos=_fmt(derived["base_pos_world"]),
                      quat=_fmt([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]))
    ET.SubElement(body, "freejoint", name="cart_free")
    m = params["mass"]["base_kg"]
    ET.SubElement(body, "inertial", pos="0 0 0.3", mass=f"{m}", diaginertia=_fmt(_box_inertia(m, 1.0, 0.54, 0.6)))
    body_fric = f'{params["body"]["friction"]} 0.005 0.0001'
    for prim in cart_primitives(params, derived["handle_height"]):
        attrs = {"name": f"cart_{prim.name}", "pos": _fmt(prim.xyz)}
        if prim.kind == "box":
            attrs.update(type="box", size=_fmt([s / 2 for s in prim.size]))
        elif prim.kind == "cylinder":
            attrs.update(type="cylinder", size=_fmt([prim.size[0], prim.size[1] / 2]))
        else:
            attrs.update(type="sphere", size=_fmt([prim.size[0]]))
        if any(abs(v) > 1e-9 for v in prim.rpy):
            attrs["quat"] = _fmt(_quat_wxyz(R.from_euler("xyz", prim.rpy).as_matrix()))
        if prim.collision:
            attrs.update(friction=body_fric, rgba="0.45 0.55 0.75 0.35")
        else:
            attrs.update(contype="0", conaffinity="0", group="1", rgba="0.85 0.2 0.2 1")
        ET.SubElement(body, "geom", **attrs)
    vis = params["visual"]
    ET.SubElement(body, "geom", name="cart_visual", type="mesh", mesh="xiaotuiche", pos=_fmt(vis["offset"]),
                  contype="0", conaffinity="0", group="1", rgba="0.6 0.62 0.66 0.45")
    w = params["wheels"]
    r, width, trail = w["radius"], w["width"], w["caster_trail"]
    wheel_fric = f'{w["friction"]} 0.005 0.0001'
    for i, (x, y) in enumerate(w["positions_xy"]):
        swivel = ET.SubElement(body, "body", name=f"caster_{i}_swivel", pos=_fmt([x, y, r + 0.02]))
        ET.SubElement(swivel, "joint", name=f"caster_{i}_swivel_joint", type="hinge", axis="0 0 1",
                      damping=f'{w["swivel_damping"]}', armature="0.0005", frictionloss="0.01")
        sm = w["swivel_mass_kg"]
        ET.SubElement(swivel, "inertial", pos="0 0 0", mass=f"{sm}", diaginertia=_fmt(_box_inertia(sm, .04, .04, .04)))
        wheel = ET.SubElement(swivel, "body", name=f"wheel_{i}", pos=_fmt([-trail, 0., -0.02]))
        ET.SubElement(wheel, "joint", name=f"wheel_{i}_joint", type="hinge", axis="0 1 0",
                      damping=f'{w["wheel_damping"]}', armature="0.0005", frictionloss="0.02")
        wm = w["wheel_mass_kg"]
        ixx = wm / 12 * (3 * r**2 + width**2)
        ET.SubElement(wheel, "inertial", pos="0 0 0", mass=f"{wm}", diaginertia=_fmt([ixx, wm * r**2 / 2, ixx]))
        # Same collider type as the Isaac asset (cart_params.wheels.collider) so both simulators roll alike;
        # the cylinder stays as a visual-only geom when a sphere collider is used.
        if w.get("collider", "cylinder") == "sphere":
            ET.SubElement(wheel, "geom", name=f"wheel_{i}_geom", type="sphere", size=_fmt([r]),
                          friction=wheel_fric, rgba="0.15 0.15 0.15 0")
            ET.SubElement(wheel, "geom", name=f"wheel_{i}_visual", type="cylinder", size=_fmt([r, width / 2]),
                          quat="0.70710678 0.70710678 0 0", contype="0", conaffinity="0", group="1",
                          rgba="0.15 0.15 0.15 1")
        else:
            ET.SubElement(wheel, "geom", name=f"wheel_{i}_geom", type="cylinder", size=_fmt([r, width / 2]),
                          quat="0.70710678 0.70710678 0 0", friction=wheel_fric, rgba="0.15 0.15 0.15 1")
    return body


def build_mjcf(with_cart: bool, output: Path | None = None) -> dict:
    pose = load_initial_pose(INITIAL_POSE_YAML)
    params = load_cart_params()
    kin = Kinematics(load_meshes=False)
    derived = derive_cart_pose(params, pose, kin)
    kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])

    tree = ET.parse(G1_XML)
    root = tree.getroot()
    root.set("model", "g1_pushcart" if with_cart else "g1_grasp")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((G1_DESCRIPTION_DIR / "meshes").resolve()))
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.005")
    option.set("integrator", "implicitfast")
    baked = bake_fingers(root, pose["robot"]["finger_pos"])

    welds = {}
    if with_cart:
        asset = root.find("asset")
        ET.SubElement(asset, "mesh", name="xiaotuiche", file=str(CART_OBJ.resolve()),
                      scale=_fmt([params["visual"]["scale"]] * 3))
        worldbodies = root.findall("worldbody")
        worldbodies[-1].append(cart_body_xml(params, derived))
        cart_T = np.eye(4)
        cart_T[:3, :3] = R.from_euler("z", derived["base_yaw"]).as_matrix()
        cart_T[:3, 3] = derived["base_pos_world"]
        # Same as the packed USD: the robot (feet) never collides with the cart (a foot-caster contact would
        # close a rigid loop through the welds; the task keeps the feet away with a proximity penalty).
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        cart_bodies = [CART_BODY] + [f"caster_{i}_swivel" for i in range(4)] + [f"wheel_{i}" for i in range(4)]
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link"):
            for body in cart_bodies:
                ET.SubElement(contact, "exclude", body1=foot, body2=body)
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        for side, wrist in WRIST_LINKS.items():
            rel = np.linalg.inv(kin.world_T(wrist)) @ cart_T      # pose of body2 (cart) in body1 (wrist) frame
            relpose = np.concatenate([rel[:3, 3], _quat_wxyz(rel[:3, :3])])
            ET.SubElement(equality, "weld", name=f"grasp_{side}", body1=wrist, body2=CART_BODY,
                          relpose=_fmt(relpose), solref="0.004 1", solimp="0.95 0.99 0.001 0.5 2")
            welds[side] = {"body1": wrist, "body2": CART_BODY, "relpose": relpose.tolist()}

    output = output or (MJCF_SCENE if with_cart else MJCF_ROBOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode")
    report = verify_mjcf(output, pose, derived, with_cart)
    report.update({"baked_finger_bodies": baked, "welds": welds, "output": str(output)})
    return report


def preset_qpos(model, pose: dict, derived: dict | None):
    """qpos for the preset pose: robot free joint + 27 joints (+ cart free joint and passive casters at 0)."""
    import mujoco
    qpos = np.zeros(model.nq)
    free = model.joint("floating_base_joint")
    qpos[free.qposadr[0]:free.qposadr[0] + 3] = pose["robot"]["base_pos"]
    x, y, z, w = pose["robot"]["base_quat_xyzw"]
    qpos[free.qposadr[0] + 3:free.qposadr[0] + 7] = [w, x, y, z]
    for name in HOMIE_JOINTS:
        qpos[model.joint(name).qposadr[0]] = pose["robot"]["joint_pos"][name]
    if derived is not None and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cart_free") >= 0:
        cart = model.joint("cart_free")
        qpos[cart.qposadr[0]:cart.qposadr[0] + 3] = derived["base_pos_world"]
        yaw = derived["base_yaw"]
        qpos[cart.qposadr[0] + 3:cart.qposadr[0] + 7] = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
    return qpos


def verify_mjcf(path: Path, pose: dict, derived: dict, with_cart: bool) -> dict:
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    hinge = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    robot_hinges = [n for n in hinge if n in HOMIE_JOINTS]
    data.qpos[:] = preset_qpos(model, pose, derived if with_cart else None)
    mujoco.mj_forward(model, data)
    eq_violation = []
    for i in range(data.nefc):
        if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY:
            eq_violation.append(float(abs(data.efc_pos[i])))
    hand_geoms = {i for i in range(model.ngeom) if "hand" in model.body(model.geom_bodyid[i]).name}
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        contacts.append([model.geom(c.geom1).name or model.body(model.geom_bodyid[c.geom1]).name,
                         model.geom(c.geom2).name or model.body(model.geom_bodyid[c.geom2]).name, float(c.dist)])
    feet_z = [float(data.body(n).xpos[2]) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    report = {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "njnt": int(model.njnt),
        "robot_hinge_order_matches_homie": robot_hinges == HOMIE_JOINTS,
        "actuator_order": [model.actuator(i).name for i in range(model.nu)],
        "timestep": float(model.opt.timestep),
        "equality_max_violation_at_preset": max(eq_violation) if eq_violation else None,
        "contacts_at_preset": contacts,
        "hand_geoms_colliding": sum(1 for g in hand_geoms if model.geom_contype[g] or model.geom_conaffinity[g]),
        "feet_z_at_preset": feet_z,
        "root_height": float(data.body("pelvis").xpos[2]),
    }
    if with_cart:
        wheel_ids = [i for i in range(model.nbody) if model.body(i).name.startswith("wheel_")]
        report["wheel_bottom_z"] = [float(data.xpos[i][2]) - 0.06 for i in wheel_ids]
        report["cart_pos"] = data.body(CART_BODY).xpos.tolist()
        report["robot_cart_contact_excludes"] = int(model.nexclude)
    return report


def build_all() -> dict:
    reports = {"robot": build_mjcf(False), "cart": build_mjcf(True)}
    MJCF_REPORT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
