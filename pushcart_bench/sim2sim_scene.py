"""MuJoCo scene parameters sourced from the V5 Isaac training assets.

The legacy scene remains available for reproducing published benchmarks. This
profile fixes identifiable model differences; it does not equate MuJoCo's soft
contact/weld solver with PhysX's TGS solver. No parameter is fitted to estimator
predictions or benchmark outcomes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from homie_isaaclab.config import G1RoughCfg
from homie_pushcart_v52.paths import G1_GRASP_URDF, CART_URDF, ROOT, HOMIE_JOINTS

SCENE_REVISION = 6
LEGACY_SCENE = ROOT / "assets/g1_pushcart/mujoco/g1_pushcart.xml"
CACHE = ROOT / ".cache/pushcart_sim2sim"
SCENE_VERSIONS = ("original", "four_spheres")
# The official G1 contact spheres, in each ankle-roll link's coordinates.
# Version this separately so the original scene's existing cache key is stable.
FOOT_CONTACT_REVISION = 1
FOOT_CONTACT_POINTS = ((-.05, .025, -.03), (-.05, -.025, -.03),
                       (.12, .03, -.03), (.12, -.03, -.03))


def normalize_scene_version(version: str) -> str:
    version = "four_spheres" if version == "new" else version
    if version not in SCENE_VERSIONS:
        raise ValueError(f"Unknown MuJoCo scene version: {version}; choose original or four_spheres")
    return version


def _use_four_sphere_feet(root: ET.Element):
    """Change only the foot contact representation; retain the 0.75 floor pair friction."""
    contact = root.find("contact")
    old_names = {f"isaac_{side}_ankle_roll_link_0" for side in ("left", "right")}
    for name in old_names:
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"Missing original foot collision: {name}")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for element in list(contact):
        if element.tag == "pair" and (element.get("geom1") in old_names or element.get("geom2") in old_names):
            contact.remove(element)
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        for i, position in enumerate(FOOT_CONTACT_POINTS):
            name = f"{side}_foot_contact_{i}"
            ET.SubElement(body, "geom", name=name, type="sphere", size="0.005", pos=_fmt(position),
                          contype="1", conaffinity="1", condim="3", rgba="0.2 0.2 0.2 1")
            ET.SubElement(contact, "pair", name=f"{name}_floor", geom1=name, geom2="floor",
                          condim="3", friction="0.75 0.75 0 0 0")


def _fmt(x):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(x).reshape(-1))


def _origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return t


def urdf_inertias(urdf: Path, body_names: set[str], rename=None, properties=None):
    """Aggregate fixed URDF links into their actual MJCF bodies, at the COM.

    This preserves complete inertia tensors and parallel-axis terms, including
    palms/head omitted as separate bodies in HOMIE's original MJCF.
    """
    root = ET.parse(urdf).getroot()
    if properties is not None:
        names = {link.get('name') for link in root.findall('link') if link.find('inertial') is not None}
        if names != set(properties):
            raise ValueError('USD and URDF rigid-body sets changed; audit fixed-link aggregation')
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    rename = rename or {}
    grouped = {}
    for link in root.findall("link"):
        inert = link.find("inertial")
        if inert is None:
            continue
        mass = float(inert.find("mass").get("value"))
        name, transform = link.get("name"), np.eye(4)
        source_name = name
        while rename.get(name, name) not in body_names:
            joint = parents[name]
            if joint.get("type") != "fixed":
                raise ValueError(f"Missing movable URDF body {name}")
            transform = _origin(joint.find("origin")) @ transform
            name = joint.find("parent").get("link")
        if properties is None:
            transform = transform @ _origin(inert.find("origin"))
            a = inert.find("inertia").attrib
            tensor = np.array([[float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
                               [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
                               [float(a["ixz"]), float(a["iyz"]), float(a["izz"])]])
        else:
            prop = properties[source_name]
            mass = prop['mass']
            transform[:3, 3] += transform[:3, :3] @ np.asarray(prop['com'])
            tensor = np.asarray(prop['inertia'])
        rot = transform[:3, :3]
        grouped.setdefault(rename.get(name, name), []).append((mass, transform[:3, 3], rot @ tensor @ rot.T))
    result = {}
    for name, parts in grouped.items():
        mass = sum(p[0] for p in parts)
        com = sum(m * c for m, c, _ in parts) / mass
        tensor = sum(i + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com))
                     for m, c, i in parts)
        result[name] = (mass, com, tensor)
    return result


def build_isaac_scene(cfg=None, upper_drive="training", inertia_profile="urdf", scene_version="original") -> Path:
    """Generate an isolated, content-addressed MJCF without modifying training assets."""
    from homie_pushcart_v52.control_contract import validate_upper_drive
    validate_upper_drive(upper_drive)
    scene_version = normalize_scene_version(scene_version)
    if inertia_profile not in ("urdf", "physx"):
        raise ValueError(f"Unknown inertia profile: {inertia_profile}")
    properties, reference_bytes = {}, b""
    if inertia_profile == "physx":
        from .physx_inertia import REFERENCE, usd_mass_properties
        properties = usd_mass_properties()['bodies']
        reference_bytes = REFERENCE.read_bytes()
    cfg = G1RoughCfg() if cfg is None else cfg
    signature = repr((SCENE_REVISION, cfg.control.stiffness, cfg.control.damping,
                      cfg.asset.armature, cfg.sim.dt, cfg.sim.gravity, upper_drive, inertia_profile)).encode()
    if scene_version != "original":
        signature += repr((scene_version, FOOT_CONTACT_REVISION, FOOT_CONTACT_POINTS)).encode()
    digest = hashlib.sha256(signature + reference_bytes + b"".join(p.read_bytes() for p in (LEGACY_SCENE, G1_GRASP_URDF, CART_URDF))).hexdigest()[:16]
    output = CACHE / f"isaac_{digest}.xml"
    if output.is_file():
        return output
    root = ET.parse(LEGACY_SCENE).getroot()
    root.set("model", "g1_pushcart_isaac_parameters")
    root.find("option").set("timestep", str(cfg.sim.dt))
    root.find("option").set("gravity", _fmt(cfg.sim.gravity))
    root.find("option").set("integrator", "implicitfast")
    # No free-base rotor inertia, passive robot joint damping or Coulomb
    # friction exists in the Isaac articulation. Declare actual hinges below.
    default_joint = root.find("default/joint")
    default_joint.set("damping", "0")
    default_joint.set("frictionloss", "0")
    default_joint.set("armature", "0")
    bodies = {b.get("name"): b for b in root.iter("body")}
    inertias = urdf_inertias(G1_GRASP_URDF, set(bodies), properties=properties.get('robot'))
    inertias.update(urdf_inertias(CART_URDF, set(bodies), {"base_link": "cart_base"}, properties.get('cart')))
    for name, (mass, com, tensor) in inertias.items():
        body = bodies[name]
        old = body.find("inertial")
        if old is not None:
            body.remove(old)
        ET.SubElement(body, "inertial", mass=str(mass), pos=_fmt(com),
                      fullinertia=_fmt(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
    # HOMIE's general-purpose g1.xml has collision meshes on arms/torso
    # and four foot spheres; the training URDF has ONLY two foot meshes.
    # Preserve visuals, replace the robot's collision model from that URDF.
    for name, body in bodies.items():
        if name == "cart_base" or name.startswith(("wheel_", "caster_")):
            continue
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    foot_geoms = []
    for link in ET.parse(G1_GRASP_URDF).getroot().findall("link"):
        for index, collision in enumerate(link.findall("collision")):
            transform = _origin(collision.find("origin"))
            name = f'isaac_{link.get("name")}_{index}'
            geometry = collision.find("geometry")
            box, mesh = geometry.find("box"), geometry.find("mesh")
            if box is not None:
                shape = dict(type="box", size=_fmt(np.fromstring(box.get("size"), sep=" ") / 2.))
            elif mesh is not None:
                source = (G1_GRASP_URDF.parent / mesh.get("filename")).resolve()
                ET.SubElement(root.find("asset"), "mesh", name=name, file=str(source), scale=mesh.get("scale", "1 1 1"))
                shape = dict(type="mesh", mesh=name)
            else:
                raise ValueError("Training robot collision geometry changed; audit its MuJoCo representation")
            ET.SubElement(bodies[link.get("name")], "geom", name=name, **shape, pos=_fmt(transform[:3, 3]),
                          quat=_fmt(np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)),
                          contype="1", conaffinity="1", condim="3", friction="0.5 0 0", rgba="0.3 0.3 0.3 0")
            foot_geoms.append(name)
    joints = {j.get("name"): j for j in root.iter("joint") if j.get("name")}
    urdf = ET.parse(G1_GRASP_URDF).getroot()
    for source in urdf.findall("joint"):
        name = source.get("name")
        if name not in HOMIE_JOINTS:
            continue
        joint, limits = joints[name], source.find("limit")
        joint.set("range", f'{limits.get("lower")} {limits.get("upper")}')
        joint.set("armature", str(cfg.asset.armature))
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
    for source in ET.parse(CART_URDF).getroot().findall("joint"):
        joint = joints[source.get("name")]
        joint.set("armature", "0")
        joint.set("frictionloss", "0")
        joint.set("damping", source.find("dynamics").get("damping"))
    # The default uses native position servos (implicit damping in implicitfast)
    # above the legs. The explicit_pd experiment retains upper motor actuators.
    # Matching the mixed interface does not imply matching PhysX numerics.
    actuators = root.find("actuator")
    for motor in list(actuators):
        name = motor.get("name")
        if name not in HOMIE_JOINTS[12:] or upper_drive == "explicit_pd":
            continue
        key = next(k for k in cfg.control.stiffness if k in name)
        limit = urdf.find(f"joint[@name='{name}']/limit")
        attrs = dict(name=name, joint=name, kp=str(cfg.control.stiffness[key]), kv=str(cfg.control.damping[key]),
                     forcelimited="true", forcerange=f'-{limit.get("effort")} {limit.get("effort")}',
                     ctrllimited="false")
        actuators.remove(motor)
        ET.SubElement(actuators, "position", **attrs)
    # PhysX averages the wheel and floor sliding coefficients. Explicit pairs
    # prevent MuJoCo's max-combination and extra rolling/torsional friction.
    contact = root.find("contact")
    ET.SubElement(contact, "exclude", body1="left_ankle_roll_link", body2="right_ankle_roll_link")
    for name in foot_geoms:
        # Runtime PhysX audit: imported foot material 0.5, ground 1.0;
        # the default average combine mode gives 0.75.
        ET.SubElement(contact, "pair", name=f"{name}_floor", geom1=name, geom2="floor",
                      condim="3", friction="0.75 0.75 0 0 0")
    for i in range(4):
        geom = root.find(f".//geom[@name='wheel_{i}_geom']")
        mu = float(geom.get("friction").split()[0])
        ET.SubElement(contact, "pair", name=f"wheel_floor_{i}", geom1=f"wheel_{i}_geom", geom2="floor",
                      condim="3", friction=_fmt([(mu + 1.) / 2.] * 2 + [0., 0., 0.]))
    if scene_version == "four_spheres":
        _use_four_sphere_feet(root)
    # Resolve mesh paths before moving the scene to a separate cache directory.
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "."))
    if not meshdir.is_absolute():
        compiler.set("meshdir", str((LEGACY_SCENE.parent / meshdir).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(output))  # Validate before returning a cached asset.
    return output


def payload_properties(base_mass, base_com, base_inertia, payload, offset=(0., 0.)):
    """Exact compact-box/parallel-axis model in V5.x PushCartEnv._randomize_cart."""
    if not np.isfinite(payload) or payload < 0:
        raise ValueError("Payload must be finite and nonnegative")
    d = np.r_[np.asarray(offset, dtype=float), 0.]
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("Payload offset must contain two finite coordinates")
    total = base_mass + payload
    box = np.diag([.30**2 + .20**2, .40**2 + .20**2, .40**2 + .30**2]) / 12.
    inertia = base_inertia + payload * box + base_mass * payload / total * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total, base_com + payload / total * d, inertia


def set_payload(env, payload, offset=(0., 0.)):
    m, b = env.model, env.cart_id
    if not hasattr(env, "_payload_base"):
        rot = Rotation.from_quat(np.roll(m.body_iquat[b], -1)).as_matrix()
        env._payload_base = (float(m.body_mass[b]), m.body_ipos[b].copy(), rot @ np.diag(m.body_inertia[b]) @ rot.T)
    mass, com, inertia = payload_properties(*env._payload_base, payload, offset)
    if getattr(env, 'inertia_profile', 'urdf') == 'physx':
        from .physx_inertia import cart_reset_inertia, usd_mass_properties
        inertia = cart_reset_inertia(inertia)
        # Isaac resets inertias and COM poses for ALL cart bodies, including
        # wheels. Their unsorted principal axes undergo the same replacement.
        for name, prop in usd_mass_properties()['bodies']['cart'].items():
            if name == 'base_link':
                continue
            body = m.body(name).id
            values, vectors = np.linalg.eigh(cart_reset_inertia(prop['inertia'], name))
            if np.linalg.det(vectors) < 0:
                vectors[:, 0] *= -1
            m.body_inertia[body] = values
            m.body_iquat[body] = np.roll(Rotation.from_matrix(vectors).as_quat(), 1)
    diagonal, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    m.body_mass[b], m.body_ipos[b], m.body_inertia[b] = mass, com, diagonal
    m.body_iquat[b] = np.roll(Rotation.from_matrix(axes).as_quat(), 1)
    mujoco.mj_setConst(m, env.data)


def set_grasp_profile(env, profile="nominal"):
    """Select physical grasp alignment without modifying XML assets or weights.

    ``rigid_limit`` uses MuJoCo's maximum allowed constraint impedance for
    comparison with PhysX fixed joints. It was chosen using a prescribed-target,
    no-policy single/dual-grasp fixture, not estimator scores. Contact, timestep,
    PD gains and solref stay unchanged. This is an explicit experimental profile;
    the policy runner defaults to nominal pending joint-mapping review.
    """
    if profile not in ("nominal", "rigid_limit"):
        raise ValueError(f"Unknown grasp profile: {profile}")
    if profile == "rigid_limit" and env.scene_profile != "isaac":
        raise ValueError("Grasp alignment requires the aligned scene")
    if not hasattr(env, "_nominal_grasp_solimp"):
        env._nominal_grasp_solimp = env.model.eq_solimp.copy()
    env.model.eq_solimp[:] = env._nominal_grasp_solimp
    if profile == "rigid_limit":
        for side in ("left", "right"):
            env.model.eq_solimp[env.model.eq(f"grasp_{side}").id, :2] = .9999
    env.grasp_profile = profile


def model_report(env):
    m = env.model
    from homie_pushcart_v52.control_contract import control_contract
    from .physx_inertia import REFERENCE
    return dict(profile=env.scene_profile, revision=SCENE_REVISION, xml=str(env.xml_path),
                scene_version=getattr(env, "scene_version", "original"),
                inertia_profile=getattr(env, "inertia_profile", "urdf"),
                inertia_reference_sha256=(hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
                                          if getattr(env, "inertia_profile", "urdf") == "physx" else None),
                grasp_profile=getattr(env, "grasp_profile", "nominal"),
                grasp_constraints=[dict(name=m.eq(i).name, solref=m.eq_solref[i].tolist(),
                                        solimp=m.eq_solimp[i].tolist()) for i in range(m.neq)],
                robot_mass_kg=float(m.body_subtreemass[env.pelvis_id]),
                cart_assembly_mass_kg=float(m.body_subtreemass[env.cart_id]),
                robot_armature=m.dof_armature[env.qvel_adr].tolist(),
                robot_passive_damping=m.dof_damping[env.qvel_adr].tolist(),
                robot_frictionloss=m.dof_frictionloss[env.qvel_adr].tolist(),
                upper_actuation=("native_position_pd" if env.scene_profile == "isaac" and env.upper_drive == "training"
                                 else "explicit_effort_pd"),
                control=control_contract(env.cfg, HOMIE_JOINTS, env.torque_limits,
                                         env.upper_drive if env.scene_profile == "isaac" else "explicit_pd"),
                collision_bodies=sorted({m.body(int(m.geom_bodyid[g])).name for g in range(m.ngeom)
                                         if m.geom_contype[g] or m.geom_conaffinity[g]}),
                physics_dt=float(m.opt.timestep), control_dt=env.dt,
                remaining_differences=["PhysX TGS vs MuJoCo contact/weld solver and drive discretization",
                                       "PhysX numerical hand-subtree labels vs exact MuJoCo weld loads",
                                       "nominal deterministic evaluation excludes training domain randomization"])


def newton_euler_hand_wrenches(env, pre_step_velocity):
    """Independent arm inverse-dynamics check of the weld labels, world/palm axes.

    Computes tau_act + tau_passive + tau_external - RNE(q,dq,ddq), then
    solves J_arm.T w = residual. It never reads efc_force or qfrc_constraint.
    This is an RNE balance audit, not a fabricated PhysX joint-wrench sensor.
    Use only the aligned scene: no other arm contacts or joint-limit contacts.
    MuJoCo RNE excludes rotor inertia, so armature*ddq must be included.
    mj_step has already integrated qvel, while the solver's cached transforms
    and accelerations belong to the PRE-step sample: restore that velocity in
    a private data copy, never in the live simulation.
    """
    import copy
    m, d = env.model, env.data
    sample = copy.copy(d)
    sample.qvel[:] = pre_step_velocity
    inverse = np.empty(m.nv)
    mujoco.mj_rne(m, sample, 1, inverse)
    inverse += m.dof_armature * d.qacc
    residual = d.qfrc_actuator + d.qfrc_passive + d.qfrc_applied - inverse
    for b in range(1, m.nbody):
        if not np.any(d.xfrc_applied[b]):
            continue
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jp, jr, b)
        residual += jp.T @ d.xfrc_applied[b, :3] + jr.T @ d.xfrc_applied[b, 3:]
    result = []
    for i, side in enumerate(("left", "right")):
        body = env.wrist_ids[side]
        point = env.sub_palm[-1][i]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, point, body)
        ids = env.qvel_adr[13 + 7*i:20 + 7*i]
        jac = np.vstack((jp[:, ids], jr[:, ids]))
        result.append(np.linalg.lstsq(jac.T, residual[ids], rcond=None)[0])
    return np.array(result)
