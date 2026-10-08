"""V7 physical plant: V3.2 robot, control and cart geometry; slide/roll grasp chain and Coulomb cart resistance.

Assets are built by ``homie_pushcart_v7.assets.build`` and hashed into ``build_manifest.json``; every
environment, training run and benchmark validates against that manifest, so a rebuilt asset can never be
mixed silently with an older checkpoint's records.
"""
from __future__ import annotations

import hashlib

import numpy as np
import json

from homie_pushcart.paths import G1_GRASP_URDF, G1_URDF, INITIAL_POSE_YAML

from .paths import (ASSET_REPORT, BUILD_MANIFEST, CART_PARAMS_V7_YAML, CART_URDF, G1_GRASP_V7_URDF,
                    GRASP_POSE_V7_YAML, MJCF_SCENE, PACKED_USD, ROOT)

PHYSICS_ID = "v7_slide_roll_grasp_coulomb_cart_v2"      # v2: rolling-dependent swivel scrub
PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
FOOT_CONTACT_POINTS = ((-.05, .025, -.03), (-.05, -.025, -.03),
                       (.12, .03, -.03), (.12, -.03, -.03))
# Robot-side inputs are shared with V3.2/V6 and must not change underneath V7 either.
_SHARED_SHA256 = {
    INITIAL_POSE_YAML: "907fce1062dc4ace7cbca72ab1de158eb625093225e7ad7dce3dfca56a73080b",
    G1_URDF: "449b618e3c0e640d600e73e6cc9072d06b62a8c56c393bbb83aa46b32a9e2153",
    G1_GRASP_URDF: "528256bc2676f26d4c250f244575d3205bf0e4d1f338ead860e35b51591560c9",
}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_assets(*, require_usd: bool = True) -> dict[str, str]:
    """Reject asset drift against the build manifest; return verified repository-relative SHA256 records."""
    verified = {}
    for path, expected in _SHARED_SHA256.items():
        actual = _sha(path)
        if actual != expected:
            raise ValueError(f"V7 shared robot input changed: {path}; expected {expected}, got {actual}")
        verified[path.relative_to(ROOT).as_posix()] = actual
    if not BUILD_MANIFEST.is_file():
        raise FileNotFoundError(f"{BUILD_MANIFEST} missing; run `pushcart_v7.py build all` first")
    manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    required = [CART_URDF, CART_PARAMS_V7_YAML, GRASP_POSE_V7_YAML, G1_GRASP_V7_URDF, MJCF_SCENE] + (
        [PACKED_USD] if require_usd else [])
    for path in required:
        key = path.relative_to(ROOT).as_posix()
        if key not in manifest:
            raise ValueError(f"V7 build manifest has no record of {key}; rebuild the V7 assets")
    for key, expected in manifest.items():
        actual = _sha(ROOT / key)
        if actual != expected:
            raise ValueError(f"V7 asset changed after build: {key}; rebuild with `pushcart_v7.py build all`")
        verified[key] = actual
    return verified


def physics_manifest(*, require_usd: bool = True) -> dict:
    """Serializable physical provenance for training, playback and bench logs."""
    assets = validate_assets(require_usd=require_usd)
    return {
        "physics_id": PHYSICS_ID,
        "mjcf_scene": str(MJCF_SCENE),
        "packed_usd": str(PACKED_USD),
        "asset_report": str(ASSET_REPORT),
        "asset_sha256": assets,
        "physics_dt_s": PHYSICS_DT,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt_s": PHYSICS_DT * CONTROL_DECIMATION,
        "foot_contact": {"spheres_per_foot": 4, "radius_m": .005,
                         "positions_m": [list(p) for p in FOOT_CONTACT_POINTS]},
        "grasp": "palm FixedJoint/weld to a per-hand slide+roll/tilt/yaw chain on the bar axis; elasto-plastic slide "
                 "friction (anchor spring capped at the load-dependent Coulomb limit), elastic-then-yield pivots",
        "grasp_geometry": "Dex3 fingers closed onto the 22 mm tube (assets/grasp_fit.py, g1_grasp_v7.urdf); "
                          "hand-bar contact disabled, the chain is the contact model",
        "cart_resistance": "Coulomb wheel rolling resistance; swivel friction = tyre scrub at standstill decaying with "
                           "the wheel rolling speed to the swivel bearing friction (+ small viscous), per env",
        "isaac_drive": "V3.2: leg explicit PD effort, upper implicit position drive",
        "mujoco_drive": "V3.2: all-joint explicit PD, recomputed every physics step",
        "isaac_source": "homie_pushcart_v7.task_env.PushCartEnv",
        "mujoco_source": "homie_pushcart.mujoco_env.MujocoHomieEnv on the V7 MJCF",
    }


def _validate_mujoco_config(cfg) -> None:
    from homie_isaaclab.config import G1RoughCfg

    baseline = G1RoughCfg()
    for section, fields in {
        "sim": ("dt", "gravity"),
        "control": ("control_type", "decimation", "stiffness", "damping", "action_scale"),
        "init_state": ("default_joint_angles",),
        "asset": ("armature",),
    }.items():
        for field in fields:
            if getattr(getattr(cfg, section), field) != getattr(getattr(baseline, section), field):
                raise ValueError(f"V7 physical baseline forbids overriding {section}.{field}")


def make_mujoco_env(*, cfg=None):
    """The V3.2 MuJoCo HOMIE environment on the V7 plant (CPU; no USD needed)."""
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart.assets.mjcf_scene import preset_qpos
    from homie_pushcart.mujoco_env import MujocoHomieEnv
    from .assets.build import grasp_geometry, load_v7_params, load_v7_pose

    validate_assets(require_usd=False)
    cfg = G1RoughCfg() if cfg is None else cfg
    _validate_mujoco_config(cfg)
    env = MujocoHomieEnv(with_cart=True, xml_path=MJCF_SCENE, decimation=CONTROL_DECIMATION, cfg=cfg)
    # The V3.2 env reads the V3.2 pose/cart files; V7 places the cart from its fitted grasp.
    env.pose, env.cart_params = load_v7_pose(), load_v7_params()
    env.derived = grasp_geometry(env.cart_params, env.pose)["derived"]
    env.preset = preset_qpos(env.model, env.pose, env.derived)
    return env


def swivel_torque(scrub, bearing, rim_speed, scrub_speed):
    """Swivel Coulomb torque of a caster: tyre bore (scrub) torque at standstill, decaying with the wheel's rolling
    speed to the swivel bearing friction (cart_params_v7.yaml wheels_v7). Works on floats, numpy arrays and torch
    tensors (same law in the Isaac task and the MuJoCo plant)."""
    try:
        import torch
        if isinstance(rim_speed, torch.Tensor):
            return bearing + (scrub - bearing) * torch.exp(-rim_speed.abs() / scrub_speed)
    except ImportError:
        pass
    return bearing + (scrub - bearing) * np.exp(-np.abs(rim_speed) / scrub_speed)


class MujocoCasters:
    """Per-control-step swivel friction of the four casters on the V7 MJCF (rolling-dependent scrub)."""

    def __init__(self, env, params, scrub=None, bearing=None):
        w = params["wheels_v7"]
        mean = lambda r: 0.5 * (float(r[0]) + float(r[1]))                                   # noqa: E731
        m = env.model
        self.m, self.d = m, env.data
        self.swivel_dof = [int(m.joint(f"caster_{i}_swivel_joint").dofadr[0]) for i in range(4)]
        self.wheel_dof = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self.radius = float(params["wheels"]["radius"])
        self.scrub = float(scrub if scrub is not None else mean(w["swivel_friction_range"]))
        self.bearing = float(bearing if bearing is not None else mean(w["swivel_bearing_range"]))
        self.scrub_speed = float(w["swivel_scrub_speed_m_s"])
        self.rolling_coeff = mean(w["rolling_coeff_range"])
        self.update()

    def set_cart_mass(self, total_kg):
        """Coulomb rolling resistance of each wheel for the cart's total mass (the Isaac task's law:
        c_rr * m g / 4 * r). The MJCF carries the value for 12 + 5 kg; call after every payload change."""
        tau = self.rolling_coeff * float(total_kg) * 9.81 / 4 * self.radius
        for a in self.wheel_dof:
            self.m.dof_frictionloss[a] = tau
        return tau

    def update(self):
        rim = np.array([self.d.qvel[a] for a in self.wheel_dof]) * self.radius
        tau = swivel_torque(self.scrub, self.bearing, rim, self.scrub_speed)
        for k, a in enumerate(self.swivel_dof):
            self.m.dof_frictionloss[a] = float(tau[k])
        return tau


def drag_anchor(anchor, q, limit, k):
    """Elasto-plastic (reset-integrator) friction: drag the spring anchor so that k |anchor - q| <= limit.
    Returns the new anchor and whether it moved (sliding). Same law as PushCartEnv._update_slide_anchor."""
    e = np.asarray(anchor) - np.asarray(q)
    reach = np.asarray(limit) / k
    over = np.abs(e) > reach
    return np.where(over, np.asarray(q) + np.sign(e) * reach, anchor), over


class MujocoGrasp:
    """Per-episode grasp parameters and the per-step elasto-plastic slide friction on the V7 MJCF.

    Same laws as the Isaac task: slide actuator force = k (anchor - q) - d qdot clamped to +-limit, limit =
    mu * (N + |load across the tube|) + offset (floored; x dynamic ratio while sliding), the anchor dragged so the
    spring never exceeds the limit; roll / tilt / yaw actuators: force = -k q - d qdot clamped to +-capacity.
    """

    def __init__(self, env, params):
        import mujoco
        from .paths import ROTATION_JOINTS, SIDES, SLIDE_JOINTS, GRASP_LINKS
        self.env, self.m, self.d, self.params, self.sides = env, env.model, env.data, params, SIDES
        self.mj = mujoco
        self.slide_dof = [int(self.m.joint(SLIDE_JOINTS[s]).dofadr[0]) for s in SIDES]
        self.slide_qpos = [int(self.m.joint(SLIDE_JOINTS[s]).qposadr[0]) for s in SIDES]
        self.rot_qpos = {r: [int(self.m.joint(ROTATION_JOINTS[r][s]).qposadr[0]) for s in SIDES] for r in ROTATION_JOINTS}
        self.rot_dof = {r: [int(self.m.joint(ROTATION_JOINTS[r][s]).dofadr[0]) for s in SIDES] for r in ROTATION_JOINTS}
        self.rot_act = {r: [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ROTATION_JOINTS[r][s]}_servo")
                            for s in SIDES] for r in ROTATION_JOINTS}
        self.slide_act = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{SLIDE_JOINTS[s]}_servo") for s in SIDES]
        if min(self.slide_act) < 0:
            raise ValueError("V7 MJCF lacks the slide actuators; rebuild it (pushcart_v7.py build mjcf)")
        self.grasp_body = [self.m.body(GRASP_LINKS[s]).id for s in SIDES]
        self.cart = env.cart_id
        self.anchor = None
        from .assets.build import nominal_grasp
        nom = nominal_grasp(params)
        self.configure(nom["mu"], nom["normal"], nom["stiffness"], nom["roll_capacity"], nom["moment_capacity"])

    def configure(self, mu, normal, stiffness, roll_capacity, moment_capacity, load_dependent=True,
                  slide_stiffness=None, dynamic_ratio=1.0, roll_stiffness=None):
        """stiffness: tilt / yaw; roll_stiffness (about the tube axis) defaults to the middle of its own range."""
        from .assets.build import slide_damping
        g = self.params["grasp"]
        self.mu, self.normal, self.load_dependent = float(mu), float(normal), bool(load_dependent)
        self.slide_k = float(slide_stiffness if slide_stiffness is not None else 0.5 * sum(g["slide_stiffness_range"]))
        self.dynamic_ratio = float(dynamic_ratio)
        for a in self.slide_act:
            self.m.actuator_gainprm[a, 0] = self.slide_k
            self.m.actuator_biasprm[a, :3] = [0., -self.slide_k, -float(slide_damping(self.params, self.slide_k))]
        kr = float(roll_stiffness if roll_stiffness is not None else 0.5 * sum(g["roll_stiffness_range"]))
        for r, acts in self.rot_act.items():
            cap = roll_capacity if r == "roll" else moment_capacity
            k = kr if r == "roll" else float(stiffness)
            damping = 2.0 * float(g["rotation_damping_ratio"]) * (k * float(g["rotation_armature"])) ** 0.5
            for a in acts:
                self.m.actuator_biasprm[a, :3] = [0., -k, -damping]
                self.m.actuator_forcerange[a] = [-float(cap), float(cap)]
        self.update(np.zeros((2, 6)))

    def slide_limit(self, load_across):
        g = self.params["grasp"]
        load = np.asarray(load_across) if self.load_dependent else np.zeros(2)
        return np.maximum(self.mu * (self.normal + load) + float(g["slide_offset_n"]), float(g["slide_min_n"]))

    def reset_anchor(self):
        """Unloaded spring at the current grip (call after every episode reset)."""
        self.anchor = np.array([self.d.qpos[a] for a in self.slide_qpos])
        self.slip = np.zeros(2)
        for k, a in enumerate(self.slide_act):
            self.d.ctrl[a] = self.anchor[k]

    def update(self, hand_loads):
        """hand_loads (2, 6): [F, M] each hand applies at its grasp link, cart axes. Sets next step's anchor / limit."""
        self.slide_static = self.slide_limit(np.hypot(hand_loads[:, 0], hand_loads[:, 2]))
        q = np.array([self.d.qpos[a] for a in self.slide_qpos])
        if self.anchor is None:
            self.anchor = q.copy()
        new, over = drag_anchor(self.anchor, q, self.slide_static, self.slide_k)
        self.slip, self.anchor, self.sliding = new - self.anchor, new, over
        self.slide_limit_now = np.where(over, self.slide_static * self.dynamic_ratio, self.slide_static)
        for k, a in enumerate(self.slide_act):
            self.d.ctrl[a] = self.anchor[k]
            self.m.actuator_forcerange[a] = [-self.slide_limit_now[k], self.slide_limit_now[k]]

    def hand_loads(self) -> np.ndarray:
        """Per hand [Fx Fy Fz Mx My Mz] on the cart at the grasp link, cart axes (x fwd, y = tube axis, z up)."""
        W = self.env.grasp_cart_wrenches()
        R = self.d.xmat[self.cart].reshape(3, 3)
        com = self.d.subtree_com[self.cart]
        out = []
        for k, s in enumerate(self.sides):
            g = self.d.xpos[self.grasp_body[k]]
            F, M = W[s][:3], W[s][3:] - np.cross(g - com, W[s][:3])
            out.append(np.concatenate((R.T @ F, R.T @ M)))
        return np.array(out)

    def pivot_moments(self) -> np.ndarray:
        """(3 roll/tilt/yaw, 2 hands) load torque the hand's weld puts on each pivot joint: J^T f of that weld's
        constraint rows, read straight from MuJoCo's generalized forces (no frame conversion)."""
        m, d = self.m, self.d
        out = np.zeros((3, 2))
        for k, s in enumerate(self.sides):
            eq = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_EQUALITY, f"grasp_{s}")
            rows = (d.efc_type == self.mj.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq)
            generalized = np.zeros(m.nv)
            self.mj.mj_mulJacTVec(m, d, generalized, np.where(rows, d.efc_force, 0.0))
            for i, r in enumerate(("roll", "tilt", "yaw")):
                out[i, k] = abs(float(generalized[self.rot_dof[r][k]]))
        return out

    def state(self) -> dict:
        q, v = self.d.qpos, self.d.qvel
        return {"slide_q": np.array([q[a] for a in self.slide_qpos]),
                "slide_anchor": None if self.anchor is None else self.anchor.copy(),
                "slide_v": np.array([v[a] for a in self.slide_dof]),
                "rot_q": np.array([[q[a] for a in self.rot_qpos[r]] for r in ("roll", "tilt", "yaw")]),
                "rot_v": np.array([[v[a] for a in self.rot_dof[r]] for r in ("roll", "tilt", "yaw")])}
