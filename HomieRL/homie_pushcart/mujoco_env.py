"""MuJoCo sim2sim environment: HOMIE's observation / PD pipeline on g1_grasp.xml or g1_pushcart.xml.

Mirrors HomieIsaacLabEnv step-for-step with the training configuration (G1RoughCfg):
76-dim frame x 6 history, commands scaled by [2, 2, 0.5], height 0.74, PD gains from
cfg.control, action scale 0.25, 50 Hz control over 4 x 5 ms physics steps. The upper body is
held at the preset grasp pose (plus an optional correction) through the same PD gains.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from .assets.cart_urdf import derive_cart_pose, load_cart_params
from .assets.mjcf_scene import CART_BODY, MJCF_ROBOT, MJCF_SCENE, preset_qpos
from .paths import HOMIE_JOINTS, ROOT
from .pose import Kinematics, load_initial_pose

if str(ROOT / "HomieRL") not in sys.path:
    sys.path.insert(0, str(ROOT / "HomieRL"))


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xyz = np.array([x, y, z])
    return v * (2 * w * w - 1) - 2 * w * np.cross(xyz, v) + 2 * xyz * np.dot(xyz, v)


class MujocoHomieEnv:
    def __init__(self, with_cart: bool, xml_path: Path | None = None, decimation: int = 4):
        from homie_isaaclab.config import G1RoughCfg

        self.cfg = G1RoughCfg()
        self.with_cart = with_cart
        self.xml_path = Path(xml_path) if xml_path else (MJCF_SCENE if with_cart else MJCF_ROBOT)
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"{self.xml_path} missing; run `pushcart.py mj build`")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.decimation = decimation
        self.dt = float(self.model.opt.timestep) * decimation

        self.pose = load_initial_pose()
        self.cart_params = load_cart_params()
        self.derived = derive_cart_pose(self.cart_params, self.pose, Kinematics(load_meshes=False)) if with_cart else None

        m = self.model
        self.qpos_adr = np.array([m.joint(n).qposadr[0] for n in HOMIE_JOINTS])
        self.qvel_adr = np.array([m.joint(n).dofadr[0] for n in HOMIE_JOINTS])
        self.act_idx = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in HOMIE_JOINTS])
        if (self.act_idx < 0).any():
            raise ValueError("MJCF actuators must be named after the HOMIE joints")
        free = m.joint("floating_base_joint")
        self.root_qpos, self.root_qvel = int(free.qposadr[0]), int(free.dofadr[0])
        self.pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.cart_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CART_BODY) if with_cart else -1

        c = self.cfg
        self.num_lower = c.env.num_actions                       # 12
        self.num_dof = len(HOMIE_JOINTS)                         # 27
        self.p_gains = np.zeros(self.num_dof)
        self.d_gains = np.zeros(self.num_dof)
        for i, name in enumerate(HOMIE_JOINTS):
            key = next(k for k in c.control.stiffness if k in name)
            self.p_gains[i], self.d_gains[i] = c.control.stiffness[key], c.control.damping[key]
        self.default_dof_pos = np.array([c.init_state.default_joint_angles[n] for n in HOMIE_JOINTS])
        self.torque_limits = np.array([m.jnt_actfrcrange[m.joint(n).id][1] for n in HOMIE_JOINTS])
        # Action limits come from HOMIE's URDF, as in Isaac Lab (g1.xml widens waist_yaw to +-2.618 rad).
        import xml.etree.ElementTree as ET
        from .paths import G1_URDF
        urdf_limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                       for j in ET.parse(G1_URDF).getroot().findall("joint") if j.get("type") != "fixed"}
        limits = np.array([urdf_limits[n] for n in HOMIE_JOINTS])
        self.action_scale = c.control.action_scale
        self.action_min = (limits[:, 0] - self.default_dof_pos) / self.action_scale
        self.action_max = (limits[:, 1] - self.default_dof_pos) / self.action_scale
        self.preset_dof_pos = np.array([self.pose["robot"]["joint_pos"][n] for n in HOMIE_JOINTS])
        offset = self.pose["robot"].get("upper_hold_offset") or {}
        self.upper_hold_offset = np.array([float(offset.get(n, 0.)) for n in HOMIE_JOINTS[self.num_lower:]])
        hold = (self.preset_dof_pos[self.num_lower:] + self.upper_hold_offset - self.default_dof_pos[self.num_lower:]) / self.action_scale
        self.upper_hold_actions = np.clip(hold, self.action_min[self.num_lower:], self.action_max[self.num_lower:])
        self.upper_delta = np.zeros(self.num_dof - self.num_lower)

        scales = c.normalization.obs_scales
        self.obs_scales = dict(ang_vel=scales.ang_vel, dof_pos=scales.dof_pos, dof_vel=scales.dof_vel, lin_vel=scales.lin_vel)
        self.commands_scale = np.array([scales.lin_vel, scales.lin_vel, scales.ang_vel])
        self.height_target = c.rewards.base_height_target
        self.clip_obs, self.clip_actions = c.normalization.clip_observations, c.normalization.clip_actions
        self.one_step_obs = c.env.num_one_step_observations                 # 76
        self.history_len = c.env.num_actor_history                          # 6
        self.commands = np.zeros(5)
        self.commands[4] = self.height_target
        self.actions = np.zeros(self.num_lower)
        self.obs_hist = np.zeros((self.history_len, self.one_step_obs), dtype=np.float32)
        self.joint_pos_target = self.preset_dof_pos.copy()
        self.eq_ids = [i for i in range(m.neq)]
        self.preset = preset_qpos(m, self.pose, self.derived)

    # ------------------------------------------------------------------ state helpers
    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr]

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr]

    @property
    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[self.root_qpos + 3:self.root_qpos + 7]

    @property
    def base_ang_vel(self) -> np.ndarray:
        """Angular velocity in the pelvis frame (the IMU site is aligned with the pelvis)."""
        return self.data.qvel[self.root_qvel + 3:self.root_qvel + 6]

    @property
    def base_lin_vel(self) -> np.ndarray:
        return quat_rotate_inverse_wxyz(self.base_quat_wxyz, self.data.qvel[self.root_qvel:self.root_qvel + 3])

    @property
    def root_height(self) -> float:
        return float(self.data.qpos[self.root_qpos + 2])

    @property
    def projected_gravity(self) -> np.ndarray:
        return quat_rotate_inverse_wxyz(self.base_quat_wxyz, np.array([0., 0., -1.]))

    def cart_state(self):
        if self.cart_id < 0:
            return None
        pos = self.data.xpos[self.cart_id].copy()
        quat = self.data.xquat[self.cart_id].copy()               # wxyz
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.cart_id, vel, 0)
        return pos, quat, vel[3:6].copy(), vel[:3].copy()          # lin (world), ang (world)

    def _weld_rows(self) -> dict[str, list[float]]:
        out = {}
        for i in range(self.data.nefc):
            if self.data.efc_type[i] != mujoco.mjtConstraint.mjCNSTR_EQUALITY:
                continue
            eq = int(self.data.efc_id[i])
            name = self.model.equality(eq).name
            out.setdefault(name, []).append(float(self.data.efc_force[i]))
        return out

    def grasp_forces(self) -> dict[str, float]:
        """Magnitude of the weld constraint force per hand [N] (from the constraint solver)."""
        return {k.replace("grasp_", ""): float(np.linalg.norm(v[:3])) for k, v in self._weld_rows().items()}

    def grasp_force_vectors(self) -> dict[str, np.ndarray]:
        """Force each hand applies to the cart, world frame [N] (same convention as Isaac's ``grasp_wrench``).

        MuJoCo's weld residual is body1 (wrist) minus body2 (cart) in the global frame, so the translational
        ``efc_force`` rows act on the wrist; the cart receives the opposite. Verified on a 42 kg cart: while it
        accelerates forward the returned x components are positive (both hands push).
        """
        return {k.replace("grasp_", ""): -np.asarray(v[:3], dtype=float) for k, v in self._weld_rows().items()}

    def grasp_cart_wrenches(self) -> dict[str, np.ndarray]:
        """Per-hand [Fx,Fy,Fz,Mx,My,Mz] on the cart, world axes, about its assembly COM.

        Weld rotational efc_force rows are not Cartesian torques. Map each weld through J.T
        to the cart free-joint generalized force, then solve the Cartesian Jacobian transpose.
        This includes anchor lever arms, quaternion residual scaling and weld torques correctly.
        Works with both dense and sparse constraint Jacobians (mj_mulJacTVec).
        https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-muljactvec
        """
        if self.cart_id < 0:
            return {}
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, d.subtree_com[self.cart_id], self.cart_id)
        adr = int(m.joint("cart_free").dofadr[0])
        mapping = np.vstack((jacp[:, adr:adr + 6], jacr[:, adr:adr + 6])).T
        out = {}
        for side in ("left", "right"):
            eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{side}")
            if eq < 0:
                continue
            rows = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq)
            selected = np.where(rows, d.efc_force, 0.0)
            generalized = np.zeros(m.nv)
            mujoco.mj_mulJacTVec(m, d, generalized, selected)
            out[side] = np.linalg.solve(mapping, generalized[adr:adr + 6])
        return out

    # ------------------------------------------------------------------ control
    def set_command(self, vx: float, vy: float, wz: float):
        self.commands[:3] = (vx, vy, wz)
        self.commands[3] = 0.
        self.commands[4] = self.height_target

    def reset(self) -> np.ndarray:
        self.data.qpos[:] = self.preset
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.actions[:] = 0.
        self.obs_hist[:] = 0.
        self.joint_pos_target = self.preset_dof_pos.copy()
        return self.compute_observations()

    def compute_observations(self) -> np.ndarray:
        current = np.concatenate([
            self.commands[:3] * self.commands_scale,
            self.commands[4:5],
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions[:self.num_lower],
        ]).astype(np.float32)
        self.obs_hist = np.roll(self.obs_hist, -1, axis=0)
        self.obs_hist[-1] = current
        return np.clip(self.obs_hist.reshape(-1), -self.clip_obs, self.clip_obs)

    def step(self, leg_actions: np.ndarray):
        upper = np.clip(self.upper_hold_actions + self.upper_delta,
                        self.action_min[self.num_lower:], self.action_max[self.num_lower:])
        self.actions = np.clip(np.asarray(leg_actions, dtype=np.float64), -self.clip_actions, self.clip_actions)
        full = np.concatenate([self.actions, upper])
        self.joint_pos_target = self.default_dof_pos + self.action_scale * full
        for _ in range(self.decimation):
            torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
            self.data.ctrl[self.act_idx] = np.clip(torques, -self.torque_limits, self.torque_limits)
            mujoco.mj_step(self.model, self.data)
        fallen = bool(np.linalg.norm(self.projected_gravity[:2]) > 0.8)
        obs = self.compute_observations()
        return obs, fallen

    def info(self) -> dict:
        return {"xml": str(self.xml_path), "dt": self.dt, "timestep": float(self.model.opt.timestep),
                "decimation": self.decimation, "nq": int(self.model.nq), "with_cart": self.with_cart,
                "cart_mass_kg": float(self.model.body(CART_BODY).mass[0]) if self.with_cart else None}
