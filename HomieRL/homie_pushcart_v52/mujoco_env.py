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
        self._robot_bodies = self._subtree_ids(self.pelvis_id)
        self._cart_bodies = self._subtree_ids(self.cart_id) if with_cart else np.array([], dtype=int)
        self._feet = [m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self._wheels = [m.body(f"wheel_{i}").id for i in range(4)] if with_cart else []
        self._ground_geoms = {i for i in range(m.ngeom) if int(m.geom_bodyid[i]) == 0
                              and m.geom(i).name == 'floor'}
        if not self._ground_geoms:
            raise ValueError('V5 ground wrench reporting needs the explicit world-attached floor geom')
        self.cart_pulse_impulse = np.zeros(6)
        self.cart_applied_wrench = np.zeros(6)
        self._pulse_wrench = np.zeros(6)
        self._pulse_steps = 0
        self.physics_samples = []

    def _subtree_ids(self, root):
        selected = []
        for body in range(1, self.model.nbody):
            parent = body
            while parent > 0 and parent != root:
                parent = int(self.model.body_parentid[parent])
            if parent == root:
                selected.append(body)
        return np.array(selected, dtype=int)

    def centroidal_state(self, body_id):
        """Full multibody momentum using physical COMs, including wheel spin."""
        m, d = self.model, self.data
        ids = self._robot_bodies if body_id == self.pelvis_id else self._cart_bodies if body_id == self.cart_id else self._subtree_ids(body_id)
        masses, positions = m.body_mass[ids], d.xipos[ids]
        velocities = np.zeros((len(ids), 6))
        for j, body in enumerate(ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(body), velocities[j], 0)
        com = np.sum(masses[:, None] * positions, axis=0) / masses.sum()
        linear = (masses[:, None] * velocities[:, 3:]).sum(0)
        # ximat maps principal inertial axes to world; body_inertia is diagonal there.
        rotations = d.ximat[ids].reshape(-1, 3, 3)
        inertia = np.einsum('bij,bj,bkj->bik', rotations, m.body_inertia[ids], rotations)
        offset = positions - com
        angular = (np.einsum('bij,bj->bi', inertia, velocities[:, :3]) +
                   np.cross(offset, masses[:, None] * velocities[:, 3:])).sum(0)
        assembly = (inertia + masses[:, None, None] * (
            (offset**2).sum(-1)[:, None, None] * np.eye(3) - offset[:, :, None] * offset[:, None, :])).sum(0)
        return dict(mass=float(masses.sum()), com=com, velocity=linear / masses.sum(),
                    linear_momentum=linear, angular_momentum=angular, inertia_world=assembly)

    def ground_contact_wrenches(self):
        """Six-axis contact loads from every ground contact point (world COM references)."""
        d, m = self.data, self.model
        feet, wheels = np.zeros((2, 6)), np.zeros((len(self._wheels), 6))
        robot_com = d.subtree_com[self.pelvis_id]
        cart_com = d.subtree_com[self.cart_id] if self.with_cart else np.zeros(3)
        for index in range(d.ncon):
            contact = d.contact[index]
            bodies = [int(m.geom_bodyid[g]) for g in contact.geom]
            # A static wall/obstacle also belongs to body 0; only configured
            # floor geoms are ground partners, not every world-attached geom.
            ground_side = [int(g) in self._ground_geoms for g in contact.geom]
            if sum(ground_side) != 1:
                continue
            body = bodies[1] if ground_side[0] else bodies[0]
            if body in self._feet:
                target, row, reference = feet, self._feet.index(body), robot_com
            elif body in self._wheels:
                target, row, reference = wheels, self._wheels.index(body), cart_com
            else:
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, local)
            sign = 1. if ground_side[0] else -1.
            rotation = contact.frame.reshape(3, 3).T
            force = sign * (rotation @ local[:3])
            moment = sign * (rotation @ local[3:]) + np.cross(contact.pos - reference, force)
            target[row] += np.r_[force, moment]
        return dict(feet=feet, wheels=wheels, valid=bool(np.isfinite(feet).all() and np.isfinite(wheels).all()))

    def inject_cart_wrench(self, force, torque, duration, offset=None):
        if not self.with_cart or not np.isfinite(duration) or duration <= 0:
            raise ValueError("physical pulse requires a cart and finite positive duration")
        force, torque = np.asarray(force, dtype=float), np.asarray(torque, dtype=float)
        if force.shape != (3,) or torque.shape != (3,) or not np.isfinite(np.r_[force, torque]).all():
            raise ValueError("pulse force and torque require three finite components")
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("pulse offset requires three finite components")
            torque = torque + np.cross(offset, force)
        if not np.isfinite(torque).all():
            raise ValueError("pulse moment including offset must be finite")
        self._pulse_wrench[:] = np.r_[force, torque]
        self._pulse_steps = max(1, round(float(duration) / self.model.opt.timestep))

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
        return pos, quat, self.centroidal_state(self.cart_id)["velocity"], vel[:3].copy()

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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.preset
        self.data.qvel[:] = 0.
        self.data.ctrl[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.actions[:] = 0.
        self.obs_hist[:] = 0.
        self.joint_pos_target = self.preset_dof_pos.copy()
        self._pulse_steps = 0
        self._pulse_wrench[:] = 0.
        self.upper_delta[:] = 0.
        self.commands[:4] = 0.
        self.commands[4] = self.height_target
        self.cart_applied_wrench[:] = 0.
        self.cart_pulse_impulse[:] = 0.
        self.physics_samples = []
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
        self.physics_samples = []
        for _ in range(self.decimation):
            torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
            self.data.ctrl[self.act_idx] = np.clip(torques, -self.torque_limits, self.torque_limits)
            self.cart_applied_wrench[:] = self._pulse_wrench if self._pulse_steps > 0 else 0.
            applied_reference = self.data.subtree_com[self.cart_id].copy() if self.with_cart else np.zeros(3)
            if self.with_cart:
                force = self.cart_applied_wrench[:3]
                torque = self.cart_applied_wrench[3:] + np.cross(
                    self.data.subtree_com[self.cart_id] - self.data.xipos[self.cart_id], force)
                self.data.xfrc_applied[self.cart_id] = np.r_[force, torque]
            mujoco.mj_step(self.model, self.data)
            self._pulse_steps = max(0, self._pulse_steps - 1)
            self.cart_pulse_impulse += self.cart_applied_wrench * self.model.opt.timestep
            # Save the actual interval solve before mj_forward replaces it at the new state.
            interval_hand = sum(self.grasp_cart_wrenches().values(), np.zeros(6)) if self.with_cart else np.zeros(6)
            interval_ground = self.ground_contact_wrenches()
            force_cart_com = self.data.subtree_com[self.cart_id].copy() if self.with_cart else np.zeros(3)
            force_robot_com = self.data.subtree_com[self.pelvis_id].copy()
            mujoco.mj_forward(self.model, self.data)
            robot = self.centroidal_state(self.pelvis_id)
            cart = self.centroidal_state(self.cart_id) if self.with_cart else None
            self.physics_samples.append(dict(time_s=float(self.data.time),
                force_interval_start_s=float(self.data.time-self.model.opt.timestep),
                force_cart_com=force_cart_com, force_robot_com=force_robot_com,
                robot_momentum=np.r_[robot['linear_momentum'], robot['angular_momentum']],
                robot_com=robot['com'], cart_momentum=np.r_[cart['linear_momentum'], cart['angular_momentum']] if cart else np.zeros(6),
                cart_com=cart['com'] if cart else np.zeros(3),
                hand_wrench=interval_hand,
                feet_wrench=interval_ground['feet'], wheel_wrench=interval_ground['wheels'],
                cart_applied_reference=applied_reference,
                cart_applied_wrench=self.cart_applied_wrench.copy(),
                cart_applied_wrench_com=np.r_[self.cart_applied_wrench[:3], self.cart_applied_wrench[3:] +
                    np.cross(applied_reference-(cart['com'] if cart else applied_reference), self.cart_applied_wrench[:3])],
                ground_valid=interval_ground['valid']))
        fallen = bool(np.linalg.norm(self.projected_gravity[:2]) > 0.8)
        obs = self.compute_observations()
        return obs, fallen

    def info(self) -> dict:
        return {"xml": str(self.xml_path), "dt": self.dt, "timestep": float(self.model.opt.timestep),
                "decimation": self.decimation, "nq": int(self.model.nq), "with_cart": self.with_cart,
                "cart_mass_kg": float(self.model.body_mass[self._cart_bodies].sum()) if self.with_cart else None,
                "cart_mass_definition": "complete cart assembly including wheels and payload"}
