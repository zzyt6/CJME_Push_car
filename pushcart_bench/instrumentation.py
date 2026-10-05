"""200 Hz measurements on the unchanged legacy four-sphere MuJoCo backend.

The step loop matches ``MujocoHomieEnv``; weld forces, palm positions and
mechanical power are read after/before each existing physics step. Measurements
never feed back into the simulation. An explicit ``inject_cart_wrench`` call is
the only optional intervention, used by the retained disturbance benchmarks.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from homie_pushcart.assets.mjcf_scene import MJCF_ROBOT, MJCF_SCENE
from homie_pushcart.mujoco_env import MujocoHomieEnv


# Palm-link origins relative to the welded wrist_yaw_link in the original URDF.
PALM_OFFSET = {"left": np.array([.0415, .003, 0.]), "right": np.array([.0415, -.003, 0.])}
SIDES = ("left", "right")


class InstrumentedEnv(MujocoHomieEnv):
    """Legacy-only environment with substep measurements and deliberate pulses."""

    def __init__(self, with_cart: bool, xml_path: Path | None = None, decimation: int = 4, *, cfg=None):
        expected = MJCF_SCENE if with_cart else MJCF_ROBOT
        if xml_path is not None and Path(xml_path).resolve() != expected.resolve():
            raise ValueError("Instrumentation uses the fixed legacy four-sphere backend")
        super().__init__(with_cart, xml_path, decimation, cfg=cfg)
        self.robot_mass = float(self.model.body_subtreemass[self.pelvis_id])
        if with_cart:
            yaw0 = float(self.derived["base_yaw"])
            delta = np.array(self.derived["bar_center_world"]) - np.array(self.derived["base_pos_world"])
            c, s = math.cos(-yaw0), math.sin(-yaw0)
            self.bar_local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2]])
            self.wrist_ids = {side: self.model.body(f"{side}_wrist_yaw_link").id for side in SIDES}
        self._pulse_wrench, self._pulse_steps = np.zeros(6), 0
        self._pulse_active = False
        self._clear()

    def _clear(self):
        self.sub_hands, self.sub_palm, self.sub_handle, self.sub_power = [], [], [], []

    def reset(self) -> np.ndarray:
        self._clear()
        if self.with_cart and self._pulse_active:
            self.data.xfrc_applied[self.cart_id] = 0.
        self._pulse_wrench, self._pulse_steps = np.zeros(6), 0
        self._pulse_active = False
        return super().reset()

    def inject_cart_wrench(self, force_w, torque_w, duration_s):
        """Apply a known world-axis wrench at cart COM for the requested duration."""
        if not self.with_cart:
            raise ValueError("A cart is required for cart-side disturbances")
        wrench = np.r_[np.asarray(force_w, float), np.asarray(torque_w, float)]
        if wrench.shape != (6,) or not np.isfinite(wrench).all() or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("A finite 6-D wrench and positive duration are required")
        self._pulse_wrench = wrench
        self._pulse_steps = max(1, round(float(duration_s) / self.model.opt.timestep))
        self._pulse_active = True

    def _sample(self):
        """Read the last weld solve, without rerunning forward dynamics."""
        d = self.data
        hands = self.grasp_cart_wrenches()
        com = d.subtree_com[self.cart_id]
        handle = d.xpos[self.cart_id] + d.xmat[self.cart_id].reshape(3, 3) @ self.bar_local
        wrench, palms, total = np.zeros((2, 6)), np.zeros((2, 3)), np.zeros(6)
        for i, side in enumerate(SIDES):
            if side not in hands:
                continue
            body = self.wrist_ids[side]
            palm = d.xpos[body] + d.xmat[body].reshape(3, 3) @ PALM_OFFSET[side]
            force, moment_com = hands[side][:3], hands[side][3:]
            moment_palm = moment_com - np.cross(palm - com, force)
            wrench[i], palms[i] = np.r_[force, moment_palm], palm
            total += np.r_[force, moment_palm + np.cross(palm - handle, force)]
        self.sub_hands.append(wrench)
        self.sub_palm.append(palms)
        self.sub_handle.append(total)

    def step(self, leg_actions: np.ndarray):
        # Preserve the exact legacy action/PD/step/observation arithmetic.
        upper = np.clip(self.upper_hold_actions + self.upper_delta,
                        self.action_min[self.num_lower:], self.action_max[self.num_lower:])
        self.actions = np.clip(np.asarray(leg_actions, dtype=np.float64), -self.clip_actions, self.clip_actions)
        full = np.concatenate([self.actions, upper])
        self.joint_pos_target = self.default_dof_pos + self.action_scale * full
        self._clear()
        for _ in range(self.decimation):
            torques = self.p_gains * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
            applied = np.clip(torques, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.act_idx] = applied
            self.sub_power.append(float(np.abs(applied * self.dof_vel).sum()))
            if self._pulse_active:
                if self._pulse_steps > 0:
                    self.data.xfrc_applied[self.cart_id] = self._pulse_wrench
                    self._pulse_steps -= 1
                else:
                    self.data.xfrc_applied[self.cart_id] = 0.
                    self._pulse_active = False
            mujoco.mj_step(self.model, self.data)
            if self.with_cart:
                self._sample()
        fallen = bool(np.linalg.norm(self.projected_gravity[:2]) > .8)
        return self.compute_observations(), fallen


def install_instrumented_env():
    """Enable substep recording for V3 and passthrough benchmark runners."""
    import homie_pushcart.mujoco_hl as hl
    hl.MujocoHomieEnv = InstrumentedEnv
