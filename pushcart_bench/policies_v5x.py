"""V5.1 / V5.2 high-level policies in the shared MuJoCo bench, plus read-only 200 Hz hand-load instrumentation.

The Isaac-side V5.x actor observation (fixed-grasp FK, HIM velocity paired with the post-step sample,
observable recovery authority, and for V5.2 the (J^T)^+ tau prior) is rebuilt here from MuJoCo measurements,
mirroring ``PushCartEnv.step`` / ``_update_hl_observations`` with all observation noise and delays off.
The default ``isaac`` scene uses the training URDF inertias, joint limits,
mixed actuator interface and compact payload model, restoring the saved HOMIE
settings and checkpoint. ``upper_drive='explicit_pd'`` selects the separate
all-effort experiment. ``scene_profile='legacy'`` reproduces the original
cross-version benchmark; never pool the two profiles.

``InstrumentedEnv`` preserves the legacy stepping path and implements the aligned scene's selected upper
drive and body damping. Its read-only instrumentation records both weld wrenches after every physics
step: hand-on-cart force and moment at each palm origin (world axes), the resultant about the handle
centre, and the PD-proxy joint mechanical power. These records do not feed back into the simulation.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from homie_pushcart.assets.mjcf_scene import CART_BODY
from homie_pushcart.mujoco_env import MujocoHomieEnv, quat_rotate_inverse_wxyz
from homie_pushcart_v52.control_contract import pd_effort, validate_upper_drive

from .policies import CartPolicyBase

# g1_grasp.urdf ``*_hand_palm_joint``: palm-link origin on the welded wrist_yaw_link.
PALM_OFFSET = {"left": np.array([0.0415, 0.003, 0.]), "right": np.array([0.0415, -0.003, 0.])}
SIDES = ("left", "right")


def _yaw(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_to_mat(quat_wxyz) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, dtype=float))
    return m.reshape(3, 3)


class InstrumentedEnv(MujocoHomieEnv):
    """Selectable scene/actuation profile with read-only grasp-wrench instrumentation."""

    def __init__(self, with_cart: bool, xml_path: Path | None = None, decimation: int = 4,
                 scene_profile: str = "legacy", *, cfg=None, upper_drive: str = "training"):
        if scene_profile not in ("legacy", "isaac"):
            raise ValueError(f"Unknown sim2sim scene profile: {scene_profile}")
        self.scene_profile = scene_profile
        self.upper_drive = validate_upper_drive(upper_drive)
        if scene_profile == "isaac":
            if not with_cart or xml_path is not None:
                raise ValueError("The Isaac parameter profile owns its robot+cart scene")
            from .sim2sim_scene import build_isaac_scene
            xml_path = build_isaac_scene(cfg, self.upper_drive)
        super().__init__(with_cart, xml_path, decimation, cfg=cfg)
        m = self.model
        self.robot_mass = float(m.body_subtreemass[self.pelvis_id])
        if with_cart:
            derived = self.derived
            yaw0 = float(derived["base_yaw"])
            d = np.array(derived["bar_center_world"]) - np.array(derived["base_pos_world"])
            c, s = math.cos(-yaw0), math.sin(-yaw0)
            self.bar_local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
            self.wrist_ids = {side: m.body(f"{side}_wrist_yaw_link").id for side in SIDES}
        self._pulse_wrench, self._pulse_steps = np.zeros(6), 0
        self.upper_torque_proxy = np.zeros(15)
        self.audit_newton_euler = False
        self._clear()

    def _clear(self):
        self.sub_hands, self.sub_palm, self.sub_handle, self.sub_power = [], [], [], []
        self.sub_newton_euler = []

    def reset(self) -> np.ndarray:
        self._clear()
        self._pulse_wrench, self._pulse_steps = np.zeros(6), 0
        self.upper_torque_proxy[:] = 0.
        if self.with_cart:
            self.data.xfrc_applied[self.cart_id] = 0.
        return super().reset()

    def inject_cart_wrench(self, force_w, torque_w, duration_s):
        """Known world-axis force/couple on the cart base body COM for ``duration_s`` (5 ms steps)."""
        self._pulse_wrench = np.r_[np.asarray(force_w, float), np.asarray(torque_w, float)]
        self._pulse_steps = max(1, round(float(duration_s) / self.model.opt.timestep))

    def _sample(self):
        """Weld solve of the physics step just taken; xpos/subtree_com belong to the same forward pass."""
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

    def _body_damping(self):
        """PhysX's configured linear/angular velocity damping, as body COM loads."""
        m, d = self.model, self.data
        d.xfrc_applied[:] = 0.
        linear, angular = self.cfg.asset.linear_damping, self.cfg.asset.angular_damping
        for b in range(1, m.nbody):
            if m.body_mass[b] == 0:
                continue
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, velocity, 0)
            rot = d.ximat[b].reshape(3, 3)
            d.xfrc_applied[b, :3] = -linear * m.body_mass[b] * velocity[3:]
            d.xfrc_applied[b, 3:] = -angular * (rot @ (m.body_inertia[b] * (rot.T @ velocity[:3])))

    def step(self, leg_actions: np.ndarray):
        # Keep the shared action pipeline; dispatch actuation by scene profile.
        upper = np.clip(self.upper_hold_actions + self.upper_delta,
                        self.action_min[self.num_lower:], self.action_max[self.num_lower:])
        self.actions = np.clip(np.asarray(leg_actions, dtype=np.float64), -self.clip_actions, self.clip_actions)
        full = np.concatenate([self.actions, upper])
        self.joint_pos_target = self.default_dof_pos + self.action_scale * full
        self._clear()
        for _ in range(self.decimation):
            applied = pd_effort(self.dof_pos, self.dof_vel, self.joint_pos_target,
                                self.p_gains, self.d_gains, self.torque_limits)
            # Isaac's implicit upper actuator exposes the clipped PRE-step PD
            # estimate, not its solver impulse. Keep that observation separate
            # from ctrl, which holds position targets with native servos.
            self.upper_torque_proxy[:] = applied[self.num_lower:]
            if self.scene_profile == "isaac":
                if self.upper_drive == "training":
                    self.data.ctrl[self.act_idx[:self.num_lower]] = applied[:self.num_lower]
                    self.data.ctrl[self.act_idx[self.num_lower:]] = self.joint_pos_target[self.num_lower:]
                else:
                    self.data.ctrl[self.act_idx] = applied
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_comPos(self.model, self.data)
                mujoco.mj_comVel(self.model, self.data)
                self._body_damping()
            else:
                self.data.ctrl[self.act_idx] = applied
            self.sub_power.append(float(np.abs(applied * self.dof_vel).sum()))
            if self.with_cart:
                if self.scene_profile == "isaac":
                    self.data.xfrc_applied[self.cart_id] += self._pulse_wrench if self._pulse_steps > 0 else 0.
                else:
                    self.data.xfrc_applied[self.cart_id] = self._pulse_wrench if self._pulse_steps > 0 else 0.
                self._pulse_steps = max(0, self._pulse_steps - 1)
            pre_step_velocity = self.data.qvel.copy() if self.audit_newton_euler else None
            mujoco.mj_step(self.model, self.data)
            if self.with_cart:
                self._sample()
                if self.audit_newton_euler:
                    from .sim2sim_scene import newton_euler_hand_wrenches
                    self.sub_newton_euler.append(newton_euler_hand_wrenches(self, pre_step_velocity))
        if self.scene_profile == "isaac":
            # mj_step leaves derived transforms at the last solver sample, one
            # substep behind integrated qpos. Refresh only kinematics: do not
            # run mj_forward and replace the recorded interval's constraint loads.
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_comVel(self.model, self.data)
        fallen = bool(np.linalg.norm(self.projected_gravity[:2]) > 0.8)
        obs = self.compute_observations()
        return obs, fallen


def install_instrumented_env():
    """Make every ``homie_pushcart.mujoco_hl.make_runner`` (V3 ... V4, passthrough) use ``InstrumentedEnv``."""
    import homie_pushcart.mujoco_hl as hl
    hl.MujocoHomieEnv = InstrumentedEnv


class V5xRunner:
    """MuJoCo counterpart of the V5.1 / V5.2 ``PushCartEnv`` action and observation pipeline (single env)."""

    def __init__(self, package: str, checkpoint: Path, level: int = 17, homie_checkpoint: Path | None = None,
                 scene_profile: str = "isaac", upper_drive: str | None = None):
        mod = lambda name: importlib.import_module(f"{package}.{name}")  # noqa: E731
        tcm, obm, fgm, cqm, cum, low = (mod(n) for n in ("task_config", "observation", "fixed_grasp",
                                                           "cart_quality", "curriculum", "lowlevel"))
        self.package, self.obm = package, obm
        cfg, tc = mod("env_smoke").make_configs(1, False, level, checkpoint=checkpoint)
        self.task_cfg = tc
        mode = upper_drive if upper_drive is not None else getattr(tc.actions, "upper_drive", "training")
        self.env = env = InstrumentedEnv(with_cart=True, scene_profile=scene_profile,
                                         cfg=cfg if scene_profile == "isaac" else None,
                                         decimation=cfg.control.decimation if scene_profile == "isaac" else 4,
                                         upper_drive=mode)
        saved_lowlevel = getattr(cfg, "pushcart_saved_checkpoint", None)
        self.homie_checkpoint = Path(homie_checkpoint or saved_lowlevel or low.frozen_checkpoint())
        if not self.homie_checkpoint.is_absolute():
            self.homie_checkpoint = mod("paths").ROOT / self.homie_checkpoint
        # Low level first: the high-level loader removes HOMIE's rsl_rl fork from sys.path.
        self.lowlevel = low.HimLowLevelPolicy(self.homie_checkpoint, device="cpu")
        if env.one_step_obs * env.history_len != self.lowlevel.num_obs:
            raise ValueError("Saved low-level history layout does not match the frozen HOMIE network")
        self.hl = mod("hl_policy").HighLevelPolicy(Path(checkpoint), tcm.num_actor_obs(tc), tcm.num_critic_obs(tc),
                                                   tc.env.num_hl_actions, "cpu")
        self.actor = self.hl.actor
        from tensordict import TensorDict
        self._TensorDict = TensorDict
        self.frame_dim = obm.FRAME_DIM
        self.has_prior = any(name == "static_hand_wrench" for name, _ in obm.FRAME_FIELDS)
        self.wrench_scale = np.array(obm.WRENCH_SCALE)
        self.hand_scale = np.tile(np.array(obm.HAND_WRENCH_SCALE), 2) if self.has_prior else None
        self.dt = env.dt
        self.num_actions = tc.env.num_hl_actions
        self.history = tc.env.actor_history
        self.residual_scale = np.array(tc.actions.cmd_residual_scale)
        lim = np.array(tc.actions.homie_cmd_limits)
        self.cmd_lo, self.cmd_hi = lim[:, 0], lim[:, 1]
        self.cmd_alpha = min(1.0, self.dt / tc.actions.cmd_filter_tau_s) if tc.actions.cmd_filter_tau_s > 0 else 1.0
        self.ref_alpha = min(1.0, self.dt / tc.commands.ref_tau_s) if tc.commands.ref_tau_s > 0 else 1.0
        self.gate_speed = float(tc.actions.idle_gate_speed)
        self.level = int(level)
        self.lp = cum.level_params(self.level, tc)
        self.upper_scale = np.array([self.lp.waist_scale] + [self.lp.arm_scale] * 14)
        self.step_lim = tc.actions.upper_rate_limit_rad_s * self.dt
        n_low = env.num_lower
        self.hold_target_q = env.preset_dof_pos[n_low:] + env.upper_hold_offset
        self.upper_lo = env.default_dof_pos[n_low:] + env.action_min[n_low:] * env.action_scale
        self.upper_hi = env.default_dof_pos[n_low:] + env.action_max[n_low:] * env.action_scale
        self.commands_scale = env.commands_scale
        m = env.model
        self.cart_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CART_BODY)
        self.pelvis = env.pelvis_id
        self.bar_local = env.bar_local
        self.swivel_qpos = [m.joint(f"caster_{i}_swivel_joint").qposadr[0] for i in range(4)]
        self.upper_act = env.act_idx[n_low:]
        self.quality = cqm.CartQualityTracker(tc.quality, self.dt, 1, dtype=torch.float64)
        self.idle_correction, self.idle_gate_applied = None, 0.0
        self.wheel_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{i}") for i in range(4)]
        self.observer = fgm.FixedGraspObserver(
            1, "cpu", pose=env.pose, cart_params=env.cart_params,
            position_tolerance_m=tc.interaction.fk_position_tolerance_m,
            orientation_tolerance_rad=tc.interaction.fk_orientation_tolerance_rad)
        self.recovery = obm.RecoveryAuthority(tc.recovery, self.dt, 1, "cpu")
        self._default_inertia = m.body_inertia[self.cart_body].copy()
        self.reset_buffers()

    # ------------------------------------------------------------------ state
    @property
    def cart_mass_total(self) -> float:
        if self.env.scene_profile == "isaac":
            return float(self.env.model.body_subtreemass[self.cart_body])
        return float(self.env.model.body_mass[self.cart_body])

    def set_payload(self, payload_kg: float):
        if self.env.scene_profile == "isaac":
            from .sim2sim_scene import set_payload
            set_payload(self.env, payload_kg, getattr(self, "payload_offset", (0., 0.)))
            return
        # Same payload model as homie_pushcart.mujoco_hl, so every bench policy sees identical cart physics.
        m = self.env.model
        base = float(self.env.cart_params["mass"]["base_kg"])
        m.body_mass[self.cart_body] = base + payload_kg
        m.body_inertia[self.cart_body] = self._default_inertia * (base + payload_kg) / base
        mujoco.mj_setConst(m, self.env.data)

    def reset_buffers(self):
        self.quality.reset()
        self.cmd_task, self.cmd_ref, self.cmd_homie = np.zeros(3), np.zeros(3), np.zeros(3)
        self.hl_actions = np.zeros(self.num_actions)
        self.upper_delta_q = np.zeros(15)
        self.vel_est = np.zeros(3)
        self.actor_hist = np.zeros((self.history, self.frame_dim), dtype=np.float32)
        self.env.upper_delta[:] = 0.
        self.observer.reset()
        self.recovery.reset(slice(None))
        self.kinematic_valid = False
        self.episode_length = 0
        self.pred_hands = np.full(12, np.nan)
        self.pred_resultant = np.full(3, np.nan)
        self.prior = np.full(12, np.nan)

    def reset(self, caster_angle: float | None = None) -> np.ndarray:
        obs = self.env.reset()
        if caster_angle is not None:
            for adr in self.swivel_qpos:
                self.env.data.qpos[adr] = caster_angle
            mujoco.mj_forward(self.env.model, self.env.data)
        self.reset_buffers()
        self._obs = obs
        self.vel_est = self._estimate_velocity(obs)
        self.task_state()
        self._update_hl_obs(fresh=True)
        return self.hl_obs

    def _estimate_velocity(self, obs) -> np.ndarray:
        return self.lowlevel.estimate_velocity(torch.from_numpy(obs).unsqueeze(0))[0].cpu().numpy() / self.env.obs_scales["lin_vel"]

    def task_state(self) -> dict:
        """Same task signals as homie_pushcart.mujoco_hl (cart base-link origin velocity)."""
        env, d = self.env, self.env.data
        cart_pos, cart_quat = d.xpos[self.cart_body].copy(), d.xquat[self.cart_body].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_BODY, self.cart_body, vel, 0)
        lin, ang = vel[3:6], vel[:3]
        if env.scene_profile == "isaac":
            lin = lin + np.cross(ang, d.xpos[self.cart_body] - d.xipos[self.cart_body])
        yaw = _yaw(cart_quat)
        c, s = math.cos(yaw), math.sin(yaw)
        bar_w = cart_pos + _quat_to_mat(cart_quat) @ self.bar_local
        base_quat = env.base_quat_wxyz.copy()
        rel = bar_w - d.xpos[self.pelvis]
        rel_yaw = math.atan2(math.sin(yaw - _yaw(base_quat)), math.cos(yaw - _yaw(base_quat)))
        hands = env.grasp_cart_wrenches()
        total = sum(hands.values(), np.zeros(6))
        self.ts = {
            "cart_v": np.array([c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1]]), "cart_wz": float(ang[2]),
            "cart_speed": float(np.hypot(lin[0], lin[1])), "bar_b": quat_rotate_inverse_wxyz(base_quat, rel),
            "rel_yaw": rel_yaw, "dist": float(np.hypot(rel[0], rel[1])),
            "cart_tilt": float(np.linalg.norm(quat_rotate_inverse_wxyz(cart_quat, np.array([0., 0., -1.]))[:2])),
            "upper_tau": env.upper_torque_proxy.copy(), "grasp": env.grasp_forces(), "cart_yaw": yaw,
            "cart_force_w": total[:3], "cart_moment_w": total[3:], "hand_cart_wrenches": hands,
            "robot_tilt_rad": float(np.arcsin(np.clip(np.linalg.norm(env.projected_gravity[:2]), 0, 1))),
            "wheel_lift_m": float(np.maximum(d.xpos[self.wheel_body_ids, 2] - env.cart_params["wheels"]["radius"], 0).max()),
        }
        return self.ts

    # ------------------------------------------------------------------ stepping (mirrors PushCartEnv.step)
    def step(self, cmd_task) -> tuple[np.ndarray, bool]:
        tc, env = self.task_cfg, self.env
        self.cmd_task = np.asarray(cmd_task, dtype=float)
        with torch.inference_mode():
            extra = {'estimator_obs': self.falcon_obs} if self.has_prior else {}
            raw = self.hl(torch.from_numpy(self.hl_obs).unsqueeze(0), **extra).squeeze(0).cpu().numpy()
        applied = np.clip(raw, -tc.actions.clip, tc.actions.clip)
        self.hl_actions = applied
        self.cmd_ref = self.cmd_ref + self.ref_alpha * (self.cmd_task - self.cmd_ref)
        gate = min(1.0, float(np.linalg.norm(self.cmd_ref)) / self.gate_speed) if self.gate_speed > 0 else 1.0
        gate = max(gate, float(self.recovery.gate[0, 0])) * float(self.kinematic_valid)
        reference = (self.observer.reference_base(torch.as_tensor(self.cmd_ref[None], dtype=torch.float32))[0].numpy()
                     if self.kinematic_valid else np.zeros(3))
        target = np.clip(reference + gate * applied[:3] * self.residual_scale, self.cmd_lo, self.cmd_hi)
        self.cmd_homie = self.cmd_homie + self.cmd_alpha * (target - self.cmd_homie)
        env.set_command(*self.cmd_homie)
        env.obs_hist[-1, :3] = env.commands[:3] * self.commands_scale
        delta_target = gate * applied[3:] * self.upper_scale
        delta = self.upper_delta_q + np.clip(delta_target - self.upper_delta_q, -self.step_lim, self.step_lim)
        q_target = np.clip(self.hold_target_q + delta, self.upper_lo, self.upper_hi)
        self.upper_delta_q = q_target - self.hold_target_q
        env.upper_delta[:] = self.upper_delta_q / env.action_scale
        obs_t = torch.from_numpy(np.clip(env.obs_hist.reshape(-1), -env.clip_obs, env.clip_obs)).unsqueeze(0)
        legs, _ = self.lowlevel.act_and_estimate(obs_t)
        self._obs, fallen = env.step(legs.squeeze(0).cpu().numpy())
        # Pair the HIM estimate with the NEW sample, as the Isaac V5.x env does.
        self.vel_est = self._estimate_velocity(self._obs)
        self.episode_length += 1
        self.task_state()
        self.quality.update(torch.tensor([[*self.ts["cart_v"], self.ts["cart_wz"]]], dtype=torch.float64),
                            torch.from_numpy(self.cmd_ref[None]), torch.tensor([self.ts["cart_yaw"]], dtype=torch.float64),
                            torch.from_numpy(self.ts["cart_force_w"][None]), torch.from_numpy(self.ts["cart_moment_w"][None]))
        self._update_hl_obs(fresh=False)
        return self.hl_obs, fallen

    def _update_hl_obs(self, fresh: bool):
        env, tc, obm = self.env, self.task_cfg, self.obm
        f32 = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)  # noqa: E731
        q, dq, gyro, gravity = env.dof_pos.copy(), env.dof_vel.copy(), env.base_ang_vel.copy(), env.projected_gravity
        sc = env.obs_scales
        q_rel = (q - env.default_dof_pos) * sc["dof_pos"]
        dq_scaled = dq * sc["dof_vel"]
        tau_nm = np.zeros(15) if fresh else self.ts["upper_tau"]
        tau = tau_nm * tc.observations.torque_scale
        fk = self.observer.update(f32(q), f32(dq), f32(self.vel_est), f32(gyro), f32(gravity))
        self.kinematic_valid = bool(fk.valid[0])
        self.recovery.update(fk.cart_twist_heading, f32(self.cmd_ref), f32(gyro), f32(gravity),
                             fk.valid & torch.tensor([not fresh]))
        valid = self.kinematic_valid
        pose = np.r_[fk.handle_pos_b[0].numpy(), fk.rel_yaw[0].item()] if valid else np.zeros(4)
        twist = fk.cart_twist_heading[0].numpy() if valid else np.zeros(3)
        upper_target = (self.hold_target_q + self.upper_delta_q - env.default_dof_pos[env.num_lower:]) * sc["dof_pos"]
        cs = self.commands_scale
        parts = [q_rel, dq_scaled, tau, gyro * sc["ang_vel"], gravity, self.vel_est * sc["lin_vel"], pose,
                 twist * cs, self.cmd_ref * cs, self.cmd_homie * cs, upper_target, self.hl_actions]
        if self.has_prior:
            prior = obm.static_hand_wrench(fk.palm_jacobian_b, f32(tau_nm),
                                           obm.body_to_cart_heading(fk.body_to_heading, fk.rel_yaw))[0].numpy()
            self.prior = prior if valid else np.zeros(12)
            parts.append(self.prior / self.hand_scale)
        frame = np.concatenate(parts).astype(np.float32)
        if frame.shape[0] != self.frame_dim:
            raise RuntimeError(f"{self.package} MuJoCo frame {frame.shape[0]} != {self.frame_dim}")
        if fresh:
            self.actor_hist[:] = 0.
        else:
            self.actor_hist = np.roll(self.actor_hist, -1, axis=0)
        self.actor_hist[-1] = frame
        self.hl_obs = np.clip(self.actor_hist.reshape(-1), -env.clip_obs, env.clip_obs)
        if self.has_prior:
            from homie_pushcart_v52.falcon_history import FalconHistory
            if not hasattr(self, 'falcon_history'):
                self.falcon_history = FalconHistory(1)
            self.falcon_obs = self.falcon_history.push(
                torch.from_numpy(self.hl_obs[-self.frame_dim:]).unsqueeze(0),
                f32(env.actions[:12]), torch.tensor([fresh]))
        self._predict()

    @torch.inference_mode()
    def _predict(self):
        td = self._TensorDict({"policy": torch.from_numpy(self.hl_obs).unsqueeze(0)}, batch_size=[1])
        if self.has_prior:
            td['falcon'] = self.falcon_obs
            hands = self.actor.predict_hand_wrench(td)
            self.pred_hands = hands[0].numpy() * self.hand_scale
            self.pred_resultant = self.actor.compose_resultant(hands)[0].numpy() * self.wrench_scale
        else:
            self.pred_resultant = self.actor.predict_interaction(td)[0].numpy() * self.wrench_scale

    def geometry_failure(self) -> str | None:
        tc, ts = self.task_cfg.termination, self.ts
        if abs(ts["rel_yaw"]) > tc.yaw_diff_rad:
            return "cart_yaw"
        if not (tc.distance_m[0] <= ts["dist"] <= tc.distance_m[1]):
            return "cart_distance"
        if ts["cart_tilt"] > math.sin(tc.cart_tilt_rad):
            return "cart_tilt"
        return None


class V5xPolicy(CartPolicyBase):
    """Bench policy for a V5.1 / V5.2 checkpoint (``package`` = homie_pushcart_v51 | homie_pushcart_v52)."""

    def __init__(self, package: str, checkpoint: str | Path, level: int = 17, homie_checkpoint: Path | None = None,
                 name: str | None = None, scene_profile: str = "isaac", upper_drive: str | None = None):
        self.runner = V5xRunner(package, Path(checkpoint), level, homie_checkpoint, scene_profile, upper_drive)
        self.homie_checkpoint = self.runner.homie_checkpoint
        self.env = self.runner.env
        self.dt = self.env.dt
        self.level = level
        self.hl_checkpoint = Path(checkpoint)
        self.name = name or f"{package}:{self.hl_checkpoint.parent.name}/{self.hl_checkpoint.stem}"
        m = self.env.model
        self.cart_body = self.runner.cart_body
        self.cart_qvel = int(m.joint("cart_free").dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._pair_friction0 = m.pair_friction.copy()
        self._fallen = False
        self._payload = 0.0

    def describe(self) -> dict:
        from .sim2sim_scene import model_report
        out = super().describe()
        out.update(kind="v5x", package=self.runner.package, frame_dim=self.runner.frame_dim,
                   static_prior=self.runner.has_prior,
                   sim2sim=model_report(self.env),
                   observation_model="MuJoCo rebuild of the Isaac V5.x frame; noise/delay off")
        return out

    def set_payload_com_offset(self, dx: float, dy: float):
        if self.env.scene_profile == "isaac":
            self.runner.payload_offset = (float(dx), float(dy))
        else:
            super().set_payload_com_offset(dx, dy)

    def restore_physics(self):
        super().restore_physics()
        self.runner.payload_offset = (0., 0.)
        self.env.model.pair_friction[:] = self._pair_friction0

    def set_wheel_friction(self, mu: float):
        super().set_wheel_friction(mu)
        if self.env.scene_profile == "isaac":
            m = self.env.model
            for i in range(4):
                m.pair_friction[m.pair(f"wheel_floor_{i}").id, :2] = (float(mu) + 1.) / 2.
