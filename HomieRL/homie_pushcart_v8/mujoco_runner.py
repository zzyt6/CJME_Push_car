"""V8 inference on the V7 MuJoCo plant (sim2sim bench), same pipeline as the Isaac V8 environment:

* high level: V3.2 action pipeline for the first 18 actions (HOMIE command residual + upper body), unchanged;
* low level: HOMIE's frozen actor receives the push-cart fine-tuned velocity estimate (homie_velocity_estimator_<N>.pt,
  paired with the checkpoint) and HOMIE's own latent; the clipped 12-dim leg residual is added to its output, and HOMIE
  observes the applied (final) leg action, as in training;
* observation frame (126): HOMIE 76 | task command 3 | upper torque 15 | camera relative yaw 1 | applied actions 30 |
  residual scale 1, five frames; the camera model is training's at full noise (latency 2-5 steps, 25/50 Hz, 1.5 deg,
  3 % dropout), seeded per episode;
* teacher privileged vector (62): built from the MuJoCo state with the Isaac definitions (task_env._privileged_v8). The
  MuJoCo plant has nominal (mid-range) grasp / caster parameters, no yaw damping, no bias / bump forces.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from homie_pushcart.mujoco_hl import MujocoHighLevelRunner
from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
from homie_isaaclab.config import G1RoughCfg
from homie_pushcart_v7.baseline import apply_config, homie_baseline
from homie_pushcart_v7.physics import make_mujoco_env

from .task_config import (ACTION_DIM, FRAME_DIM, HL_ACTIONS, LONG_HISTORY, POLICY_HISTORY, PRIVILEGED_DIM, leg_residual_filter_alpha,
                          leg_residual_map, task_cfg_from_run)


class CameraYaw:
    """Delayed, rate-limited, noisy, occasionally missed relative yaw (training model at noise level 1)."""

    def __init__(self, cam, noise_level=1.0):
        self.cam, self.noise = cam, float(noise_level)
        self.reset(0)

    def reset(self, seed):
        cam, n = self.cam, self.noise
        self.rng = np.random.default_rng(10_000 + int(seed))
        self.latency = int(round(int(self.rng.integers(cam.latency_steps[0], cam.latency_steps[1] + 1)) * n))
        self.period = int(self.rng.integers(cam.update_period_steps[0], cam.update_period_steps[1] + 1))
        self.phase = int(self.rng.integers(0, 2)) % self.period
        self.ring, self.value, self.t = None, 0.0, 0

    def __call__(self, true_yaw):
        cam = self.cam
        if self.ring is None:                          # episode start: no stale pose
            self.ring = np.full(int(cam.history_steps), float(true_yaw))
            self.value = float(true_yaw)
        else:
            self.ring = np.roll(self.ring, -1)
            self.ring[-1] = float(true_yaw)
            measured = self.ring[-1 - self.latency] + self.rng.normal() * cam.noise_yaw_rad * self.noise
            if self.t % self.period == self.phase and self.rng.random() >= cam.dropout_prob * self.noise:
                self.value = float(measured)
        self.t += 1
        return self.value


class MujocoV8Runner(MujocoHighLevelRunner):
    def __init__(self, checkpoint, level=17, homie_checkpoint=None, hl_policy=None, estimator=None,
                 camera_noise=1.0, zero_leg_residual=False):
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()                                         # the V8 Actor classes need RSL-RL 5
        from .teacher_policy import load_teacher
        from .student_policy import is_student_run, load_student
        from .training import homie_velocity_file
        from homie_pushcart_v7.interaction_policy import homie_encoder_from_checkpoint
        checkpoint = Path(checkpoint)
        task = task_cfg_from_run(checkpoint, verbose=False)
        cfg = G1RoughCfg()
        apply_config(cfg, homie_baseline())
        env = make_mujoco_env(cfg=cfg)
        self.homie_checkpoint = Path(homie_checkpoint or frozen_checkpoint())
        low = HimLowLevelPolicy(self.homie_checkpoint, "cpu")
        self.v8 = task.v8
        # teacher run -> privileged teacher; distillation run -> deployable student (32-frame history)
        self.student = hl_policy is None and is_student_run(checkpoint)
        self.policy = hl_policy if hl_policy is not None else (load_student(checkpoint) if self.student
                                                                else load_teacher(checkpoint))
        self.zero_leg_residual = bool(zero_leg_residual)
        # HOMIE input layer: the fine-tuned velocity estimator paired with this checkpoint
        self.hv, self.hv_path = None, None
        if task.v8.homie_velocity.enabled:
            path = Path(estimator) if estimator else homie_velocity_file(checkpoint)
            if path is None and task.v8.homie_velocity.init_checkpoint:     # student run: the teacher's estimator
                path = Path(task.v8.homie_velocity.init_checkpoint)
            if path is None:
                raise FileNotFoundError(f"no homie_velocity_estimator for {checkpoint}")
            self.hv = homie_encoder_from_checkpoint(self.homie_checkpoint)
            self.hv.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=False)["encoder"])
            self.hv.eval()
            self.hv_path = path
        if task.v8.upper_drive.mode == "explicit":            # V8.1: the run's upper-body gains (MuJoCo PD is explicit)
            from homie_pushcart.mujoco_env import HOMIE_JOINTS
            from .task_config import upper_drive_gains
            lower = env.num_lower
            kp, kd = upper_drive_gains(task, HOMIE_JOINTS[lower:])
            env.p_gains[lower:], env.d_gains[lower:] = kp, kd
        self.camera = CameraYaw(task.v8.camera, camera_noise)
        self.seed = 0
        super().__init__(env, low, None, task, level)
        self.hl = self.policy                                   # describe(): iteration
        from homie_pushcart_v7.assets.build import load_v7_params, nominal_grasp
        from homie_pushcart_v7.physics import MujocoCasters, MujocoGrasp
        self.params = load_v7_params()
        self.grasp = MujocoGrasp(env, self.params)
        self.grasp.reset_anchor()
        self.casters = MujocoCasters(env, self.params)
        g, nom = self.params["grasp"], nominal_grasp(self.params)
        mean = lambda r: 0.5 * (float(r[0]) + float(r[1]))                                        # noqa: E731
        self.nom = {"roll_stiffness": float(nom["roll_stiffness"]), "roll_capacity": float(nom["roll_capacity"]),
                    "moment_capacity": float(nom["moment_capacity"]), "pull": mean(g["pull_capacity_range"]),
                    "lift": mean(g["lift_capacity_range"])}
        m = env.model
        # the MJCF merges each palm into its wrist-yaw link: palm origin = wrist pose (+) the URDF's fixed offset
        # (g1_grasp_v7.urdf, *_hand_palm_joint)
        self.palm_body = [m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
        self.palm_offset = [np.array([0.0415, 0.003, 0.0]), np.array([0.0415, -0.003, 0.0])]
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.reset_buffers()

    # ------------------------------------------------------------------ model / episode
    def set_payload(self, payload_kg: float):
        super().set_payload(payload_kg)
        if hasattr(self, "casters"):
            self.casters.set_cart_mass(float(self.env.cart_params["mass"]["base_kg"]) + float(payload_kg))

    def reset_buffers(self):
        super().reset_buffers()
        self.leg_residual = np.zeros(12)
        self.leg_raw = np.zeros(12)
        self.homie_leg_actions = np.zeros(12)
        self.frames = np.zeros((POLICY_HISTORY, FRAME_DIM), dtype=np.float32)
        self.frames_long = np.zeros((LONG_HISTORY, FRAME_DIM), dtype=np.float32)
        self.history_obs = np.zeros(LONG_HISTORY * FRAME_DIM, dtype=np.float32)
        self.explicit_est = np.zeros(19, dtype=np.float32)
        self.privileged = np.zeros(PRIVILEGED_DIM, dtype=np.float32)
        self.slip_v = np.zeros(2)
        self.episode_step = 0
        self.vel_est_homie = np.zeros(3)
        if hasattr(self, "grasp"):
            self.grasp.reset_anchor()
        if hasattr(self, "casters"):
            self.casters.update()
        if hasattr(self, "camera"):
            self.camera.reset(self.seed)

    def reset(self, caster_angle=None, seed=None):
        if seed is not None:
            self.seed = int(seed)
        return super().reset(caster_angle)

    # ------------------------------------------------------------------ step
    def step(self, cmd_task):
        tc, env = self.task_cfg, self.env
        self.cmd_task = np.asarray(cmd_task, dtype=float)
        self.cmd_ref = self.cmd_ref + self.ref_alpha * (self.cmd_task - self.cmd_ref)
        gate = min(1.0, float(np.linalg.norm(self.cmd_ref)) / self.gate_speed) if self.gate_speed > 0 else 1.0
        if self.idle_correction is not None:
            self.idle_gate_applied = float(self.idle_correction.gate(
                torch.as_tensor(self.cmd_ref[None], dtype=torch.float32),
                torch.as_tensor(self.cmd_task[None], dtype=torch.float32))[0, 0])
            gate = max(gate, self.idle_gate_applied)
        hl_on = self.hl is not None                              # viewers switch the high level off (= HOMIE passthrough)
        if hl_on:
            with torch.inference_mode():
                second = self.history_obs if self.student else self.privileged
                out = self.policy(torch.from_numpy(self.hl_obs).unsqueeze(0),
                                  torch.from_numpy(second).unsqueeze(0))[0].numpy()
                if self.student:                                 # diagnostics: estimate vs MuJoCo truth
                    self.explicit_est = self.policy.explicit(torch.from_numpy(self.history_obs).unsqueeze(0))[0].numpy()
        else:
            out = np.zeros(ACTION_DIM, dtype=np.float32)
        a = np.clip(out[:HL_ACTIONS], -tc.actions.clip, tc.actions.clip)
        lr = self.v8.leg_residual
        self.leg_raw = out[HL_ACTIONS:ACTION_DIM].copy()          # network output u (diagnostics)
        target = leg_residual_map(self.leg_raw, self.task_cfg)   # range * tanh(u) | clamp(u, +-clip), as in training
        alpha = leg_residual_filter_alpha(self.task_cfg, self.dt)
        self.leg_residual = (np.zeros(12) if (self.zero_leg_residual or not lr.enabled)
                             else self.leg_residual + alpha * (target - self.leg_residual))
        self.hl_actions = a
        target = np.clip(self.cmd_ref + gate * a[:3] * self.residual_scale, self.cmd_lo, self.cmd_hi)
        self.cmd_homie = self.cmd_homie + self.cmd_alpha * (target - self.cmd_homie)
        env.set_command(*self.cmd_homie)
        env.obs_hist[-1, :3] = env.commands[:3] * self.commands_scale
        delta_target = gate * a[3:] * self.upper_scale
        delta = self.upper_delta_q + np.clip(delta_target - self.upper_delta_q, -self.step_lim, self.step_lim)
        q_target = np.clip(self.hold_target_q + delta, self.upper_lo, self.upper_hi)
        self.upper_delta_q = q_target - self.hold_target_q
        env.upper_delta[:] = self.upper_delta_q / env.action_scale
        obs_t = torch.from_numpy(np.clip(env.obs_hist.reshape(-1), -env.clip_obs, env.clip_obs)).unsqueeze(0).float()
        with torch.inference_mode():
            m = self.lowlevel.model
            history = obs_t[:, :m.actor_proprioceptive_obs_length]
            vel_homie, latent = m.estimator(history)
            vel = self.hv(history)[:, :3] if (self.hv is not None and hl_on) else vel_homie
            legs = m.actor(torch.cat((obs_t[:, -m.num_one_step_obs:], vel, latent), dim=-1))[0].numpy()
        self.vel_est = vel[0].numpy() / env.obs_scales["lin_vel"]
        self.vel_est_homie = vel_homie[0].numpy() / env.obs_scales["lin_vel"]
        self.homie_leg_actions = legs
        anchor_prev = None if self.grasp.anchor is None else self.grasp.anchor.copy()
        self._obs, fallen = env.step(legs + self.leg_residual)
        if getattr(self, "obs_noise", False):
            # the Isaac training noise (homie_isaaclab task_core._get_noise_scale_vec, add_noise, level 1): uniform on
            # each new HOMIE frame -- ang vel 0.5*0.5, gravity 0.05, joint pos 0.02, joint vel 2.0*0.05
            if not hasattr(self, "_noise_vec"):
                v = np.zeros(env.obs_hist.shape[1], dtype=np.float32)
                v[4:7], v[7:10], v[10:37], v[37:64] = 0.25, 0.05, 0.02, 0.1
                self._noise_vec, self._noise_rng = v, np.random.default_rng(20_000 + int(self.seed))
            env.obs_hist[-1] += ((2 * self._noise_rng.random(self._noise_vec.shape) - 1) * self._noise_vec).astype(np.float32)
        self.grasp.update(self.grasp.hand_loads())               # elasto-plastic slide for the next step
        self.casters.update()                                    # swivel scrub fading with rolling speed
        self.slip_v = np.zeros(2) if anchor_prev is None else (self.grasp.anchor - anchor_prev) / self.dt
        self.episode_step += 1
        self.task_state()
        self.quality.update(torch.tensor([[*self.ts["cart_v"], self.ts["cart_wz"]]], dtype=torch.float64),
                            torch.from_numpy(self.cmd_ref[None]), torch.tensor([self.ts["cart_yaw"]], dtype=torch.float64),
                            torch.from_numpy(self.ts["cart_force_w"][None]), torch.from_numpy(self.ts["cart_moment_w"][None]))
        self._update_hl_obs(fresh=False)
        return self.hl_obs, fallen

    # ------------------------------------------------------------------ observations
    def _hand_labels(self):
        """Hand on cart, about each palm origin, cart-heading axes, / HAND_WRENCH_SCALE (Isaac ReadOnlyHandLabels)."""
        from homie_pushcart_v7.observation import HAND_WRENCH_SCALE
        d = self.env.data
        W = self.env.grasp_cart_wrenches()                       # world axes, about the cart assembly COM
        com = d.subtree_com[self.cart_body]
        yaw = self.ts["cart_yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        out = []
        for k, side in enumerate(("left", "right")):
            F, M = W[side][:3], W[side][3:]
            b = self.palm_body[k]
            palm = d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.palm_offset[k]
            M_palm = M - np.cross(palm - com, F)
            out.append(np.concatenate((Rt @ F, Rt @ M_palm)))
        return np.concatenate(out) / np.array(HAND_WRENCH_SCALE), float(self.episode_step > 0)

    def _privileged_v8(self):
        tc, ts, env, d = self.task_cfg, self.ts, self.env, self.env.data
        norm = tc.observations.cart_mass_norm
        mass = (self.cart_mass_total - norm[0]) / norm[1]
        cart_v = [ts["cart_v"][0] * 2.0, ts["cart_v"][1] * 2.0, ts["cart_wz"] * 0.5]
        robot_v = env.base_lin_vel * env.obs_scales["lin_vel"]
        hands, valid = self._hand_labels()
        com = [0.0, 0.0]                                          # bench payload: centred
        st, g, nom = self.grasp.state(), self.grasp, self.nom
        static = np.broadcast_to(np.asarray(getattr(g, "slide_static", 0.0), dtype=float), (2,))
        per_hand = []
        for k in range(2):
            per_hand += [st["slide_q"][k] / 0.1, self.slip_v[k] / 0.2, st["rot_q"][0, k] / 0.3, st["rot_q"][1, k] / 0.3,
                         st["rot_q"][2, k] / 0.3, float(static[k]) / 60., g.mu * g.normal / 60.,
                         nom["roll_stiffness"] / 60., nom["roll_capacity"] / 2.5, nom["moment_capacity"] / 6.,
                         nom["pull"] / 80., nom["lift"] / 80.]
        cart = [self.casters.rolling_coeff / 0.04, self.casters.scrub / 0.4, 0., 0., 0., 0.]
        caster = np.array([d.qpos[a] for a in self.swivel_qpos])
        wheel_mu = float(self.env.model.geom_friction[self.wheel_geoms[0], 0])
        out = np.concatenate((cart_v, [mass], robot_v, hands, [valid], com, per_hand, cart,
                              [0.0], [wheel_mu], np.sin(caster), np.cos(caster))).astype(np.float32)
        assert out.shape[0] == PRIVILEGED_DIM, out.shape
        clip = self.env.clip_obs
        return np.clip(np.nan_to_num(out), -clip, clip).astype(np.float32)

    def _update_hl_obs(self, fresh):
        super()._update_hl_obs(fresh)                            # legacy 5 x 124 frames (task command, torque, scale)
        if not hasattr(self, "camera") or not hasattr(self, "leg_residual"):
            return
        legacy = self.actor_hist[-1]
        tau = legacy[80:95] + getattr(self, "tau_obs_offset", 0.0)        # optional diagnostic bias of the torque obs
        if getattr(self, "obs_noise", False) and hasattr(self, "_noise_rng"):  # Isaac: x (1 + 0.05 U(-1, 1))
            tau = tau * (1. + 0.05 * (2 * self._noise_rng.random(15) - 1))
        frame = np.concatenate((legacy[0:79], tau, [self.camera(self.ts["rel_yaw"])],
                                self.hl_actions, self.leg_residual, legacy[123:124])).astype(np.float32)
        assert frame.shape[0] == FRAME_DIM, frame.shape
        if fresh:
            self.frames[:] = 0.
            self.frames_long[:] = 0.
        else:
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames_long = np.roll(self.frames_long, -1, axis=0)
        self.frames[-1] = frame
        self.frames_long[-1] = frame
        clip = self.env.clip_obs
        self.hl_obs = np.clip(self.frames.reshape(-1), -clip, clip).astype(np.float32)
        self.history_obs = np.clip(self.frames_long.reshape(-1), -clip, clip).astype(np.float32)
        self.privileged = self._privileged_v8()
