"""V8 environment: the V7.1 plant with V8 observations (teacher privileged / student deployable) and a gated residual
on HOMIE's leg actions. Everything physical (cart, grasp, casters, disturbances, curriculum) is V7.1's."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from homie_isaaclab.math_utils import torch_rand_float
from homie_pushcart_v7.task_env import PushCartEnv as V71Env

from .task_config import (ACTION_DIM, FRAME_DIM, HL_ACTIONS, LEG_ACTIONS, LONG_HISTORY, POLICY_HISTORY,
                          PRIVILEGED_DIM, validate_v8)

LEGACY_FRAME = 124


class PushCartEnv(V71Env):
    """Observation groups: policy (5 x 126), privileged (62), critic (692), optional history (32 x 126).
    Actions (30): V7's 18 high-level actions, then 12 residual leg actions."""

    def __init__(self, cfg, task_cfg, *args, **kwargs):
        validate_v8(task_cfg)
        super().__init__(cfg, task_cfg, *args, **kwargs)
        self.num_hl_obs = FRAME_DIM * POLICY_HISTORY
        self.num_hl_critic_obs = self.num_hl_obs + PRIVILEGED_DIM
        self.homie_velocity_save_path = None            # set by the direct env (run directory)
        self._setup_homie_velocity()
        self._wrap_lowlevel()

    # ------------------------------------------------------------------ buffers
    def _init_hl_buffers(self):
        super()._init_hl_buffers()
        dev, N = self.device, self.num_envs
        cam = self.task_cfg.v8.camera
        z = lambda *shape: torch.zeros(*shape, device=dev)                                # noqa: E731
        self.leg_residual = z(N, LEG_ACTIONS)            # applied this step (HOMIE action units)
        self.leg_residual_prev = z(N, LEG_ACTIONS)
        self._leg_residual_raw_prev = z(N, LEG_ACTIONS)  # for the shared high-level action delay
        self.v8_actions = z(N, ACTION_DIM)               # applied 30 actions (observation)
        self.cam_true = z(N, cam.history_steps, 1)       # ring of the true relative yaw
        self.cam_pose = z(N, 1)                          # current camera measurement (held between updates)
        self.cam_latency = torch.zeros(N, dtype=torch.long, device=dev)
        self.cam_period = torch.ones(N, dtype=torch.long, device=dev)
        self.cam_phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.frames_short = z(N, POLICY_HISTORY, FRAME_DIM)
        self.frames_long = z(N, LONG_HISTORY, FRAME_DIM)
        self.privileged = z(N, PRIVILEGED_DIM)
        self.v8_policy_obs = z(N, FRAME_DIM * POLICY_HISTORY)
        self.v8_critic_obs = z(N, FRAME_DIM * POLICY_HISTORY + PRIVILEGED_DIM)

    def _setup_homie_velocity(self):
        """Input-layer adaptation of HOMIE: a copy of its HIM encoder, fine-tuned on the true pelvis velocity in the
        push-cart scene, provides the velocity HOMIE's leg actor sees (HOMIE's latent and weights stay frozen)."""
        hv, dev, N = self.task_cfg.v8.homie_velocity, self.device, self.num_envs
        m = self.policy.model
        self.hv_enabled, self.hv_train = bool(hv.enabled), bool(hv.enabled and hv.train)
        self.hv_len = int(m.actor_proprioceptive_obs_length)
        self.hv_err_ft, self.hv_err_homie = torch.zeros(3, device=dev), torch.zeros(3, device=dev)
        self.hv_updates = 0
        self.hv_use_tuned = False                       # gate: feed HOMIE the tuned velocity only once it is better
        self.hv_active_frac = torch.zeros((), device=dev)
        if not self.hv_enabled:
            return
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from homie_pushcart_v7.interaction_policy import homie_encoder_from_checkpoint
        self.hv_encoder = homie_encoder_from_checkpoint(self.policy.checkpoint).to(dev)
        if hv.init_checkpoint:
            saved = torch.load(str(hv.init_checkpoint), map_location=dev, weights_only=False)
            self.hv_encoder.load_state_dict(saved["encoder"])
            self.hv_updates = int(saved.get("updates", 0))
            print(f"[pushcart-v8] HOMIE velocity estimator from {hv.init_checkpoint} ({self.hv_updates} updates)", flush=True)
        for p in self.hv_encoder.parameters():
            p.requires_grad_(self.hv_train)
        if self.hv_train:
            T = int(hv.rollout_steps)
            self.hv_opt = torch.optim.Adam(self.hv_encoder.parameters(), lr=float(hv.learning_rate))
            self.hv_obs = torch.zeros(T, N, self.hv_len, device=dev)
            self.hv_tgt = torch.zeros(T, N, 3, device=dev)
            self.hv_valid = torch.zeros(T, N, dtype=torch.bool, device=dev)
            self.hv_ptr = 0

    def _wrap_lowlevel(self):
        """HOMIE's leg actions + the applied residual. HOMIE's own observation keeps the final (applied) actions.
        With the input-layer adaptation, HOMIE's actor gets the fine-tuned velocity and HOMIE's own latent."""
        original = self.policy.act_and_estimate
        m = self.policy.model

        def act_with_residual(obs):
            if self.hv_enabled:
                with torch.no_grad():
                    history = obs[:, :self.hv_len]
                    vel_homie, latent = m.estimator(history)
                    vel_tuned = self.hv_encoder(history)[:, :3]
                    vel = vel_tuned if self.hv_use_tuned else vel_homie
                    legs = m.actor(torch.cat((obs[:, -m.num_one_step_obs:], vel, latent), dim=-1))
                    truth = self.base_lin_vel * self.obs_scales.lin_vel
                    valid = self.episode_length_buf >= int(self.task_cfg.v8.homie_velocity.valid_after_steps)
                    if valid.any():                      # running error (m/s) for the logs
                        scale = float(self.obs_scales.lin_vel)
                        self.hv_err_ft.lerp_((vel_tuned - truth)[valid].abs().mean(0) / scale, 0.01)
                        self.hv_err_homie.lerp_((vel_homie - truth)[valid].abs().mean(0) / scale, 0.01)
                        # both errors on the same states; hysteresis keeps the switch from chattering
                        ratio = float(self.hv_err_ft.sum() / self.hv_err_homie.sum().clamp_min(1e-6))
                        self.hv_use_tuned = ratio < (1.0 if self.hv_use_tuned else 0.98)
                    self.hv_active_frac.lerp_(torch.tensor(float(self.hv_use_tuned), device=self.device), 0.01)
                    if self.hv_train:
                        self.hv_obs[self.hv_ptr].copy_(history)
                        self.hv_tgt[self.hv_ptr].copy_(truth)
                        self.hv_valid[self.hv_ptr].copy_(valid)
            else:
                legs, vel = original(obs)
            self.homie_leg_actions = legs                # HOMIE's own output (diagnostics / tests)
            return legs + self.leg_residual, vel
        self.policy.act_and_estimate = act_with_residual

    def _train_homie_velocity(self):
        hv = self.task_cfg.v8.homie_velocity
        with torch.inference_mode(False), torch.enable_grad():
            x = self.hv_obs.reshape(-1, self.hv_len).clone()
            y = self.hv_tgt.reshape(-1, 3).clone()
            keep = self.hv_valid.reshape(-1).clone()
            x, y = x[keep], y[keep]
            if len(x) == 0:
                return
            for _ in range(int(hv.updates_per_rollout)):
                ids = torch.randint(0, len(x), (min(int(hv.batch_size), len(x)),), device=x.device)
                loss = torch.nn.functional.mse_loss(self.hv_encoder(x[ids])[:, :3], y[ids])
                self.hv_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hv_encoder.parameters(), float(hv.max_grad_norm))
                self.hv_opt.step()
        self.hv_updates += 1
        if self.homie_velocity_save_path and (self.hv_updates - 1) % int(hv.save_every_updates) == 0:
            self.save_homie_velocity()

    def save_homie_velocity(self, path=None):
        """Latest estimator + a copy named by the PPO iteration it belongs to. The estimator updates at the end of each
        rollout, before the runner saves model_<it>.pt, so iteration it = updates - 1."""
        path = path or self.homie_velocity_save_path
        if path and self.hv_enabled:
            data = {"encoder": {k: v.detach().cpu() for k, v in self.hv_encoder.state_dict().items()},
                    "updates": self.hv_updates, "architecture": [self.hv_len, 256, 256, 35],
                    "init": "frozen HOMIE HIM encoder", "target": "pelvis body-frame velocity x lin_vel scale"}
            path = Path(path)
            torch.save(data, str(path))
            torch.save(data, str(path.with_name(f"{path.stem}_{self.hv_updates - 1}{path.suffix}")))

    # ------------------------------------------------------------------ step
    def step(self, actions: torch.Tensor):
        lr = self.task_cfg.v8.leg_residual
        actions = actions.to(self.device)
        raw = actions[:, HL_ACTIONS:].clamp(-lr.clip, lr.clip)
        applied = torch.where(self.hl_delay.unsqueeze(1) > 0, self._leg_residual_raw_prev, raw)
        self._leg_residual_raw_prev.copy_(raw)
        self.leg_residual_prev.copy_(self.leg_residual)
        self.leg_residual.copy_(applied if lr.enabled else torch.zeros_like(applied))
        out = super().step(actions[:, :HL_ACTIONS])
        if getattr(self, "hv_train", False):
            self.hv_ptr += 1
            if self.hv_ptr >= self.hv_obs.shape[0]:
                self._train_homie_velocity()
                self.hv_ptr = 0
        return self.v8_policy_obs, self.v8_critic_obs, out[2], out[3], out[4]

    # ------------------------------------------------------------------ rewards
    def _reward_leg_residual(self):
        return self.leg_residual.square().sum(dim=1)

    def _reward_leg_residual_rate(self):
        return (self.leg_residual - self.leg_residual_prev).square().sum(dim=1)

    def _log_episode_diagnostics(self, env_ids):
        super()._log_episode_diagnostics(env_ids)
        ep = self.extras.setdefault("episode", {})
        if not hasattr(self, "leg_residual"):
            return
        if getattr(self, "hv_enabled", False):
            for i, axis in enumerate(("vx", "vy", "vz")):
                ep[f"homie_velocity_mae_tuned_{axis}"] = self.hv_err_ft[i].clone()
                ep[f"homie_velocity_mae_original_{axis}"] = self.hv_err_homie[i].clone()
            ep["homie_velocity_updates"] = torch.tensor(float(self.hv_updates), device=self.device)
            ep["homie_velocity_tuned_active"] = self.hv_active_frac.clone()
        ep["leg_residual_abs"] = self.leg_residual.abs().mean()

    # ------------------------------------------------------------------ resets
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not hasattr(self, "cam_latency") or len(env_ids) == 0:
            return
        cam, dev, n = self.task_cfg.v8.camera, self.device, len(env_ids)
        ids = env_ids.to(torch.long)
        noise = float(self.level_params.obs_noise_frac)
        lat = torch.randint(int(cam.latency_steps[0]), int(cam.latency_steps[1]) + 1, (n,), device=dev)
        self.cam_latency[ids] = torch.round(lat.float() * noise).long()               # 0 latency at noise level 0
        self.cam_period[ids] = torch.randint(int(cam.update_period_steps[0]), int(cam.update_period_steps[1]) + 1,
                                             (n,), device=dev)
        self.cam_phase[ids] = torch.randint(0, 2, (n,), device=dev) % self.cam_period[ids]
        self.cam_true[ids] = 0.
        self.cam_pose[ids] = 0.
        self.leg_residual[ids] = 0.
        self.leg_residual_prev[ids] = 0.
        self._leg_residual_raw_prev[ids] = 0.
        self.frames_short[ids] = 0.
        self.frames_long[ids] = 0.

    # ------------------------------------------------------------------ observations
    def _camera(self):
        """Delayed, rate-limited, noisy, occasionally missed relative yaw of the cart."""
        cam, ts, N, dev = self.task_cfg.v8.camera, self.ts, self.num_envs, self.device
        true = ts.rel_yaw.unsqueeze(1)
        fresh = self.episode_length_buf == 0
        self.cam_true = torch.cat((self.cam_true[:, 1:], true.unsqueeze(1)), 1)
        self.cam_true[fresh] = true[fresh].unsqueeze(1)                                  # no stale pose after reset
        delayed = self.cam_true[torch.arange(N, device=dev), -1 - self.cam_latency]
        noise = float(self.level_params.obs_noise_frac)
        measured = delayed + torch.randn(N, 1, device=dev) * (cam.noise_yaw_rad * noise)
        update = ((self.episode_length_buf % self.cam_period) == self.cam_phase)
        update &= torch.rand(N, device=dev) >= cam.dropout_prob * noise
        update |= fresh
        self.cam_pose = torch.where(update.unsqueeze(1), measured, self.cam_pose)
        return self.cam_pose

    def _privileged_v8(self):
        ts, tc, N = self.ts, self.task_cfg, self.num_envs
        norm = tc.observations.cart_mass_norm
        mass = ((self.cart_true_mass - norm[0]) / norm[1]).unsqueeze(1)
        cart_v = torch.cat((ts.cart_v * 2.0, (ts.cart_wz * 0.5).unsqueeze(1)), -1)          # V7 cart HIM scale
        robot_v = self.base_lin_vel * self.obs_scales.lin_vel
        hands = self.hand_target                                                            # 12 scaled + valid
        com = self.payload_com / max(tc.randomization.payload_com_offset_m, 1e-6)
        caster = torch.cat((torch.sin(ts.caster_angles), torch.cos(ts.caster_angles)), -1)
        out = torch.cat((cart_v, mass, robot_v, hands[:, :12], hands[:, 12:13], com, self._v7_privileged(),
                         (self.yaw_damping / 10.).unsqueeze(1), self.wheel_friction.unsqueeze(1), caster), -1)
        if out.shape[1] != PRIVILEGED_DIM:
            raise RuntimeError(f"V8 privileged vector has {out.shape[1]} columns, expected {PRIVILEGED_DIM}")
        clip = self.cfg.normalization.clip_observations
        return torch.nan_to_num(out).clamp(-clip, clip)

    def _update_hl_observations(self):
        super()._update_hl_observations()          # V7.1: legacy frame, estimator history, hand labels, critic
        if not hasattr(self, "frames_short"):
            return
        N = self.num_envs
        legacy = self.legacy_actor_obs.reshape(N, POLICY_HISTORY, LEGACY_FRAME)[:, -1]
        self.v8_actions = torch.cat((self.hl_actions, self.leg_residual), -1)
        frame = torch.cat((legacy[:, 0:79],              # HOMIE 76 (noisy) + task command 3
                           legacy[:, 80:95],             # upper-body torque 15 (noisy)
                           self._camera(),               # relative yaw of the cart from the camera 1
                           self.v8_actions,              # previous applied actions 30
                           legacy[:, 123:124]), -1)      # residual scale 1
        fresh = (self.episode_length_buf == 0)
        self.frames_short = torch.cat((self.frames_short[:, 1:], frame.unsqueeze(1)), 1)
        self.frames_short[fresh, :-1] = 0.
        if self.task_cfg.v8.emit_history:
            self.frames_long = torch.cat((self.frames_long[:, 1:], frame.unsqueeze(1)), 1)
            self.frames_long[fresh, :-1] = 0.
        clip = self.cfg.normalization.clip_observations
        self.v8_policy_obs = self.frames_short.flatten(1).clamp(-clip, clip)
        self.privileged = self._privileged_v8()
        self.v8_critic_obs = torch.cat((self.v8_policy_obs, self.privileged), -1)

    def observation_dict(self):
        out = {"policy": self.v8_policy_obs, "privileged": self.privileged, "critic": self.v8_critic_obs}
        if self.task_cfg.v8.emit_history:
            out["history"] = self.frames_long.flatten(1).clamp(-self.cfg.normalization.clip_observations,
                                                               self.cfg.normalization.clip_observations)
        return out
