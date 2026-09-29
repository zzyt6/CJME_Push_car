"""V3.2 dynamics/control/rewards, with projected observations and read-only labels."""
import torch

from homie_pushcart.task_env import PushCartEnv as V32Env, TERMINATION_REASONS
from homie_isaaclab.math_utils import quat_rotate_inverse
from .task_config import num_actor_obs, num_critic_obs, validate_task_schema
from .observation import (
    HIMHistory, him_frame, without_handle_position, HAND_WRENCH_SCALE,
)
from .him_estimator import make_next_critic_observation
from .hand_labels import ReadOnlyHandLabels
from .physics import validate_assets


class PushCartEnv(V32Env):
    def __init__(self, cfg, task_cfg, *args, **kwargs):
        validate_task_schema(task_cfg)
        validate_assets()
        self._transition_active = False
        super().__init__(cfg, task_cfg, *args, **kwargs)
        self._labels = ReadOnlyHandLabels(self)
        self.num_hl_obs = num_actor_obs(task_cfg)
        self.num_hl_critic_obs = num_critic_obs(task_cfg)

    def _init_hl_buffers(self):
        # Keep the original 124x5 history and 652 critic construction internally.
        super()._init_hl_buffers()
        self.estimator_history = HIMHistory(self.num_envs, self.device)
        self.estimator_obs = self.estimator_history.buffer.flatten(1)
        self.legacy_actor_obs = self.hl_obs
        self.legacy_critic_obs = self.hl_critic_obs
        self.hand_scale = torch.tensor(HAND_WRENCH_SCALE, device=self.device)
        self.hand_target = torch.zeros(self.num_envs, 13, device=self.device)
        self.him_next_target = torch.zeros(self.num_envs, 81, device=self.device)
        self._pre_reset_hands = self.hand_target.clone()

    def _post_physics_step_callback(self):
        # This call alone computes the original state, filters and diagnostics.
        super()._post_physics_step_callback()
        if not hasattr(self, "_labels"):
            return
        hands, valid_hand = self._labels.sample()
        self._pre_reset_hands = torch.cat((hands/self.hand_scale, valid_hand[:, None]), -1).detach()
        # HOMIE's target branch uses the clean next privileged frame, not noisy
        # actor history. Construct it before termination/reset, as original HIM.
        # Do not call compute_termination_observations: that method samples
        # observation noise and would advance V3.2's random-number stream.
        imu_q = self.rigid_body_states[:, self.imu_index, 3:7]
        imu_w = quat_rotate_inverse(imu_q, self.rigid_body_states[:, self.imu_index, 10:13])
        imu_g = quat_rotate_inverse(imu_q, self.gravity_vec)
        clean_low = torch.cat((self.commands[:, :3]*self.commands_scale,
            self.commands[:, 4:5], imu_w*self.obs_scales.ang_vel, imu_g,
            (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
            self.dof_vel*self.obs_scales.dof_vel, self.actions[:, :12]), -1)
        next_frame = torch.cat((clean_low, self.ts.rel_yaw[:, None]), -1)
        speed = torch.cat((self.ts.cart_v, self.ts.cart_wz[:, None]), -1)
        target = make_next_critic_observation(next_frame, speed)
        valid_speed = torch.isfinite(target).all(-1) & (self.episode_length_buf > 0)
        self.him_next_target = torch.cat((torch.where(valid_speed[:, None], target, 0.),
                                          valid_speed[:, None]), -1).detach()

    def _update_hl_observations(self):
        # Exactly one original observation update: same noisy yaw, same random
        # draw count, same original history and reward-only true state sources.
        super()._update_hl_observations()
        self.legacy_actor_obs = self.hl_obs
        self.legacy_critic_obs = self.hl_critic_obs
        fresh = self.episode_length_buf == 0
        source = self.legacy_actor_obs.reshape(self.num_envs, 5, 124)
        self.hl_obs = without_handle_position(self.legacy_actor_obs)
        self.estimator_obs = self.estimator_history.push(him_frame(source[:, -1]), fresh)
        # A reset observation has no hand label. Never change the separate
        # pre-reset t+1 target: PPO still needs it for the terminal transition.
        self.hand_target = torch.where(fresh[:, None], 0., self._pre_reset_hands).clone()
        self.hl_critic_obs = torch.cat((self.legacy_critic_obs, self.hand_target), -1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not len(env_ids) or not hasattr(self, "estimator_history"):
            return
        self.estimator_history.reset(env_ids)
        self._pre_reset_hands = self._pre_reset_hands.clone()
        self._pre_reset_hands[env_ids] = 0.
        if not self._transition_active:
            self.him_next_target = self.him_next_target.clone()
            self.him_next_target[env_ids] = 0.

    def step(self, actions):
        self._transition_active = True
        try:
            return super().step(actions)
        finally:
            self._transition_active = False

    def observation_dict(self):
        return {"policy": self.hl_obs, "estimator": self.estimator_obs,
                "critic": self.hl_critic_obs, "hand_target": self.hand_target,
                "him_next_target": self.him_next_target}
