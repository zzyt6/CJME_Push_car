"""V6 inference on the unmodified V3.2 MuJoCo plant and action pipeline."""
from pathlib import Path
import numpy as np
import torch

from homie_pushcart.mujoco_hl import MujocoHighLevelRunner
from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
from homie_isaaclab.config import G1RoughCfg
from .baseline import apply_config, homie_baseline
from .physics import make_mujoco_env
from .task_config import task_cfg_from_run
from .observation import HIMHistory, him_frame, without_handle_position


class _PolicyBridge:
    def __init__(self, runner, policy):
        self.runner, self.policy = runner, policy
        self.iteration = policy.iteration

    def __call__(self, obs):
        return self.policy(obs, self.runner.estimator_obs)


class MujocoV6Runner(MujocoHighLevelRunner):
    def __init__(self, checkpoint, level=17, homie_checkpoint=None):
        task = task_cfg_from_run(checkpoint)
        cfg = G1RoughCfg()
        apply_config(cfg, homie_baseline())
        env = make_mujoco_env(cfg=cfg)
        self.homie_checkpoint = Path(homie_checkpoint or frozen_checkpoint())
        low = HimLowLevelPolicy(self.homie_checkpoint, "cpu")
        from .hl_policy import HighLevelPolicy
        self.v6_policy = HighLevelPolicy(Path(checkpoint), device="cpu")
        super().__init__(env, low, None, task, level)
        self.hl = _PolicyBridge(self, self.v6_policy)

    def reset_buffers(self):
        super().reset_buffers()
        self.estimator_history = HIMHistory(1)
        self.estimator_obs = self.estimator_history.buffer.flatten(1)
        self.pred_cart_velocity, self.pred_hands = np.zeros(3), np.zeros(12)

    def _update_hl_obs(self, fresh):
        super()._update_hl_obs(fresh)
        legacy = torch.from_numpy(self.hl_obs.copy()).unsqueeze(0)
        self.legacy_actor_obs = legacy
        self.estimator_obs = self.estimator_history.push(
            him_frame(legacy.reshape(1, 5, 124)[:, -1]), torch.tensor([fresh]))
        self.hl_obs = without_handle_position(legacy)[0].numpy()
        with torch.inference_mode():
            speed, hands = self.v6_policy.physical_estimates(
                torch.from_numpy(self.hl_obs).unsqueeze(0), self.estimator_obs)
        self.pred_cart_velocity = speed[0].cpu().numpy()
        self.pred_hands = hands[0].cpu().numpy()
