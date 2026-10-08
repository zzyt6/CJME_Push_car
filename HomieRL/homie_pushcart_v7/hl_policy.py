"""Strict V7 inference loading without importing Isaac simulation packages."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from homie_pushcart.hl_policy import import_rsl_rl5


def inference_actor_config(checkpoint):
    import yaml
    from .task_config import run_config_path
    source = run_config_path(checkpoint)
    agent_file = source.with_name("agent.yaml") if source else None
    if agent_file is None or not agent_file.is_file():
        raise ValueError("V7 inference requires the run's params/env.yaml and params/agent.yaml.")
    saved = yaml.load(agent_file.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(saved, dict) or not isinstance(saved.get("actor"), dict):
        raise ValueError("V7 saved Actor configuration is missing.")
    actor = copy.deepcopy(saved["actor"])
    if actor.pop("class_name", None) != "homie_pushcart_v7.interaction_policy:InteractionActor":
        raise ValueError("V7 requires its InteractionActor checkpoint; earlier versions cannot be loaded implicitly.")
    groups = copy.deepcopy(saved.get("obs_groups", {}))
    if groups != {"actor": ["policy", "estimator"], "critic": ["critic"]}:
        raise ValueError("V7 saved observation groups do not match its approved interface.")
    return actor, groups


class HighLevelPolicy:
    def __init__(self, checkpoint: Path, num_actor_obs=605, num_critic_obs=695,
                 num_actions=18, device="cpu"):
        from .task_config import task_cfg_from_run
        if (num_actor_obs, num_critic_obs, num_actions) != (605, 695, 18):
            raise ValueError("V7 inference requires policy605, Critic695, and actions18.")
        task_cfg_from_run(checkpoint, verbose=False)
        import_rsl_rl5()
        from tensordict import TensorDict
        from .interaction_policy import InteractionActor
        actor_cfg, groups = inference_actor_config(checkpoint)
        probe = TensorDict({"policy": torch.zeros(1, 605), "estimator": torch.zeros(1, 462)}, batch_size=[1])
        self.actor = InteractionActor(probe, groups, "actor", 18, **actor_cfg).to(device)
        loaded = torch.load(str(checkpoint), map_location=device, weights_only=False)
        if loaded.get("v7_network_metadata") != self.actor.metadata():
            raise ValueError("V7 checkpoint metadata is missing or incompatible.")
        self.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
        self.actor.eval()
        self.iteration, self.checkpoint = loaded.get("iter"), Path(checkpoint)
        self.device, self._TensorDict = device, TensorDict
        self.num_actor_obs, self.num_estimator_obs = 605, 462
        self.num_critic_obs, self.num_actions = 695, 18

    def _observations(self, actor_obs, estimator_obs):
        if estimator_obs is None:
            if hasattr(actor_obs, "keys") and {"policy", "estimator"}.issubset(actor_obs.keys()):
                return actor_obs
            raise ValueError("V7 inference needs the independent six-frame estimator history.")
        return self._TensorDict({"policy": actor_obs, "estimator": estimator_obs},
                                batch_size=[actor_obs.shape[0]])

    @torch.inference_mode()
    def __call__(self, actor_obs, estimator_obs=None, critic_obs=None):
        return self.actor(self._observations(actor_obs, estimator_obs))

    @torch.inference_mode()
    def physical_estimates(self, actor_obs, estimator_obs=None):
        return self.actor.physical_estimates(self._observations(actor_obs, estimator_obs))
