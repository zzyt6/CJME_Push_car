"""Load a trained high-level (RSL-RL 5) actor from a checkpoint for inference outside the training loop.

Used by the Isaac ``eval`` command and by the MuJoCo sim2sim tools; independent of any simulator.
"""
from __future__ import annotations

from pathlib import Path
import copy

import torch

from .paths import ROOT


def inference_actor_config(checkpoint):
    """Restore architecture, activation and normalizer; dimensions alone cannot detect activation drift."""
    import yaml
    from .task_config import run_config_path
    from .lab_config import PushCartRunnerCfg
    defaults = PushCartRunnerCfg()
    actor = defaults.actor.to_dict()
    groups = copy.deepcopy(defaults.obs_groups)
    source = run_config_path(checkpoint)
    agent_file = source.with_name("agent.yaml") if source else None
    if agent_file is None or not agent_file.is_file():
        raise ValueError("V5 standalone inference requires the run's params/env.yaml and params/agent.yaml beside the checkpoint; "
                         "activation and controller settings cannot be recovered safely from weights alone")
    if agent_file.is_file():
        saved = yaml.load(agent_file.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
        if not isinstance(saved, dict) or not isinstance(saved.get("actor"), dict):
            raise ValueError(f"Saved actor configuration is missing: {agent_file}")
        actor.update(copy.deepcopy(saved["actor"]))
        groups = copy.deepcopy(saved.get("obs_groups", groups))
    if actor.pop("class_name", "") != "homie_pushcart_v51.interaction_policy:InteractionActor":
        raise ValueError("V5.1 requires its InteractionActor checkpoint")
    if groups.get("actor") != ["policy"]:
        raise ValueError("V5 deployable actor must consume only the policy observation group")
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        actor.pop(key, None)
    return actor, groups


def import_rsl_rl5():
    """Import the installed RSL-RL 5 even though ``pushcart.py`` puts HOMIE's legacy fork first on sys.path."""
    import importlib
    import sys
    fork = (ROOT / "HomieRL" / "rsl_rl").resolve()
    module = sys.modules.get("rsl_rl")
    if module is not None and not (Path(module.__file__).parent / "models").is_dir():
        for name in [n for n in sys.modules if n == "rsl_rl" or n.startswith("rsl_rl.")]:
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != fork]
    rsl_rl = importlib.import_module("rsl_rl")
    if not (Path(rsl_rl.__file__).parent / "models").is_dir():
        raise ImportError(f"installed rsl_rl 5 not found; got {rsl_rl.__file__}")
    return rsl_rl


class HighLevelPolicy:
    """Deterministic V5 actor, with observation dimensions taken from the run config."""

    def __init__(self, checkpoint: Path, num_actor_obs: int, num_critic_obs: int, num_actions: int, device: str = "cpu"):
        from .task_config import task_cfg_from_run
        task_cfg_from_run(checkpoint, verbose=False)  # rejects pre-V5 saved configurations
        loaded = torch.load(str(checkpoint), map_location=device, weights_only=False)
        state = loaded["actor_state_dict"]
        import_rsl_rl5()
        from tensordict import TensorDict
        from .interaction_policy import InteractionActor
        actor_cfg, obs_groups = inference_actor_config(checkpoint)
        probe = TensorDict({"policy": torch.zeros(1, num_actor_obs), "critic": torch.zeros(1, num_critic_obs)}, batch_size=[1])
        self.actor = InteractionActor(probe, obs_groups, "actor", num_actions, **actor_cfg).to(device)
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval()
        self.iteration = loaded.get("iter")
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.num_actor_obs, self.num_critic_obs, self.num_actions = num_actor_obs, num_critic_obs, num_actions
        self._TensorDict = TensorDict

    @torch.inference_mode()
    def __call__(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None = None) -> torch.Tensor:
        n = actor_obs.shape[0]
        if critic_obs is None:
            critic_obs = torch.zeros(n, self.num_critic_obs, device=actor_obs.device)
        td = self._TensorDict({"policy": actor_obs, "critic": critic_obs}, batch_size=[n])
        return self.actor(td)
