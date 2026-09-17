"""Load a trained high-level (RSL-RL 5) actor from a checkpoint for inference outside the training loop.

Used by the Isaac ``eval`` command and by the MuJoCo sim2sim tools; independent of any simulator.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .paths import ROOT


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
    """Deterministic actor: (actor obs [N, 620], critic obs [N, 652] or None) -> actions [N, 18] in [-1, 1]."""

    def __init__(self, checkpoint: Path, num_actor_obs: int, num_critic_obs: int, num_actions: int, device: str = "cpu"):
        import_rsl_rl5()
        from tensordict import TensorDict
        from rsl_rl.models import MLPModel
        from .lab_config import PushCartRunnerCfg
        cfg = PushCartRunnerCfg()
        actor_cfg = cfg.actor.to_dict()
        actor_cfg.pop("class_name", None)
        for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            actor_cfg.pop(key, None)
        probe = TensorDict({"policy": torch.zeros(1, num_actor_obs), "critic": torch.zeros(1, num_critic_obs)}, batch_size=[1])
        self.actor = MLPModel(probe, cfg.obs_groups, "actor", num_actions, **actor_cfg).to(device)
        loaded = torch.load(str(checkpoint), map_location=device, weights_only=False)
        self.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
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
