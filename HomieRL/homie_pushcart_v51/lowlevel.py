"""Frozen HOMIE low-level policy: checkpoint discovery and inference wrapper.

Uses the original HIMActorCritic class (homie_rsl_rl) with the constructor
arguments of the training run, so checkpoints load with strict=True.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import torch

from .paths import HOMIE_FROZEN_CHECKPOINT, HOMIE_RUN_DIR, ROOT

if str(ROOT / "HomieRL") not in sys.path:
    sys.path.insert(0, str(ROOT / "HomieRL"))
if str(ROOT / "HomieRL" / "rsl_rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "HomieRL" / "rsl_rl"))


def frozen_checkpoint() -> Path:
    """The pinned HOMIE checkpoint (paths.HOMIE_FROZEN_CHECKPOINT); falls back to the newest one if it is missing."""
    if HOMIE_FROZEN_CHECKPOINT.is_file():
        return HOMIE_FROZEN_CHECKPOINT
    print(f"[pushcart] WARNING: frozen HOMIE checkpoint {HOMIE_FROZEN_CHECKPOINT.name} not found, using the latest one", flush=True)
    return latest_checkpoint(HOMIE_RUN_DIR)


def latest_checkpoint(run_dir: Path = HOMIE_RUN_DIR) -> Path:
    files = list(Path(run_dir).glob("model_*.pt"))
    if not files:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")

    def index(path: Path) -> int:
        match = re.search(r"model_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(files, key=index)


class HimLowLevelPolicy:
    """Deterministic HOMIE inference: 456-dim actor observation -> 12 leg actions."""

    def __init__(self, checkpoint: Path, device: str = "cuda:0"):
        from homie_isaaclab.config import G1RoughCfg, G1RoughCfgPPO
        from homie_isaaclab.math_utils import class_to_dict
        from homie_rsl_rl.modules import HIMActorCritic

        env_cfg, train_cfg = G1RoughCfg(), G1RoughCfgPPO()
        e = env_cfg.env
        self.model = HIMActorCritic(
            e.num_observations, e.num_privileged_obs, e.num_one_step_observations,
            e.num_one_step_privileged_obs, e.num_actor_history, e.num_critic_history, e.num_actions,
            **class_to_dict(train_cfg.policy),
        ).to(device)
        # HOMIE's HIMActorCritic.__init__ does `Normal.set_default_validate_args = False`, which replaces the
        # torch classmethod by a bool and later breaks RSL-RL 5 (`Normal.set_default_validate_args(False)`).
        # Undo it so the frozen low-level policy can live in the same process as the high-level PPO.
        from torch.distributions import Normal
        if isinstance(Normal.__dict__.get("set_default_validate_args"), bool):
            del Normal.set_default_validate_args
            Normal.set_default_validate_args(False)
        loaded = torch.load(str(checkpoint), map_location=device, weights_only=False)
        state = loaded["model_state_dict"] if "model_state_dict" in loaded else loaded
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.checkpoint = Path(checkpoint)
        self.iteration = loaded.get("iter", None)
        self.num_obs, self.num_actions = e.num_observations, e.num_actions

    @torch.inference_mode()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model.act_inference(obs)

    @torch.inference_mode()
    def estimate_velocity(self, obs: torch.Tensor) -> torch.Tensor:
        """HIM scaled velocity for the supplied observation time, without taking an action."""
        return self.model.estimator(obs[:, :self.model.actor_proprioceptive_obs_length])[0]

    @torch.inference_mode()
    def act_and_estimate(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Leg actions plus HIM's own base-velocity estimate (in HOMIE's scaled units, i.e. m/s x obs_scales.lin_vel).

        Same computation as ``act_inference`` for the height-free HOMIE profile; the estimate is a deployable
        proprioceptive signal, which is why the high-level policy observes it instead of the true base velocity.
        """
        m = self.model
        vel, latent = m.estimator(obs[:, :m.actor_proprioceptive_obs_length])
        actions = m.actor(torch.cat((obs[:, -m.num_one_step_obs:], vel, latent), dim=-1))
        return actions, vel
