"""Causal interaction estimation shared with the V5.1 high-level actor.

Only the fixed-scaled policy history is read during inference. The force head
predicts total interaction load (including useful pushing force), not a measured
external force or a decomposition into useful and harmful components.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class InteractionActor(MLPModel):
    """History -> 16D latent + 3D load estimate -> current-frame action policy.

    History is chronological, with the current frame last. Predictions have the
    same timestamp as this frame and units [Fx/80 N, Fy/80 N, Mz/12 Nm]. The
    auxiliary coefficient does not change this actor's architecture.
    """

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims=(256, 256, 256), activation: str = "elu",
        obs_normalization: bool = False, distribution_cfg: dict | None = None,
        frame_dim: int = 124, history_length: int = 20, latent_dim: int = 16,
        encoder_hidden_dims=(256, 128), interaction_hidden_dims=(64,),
    ) -> None:
        if obs_groups[obs_set] != ["policy"]:
            raise ValueError("InteractionActor requires exactly the ['policy'] observation group.")
        if obs_normalization:
            raise ValueError("InteractionActor uses fixed feature scaling; obs_normalization must be False.")
        if min(frame_dim, history_length, latent_dim) <= 0:
            raise ValueError("Frame, history and latent dimensions must be positive.")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != frame_dim * history_length:
            raise ValueError(f"Expected policy history width {frame_dim * history_length}.")
        self.frame_dim = frame_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, False, copy.deepcopy(distribution_cfg))
        self.history_encoder = MLP(frame_dim * history_length, latent_dim, encoder_hidden_dims, activation)
        self.interaction_head = MLP(latent_dim, 3, interaction_hidden_dims, activation)

    def _get_latent_dim(self) -> int:
        return self.frame_dim + self.latent_dim + 3

    def predict_interaction(self, obs: TensorDict) -> torch.Tensor:
        """Predict normalized load without inspecting critic inputs or labels."""
        return self.interaction_head(self.history_encoder(obs["policy"]))

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        history = obs["policy"]
        latent = self.history_encoder(history)
        # Both prediction and latent remain connected to the PPO gradient.
        interaction = self.interaction_head(latent)
        return torch.cat((history[..., -self.frame_dim:], latent, interaction), dim=-1)

    def as_jit(self) -> nn.Module:
        return _InteractionExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _InteractionOnnxExport(self, verbose)


class _InteractionExport(nn.Module):
    """Deterministic actor export including the encoder and interaction head."""

    is_recurrent: bool = False

    def __init__(self, model: InteractionActor) -> None:
        super().__init__()
        self.frame_dim = model.frame_dim
        self.input_size = model.obs_dim
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.interaction_head = copy.deepcopy(model.interaction_head)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.history_encoder(obs)
        predicted = self.interaction_head(latent)
        inputs = torch.cat((obs[..., -self.frame_dim:], latent, predicted), dim=-1)
        return self.deterministic_output(self.mlp(inputs))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _InteractionOnnxExport(_InteractionExport):
    def __init__(self, model: InteractionActor, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
