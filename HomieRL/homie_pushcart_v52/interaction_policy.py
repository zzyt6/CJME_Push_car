"""V5.2 high-level actor with an independently trained per-hand interaction estimator.

The estimator sees only the last ``estimator_history`` fixed-scaled policy frames
(proprioception, commands, FK and the static (J^T)^+ tau prior), through its own
running input normalization. It predicts each hand's wrench on the cart,
[left F, left M, right F, right M] at the palm origins in cart-heading axes,
divided by ``HAND_WRENCH_SCALE``. It is optimized only by the supervised loss in
``InteractionPPO`` with its own optimizer; the policy consumes a *detached*
prediction, so PPO gradients never reach it. The handle resultant [Fx, Fy, Mz]
is composed from the two hands with the constant fixed-grasp palm offsets.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization

from .observation import HAND_WRENCH_SCALE, WRENCH_SCALE


def compose_handle_wrench(hands: torch.Tensor, palm_offsets: torch.Tensor) -> torch.Tensor:
    """Per-hand [N,12] physical wrench -> handle-centre [Fx, Fy, Mz] [N,3] (cart axes).

    ``palm_offsets`` [2,3] are palm origins minus the handle centre. Mz adds the
    two palm moments and the r x F transport of both hand forces.
    """
    left_f, left_m, right_f, right_m = hands[..., 0:3], hands[..., 3:6], hands[..., 6:9], hands[..., 9:12]
    rl, rr = palm_offsets[0], palm_offsets[1]
    mz = (left_m[..., 2] + right_m[..., 2]
          + rl[0] * left_f[..., 1] - rl[1] * left_f[..., 0]
          + rr[0] * right_f[..., 1] - rr[1] * right_f[..., 0])
    return torch.stack((left_f[..., 0] + right_f[..., 0], left_f[..., 1] + right_f[..., 1], mz), -1)


class InteractionActor(MLPModel):
    """History -> 16D latent; last frames -> per-hand wrench; current-frame action policy.

    Policy input: current frame + latent + detached per-hand wrench (12) + composed
    resultant (3). History is chronological, current frame last. With
    ``force_to_policy=False`` the 15 load slots are zero (equal-capacity ablation);
    the estimator is still trained so its accuracy remains measurable.
    """

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims=(256, 256, 256), activation: str = "elu",
        obs_normalization: bool = False, distribution_cfg: dict | None = None,
        frame_dim: int = 136, history_length: int = 20, latent_dim: int = 16,
        encoder_hidden_dims=(256, 128), estimator_history: int = 5,
        estimator_hidden_dims=(512, 256, 128), force_to_policy: bool = True,
        palm_offsets=None,
    ) -> None:
        if obs_groups[obs_set] != ["policy"]:
            raise ValueError("InteractionActor requires exactly the ['policy'] observation group.")
        if obs_normalization:
            raise ValueError("InteractionActor uses fixed feature scaling; obs_normalization must be False.")
        if min(frame_dim, history_length, latent_dim, estimator_history) <= 0 or estimator_history > history_length:
            raise ValueError("Frame, history, latent and estimator history must be positive (estimator <= history).")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != frame_dim * history_length:
            raise ValueError(f"Expected policy history width {frame_dim * history_length}.")
        self.frame_dim = frame_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.estimator_history = estimator_history
        self.force_to_policy = bool(force_to_policy)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, False, copy.deepcopy(distribution_cfg))
        self.history_encoder = MLP(frame_dim * history_length, latent_dim, encoder_hidden_dims, activation)
        width = frame_dim * estimator_history
        self.estimator_normalizer = EmpiricalNormalization(width)
        self.force_estimator = MLP(width, 12, estimator_hidden_dims, activation)
        if palm_offsets is None:
            from .fixed_grasp import palm_offsets_from_handle
            palm_offsets = palm_offsets_from_handle()
        offsets = torch.as_tensor(palm_offsets, dtype=torch.float32)
        if offsets.shape != (2, 3) or not torch.isfinite(offsets).all():
            raise ValueError("palm_offsets must be finite [2,3] (left, right) palm minus handle centre, cart axes.")
        self.register_buffer("palm_offsets", offsets)
        self.register_buffer("hand_scale", torch.tensor(HAND_WRENCH_SCALE, dtype=torch.float32).repeat(2))
        self.register_buffer("wrench_scale", torch.tensor(WRENCH_SCALE, dtype=torch.float32))

    def _get_latent_dim(self) -> int:
        return self.frame_dim + self.latent_dim + 12 + 3

    def estimator_parameters(self):
        return self.force_estimator.parameters()

    def predict_hand_wrench(self, obs: TensorDict) -> torch.Tensor:
        """Normalized per-hand wrench [N,12]; reads only the last estimator frames."""
        key = getattr(self, 'estimator_observation_key', 'policy')
        if key == 'falcon_fullbody':
            from .falcon_history import fullbody_history
            recent = fullbody_history(obs['policy'],obs['falcon'])
        else:
            recent = (obs["policy"][..., -self.frame_dim * self.estimator_history:]
                      if key == 'policy' else obs[key])
        return self.force_estimator(self.estimator_normalizer(recent))

    def compose_resultant(self, hands_normalized: torch.Tensor) -> torch.Tensor:
        """Normalized per-hand prediction -> normalized handle resultant [Fx/80, Fy/80, Mz/12]."""
        return compose_handle_wrench(hands_normalized * self.hand_scale, self.palm_offsets) / self.wrench_scale

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        history = obs["policy"]
        latent = self.history_encoder(history)
        # Detached: the estimator is trained only by its own supervised objective.
        hands = self.predict_hand_wrench(obs).detach()
        if not self.force_to_policy:
            hands = torch.zeros_like(hands)
        return torch.cat((history[..., -self.frame_dim:], latent, hands, self.compose_resultant(hands)), dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        # Running statistics of the estimator input only (no-op in eval mode).
        self.estimator_normalizer.update(obs["policy"][..., -self.frame_dim * self.estimator_history:])

    def as_jit(self) -> nn.Module:
        if getattr(self, 'estimator_observation_key', 'policy') != 'policy':
            raise ValueError('Attached FALCON head needs a separate history input; export the standalone estimator')
        return _InteractionExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        if getattr(self, 'estimator_observation_key', 'policy') != 'policy':
            raise ValueError('Attached FALCON head needs a separate history input; export the standalone estimator')
        return _InteractionOnnxExport(self, verbose)


class _InteractionExport(nn.Module):
    """Deterministic actor export including the history encoder and force estimator."""

    is_recurrent: bool = False

    def __init__(self, model: InteractionActor) -> None:
        super().__init__()
        self.frame_dim = model.frame_dim
        self.estimator_width = model.frame_dim * model.estimator_history
        self.force_to_policy = model.force_to_policy
        self.input_size = model.obs_dim
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.estimator_normalizer = copy.deepcopy(model.estimator_normalizer)
        self.force_estimator = copy.deepcopy(model.force_estimator)
        self.register_buffer("palm_offsets", model.palm_offsets.clone())
        self.register_buffer("hand_scale", model.hand_scale.clone())
        self.register_buffer("wrench_scale", model.wrench_scale.clone())
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.history_encoder(obs)
        hands = self.force_estimator(self.estimator_normalizer(obs[..., -self.estimator_width:]))
        if not self.force_to_policy:
            hands = torch.zeros_like(hands)
        resultant = compose_handle_wrench(hands * self.hand_scale, self.palm_offsets) / self.wrench_scale
        inputs = torch.cat((obs[..., -self.frame_dim:], latent, hands, resultant), dim=-1)
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
