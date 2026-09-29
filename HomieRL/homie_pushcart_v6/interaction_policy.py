"""V3.2 five-frame Actor plus detached cart HIM and per-hand wrench estimates."""
from __future__ import annotations

import copy
from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP

from .him_estimator import CartHIMEstimator, CART_VELOCITY_SCALE
from .observation import HAND_WRENCH_SCALE

POLICY_DIM = 605
ESTIMATOR_DIM = 462
ACTOR_INPUT_DIM = 652
CRITIC_DIM = 665
ACTION_DIM = 18
NETWORK_SCHEMA = "v6_v32_policy605_him462_actor652_critic665_v1"


class InteractionActor(MLPModel):
    """Preserve the full five-frame history, normalization and 18-action head.

    Observable ``policy`` is V3.2 minus handle xyz in each of five frames;
    ``estimator`` is six HOMIE+relative-yaw frames. No privileged fields enter
    inference. The force head sees normalized policy history and predicted
    cart speed only, never true velocity or HIM's latent representation.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                 distribution_cfg=None, estimator_hidden_dims=(512, 256, 128)):
        if list(hidden_dims) != [512, 256, 128] or list(estimator_hidden_dims) != [512, 256, 128]:
            raise ValueError("V6 preserves the approved Actor and force MLP widths [512, 256, 128].")
        if activation != "elu" or not obs_normalization or output_dim != ACTION_DIM:
            raise ValueError("V6 requires ELU, Actor observation normalization, and 18 unchanged actions.")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         True, copy.deepcopy(distribution_cfg))
        self.cart_him = CartHIMEstimator()
        self.estimator_normalizer = EmpiricalNormalization(POLICY_DIM)
        self.force_estimator = MLP(POLICY_DIM + 3, 12, estimator_hidden_dims, activation)
        self.register_buffer("hand_scale", torch.tensor(HAND_WRENCH_SCALE))
        self.register_buffer("cart_velocity_scale", torch.tensor(CART_VELOCITY_SCALE))

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_groups[obs_set] != ["policy", "estimator"]:
            raise ValueError("V6 Actor groups must be ['policy', 'estimator'].")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != POLICY_DIM:
            raise ValueError("V6 policy history must have shape [N, 605].")
        if obs["estimator"].shape != (obs["policy"].shape[0], ESTIMATOR_DIM):
            raise ValueError("V6 HIM history must have shape [N, 462].")
        # The MLP normalizer covers the ACTUAL concatenated input, not the two
        # raw observation groups (605 + 462). The history is never compressed.
        return list(obs_groups[obs_set]), ACTOR_INPUT_DIM

    def _get_latent_dim(self):
        return ACTOR_INPUT_DIM

    def force_parameters(self):
        return self.force_estimator.parameters()

    def cart_velocity_parameters(self):
        return self.cart_him.parameters()

    def estimator_parameters(self):
        return chain(self.force_parameters(), self.cart_velocity_parameters())

    def predict_cart_velocity(self, obs):
        return self.cart_him(obs["estimator"])[0]

    def predict_hand_wrench_history(self, policy, estimator_history, cart_velocity=None):
        if cart_velocity is None:
            cart_velocity = self.cart_him(estimator_history)[0]
        force_input = torch.cat((self.estimator_normalizer(policy.detach()), cart_velocity.detach()), -1)
        return self.force_estimator(force_input)

    def predict_hand_wrench(self, obs):
        return self.predict_hand_wrench_history(obs["policy"], obs["estimator"])

    def unnormalized_actor_input(self, obs):
        velocity, latent = self.cart_him(obs["estimator"])
        hands = self.predict_hand_wrench_history(obs["policy"], obs["estimator"], velocity).detach()
        return torch.cat((obs["policy"], velocity, latent, hands), -1)

    def get_latent(self, obs, masks=None, hidden_state=None):
        return self.obs_normalizer(self.unnormalized_actor_input(obs))

    @torch.no_grad()
    def update_normalization(self, obs):
        self.estimator_normalizer.update(obs["policy"])
        self.obs_normalizer.update(self.unnormalized_actor_input(obs))

    @torch.no_grad()
    def physical_estimates(self, obs):
        speed = self.predict_cart_velocity(obs)
        hands = self.predict_hand_wrench_history(obs["policy"], obs["estimator"], speed)
        return speed / self.cart_velocity_scale, hands * self.hand_scale

    def metadata(self):
        return {
            "schema": NETWORK_SCHEMA, "version": 6,
            "observable_groups": {"policy": POLICY_DIM, "estimator": ESTIMATOR_DIM},
            "actor_input_fields": [["v32_history_without_handle_xyz", 605],
                                   ["cart_velocity", 3], ["him_latent", 32], ["per_hand_wrench", 12]],
            "actor_mlp": [ACTOR_INPUT_DIM, 512, 256, 128, ACTION_DIM],
            "actor_normalization": "EmpiricalNormalization_on_all_652_mlp_inputs",
            "critic_input": CRITIC_DIM, "critic_legacy_prefix": 652,
            "force_mlp": [608, 512, 256, 128, 12],
            "force_input": "independently_normalized_policy605_plus_predicted_cart_velocity3",
            "force_scale": list(HAND_WRENCH_SCALE), "cart_him": self.cart_him.metadata(),
        }

    def as_jit(self):
        return ObservableActorExport(self)

    def as_onnx(self, verbose=False):
        return ObservableActorExport(self)


class ObservableActorExport(nn.Module):
    """Inference contains both normalizers; no Critic or supervision input."""
    is_recurrent = False

    def __init__(self, actor):
        super().__init__()
        self.input_size = POLICY_DIM
        self.estimator_input_size = ESTIMATOR_DIM
        self.cart_encoder = copy.deepcopy(actor.cart_him.encoder)
        for name in ("obs_normalizer", "estimator_normalizer", "force_estimator", "mlp"):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic_output = (actor.distribution.as_deterministic_output_module()
                                     if actor.distribution is not None else nn.Identity())
        self.register_buffer("hand_scale", actor.hand_scale.clone())
        self.register_buffer("cart_velocity_scale", actor.cart_velocity_scale.clone())

    def _features(self, policy: torch.Tensor, estimator: torch.Tensor):
        if policy.ndim != 2 or policy.shape[1] != self.input_size:
            raise ValueError("V6 policy history must have shape [N, 605].")
        if estimator.ndim != 2 or estimator.shape[1] != self.estimator_input_size:
            raise ValueError("V6 HIM history must have shape [N, 462].")
        parts = self.cart_encoder(estimator)
        speed, latent = parts[:, :3], F.normalize(parts[:, 3:], dim=-1, p=2.0)
        hands = self.force_estimator(torch.cat((self.estimator_normalizer(policy), speed), -1))
        return speed, latent, hands

    @torch.jit.export
    def estimates(self, policy: torch.Tensor, estimator: torch.Tensor):
        speed, _, hands = self._features(policy, estimator)
        return speed, hands

    def forward(self, policy: torch.Tensor, estimator: torch.Tensor):
        speed, latent, hands = self._features(policy, estimator)
        inputs = self.obs_normalizer(torch.cat((policy, speed, latent, hands), -1))
        return self.deterministic_output(self.mlp(inputs))

    @torch.jit.export
    def physical_estimates(self, policy: torch.Tensor, estimator: torch.Tensor):
        speed, hands = self.estimates(policy, estimator)
        return speed / self.cart_velocity_scale, hands * self.hand_scale

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return torch.zeros(1, self.input_size), torch.zeros(1, self.estimator_input_size)

    @property
    def input_names(self):
        return ["policy_history", "him_history"]

    @property
    def output_names(self):
        return ["actions"]
