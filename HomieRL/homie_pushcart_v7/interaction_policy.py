"""V3.2 five-frame Actor plus detached cart HIM and per-hand wrench estimates.

V7.1 (``robot_velocity_head=True``): a robot (pelvis) velocity estimator fine-tuned in the push-cart scene. It is a copy
of HOMIE's HIM encoder (456 -> 256 -> 256 -> 3 + 32), initialised from the frozen HOMIE checkpoint and supervised on
the true pelvis velocity; its first three outputs (m/s x 2, HOMIE's lin_vel scale) are appended to the Actor input.
HOMIE's own estimator / legs stay frozen. The HOMIE estimate slots of the policy frames are masked by the task.
"""
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
ROBOT_VELOCITY_DIM = 3
CRITIC_DIM = 695
ACTION_DIM = 18
NETWORK_SCHEMA = "v7_v32_policy605_him462_actor652_critic695_v2"
NETWORK_SCHEMA_V71 = "v7_1_v32_policy605_him462_robotvel3_actor655_critic695_v1"
HOMIE_FRAME = 76
HOMIE_HISTORY = 6
ROBOT_VELOCITY_SCALE = 2.0              # HOMIE obs_scales.lin_vel: the head predicts pelvis velocity x 2


def homie_history(estimator_history):
    """[N, 6 x 77] V7 estimator history -> [N, 6 x 76] HOMIE observation history (relative yaw dropped).
    Both histories run oldest -> newest and each 77-frame starts with HOMIE's own 76 observation entries."""
    frames = estimator_history.reshape(*estimator_history.shape[:-1], HOMIE_HISTORY, HOMIE_FRAME + 1)
    return frames[..., :HOMIE_FRAME].flatten(-2)


def homie_encoder_from_checkpoint(checkpoint=None):
    """HOMIE's HIM encoder (Linear 456-256, ELU, 256-256, ELU, 256-35) with the frozen checkpoint's weights."""
    from homie_pushcart.lowlevel import frozen_checkpoint
    path = checkpoint or frozen_checkpoint()
    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    state = loaded["model_state_dict"] if "model_state_dict" in loaded else loaded
    encoder = nn.Sequential(nn.Linear(HOMIE_FRAME * HOMIE_HISTORY, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                            nn.Linear(256, 35))
    prefix = "estimator.encoder."
    encoder.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return encoder


class InteractionActor(MLPModel):
    """Preserve the full five-frame history, normalization and 18-action head.

    Observable ``policy`` is V3.2 minus handle xyz in each of five frames;
    ``estimator`` is six HOMIE+relative-yaw frames. No privileged fields enter
    inference. The force head sees normalized policy history and predicted
    cart speed only, never true velocity or HIM's latent representation.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                 distribution_cfg=None, estimator_hidden_dims=(512, 256, 128), robot_velocity_head=False):
        if list(hidden_dims) != [512, 256, 128] or list(estimator_hidden_dims) != [512, 256, 128]:
            raise ValueError("V7 preserves the approved Actor and force MLP widths [512, 256, 128].")
        if activation != "elu" or not obs_normalization or output_dim != ACTION_DIM:
            raise ValueError("V7 requires ELU, Actor observation normalization, and 18 unchanged actions.")
        self.robot_velocity_head = bool(robot_velocity_head)
        self.actor_input_dim = ACTOR_INPUT_DIM + (ROBOT_VELOCITY_DIM if self.robot_velocity_head else 0)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         True, copy.deepcopy(distribution_cfg))
        if self.robot_velocity_head:
            self.robot_velocity = homie_encoder_from_checkpoint()
            # frozen copy of the pre-trained weights: diagnostics only (how much the fine-tuning gained)
            self.robot_velocity_homie = homie_encoder_from_checkpoint()
            for param in self.robot_velocity_homie.parameters():
                param.requires_grad_(False)
        self.cart_him = CartHIMEstimator()
        self.estimator_normalizer = EmpiricalNormalization(POLICY_DIM)
        self.force_estimator = MLP(POLICY_DIM + 3, 12, estimator_hidden_dims, activation)
        self.register_buffer("hand_scale", torch.tensor(HAND_WRENCH_SCALE))
        self.register_buffer("cart_velocity_scale", torch.tensor(CART_VELOCITY_SCALE))

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_groups[obs_set] != ["policy", "estimator"]:
            raise ValueError("V7 Actor groups must be ['policy', 'estimator'].")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != POLICY_DIM:
            raise ValueError("V7 policy history must have shape [N, 605].")
        if obs["estimator"].shape != (obs["policy"].shape[0], ESTIMATOR_DIM):
            raise ValueError("V7 HIM history must have shape [N, 462].")
        # The MLP normalizer covers the ACTUAL concatenated input, not the two
        # raw observation groups (605 + 462). The history is never compressed.
        return list(obs_groups[obs_set]), self.actor_input_dim

    def _get_latent_dim(self):
        return self.actor_input_dim

    def force_parameters(self):
        return self.force_estimator.parameters()

    def cart_velocity_parameters(self):
        return self.cart_him.parameters()

    def robot_velocity_parameters(self):
        return self.robot_velocity.parameters() if self.robot_velocity_head else iter(())

    def estimator_parameters(self):
        frozen = self.robot_velocity_homie.parameters() if self.robot_velocity_head else iter(())
        return chain(self.force_parameters(), self.cart_velocity_parameters(), self.robot_velocity_parameters(), frozen)

    def predict_robot_velocity(self, obs, pretrained=False):
        """Pelvis velocity (body frame) x ROBOT_VELOCITY_SCALE from the six-frame HOMIE history."""
        net = self.robot_velocity_homie if pretrained else self.robot_velocity
        return net(homie_history(obs["estimator"]))[:, :ROBOT_VELOCITY_DIM]

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
        parts = [obs["policy"], velocity, latent, hands]
        if self.robot_velocity_head:
            parts.append(self.predict_robot_velocity(obs).detach())
        return torch.cat(parts, -1)

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
        out = self._metadata_v7()
        if self.robot_velocity_head:
            out.update({
                "schema": NETWORK_SCHEMA_V71, "release": "7.1",
                "actor_input_fields": out["actor_input_fields"] + [["robot_velocity_x2", ROBOT_VELOCITY_DIM]],
                "actor_mlp": [self.actor_input_dim, 512, 256, 128, ACTION_DIM],
                "actor_normalization": f"EmpiricalNormalization_on_all_{self.actor_input_dim}_mlp_inputs",
                "robot_velocity": {"architecture": [HOMIE_FRAME * HOMIE_HISTORY, 256, 256, 35], "used_outputs": 3,
                                   "input": "estimator history frames [:76] (HOMIE observation history)",
                                   "init": "frozen HOMIE HIM encoder", "target": "pelvis body-frame velocity x 2",
                                   "scale": ROBOT_VELOCITY_SCALE},
            })
        return out

    def _metadata_v7(self):
        return {
            "schema": NETWORK_SCHEMA, "version": 7,
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
        self.robot_velocity_head = bool(actor.robot_velocity_head)
        self.robot_velocity = (copy.deepcopy(actor.robot_velocity) if self.robot_velocity_head
                               else nn.Linear(1, 1))          # placeholder keeps TorchScript attribute types static

    def _features(self, policy: torch.Tensor, estimator: torch.Tensor):
        if policy.ndim != 2 or policy.shape[1] != self.input_size:
            raise ValueError("V7 policy history must have shape [N, 605].")
        if estimator.ndim != 2 or estimator.shape[1] != self.estimator_input_size:
            raise ValueError("V7 HIM history must have shape [N, 462].")
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
        if self.robot_velocity_head:
            frames = estimator.reshape(estimator.shape[0], HOMIE_HISTORY, HOMIE_FRAME + 1)[:, :, :HOMIE_FRAME]
            robot = self.robot_velocity(frames.reshape(estimator.shape[0], HOMIE_FRAME * HOMIE_HISTORY))[:, :3]
            inputs = self.obs_normalizer(torch.cat((policy, speed, latent, hands, robot), -1))
        else:
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
