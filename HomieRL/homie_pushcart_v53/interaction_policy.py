"""V5.3: observable history -> cart velocity -> per-hand wrench -> actor.

Both supervised heads are detached from PPO; the force head receives predicted
velocity during training AND deployment. No velocity teacher forcing or grasp FK.
"""
from __future__ import annotations
import copy
from itertools import chain

import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization

from .observation import FRAME_DIM, CART_INPUT_FRAME_DIM, CART_VELOCITY_SCALE, HAND_WRENCH_SCALE


class InteractionActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128),
                 activation='elu', obs_normalization=False, distribution_cfg=None,
                 frame_dim=FRAME_DIM, history_length=20, latent_dim=16,
                 encoder_hidden_dims=(256, 128), estimator_history=5,
                 estimator_hidden_dims=(512, 256, 128), cart_velocity_hidden_dims=(128, 64),
                 force_to_policy=True):
        if obs_groups[obs_set] != ['policy'] or obs_normalization:
            raise ValueError('V5.3 actor only consumes fixed-scaled policy history')
        if frame_dim != FRAME_DIM or not 0 < estimator_history <= history_length:
            raise ValueError('V5.3 schema/history mismatch; V5.2 weights are incompatible')
        if obs['policy'].shape[-1] != frame_dim*history_length:
            raise ValueError('V5.3 policy history has incorrect width')
        self.frame_dim, self.history_length, self.latent_dim = frame_dim, history_length, latent_dim
        self.estimator_history, self.force_to_policy = estimator_history, force_to_policy
        self.cart_input_width = CART_INPUT_FRAME_DIM*estimator_history
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         False, copy.deepcopy(distribution_cfg))
        self.history_encoder = MLP(frame_dim*history_length, latent_dim, encoder_hidden_dims, activation)
        self.cart_velocity_normalizer = EmpiricalNormalization(self.cart_input_width)
        self.cart_velocity_estimator = MLP(self.cart_input_width, 3, cart_velocity_hidden_dims, activation)
        self.estimator_normalizer = EmpiricalNormalization(frame_dim*estimator_history)
        self.force_estimator = MLP(frame_dim*estimator_history+3, 12, estimator_hidden_dims, activation)
        self.register_buffer('hand_scale', torch.tensor(HAND_WRENCH_SCALE).repeat(2))
        self.register_buffer('cart_velocity_scale', torch.tensor(CART_VELOCITY_SCALE))

    def _get_latent_dim(self):
        return self.frame_dim+self.latent_dim+3+12

    def force_parameters(self):
        return self.force_estimator.parameters()

    def cart_velocity_parameters(self):
        return self.cart_velocity_estimator.parameters()

    def estimator_parameters(self):
        return chain(self.force_parameters(), self.cart_velocity_parameters())

    def recent_history(self, history):
        return history[..., -self.frame_dim*self.estimator_history:]

    def cart_velocity_input(self, history):
        recent = self.recent_history(history).reshape(-1, self.estimator_history, self.frame_dim)
        return recent[..., :CART_INPUT_FRAME_DIM].flatten(1)

    def predict_cart_velocity_history(self, history):
        return self.cart_velocity_estimator(self.cart_velocity_normalizer(self.cart_velocity_input(history)))

    def predict_cart_velocity(self, obs):
        """Normalized [vx,vy,world-vertical angular velocity] at the cart base-link origin."""
        return self.predict_cart_velocity_history(obs['policy'])

    def predict_hand_wrench_history(self, history, cart_velocity=None):
        if cart_velocity is None:
            cart_velocity = self.predict_cart_velocity_history(history)
        inputs = torch.cat((self.estimator_normalizer(self.recent_history(history)), cart_velocity.detach()), -1)
        return self.force_estimator(inputs)

    def predict_hand_wrench(self, obs):
        return self.predict_hand_wrench_history(obs['policy'])

    def get_latent(self, obs, masks=None, hidden_state=None):
        history = obs['policy']
        velocity = self.predict_cart_velocity_history(history).detach()
        hands = self.predict_hand_wrench_history(history, velocity).detach()
        if not self.force_to_policy:
            hands = torch.zeros_like(hands)
        # No handle-centre resultant requiring fixed palm-to-cart calibration.
        return torch.cat((history[..., -self.frame_dim:], self.history_encoder(history), velocity, hands), -1)

    def update_normalization(self, obs):
        self.estimator_normalizer.update(self.recent_history(obs['policy']))
        self.cart_velocity_normalizer.update(self.cart_velocity_input(obs['policy']))

    def as_jit(self):
        return ObservableActorExport(self)

    def as_onnx(self, verbose=False):
        return ObservableActorExport(self)


class ObservableActorExport(nn.Module):
    """One deployment input; outputs action, physical cart velocity and both hand wrenches."""
    is_recurrent = False

    def __init__(self, actor):
        super().__init__()
        self.input_size = actor.frame_dim*actor.history_length
        self.frame_dim, self.estimator_history = actor.frame_dim, actor.estimator_history
        self.cart_frame_dim = CART_INPUT_FRAME_DIM
        self.force_to_policy = actor.force_to_policy
        for name in ('history_encoder', 'cart_velocity_normalizer', 'cart_velocity_estimator',
                     'estimator_normalizer', 'force_estimator', 'mlp'):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic_output = (actor.distribution.as_deterministic_output_module()
                                     if actor.distribution is not None else nn.Identity())
        self.register_buffer('hand_scale', actor.hand_scale.clone())
        self.register_buffer('cart_velocity_scale', actor.cart_velocity_scale.clone())

    @torch.jit.export
    def estimates(self, history):
        recent = history[..., -self.frame_dim*self.estimator_history:]
        speed_x = recent.reshape(-1, self.estimator_history, self.frame_dim)[..., :self.cart_frame_dim].flatten(1)
        speed = self.cart_velocity_estimator(self.cart_velocity_normalizer(speed_x))
        hands = self.force_estimator(torch.cat((self.estimator_normalizer(recent), speed), -1))
        return speed, hands

    def forward(self, history):
        speed, hands = self.estimates(history)
        load = hands if self.force_to_policy else torch.zeros_like(hands)
        x = torch.cat((history[..., -self.frame_dim:], self.history_encoder(history), speed, load), -1)
        return self.deterministic_output(self.mlp(x))

    @torch.jit.export
    def physical_estimates(self, history):
        speed, hands = self.estimates(history)
        return speed / self.cart_velocity_scale, hands * self.hand_scale

    @torch.jit.export
    def reset(self):
        pass

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self):
        return ['observable_history']

    @property
    def output_names(self):
        return ['actions']
