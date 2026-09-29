"""HOMIE HIM estimator with cart velocity supervision and the V3.2 relative yaw.

The observable frame is the unchanged 76-column HOMIE frame followed by the
V3.2 measured relative-yaw scalar. Six oldest-to-newest frames are encoded.
This module does not change the high-level Actor, Critic, or their histories.

Training pairs follow HOMIE's temporal convention: history at t predicts cart
velocity at t+1. ``next_critic_obs`` here is a dedicated 80-column estimator
target, NOT the high-level Critic input: [next observable frame (77), next
scaled cart velocity (3)]. Terminal transitions must use the pre-reset target.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


FRAME_DIM = 77
HISTORY_LENGTH = 6
HISTORY_DIM = FRAME_DIM * HISTORY_LENGTH
LATENT_DIM = 32
NEXT_CRITIC_DIM = FRAME_DIM + 3
CART_VELOCITY_SCALE = (2.0, 2.0, 0.5)


def make_next_critic_observation(next_frame: torch.Tensor,
                                 next_cart_velocity: torch.Tensor) -> torch.Tensor:
    """Pack t+1 observables and physical [vx, vy, wz] in upright cart axes.

The velocity reference point is the original V3.2 cart COM. This helper only
scales supplied values; it must never be used to build an inference input.
"""
    if next_frame.ndim != 2 or next_frame.shape[1] != FRAME_DIM:
        raise ValueError("The next cart HIM frame must have shape [N, 77].")
    if next_cart_velocity.shape != (next_frame.shape[0], 3):
        raise ValueError("The next physical cart velocity must have shape [N, 3].")
    scale = next_cart_velocity.new_tensor(CART_VELOCITY_SCALE)
    return torch.cat((next_frame, next_cart_velocity * scale), dim=-1).detach()


def mirror_cart_frame(frame: torch.Tensor) -> torch.Tensor:
    """HOMIE sagittal reflection plus yaw -> -yaw; opt-in data helper only.

    The 76-column prefix uses HOMIE's existing joint ordering and signs. This
    function does not enable PPO symmetry augmentation or alter any policy.
    """
    if frame.shape[-1] != FRAME_DIM:
        raise ValueError("A mirrored cart HIM frame must have 77 columns.")
    # HOMIE order: left leg 6, right leg 6, waist 1, left arm 7, right arm 7.
    joint_order = (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12,
                   20, 21, 22, 23, 24, 25, 26, 13, 14, 15, 16, 17, 18, 19)
    joint_sign = (1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1,
                  1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1)
    result = frame.clone()
    result[..., :10] = frame[..., :10] * frame.new_tensor((1, -1, -1, 1, -1, 1, -1, 1, -1, 1))
    for start in (10, 37):
        result[..., start:start + 27] = frame[..., start:start + 27][..., list(joint_order)] * frame.new_tensor(joint_sign)
    result[..., 64:76] = frame[..., 64:76][..., list(joint_order[:12])] * frame.new_tensor(joint_sign[:12])
    result[..., 76] = -frame[..., 76]
    return result


def mirror_cart_velocity(velocity: torch.Tensor) -> torch.Tensor:
    """Reflect cart [vx, vy, wz], in either physical or fixed-scaled units."""
    if velocity.shape[-1] != 3:
        raise ValueError("Cart velocity must have three columns [vx, vy, wz].")
    return velocity * velocity.new_tensor((1, -1, -1))


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """The unchanged balanced assignment calculation from HOMIE HIM."""
    q = torch.exp(out / eps).T
    k, batch = q.shape[0], q.shape[1]
    q /= q.sum()
    for _ in range(iters):
        q /= torch.sum(q, dim=1, keepdim=True)
        q /= k
        q /= torch.sum(q, dim=0, keepdim=True)
        q /= batch
    return (q * batch).T


class CartHIMEstimator(nn.Module):
    """Fixed HOMIE architecture; only the added yaw and supervision differ.

    ``forward`` detaches predictions as in HIM. ``encode`` keeps gradients to
    estimator parameters but never to observation inputs. The dedicated Adam
    optimizer is not part of ``state_dict``: checkpoint its state separately.
    Pass PPO's current adaptive learning rate to every ``update`` call.
    """

    def __init__(self, learning_rate: float = 1.0e-3):
        super().__init__()
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Cart HIM learning_rate must be finite and positive.")
        self.temporal_steps = HISTORY_LENGTH
        self.num_one_step_obs = FRAME_DIM
        self.num_height_points = 0
        self.num_latent = LATENT_DIM
        self.input_dim = HISTORY_DIM
        self.max_grad_norm = 10.0
        self.temperature = 3.0
        self.encoder = nn.Sequential(
            nn.Linear(HISTORY_DIM, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 3 + LATENT_DIM))
        self.target = nn.Sequential(
            nn.Linear(FRAME_DIM, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(), nn.Linear(256, LATENT_DIM))
        self.proto = nn.Embedding(64, LATENT_DIM)
        self.learning_rate = float(learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_history.ndim != 2 or obs_history.shape[1] != self.input_dim:
            raise ValueError("Cart HIM history must have shape [N, 462].")
        parts = self.encoder(obs_history.detach())
        return parts[..., :3], F.normalize(parts[..., 3:], dim=-1, p=2.0)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        velocity, latent = self.encode(obs_history)
        return velocity.detach(), latent.detach()

    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs_history)

    def compute_response(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs_history)

    def compute_feedback(self, obs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target(obs.detach()), dim=-1, p=2.0).detach()

    @torch.jit.ignore
    def metadata(self) -> dict:
        return {
            "schema": "v6_cart_him_77x6_v1",
            "reference": "HomieRL/rsl_rl/rsl_rl/modules/him_estimator.py",
            "frame_fields": [["homie_original_frame", 76], ["v32_relative_yaw", 1]],
            "frame_dim": FRAME_DIM, "history_length": HISTORY_LENGTH,
            "history_order": "oldest_to_newest", "history_dim": HISTORY_DIM,
            "encoder_dims": [HISTORY_DIM, 256, 256, 35],
            "target_dims": [FRAME_DIM, 256, 256, LATENT_DIM],
            "activation": "elu", "latent_normalization": "l2", "num_prototypes": 64,
            "temperature": self.temperature, "sinkhorn_epsilon": 0.05, "sinkhorn_iterations": 3,
            "optimizer": "Adam", "initial_learning_rate": 1.0e-3,
            "learning_rate_schedule": "caller_passes_current_ppo_adaptive_lr_each_minibatch",
            "max_grad_norm": self.max_grad_norm, "loss": "velocity_mse_plus_swap",
            "velocity_fields": ["vx", "vy", "wz"], "velocity_scale": list(CART_VELOCITY_SCALE),
            "velocity_axes": "upright_cart_heading", "velocity_reference_point": "v32_cart_com",
            "supervision_timing": "history_t_to_pre_reset_next_state_t_plus_1",
            "next_target_layout": "next_frame77_then_next_scaled_cart_velocity3",
            "target_encoder_input": "next_frame_without_command3_then_next_scaled_cart_velocity3",
            "invalid_batch_behavior": "skip_optimizer_and_prototype_update",
            "policy_gradient_to_estimator": False, "symmetry_augmentation_enabled": False,
        }

    @torch.jit.ignore
    def _select_valid(self, obs_history: torch.Tensor, next_critic_obs: torch.Tensor,
                      valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if obs_history.ndim != 2 or obs_history.shape[1] != HISTORY_DIM:
            raise ValueError("Cart HIM history must have shape [N, 462].")
        if next_critic_obs.shape != (obs_history.shape[0], NEXT_CRITIC_DIM):
            raise ValueError("Cart HIM next target must have shape [N, 80].")
        if valid is not None:
            if valid.shape == (obs_history.shape[0], 1):
                valid = valid[:, 0]
            if valid.shape != (obs_history.shape[0],):
                raise ValueError("Cart HIM valid mask must have shape [N] or [N, 1].")
            if valid.dtype != torch.bool and not ((valid == 0) | (valid == 1)).all():
                raise ValueError("Cart HIM valid mask must contain only boolean or 0/1 values.")
            valid = valid.to(device=obs_history.device, dtype=torch.bool)
            obs_history, next_critic_obs = obs_history[valid], next_critic_obs[valid]
        obs_history, next_critic_obs = obs_history.detach(), next_critic_obs.detach()
        if not (torch.isfinite(obs_history).all() and torch.isfinite(next_critic_obs).all()):
            raise ValueError("Valid cart HIM samples must be finite; mask invalid transitions explicitly.")
        return obs_history, next_critic_obs

    @torch.jit.ignore
    def _losses(self, obs_history: torch.Tensor,
                next_critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Match HOMIE's slices exactly after increasing num_one_step_obs to 77.
        velocity = next_critic_obs[:, FRAME_DIM:FRAME_DIM + 3].detach()
        next_obs = next_critic_obs.detach()[:, 3:FRAME_DIM + 3]
        parts = self.encoder(obs_history)
        z_t = self.target(next_obs)
        prediction, z_s = parts[..., :3], parts[..., 3:]
        z_s, z_t = F.normalize(z_s, dim=-1, p=2), F.normalize(z_t, dim=-1, p=2)
        with torch.no_grad():
            weight = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
            self.proto.weight.copy_(weight)
        score_s, score_t = z_s @ self.proto.weight.T, z_t @ self.proto.weight.T
        with torch.no_grad():
            q_s, q_t = sinkhorn(score_s), sinkhorn(score_t)
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        return F.mse_loss(prediction, velocity), swap_loss

    @torch.jit.ignore
    def losses(self, obs_history: torch.Tensor, next_critic_obs: torch.Tensor,
               valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable HIM losses; invalid rows are excluded before encoding.

        Like the original training calculation, valid calls L2-normalize the
        prototypes in-place. An empty mask leaves all parameters untouched.
        """
        history, target = self._select_valid(obs_history, next_critic_obs, valid)
        if history.shape[0] == 0:
            zero = self.encoder[0].weight.sum() * 0.0
            return zero, zero
        return self._losses(history, target)

    @torch.jit.ignore
    def update(self, obs_history: torch.Tensor, next_critic_obs: torch.Tensor,
               lr: float | None = None, valid: torch.Tensor | None = None) -> tuple[float, float]:
        """Perform one HIM update; a wholly invalid batch never advances Adam."""
        if lr is not None and (not math.isfinite(lr) or lr <= 0):
            raise ValueError("Cart HIM adaptive learning rate must be finite and positive.")
        history, target = self._select_valid(obs_history, next_critic_obs, valid)
        if history.shape[0] == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0, 0.0
        if lr is not None:
            self.learning_rate = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        estimation_loss, swap_loss = self._losses(history, target)
        self.optimizer.zero_grad()
        (estimation_loss + swap_loss).backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return estimation_loss.item(), swap_loss.item()
