# PPO update structure adapted from RSL-RL (ETH Zurich / NVIDIA, BSD-3-Clause).
"""RSL-RL PPO with separate force and cart-velocity supervised optimizers.

Both estimator parameter sets are excluded from PPO. Force uses masked Huber;
cart velocity uses masked MSE against privileged simulator velocity. The force
head always consumes predicted (detached) velocity, including during training.
All three optimizers and both normalizers are included in checkpoint state.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.algorithms import PPO

from .interaction_policy import InteractionActor

HAND_TARGET_DIM = 12


def masked_interaction_huber(
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Huber loss and valid fraction for normalized targets [..., D + valid].

    Mask before arithmetic: NaNs in invalid labels must never reach the loss.
    All-invalid batches produce a differentiable zero with zero gradients.
    """
    dim = predictions.shape[-1]
    if targets.shape[:-1] != predictions.shape[:-1] or targets.shape[-1] != dim + 1:
        raise ValueError(f"Expected predictions [...,{dim}] and interaction targets [...,{dim + 1}].")
    valid = (targets[..., dim] > 0.5) & torch.isfinite(targets).all(dim=-1)
    safe_targets = torch.where(valid[..., None], targets[..., :dim], predictions.detach())
    component_loss = F.huber_loss(predictions, safe_targets, reduction="none", delta=delta)
    row_loss = component_loss.mean(dim=-1)
    loss = torch.where(valid, row_loss, torch.zeros_like(row_loss)).sum() / valid.sum().clamp_min(1)
    return loss, valid.float().mean()


class InteractionPPO(PPO):
    """Feed-forward PPO; force and cart speed each have their own optimizer."""

    def __init__(self, actor, critic, storage, *, estimator_learning_rate: float = 1.0e-3,
                 cart_velocity_learning_rate: float = 1.0e-3,
                 estimator_epochs: int = 5, estimator_mini_batches: int = 8,
                 estimator_max_grad_norm: float = 1.0, huber_delta: float = 1.0, **kwargs) -> None:
        if not isinstance(actor, InteractionActor):
            raise TypeError("InteractionPPO requires an InteractionActor.")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("V5.3 interaction PPO supports feed-forward models only.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("V5.3 interaction PPO does not support RND or symmetry augmentation.")
        for name, value in (("estimator_learning_rate", estimator_learning_rate), ("huber_delta", huber_delta),
                            ("cart_velocity_learning_rate", cart_velocity_learning_rate),
                            ("estimator_max_grad_norm", estimator_max_grad_norm)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if estimator_epochs < 0 or estimator_mini_batches < 1:
            raise ValueError("estimator_epochs must be >= 0 and estimator_mini_batches >= 1.")
        self.estimator_learning_rate = estimator_learning_rate
        self.estimator_epochs = int(estimator_epochs)
        self.estimator_mini_batches = int(estimator_mini_batches)
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.huber_delta = huber_delta
        super().__init__(actor, critic, storage, **kwargs)
        # Rebuild the PPO optimizer without the estimator; give the estimator its own.
        estimator_ids = {id(p) for p in actor.estimator_parameters()}
        self.policy_parameters = [p for p in actor.parameters() if id(p) not in estimator_ids]
        self.optimizer = type(self.optimizer)(self.policy_parameters + list(critic.parameters()), lr=self.learning_rate)
        self.estimator_optimizer = torch.optim.Adam(actor.force_parameters(), lr=estimator_learning_rate)
        self.cart_velocity_optimizer = torch.optim.Adam(actor.cart_velocity_parameters(), lr=cart_velocity_learning_rate)

    def save(self) -> dict:
        saved = super().save()
        saved["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        saved["cart_velocity_optimizer_state_dict"] = self.cart_velocity_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            # Adaptive KL scheduling uses this scalar in addition to optimizer
            # groups; resume from the saved rate instead of the config default.
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            if "cart_velocity_optimizer_state_dict" in loaded_dict:
                self.cart_velocity_optimizer.load_state_dict(loaded_dict["cart_velocity_optimizer_state_dict"])
        return restore_iteration

    def update(self) -> dict[str, float]:
        totals = dict(value=0.0, surrogate=0.0, entropy=0.0)
        num_updates = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = ((batch.advantages - batch.advantages.mean())
                                        / (batch.advantages.std(unbiased=False) + 1e-8))
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(
                        batch.old_distribution_params, self.actor.output_distribution_params).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantages = batch.advantages.squeeze(-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(),
                                           (value_clipped - batch.returns).square()).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy_parameters, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            num_updates += 1
        if not num_updates:
            raise RuntimeError("Interaction PPO received no minibatches.")
        losses = {key: value / num_updates for key, value in totals.items()}
        losses.update(self.update_estimator())
        self.storage.clear()
        return losses

    def update_estimator(self) -> dict[str, float]:
        """Independent supervised heads; true cart speed is ONLY a target."""
        if self.estimator_epochs == 0:
            return {}
        actor = self.actor
        observations = self.storage.observations.flatten(0, 1)
        inputs = observations["policy"]
        hands_target = observations["interaction_target"]
        speed_target = observations["cart_velocity_target"]
        out = self.estimator_metrics(inputs, hands_target, speed_target)
        sums, count = [0., 0.], 0
        for _ in range(self.estimator_epochs):
            order = torch.randperm(len(inputs), device=inputs.device)
            for idx in order.tensor_split(min(self.estimator_mini_batches, len(inputs))):
                speed = actor.predict_cart_velocity_history(inputs[idx])
                valid = (speed_target[idx, 3] > .5) & torch.isfinite(speed_target[idx]).all(-1)
                target = torch.where(valid[:, None], speed_target[idx, :3], speed.detach())
                speed_loss = torch.where(valid, (speed-target).square().mean(-1), 0.).sum()/valid.sum().clamp_min(1)
                self.cart_velocity_optimizer.zero_grad()
                if valid.any():
                    speed_loss.backward()
                    nn.utils.clip_grad_norm_(actor.cart_velocity_parameters(), self.estimator_max_grad_norm)
                    self.cart_velocity_optimizer.step()
                # Fresh prediction from the speed head, never speed_target.
                hands = actor.predict_hand_wrench_history(inputs[idx])
                force_loss, valid_fraction = masked_interaction_huber(hands, hands_target[idx], self.huber_delta)
                self.estimator_optimizer.zero_grad()
                if valid_fraction > 0:
                    force_loss.backward()
                    nn.utils.clip_grad_norm_(actor.force_parameters(), self.estimator_max_grad_norm)
                    self.estimator_optimizer.step()
                sums[0] += speed_loss.item()
                sums[1] += force_loss.item()
                count += 1
        self.estimator_optimizer.zero_grad(set_to_none=True)
        self.cart_velocity_optimizer.zero_grad(set_to_none=True)
        out.update(cart_velocity_supervised=sums[0]/count, estimator=sums[1]/count)
        return out

    @torch.no_grad()
    def estimator_metrics(self, inputs, hand_targets, speed_targets, max_rows=32768):
        rows = torch.randperm(len(inputs), device=inputs.device)[:max_rows]
        actor = self.actor
        speed = actor.predict_cart_velocity_history(inputs[rows])
        hands = actor.predict_hand_wrench_history(inputs[rows], speed)
        out = {}
        for prefix, pred, target, scale, names in (
            ("hand_mae", hands, hand_targets[rows], actor.hand_scale,
             ("L_fx", "L_fy", "L_fz", "L_mx", "L_my", "L_mz", "R_fx", "R_fy", "R_fz", "R_mx", "R_my", "R_mz")),
            ("cart_velocity_mae", speed, speed_targets[rows], 1./actor.cart_velocity_scale, ("vx", "vy", "omega_z")),
        ):
            valid = (target[:, -1] > .5) & torch.isfinite(target).all(-1)
            if valid.any():
                error = ((pred[valid]-target[valid, :-1])*scale).abs().mean(0).tolist()
                out.update({f"{prefix}_{name}": value for name, value in zip(names, error)})
        return out
