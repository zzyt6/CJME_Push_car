# PPO update structure adapted from RSL-RL (ETH Zurich / NVIDIA, BSD-3-Clause).
"""RSL-RL 5 PPO with a masked, same-timestamp interaction supervision loss.

RolloutStorage stores all TensorDict groups before each action. The encoder is
recomputed during each minibatch update, so no stale rollout latent is trained.
The standard runner saves all actor modules and optimizer state without hooks.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.algorithms import PPO

from .interaction_policy import InteractionActor


def masked_interaction_huber(
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Huber loss and valid fraction for [Fx/80, Fy/80, Mz/12, valid].

    Mask before arithmetic: NaNs in invalid labels must never reach the loss.
    All-invalid batches produce a differentiable zero with zero gradients.
    """
    if targets.shape[:-1] != predictions.shape[:-1] or targets.shape[-1] != 4 or predictions.shape[-1] != 3:
        raise ValueError("Expected predictions [...,3] and interaction targets [...,4].")
    valid = (targets[..., 3] > 0.5) & torch.isfinite(targets).all(dim=-1)
    safe_targets = torch.where(valid[..., None], targets[..., :3], predictions.detach())
    component_loss = F.huber_loss(predictions, safe_targets, reduction="none", delta=delta)
    row_loss = component_loss.mean(dim=-1)
    loss = torch.where(valid, row_loss, torch.zeros_like(row_loss)).sum() / valid.sum().clamp_min(1)
    return loss, valid.float().mean()


class InteractionPPO(PPO):
    """Feed-forward PPO plus interaction loss; aux_coef=0 is equal-capacity PPO."""

    def __init__(self, actor, critic, storage, *, aux_coef: float = 0.1,
                 huber_delta: float = 1.0, **kwargs) -> None:
        if not isinstance(actor, InteractionActor):
            raise TypeError("InteractionPPO requires an InteractionActor.")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("V5.1 interaction PPO supports feed-forward models only.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("V5.1 interaction PPO does not support RND or symmetry augmentation.")
        if not math.isfinite(aux_coef) or aux_coef < 0:
            raise ValueError("aux_coef must be finite and non-negative.")
        if not math.isfinite(huber_delta) or huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive.")
        self.aux_coef = aux_coef
        self.huber_delta = huber_delta
        super().__init__(actor, critic, storage, **kwargs)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            # Adaptive KL scheduling uses this scalar in addition to optimizer
            # groups; resume from the saved rate instead of the config default.
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return restore_iteration

    def update(self) -> dict[str, float]:
        totals = dict(value=0.0, surrogate=0.0, entropy=0.0, interaction=0.0, interaction_valid=0.0)
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

            if self.aux_coef > 0:
                # Recomputed from the same pre-action observation saved by PPO.act.
                interaction_loss, valid_fraction = masked_interaction_huber(
                    self.actor.predict_interaction(batch.observations),
                    batch.observations["interaction_target"], self.huber_delta)
                loss = loss + self.aux_coef * interaction_loss
                totals["interaction"] += interaction_loss.item()
                totals["interaction_valid"] += valid_fraction.item()
            # aux_coef=0 intentionally never reads labels, while retaining the
            # identical encoder/head/policy and their PPO gradient paths.
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["entropy"] += entropy.mean().item()
            num_updates += 1
        self.storage.clear()
        if not num_updates:
            raise RuntimeError("Interaction PPO received no minibatches.")
        return {key: value / num_updates for key, value in totals.items()}
