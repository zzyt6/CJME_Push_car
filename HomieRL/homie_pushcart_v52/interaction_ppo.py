# PPO update structure adapted from RSL-RL (ETH Zurich / NVIDIA, BSD-3-Clause).
"""RSL-RL 5 PPO plus an independently optimized per-hand interaction estimator.

The estimator (``actor.force_estimator``) is excluded from the PPO optimizer and
trained after each PPO update by its own Adam on the same rollout: masked Huber
loss on the 12 normalized per-hand targets, fixed learning rate, its own gradient
clip. The policy reads a detached prediction, so the two objectives cannot trade
off against each other and the KL-adaptive PPO learning rate does not throttle
the estimator. The standard runner saves both optimizers through ``save/load``.
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
    """Feed-forward PPO; the force estimator has its own optimizer and objective."""

    def __init__(self, actor, critic, storage, *, estimator_learning_rate: float = 1.0e-3,
                 estimator_epochs: int = 5, estimator_mini_batches: int = 8,
                 estimator_max_grad_norm: float = 1.0, huber_delta: float = 1.0, **kwargs) -> None:
        if not isinstance(actor, InteractionActor):
            raise TypeError("InteractionPPO requires an InteractionActor.")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("V5.2 interaction PPO supports feed-forward models only.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("V5.2 interaction PPO does not support RND or symmetry augmentation.")
        for name, value in (("estimator_learning_rate", estimator_learning_rate), ("huber_delta", huber_delta),
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
        self.estimator_optimizer = torch.optim.Adam(actor.estimator_parameters(), lr=estimator_learning_rate)

    def save(self) -> dict:
        saved = super().save()
        saved["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            # Adaptive KL scheduling uses this scalar in addition to optimizer
            # groups; resume from the saved rate instead of the config default.
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
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
        """Supervised pass(es) over the stored pre-action observations and labels.

        Metrics are computed *before* fitting, on this fresh rollout, so the logged
        R2 is an out-of-sample estimate (it is the accuracy the policy just used).
        """
        if self.estimator_epochs == 0:
            return {}
        actor = self.actor
        width = actor.frame_dim * actor.estimator_history
        observations = self.storage.observations.flatten(0, 1)
        inputs = observations["policy"][..., -width:]
        targets = observations["interaction_target"]
        out = self.estimator_metrics(inputs, targets)
        loss_sum, valid_sum, count = 0.0, 0.0, 0
        size = inputs.shape[0] // self.estimator_mini_batches
        for _ in range(self.estimator_epochs):
            order = torch.randperm(inputs.shape[0], device=inputs.device)
            for i in range(self.estimator_mini_batches):
                idx = order[i * size:(i + 1) * size]
                prediction = actor.force_estimator(actor.estimator_normalizer(inputs[idx]))
                loss, valid_fraction = masked_interaction_huber(prediction, targets[idx], self.huber_delta)
                self.estimator_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.estimator_parameters(), self.estimator_max_grad_norm)
                self.estimator_optimizer.step()
                loss_sum += loss.item()
                valid_sum += valid_fraction.item()
                count += 1
        self.estimator_optimizer.zero_grad(set_to_none=True)
        out.update(estimator=loss_sum / count, estimator_valid=valid_sum / count)
        return out

    @torch.no_grad()
    def estimator_metrics(self, inputs, targets, max_rows: int = 32768) -> dict[str, float]:
        """Pre-fit R2 per hand axis and of the composed handle resultant [Fx, Fy, Mz]."""
        actor = self.actor
        rows = torch.randperm(inputs.shape[0], device=inputs.device)[:max_rows]
        prediction = actor.force_estimator(actor.estimator_normalizer(inputs[rows]))
        y = targets[rows, :HAND_TARGET_DIM]
        valid = (targets[rows, HAND_TARGET_DIM] > 0.5) & torch.isfinite(targets[rows]).all(-1)
        if not valid.any():
            return {}
        prediction, y = prediction[valid], y[valid]
        out = {"estimator_eval_huber": masked_interaction_huber(
            prediction, torch.cat((y, torch.ones_like(y[:, :1])), -1), self.huber_delta)[0].item()}

        def r2(p, t):
            return (1. - (p - t).square().mean(0) / t.var(0, unbiased=False).clamp_min(1e-8)).tolist()

        names = ("L_fx", "L_fy", "L_fz", "L_mx", "L_my", "L_mz", "R_fx", "R_fy", "R_fz", "R_mx", "R_my", "R_mz")
        hand = r2(prediction, y)
        for name in ("L_fx", "L_fy", "L_fz", "L_mz", "R_fx", "R_fy", "R_fz", "R_mz"):
            out[f"estimator_r2_{name}"] = hand[names.index(name)]
        for name, value in zip(("Fx", "Fy", "Mz"), r2(actor.compose_resultant(prediction), actor.compose_resultant(y))):
            out[f"estimator_r2_handle_{name}"] = value
        return out
