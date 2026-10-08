# PPO update structure adapted from RSL-RL (ETH Zurich / NVIDIA, BSD-3-Clause).
"""V3.2 PPO, a HIM update per PPO minibatch, and independent wrench supervision."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.algorithms import PPO

from .interaction_policy import InteractionActor, CRITIC_DIM, ROBOT_VELOCITY_SCALE


def masked_interaction_huber(predictions, targets, delta=1.0):
    """V5.3's per-hand Huber loss; discard invalid labels before arithmetic."""
    if predictions.shape[-1] != 12 or targets.shape != (*predictions.shape[:-1], 13):
        raise ValueError("V7 hand predictions/targets must have 12/13 columns.")
    valid = (targets[..., 12] > 0.5) & torch.isfinite(targets).all(dim=-1)
    safe = torch.where(valid[..., None], targets[..., :12], predictions.detach())
    row_loss = F.huber_loss(predictions, safe, reduction="none", delta=delta).mean(-1)
    loss = torch.where(valid, row_loss, torch.zeros_like(row_loss)).sum() / valid.sum().clamp_min(1)
    return loss, valid.float().mean()


class InteractionPPO(PPO):
    def __init__(self, actor, critic, storage, *, estimator_learning_rate=1.e-3,
                 estimator_epochs=5, estimator_mini_batches=8,
                 estimator_max_grad_norm=1.0, huber_delta=1.0, robot_velocity_learning_rate=3.0e-4, **kwargs):
        if not isinstance(actor, InteractionActor) or actor.is_recurrent or critic.is_recurrent:
            raise TypeError("V7 requires its feed-forward InteractionActor and Critic.")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("V7 preserves V3.2 PPO without new RND or symmetry augmentation.")
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("V7 auxiliary optimizers currently require the baseline single-device runner.")
        if not critic.obs_normalization or critic.obs_dim != CRITIC_DIM:
            raise ValueError("V7 Critic must normalize the original 652 columns plus 13 hand labels and 30 V7 privileges.")
        for key, width in (("policy", 605), ("estimator", 462), ("critic", 695),
                           ("hand_target", 13), ("him_next_target", 81)):
            if key not in storage.observations.keys() or storage.observations[key].shape[-1] != width:
                raise ValueError(f"V7 rollout storage requires {key} with {width} columns.")
        if actor.robot_velocity_head and ("robot_velocity_target" not in storage.observations.keys()
                                          or storage.observations["robot_velocity_target"].shape[-1] != 4):
            raise ValueError("V7.1 rollout storage requires robot_velocity_target with 4 columns (velocity x2, valid).")
        for name, value in (("estimator_learning_rate", estimator_learning_rate),
                            ("estimator_max_grad_norm", estimator_max_grad_norm), ("huber_delta", huber_delta)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if estimator_epochs < 0 or estimator_mini_batches < 1:
            raise ValueError("Force estimator epochs must be nonnegative and minibatches positive.")
        self.estimator_epochs, self.estimator_mini_batches = int(estimator_epochs), int(estimator_mini_batches)
        self.estimator_max_grad_norm, self.huber_delta = estimator_max_grad_norm, huber_delta
        super().__init__(actor, critic, storage, **kwargs)
        estimator_ids = {id(p) for p in actor.estimator_parameters()}
        self.policy_parameters = [p for p in actor.parameters() if id(p) not in estimator_ids]
        self.optimizer = type(self.optimizer)(self.policy_parameters + list(critic.parameters()), lr=self.learning_rate)
        self.estimator_optimizer = torch.optim.Adam(actor.force_parameters(), lr=estimator_learning_rate)
        self.cart_velocity_optimizer = actor.cart_him.optimizer
        # V7.1: fine-tune the HOMIE-initialised pelvis-velocity head; small rate, it starts from a trained estimator
        self.robot_velocity_optimizer = (torch.optim.Adam(actor.robot_velocity_parameters(), lr=robot_velocity_learning_rate)
                                         if actor.robot_velocity_head else None)

    def act(self, obs):
        actions = super().act(obs)
        # Environment buffers may be updated in-place by step(). Preserve t's
        # observations and CURRENT hand labels before that happens.
        self.transition.observations = obs.detach().clone()
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        # step() exposes pre-reset t+1 HIM targets even where returned policy
        # observations already belong to the reset episode. Backfill only this
        # target into the independent t observation snapshot.
        target = obs["him_next_target"]
        if target.shape != (rewards.shape[0], 81):
            raise ValueError("V7 step must return pre-reset HIM next targets [N, 81].")
        recorded = self.transition.observations.clone()
        recorded["him_next_target"] = target.detach().clone()
        self.transition.observations = recorded
        super().process_env_step(obs, rewards, dones, extras)

    def save(self):
        saved = super().save()
        saved["force_estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        saved["cart_him_optimizer_state_dict"] = self.cart_velocity_optimizer.state_dict()
        saved["v7_network_metadata"] = self.actor.metadata()
        if self.robot_velocity_optimizer is not None:
            saved["robot_velocity_optimizer_state_dict"] = self.robot_velocity_optimizer.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("v7_network_metadata") != self.actor.metadata():
            raise ValueError("V7 checkpoint network metadata differs; cross-version loading is not implicit.")
        optimizer_requested = load_cfg is None or load_cfg.get("optimizer")
        if optimizer_requested:
            keys = ["force_estimator_optimizer_state_dict", "cart_him_optimizer_state_dict"]
            if self.robot_velocity_optimizer is not None:
                keys.append("robot_velocity_optimizer_state_dict")
            for key in keys:
                if key not in loaded_dict:
                    raise ValueError(f"V7 resume requires {key}.")
        restored_iteration = super().load(loaded_dict, load_cfg, strict)
        if optimizer_requested:
            self.estimator_optimizer.load_state_dict(loaded_dict["force_estimator_optimizer_state_dict"])
            self.cart_velocity_optimizer.load_state_dict(loaded_dict["cart_him_optimizer_state_dict"])
            if self.robot_velocity_optimizer is not None:
                self.robot_velocity_optimizer.load_state_dict(loaded_dict["robot_velocity_optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            self.actor.cart_him.learning_rate = self.cart_velocity_optimizer.param_groups[0]["lr"]
        return restored_iteration

    def update(self):
        totals = dict(value=0., surrogate=0., entropy=0., cart_velocity_supervised=0., cart_him_swap=0.)
        count = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = ((batch.advantages - batch.advantages.mean())
                                        / (batch.advantages.std() + 1.e-8))
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, self.actor.output_distribution_params).mean()
                    if kl > self.desired_kl * 2.:
                        self.learning_rate = max(1.e-5, self.learning_rate / 1.5)
                    elif 0. < kl < self.desired_kl / 2.:
                        self.learning_rate = min(1.e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            # HOMIE timing: update HIM once per PPO minibatch, after adaptive
            # LR selection. All predictions already used by PPO are detached.
            target = batch.observations["him_next_target"]
            valid = (target[:, 80] > .5) & torch.isfinite(target).all(-1)
            mse, swap = self.actor.cart_him.update(
                batch.observations["estimator"], target[:, :80], lr=self.learning_rate, valid=valid)
            self.cart_velocity_optimizer.zero_grad(set_to_none=True)

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            advantages = batch.advantages.squeeze(-1)
            surrogate_loss = torch.maximum(-advantages * ratio,
                -advantages * ratio.clamp(1. - self.clip_param, 1. + self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(), (clipped - batch.returns).square()).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_parameters, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for key, value in (("value", value_loss.item()), ("surrogate", surrogate_loss.item()),
                               ("entropy", entropy.mean().item()), ("cart_velocity_supervised", mse),
                               ("cart_him_swap", swap)):
                totals[key] += value
            count += 1
        if count == 0:
            raise RuntimeError("V7 PPO received no minibatches.")
        result = {key: value / count for key, value in totals.items()}
        result.update(self.update_estimator())
        self.storage.clear()
        return result

    def update_estimator(self):
        """Current-state hand supervision with the V5.3 optimizer schedule."""
        if self.estimator_epochs == 0:
            return {}
        obs = self.storage.observations.flatten(0, 1)
        out = self.estimator_metrics(obs)
        total, count = 0., 0
        for _ in range(self.estimator_epochs):
            order = torch.randperm(len(obs), device=self.device)
            for ids in order.tensor_split(min(self.estimator_mini_batches, len(obs))):
                hands = self.actor.predict_hand_wrench(obs[ids])
                loss, fraction = masked_interaction_huber(hands, obs["hand_target"][ids], self.huber_delta)
                self.estimator_optimizer.zero_grad()
                if fraction > 0:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.force_parameters(), self.estimator_max_grad_norm)
                    self.estimator_optimizer.step()
                total += loss.item()
                count += 1
        self.estimator_optimizer.zero_grad(set_to_none=True)
        out["estimator"] = total / count
        if self.robot_velocity_optimizer is not None:
            out.update(self.update_robot_velocity(obs))
        return out

    def update_robot_velocity(self, obs):
        """V7.1: supervised fine-tuning of the pelvis-velocity head on the push-cart rollouts (MSE, valid rows)."""
        target = obs["robot_velocity_target"]
        valid = (target[:, 3] > .5) & torch.isfinite(target).all(-1)
        total, count = 0., 0
        for _ in range(self.estimator_epochs):
            order = torch.randperm(len(obs), device=self.device)
            for ids in order.tensor_split(min(self.estimator_mini_batches, len(obs))):
                keep = ids[valid[ids]]
                if len(keep) == 0:
                    continue
                prediction = self.actor.predict_robot_velocity(obs[keep])
                loss = F.mse_loss(prediction, target[keep, :3])
                self.robot_velocity_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.robot_velocity.parameters(), self.estimator_max_grad_norm)
                self.robot_velocity_optimizer.step()
                total += loss.item()
                count += 1
        self.robot_velocity_optimizer.zero_grad(set_to_none=True)
        return {"robot_velocity_supervised": total / max(count, 1)}

    @torch.no_grad()
    def estimator_metrics(self, obs, max_rows=32768):
        rows = torch.randperm(len(obs), device=self.device)[:max_rows]
        selected = obs[rows]
        speed = self.actor.predict_cart_velocity(selected)
        hands = self.actor.predict_hand_wrench(selected)
        result = {}
        if self.actor.robot_velocity_head:
            target = selected["robot_velocity_target"]
            mask = (target[:, 3] > .5) & torch.isfinite(target).all(-1)
            for prefix, pretrained in (("robot_velocity_mae", False), ("robot_velocity_homie_mae", True)):
                pred = self.actor.predict_robot_velocity(selected, pretrained=pretrained)
                errors = (pred[mask] - target[mask, :3]).abs() / ROBOT_VELOCITY_SCALE
                mean = errors.mean(0) if mask.any() else torch.zeros(3, device=self.device)
                for name, value in zip(("vx", "vy", "vz"), mean):
                    result[f"{prefix}_{name}"] = value.item()
        for prefix, prediction, target, valid, scale, names in (
            ("cart_velocity_mae", speed, selected["him_next_target"][:, 77:80],
             selected["him_next_target"][:, 80], 1. / self.actor.cart_velocity_scale, ("vx", "vy", "wz")),
            ("hand_mae", hands, selected["hand_target"][:, :12], selected["hand_target"][:, 12],
             self.actor.hand_scale, tuple(f"{side}_{axis}" for side in ("L", "R")
                                         for axis in ("fx", "fy", "fz", "mx", "my", "mz"))),
        ):
            mask = (valid > .5) & torch.isfinite(target).all(-1)
            errors = (prediction[mask] - target[mask]).abs() * scale
            mean = errors.mean(0) if mask.any() else torch.zeros(len(names), device=self.device)
            for name, value in zip(names, mean):
                result[f"{prefix}_{name}"] = value.item()
            result[f"{prefix}_valid_fraction"] = mask.float().mean().item()
        return result
