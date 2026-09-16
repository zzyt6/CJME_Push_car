"""Adapt the unchanged HOMIE HIM-PPO math to RSL-RL 5's OnPolicyRunner.

The old sources load under homie_rsl_rl; the installed rsl_rl stays untouched.
This adapter translates observations, estimator targets and checkpoint state.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn
from homie_rsl_rl.algorithms import HIMPPO
from homie_rsl_rl.modules import HIMActorCritic


class HimPolicy(nn.Module):
    """Deterministic HIM inference accepting the official TensorDict input."""
    is_recurrent = False

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    @property
    def output_std(self):
        return self.actor_critic.std

    def forward(self, obs):
        return self.actor_critic.act_inference(obs["policy"])

    def reset(self, dones=None):
        self.actor_critic.reset(dones)


class HimPpoAdapter:
    def __init__(self, algorithm):
        self.him = algorithm
        self.policy = HimPolicy(algorithm.actor_critic)
        self.num_updates = 0

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        if cfg.get("multi_gpu"):
            raise ValueError("The original HOMIE optimizer currently supports single-GPU training only.")
        core = env.unwrapped.homie
        model = HIMActorCritic(
            core.num_obs, core.num_privileged_obs,
            core.num_one_step_obs, core.num_one_step_privileged_obs,
            core.actor_history_length, core.critic_history_length, core.num_lower_dof,
            **cfg["him_policy"],
        ).to(device)
        algorithm_cfg = deepcopy(cfg["algorithm"])
        algorithm_cfg.pop("class_name")
        if algorithm_cfg.pop("rnd_cfg") is not None:
            raise ValueError("HOMIE does not implement RND.")
        algorithm = HIMPPO(model, device=device, **algorithm_cfg)
        algorithm.init_storage(core.num_envs, cfg["num_steps_per_env"],
                               [core.num_obs], [core.num_privileged_obs], [core.num_lower_dof])
        if env.unwrapped.cfg.log_dir:
            import rsl_rl
            import homie_rsl_rl
            report = {
                "runner": "rsl_rl.runners.OnPolicyRunner",
                "rsl_rl_source": rsl_rl.__file__,
                "algorithm": "homie_rsl_rl.algorithms.HIMPPO",
                "him_source": homie_rsl_rl.__file__,
                "num_envs": core.num_envs,
                "actor_observations": core.num_obs,
                "critic_observations": core.num_privileged_obs,
                "actions": core.num_lower_dof,
                "terminal_critic_in_extras": True,
            }
            (Path(env.unwrapped.cfg.log_dir) / "integration.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
        print("[HOMIE] Official RSL-RL OnPolicyRunner + original HIMPPO/HIMActorCritic")
        return HimPpoAdapter(algorithm)

    @property
    def learning_rate(self):
        return self.him.learning_rate

    def train_mode(self):
        self.him.actor_critic.train()

    def eval_mode(self):
        self.him.actor_critic.eval()

    def get_policy(self):
        return self.policy

    def act(self, obs):
        return self.him.act(obs["policy"], obs["critic"])

    def process_env_step(self, obs, rewards, dones, extras):
        next_critic = obs["critic"].clone().detach()
        ids = extras["homie_terminal_ids"].to(self.him.device)
        terminal = extras["homie_terminal_critic"].to(self.him.device)
        next_critic[ids] = terminal
        self.him.process_env_step(rewards, dones, extras, next_critic)

    def compute_returns(self, obs):
        self.him.compute_returns(obs["critic"])

    def update(self):
        names = ("value", "surrogate", "estimation", "swap", "actor_symmetry", "critic_symmetry")
        losses = dict(zip(names, self.him.update()))
        if not all(math.isfinite(float(value)) for value in losses.values()):
            raise FloatingPointError(f"Non-finite HIM losses: {losses}")
        self.num_updates += 1
        return losses

    def save(self):
        if not all(torch.isfinite(p).all() for p in self.him.actor_critic.parameters()):
            raise FloatingPointError("Non-finite HIM checkpoint parameters")
        return {
            "model_state_dict": self.him.actor_critic.state_dict(),
            "optimizer_state_dict": self.him.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.him.actor_critic.estimator.optimizer.state_dict(),
            "homie_learning_rate": self.him.learning_rate,
            "homie_checkpoint_format": "isaaclab_rsl_rl_5",
            "homie_updates": self.num_updates,
        }

    def load(self, loaded_dict, load_cfg=None, strict=True):
        self.him.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        # Original HOMIE stores the next iteration; the modern runner stores
        # the last iteration index. Track actual updates separately from labels.
        fallback_updates = loaded_dict.get("iter", 0)
        if "homie_checkpoint_format" in loaded_dict and "iter" in loaded_dict:
            fallback_updates += 1
        self.num_updates = loaded_dict.get("homie_updates", fallback_updates)
        load_cfg = load_cfg or {}
        if load_cfg.get("optimizer", True):
            self.him.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.him.actor_critic.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            self.him.learning_rate = loaded_dict.get(
                "homie_learning_rate", self.him.optimizer.param_groups[0]["lr"],
            )
        return load_cfg.get("iteration", True)
