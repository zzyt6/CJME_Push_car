"""Verify the official bridge preserves the author's data and optimization."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import torch
from tensordict import TensorDict

from homie_isaaclab.config import G1RoughCfg, G1RoughCfgPPO
from homie_isaaclab.lab_config import HomieDirectEnvCfg, HomieRunnerCfg
from homie_isaaclab.math_utils import class_to_dict
from homie_isaaclab.rsl_rl import HimPpoAdapter
from homie_rsl_rl.algorithms import HIMPPO
from homie_rsl_rl.modules import HIMActorCritic

ROOT = Path(__file__).resolve().parents[1]


def original_configs():
    """Execute the original pure config classes without importing Isaac Gym."""
    namespace = {"__name__": "homie_original_config_test"}
    base = ROOT / "HomieRL/legged_gym/legged_gym/envs"
    for path in (base / "base/base_config.py", base / "base/legged_robot_config.py", base / "g1/g1_29dof_config.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("legged_gym"))
        )]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["G1RoughCfg"](), namespace["G1RoughCfgPPO"]()


def make_algorithm():
    model = HIMActorCritic(456, 79, 76, 79, 6, 1, 12,
                           actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])
    cfg = class_to_dict(G1RoughCfgPPO().algorithm)
    cfg.update(num_learning_epochs=1, num_mini_batches=2)
    algorithm = HIMPPO(model, device="cpu", **cfg)
    algorithm.init_storage(2, 2, [456], [79], [12])
    return algorithm


class OfficialTrainingTests(unittest.TestCase):
    def test_config_matches_author_sources(self):
        original_env, original_train = original_configs()
        self.assertEqual(class_to_dict(G1RoughCfg()), class_to_dict(original_env))
        self.assertEqual(class_to_dict(G1RoughCfgPPO()), class_to_dict(original_train))
        official = HomieRunnerCfg()
        algorithm = official.algorithm.to_dict()
        algorithm.pop("class_name")
        algorithm.pop("rnd_cfg")
        self.assertEqual(algorithm, class_to_dict(original_train.algorithm))
        self.assertEqual(official.him_policy, class_to_dict(original_train.policy))
        for key in ("num_steps_per_env", "max_iterations", "save_interval"):
            self.assertEqual(getattr(official, key), getattr(original_train.runner, key))
        self.assertEqual(HomieDirectEnvCfg().scene.num_envs, original_env.env.num_envs)

    def test_original_and_modern_rsl_rl_coexist(self):
        import rsl_rl
        from rsl_rl.runners import OnPolicyRunner
        self.assertNotIn(str(ROOT / "HomieRL/rsl_rl"), str(Path(rsl_rl.__file__)))
        self.assertTrue(hasattr(OnPolicyRunner, "learn"))
        self.assertEqual(HIMPPO.__module__, "homie_rsl_rl.algorithms.him_ppo")

    def test_bridge_matches_original_update_and_restores_both_optimizers(self):
        torch.manual_seed(10)
        original = make_algorithm()
        adapted = HimPpoAdapter(deepcopy(original))
        for step in range(2):
            obs = TensorDict({"policy": torch.randn(2, 456), "critic": torch.randn(2, 79)}, [2])
            next_obs = TensorDict({"policy": torch.randn(2, 456), "critic": torch.randn(2, 79)}, [2])
            rewards = torch.tensor([0.5, -0.2])
            dones = torch.tensor([True, False])
            terminal = torch.full((1, 79), 3.0 + step)
            extras = {"time_outs": torch.tensor([True, False]), "homie_terminal_ids": torch.tensor([0]),
                      "homie_terminal_critic": terminal}
            rng = torch.get_rng_state()
            with torch.inference_mode():
                expected_actions = original.act(obs["policy"], obs["critic"])
                torch.set_rng_state(rng)
                actions = adapted.act(obs)
                torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
                expected_next = next_obs["critic"].clone()
                expected_next[0] = terminal[0]
                original.process_env_step(rewards, dones, extras, expected_next)
                adapted.process_env_step(next_obs, rewards, dones, extras)
                torch.testing.assert_close(adapted.him.storage.next_privileged_observations[step * 2, 0], terminal[0])
                torch.testing.assert_close(adapted.him.storage.next_privileged_observations[step * 2, 1], next_obs["critic"][1])
        with torch.inference_mode():
            original.compute_returns(next_obs["critic"])
            adapted.compute_returns(next_obs)
        rng = torch.get_rng_state()
        expected_losses = original.update()
        torch.set_rng_state(rng)
        losses = adapted.update()
        self.assertEqual(tuple(losses.values()), expected_losses)
        for key, value in original.actor_critic.state_dict().items():
            torch.testing.assert_close(adapted.him.actor_critic.state_dict()[key], value, rtol=0, atol=0)
        saved = deepcopy(adapted.save())
        restored = HimPpoAdapter(make_algorithm())
        self.assertTrue(restored.load(saved))
        self.assertTrue(restored.him.optimizer.state)
        self.assertTrue(restored.him.actor_critic.estimator.optimizer.state)
        self.assertEqual(restored.learning_rate, adapted.learning_rate)
        for left_opt, right_opt in ((adapted.him.optimizer, restored.him.optimizer),
                                    (adapted.him.actor_critic.estimator.optimizer, restored.him.actor_critic.estimator.optimizer)):
            for key, state in left_opt.state_dict()["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(right_opt.state_dict()["state"][key][field], value, rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored.get_policy()(obs), adapted.get_policy()(obs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
