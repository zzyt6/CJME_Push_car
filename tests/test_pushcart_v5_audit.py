"""Regression cases found by the V5 adversarial audit."""
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "HomieRL"), str(ROOT / "HomieRL/rsl_rl")]
from homie_pushcart_v5.task_env import PushCartEnv
from homie_pushcart_v5.task_config import task_cfg_from_run
from homie_pushcart_v5.lowlevel import HimLowLevelPolicy


class ActorInputAuditTests(unittest.TestCase):
    def fixture(self):
        env = PushCartEnv.__new__(PushCartEnv)
        env.device, env.num_envs, env.num_hl_actions = "cpu", 2, 18
        env.task_cfg = SimpleNamespace()
        return env

    def test_nonfinite_action_fails_before_any_buffers_or_physics_are_touched(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            action = torch.zeros(2, 18)
            action[1, 10] = value
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "before advancing"):
                self.fixture().step(action)

    def test_broadcastable_wrong_action_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.fixture().step(torch.zeros(1, 18))


class SavedConfigAuditTests(unittest.TestCase):
    def test_existing_broken_config_cannot_silently_restore_current_defaults(self):
        for text in ("task: [", "homie: {}", "task: null"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                params = Path(folder) / "params"
                params.mkdir()
                (params / "env.yaml").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    task_cfg_from_run(folder, verbose=False)

    def test_current_him_observation_is_used_for_velocity_inference(self):
        policy = HimLowLevelPolicy.__new__(HimLowLevelPolicy)
        policy.model = SimpleNamespace(actor_proprioceptive_obs_length=6,
                                       estimator=lambda obs: (obs[:, -3:], torch.zeros(len(obs), 1)))
        old = torch.arange(6.).reshape(1, 6)
        current = old + 10.
        torch.testing.assert_close(policy.estimate_velocity(current), torch.tensor([[13., 14., 15.]]))
        self.assertFalse(policy.estimate_velocity(current).requires_grad)


if __name__ == "__main__":
    unittest.main()
