"""Pulse-free fast path preserves physical duration, frames and clearing."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'HomieRL'), str(ROOT / 'HomieRL/rsl_rl')]
from homie_pushcart_v51.task_env import PushCartEnv
from homie_pushcart_v51.sim_env import PushCartSimEnv


def adapter():
    env = PushCartEnv.__new__(PushCartEnv)
    env.num_envs, env.device, env.dt = 2, 'cpu', .02
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005))
    env.cart_pulse_timer = torch.zeros(2, dtype=torch.long)
    env.cart_pulse_wrench = torch.zeros(2, 6)
    env.cart_applied_wrench = torch.zeros(2, 6)
    env.cart_pulse_impulse = torch.zeros(2, 6)
    env._cart_applied_reference = torch.zeros(2, 3)
    env._cart_pulse_steps_remaining = 0
    env._cart_pulse_needs_clear = False
    env._cart_pulse_body_ids = torch.tensor([0], dtype=torch.int32)
    env.cart_base_body = 0
    env.cart = SimpleNamespace(
        data=SimpleNamespace(body_com_pose_w=SimpleNamespace(torch=torch.zeros(2, 1, 7))),
        permanent_wrench_composer=Mock())
    env.cart_com_world = Mock(return_value=torch.tensor([[0., .1, 0.], [0., .1, 0.]]))
    return env


class PulseFastTests(unittest.TestCase):
    def test_no_pulse_does_not_query_cart_or_submit_wrench(self):
        env = adapter()
        with patch.object(PushCartSimEnv, '_apply_action'):
            for _ in range(8):
                env._apply_action()
        env.cart_com_world.assert_not_called()
        env.cart.permanent_wrench_composer.set_forces_and_torques_index.assert_not_called()

    def test_duration_impulse_reference_and_one_final_clear(self):
        env = adapter()
        env.inject_cart_wrench([10., 0., 0.], [0., 0., 0.], .025,
                              env_ids=[1], offset_w=[0., .2, 0.])
        trace = []
        with patch.object(PushCartSimEnv, '_apply_action'):
            for i in range(8):
                env._apply_action()
                trace.append(env.cart_applied_wrench.clone())
                submitted = env.cart.permanent_wrench_composer.set_forces_and_torques_index.call_args.kwargs
                self.assertAlmostEqual(float(submitted['torques'][1, 0, 2]), -3. if i < 5 else 0.)
        trace = torch.stack(trace)
        torch.testing.assert_close(trace[:, 0], torch.zeros(8, 6))
        torch.testing.assert_close(trace[:5, 1], torch.tensor([10., 0., 0., 0., 0., -2.]).expand(5, -1))
        torch.testing.assert_close(trace[5:], torch.zeros(3, 2, 6))
        torch.testing.assert_close(env.cart_pulse_impulse[1], torch.tensor([.25, 0., 0., 0., 0., -.05]))
        self.assertEqual(env.cart_com_world.call_count, 5)
        self.assertEqual(env.cart.permanent_wrench_composer.set_forces_and_torques_index.call_count, 6)

    def test_overlapping_injections_replacement_and_partial_reset(self):
        env = adapter()
        with patch.object(PushCartSimEnv, '_apply_action'):
            env.inject_cart_wrench([10., 0., 0.], [0., 0., 0.], .02, env_ids=[0])
            env._apply_action()
            env.inject_cart_wrench([20., 0., 0.], [0., 0., 0.], .03, env_ids=[1])
            # A reset clears rows but need not synchronize/recompute the CPU bound.
            env.cart_pulse_timer[0] = 0
            env.cart_pulse_wrench[0] = 0.
            env.cart_applied_wrench[0] = 0.
            for i in range(7):
                env._apply_action()
                self.assertEqual(float(env.cart_applied_wrench[0].abs().sum()), 0.)
                self.assertEqual(float(env.cart_applied_wrench[1, 0]), 20. if i < 6 else 0.)
            # New injection after clearing must become active immediately.
            env.inject_cart_wrench([30., 0., 0.], [0., 0., 0.], .005, env_ids=[0])
            env._apply_action()
            self.assertEqual(float(env.cart_applied_wrench[0, 0]), 30.)
            env._apply_action()
            self.assertEqual(float(env.cart_applied_wrench.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
