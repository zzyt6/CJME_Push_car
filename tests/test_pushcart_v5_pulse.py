"""Physical interval contract of the Isaac task disturbance adapter."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "HomieRL"), str(ROOT / "HomieRL/rsl_rl")]
from homie_pushcart_v5.task_env import PushCartEnv
from homie_pushcart_v5.sim_env import PushCartSimEnv


def adapter():
    env = PushCartEnv.__new__(PushCartEnv)
    env.num_envs, env.device, env.dt = 2, "cpu", .02
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005))
    env.cart_pulse_timer = torch.zeros(2, dtype=torch.long)
    env.cart_pulse_wrench = torch.zeros(2, 6)
    env.cart_applied_wrench = torch.zeros(2, 6)
    env.cart_pulse_impulse = torch.zeros(2, 6)
    env.cart_base_body = 0
    env.cart = SimpleNamespace(
        data=SimpleNamespace(body_com_pose_w=SimpleNamespace(torch=torch.zeros(2, 1, 7))),
        permanent_wrench_composer=Mock(), write_joint_damping_to_sim_index=Mock())
    env.cart_com_world = lambda: torch.tensor([[0., .1, 0.], [0., .1, 0.]])
    return env


class IsaacPhysicalPulseTests(unittest.TestCase):
    def test_partial_control_interval_preserves_impulse_and_moment_origin(self):
        env = adapter()
        env.inject_cart_wrench([10., 0., 0.], [0., 0., 0.], .025, env_ids=[1], offset_w=[0., .2, 0.])
        trace = []
        with patch.object(PushCartSimEnv, "_apply_action"):
            for _ in range(8):
                env._apply_action()
                trace.append(env.cart_applied_wrench.clone())
                applied = env.cart.permanent_wrench_composer.set_forces_and_torques_index.call_args.kwargs
                # COM offset .1m contributes -1Nm in addition to the -2Nm event couple.
                expected_mz = -3. if len(trace) <= 5 else 0.
                self.assertAlmostEqual(applied["torques"][1, 0, 2].item(), expected_mz)
        trace = torch.stack(trace)
        torch.testing.assert_close(trace[:, 0], torch.zeros(8, 6))
        torch.testing.assert_close(trace[:5, 1], torch.tensor([10., 0., 0., 0., 0., -2.]).expand(5, -1))
        torch.testing.assert_close(trace[5:], torch.zeros(3, 2, 6))
        torch.testing.assert_close(env.cart_pulse_impulse[1], torch.tensor([.25, 0., 0., 0., 0., -.05]))

    def test_invalid_input_does_not_schedule_a_pulse(self):
        env = adapter()
        for options in (dict(env_ids=[-1]), dict(env_ids=[2]), dict(env_ids=[1, 1]),
                        dict(env_ids=[.5]), dict(offset_w=[float("nan"), 0., 0.]), dict(offset_w=[0., 0.])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                env.inject_cart_wrench([1., 0., 0.], [0., 0., 0.], .1, **options)
        self.assertEqual(env.cart_pulse_timer.count_nonzero().item(), 0)

    def test_stalled_wheel_releases_after_curriculum_returns_to_zero(self):
        env = adapter()
        env.level_params = SimpleNamespace(disturbance_frac=0.)
        env.task_cfg = SimpleNamespace(randomization=SimpleNamespace(cart_force_pulse=False))
        env.cart_push_timer = torch.ones(2, dtype=torch.long)
        env.stall_timer = torch.tensor([1, 0])
        env.wheel_damping = torch.tensor([.03, .04])
        env.wheel_joint_ids_sim = torch.arange(4, dtype=torch.int32)
        env._apply_cart_disturbances()
        env.cart.write_joint_damping_to_sim_index.assert_called_once()
        args = env.cart.write_joint_damping_to_sim_index.call_args.kwargs
        torch.testing.assert_close(args["damping"], torch.full((1, 4), .03))
        self.assertEqual(env.stall_timer.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
