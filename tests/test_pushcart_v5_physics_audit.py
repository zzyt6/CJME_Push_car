"""Adversarial backend checks: whole-system state, terminal data, and reset safety."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'HomieRL'), str(ROOT/'HomieRL/rsl_rl')]
from homie_pushcart_v5.sim_env import PushCartSimEnv
from homie_pushcart_v5.mujoco_env import MujocoHomieEnv
from homie_pushcart_v5.dynamics import shift_wrench


class IsaacBackendAuditTests(unittest.TestCase):
    def test_cart_state_uses_whole_cart_com_velocity_not_base_com(self):
        env = PushCartSimEnv.__new__(PushCartSimEnv)
        proxy = lambda x: SimpleNamespace(torch=torch.tensor(x, dtype=torch.float32))
        env.cart = SimpleNamespace(data=SimpleNamespace(
            root_link_pose_w=proxy([[2., 3., 4., 0., 0., 0., 1.]]),
            root_com_vel_w=proxy([[100., 0., 0., 0., 0., .2]]),
            body_mass=proxy([[12., 2.]]), body_com_vel_w=proxy([[[1., 0., 0., 0., 0., .2],
                                                              [0., 7., 0., 0., 2., 0.]]])))
        pos, _, linear, angular = env.cart_state()
        torch.testing.assert_close(pos, torch.tensor([[2., 3., 4.]]))
        torch.testing.assert_close(linear, torch.tensor([[12./14., 1., 0.]]))
        torch.testing.assert_close(angular, torch.tensor([[0., 0., .2]]))

    def test_terminal_snapshot_survives_selected_reset_without_cloning_other_envs(self):
        env = PushCartSimEnv.__new__(PushCartSimEnv)
        env.physics_samples = {'hand_wrench': torch.arange(4*3*2*6).reshape(4, 3, 2, 6).float(),
                               'fault': torch.tensor([[False, True, False]]).expand(4, -1).clone()}
        env.physics_impulses = {'hand_wrench': torch.ones(3, 2, 6)}
        env.physics_fixed_origin_impulses = {'cart_hand_wrench': torch.ones(3, 6)}
        env.physics_impulse_reference = torch.randn(3, 2, 3)
        env.physics_valid_duration = torch.ones(3)*.02
        snap = env._snapshot_terminal_physics(torch.tensor([1]))
        original = snap['samples']['hand_wrench'].clone()
        env.physics_samples['hand_wrench'][:, 1] = 0.
        env.physics_impulses['hand_wrench'][1] = 0.
        torch.testing.assert_close(snap['samples']['hand_wrench'], original)
        self.assertEqual(snap['samples']['hand_wrench'].shape, (4, 1, 2, 6))
        self.assertTrue(snap['samples']['fault'].all())
        self.assertEqual(snap['impulses']['hand_wrench'].sum().item(), 12.)
        self.assertIsNone(env._snapshot_terminal_physics(torch.tensor([], dtype=torch.long)))

    def test_fixed_origin_angular_impulse_includes_moving_reference_lever(self):
        wrench = torch.tensor([[10., 0., 0., 0., 0., 0.], [10., 0., 0., 0., 0., 0.]])
        source = torch.tensor([[0., 0., 0.], [0., .2, 0.]])
        fixed = shift_wrench(wrench, source, torch.zeros_like(source)).sum(0)*.005
        torch.testing.assert_close(fixed, torch.tensor([.1, 0., 0., 0., 0., -.01]))
        self.assertNotEqual(fixed[-1].item(), (wrench.sum(0)*.005)[-1].item())


class MujocoBackendAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MujocoHomieEnv(True)

    def setUp(self):
        self.env.reset()

    def test_centroidal_matches_engine_independent_subtree_calculation(self):
        env = self.env
        env.data.qvel[:] = np.random.default_rng(7).normal(size=env.model.nv)
        mujoco.mj_forward(env.model, env.data)
        mujoco.mj_subtreeVel(env.model, env.data)
        for body in (env.pelvis_id, env.cart_id):
            state = env.centroidal_state(body)
            np.testing.assert_allclose(state['velocity'], env.data.subtree_linvel[body], atol=1.e-12)
            np.testing.assert_allclose(state['angular_momentum'], env.data.subtree_angmom[body], atol=1.e-11)

    def test_lowlevel_reset_removes_command_and_arm_correction(self):
        env = self.env
        env.set_command(.5, -.2, .3)
        env.upper_delta[:] = .2
        obs = env.reset()
        np.testing.assert_allclose(env.commands[:4], 0.)
        np.testing.assert_allclose(env.upper_delta, 0.)
        np.testing.assert_allclose(obs.reshape(env.history_len, -1)[-1, :3], 0.)

    def test_interval_applied_torque_can_be_shifted_to_sampled_com(self):
        env = self.env
        env.inject_cart_wrench([20., 7., 0.], [0., 0., 1.], .025, offset=[0., .2, 0.])
        env.step(np.zeros(env.num_lower))
        for sample in env.physics_samples:
            w = sample['cart_applied_wrench']
            expected = w[3:]+np.cross(sample['cart_applied_reference']-sample['cart_com'], w[:3])
            np.testing.assert_allclose(sample['cart_applied_wrench_com'][3:], expected)
            self.assertIn('ground_valid', sample)


if __name__ == '__main__':
    unittest.main()
