"""Dynamics property caching preserves frames, randomization and fault detection."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_pushcart_v51_fast_dynamics import hand_environment, proxy
from homie_pushcart_v51.sim_env import PushCartSimEnv
from homie_pushcart_v51.dynamics import rotation_matrix_xyzw


class CountingData:
    def __init__(self, source):
        self.values = vars(source).copy()
        self.reads = Counter()

    def __getattr__(self, name):
        self.reads[name] += 1
        return self.values[name]


def cached_environment():
    env, target, incoming = hand_environment()
    data = env.robot.data
    local = torch.zeros_like(data.body_link_pose_w.torch)
    local[..., 6] = 1.
    data.body_com_pose_b = proxy(local)
    env.robot.data = CountingData(data)
    env.refresh_dynamics_properties(env.robot)
    return env, target, incoming


def update_world_com(data):
    """Independent link-pose plus local-offset reference used by full audit."""
    pose = data.values['body_link_pose_w'].torch
    local = data.values['body_com_pose_b'].torch
    rotation = rotation_matrix_xyzw(pose[..., 3:])
    data.values['body_com_pose_w'].torch[..., :3] = (
        pose[..., :3] + torch.einsum('nbij,nbj->nbi', rotation, local[..., :3]))


class DynamicsPropertyCacheTests(unittest.TestCase):
    def test_selected_hotpath_does_not_fetch_static_or_whole_body_com_properties(self):
        env, _, _ = cached_environment()
        data = env.robot.data
        before = data.reads.copy()
        for _ in range(3):
            cached = env._body_dynamics(env.robot, env._hand_body_ids)
        for name in ('body_mass', 'body_inertia', 'body_com_pose_b', 'body_com_pose_w'):
            self.assertEqual(data.reads[name], before[name], name)
        full = env._body_dynamics(env.robot)
        for selected, reference in zip(cached, full):
            torch.testing.assert_close(selected, reference[:, env._hand_body_ids])
        self.assertGreater(data.reads['body_mass'], before['body_mass'])
        self.assertGreater(data.reads['body_inertia'], before['body_inertia'])

    def test_partial_refresh_updates_mass_inertia_com_and_does_not_alias_inputs(self):
        env, _, _ = cached_environment()
        data, ids = env.robot.data, torch.tensor([1])
        unchanged = tuple(x[0].clone() for x in env._body_dynamics(env.robot, env._hand_body_ids))
        mass = data.values['body_mass'].torch[ids].clone() * 1.7
        inertia = data.values['body_inertia'].torch[ids].clone() * 2.3
        com = data.values['body_com_pose_b'].torch[ids].clone()
        com[..., :3] = torch.tensor([.07, -.04, .02])
        # COM principal-axis quaternion must not rotate the prim-frame inertia twice.
        com[..., 3:] = torch.tensor([.70710678, 0., 0., .70710678])
        for name, value in (('body_mass', mass), ('body_inertia', inertia), ('body_com_pose_b', com)):
            data.values[name].torch[ids] = value
        update_world_com(data)
        before = data.reads.copy()
        env.refresh_dynamics_properties(env.robot, ids, masses=mass, inertias=inertia, coms=com)
        self.assertEqual(data.reads, before)
        mass.zero_(); inertia.zero_(); com.zero_()
        selected = env._body_dynamics(env.robot, env._hand_body_ids)
        full = env._body_dynamics(env.robot)
        for current, reference, original in zip(selected, full, unchanged):
            torch.testing.assert_close(current, reference[:, env._hand_body_ids])
            torch.testing.assert_close(current[0], original)
        # Compare the resulting hand labels, not just cached parameter arrays.
        measured, _, _ = env._measure_hand_wrench(
            *full, data.values['body_com_acc_w'].torch, env._hand_subtree_ids)
        env._sample_dynamics(0)
        env._sample_dynamics(1)
        torch.testing.assert_close(env._hand_wrench, measured)

    def test_refresh_without_provided_values_reads_updated_simulator_properties(self):
        env, _, _ = cached_environment()
        data, ids = env.robot.data, torch.tensor([0])
        data.values['body_mass'].torch[0] *= 1.2
        data.values['body_com_pose_b'].torch[0, :, :3] = torch.tensor([.02, -.03, .04])
        data.values['body_inertia'].torch[0] *= .8
        update_world_com(data)
        env.refresh_dynamics_properties(env.robot, ids)
        selected = env._body_dynamics(env.robot, env._hand_body_ids)
        full = env._body_dynamics(env.robot)
        for actual, expected in zip(selected, full):
            torch.testing.assert_close(actual, expected[:, env._hand_body_ids])

    def test_cart_com_changes_after_selected_payload_randomization_and_live_motion(self):
        env, _, _ = cached_environment()
        source = SimpleNamespace(**{name: proxy(value.torch.clone())
                                   for name, value in env.robot.data.values.items()})
        env.cart = SimpleNamespace(data=CountingData(source))
        env.refresh_dynamics_properties(env.cart)
        data = env.cart.data
        before = env.cart_com_world().clone()
        data.values['body_mass'].torch[1, 2] += 10.
        data.values['body_com_pose_b'].torch[1, 2, :3] = torch.tensor([.1, .2, -.1])
        env.refresh_dynamics_properties(env.cart, torch.tensor([1]))
        data.values['body_link_pose_w'].torch[1, :, :3] += torch.tensor([.3, .2, .1])
        update_world_com(data)
        reads = data.reads.copy()
        actual = env.cart_com_world()
        mass = data.values['body_mass'].torch
        expected = (mass[..., None] * data.values['body_com_pose_w'].torch[..., :3]).sum(1) / mass.sum(1)[:, None]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[0], before[0])
        for name in ('body_mass', 'body_com_pose_b', 'body_com_pose_w'):
            self.assertEqual(data.reads[name], reads[name], name)

    def test_robot_randomization_hook_refreshes_after_actual_setters(self):
        env, _, _ = cached_environment()
        def randomize(instance):
            data = instance.robot.data
            data.values['body_mass'].torch *= 1.4
            data.values['body_inertia'].torch *= 1.4
            data.values['body_com_pose_b'].torch[..., :3] = .02
            update_world_com(data)
        with patch('homie_isaaclab.env.HomieIsaacLabEnv._randomize_body_properties', randomize):
            env._randomize_body_properties()
        selected = env._body_dynamics(env.robot, env._hand_body_ids)
        full = env._body_dynamics(env.robot)
        for actual, expected in zip(selected, full):
            torch.testing.assert_close(actual, expected[:, env._hand_body_ids])

    def test_cached_sampling_preserves_200hz_peak_and_unrelated_joint_fault(self):
        env, target, incoming = cached_environment()
        env._sample_dynamics(0)
        original = incoming.clone()
        incoming[0, 0, 0] += 150.
        env._sample_dynamics(1)
        incoming.copy_(original)
        env._sample_dynamics(2)
        torch.testing.assert_close(env._hand_wrench, target, atol=2e-6, rtol=2e-6)
        self.assertTrue(env.dynamics_fault[0])
        self.assertGreater(env.hand_force_peak[0, 0], 150.)
        env.robot.data.values['joint_pos'].torch[1, 0] = float('nan')
        env._sample_dynamics(3)
        self.assertTrue(env.dynamics_fault[1])
        self.assertFalse(env.dynamics_wrench_valid[1])
        self.assertTrue(torch.isfinite(env._hand_wrench).all())


if __name__ == '__main__':
    unittest.main()
