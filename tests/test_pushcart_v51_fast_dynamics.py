"""CPU tests of the real V5.1 measurement path; no Kit app or scene launched."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'HomieRL'), str(ROOT / 'HomieRL/rsl_rl')]
from homie_pushcart_v51.sim_env import PushCartSimEnv
from homie_pushcart_v51.dynamics import rotation_matrix_xyzw


def proxy(value):
    return SimpleNamespace(torch=value)


def hand_environment():
    """Moving anisotropic palm/finger subtrees embedded in an unrelated robot."""
    env = PushCartSimEnv.__new__(PushCartSimEnv)
    env.num_envs, env.device, env.with_cart, env.cart = 2, 'cpu', False, None
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005, gravity=(0., 0., -9.81)),
                              asset=SimpleNamespace(disable_gravity=False),
                              control=SimpleNamespace(decimation=4))
    env.task_cfg = SimpleNamespace(diagnostics=SimpleNamespace(full_dynamics=False),
                                   termination=SimpleNamespace(blowup_force_n=100.))
    names = ['pelvis', 'left_hand_palm_link', 'left_finger', 'right_hand_palm_link', 'right_finger']
    n, b = 2, len(names)
    pose = torch.zeros(n, b, 7)
    pose[..., 6] = 1.
    pose[:, 1, :3] = torch.tensor([.5, .2, .8])
    pose[:, 2, :3] = torch.tensor([.6, .2, .8])
    pose[:, 3, :3] = torch.tensor([.5, -.2, .8])
    pose[:, 4, :3] = torch.tensor([.6, -.2, .8])
    pose[1, :, 5:] = torch.tensor([.70710678, .70710678])
    velocity = torch.zeros(n, b, 6)
    velocity[..., :3] = torch.tensor([.3, -.2, .1])
    velocity[..., 3:] = torch.tensor([.4, -.3, .5])
    acceleration = torch.zeros(n, b, 6)
    acceleration[..., :3] = torch.tensor([1., -2., .5])
    acceleration[..., 3:] = torch.tensor([.2, .3, -.1])
    inertia = torch.diag(torch.tensor([.02, .03, .04])).reshape(1, 1, 9).expand(n, b, -1).clone()
    masses = torch.tensor([10., .4, .1, .4, .1]).expand(n, -1).clone()
    data = SimpleNamespace(body_mass=proxy(masses), body_com_pose_w=proxy(pose.clone()),
                           body_link_pose_w=proxy(pose), body_com_vel_w=proxy(velocity),
                           body_com_acc_w=proxy(acceleration), body_inertia=proxy(inertia),
                           root_link_pose_w=proxy(pose[:, 0]), root_com_vel_w=proxy(velocity[:, 0]),
                           joint_pos=proxy(torch.zeros(n, 3)), joint_vel=proxy(torch.zeros(n, 3)))
    env.robot = SimpleNamespace(body_names=names, num_bodies=b, data=data)
    env.palm_body_ids = torch.tensor([1, 3])
    xml = ET.fromstring('<robot><joint><parent link="left_hand_palm_link"/><child link="left_finger"/></joint>'
                        '<joint><parent link="right_hand_palm_link"/><child link="right_finger"/></joint></robot>')
    env._init_dynamics(xml)
    env.centroidal_state = lambda *_: (_ for _ in ()).throw(AssertionError('whole body audit in train path'))
    env._contact_wrench = lambda *_: (_ for _ in ()).throw(AssertionError('ground audit in train path'))
    target = torch.tensor([[[3., -2., 1., .3, -.2, .1], [4., 1., -2., -.1, .3, .2]]]).expand(n, -1, -1).clone()
    rotations = rotation_matrix_xyzw(pose[..., 3:])
    inertia_world = rotations @ inertia.reshape(n, b, 3, 3) @ rotations.transpose(-1, -2)
    incoming_world = target.clone()
    for side, indices in enumerate(([1, 2], [3, 4])):
        forces = masses[:, indices, None] * (acceleration[:, indices, :3]-torch.tensor([0., 0., -9.81]))
        spin = (inertia_world[:, indices] @ velocity[:, indices, 3:, None]).squeeze(-1)
        spin_rate = (inertia_world[:, indices] @ acceleration[:, indices, 3:, None]).squeeze(-1)
        spin_rate += torch.cross(velocity[:, indices, 3:], spin, dim=-1)
        arms = pose[:, indices, :3]-pose[:, indices[0], None, :3]
        incoming_world[:, side, :3] += forces.sum(1)
        incoming_world[:, side, 3:] += (spin_rate+torch.cross(arms, forces, dim=-1)).sum(1)
    palm_rotation = rotations[:, env.palm_body_ids]
    incoming = torch.cat(((palm_rotation.transpose(-1, -2) @ incoming_world[..., :3, None]).squeeze(-1),
                          (palm_rotation.transpose(-1, -2) @ incoming_world[..., 3:, None]).squeeze(-1)), -1)
    env._incoming_hand_wrench = lambda: incoming
    return env, target, incoming


class FastDynamicsTests(unittest.TestCase):
    def test_local_dynamic_compensation_and_warmup_without_ground_or_whole_body(self):
        env, target, _ = hand_environment()
        self.assertEqual(env.physics_samples, {})
        self.assertEqual(env.physics_impulses, {})
        self.assertEqual(env._ground_contact_views, [])
        self.assertFalse(env.dynamics_metadata['dynamic_contact_wrench_calibrated'])
        env._sample_dynamics(0)
        self.assertFalse(env.dynamics_wrench_valid.any())
        env._sample_dynamics(1)
        torch.testing.assert_close(env._hand_wrench, target, atol=2e-6, rtol=2e-6)
        self.assertTrue(env.dynamics_wrench_valid.all())
        self.assertFalse(env.ground_wrench_valid.any())
        self.assertFalse(env.dynamics_fault.any())
        self.assertIsNone(env._snapshot_terminal_physics(torch.tensor([0])))

    def test_local_subset_matches_full_subtree_calculation(self):
        env, _, _ = hand_environment()
        mass, position, velocity, inertia = env._body_dynamics(env.robot)
        full, _, _ = env._measure_hand_wrench(mass, position, velocity, inertia,
            env.robot.data.body_com_acc_w.torch, env._hand_subtree_ids)
        env._sample_dynamics(0)
        env._sample_dynamics(1)
        torch.testing.assert_close(env._hand_wrench, full)

    def test_transient_peak_latches_fault_and_selected_reset_clears_only_selected(self):
        env, target, incoming = hand_environment()
        env._sample_dynamics(0)
        saved = incoming.clone()
        incoming[0, 0, 0] += 150.
        env._sample_dynamics(1)
        incoming.copy_(saved)
        env._sample_dynamics(2)
        torch.testing.assert_close(env._hand_wrench, target, atol=2e-6, rtol=2e-6)
        self.assertGreater(env.hand_force_peak[0, 0], 150.)
        self.assertTrue(env.dynamics_fault[0])
        self.assertFalse(env.dynamics_fault[1])
        other = env._hand_wrench[1].clone()
        env._reset_dynamics(torch.tensor([0]))
        self.assertFalse(env.dynamics_wrench_valid[0])
        self.assertFalse(env.dynamics_fault[0])
        self.assertEqual(env.hand_force_peak[0].sum(), 0.)
        torch.testing.assert_close(env._hand_wrench[1], other)

    def test_nonfinite_unrelated_joint_is_fault_not_valid_zero_measurement(self):
        env, _, _ = hand_environment()
        env._sample_dynamics(0)
        env.robot.data.joint_pos.torch[0, 0] = float('nan')
        env._sample_dynamics(1)
        self.assertTrue(env.dynamics_fault[0])
        self.assertFalse(env.dynamics_wrench_valid[0])
        self.assertTrue(torch.isfinite(env._hand_wrench).all())
        self.assertTrue(env.dynamics_wrench_valid[1])

    def test_nonfinite_hand_acceleration_cannot_be_hidden_by_ground_disabled(self):
        env, _, _ = hand_environment()
        env._sample_dynamics(0)
        env.robot.data.body_com_acc_w.torch[0, 2, 0] = float('nan')
        env._sample_dynamics(1)
        self.assertTrue(env.dynamics_fault[0])
        self.assertFalse(env.dynamics_wrench_valid[0])
        self.assertFalse(env.ground_wrench_valid.any())

    def test_full_audit_remains_available_and_ground_invalidity_is_separate(self):
        env, target, _ = hand_environment()
        env.task_cfg.diagnostics.full_dynamics = True
        env.contacts = [SimpleNamespace(cfg=SimpleNamespace(prim_path=f'/World/envs/env_.*/foot_{i}'))
                        for i in range(2)]
        view = SimpleNamespace(sensor_count=env.num_envs, filter_count=1)
        physics_view = SimpleNamespace(create_rigid_contact_view=lambda *a, **k: view)
        xml = ET.fromstring('<robot><joint><parent link="left_hand_palm_link"/><child link="left_finger"/></joint>'
                            '<joint><parent link="right_hand_palm_link"/><child link="right_finger"/></joint></robot>')
        backend = SimpleNamespace(PhysxManager=SimpleNamespace(get_physics_sim_view=lambda: physics_view))
        with patch.dict(sys.modules, {'isaaclab_physx.physics': backend}):
            env._init_dynamics(xml)
        env.centroidal_state = PushCartSimEnv.centroidal_state.__get__(env)
        env._contact_wrench = lambda *a: (torch.zeros(2, 6), torch.zeros(2, dtype=torch.bool))
        self.assertTrue(env.full_dynamics)
        self.assertIn('robot_momentum', env.physics_samples)
        env._sample_dynamics(0)
        env._sample_dynamics(1)
        torch.testing.assert_close(env._hand_wrench, target, atol=2e-6, rtol=2e-6)
        self.assertTrue(env.dynamics_wrench_valid.all())
        self.assertFalse(env.ground_wrench_valid.any())
        snap = env._snapshot_terminal_physics(torch.tensor([0]))
        self.assertEqual(snap['samples']['hand_wrench'].shape, (4, 1, 2, 6))
        self.assertTrue(torch.isfinite(snap['samples']['robot_momentum']).all())


class GeometricReferenceTests(unittest.TestCase):
    def test_root_com_velocity_shift_uses_root_body_not_cart_assembly(self):
        env = PushCartSimEnv.__new__(PushCartSimEnv)
        env.cart = SimpleNamespace(data=SimpleNamespace(
            root_link_pose_w=proxy(torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])),
            root_com_pose_w=proxy(torch.tensor([[0., 1., 0., 0., 0., 0., 1.]])),
            root_com_vel_w=proxy(torch.tensor([[1., 0., 0., 0., 0., 2.]]))))
        position, _, linear, omega = env.cart_state()
        torch.testing.assert_close(position, torch.zeros(1, 3))
        torch.testing.assert_close(linear, torch.tensor([[3., 0., 0.]]))
        torch.testing.assert_close(omega, torch.tensor([[0., 0., 2.]]))

    def test_cart_wrench_defaults_to_handle_center_and_can_select_com_for_audit(self):
        env = PushCartSimEnv.__new__(PushCartSimEnv)
        env.cart = SimpleNamespace(data=SimpleNamespace(
            root_link_pose_w=proxy(torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]))))
        env.bar_local = torch.tensor([0., 1., 0.])
        env.palm_body_ids = torch.tensor([0, 1])
        env.rigid_body_states = torch.zeros(1, 2, 13)
        env.cart_com_world = lambda: torch.zeros(1, 3)
        force = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]])
        total, moment, point = env.cart_hand_wrench(force, torch.zeros_like(force))
        torch.testing.assert_close(total, torch.tensor([[2., 0., 0.]]))
        torch.testing.assert_close(moment, torch.tensor([[0., 0., 2.]]))
        torch.testing.assert_close(point, torch.tensor([[0., 1., 0.]]))
        _, audit_moment, _ = env.cart_hand_wrench(force, torch.zeros_like(force), reference='cart_com')
        torch.testing.assert_close(audit_moment, torch.zeros(1, 3))


if __name__ == '__main__':
    unittest.main()
