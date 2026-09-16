"""CPU checks for policy semantics that must survive the simulator migration."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))

from homie_isaaclab.config import G1RoughCfg
from homie_isaaclab.math_utils import quat_apply, quat_mul, quat_rotate_inverse
from homie_isaaclab.task_core import HomieTaskCore
from homie_rsl_rl.algorithms import HIMPPO

class PolicyCompatibilityTests(unittest.TestCase):
    def test_timeout_matches_twenty_seconds_at_fifty_hz(self):
        task = HomieTaskCore()
        task.max_episode_length = 1000
        task.episode_length_buf = torch.tensor([999, 1000])
        task.contact_forces = torch.zeros(2, 1, 3)
        task.termination_contact_indices = torch.tensor([0])
        task.projected_gravity = torch.tensor([[0., 0., -1.], [0., 0., -1.]])
        task.check_termination()
        self.assertEqual(task.time_out_buf.tolist(), [False, True])

    def test_xyzw_rotation_and_inverse(self):
        q = torch.tensor([[0., 0., math.sqrt(.5), math.sqrt(.5)]])
        v = torch.tensor([[1., 0., 0.]])
        rotated = quat_apply(q, v)
        torch.testing.assert_close(rotated, torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(quat_rotate_inverse(q, rotated), v)

    def test_virtual_link_rotation_composition(self):
        q = torch.tensor([[0., 0., math.sqrt(.5), math.sqrt(.5)]])
        marker = torch.tensor([[.12, 0., -.04]])
        expected = torch.tensor([[-.12, 0., -.04]])
        torch.testing.assert_close(quat_apply(quat_mul(q, q), marker), expected, atol=1e-6, rtol=0)

    def test_urdf_matches_him_symmetry_joint_order(self):
        urdf = ET.parse(ROOT / 'HomieRL/legged_gym/resources/robots/g1_description/g1.urdf').getroot()
        names = [j.get('name') for j in urdf.findall('joint') if j.get('type') == 'revolute']
        leg = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
        arm = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
        expected = [f'{side}_{joint}_joint' for side in ['left', 'right'] for joint in leg]
        expected += ['waist_yaw_joint']
        expected += [f'{side}_{joint}_joint' for side in ['left', 'right'] for joint in arm]
        self.assertEqual(names, expected)
        cfg = G1RoughCfg()
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions), (456, 79, 12))

    def test_symmetry_is_involution_for_all_policy_inputs(self):
        algorithm = HIMPPO.__new__(HIMPPO)
        algorithm.actor_critic = SimpleNamespace(num_one_step_obs=76, actor_history_length=6,
                                                 num_one_step_critic_obs=79, critic_history_length=1)
        for size, flip in [(456, algorithm.flip_g1_actor_obs), (79, algorithm.flip_g1_critic_obs),
                           (12, algorithm.flip_g1_actions)]:
            values = torch.arange(2 * size, dtype=torch.float32).reshape(2, size) + 1
            torch.testing.assert_close(flip(flip(values)), values)


if __name__ == '__main__':
    unittest.main()
