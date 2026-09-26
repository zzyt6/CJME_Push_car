"""V5.2 interval-mean hand labels on the real CPU measurement path (no Kit app)."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'HomieRL'), str(ROOT / 'HomieRL/rsl_rl')]
from homie_pushcart_v52.sim_env import PushCartSimEnv


def proxy(value):
    return SimpleNamespace(torch=value)


def environment(label_substep_mean=True):
    env = PushCartSimEnv.__new__(PushCartSimEnv)
    env.num_envs, env.device, env.with_cart, env.cart = 2, 'cpu', False, None
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=.005, gravity=(0., 0., 0.)),
                              asset=SimpleNamespace(disable_gravity=True), control=SimpleNamespace(decimation=4))
    env.task_cfg = SimpleNamespace(diagnostics=SimpleNamespace(full_dynamics=False),
                                   termination=SimpleNamespace(blowup_force_n=1e6),
                                   interaction=SimpleNamespace(label_substep_mean=label_substep_mean))
    names = ['pelvis', 'left_hand_palm_link', 'right_hand_palm_link']
    n, b = 2, len(names)
    pose = torch.zeros(n, b, 7)
    pose[..., 6] = 1.
    pose[:, 1, :3] = torch.tensor([.5, .2, .8])
    pose[:, 2, :3] = torch.tensor([.5, -.2, .8])
    zeros6 = torch.zeros(n, b, 6)
    data = SimpleNamespace(body_mass=proxy(torch.tensor([10., .4, .4]).expand(n, -1).clone()),
                           body_com_pose_w=proxy(pose.clone()), body_link_pose_w=proxy(pose),
                           body_com_vel_w=proxy(zeros6.clone()), body_com_acc_w=proxy(zeros6.clone()),
                           body_inertia=proxy(torch.eye(3).reshape(1, 1, 9).expand(n, b, -1).clone() * .01),
                           root_link_pose_w=proxy(pose[:, 0]), root_com_vel_w=proxy(zeros6[:, 0]),
                           joint_pos=proxy(torch.zeros(n, 3)), joint_vel=proxy(torch.zeros(n, 3)))
    env.robot = SimpleNamespace(body_names=names, num_bodies=b, data=data)
    env.palm_body_ids = torch.tensor([1, 2])
    env._init_dynamics(ET.fromstring('<robot></robot>'))
    state = {'incoming': torch.zeros(n, 2, 6)}
    env._incoming_hand_wrench = lambda: state['incoming']
    return env, state


class IntervalMeanTests(unittest.TestCase):
    def run_interval(self, env, state, forces):
        env.hand_wrench_sum.zero_()
        env.hand_wrench_count.zero_()
        for value in forces:
            state['incoming'] = torch.zeros(2, 2, 6)
            state['incoming'][:, 0, 0] = value
            env._sample_dynamics(0)

    def test_mean_of_valid_substeps_differs_from_last_sample(self):
        env, state = environment()
        self.run_interval(env, state, [0., 0.])        # reset warm-up (2 physics steps)
        self.run_interval(env, state, [10., 20., 30., 100.])
        force, moment, points, count = env.grasp_wrench_interval_mean()
        torch.testing.assert_close(count, torch.full((2,), 4.))
        torch.testing.assert_close(force[:, 0, 0], torch.full((2,), 40.))
        torch.testing.assert_close(env._hand_wrench[:, 0, 0], torch.full((2,), 100.))   # rewards still see last
        torch.testing.assert_close(points[:, 0], torch.tensor([[.5, .2, .8]] * 2))

    def test_warmup_samples_excluded_and_reset_clears(self):
        env, state = environment()
        self.run_interval(env, state, [50., 10., 20., 30.])   # first sample invalid (warm-up)
        force, _, _, count = env.grasp_wrench_interval_mean()
        torch.testing.assert_close(count, torch.full((2,), 3.))
        torch.testing.assert_close(force[:, 0, 0], torch.full((2,), 20.))
        env._reset_dynamics(torch.tensor([0]))
        self.assertEqual(env.hand_wrench_count[0].item(), 0.)
        self.assertEqual(env.hand_wrench_count[1].item(), 3.)

    def test_disabled_falls_back_to_latest_sample(self):
        env, state = environment(label_substep_mean=False)
        self.run_interval(env, state, [0., 0., 10., 30.])
        force, _, _, count = env.grasp_wrench_interval_mean()
        torch.testing.assert_close(force[:, 0, 0], torch.full((2,), 30.))
        torch.testing.assert_close(count, torch.ones(2))


if __name__ == '__main__':
    unittest.main()
