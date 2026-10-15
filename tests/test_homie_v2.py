"""HOMIE v2 keeps the baseline training contract except for the recorded port-vs-official fixes."""
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))

from homie_isaaclab import register_tasks as register_baseline
from homie_isaaclab.config import G1RoughCfg, G1RoughCfgPPO
from homie_isaaclab.math_utils import class_to_dict
from homie_isaaclab_v2 import register_tasks
from homie_isaaclab_v2.config import G1RoughCfg as V2Cfg, G1RoughCfgPPO as V2PPO
from homie_isaaclab_v2.env import HomieIsaacLabEnv
from homie_isaaclab_v2.lab_config import HomieRunnerCfg
from homie_isaaclab_v2.task_core import HomieTaskCore
from homie_isaaclab_v2.training import train

# Default masses [kg] from g1.urdf: pelvis, a leg link, torso_link, both palms.
BODIES = ['pelvis', 'left_hip_pitch_link', 'torso_link', 'left_hand_palm_link', 'right_hand_palm_link']
DEFAULT = torch.tensor([3.813, 1.35, 6.78, .37283854, .37283854])


class _Robot:
    def __init__(self, n):
        b = len(BODIES)
        self.data = SimpleNamespace(
            body_mass=SimpleNamespace(torch=DEFAULT.repeat(n, 1)),
            body_inertia=SimpleNamespace(torch=torch.ones(n, b, 9)),
            body_com_pose_b=SimpleNamespace(torch=torch.zeros(n, b, 7)))

    def set_masses_index(self, masses):
        self.masses = masses

    def set_inertias_index(self, inertias):
        self.inertias = inertias

    def set_coms_index(self, coms):
        self.coms = coms


def _mass_env(n=4096):
    env = object.__new__(HomieIsaacLabEnv)
    env.cfg, env.device, env.robot = V2Cfg(), 'cpu', _Robot(n)
    env.body_names, env.base_body_index, env.torso_body_index = BODIES, 0, 2
    env.payload, env.hand_payload = torch.full((n, 1), 10.), torch.full((n, 2), .3)
    env.body_displacement = torch.zeros(n, 3)
    return env


class HomieV2Tests(unittest.TestCase):
    def test_walking_config_differs_from_baseline_only_in_payload_switch(self):
        base, v2 = class_to_dict(G1RoughCfg()), class_to_dict(V2Cfg())
        self.assertTrue(base['domain_rand']['randomize_payload_mass'])
        self.assertFalse(v2['domain_rand']['randomize_payload_mass'])
        v2['domain_rand']['randomize_payload_mass'] = True
        # Balance-only additions are inert while walking.
        self.assertFalse(v2.pop('balance')['enabled'])
        for name in ('feet_still', 'stance_height', 'com_centering', 'load_balance', 'pitch_band'):
            self.assertEqual(v2['rewards']['scales'].pop(name), 0.)
        self.assertEqual(base, v2)
        self.assertEqual(class_to_dict(G1RoughCfgPPO()), class_to_dict(V2PPO()))

    def test_balance_task_config(self):
        from homie_isaaclab_v2.lab_config import HomieBalanceEnvCfg, HomieBalanceRunnerCfg, HomieDirectEnvCfg
        self.assertTrue(HomieBalanceEnvCfg().homie['balance']['enabled'])
        self.assertFalse(HomieDirectEnvCfg().homie['balance']['enabled'])
        self.assertEqual(HomieBalanceRunnerCfg().experiment_name, "homie_g1_v2_balance")
        b = V2Cfg().balance
        self.assertAlmostEqual(b.max_push_vel_xy, .2)   # capture-point limit without a step
        self.assertEqual(V2Cfg().rewards.base_height_target, .74)   # same height definition as walking

    def test_balance_commands_stay_zero_at_walking_height(self):
        task = HomieTaskCore()
        task.cfg, task.device = V2Cfg(), 'cpu'
        task.cfg.balance.enabled = True
        task.commands = torch.ones(4, 5)
        task._resample_commands(torch.tensor([0, 2]))
        torch.testing.assert_close(task.commands[[0, 2]], torch.tensor([[0., 0, 0, 0, .74]] * 2))
        self.assertTrue(torch.all(task.commands[[1, 3]] == 1.))

    def test_feet_still_penalizes_any_foot_motion(self):
        task = HomieTaskCore()
        task.feet_vel = torch.tensor([[[0., 0, 0], [0, 0, 0]], [[.3, .4, 0], [0, 0, 1.]]])
        torch.testing.assert_close(task._reward_feet_still(), torch.tensor([0., 1.5]))

    def test_balance_candidate_rewards(self):
        env = object.__new__(HomieIsaacLabEnv)
        env.cfg = V2Cfg()
        # Absolute pelvis height of the grasp stance; 5 mm off: exp(-1).
        env._stance_root_pose = torch.tensor([0., 0, .759214, 0, .009508, 0, .999955])
        env.root_states = torch.tensor([[0., 0, .759214], [0, 0, .754214]])
        torch.testing.assert_close(env._reward_stance_height(), torch.tensor([1., math.exp(-1.)]), atol=1e-4, rtol=0)
        env.feet_indices = torch.tensor([0, 1])
        env.contact_forces = torch.tensor([[[0., 0, 300], [0, 0, 300]], [[0., 0, 450], [0, 0, 150]]])
        torch.testing.assert_close(env._reward_load_balance(), torch.tensor([0., .5]))
        env.contact_forces[1, 1, 2] = 0.            # one foot unloaded: that is a step, not an imbalance
        self.assertEqual(env._reward_load_balance()[1].item(), 0.)
        # Two bodies of 1 and 3 kg -> CoM x = 0.075; feet markers centred at x = 0.02.
        env._body_mass = torch.tensor([[1., 3.]])
        env.robot = SimpleNamespace(data=SimpleNamespace(body_com_pos_w=SimpleNamespace(
            torch=torch.tensor([[[0., 0, .7], [.1, 0, .9]]]))))
        env.left_foot_indices, env.right_foot_indices = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
        markers = torch.tensor([.12, 0., -.06])
        env.rigid_body_states = torch.zeros(1, 6, 13)
        env.rigid_body_states[0, :3, 0] = env.rigid_body_states[0, 3:, 0] = markers
        torch.testing.assert_close(env._reward_com_centering(), torch.tensor([.055 ** 2]))
        # Free within 0.019 +- 0.03; run 1's -0.081 is 0.07 beyond.
        env.pitch = torch.tensor([[.019], [.045], [-.081]])
        torch.testing.assert_close(env._reward_pitch_band(), torch.tensor([0., 0., .07]))

    def test_balance_reward_set(self):
        scales = V2Cfg().balance.reward_scales
        self.assertEqual(set(scales), {'stand_still', 'feet_still', 'stance_height', 'com_centering',
                                       'load_balance', 'pitch_band'})
        self.assertGreater(scales['stance_height'], 0.)
        self.assertTrue(all(v < 0. for k, v in scales.items() if k != 'stance_height'))

    def test_balance_start_buffers(self):
        env = object.__new__(HomieIsaacLabEnv)
        n = 4
        env.num_lower_dof = 12
        env._grasp_upper_actions = torch.arange(15.)
        for name in ('actions', 'last_actions', 'last_last_actions'):
            setattr(env, name, torch.zeros(n, 27))
        env.current_upper_actions, env.delta_upper_actions = torch.zeros(n, 15), torch.ones(n, 15)
        env.obs_buf, env.privileged_obs_buf = torch.zeros(n, 456), torch.zeros(n, 79)
        bank = dict(obs=torch.full((3, 456), 7.), critic_obs=torch.full((3, 79), 7.), actions=torch.full((3, 27), 2.),
                    last_actions=torch.full((3, 27), 1.), current_upper_actions=torch.full((3, 15), 3.),
                    delta_upper_actions=torch.full((3, 15), .5))
        bank['obs'][2] = 9.
        env._bank = bank
        env._balance_reset_buffers(torch.tensor([1, 3], dtype=torch.int32), torch.tensor([True, False]),
                                   torch.tensor([2, 0]))
        self.assertTrue(torch.all(env.obs_buf[1] == 9.))                      # switched: walking history
        self.assertTrue(torch.all(env.actions[1] == 2.) and torch.all(env.last_actions[1] == 1.))
        self.assertTrue(torch.all(env.current_upper_actions[1] == 3.))
        torch.testing.assert_close(env.actions[3, 12:], torch.arange(15.))   # stance: arms hold the grasp
        torch.testing.assert_close(env.last_actions[3, 12:], torch.arange(15.))
        self.assertTrue(torch.all(env.delta_upper_actions[3] == 0.))
        self.assertTrue(torch.all(env.obs_buf[[0, 2, 3]] == 0.))
        # Stance-only starts (no bank loaded, e.g. the viewer) must not touch the bank.
        env._bank = None
        env._balance_reset_buffers(torch.tensor([0], dtype=torch.int32), torch.tensor([False]), torch.tensor([0]))
        torch.testing.assert_close(env.actions[0, 12:], torch.arange(15.))

    def test_mass_randomization_matches_official_effect(self):
        env = _mass_env()
        env._randomize_body_properties()
        ratio = env.robot.masses / DEFAULT
        self.assertTrue(torch.equal(env.robot.masses[:, 0], DEFAULT[0].expand(len(ratio))))   # pelvis untouched
        self.assertGreaterEqual(ratio[:, 1:].min().item(), .8)
        self.assertLessEqual(ratio[:, 1:].max().item(), 1.2)
        torso = env.robot.masses[:, 2]
        self.assertGreater(torso.min().item(), 6.78 * .8 - 1e-4)        # was 0.42 kg with the payload
        self.assertLess(torso.max().item(), 6.78 * 1.2 + 1e-4)          # was 18.1 kg with the payload
        torch.testing.assert_close(env.robot.inertias[..., 0], ratio)   # inertia follows the mass

    def test_enabled_payload_adds_after_link_scaling(self):
        env = _mass_env(8)
        env.cfg.domain_rand.randomize_link_mass = False
        env.cfg.domain_rand.randomize_payload_mass = True
        env._randomize_body_properties()
        torch.testing.assert_close(env.robot.masses[0], DEFAULT + torch.tensor([0., 0., 10., .3, .3]))

    def test_joint_power_ignores_curriculum_driven_upper_body(self):
        task = HomieTaskCore()
        task.num_lower_dof = 12
        task.commands = torch.zeros(1, 5)
        task.dof_vel = torch.ones(1, 27)
        task.torques = torch.cat((torch.full((1, 12), 2.), torch.full((1, 15), 1e3)), dim=1)
        self.assertAlmostEqual(task._reward_joint_power().item(), 12 * 2. / .1, places=3)

    def test_registration_does_not_replace_baseline(self):
        register_baseline()
        baseline = gym.spec("Homie-G1-Direct-v0")
        register_tasks()
        register_tasks()
        self.assertIs(gym.spec("Homie-G1-Direct-v0"), baseline)
        v2 = gym.spec("Homie-G1-V2-Direct-v0")
        self.assertEqual(v2.entry_point, "homie_isaaclab_v2.direct_env:HomieDirectEnv")
        self.assertTrue(all(v.startswith("homie_isaaclab_v2.") for v in v2.kwargs.values()))
        self.assertEqual(HomieRunnerCfg().experiment_name, "homie_g1_v2")
        self.assertEqual(HomieRunnerCfg().algorithm.class_name, "homie_isaaclab_v2.rsl_rl:HimPpoAdapter")
        balance = gym.spec("Homie-G1-V2-Balance-Direct-v0")
        self.assertEqual(balance.kwargs["env_cfg_entry_point"], "homie_isaaclab_v2.lab_config:HomieBalanceEnvCfg")

    def test_balance_alias_uses_its_own_log_namespace(self):
        with patch("homie_isaaclab_v2.training.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(train(["--task", "balance", "--max_iterations", "2"]), 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--task")+1], "Homie-G1-V2-Balance-Direct-v0")
        self.assertEqual(argv[argv.index("--experiment_name")+1], "homie_g1_v2_balance")

    def test_entry_forwards_to_v2_task_with_source_on_child_path(self):
        with patch("homie_isaaclab_v2.training.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(train(["--task", "g1", "--num_envs", "8", "--max_iterations", "2"]), 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--task")+1], "Homie-G1-V2-Direct-v0")
        self.assertEqual(argv[argv.index("--external_callback")+1], "homie_isaaclab_v2.register_tasks")
        self.assertEqual(argv[argv.index("--experiment_name")+1], "homie_g1_v2")
        path = run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(Path(path[0]), ROOT / "HomieRL")


if __name__ == "__main__":
    unittest.main()
