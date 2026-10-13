"""HOMIE v2 keeps the baseline training contract except for the recorded port-vs-official fixes."""
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
    def test_config_differs_from_baseline_only_in_payload_switch(self):
        base, v2 = class_to_dict(G1RoughCfg()), class_to_dict(V2Cfg())
        self.assertTrue(base['domain_rand']['randomize_payload_mass'])
        self.assertFalse(v2['domain_rand']['randomize_payload_mass'])
        v2['domain_rand']['randomize_payload_mass'] = True
        self.assertEqual(base, v2)
        self.assertEqual(class_to_dict(G1RoughCfgPPO()), class_to_dict(V2PPO()))

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
