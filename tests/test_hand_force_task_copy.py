"""The independent research task starts with HOMIE's original training contract."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym

from homie_isaaclab import register_tasks as register_baseline
from homie_isaaclab.config import G1RoughCfg, G1RoughCfgPPO
from homie_isaaclab.math_utils import class_to_dict
from homie_hand_force import register_tasks
from homie_hand_force.config import G1RoughCfg as ForceCfg, G1RoughCfgPPO as ForcePPO
from homie_hand_force.lab_config import HomieRunnerCfg
from homie_hand_force.training import train

ROOT = Path(__file__).resolve().parents[1]


class HandForceTaskCopyTests(unittest.TestCase):
    def test_training_defaults_equal_and_independent(self):
        original, copied = G1RoughCfg(), ForceCfg()
        self.assertEqual(class_to_dict(original), class_to_dict(copied))
        self.assertEqual(class_to_dict(G1RoughCfgPPO()), class_to_dict(ForcePPO()))
        copied.domain_rand.init_upper_ratio = .7
        self.assertEqual(original.domain_rand.init_upper_ratio, 0.)
        self.assertEqual(ForceCfg().domain_rand.init_upper_ratio, 0.)

    def test_registration_does_not_replace_baseline(self):
        register_baseline()
        baseline = gym.spec("Homie-G1-Direct-v0")
        register_tasks()
        register_tasks()
        self.assertIs(gym.spec("Homie-G1-Direct-v0"), baseline)
        copied = gym.spec("Homie-G1-HandForce-Direct-v0")
        self.assertEqual(copied.entry_point, "homie_hand_force.direct_env:HomieDirectEnv")
        self.assertTrue(all(v.startswith("homie_hand_force.") for v in copied.kwargs.values()))
        self.assertEqual(HomieRunnerCfg().experiment_name, "homie_g1_hand_force")
        self.assertEqual(HomieRunnerCfg().algorithm.class_name, "homie_hand_force.rsl_rl:HimPpoAdapter")

    def test_entry_forwards_to_separate_task_and_log_namespace(self):
        with patch("homie_hand_force.training.subprocess.run") as run:
            run.return_value.returncode = 0
            self.assertEqual(train(["--task", "g1", "--num_envs", "8", "--max_iterations", "2"]), 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--task")+1], "Homie-G1-HandForce-Direct-v0")
        self.assertEqual(argv[argv.index("--external_callback")+1], "homie_hand_force.register_tasks")
        self.assertEqual(argv[argv.index("--experiment_name")+1], "homie_g1_hand_force")

    def test_initial_copy_has_only_recorded_namespace_changes(self):
        manifest = json.loads((ROOT/"HomieRL/homie_hand_force/baseline_manifest.json").read_text())
        for entry in manifest["files"]:
            original = (ROOT/entry["source"]).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), entry["source_sha256"], entry["source"])
            expected = original.decode("utf8")
            for before, after in entry["replacements"]:
                expected = expected.replace(before, after)
            self.assertEqual((ROOT/entry["copy"]).read_bytes(), expected.encode("utf8"), entry["copy"])


if __name__ == "__main__":
    unittest.main()
