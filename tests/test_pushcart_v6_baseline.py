"""Guard the V6 scope against saved V3.2 data and cross-version checkpoint drift."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart_v6.baseline import (
    agent_baseline, baseline_source, config_dict, env_baseline,
    homie_baseline, load_baseline, task_baseline,
)
from homie_pushcart_v6.task_config import (
    PushCartTaskCfg, num_actor_obs, num_critic_obs, num_estimator_obs,
    task_cfg_from_run, validate_task_schema,
)


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


class V6BaselineTests(unittest.TestCase):
    def assert_subset_equal(self, expected, actual, path="task"):
        for key, value in expected.items():
            name = f"{path}.{key}"
            self.assertIn(key, actual, name)
            if isinstance(value, dict):
                self.assert_subset_equal(value, actual[key], name)
            else:
                self.assertEqual(value, actual[key], name)

    def test_snapshot_matches_the_actual_v32_run_and_hashes(self):
        source = baseline_source()
        env_path = ROOT / source["files"]["env_yaml"]
        agent_path = ROOT / source["files"]["agent_yaml"]
        if not env_path.is_file() or not agent_path.is_file():
            self.skipTest("Historical logs are optional when distributing V6")
        saved_env = normalized(yaml.load(env_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader))
        saved_agent = normalized(yaml.load(agent_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader))
        self.assertEqual(saved_env, env_baseline())
        self.assertEqual(saved_agent, agent_baseline())
        for key, relative_path in source["files"].items():
            path = ROOT / relative_path
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.assertEqual(source["sha256"][key], digest, key)

    def test_all_original_task_values_survive_and_v4_options_are_off(self):
        cfg = PushCartTaskCfg()
        self.assert_subset_equal(task_baseline(), config_dict(cfg))
        self.assertFalse(cfg.commands.sequence_enabled)
        self.assertFalse(cfg.idle_correction.enabled)
        self.assertFalse(cfg.rewards.force_rate_seed_on_reset)
        validate_task_schema(cfg)

    def test_quality_rewards_use_trained_v32_values_not_current_defaults(self):
        cfg = PushCartTaskCfg()
        self.assertTrue(cfg.quality.enabled)
        self.assertEqual(config_dict(cfg.rewards.scales), task_baseline()["rewards"]["scales"])
        required = {"cart_yaw_lead": .5, "force_waste": -.5, "force_rate": -.2,
                    "track_cart_vx": 1.5, "track_progress": 1., "cart_lin_ripple": -.1,
                    "cart_yaw_ripple": -.15, "cart_response_error": -.1,
                    "cart_wrench_osc": -.02, "cart_ripple": 0.}
        for name, value in required.items():
            self.assertEqual(getattr(cfg.rewards.scales, name), value, name)
        self.assertEqual(len(config_dict(cfg.rewards.scales)), 28)

    def test_original_ppo_and_normalization_are_pinned(self):
        cfg = agent_baseline()
        self.assertEqual(cfg["seed"], 1)
        self.assertEqual(cfg["num_steps_per_env"], 24)
        for name in ("actor", "critic"):
            self.assertEqual(cfg[name]["hidden_dims"], [512, 256, 128])
            self.assertEqual(cfg[name]["activation"], "elu")
            self.assertTrue(cfg[name]["obs_normalization"])
        self.assertEqual(cfg["algorithm"]["learning_rate"], .001)
        self.assertEqual(cfg["algorithm"]["num_learning_epochs"], 5)
        self.assertEqual(cfg["algorithm"]["num_mini_batches"], 4)

    def test_histories_and_actions_are_separate_contracts(self):
        cfg = PushCartTaskCfg()
        self.assertEqual(num_actor_obs(cfg), 605)
        self.assertEqual(num_critic_obs(cfg), 665)
        self.assertEqual(num_estimator_obs(cfg), 462)
        self.assertEqual(cfg.env.num_hl_actions, 18)
        self.assertEqual(config_dict(cfg.actions), task_baseline()["actions"])

    def test_baseline_accessors_and_task_instances_do_not_share_mutable_values(self):
        original = load_baseline()
        cfg = PushCartTaskCfg()
        cfg.actions.cmd_residual_scale[0] = 999
        cfg.baseline_source["sha256"]["env_yaml"] = "changed"
        changed = homie_baseline()
        changed.clear()
        snapshot = load_baseline()
        snapshot["task"]["rewards"]["scales"]["force_rate"] = 999
        self.assertEqual(original, load_baseline())
        self.assertEqual(PushCartTaskCfg().actions.cmd_residual_scale, [.4, .3, .5])
        self.assertEqual(PushCartTaskCfg().baseline_source, baseline_source())

    def write_run(self, root, task):
        params = Path(root) / "params"
        params.mkdir(exist_ok=True)
        (params / "env.yaml").write_text(yaml.safe_dump({"task": task}), encoding="utf-8")
        return Path(root) / "model_0.pt"

    def test_complete_v6_config_restores_exactly(self):
        expected = config_dict(PushCartTaskCfg())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.write_run(directory, expected)
            cfg = task_cfg_from_run(checkpoint, verbose=False)
            restored = config_dict(cfg)
            self.assertEqual(restored.pop("source_yaml"), str(Path(directory) / "params" / "env.yaml"))
            self.assertEqual(restored, expected)

    def test_missing_or_old_checkpoint_config_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                task_cfg_from_run(Path(directory) / "model_0.pt", verbose=False)
            for task in (task_baseline(), {"version": 53}, {"version": 6}):
                checkpoint = self.write_run(directory, task)
                with self.assertRaises(ValueError):
                    task_cfg_from_run(checkpoint, verbose=False)

    def test_saved_schema_provenance_unknown_fields_and_v4_drift_are_rejected(self):
        base = config_dict(PushCartTaskCfg())
        mutations = [
            lambda task: task.update(observation_schema="v53"),
            lambda task: task["baseline_source"]["sha256"].update(env_yaml="wrong"),
            lambda task: task["rewards"]["scales"].update(unapproved_reward=1.),
            lambda task: task["quality"].pop("velocity_scale"),
            lambda task: task["env"].update(actor_history=20),
            lambda task: task["commands"].update(sequence_enabled=True),
            lambda task: task["idle_correction"].update(enabled=True),
            lambda task: task["rewards"].update(force_rate_seed_on_reset=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for mutate in mutations:
                task = deepcopy(base)
                mutate(task)
                checkpoint = self.write_run(directory, task)
                with self.assertRaises(ValueError):
                    task_cfg_from_run(checkpoint, verbose=False)


if __name__ == "__main__":
    unittest.main()
