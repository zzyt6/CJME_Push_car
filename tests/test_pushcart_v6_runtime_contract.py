"""Offline validation of runtime audit integrity; never starts Isaac/Kit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v6_runtime_validator", ROOT / "scripts/validate_pushcart_v6_runtime.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class V6RuntimeContractTests(unittest.TestCase):
    def test_exact_comparison_rejects_small_drift_nonfinite_missing_and_dtype_change(self):
        original = {"reward": np.array([[1., 2.]], dtype=np.float64)}
        self.assertTrue(validator.compare_traces(original, original)["equal"])
        changed = original["reward"].copy()
        changed[0, 1] += 1e-12
        result = validator.compare_traces(original, {"reward": changed})
        self.assertFalse(result["equal"])
        self.assertEqual(result["differences"]["reward"]["first_index"], [0, 1])
        self.assertFalse(validator.compare_traces(original, {})["equal"])
        self.assertFalse(validator.compare_traces({}, {})["equal"])
        self.assertFalse(validator.compare_traces(original, {"reward": original["reward"].astype(np.float32)})["equal"])
        invalid = {"reward": np.array([np.nan])}
        self.assertFalse(validator.compare_traces(invalid, invalid)["equal"])

    def test_same_seed_cfg_preserves_baseline_rewards_quality_and_noise(self):
        from homie_pushcart_v6.baseline import config_dict, homie_baseline, task_baseline
        a, old = validator.make_runtime_configs("v32", 8, 8)
        b, new = validator.make_runtime_configs("v6", 8, 8)
        self.assertEqual(config_dict(a), config_dict(b))
        self.assertEqual(config_dict(old.rewards), config_dict(new.rewards))
        self.assertEqual(config_dict(old.quality), config_dict(new.quality))
        self.assertEqual(config_dict(old.rewards.scales), task_baseline()["rewards"]["scales"])
        self.assertTrue(old.quality.enabled)
        self.assertEqual(a.noise.add_noise, homie_baseline()["noise"]["add_noise"])
        self.assertFalse(old.rewards.force_rate_seed_on_reset)
        self.assertEqual(old.randomization.cart_push_vel_max, 0.)

    def test_prescribed_actions_are_repeatable_and_use_every_channel(self):
        trajectory = np.stack([validator.prescribed_action(step, 8) for step in range(64)])
        self.assertEqual(trajectory.shape, (64, 8, 18))
        self.assertEqual(trajectory.dtype, np.float32)
        self.assertTrue((np.ptp(trajectory, axis=0) > 0).all())
        self.assertLessEqual(float(np.abs(trajectory).max()), .025)
        np.testing.assert_array_equal(trajectory[10], validator.prescribed_action(10, 8))

    def test_nonce_trace_hash_and_failure_artifacts_cannot_be_bypassed_by_zero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trace = output / "v6.npz"
            np.savez_compressed(trace, reward=np.zeros((3, 8)))
            report = {"result": validator.SUCCESS, "variant": "v6", "run_id": "new",
                      "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}
            (output / "v6.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(validator.validate_child_artifacts(output, "v6", "new", 0), report)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                validator.validate_child_artifacts(output, "v6", "old", 0)
            with self.assertRaisesRegex(RuntimeError, "exited"):
                validator.validate_child_artifacts(output, "v6", "new", 2)
            np.savez_compressed(trace, reward=np.ones((3, 8)))
            with self.assertRaisesRegex(RuntimeError, "hash"):
                validator.validate_child_artifacts(output, "v6", "new", 0)
            (output / "v6.failure.json").write_text(json.dumps({"run_id": "new"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reported a failure"):
                validator.validate_child_artifacts(output, "v6", "new", 0)


if __name__ == "__main__":
    unittest.main()
