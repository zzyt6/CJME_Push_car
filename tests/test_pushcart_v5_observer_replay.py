"""Offline comparison must honor recorded frames, timestamps and failure masks."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests"), str(ROOT / "HomieRL")]
from compare_pushcart_v5_observers import compare_recorded_trace, rotate_xy
from test_pushcart_v5_motion_observer import trace
from homie_pushcart_v5.motion_observer import compare_motion_estimators


def fixture():
    synthetic = trace(steps=80, base_yaw_rate=1.1)
    replay = compare_motion_estimators(synthetic)
    t = np.arange(80) * .02
    rel = synthetic["handle_pose_b"][:, 3]
    data = {"pose_base": synthetic["handle_pose_b"][:, None],
            "base_velocity_est": synthetic["base_velocity_b"][:, None],
            "imu_gyro_b": synthetic["gyro_b"][:, None], "gravity_b": synthetic["gravity_b"][:, None],
            "valid": np.ones((80, 1), dtype=bool), "failed": np.zeros((80, 1), dtype=bool),
            "pose_delay_steps": np.zeros((80, 1), dtype=int),
            "actual": rotate_xy(synthetic["truth_cart_twist_base"], -rel)[:, None],
            "estimated_cart_twist": rotate_xy(replay["baseline_twist_base"].numpy()[:, 0], -rel)[:, None],
            "observer_ready": replay["baseline_ready"].numpy(),
            "cart_yaw": np.full((80, 1), 3.), "base_yaw": (t * 1.1)[:, None],
            "time_s": (t + .02)[:, None],
            "cart_com": np.c_[.8 + .2 * t, .1 + .04 * t, np.full(80, .3)][:, None]}
    metadata = {"dt": .02, "observer_geometry": [-.49, 0., 0.], "cases": [{"task": "synthetic", "payload_kg": 0.}]}
    return data, metadata


class ObserverReplayTests(unittest.TestCase):
    def test_recorded_frame_roundtrip_and_same_common_sample_mask(self):
        data, metadata = fixture()
        report, arrays = compare_recorded_trace(data, metadata)
        self.assertEqual(report["common_samples"], 60)
        self.assertLess(max(report["replayed_baseline_vs_recorded_max_abs"]), 1e-12)
        self.assertLess(max(report["overall_instantaneous"]["candidate"]["rmse"]), 1e-12)
        self.assertGreater(report["overall_instantaneous"]["recorded_v5"]["xy_vector_rmse"], .02)
        self.assertFalse(report["candidate_promoted_to_default"])
        np.testing.assert_allclose(arrays["truth_current_twist_base"][arrays["common_mask"]],
                                   arrays["truth_kernel_matched_twist_base"][arrays["common_mask"]], atol=1e-12)

    def test_first_failure_masks_all_later_rows_even_if_flag_returns_false(self):
        data, metadata = fixture()
        data["failed"][40] = True
        report, arrays = compare_recorded_trace(data, metadata)
        self.assertEqual(report["common_samples"], 20)
        self.assertFalse(arrays["common_mask"][40:].any())
        self.assertFalse(arrays["valid_before_first_failure"][40:].any())

    def test_truth_change_cannot_alter_candidate_output(self):
        data, metadata = fixture()
        _, before = compare_recorded_trace(data, metadata)
        data["actual"][:] = 100.
        data["cart_com"][:] = 200.
        data["cart_yaw"][:] = -1.
        _, after = compare_recorded_trace(data, metadata)
        np.testing.assert_array_equal(before["candidate_twist_base"], after["candidate_twist_base"])

    def test_recorded_estimate_uses_observed_yaw_not_true_relative_yaw(self):
        data, metadata = fixture()
        data["pose_base"][:, :, 3] += .4
        _, arrays = compare_recorded_trace(data, metadata)
        expected = rotate_xy(data["estimated_cart_twist"], data["pose_base"][..., 3])
        np.testing.assert_allclose(arrays["recorded_v5_twist_base"], expected)

    def test_irregular_timestamps_rejected(self):
        data, metadata = fixture()
        data["time_s"][35, 0] += .003
        with self.assertRaisesRegex(ValueError, "regular control timestamps"):
            compare_recorded_trace(data, metadata)


if __name__ == "__main__":
    unittest.main()
