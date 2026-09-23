"""Offline checks prevent a wrong wrench sign/frame being reported as calibrated."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart.wrench_calibration import PHASES, summarize_wrench_trace


def synthetic_trace():
    shape = (len(PHASES) * 4, 4, 2, 3)
    applied_f, applied_m = np.zeros(shape), np.zeros(shape)
    phase = np.repeat(np.arange(len(PHASES)), 4)
    for name, value in (("force_positive", 5.), ("force_negative", -5.)):
        applied_f[phase == PHASES.index(name), ..., 0] = value
    for name, value in (("torque_positive", .15), ("torque_negative", -.15)):
        applied_m[phase == PHASES.index(name), ..., 2] = value
    return {"phase": phase, "selected": np.ones(len(phase), bool), "force_w": -applied_f + .1,
            "torque_w": -applied_m + .02, "applied_force_w": applied_f, "applied_torque_w": applied_m}


class WrenchCalibrationTests(unittest.TestCase):
    def test_correct_signed_load_with_bias_passes_but_is_not_dynamic_calibration(self):
        report = summarize_wrench_trace(synthetic_trace())
        self.assertTrue(report["known_load_source_validation_passed"])
        self.assertFalse(report["dynamic_contact_wrench_calibrated"])
        self.assertAlmostEqual(report["known_load_probes"]["force"]["error_norm_N"][0][0], 0.)

    def test_wrong_sign_fails(self):
        trace = synthetic_trace()
        trace["force_w"] *= -1
        self.assertFalse(summarize_wrench_trace(trace)["known_load_source_validation_passed"])

    def test_single_yaw_frame_error_cannot_cancel_with_other_environments(self):
        trace = synthetic_trace()
        trace["force_w"][:, 2] = trace["force_w"][:, 2, :, [1, 0, 2]].transpose(1, 2, 0)
        self.assertFalse(summarize_wrench_trace(trace)["known_load_source_validation_passed"])

    def test_nonfinite_or_robot_reset_invalidates_source_validation(self):
        trace = synthetic_trace()
        trace["force_w"][0, 0, 0, 0] = np.nan
        self.assertFalse(summarize_wrench_trace(trace)["known_load_source_validation_passed"])
        self.assertFalse(summarize_wrench_trace(synthetic_trace(), resets=1)["known_load_source_validation_passed"])

    def test_no_probe_does_not_pass_as_zero_error(self):
        trace = synthetic_trace()
        trace["applied_force_w"].fill(0)
        trace["force_w"].fill(0)
        self.assertFalse(summarize_wrench_trace(trace)["known_load_source_validation_passed"])

    def test_unselected_startup_spike_does_not_pollute_settled_probe(self):
        trace = synthetic_trace()
        trace["selected"][::4] = False
        trace["force_w"][::4] += 1000.
        self.assertTrue(summarize_wrench_trace(trace)["known_load_source_validation_passed"])


if __name__ == "__main__":
    unittest.main()
