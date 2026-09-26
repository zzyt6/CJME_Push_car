"""Recovery metric semantics: paired references, dwell, failures and physical impulse."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v5.reaction_bench import SCENARIOS, summarize_trial, task_command, _first_recovery, _reference_metrics


def trace(duration=7., dt=.02):
    times = np.arange(1, round(duration / dt) + 1) * dt
    n = len(times)
    applied = np.zeros((n, 6))
    applied[(times > 4. + 1e-9) & (times <= 4.2 + 1e-9), 0] = 15.
    return {"time_s": times, "applied_wrench": applied, "disturbance_active": np.linalg.norm(applied, axis=1) > 0,
            "cart_twist": np.tile([.3, 0., 0.], (n, 1)),
            "command_reference": np.tile([.3, 0., 0.], (n, 1)),
            "robot_momentum": np.tile([12., 0., 0., 0., 0., 0.], (n, 1)),
            "robot_mass": np.full(n, 40.), "robot_tilt_rad": np.zeros(n),
            "hand_wrench": np.tile([10., 0., 0., 0., 0., 0.], (n, 1)), "dt_s": np.asarray(dt)}


class RecoveryMetricTests(unittest.TestCase):
    def test_command_ramp_has_known_time_and_magnitude(self):
        np.testing.assert_allclose(task_command(.99), [0., 0., 0.])
        np.testing.assert_allclose(task_command(1.5), [.15, 0., 0.])
        np.testing.assert_allclose(task_command(8.), [.3, 0., 0.])

    def test_nominal_walking_momentum_is_not_penalized(self):
        reference = trace()
        result = summarize_trial(trace(), reference, SCENARIOS[0])
        self.assertFalse(result["recovered"])
        self.assertTrue(result["disturbance_rejected"])
        self.assertEqual(result["paired_robot_linear_momentum_deviation_peak_ns"], 0.)
        self.assertIsNone(result["recovery_time_s"])

    def test_error_must_remain_in_band_for_full_dwell(self):
        sample, reference = trace(), trace()
        times = sample["time_s"]
        sample["cart_twist"][(times > 4. + 1e-9) & (times < 5.), 0] += .2
        sample["cart_twist"][(times >= 5.2) & (times < 5.4), 0] += .2
        result = summarize_trial(sample, reference, SCENARIOS[0])
        self.assertTrue(result["recovered"])
        self.assertAlmostEqual(result["recovery_time_s"], 1.2)
        self.assertAlmostEqual(result["recovery_confirmed_at_s"], 5.9)

    def test_short_last_window_is_censored_not_recovered(self):
        sample, reference = trace(4.6), trace(4.6)
        result = summarize_trial(sample, reference, SCENARIOS[0])
        self.assertEqual(result["recovery_status"], "not_recovered_within_horizon")
        self.assertIsNone(result["recovery_time_s"])

    def test_applied_force_must_actually_be_off_before_recovery(self):
        sample = trace()
        sample["cart_twist"][(sample["time_s"] > 4. + 1e-9) & (sample["time_s"] <= 4.2), 0] += .2
        sample["disturbance_active"] = sample["time_s"] <= 4.4
        result = summarize_trial(sample, trace(), SCENARIOS[0])
        self.assertAlmostEqual(result["recovery_time_s"], .22)
        self.assertAlmostEqual(result["recovery_confirmed_at_s"], 4.92)

    def test_failure_cannot_be_hidden_by_previous_in_band_samples(self):
        result = summarize_trial(trace(), trace(), SCENARIOS[0], failure={"reason": "fall", "time_s": 6.})
        self.assertEqual(result["recovery_status"], "terminated")
        self.assertFalse(result["recovered"])
        self.assertIsNone(result["recovery_time_s"])

    def test_failed_reference_is_not_a_success_baseline(self):
        result = summarize_trial(trace(), trace(), SCENARIOS[0], reference_failure={"reason": "fall"})
        self.assertEqual(result["recovery_status"], "reference_terminated")

    def test_event_not_reached_is_explicit(self):
        result = summarize_trial(trace(2.), trace(2.), SCENARIOS[0])
        self.assertEqual(result["recovery_status"], "event_not_reached")
        self.assertFalse(result["event_reached"])

    def test_physics_rate_load_impulse_uses_all_substeps(self):
        sample = trace()
        sample["physics_hand_wrench"] = np.zeros((1400, 2, 6))
        sample["physics_hand_wrench"][:, :, 0] = 5.
        sample["physics_dt_s"] = np.asarray(.005)
        result = summarize_trial(sample, trace(), SCENARIOS[0])
        self.assertEqual(result["raw_hand_metric_rate_hz"], 200.)
        self.assertAlmostEqual(result["raw_hand_signed_impulse_ns_nms"][0], 70.)
        self.assertAlmostEqual(result["raw_hand_force_peak_n"], 10.)

    def test_misaligned_reference_is_rejected(self):
        reference = trace()
        reference["time_s"] += .01
        with self.assertRaisesRegex(ValueError, "same sample times"):
            summarize_trial(trace(), reference, SCENARIOS[0])

    def test_reference_must_match_before_the_event(self):
        sample = trace()
        sample["cart_twist"][0, 0] += .1
        result = summarize_trial(sample, trace(), SCENARIOS[0])
        self.assertFalse(result["paired_pre_event_match"])
        self.assertEqual(result["recovery_status"], "reference_prefix_mismatch")

    def test_paired_stationary_policy_does_not_restore_task(self):
        sample, reference = trace(), trace()
        sample["cart_twist"][:, 0] = reference["cart_twist"][:, 0] = 0.
        result = summarize_trial(sample, reference, SCENARIOS[0])
        self.assertEqual(result["recovery_status"], "task_not_restored")
        self.assertIsNotNone(result["paired_return_time_s"])
        self.assertFalse(result["task_restored"])

    def test_no_delivered_disturbance_is_not_a_recovery(self):
        sample = trace()
        sample["applied_wrench"][:] = 0.
        result = summarize_trial(sample, trace(), SCENARIOS[0])
        self.assertEqual(result["recovery_status"], "disturbance_not_verified")

    def test_missing_samples_cannot_complete_dwell(self):
        times = np.r_[np.arange(4.2, 4.4, .02), np.arange(5., 5.22, .02)]
        self.assertIsNone(_first_recovery(times, np.ones(len(times), dtype=bool), 4.2, .5))

    def test_success_window_cannot_precede_first_response(self):
        sample = trace(6.)
        sample["cart_twist"][(sample["time_s"] > 5.5), 0] += .2
        result = summarize_trial(sample, trace(6.), SCENARIOS[0])
        self.assertFalse(result["recovered"])
        self.assertIsNone(result["recovery_time_s"])

    def test_nonfinite_sample_is_never_a_success(self):
        sample = trace()
        sample["robot_momentum"][-1, 0] = np.nan
        result = summarize_trial(sample, trace(), SCENARIOS[0])
        self.assertEqual(result["recovery_status"], "nonfinite_trace")
        self.assertFalse(result["recovered"])

    def test_reference_and_scenario_tracking_rmse_use_identical_post_ramp_window(self):
        sample = trace()
        # Large start-up error must not lower/inflate only one comparison side.
        sample["cart_twist"][sample["time_s"] < 2.] += np.array([2., 1., .8])
        sample["cart_twist"][sample["time_s"] >= 2.] += np.array([.03, .02, .04])
        reference = _reference_metrics(sample)
        trial = summarize_trial(sample, sample, SCENARIOS[0])
        np.testing.assert_allclose(reference["reference_cart_twist_tracking_rmse"], [.03, .02, .04])
        np.testing.assert_allclose(reference["reference_cart_twist_tracking_rmse"], trial["cart_twist_tracking_rmse"])
        for key in ("tracking_window_requested_start_s", "tracking_window_start_s", "tracking_window_end_s",
                    "tracking_window_samples", "tracking_window_fallback_to_available"):
            self.assertEqual(reference["reference_" + key], trial[key])
        self.assertEqual(trial["tracking_window_start_s"], 2.)
        self.assertFalse(trial["tracking_window_fallback_to_available"])

    def test_short_rollout_tracking_window_fallback_is_explicit_on_both_sides(self):
        sample = trace(1.)
        sample["cart_twist"][:, 0] += .1
        reference = _reference_metrics(sample)
        trial = summarize_trial(sample, sample, SCENARIOS[0])
        np.testing.assert_allclose(reference["reference_cart_twist_tracking_rmse"], trial["cart_twist_tracking_rmse"])
        self.assertTrue(reference["reference_tracking_window_fallback_to_available"])
        self.assertTrue(trial["tracking_window_fallback_to_available"])
        self.assertEqual(trial["tracking_window_requested_start_s"], 2.)
        self.assertEqual(trial["tracking_window_start_s"], .02)


if __name__ == "__main__":
    unittest.main()
