"""Regressions for V4 bench confirmation and reference-following measurements."""
import unittest

import numpy as np

from pushcart_bench.metrics import (
    Trace, cart_quality_metrics, reference_tracking_metrics,
    step_response_metrics, stopping_distance, stopping_metrics,
)


class StopConfirmationTests(unittest.TestCase):
    def test_final_rebound_invalidates_an_earlier_confirmed_stop(self):
        t = np.arange(0., 2.1, .1)
        speed = np.where(t < 1.5, .01, .2)
        m = stopping_metrics(t, np.cumsum(speed) * .1, np.zeros_like(t), speed, 0.)
        self.assertAlmostEqual(m["first_confirmed_time"], .5)
        self.assertFalse(m["confirmed"])
        self.assertIsNone(m["confirmed_distance"])
        self.assertIsNone(m["confirmed_time"])
        self.assertEqual(m["rebound_events"], 1)
        self.assertEqual(m["rebound_count"], 6)
        self.assertGreater(m["post_first_confirm_travel"], .1)

    def test_stop_confirmed_after_last_rebound(self):
        t = np.arange(0., 3.1, .1)
        speed = np.where((t >= 1.) & (t < 1.5), .2, .01)
        m = stopping_metrics(t, np.cumsum(speed) * .1, np.zeros_like(t), speed, 0.)
        self.assertTrue(m["confirmed"])
        self.assertAlmostEqual(m["first_confirmed_time"], .5)
        self.assertAlmostEqual(m["confirmed_time"], 2.)
        self.assertEqual(m["rebound_events"], 1)
        self.assertGreater(m["confirmed_distance"], m["first_confirmed_distance"])

    def test_missing_speed_breaks_dwell_instead_of_being_removed(self):
        t = np.arange(0., 1.1, .1)
        speed = np.full_like(t, .01)
        speed[4] = np.nan
        speed[8] = np.nan
        m = stopping_metrics(t, t * .01, np.zeros_like(t), speed, 0.)
        self.assertFalse(m["confirmed"])
        self.assertIsNone(m["first_confirmed_time"])
        self.assertEqual(m["invalid_samples"], 2)

    def test_missing_last_sample_invalidates_final_confirmation(self):
        t = np.arange(0., 1.1, .1)
        speed = np.full_like(t, .01)
        speed[-1] = np.nan
        m = stopping_metrics(t, t * .01, np.zeros_like(t), speed, 0.)
        self.assertFalse(m["confirmed"])
        self.assertIsNotNone(m["first_confirmed_time"])

    def test_zero_dwell_confirms_one_low_sample(self):
        m = stopping_metrics([1.], [0.], [0.], [.01], 1., dwell_s=0.)
        self.assertTrue(m["confirmed"])
        self.assertEqual(m["confirmed_time"], 0.)
        self.assertEqual(m["confirmed_distance"], 0.)

    def test_reconstructed_switch_roundoff_preserves_boundary_distance(self):
        t = np.array([5., 5.02, 5.04, 5.06])
        switch = t[2] - .02  # 5.0200000000000005, versus the stored boundary 5.02.
        self.assertGreater(switch, t[1])
        m = stopping_metrics(t, [0., .004, .008, .0082], np.zeros_like(t),
                             [.2, .2, .01, .01], switch, dwell_s=0.)
        self.assertAlmostEqual(m["first_low_distance"], .004)
        self.assertAlmostEqual(m["confirmed_distance"], .004)
        self.assertAlmostEqual(m["confirmed_time"], .02)

    def test_roundoff_does_not_produce_negative_stop_times(self):
        t = np.array([5., 5.02, 5.04])
        m = stopping_metrics(t, np.zeros_like(t), np.zeros_like(t), np.full_like(t, .01),
                             t[2] - .02, dwell_s=0.)
        for field in ("first_low_time", "first_confirmed_time", "confirmed_time"):
            self.assertEqual(m[field], 0.)

    def test_clock_gap_cannot_prove_dwell(self):
        t = np.array([0., .1, .2, 1., 1.1, 1.2])
        m = stopping_metrics(t, t * .01, np.zeros_like(t), np.full_like(t, .01), 0.)
        self.assertFalse(m["confirmed"])

    def test_missing_position_keeps_distance_unknown_after_recovery(self):
        t = np.arange(0., 2.1, .1)
        x = t * .01
        x[3] = np.nan
        m = stopping_metrics(t, x, np.zeros_like(t), np.full_like(t, .01), 0.)
        self.assertTrue(m["confirmed"])
        self.assertIsNone(m["confirmed_distance"])
        self.assertIsNotNone(m["post_confirm_travel"])

    def test_never_stopping_has_no_legacy_stopping_distance(self):
        t = np.arange(0., 2., .1)
        self.assertIsNone(stopping_distance(t, .2 * t, np.zeros_like(t), np.full_like(t, .2), 0.))

    def test_invalid_clock_and_threshold_fail_explicitly(self):
        for t in ([0., 0.], [1., 0.], [0., np.nan]):
            with self.subTest(t=t), self.assertRaises(ValueError):
                stopping_metrics(t, [0., 0.], [0., 0.], [.01, .01], 0.)
        for threshold in (0., -1., np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                stopping_metrics([0., 1.], [0., 0.], [0., 0.], [.01, .01], 0., v_stop=threshold)


class StepConfirmationTests(unittest.TestCase):
    def test_single_final_in_band_sample_does_not_establish_settling(self):
        t = np.arange(0., 2.1, .1)
        y = np.zeros_like(t)
        y[-1] = 1.
        m = step_response_metrics(t, y, 0., 0., 1.)
        self.assertFalse(m["settling_confirmed"])
        self.assertIsNone(m["settling_time"])

    def test_settling_start_and_confirmation_are_distinct(self):
        t = np.arange(0., 2.1, .1)
        m = step_response_metrics(t, np.where(t >= 1., 1., 0.), 0., 0., 1.)
        self.assertTrue(m["settling_confirmed"])
        self.assertAlmostEqual(m["settling_time"], 1.)
        self.assertAlmostEqual(m["settling_confirmed_time"], 1.5)

    def test_invalid_tail_cannot_confirm_settling(self):
        t = np.arange(0., 2.1, .1)
        y = np.ones_like(t)
        y[-1] = np.nan
        m = step_response_metrics(t, y, 0., 0., 1.)
        self.assertFalse(m["settling_confirmed"])
        self.assertIsNone(m["settling_time"])
        self.assertIsNone(m["ss_error"])

    def test_clock_gap_restarts_settling_dwell(self):
        t = np.array([0., .1, .2, 1., 1.1, 1.2])
        m = step_response_metrics(t, np.ones_like(t), 0., 0., 1.)
        self.assertFalse(m["settling_confirmed"])

    def test_reconstructed_switch_roundoff_preserves_boundary_and_nonnegative_time(self):
        t = np.array([5., 5.02, 5.04])
        m = step_response_metrics(t, [0., 1., 1.], t[2] - .02, 0., 1., settling_dwell_s=0.)
        self.assertTrue(m["settling_confirmed"])
        self.assertEqual(m["settling_time"], 0.)
        self.assertEqual(m["settling_confirmed_time"], 0.)

    def test_short_post_step_tail_uses_original_cadence(self):
        t = np.array([0., .1, .2, .3, .4, 1.])
        m = step_response_metrics(t, np.ones_like(t), .4, 0., 1.)
        self.assertFalse(m["settling_confirmed"])
        self.assertIsNone(m["settling_time"])
        y = np.array([0., 0., 0., 0., .2, 1.])
        m = step_response_metrics(t, y, .4, 0., 1.)
        self.assertIsNone(m["rise_time"])


class ReferenceTrackingTests(unittest.TestCase):
    def test_following_delayed_reference_exactly_has_zero_tracking_error(self):
        t = np.arange(0., 5.01, .02)
        reference = np.where(t <= 1., 0., 1. - np.exp(-(t - 1.) / .4))
        m = reference_tracking_metrics(t, reference, reference, 1.)
        for window in ("full", "transient", "post"):
            for kind in ("rmse", "iae", "ise"):
                self.assertEqual(m[f"reference_tracking_{window}_{kind}"], 0.)
            self.assertAlmostEqual(m[f"reference_tracking_{window}_coverage"], 1.)

    def test_constant_tracking_error_has_correct_integrals(self):
        t = np.arange(0., 3.01, .1)
        m = reference_tracking_metrics(t, np.full_like(t, .2), np.zeros_like(t), 1., transient_s=1.)
        self.assertAlmostEqual(m["reference_tracking_full_iae"], .4)
        self.assertAlmostEqual(m["reference_tracking_full_ise"], .08)
        self.assertAlmostEqual(m["reference_tracking_transient_iae"], .2)
        self.assertAlmostEqual(m["reference_tracking_post_iae"], .2)

    def test_nan_interval_is_not_interpolated_or_counted_as_zero_error(self):
        t = np.array([0., .1, .2, .3, .4])
        actual = np.array([1., 1., np.nan, 1., 1.])
        m = reference_tracking_metrics(t, actual, np.zeros_like(t), 0., transient_s=.4)
        self.assertAlmostEqual(m["reference_tracking_full_valid_duration_s"], .2)
        self.assertAlmostEqual(m["reference_tracking_full_coverage"], .5)
        self.assertAlmostEqual(m["reference_tracking_full_iae"], .2)
        self.assertEqual(m["reference_tracking_full_rmse"], 1.)

    def test_early_termination_reports_partial_transient_coverage(self):
        t = np.array([0., .1, .2])
        m = reference_tracking_metrics(t, np.ones_like(t), np.zeros_like(t), 0., transient_s=1.)
        self.assertAlmostEqual(m["reference_tracking_transient_coverage"], .2)
        self.assertIsNone(m["reference_tracking_post_rmse"])
        self.assertIsNone(m["reference_tracking_post_iae"])

    def test_clock_gap_is_not_integrated(self):
        t = np.array([0., .1, .2, 1., 1.1, 1.2])
        m = reference_tracking_metrics(t, np.ones_like(t), np.zeros_like(t), 0.)
        self.assertAlmostEqual(m["reference_tracking_full_iae"], .4)
        self.assertAlmostEqual(m["reference_tracking_full_coverage"], 1. / 3.)

    def test_response_cost_integration_uses_control_step_endpoints(self):
        trace = Trace()
        for i in range(21):
            t = i * .1
            cost = 2. if .5 < t <= 1. else 0.
            trace.append(t, [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], telemetry={
                "quality_ready": True, "cart_response_ready": True,
                "cart_response_error_cost": cost, "cart_response_error_vx": cost,
            })
        m = cart_quality_metrics(trace, start_s=1.1, response_window=(.5, 1.))
        self.assertEqual(m["cart_response_error_cost_mean"], 0.)
        self.assertEqual(m["cart_response_error_cost_mean_transient"], 2.)
        self.assertAlmostEqual(m["cart_response_error_cost_integral_transient"], 1.)
        self.assertAlmostEqual(m["cart_response_error_cost_integral_full"], 1.)
        self.assertAlmostEqual(m["cart_response_transient_valid_duration_s"], .5)

    def test_response_cost_excludes_invalid_transition_intervals(self):
        trace = Trace()
        for i in range(5):
            trace.append(i * .1, [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], telemetry={
                "quality_ready": True, "cart_response_ready": i != 2,
                "cart_response_error_cost": 1., "cart_response_error_vx": 1.,
            })
        m = cart_quality_metrics(trace, response_window=(0., .4))
        # Only the invalid transition (.1,.2] is omitted. The valid recovery step
        # (.2,.3] has its own known cost and dt, so it must be counted.
        self.assertAlmostEqual(m["cart_response_error_cost_integral_transient"], .3)
        self.assertAlmostEqual(m["cart_response_transient_coverage"], .75)
        # Continuous-error trapezoids require both endpoints to be valid.
        self.assertAlmostEqual(m["cart_response_error_vx_iae_transient"], .2)


if __name__ == "__main__":
    unittest.main()
