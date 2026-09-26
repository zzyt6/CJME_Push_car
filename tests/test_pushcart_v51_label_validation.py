"""Known-load target/reference and diagnostic statistics, without simulator startup."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_pushcart_v51_labels import PHASES, external_reaction_at_palm, lag_diagnostic, prescribed_load, summarize_trace


class LabelValidationTests(unittest.TestCase):
    def test_combined_force_couple_and_offset_have_hand_on_environment_sign(self):
        force = np.array([3., 0., 0.])
        moment_com = np.array([0., 0., 2.])
        result = external_reaction_at_palm(force, moment_com, np.array([0., 1., 0.]), np.zeros(3))
        np.testing.assert_allclose(result, [-3., 0., 0., 0., 0., 1.])
        shifted = external_reaction_at_palm(force, moment_com, np.array([10., 11., 10.]), np.full(3, 10.))
        np.testing.assert_allclose(shifted, result)

    def test_lag_uses_positive_value_for_delayed_measurement(self):
        expected = np.random.default_rng(51).normal(size=(250, 3))
        measured = np.zeros_like(expected)
        measured[3:] = expected[:-3]
        report = lag_diagnostic(measured, expected, .005)
        self.assertEqual(report['lag_s'], .015)
        self.assertAlmostEqual(report['correlation'], 1.)
        self.assertAlmostEqual(report['aligned_vector_rms'], 0.)
        self.assertIsNone(lag_diagnostic(np.zeros_like(expected), expected, .005)['lag_s'])

    def test_static_and_dynamic_loads_include_signed_excitation_and_release(self):
        positive = prescribed_load('static_positive', 1.)
        negative = prescribed_load('static_negative', 1.)
        for p, n in zip(positive, negative):
            np.testing.assert_allclose(p, -n)
        dynamic = [prescribed_load('dynamic_loaded', t)[0] for t in np.arange(0., 3., .005)]
        self.assertTrue((np.std(dynamic, axis=0) > .1).all())
        force, torque = prescribed_load('dynamic_release', 0.)
        np.testing.assert_array_equal(force, 0.)
        np.testing.assert_array_equal(torque, 0.)

    def test_statistics_keep_opposite_hand_errors_and_invalid_samples(self):
        steps, n = 30, 2
        phase = np.repeat(np.arange(len(PHASES)), steps)
        expected = np.ones((len(phase), n, 2, 6))
        measured = expected.copy()
        measured[:, :, 0, 0] += 2.
        measured[:, :, 1, 0] -= 2.
        trace = {'phase': phase, 'measured_wrench_w': measured, 'expected_wrench_w': expected,
                 'valid': np.ones((len(phase), n), dtype=bool), 'fault': np.zeros((len(phase), n), dtype=bool)}
        report = summarize_trace(trace, dt=.005, expected_samples_per_phase=steps)
        self.assertEqual(report['result'], 'MEASURED')
        hands = report['phases']['dynamic_loaded']['environment_hand'][0]
        self.assertEqual(hands[0]['force_vector_rms_N'], 2.)
        self.assertEqual(hands[1]['force_vector_rms_N'], 2.)
        self.assertFalse(report['dynamic_contact_wrench_calibrated'])
        trace['valid'][steps * 4, 0] = False
        report = summarize_trace(trace, dt=.005, expected_samples_per_phase=steps)
        self.assertEqual(report['result'], 'INVALID_EXECUTION')
        self.assertEqual(report['phases']['dynamic_loaded']['excluded_samples_by_environment'], [1, 0])
        self.assertNotIn('force_lag', report['phases']['dynamic_loaded']['environment_hand'][0][0])


if __name__ == '__main__':
    unittest.main()
