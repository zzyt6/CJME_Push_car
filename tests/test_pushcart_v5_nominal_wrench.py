"""Independent offline load diagnosis must not use hand-force truth as input."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))
from homie_pushcart_v5.nominal_wrench_diagnostics import (
    independent_control_interval_load, excitation_diagnostic, shift_spatial, to_heading,
)


def trace():
    dt = .005
    t = np.arange(1, 13)*dt
    hand = np.tile([10., 4., 0., 0., 0., 2.], (len(t), 1))
    ground = np.tile([-3., -1., 0., 0., 0., -.4], (len(t), 1))
    applied = np.tile([2., 0., 0., 0., 0., .3], (len(t), 1))
    momentum = t[:, None]*(hand+ground+applied)
    com = np.tile([2., 3., .3], (len(t), 1))
    return dict(time_s=np.array([.02, .04, .06]), cart_yaw=np.full(3, np.pi/2),
                required_wrench=np.tile([4., -10., 2.], (3, 1)), physics_time_s=t,
                physics_cart_momentum=momentum, physics_cart_com=com,
                physics_force_cart_com=com.copy(), physics_wheel_wrench=ground[:, None],
                physics_cart_applied_reference=com.copy(), physics_cart_applied_wrench=applied,
                physics_hand_wrench=hand, physics_dt_s=dt,
                physics_force_interval_start_s=t-dt, physics_ground_valid=np.ones(len(t), bool))


class OfflineWrenchDiagnosticTests(unittest.TestCase):
    def test_independent_reconstruction_accounts_ground_external_and_heading(self):
        result = independent_control_interval_load(trace())
        np.testing.assert_allclose(result['demand'], [[4., -10., 2.]]*2, atol=1.e-12)
        np.testing.assert_allclose(result['closure_error'], 0., atol=1.e-12)
        np.testing.assert_allclose(result['prediction_error'], 0., atol=1.e-12)

    def test_reconstructed_load_has_no_hand_force_input_leakage(self):
        source = trace()
        expected = independent_control_interval_load(source)['demand']
        source['physics_hand_wrench'][:] = 1.e5
        result = independent_control_interval_load(source)
        np.testing.assert_allclose(result['demand'], expected)
        self.assertGreater(abs(result['closure_error']).max(), 1.e4)

    def test_shifted_force_origins_and_global_translation_do_not_change_result(self):
        source = trace()
        expected = independent_control_interval_load(source)['demand']
        old = source['physics_force_cart_com'].copy()
        new = old+np.array([.1, -.2, 0.])
        source['physics_wheel_wrench'][:, 0] = shift_spatial(source['physics_wheel_wrench'][:, 0], old, new)
        source['physics_hand_wrench'] = shift_spatial(source['physics_hand_wrench'], old, new)
        source['physics_force_cart_com'] = new
        for key in ('physics_force_cart_com', 'physics_cart_com', 'physics_cart_applied_reference'):
            source[key] += [4., 5., 0.]
        result = independent_control_interval_load(source)
        np.testing.assert_allclose(result['demand'], expected, atol=1.e-12)
        np.testing.assert_allclose(result['closure_error'], 0., atol=1.e-12)

    def test_skipped_physics_frame_or_reset_is_rejected(self):
        source = trace()
        source['physics_time_s'][5] += .002
        with self.assertRaisesRegex(ValueError, 'continuous'):
            independent_control_interval_load(source)

    def test_moving_com_momentum_transport_does_not_invent_external_torque(self):
        source = trace()
        t = source['physics_time_s']
        com = np.stack((t, 2.*t, 0.*t), -1)
        linear = np.tile([3., 4., 0.], (len(t), 1))
        # Constant momentum/angular momentum about world origin, represented
        # about different COM reference points. Frame algebra must cancel.
        angular = np.tile([0., 0., 7.], (len(t), 1))-np.cross(com, linear)
        source['physics_cart_momentum'] = np.concatenate((linear, angular), -1)
        source['physics_cart_com'] = com
        source['physics_wheel_wrench'][:] = 0.
        source['physics_cart_applied_wrench'][:] = 0.
        source['physics_hand_wrench'][:] = 0.
        result = independent_control_interval_load(source)
        np.testing.assert_allclose(result['demand'], 0., atol=1.e-12)

    def test_control_target_predicts_next_interval_not_previous_interval(self):
        source = trace()
        source['required_wrench'][0] = [1., 2., 3.]
        source['required_wrench'][1] = [4., 5., 6.]
        result = independent_control_interval_load(source)
        np.testing.assert_array_equal(result['control_index'], [0, 1])
        np.testing.assert_allclose(result['start_s'], [.02, .04])
        np.testing.assert_allclose(result['end_s'], [.04, .06])
        np.testing.assert_allclose(result['prediction'], [[1., 2., 3.], [4., 5., 6.]])

    def test_narrow_straight_trace_refuses_identification_and_never_fits(self):
        result = excitation_diagnostic(np.tile([.3, 0., 0.], (200, 1)), np.ones(200, bool), [.05, .05, .1])
        self.assertFalse(result['fit_performed'])
        self.assertFalse(result['parameters_promoted'])
        self.assertTrue(all(not axis['identifiable'] for axis in result['axes'].values()))


if __name__ == '__main__':
    unittest.main()
