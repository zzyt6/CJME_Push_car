"""Physics/sign, timing and reset checks for the shared V5 reaction runtime."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v5.reaction import ACTOR_FEATURE_DIM, ReactionController


class ReactionControllerTests(unittest.TestCase):
    def controller(self, n=1, **cfg):
        options = dict(sensor_force_noise_std_n=0., sensor_moment_noise_std_nm=0.,
                       sensor_force_bias_std_n=0., sensor_moment_bias_std_nm=0.)
        options.update(cfg)
        return ReactionController(options, .02, n)

    def sample(self, n=1):
        return dict(hand_force_b=torch.zeros(n, 3), hand_moment_b=torch.zeros(n, 3),
                    cart_velocity_b=torch.zeros(n, 3), command_ref=torch.zeros(n, 3),
                    robot_velocity_b=torch.zeros(n, 3), robot_omega_b=torch.zeros(n, 3),
                    gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
                    cart_mass=torch.full((n,), 20.), cart_yaw_inertia=torch.full((n,), 2.),
                    robot_mass=torch.full((n,), 35.), valid=torch.ones(n, dtype=torch.bool))

    def settle(self, controller, sample, steps=100):
        for _ in range(steps):
            controller.update(**sample)

    def test_stationary_zero_is_exactly_closed_and_finite(self):
        q, x = self.controller(), self.sample()
        self.settle(q, x, 300)
        self.assertEqual(q.actor_features.shape, (1, ACTOR_FEATURE_DIM))
        self.assertEqual(q.sensor_valid.item(), 1.)
        for value in (q.required_wrench, q.excess_wrench, q.command_correction,
                      q.recovery_gate, *q.costs.values()):
            self.assertEqual(value.abs().max().item(), 0.)

    def test_recovery_force_is_part_of_required_task_wrench(self):
        q = self.controller(sensor_delay_steps=0, viscous_resistance=(0., 0., 0.),
                            rolling_resistance=(0., 0., 0.))
        x = self.sample()
        x["command_ref"][:, 0] = .4
        x["robot_velocity_b"][:, 0] = .4
        x["hand_force_b"][:, 0] = 12.  # 20kg * 1.5/s * .4m/s.
        self.settle(q, x)
        self.assertAlmostEqual(q.required_wrench[0, 0].item(), 12., places=5)
        self.assertLess(q.excess_wrench.abs().max().item(), 1e-5)
        self.assertEqual(q.costs["task_wrench"].item(), 0.)
        self.assertEqual(q.command_correction.abs().max().item(), 0.)

    def test_planar_transport_has_correct_centripetal_sign(self):
        q = self.controller(viscous_resistance=(0., 0., 0.), rolling_resistance=(0., 0., 0.))
        x = self.sample()
        x["command_ref"][:] = torch.tensor([.5, 0., .4])
        x["cart_velocity_b"][:] = x["command_ref"]
        # Robot gyro intentionally differs: frame transport follows CART yaw.
        x["robot_omega_b"][:, 2] = -1.
        q.update(**x)
        torch.testing.assert_close(q.required_wrench, torch.tensor([[0., 4., 0.]]))

    def test_independent_resistance_model_does_not_cancel_measured_hand_force(self):
        q, x = self.controller(), self.sample()
        x["command_ref"][:, 0] = .3
        x["cart_velocity_b"][:, 0] = .3
        x["robot_velocity_b"][:, 0] = .3
        self.settle(q, x)
        before = q.required_wrench.clone()
        x["hand_force_b"][:, 0] = 80.
        self.settle(q, x)
        torch.testing.assert_close(q.required_wrench, before)
        self.assertGreater(q.excess_wrench[0, 0].item(), 70.)
        self.assertGreater(q.costs["task_wrench"].item(), .5)

    def test_force_compensation_opposes_robot_reaction_and_release_decays(self):
        q, x = self.controller(), self.sample()
        self.settle(q, x, 5)
        # Robot-on-cart +Fx means force ON ROBOT is negative: restore forward.
        x["hand_force_b"][:, 0] = 50.
        x["robot_velocity_b"][:, 0] = -.2
        self.settle(q, x, 50)
        self.assertGreater(q.command_correction[0, 0].item(), .08)
        self.assertGreater(q.recovery_gate.item(), .2)
        self.assertLessEqual(q.command_correction[0, 0].item(), .120001)
        old = q.command_correction.clone()
        x["hand_force_b"].zero_()
        x["robot_velocity_b"].zero_()
        for _ in range(200):
            q.update(**x)
            self.assertLessEqual((q.command_correction - old).abs().max().item(), .012001)
            old.copy_(q.command_correction)
        self.assertLess(q.command_correction.abs().max().item(), 1e-5)
        self.assertEqual(q.recovery_gate.item(), 0.)
        # Calm interval rearms a later perturbation even after arbitrarily long idle.
        x["hand_force_b"][:, 0] = -50.
        x["robot_velocity_b"][:, 0] = .2
        self.settle(q, x, 50)
        self.assertLess(q.command_correction[0, 0].item(), -.08)

    def test_constant_idle_bias_cannot_produce_perpetual_command(self):
        q, x = self.controller(), self.sample()
        x["hand_force_b"][:, 0] = 80.
        x["robot_velocity_b"][:, 0] = .2
        self.settle(q, x, 500)
        self.assertEqual(q.command_correction.abs().max().item(), 0.)
        self.assertEqual(q.recovery_gate.item(), 0.)
        self.assertGreater(q.excess_wrench[0, 0].item(), 70.)

    def test_delayed_force_and_reference_remain_aligned(self):
        q = self.controller(sensor_delay_steps=3, viscous_resistance=(0., 0., 0.),
                            rolling_resistance=(0., 0., 0.))
        x = self.sample()
        for step in range(100):
            # First-sample derivative is zero; subsequent reference is constant.
            x["command_ref"][:, 0] = .4
            x["cart_velocity_b"][:, 0] = .4 * step / 100.
            x["hand_force_b"][:, 0] = 20. * 1.5 * (.4 - x["cart_velocity_b"][:, 0])
            q.update(**x)
            self.assertEqual(q.sensor_valid.item(), float(step >= 3))
            self.assertLess(q.excess_wrench.abs().max().item(), 2e-6)

    def test_reference_acceleration_braking_and_saturation(self):
        q, x = self.controller(), self.sample()
        q.update(**x)
        x["command_ref"][:, 0] = .6
        q.update(**x)
        self.assertAlmostEqual(q.required_wrench[0, 0].item(), 24., places=4)
        self.settle(q, x, 100)
        x["cart_velocity_b"][:, 0] = .6
        x["command_ref"].zero_()
        q.update(**x)
        self.assertLess(q.required_wrench[0, 0].item(), -20.)

    def test_small_model_error_and_nominal_yaw_motion_are_not_punished(self):
        q, x = self.controller(), self.sample()
        x["command_ref"][:] = torch.tensor([.3, 0., .5])
        x["cart_velocity_b"][:] = x["command_ref"]
        x["robot_velocity_b"][:, 0] = .3
        x["robot_omega_b"][:, 2] = .5
        self.settle(q, x)
        x["hand_force_b"][:, :2] = q.required_wrench[:, :2] + 2.
        x["hand_moment_b"][:, 2] = q.required_wrench[:, 2] + .2
        self.settle(q, x)
        self.assertEqual(q.costs["reaction_motion"].item(), 0.)
        self.assertEqual(q.costs["task_wrench"].item(), 0.)

    def test_imu_motion_can_recover_at_initial_idle_without_force(self):
        q, x = self.controller(), self.sample()
        x["robot_velocity_b"][:, 0] = -.3
        x["robot_omega_b"][:, 1] = .5
        self.settle(q, x, 40)
        self.assertGreater(q.recovery_gate.item(), .2)
        self.assertGreater(q.command_correction[0, 0].item(), .04)
        self.assertGreater(q.costs["reaction_motion"].item(), .1)

    def test_partial_reset_flushes_delayed_force_without_changing_other_row(self):
        q, x = self.controller(2, sensor_delay_steps=3), self.sample(2)
        x["hand_force_b"][:, 0] = 50.
        self.settle(q, x, 30)
        keep = q.actor_features[1].clone(), q._history[1].clone()
        q.reset(torch.tensor([0]))
        self.assertEqual(q.actor_features[0].abs().max().item(), 0.)
        self.assertEqual(q._history[0].abs().max().item(), 0.)
        torch.testing.assert_close(q.actor_features[1], keep[0])
        torch.testing.assert_close(q._history[1], keep[1])
        x["hand_force_b"][0] = 0.
        self.settle(q, x, 20)
        self.assertEqual(q.filtered_wrench[0].abs().max().item(), 0.)
        self.assertEqual(q.command_correction[0].abs().max().item(), 0.)
        self.assertGreater(q.filtered_wrench[1, 0].item(), 49.)

    def test_invalid_input_clears_every_channel_and_recovers_independently(self):
        for invalid_name in ("hand_force_b", "command_ref", "gravity_b", "cart_mass", "valid"):
            with self.subTest(invalid_name=invalid_name):
                q, x = self.controller(2), self.sample(2)
                x["hand_force_b"][:, 0] = 50.
                self.settle(q, x, 20)
                if invalid_name == "valid":
                    x[invalid_name][0] = False
                else:
                    x[invalid_name][0] = float("nan")
                q.update(**x)
                self.assertEqual(q.sensor_valid[0].item(), 0.)
                self.assertEqual(q.actor_features[0].abs().max().item(), 0.)
                self.assertEqual(q._history[0].abs().max().item(), 0.)
                self.assertEqual(q.sensor_valid[1].item(), 1.)
                self.assertTrue(torch.isfinite(q.actor_features).all())

    def test_noisy_sensor_has_explicit_saturation_and_validity_channel(self):
        torch.manual_seed(3)
        q = self.controller(sensor_force_noise_std_n=1., sensor_moment_noise_std_nm=.1,
                            sensor_force_clip_n=40., sensor_delay_steps=0)
        x = self.sample()
        x["hand_force_b"][:, 0] = 1e6
        self.settle(q, x, 100)
        self.assertLessEqual(q.filtered_wrench[0, 0].item(), 40.)
        self.assertGreater(q.filtered_wrench[0, 1].abs().item(), 0.)
        self.assertEqual(q.actor_features[0, -1].item(), 1.)

    def test_support_force_uses_robot_reaction_sign_and_true_vertical_gravity(self):
        q, x = self.controller(sensor_delay_steps=0), self.sample()
        # Robot-on-cart upward force means robot feels DOWNWARD loading.
        x["hand_force_b"][:] = torch.tensor([100., 0., 50.])
        self.settle(q, x)
        expected = 100. / (.6 * (35. * 9.81 + 50.))
        self.assertAlmostEqual(q.support_utilization.item(), expected, places=5)
        before = q.support_utilization.clone()
        x["gravity_b"][:] = torch.tensor([.3, 0., -.9539392])
        self.settle(q, x)
        torch.testing.assert_close(q.support_utilization, before)
        self.assertGreater(q.costs["support"].item(), 1.)

    def test_pure_upward_robot_load_cannot_claim_feasible_negative_ground_normal(self):
        q, x = self.controller(sensor_delay_steps=0), self.sample()
        # Robot-on-cart negative Fz is an upward force ON the robot, larger
        # than its weight. Sticking support would require tensile feet force.
        x["hand_force_b"][:, 2] = -500.
        self.settle(q, x)
        self.assertEqual(q.support_utilization.item(), 0.)
        self.assertGreater(q.costs["support"].item(), .2)

    def test_disabled_compensation_preserves_diagnostics(self):
        q, x = self.controller(compensation_enabled=False), self.sample()
        x["hand_force_b"][:, 0] = 80.
        self.settle(q, x, 40)
        self.assertEqual(q.command_correction.abs().max().item(), 0.)
        self.assertGreater(q.costs["task_wrench"].item(), .1)

    def test_bad_configuration_or_shapes_are_rejected(self):
        for cfg in ({"sensor_delay_steps": -1}, {"sensor_delay_steps": .5},
                    {"correction_limit": (1., 2.)}, {"gate_max": 1.1},
                    {"sensor_filter_tau_s": 0.}, {"uncertainty_fraction": float("nan")}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                self.controller(**cfg)
        q, x = self.controller(), self.sample()
        x["hand_force_b"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            q.update(**x)

    def test_sensor_history_rotates_into_current_heading_before_delay_and_filter(self):
        q, x = self.controller(sensor_delay_steps=2), self.sample()
        x["hand_force_b"][:] = torch.tensor([40., 10., 5.])
        x["hand_moment_b"][:] = torch.tensor([2., 3., 4.])
        self.settle(q, x)
        x["hand_force_b"][:] = torch.tensor([10., -40., 5.])
        x["hand_moment_b"][:] = torch.tensor([3., -2., 4.])
        q.update(**x, heading_delta_rad=torch.tensor([torch.pi / 2.]))
        torch.testing.assert_close(q.filtered_wrench,
            torch.tensor([[10., -40., 5., 3., -2., 4.]]), atol=2e-5, rtol=0.)
        torch.testing.assert_close(q._history[0, :, :6],
            torch.tensor([[10., -40., 5., 3., -2., 4.]]).expand(3, -1), atol=2e-5, rtol=0.)

    def test_world_fixed_velocity_reference_does_not_invent_acceleration_during_yaw(self):
        q = self.controller(viscous_resistance=(0., 0., 0.), rolling_resistance=(0., 0., 0.))
        x = self.sample()
        x["command_ref"][:, 0] = .4
        x["cart_velocity_b"][:, 0] = .4
        q.update(**x, heading_delta_rad=torch.zeros(1))
        for step in range(1, 41):
            yaw = .03 * step
            x["command_ref"][0] = torch.tensor([.4 * math.cos(yaw), -.4 * math.sin(yaw), 0.])
            x["cart_velocity_b"][0, :2] = x["command_ref"][0, :2]
            # Actual frame motion is supplied separately from desired yaw-rate.
            x["cart_velocity_b"][:, 2] = 1.5
            q.update(**x, heading_delta_rad=torch.tensor([.03]))
            self.assertLess(q.required_wrench[0, :2].abs().max().item(), 5e-5)

    def test_force_alone_opens_recovery_gate_without_driving_into_stiff_load(self):
        q, x = self.controller(), self.sample()
        x["hand_force_b"][:, 0] = 50.
        self.settle(q, x, 50)
        self.assertGreater(q.recovery_gate.item(), .2)
        self.assertEqual(q.command_correction.abs().max().item(), 0.)
        self.assertGreater(q.actor_features[0, 0].item(), .5)

    def test_force_cannot_override_restoring_direction_when_robot_overspeeds(self):
        q, x = self.controller(), self.sample()
        x["command_ref"][:, 0] = .3
        x["cart_velocity_b"][:, 0] = .3
        x["robot_velocity_b"][:, 0] = .5
        x["hand_force_b"][:, 0] = 80.
        self.settle(q, x)
        self.assertLess(q.command_correction[0, 0].item(), -.04)

    def test_preloaded_spring_does_not_self_amplify_force_at_nominal_motion(self):
        q, x = self.controller(sensor_delay_steps=0), self.sample()
        x["command_ref"][:, 0] = .3
        x["cart_velocity_b"][:, 0] = .3
        preload = .04  # 500N/m * .04m = 20N initial contact load.
        for _ in range(150):
            x["hand_force_b"][:, 0] = 500. * preload
            x["robot_velocity_b"][:, 0] = .3 + q.command_correction[:, 0]
            q.update(**x)
            preload += q.dt * q.command_correction[0, 0].item()
        self.assertAlmostEqual(preload * 500., 20., places=6)
        self.assertEqual(q.command_correction.abs().max().item(), 0.)


if __name__ == "__main__":
    unittest.main()
