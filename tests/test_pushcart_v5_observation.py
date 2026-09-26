"""Observed-motion and frame-convention checks for V5's shared adapter."""
from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v5.reaction_observation import ReactionObservation, rotate_planar


class ReactionObservationTests(unittest.TestCase):
    def adapter(self, n=1, device="cpu", **cfg):
        settings = dict(sensor_force_noise_std_n=0., sensor_moment_noise_std_nm=0.,
                        sensor_force_bias_std_n=0., sensor_moment_bias_std_nm=0.)
        settings.update(cfg)
        return ReactionObservation(settings, .02, n, device=device)

    def sample(self, n=1, device="cpu"):
        return dict(wrench_cart=torch.zeros(n, 6, device=device),
                    pose_base=torch.tensor([[.4, 0., .2, 0.]], device=device).repeat(n, 1),
                    base_velocity_est=torch.zeros(n, 3, device=device),
                    imu_gyro_b=torch.zeros(n, 3, device=device),
                    gravity_b=torch.tensor([[0., 0., -1.]], device=device).repeat(n, 1),
                    command_ref=torch.zeros(n, 3, device=device),
                    cart_mass_est=torch.full((n,), 20., device=device),
                    robot_mass_est=torch.full((n,), 35., device=device),
                    valid=torch.ones(n, dtype=torch.bool, device=device))

    def settle(self, q, x, steps=60):
        for _ in range(steps):
            q.update(**x)

    def test_rotation_conventions_and_round_trip(self):
        vector = torch.tensor([[1., 0., .7], [0., 1., -.2]])
        yaw = torch.tensor([math.pi / 2., -math.pi / 2.])
        rotated = rotate_planar(vector, yaw)
        torch.testing.assert_close(rotated, torch.tensor([[0., 1., .7], [1., 0., -.2]]), atol=1e-6, rtol=0.)
        torch.testing.assert_close(rotate_planar(rotated, -yaw), vector)

    def test_motion_and_sensor_warmup_are_separate_and_zero(self):
        q, x = self.adapter(sensor_delay_steps=2), self.sample()
        x["wrench_cart"][:, 0] = 40.
        for _ in range(q.motion.window_steps - 1):
            out = q.update(**x)
            self.assertIs(out, q.controller)
            self.assertFalse(q.ready.item())
            self.assertEqual(out.actor_features.abs().max().item(), 0.)
            self.assertEqual(q.estimated_cart_twist.abs().max().item(), 0.)
        q.update(**x)
        self.assertTrue(q.ready.item())
        self.assertEqual(q.controller.sensor_valid.item(), 0.)
        q.update(**x)
        q.update(**x)
        self.assertEqual(q.controller.sensor_valid.item(), 1.)
        self.assertGreater(q.controller.filtered_wrench[0, 0].item(), 0.)

    def test_common_translation_and_cart_yaw_transform(self):
        q, x = self.adapter(), self.sample()
        x["base_velocity_est"][:, 0] = .3
        x["pose_base"][:, 3] = math.pi / 2.
        self.settle(q, x)
        torch.testing.assert_close(q.estimated_cart_twist, torch.tensor([[0., -.3, 0.]]),
                                   atol=1e-5, rtol=0.)

    def test_observed_handle_translation_detected_without_base_motion(self):
        q, x = self.adapter(), self.sample()
        for step in range(80):
            x["pose_base"][:, 0] = .4 + .2 * step * .02
            q.update(**x)
        self.assertAlmostEqual(q.estimated_cart_twist[0, 0].item(), .2, places=5)

    def test_stationary_cart_cancels_observed_robot_translation(self):
        q, x = self.adapter(), self.sample()
        x["base_velocity_est"][:, 0] = .2
        for step in range(80):
            x["pose_base"][:, 0] = .4 - .2 * step * .02
            q.update(**x)
        self.assertLess(q.estimated_cart_twist.abs().max().item(), 1e-5)

    def test_cart_rotation_handle_lever_is_removed(self):
        q, x = self.adapter(), self.sample()
        # Robot and cart co-rotate at .4rad/s; handle offset relative to robot
        # is .4m, relative to cart COM is -.49m. Set base translation so handle
        # velocity is omega x [-.49,0] and cart COM itself remains stationary.
        x["imu_gyro_b"][:, 2] = .4
        x["base_velocity_est"][:, 1] = -.4 * (.49 + .4)
        self.settle(q, x)
        torch.testing.assert_close(q.estimated_cart_twist, torch.tensor([[0., 0., .4]]),
                                   atol=1e-5, rtol=0.)

    def test_nominal_turn_base_lever_does_not_trigger_spurious_velocity_recovery(self):
        q, x = self.adapter(force_admittance_gain=(0., 0., 0.)), self.sample()
        x["imu_gyro_b"][:, 2] = .4
        x["base_velocity_est"][:, 1] = -.4 * (.49 + .4)
        x["command_ref"][:, 2] = .4
        self.settle(q, x)
        self.assertEqual(q.controller.command_correction.abs().max().item(), 0.)
        self.assertEqual(q.controller.costs["reaction_motion"].item(), 0.)

    def test_reference_translates_cart_twist_to_base_point_before_motion_ready(self):
        q, x = self.adapter(), self.sample()
        q.update(**x)
        command = torch.tensor([[.3, 0., .4]])
        torch.testing.assert_close(q.reference_base(command), torch.tensor([[.3, -.356, .4]]))
        self.assertFalse(q.ready.item())
        q.reset()
        torch.testing.assert_close(q.reference_base(command), command)
        self.assertEqual(q.relative_yaw.item(), 0.)

    def test_reference_point_translation_rotates_with_measured_relative_heading(self):
        q, x = self.adapter(), self.sample()
        x["pose_base"][:] = torch.tensor([.4, 0., .2, math.pi / 2.])
        q.update(**x)
        # r_CB in cart axes=(-.49,+.4); omega x r=(-.16,-.196).
        expected = torch.tensor([[.196, .14, .4]])
        torch.testing.assert_close(q.reference_base(torch.tensor([[.3, 0., .4]])), expected)

    def test_invalid_pose_does_not_leave_command_reference_orientation_stale(self):
        q, x = self.adapter(), self.sample()
        x["pose_base"][:, 3] = .5
        q.update(**x)
        x["valid"].zero_()
        q.update(**x)
        self.assertFalse(q._pose_valid.item())

    def test_static_force_correction_transforms_back_into_homie_base_axes(self):
        q, x = self.adapter(), self.sample()
        x["pose_base"][:, 3] = math.pi / 2.
        x["wrench_cart"][:, 0] = 50.
        x["base_velocity_est"][:, 1] = -.2
        self.settle(q, x, 70)
        self.assertGreater(q.controller.command_correction[0, 0].item(), .08)
        self.assertGreater(q.command_correction_base[0, 1].item(), .08)
        self.assertLess(q.command_correction_base[0, 0].abs().item(), 1e-5)

    def test_imu_tilt_survives_yaw_rotation(self):
        q, x = self.adapter(), self.sample()
        x["pose_base"][:, 3] = math.pi / 2.
        x["gravity_b"][:] = torch.tensor([.3, 0., -.9539392])
        self.settle(q, x, 70)
        self.assertGreater(q.controller.recovery_gate.item(), .2)
        self.assertGreater(q.controller.costs["support"].item(), 1.)

    def test_invalid_observation_and_partial_reset_do_not_poison_batch(self):
        q, x = self.adapter(2), self.sample(2)
        x["wrench_cart"][:, 0] = 50.
        self.settle(q, x)
        x["pose_base"][0, 0] = float("nan")
        q.update(**x)
        self.assertFalse(q.ready[0])
        self.assertTrue(q.ready[1])
        self.assertEqual(q.controller.actor_features[0].abs().max().item(), 0.)
        self.assertEqual(q.command_correction_base[0].abs().max().item(), 0.)
        self.assertEqual(q.motion.samples[0].item(), 0)
        x["pose_base"][0, 0] = .4
        self.settle(q, x)
        self.assertTrue(q.ready.all())
        keep = q.controller.actor_features[1].clone(), q.motion.history[1].clone()
        q.reset(torch.tensor([0]))
        self.assertFalse(q.ready[0])
        self.assertEqual(q.estimated_cart_twist[0].abs().max().item(), 0.)
        self.assertEqual(q.command_correction_base[0].abs().max().item(), 0.)
        torch.testing.assert_close(q.controller.actor_features[1], keep[0])
        torch.testing.assert_close(q.motion.history[1], keep[1])

    def test_external_invalidity_flushes_otherwise_finite_motion_history(self):
        q, x = self.adapter(), self.sample()
        self.settle(q, x)
        x["valid"].zero_()
        q.update(**x)
        self.assertFalse(q.ready.item())
        self.assertEqual(q.motion.samples.item(), 0)
        x["valid"].fill_(True)
        q.update(**x)
        self.assertFalse(q.ready.item())
        self.assertEqual(q.motion.samples.item(), 1)

    def test_yaw_inertia_comes_from_configured_mass_geometry(self):
        q, x = self.adapter(yaw_radius_gyration_m=.5), self.sample()
        x["command_ref"][:, 2] = .4
        self.settle(q, x)
        # I=20*.5^2=5, alpha=1.5*.4=.6 -> required Mz=3Nm.
        self.assertAlmostEqual(q.controller.required_wrench[0, 2].item(), 3., places=5)

    def test_bad_geometry_configuration_rejected(self):
        for cfg in ({"motion_window_s": 0.}, {"yaw_radius_gyration_m": -1.},
                    {"yaw_radius_gyration_m": float("nan")}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                self.adapter(**cfg)
        with self.assertRaises(ValueError):
            ReactionObservation({}, .02, 1, handle_offset_cart=(1., 2.))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_adapter_controller_outputs_match(self):
        cpu, gpu = self.adapter(2), self.adapter(2, device="cuda")
        xc, xg = self.sample(2), self.sample(2, device="cuda")
        for x in (xc, xg):
            x["pose_base"][:, 3] = .3
            x["wrench_cart"][:, 0] = 50.
            x["base_velocity_est"][:, 0] = .1
            x["command_ref"][:, 0] = .2
        self.settle(cpu, xc, 50)
        self.settle(gpu, xg, 50)
        torch.testing.assert_close(gpu.estimated_cart_twist.cpu(), cpu.estimated_cart_twist)
        torch.testing.assert_close(gpu.controller.actor_features.cpu(), cpu.controller.actor_features)
        torch.testing.assert_close(gpu.command_correction_base.cpu(), cpu.command_correction_base)


if __name__ == "__main__":
    unittest.main()
