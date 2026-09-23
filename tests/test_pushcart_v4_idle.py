"""Idle correction uses observed motion, including realistic observation noise."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart.idle_correction import IdleCorrectionTracker


class IdleCorrectionTests(unittest.TestCase):
    def tracker(self, n=1, **kwargs):
        return IdleCorrectionTracker(SimpleNamespace(**kwargs), .02, n)

    def inputs(self, n=1):
        return (torch.tensor([[.4, 0., .15]]).repeat(n, 1), torch.zeros(n),
                torch.zeros(n, 3), torch.zeros(n, 3),
                torch.tensor([[0., 0., -1.]]).repeat(n, 1))

    def test_no_motion_is_exactly_closed_and_gate_is_read_only(self):
        q = self.tracker()
        for _ in range(150):
            q.update(*self.inputs())
        zero = torch.zeros(1, 3)
        self.assertTrue(q.ready.item())
        self.assertEqual(q.gate(zero, zero).item(), 0.)
        before = q.samples.clone(), q.history.clone(), q.gate_level.clone()
        for _ in range(10):
            q.gate(zero, zero)
        for old, actual in zip(before, (q.samples, q.history, q.gate_level)):
            torch.testing.assert_close(old, actual)

    def test_common_robot_cart_translation_opens_bounded_gate(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        vel[:, 0] = .2
        for _ in range(100):
            q.update(pose, yaw, vel, gyro, g, task_command=vel)
        zero = torch.zeros(1, 3)
        self.assertGreater(q.gate(zero, zero).item(), .24)
        self.assertLessEqual(q.gate(zero, zero).item(), .25)
        self.assertAlmostEqual(q.estimated_twist[0, 0].item(), .2, places=5)
        self.assertEqual(q.gate(zero, zero + .1).item(), 0.)
        self.assertLessEqual(q.gate(zero, zero).item() * .4, .1)

    def test_relative_cart_motion_is_detected_when_robot_stands(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        for i in range(80):
            p = pose.clone()
            p[:, 0] += .2 * i * q.dt
            q.update(p, yaw, vel, gyro, g)
        self.assertTrue(q.active.item())
        self.assertAlmostEqual(q.estimated_twist[0, 0].item(), .2, places=5)

    def test_robot_translation_with_stationary_cart_does_not_open_gate(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        vel[:, 0] = .2
        for i in range(80):
            p = pose.clone()
            p[:, 0] -= .2 * i * q.dt
            q.update(p, yaw, vel, gyro, g)
        self.assertFalse(q.active.item())
        self.assertLess(q.estimated_twist.abs().max().item(), 1e-5)

    def test_robot_rotation_with_stationary_cart_cancels_lever_arm_motion(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        gyro[:, 2] = .6
        for i in range(100):
            a = -.6 * i * q.dt
            p = pose.clone()
            p[:, 0], p[:, 1] = .4 * math.cos(a), .4 * math.sin(a)
            q.update(p, yaw + a, vel, gyro, g)
        self.assertFalse(q.active.item())
        self.assertLess(q.estimated_twist.abs().max().item(), 1e-4)

    def test_relative_yaw_wrap_does_not_create_an_impulse(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        for i in range(100):
            a = 3.0 + .4 * i * q.dt
            wrapped = math.atan2(math.sin(a), math.cos(a))
            q.update(pose, yaw + wrapped, vel, gyro, g)
        self.assertTrue(q.active.item())
        self.assertAlmostEqual(q.estimated_twist[0, 2].item(), .4, places=5)

    def test_observed_gravity_corrects_euler_yaw_rate_at_tilt(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        pitch, roll, rate = .3, .2, .6
        g[:] = torch.tensor([math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                             -math.cos(roll) * math.cos(pitch)])
        gyro[:] = -rate * g
        for i in range(100):
            q.update(pose, yaw - rate * i * q.dt, vel, gyro, g)
        self.assertLess(abs(q.estimated_twist[0, 2].item()), 1e-5)

    def test_bad_configuration_is_rejected(self):
        for cfg in ({"max_gate": 1.1}, {"window_s": 0.}, {"motion_off_m_s": .2},
                    {"gate_rise_tau_s": float("nan")}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                self.tracker(**cfg)

    def test_hysteresis_smooth_fade_and_independent_reset(self):
        q = self.tracker(2)
        pose, yaw, vel, gyro, g = self.inputs(2)
        vel[:, 0] = .2
        for _ in range(100):
            q.update(pose, yaw, vel, gyro, g)
        keep = q.history[1].clone(), q.gate_level[1].clone()
        q.reset(torch.tensor([0]))
        self.assertEqual(q.gate_level[0].item(), 0.)
        self.assertEqual(q.samples[0].item(), 0)
        torch.testing.assert_close(q.history[1], keep[0])
        torch.testing.assert_close(q.gate_level[1], keep[1])
        vel.zero_()
        old = q.gate_level.clone()
        largest_step = 0.
        for _ in range(200):
            q.update(pose, yaw, vel, gyro, g)
            largest_step = max(largest_step, (q.gate_level - old).abs().max().item())
            old = q.gate_level.clone()
        self.assertLess(largest_step, .03)
        self.assertFalse(q.active.any())
        self.assertLess(q.gate_level.max().item(), 1e-4)

    def test_invalid_observation_breaks_window_and_cannot_poison_other_envs(self):
        q = self.tracker(2)
        pose, yaw, vel, gyro, g = self.inputs(2)
        vel[:, 0] = .2
        for _ in range(60):
            q.update(pose, yaw, vel, gyro, g)
        bad = pose.clone()
        bad[0, 0] = float("nan")
        q.update(bad, yaw, vel, gyro, g)
        self.assertFalse(q.ready[0])
        self.assertEqual(q.gate_level[0].item(), 0.)
        self.assertTrue(q.ready[1])
        self.assertTrue(torch.isfinite(q.history).all())
        for _ in range(60):
            q.update(pose, yaw, vel, gyro, g)
        self.assertTrue(q.active.all())

    def test_full_reset_clears_state_and_reseeds_first_observation(self):
        q = self.tracker(2)
        pose, yaw, vel, gyro, g = self.inputs(2)
        vel[:, 0] = .2
        for _ in range(100):
            q.update(pose, yaw, vel, gyro, g, task_command=vel)
        q.reset()
        self.assertFalse(q.ready.any())
        self.assertEqual(q.history.abs().max().item(), 0.)
        self.assertEqual(q.gate_level.max().item(), 0.)
        self.assertEqual(q.recovery_remaining.max().item(), 0.)
        q.update(*self.inputs(2))
        self.assertEqual(q.estimated_twist.abs().max().item(), 0.)

    def test_initial_idle_cannot_open_gate_with_persistent_estimator_bias(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        vel[:, 0] = .3
        zero = torch.zeros(1, 3)
        for _ in range(400):
            q.update(pose, yaw, vel, gyro, g, task_command=zero)
            self.assertEqual(q.gate(zero, zero).item(), 0.)
        self.assertTrue(q.active.item())  # Detector sees bias; command interlock blocks it.
        self.assertEqual(q.recovery_remaining.item(), 0.)

    def test_stop_recovery_expires_smoothly_even_when_motion_or_bias_persists(self):
        q = self.tracker(recovery_window_s=1.)
        pose, yaw, vel, gyro, g = self.inputs()
        vel[:, 0] = .3
        zero = torch.zeros(1, 3)
        for _ in range(100):
            q.update(pose, yaw, vel, gyro, g, task_command=vel)
        self.assertGreater(q.gate(zero, zero).item(), .24)
        previous, largest_step = q.gate(zero, zero).item(), 0.
        for _ in range(300):
            q.update(pose, yaw, vel, gyro, g, task_command=zero)
            current = q.gate(zero, zero).item()
            largest_step = max(largest_step, abs(current - previous))
            previous = current
        self.assertEqual(q.gate(zero, zero).item(), 0.)
        self.assertEqual(q.recovery_remaining.item(), 0.)
        self.assertLess(largest_step, .025)
        self.assertTrue(q.active.item())
        # A later real command may arm a new stop; idle alone cannot reopen it.
        q.update(pose, yaw, vel, gyro, g, task_command=vel)
        self.assertGreater(q.gate(zero, zero).item(), .24)

    def test_command_without_observed_motion_does_not_open_stop_recovery(self):
        q = self.tracker()
        pose, yaw, vel, gyro, g = self.inputs()
        command = torch.tensor([[.3, 0., 0.]])
        for _ in range(100):
            q.update(pose, yaw, vel, gyro, g, task_command=command)
        self.assertEqual(q.gate(vel, vel).item(), 0.)
        self.assertGreater(q.recovery_remaining.item(), 0.)

    def test_full_observation_noise_stationary_false_activation_is_rare(self):
        # Isaac full curriculum: Gaussian 2 cm / 2 deg pose noise; uniform
        # +/- .05 m/s estimate, +/- .5 rad/s gyro and +/- .05 gravity noise.
        # This tests white-noise assumptions; estimator bias needs real calibration.
        gen = torch.Generator().manual_seed(427)
        n, steps = 32, 1500
        q = self.tracker(n)
        pose, yaw, vel, gyro, g = self.inputs(n)
        active_samples = ready_samples = 0
        for _ in range(steps):
            gaussian = lambda shape: torch.randn(shape, generator=gen)
            uniform = lambda shape: 2 * torch.rand(shape, generator=gen) - 1
            q.update(pose + .02 * gaussian(pose.shape),
                     yaw + math.radians(2) * gaussian(yaw.shape),
                     vel + .05 * uniform(vel.shape), gyro + .5 * uniform(gyro.shape),
                     g + .05 * uniform(g.shape))
            active_samples += int(q.active.sum())
            ready_samples += int(q.ready.sum())
        self.assertLess(active_samples / ready_samples, .005)

    def test_motion_detection_latency_with_full_noise(self):
        gen = torch.Generator().manual_seed(249)
        n = 32
        q = self.tracker(n)
        pose, yaw, vel, gyro, g = self.inputs(n)
        latency = torch.full((n,), float("inf"))
        for i in range(150):
            gaussian = lambda shape: torch.randn(shape, generator=gen)
            uniform = lambda shape: 2 * torch.rand(shape, generator=gen) - 1
            v = vel.clone()
            v[:n // 2, 0] = .2 if i >= 50 else 0.
            rel = yaw.clone()
            rel[n // 2:] = .4 * max(0, i - 50) * q.dt
            q.update(pose + .02 * gaussian(pose.shape), rel + math.radians(2) * gaussian(yaw.shape),
                     v + .05 * uniform(vel.shape), gyro + .5 * uniform(gyro.shape),
                     g + .05 * uniform(g.shape))
            if i >= 50:
                latency[(~torch.isfinite(latency)) & q.active] = (i - 50) * q.dt
        self.assertLessEqual(latency.max().item(), 1.)


if __name__ == "__main__":
    unittest.main()
