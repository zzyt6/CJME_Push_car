"""Optional inertial-history observer: geometry, timing, causality and comparison."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v5.motion_observer import InertialCartMotionObserver, compare_motion_estimators


def trace(steps=200, dt=.02, base_v=(0., 0., 0.), cart_v=(.2, .04, 0.),
          base_yaw_rate=1.1, cart_yaw_rate=0., delay=0, tilted=False):
    """Exact synthetic observations; evaluate cart truth only outside observer."""
    result = {key: [] for key in ("handle_pose_b", "base_velocity_b", "gyro_b", "gravity_b", "truth_cart_twist_base")}
    poses = []
    for index in range(steps):
        t = index * dt
        roll, pitch = (.2 * math.sin(.4 * t), .15 * math.cos(.3 * t)) if tilted else (0., 0.)
        rd, pd = (.08 * math.cos(.4 * t), -.045 * math.sin(.3 * t)) if tilted else (0., 0.)
        yaw, cyaw = base_yaw_rate * t, 3. + cart_yaw_rate * t
        cr, sr, cp, sp, c, s = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
        ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
        rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        rot = rz @ ry @ rx
        cartrot = np.array([[math.cos(cyaw), -math.sin(cyaw), 0.], [math.sin(cyaw), math.cos(cyaw), 0.], [0., 0., 1.]])
        cartpos, basepos = np.array([.8, .1, .3]) + t * np.array(cart_v), np.array([0., 0., .7]) + t * np.array(base_v)
        handle = cartpos + cartrot @ np.array([-.49, 0., .6])
        poses.append(np.r_[rot.T @ (handle - basepos), math.atan2(math.sin(cyaw - yaw), math.cos(cyaw - yaw))])
        result["handle_pose_b"].append(poses[max(0, index - delay)])
        result["base_velocity_b"].append(rot.T @ np.array(base_v))
        result["gyro_b"].append([rd - base_yaw_rate * sp,
                                 pd * cr + base_yaw_rate * sr * cp,
                                 -pd * sr + base_yaw_rate * cr * cp])
        result["gravity_b"].append(rot.T @ np.array([0., 0., -1.]))
        v = rz.T @ np.array(cart_v)
        result["truth_cart_twist_base"].append([v[0], v[1], cart_yaw_rate])
    result = {key: np.array(value) for key, value in result.items()}
    result["pose_delay_steps"] = delay
    return result


class InertialMotionObserverTests(unittest.TestCase):
    def test_world_fixed_velocity_is_estimated_in_one_consistent_inertial_window(self):
        out = compare_motion_estimators(trace())
        self.assertGreater(out["metrics"]["sample_count"], 150)
        self.assertLess(max(out["metrics"]["candidate_rmse_instantaneous"]), 1e-10)
        self.assertGreater(out["metrics"]["baseline_rmse_instantaneous"][0], .01)

    def test_stationary_cart_cancels_translating_rotating_robot(self):
        out = compare_motion_estimators(trace(base_v=(.2, -.1, 0.), cart_v=(0., 0., 0.)))
        self.assertLess(max(out["metrics"]["candidate_rmse_instantaneous"]), 2e-5)

    def test_delayed_pose_uses_capture_time_base_pose_and_gravity(self):
        sample = trace(base_v=(.2, -.1, 0.), delay=3, tilted=True)
        out = compare_motion_estimators(sample)
        self.assertLess(max(out["metrics"]["candidate_rmse_instantaneous"]), 2e-5)
        ready = out["candidate_ready"]
        torch.testing.assert_close(out["candidate_sample_age_s"][ready], torch.full_like(out["candidate_sample_age_s"][ready], .06))
        torch.testing.assert_close(out["candidate_fit_center_age_s"][ready], torch.full_like(out["candidate_fit_center_age_s"][ready], .26))
        first = int(ready[:, 0].nonzero()[0])
        self.assertEqual(first, 23)  # 3 latency samples plus 20-step fit span.

    def test_rotating_cart_handle_lever_and_yaw_wrap_do_not_create_translation(self):
        out = compare_motion_estimators(trace(cart_v=(0., 0., 0.), cart_yaw_rate=1.8, delay=2))
        self.assertLess(max(out["metrics"]["candidate_rmse_instantaneous"]), 1e-10)

    def test_pose_noise_is_bounded_but_not_claimed_to_be_removed(self):
        sample = trace(base_yaw_rate=.7, tilted=True, delay=2)
        rng = np.random.default_rng(12)
        sample["handle_pose_b"][:, :3] += rng.normal(0., .005, (200, 3))
        sample["handle_pose_b"][:, 3] += rng.normal(0., .01, 200)
        out = compare_motion_estimators(sample)
        error = out["metrics"]["candidate_rmse_instantaneous"]
        self.assertTrue(all(0. < x < .06 for x in error), error)

    def test_acceleration_exposes_window_latency_instead_of_claiming_current_truth(self):
        sample = trace(base_yaw_rate=0., delay=2)
        times = np.arange(200) * .02
        captured_times = np.maximum(np.arange(200) - 2, 0) * .02
        acceleration = .3
        sample["handle_pose_b"][:, 0] += .5 * acceleration * captured_times ** 2
        sample["truth_cart_twist_base"][:, 0] += acceleration * times
        out = compare_motion_estimators(sample)
        ready = out["candidate_ready"][:, 0]
        error = out["candidate_twist_base"][:, 0, 0] - torch.tensor(sample["truth_cart_twist_base"][:, 0])
        expected = -acceleration * out["candidate_fit_center_age_s"][:, 0]
        torch.testing.assert_close(error[ready], expected[ready], atol=1e-12, rtol=0.)
        self.assertAlmostEqual(out["metrics"]["candidate_rmse_instantaneous"][0], .072, places=10)

    def test_partial_reset_and_invalid_sample_cannot_leave_stale_history(self):
        q = InertialCartMotionObserver(num_envs=2)
        sample = trace()
        for i in range(50):
            values = [torch.tensor(sample[key][i]).repeat(2, 1) for key in
                      ("handle_pose_b", "base_velocity_b", "gyro_b", "gravity_b")]
            q.update(*values)
        before = q.estimated_twist_base[1].clone(), q._positions[1].clone()
        q.reset(torch.tensor([0]))
        self.assertFalse(q.ready[0])
        self.assertEqual(q._present[0].sum().item(), 0)
        torch.testing.assert_close(q.estimated_twist_base[1], before[0])
        torch.testing.assert_close(q._positions[1], before[1])
        values[0][0, 0] = float("nan")
        q.update(*values)
        self.assertFalse(q.ready[0])
        self.assertTrue(q.ready[1])
        self.assertTrue(torch.isfinite(q.estimated_twist_base).all())

    def test_duplicate_capture_time_is_ignored_and_not_fit_as_false_new_motion(self):
        q = InertialCartMotionObserver(dtype=torch.float64)
        sample = trace(base_yaw_rate=0.)
        for i in range(21):
            q.update(*(torch.tensor(sample[key][i:i + 1]) for key in
                       ("handle_pose_b", "base_velocity_b", "gyro_b", "gravity_b")))
        before_count = q._present.sum().item()
        for age in range(1, 4):
            i = 20 + age
            q.update(torch.tensor(sample["handle_pose_b"][20:21]),
                     *(torch.tensor(sample[key][i:i + 1]) for key in
                       ("base_velocity_b", "gyro_b", "gravity_b")), pose_delay_steps=age)
            self.assertEqual(q._present.sum().item(), before_count)
            self.assertAlmostEqual(q.sample_age_s.item(), age * .02)
            torch.testing.assert_close(q.estimated_twist_base, torch.tensor([[.2, .04, 0.]], dtype=torch.float64))

    def test_offline_truth_is_evaluation_only_and_replay_is_causal(self):
        sample = trace(steps=80)
        original = compare_motion_estimators(sample)
        sample["truth_cart_twist_base"][:] = 999.
        changed = compare_motion_estimators(sample)
        torch.testing.assert_close(original["candidate_twist_base"], changed["candidate_twist_base"])
        short = {key: value[:40] if isinstance(value, np.ndarray) else value for key, value in sample.items()}
        prefix = compare_motion_estimators(short)
        torch.testing.assert_close(changed["candidate_twist_base"][:40], prefix["candidate_twist_base"])

    def test_unknown_or_unretained_pose_latency_is_rejected(self):
        q = InertialCartMotionObserver()
        args = (torch.tensor([[.4, 0., .2, 0.]]), torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]))
        for delay in (-1, .5, 5, float("nan")):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                q.update(*args, pose_delay_steps=delay)

    def test_reset_trace_boundary_requires_new_complete_window(self):
        sample = trace(steps=100)
        sample["reset"] = np.arange(100) == 50
        out = compare_motion_estimators(sample)
        self.assertTrue(out["candidate_ready"][49].item())
        self.assertFalse(out["candidate_ready"][50:70].any())
        self.assertTrue(out["candidate_ready"][70].item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_candidate_outputs_match(self):
        cpu, gpu = InertialCartMotionObserver(), InertialCartMotionObserver(device="cuda")
        sample = trace(steps=40, delay=2, tilted=True)
        for i in range(40):
            args = [torch.tensor(sample[key][i:i + 1], dtype=torch.float32) for key in
                    ("handle_pose_b", "base_velocity_b", "gyro_b", "gravity_b")]
            cpu.update(*args, pose_delay_steps=2)
            gpu.update(*(x.cuda() for x in args), pose_delay_steps=2)
        torch.testing.assert_close(cpu.estimated_twist_base, gpu.estimated_twist_base.cpu(), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
