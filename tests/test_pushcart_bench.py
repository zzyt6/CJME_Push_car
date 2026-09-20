"""CPU checks for the push-cart benchmark package: paths, follower (on a kinematic unicycle), metrics, policy specs."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "HomieRL"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pushcart_bench.follower import FollowerConfig, PathFollower  # noqa: E402
from pushcart_bench.metrics import Trace, aggregate, path_metrics, step_response_metrics, stopping_distance  # noqa: E402
from pushcart_bench.paths import build_path, right_angle, s_curve, straight, u_turn  # noqa: E402
from pushcart_bench.config_io import load_config  # noqa: E402


class PathTests(unittest.TestCase):
    def test_lengths_and_sampling(self):
        p = straight(6.0)
        self.assertAlmostEqual(p.length, 6.0, places=6)
        self.assertAlmostEqual(p.ds, 0.02, places=6)
        np.testing.assert_allclose(p.yaw, 0.0, atol=1e-9)
        np.testing.assert_allclose(p.curvature, 0.0, atol=1e-9)
        self.assertEqual(p.goal[:2], (6.0, 0.0))

    def test_right_angle_geometry(self):
        p = right_angle(4.0, 4.0, 0.6)
        self.assertAlmostEqual(p.length, 8.0 + math.pi / 2 * 0.6, places=3)
        gx, gy, gyaw = p.goal
        self.assertAlmostEqual(gx, 4.6, places=3)
        self.assertAlmostEqual(gy, 4.6, places=3)
        self.assertAlmostEqual(gyaw, math.pi / 2, places=3)
        arc = (p.s > 4.1) & (p.s < 4.0 + math.pi / 2 * 0.6 - 0.1)
        np.testing.assert_allclose(p.curvature[arc], 1 / 0.6, rtol=0.05)      # left turn -> positive curvature
        self.assertAlmostEqual(p.min_radius, 0.6, delta=0.05)

    def test_u_turn_geometry(self):
        p = u_turn(3.0, 1.0, 3.0)
        self.assertAlmostEqual(p.length, 6.0 + math.pi, places=3)
        gx, gy, gyaw = p.goal
        self.assertAlmostEqual(gx, 0.0, places=3)
        self.assertAlmostEqual(gy, 2.0, places=3)
        self.assertAlmostEqual(abs(gyaw), math.pi, places=3)
        # the return leg passes 2 m from the outbound leg: the forward search window must not jump onto it
        i = p.nearest_index((1.5, 0.0), start=0, window_m=3.0)
        self.assertLess(p.s[i], 3.0)

    def test_s_curve_ends_aligned_and_radius(self):
        p = s_curve(12.0, 1.0, 2)
        self.assertAlmostEqual(p.yaw[0], 0.0, places=2)
        self.assertAlmostEqual(p.yaw[-1], 0.0, places=2)
        self.assertAlmostEqual(p.xy[-1, 1], 0.0, places=3)
        self.assertGreater(np.max(p.xy[:, 1]), 0.6)
        self.assertLess(np.min(p.xy[:, 1]), -0.6)
        self.assertGreater(p.min_radius, 0.8)
        self.assertLess(p.min_radius, 1.2)
        tight = s_curve(10.0, 1.2, 2)                                        # the benchmark's default S
        self.assertLess(tight.min_radius, 0.6)
        self.assertGreater(tight.min_radius, 0.4)

    def test_s_bend_geometry(self):
        from pushcart_bench.paths import s_bend
        p = s_bend(0.6, 1.0)
        self.assertAlmostEqual(p.length, 2.0 + 2 * math.pi * 0.6, places=2)
        gx, gy, gyaw = p.goal
        self.assertAlmostEqual(gx, 2.0 + 4 * 0.6, places=2)
        self.assertAlmostEqual(gy, 0.0, places=2)
        self.assertAlmostEqual(gyaw, 0.0, places=2)
        self.assertAlmostEqual(np.max(p.xy[:, 1]), 1.2, places=2)             # bulge of 2 radius
        self.assertAlmostEqual(p.min_radius, 0.6, delta=0.03)
        bends = np.abs(p.curvature) > 1.0
        self.assertGreater(np.mean(bends), 0.5)                              # most of the path is in the arcs
        self.assertAlmostEqual(np.max(np.abs(np.diff(p.yaw))), 0.02 / 0.6, delta=0.01)   # heading continuous

    def test_s_large_geometry(self):
        from pushcart_bench.paths import s_bend
        p = s_bend(2.5, 2.0)
        self.assertAlmostEqual(p.length, 4.0 + 2 * math.pi * 2.5, places=2)
        self.assertAlmostEqual(p.goal[0], 4.0 + 10.0, places=2)
        self.assertAlmostEqual(p.goal[1], 0.0, places=2)
        self.assertAlmostEqual(np.max(p.xy[:, 1]), 5.0, places=2)
        self.assertAlmostEqual(p.min_radius, 2.5, delta=0.08)
        cfg = load_config()
        built = build_path("s_large", cfg["paths"]["s_large"])
        self.assertAlmostEqual(built.min_radius, 2.5, delta=0.08)
        self.assertGreater(built.length, 18.0)

    def test_transform_and_mirror(self):
        p = right_angle(2.0, 2.0, 0.5)
        m = p.mirrored()
        np.testing.assert_allclose(m.xy[:, 1], -p.xy[:, 1])
        np.testing.assert_allclose(m.curvature, -p.curvature)
        self.assertAlmostEqual(m.goal[2], -math.pi / 2, places=3)
        t = p.transformed((1.0, -2.0), math.pi / 2)
        np.testing.assert_allclose(t.xy[0], [1.0, -2.0], atol=1e-9)
        self.assertAlmostEqual(t.yaw[0], math.pi / 2, places=6)
        self.assertAlmostEqual(t.length, p.length, places=9)
        xy, yaw, curv = t.point_at(1.0)
        np.testing.assert_allclose(xy, [1.0, -1.0], atol=1e-6)         # 1 m along the rotated first leg (+y)

    def test_build_from_config(self):
        cfg = load_config()
        for name, spec in cfg["paths"].items():
            p = build_path(name, spec)
            self.assertEqual(p.name, name)
            self.assertGreater(p.length, 1.0)
            self.assertTrue(np.isfinite(p.curvature).all())
        with self.assertRaises(KeyError):
            build_path("bogus", {"type": "spiral"})


def simulate_unicycle(path, cfg: FollowerConfig, dt=0.02, tau=0.3, mode=None):
    """Kinematic unicycle with first-order velocity lag driven by the follower; returns (trace, follower)."""
    if mode is not None:
        cfg = FollowerConfig(**{**cfg.__dict__, "mode": mode})
    fol = PathFollower(path, cfg, dt)
    x, y, yaw = path.xy[0, 0], path.xy[0, 1], path.yaw[0]
    v = np.zeros(3)
    trace = Trace()
    while not fol.state.done:
        cmd = fol.update((x, y, yaw), v)
        v = v + (cmd - v) * (dt / tau)
        x += (v[0] * math.cos(yaw) - v[1] * math.sin(yaw)) * dt
        y += (v[0] * math.sin(yaw) + v[1] * math.cos(yaw)) * dt
        yaw = math.atan2(math.sin(yaw + v[2] * dt), math.cos(yaw + v[2] * dt))
        trace.append(fol.state.t, (x, y, yaw), v, cmd, fol.info())
    return trace, fol


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = FollowerConfig.from_dict(load_config()["follower"])

    def test_reaches_every_reference_path(self):
        from pushcart_bench.paths import s_bend
        for build in (straight, s_curve, lambda: s_curve(10.0, 1.2, 2), s_bend, right_angle, u_turn):
            path = build()
            trace, fol = simulate_unicycle(path, self.cfg)
            m = path_metrics(trace, path, fol.info())
            self.assertTrue(m["reached"], f"{path.name}: not reached ({fol.info()})")
            self.assertLess(m["final_pos_err"], self.cfg.goal_radius, path.name)
            self.assertLess(m["cross_track_rms"], 0.15, f"{path.name}: cross-track rms {m['cross_track_rms']:.3f}")
            self.assertLess(m["time_to_goal"], path.length / self.cfg.v_nom * 3.0, path.name)   # tight arcs run at ~1/3 v_nom

    def test_holonomic_mode_reaches(self):
        path = s_curve()
        trace, fol = simulate_unicycle(path, self.cfg, mode="holonomic")
        m = path_metrics(trace, path, fol.info())
        self.assertTrue(m["reached"])
        self.assertLess(m["cross_track_rms"], 0.15)
        self.assertGreater(np.max(np.abs(np.asarray(trace.cmd)[:, 1])), 0.01)     # lateral channel used

    def test_limits_and_rate(self):
        path = right_angle()
        trace, _ = simulate_unicycle(path, self.cfg)
        cmd = np.asarray(trace.cmd)
        lo = np.array([p[0] for p in self.cfg.cmd_limits])
        hi = np.array([p[1] for p in self.cfg.cmd_limits])
        self.assertTrue(np.all(cmd >= lo - 1e-9) and np.all(cmd <= hi + 1e-9))
        rate = np.abs(np.diff(cmd, axis=0)) / 0.02
        self.assertTrue(np.all(rate <= np.asarray(self.cfg.rate_limits) + 1e-6))
        self.assertTrue(np.all(cmd[:, 1] == 0.0))                                 # car mode: no lateral command

    def test_cruise_holds_commanded_speed_on_large_s(self):
        from pushcart_bench.paths import s_bend
        path = s_bend(2.5, 2.0)
        cfg = FollowerConfig.from_dict({**self.cfg.__dict__, "cruise": True, "v_nom": 0.6, "timeout_s": 120.0})
        trace, fol = simulate_unicycle(path, cfg)
        m = path_metrics(trace, path, fol.info())
        self.assertTrue(m["reached"], fol.info())
        cmd = np.asarray(trace.cmd)
        n = len(cmd)
        mid = cmd[int(0.15 * n):int(0.75 * n), 0]
        self.assertGreater(float(np.mean(mid)), 0.50)                             # not the old k_curv ~0.35 v_nom
        slow = FollowerConfig.from_dict({**self.cfg.__dict__, "cruise": False, "v_nom": 0.6})
        trace_slow, _ = simulate_unicycle(path, slow)
        mid_slow = np.asarray(trace_slow.cmd)[int(0.15 * len(trace_slow.cmd)):int(0.75 * len(trace_slow.cmd)), 0]
        self.assertGreater(float(np.mean(mid)), float(np.mean(mid_slow)) + 0.08)

    def test_timeout(self):
        cfg = FollowerConfig(**{**self.cfg.__dict__, "timeout_s": 1.0})
        path = straight(20.0)
        trace, fol = simulate_unicycle(path, cfg)
        self.assertTrue(fol.state.timed_out and not fol.state.reached)
        m = path_metrics(trace, path, fol.info())
        self.assertFalse(m["success"])
        self.assertIsNone(m["time_to_goal"])


class MetricTests(unittest.TestCase):
    def test_first_order_step(self):
        tau, t_step = 0.5, 1.0
        t = np.arange(0.0, 8.0, 0.02)
        y = np.where(t < t_step, 0.0, 0.6 * (1 - np.exp(-(t - t_step) / tau)))
        m = step_response_metrics(t, y, t_step, 0.0, 0.6)
        self.assertAlmostEqual(m["rise_time"], tau * math.log(9), delta=0.05)     # 10-90 % of a first-order lag = 2.2 tau
        self.assertAlmostEqual(m["overshoot"], 0.0, places=6)
        self.assertLess(abs(m["ss_error"]), 0.01)
        self.assertAlmostEqual(m["settling_time"], tau * math.log(10), delta=0.05)  # 10 % band

    def test_step_metrics_with_gait_oscillation(self):
        from pushcart_bench.metrics import moving_average
        t = np.arange(0.0, 8.0, 0.02)
        base = np.where(t < 1.0, 0.0, 0.3 * (1 - np.exp(-(t - 1.0) / 0.4)))
        y = base + 0.15 * np.sin(2 * math.pi * 2.0 * t)                       # 2 Hz gait ripple of +-0.15
        raw = step_response_metrics(t, y, 1.0, 0.0, 0.3)
        smooth = step_response_metrics(t, y, 1.0, 0.0, 0.3, smooth_s=0.5)  # one ripple period
        self.assertGreater(raw["overshoot"], 0.3)                          # ripple read as overshoot
        self.assertLess(smooth["overshoot"], 0.08)
        self.assertIsNotNone(smooth["settling_time"])
        np.testing.assert_allclose(moving_average(np.ones(50), 7), 1.0)

    def test_stopping_distance(self):
        t = np.arange(0.0, 4.0, 0.02)
        speed = np.where(t < 1.0, 0.5, np.maximum(0.0, 0.5 - 0.5 * (t - 1.0)))   # linear stop over 1 s
        x = np.cumsum(speed) * 0.02
        d = stopping_distance(t, x, np.zeros_like(x), speed, 1.0, v_stop=0.05)
        self.assertAlmostEqual(d, 0.25, delta=0.03)

    def test_aggregate(self):
        a = aggregate([1.0, None, 3.0, float("nan")])
        self.assertEqual(a["n"], 2)
        self.assertAlmostEqual(a["mean"], 2.0)
        self.assertEqual(aggregate([None])["n"], 0)


class PolicySpecTests(unittest.TestCase):
    def test_parse_specs(self):
        from pushcart_bench.policies import parse_policy_spec
        self.assertEqual(parse_policy_spec("passthrough")["kind"], "passthrough")
        self.assertEqual(parse_policy_spec("robot_only")["kind"], "robot_only")
        spec = parse_policy_spec("two_stage:latest")
        self.assertEqual((spec["kind"], spec["checkpoint"]), ("two_stage", "latest"))
        spec = parse_policy_spec(r"two_stage:E:\runs\model_1200.pt")
        self.assertEqual(spec["checkpoint"], r"E:\runs\model_1200.pt")
        spec = parse_policy_spec("plugin:my_pkg.my_mod:MyPolicy")
        self.assertEqual((spec["kind"], spec["module"], spec["attr"]), ("plugin", "my_pkg.my_mod", "MyPolicy"))
        with self.assertRaises(ValueError):
            parse_policy_spec("unknown_thing")


if __name__ == "__main__":
    unittest.main()
