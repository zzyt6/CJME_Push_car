"""The retained 200 Hz bench recorder cannot alter legacy simulation dynamics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "scripts")]

import numpy as np
import torch

from homie_pushcart.mujoco_env import MujocoHomieEnv
from pushcart_bench.instrumentation import InstrumentedEnv
from pushcart_bench.policies import PassthroughPolicy, make_policy, parse_policy_spec


class InstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_state_equal(self, original, recorded):
        for name in ("qpos", "qvel", "qacc", "ctrl", "qfrc_constraint", "efc_force", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(original.data, name), getattr(recorded.data, name), err_msg=name)
        self.assertEqual(original.data.time, recorded.data.time)

    def test_zero_and_changing_actions_leave_every_step_bit_exact(self):
        for with_cart in (False, True):
            with self.subTest(with_cart=with_cart):
                original, recorded = MujocoHomieEnv(with_cart), InstrumentedEnv(with_cart)
                np.testing.assert_array_equal(original.reset(), recorded.reset())
                for tick in range(48):
                    action = np.zeros(12) if tick < 16 else .04 * np.sin(np.arange(12) + tick * .1)
                    for env in (original, recorded):
                        env.set_command(.15, .02, -.03)
                        env.upper_delta[:] = 0. if tick < 16 else .002
                    original_obs, original_fall = original.step(action)
                    recorded_obs, recorded_fall = recorded.step(action)
                    self.assert_state_equal(original, recorded)
                    np.testing.assert_array_equal(original_obs, recorded_obs)
                    self.assertEqual(original_fall, recorded_fall)
                    self.assertEqual(len(recorded.sub_power), 4)
                    if with_cart:
                        self.assertEqual(np.asarray(recorded.sub_hands).shape, (4, 2, 6))
                        self.assertTrue(np.isfinite(recorded.sub_hands).all())

    def test_reading_does_not_clear_existing_external_forces(self):
        original, recorded = MujocoHomieEnv(True), InstrumentedEnv(True)
        for env in (original, recorded):
            env.reset()
            env.data.xfrc_applied[env.cart_id] = [4., 2., 0., 0., 0., .1]
        for _ in range(8):
            original.step(np.zeros(12))
            recorded.step(np.zeros(12))
            self.assert_state_equal(original, recorded)

    def test_requested_pulse_matches_the_same_direct_physics_load(self):
        original, recorded = MujocoHomieEnv(True, decimation=1), InstrumentedEnv(True, decimation=1)
        original.reset()
        recorded.reset()
        load = np.array([8., -2., 0., 0., 0., .4])
        recorded.inject_cart_wrench(load[:3], load[3:], .035)
        for tick in range(17):
            original.data.xfrc_applied[original.cart_id] = load if tick < 7 else 0.
            original.step(np.zeros(12))
            recorded.step(np.zeros(12))
            self.assert_state_equal(original, recorded)
        self.assertEqual(recorded._pulse_steps, 0)
        self.assertFalse(recorded._pulse_active)

    def test_retained_recorder_and_quality_metrics_work_without_retired_packages(self):
        from pushcart_bench.extra_metrics import Recorder, window_metrics

        with patch("homie_pushcart.mujoco_hl.MujocoHomieEnv", InstrumentedEnv):
            policy = PassthroughPolicy(17)
        rec = Recorder(policy)
        policy.reset(payload_kg=12.5, seed=0)
        for _ in range(100):
            policy.step(np.array([.1, 0., 0.]))
        values = rec.arrays()
        self.assertEqual(values["hands"].shape, (100, 4, 2, 6))
        self.assertEqual(values["F_sub"].shape, (100, 4, 3))
        self.assertTrue(np.isfinite(values["F_sub"]).all())
        metrics = window_metrics(values, 0, 100, policy.dt, policy.cart_mass_total(), policy.env.robot_mass)
        for name in ("handle_Fx_band_rms", "mech_power_mean_W", "hand_force_p95"):
            self.assertTrue(np.isfinite(metrics[name]), name)
        policy.reset()
        self.assertEqual(len(rec.rows["hands"]), 0)

    def test_registry_and_historical_runner_only_offer_retained_versions(self):
        import bench_versions

        self.assertEqual(bench_versions.ORDER, ["passthrough", "V3", "V3.1", "V3.2"])
        self.assertTrue(all(path.is_file() for path in bench_versions.VERSIONS.values()))
        for name in ("v53", "v53_original", "v53_new"):
            with self.assertRaises(ValueError):
                parse_policy_spec(name)
        for name in ("passthrough", "robot_only", "two_stage", "v6"):
            with self.assertRaisesRegex(ValueError, "fixed legacy"):
                make_policy(name, scene_version="four_spheres")
        with self.assertRaisesRegex(ValueError, "fixed legacy"):
            InstrumentedEnv(True, xml_path=ROOT / "another_scene.xml")


if __name__ == "__main__":
    unittest.main()
