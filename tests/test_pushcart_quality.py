"""Physical sign/frame, reset, frequency and backward-compatibility checks for V3.1."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from homie_pushcart.cart_quality import CartQualityTracker, apply_quality_profile, resultant_wrench, rotate_planar
from homie_pushcart.task_config import PushCartTaskCfg, task_cfg_from_run, num_actor_obs, num_critic_obs
from pushcart_bench.metrics import Trace, cart_quality_metrics, disturbance_recovery_metrics


class WrenchGeometryTests(unittest.TestCase):
    def test_opposite_pushes_generate_yaw_not_waste(self):
        forces = torch.tensor([[[10., 0., 0.], [-10., 0., 0.]]])
        points = torch.tensor([[[0., .3, 1.], [0., -.3, 1.]]])
        force, moment = resultant_wrench(forces, torch.zeros_like(forces), points, torch.zeros(1, 3))
        torch.testing.assert_close(force, torch.zeros(1, 3))
        torch.testing.assert_close(moment, torch.tensor([[0., 0., -6.]]))

    def test_moment_origin_translation(self):
        f = torch.tensor([[[2., 3., 0.]]])
        p = torch.tensor([[[1., 0., 0.]]])
        tau = torch.tensor([[[0., 0., 4.]]])
        _, m0 = resultant_wrench(f, tau, p, torch.zeros(1, 3))
        _, m1 = resultant_wrench(f, tau, p, torch.tensor([[1., 0., 0.]]))
        self.assertAlmostEqual(float(m0[0, 2]), 7.)
        self.assertAlmostEqual(float(m1[0, 2]), 4.)

    def test_mujoco_weld_recovers_known_external_load(self):
        import mujoco
        from homie_pushcart.mujoco_env import MujocoHomieEnv
        # Cart welded to a fixed fixture. Steady weld reaction must cancel the known external
        # force AND torque, including nonzero inertial offset, rotated free joint and torquescale.
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <option timestep="0.002" gravity="0 0 0"/>
          <worldbody><body name="fixture"/><body name="cart" pos="1 .4 .2" quat=".92387953 0 0 .38268343">
            <freejoint name="cart_free"/><inertial pos=".1 -.05 .03" mass="2" diaginertia=".2 .3 .4"/>
            <geom size=".1" contype="0" conaffinity="0"/>
          </body></worldbody>
          <equality><weld name="grasp_left" body1="fixture" body2="cart" torquescale="2" solref=".004 1"/></equality>
          </mujoco>''')
        data = mujoco.MjData(model)
        cart = model.body("cart").id
        load = np.array([12., -7., 3., 1., 2., -4.])
        data.xfrc_applied[cart] = load
        for _ in range(400):
            mujoco.mj_step(model, data)
        before = (data.qpos.copy(), data.qvel.copy())
        stub = SimpleNamespace(model=model, data=data, cart_id=cart)
        result = MujocoHomieEnv.grasp_cart_wrenches(stub)["left"]
        np.testing.assert_allclose(result, -load, atol=0.02, rtol=0.005)
        np.testing.assert_array_equal(data.qpos, before[0])
        np.testing.assert_array_equal(data.qvel, before[1])


class QualityFilterTests(unittest.TestCase):
    def tracker(self, n=1, dt=.02):
        cfg = PushCartTaskCfg().quality
        cfg.warmup_s = 0.
        return CartQualityTracker(cfg, dt, n, dtype=torch.float64)

    def test_constant_useful_force_and_tracking_are_free(self):
        q = self.tracker()
        v = torch.tensor([[.3, 0., .2]], dtype=torch.float64)
        f = torch.tensor([[40., -10., 0.]], dtype=torch.float64)
        m = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
        for _ in range(100):
            q.update(v, v, torch.zeros(1, dtype=torch.float64), f, m)
        self.assertEqual(float(q.ripple_cost[0]), 0.)
        self.assertEqual(float(q.wrench_cost[0]), 0.)

    def test_partial_reset_seeds_only_reset_environment(self):
        q = self.tracker(2)
        zeros = torch.zeros(2, 3, dtype=torch.float64)
        yaw = torch.zeros(2, dtype=torch.float64)
        q.update(zeros, zeros, yaw, zeros, zeros)
        q.update(zeros + 1, zeros, yaw, zeros + 10, zeros)
        q.reset(torch.tensor([0]))
        q.update(zeros + 2, zeros, yaw, zeros + 30, zeros)
        torch.testing.assert_close(q.velocity_band[0], zeros[0])
        self.assertGreater(float(q.velocity_band[1].norm()), .01)
        self.assertEqual(q.steps.tolist(), [1, 3])

    def test_world_filter_is_invariant_to_heading_relabel(self):
        q = self.tracker()
        force = torch.tensor([[40., 5., 0.]], dtype=torch.float64)
        error_world = torch.tensor([[.1, -.2, 0.]], dtype=torch.float64)
        zero = torch.zeros(1, 3, dtype=torch.float64)
        for angle in (0., .8, -1.2, 2.3):
            yaw = torch.tensor([angle], dtype=torch.float64)
            error_cart = rotate_planar(error_world, yaw, inverse=True)
            q.update(error_cart, zero, yaw, force, zero)
        torch.testing.assert_close(q.velocity_band, zero, atol=1e-12, rtol=0)
        torch.testing.assert_close(q.wrench_band, zero, atol=1e-12, rtol=0)

    def test_frequency_response_rejects_dc_and_attenuates_fast_noise(self):
        def gain(freq, dt=.02):
            q = self.tracker(dt=dt)
            zero = torch.zeros(1, 3, dtype=torch.float64)
            samples = []
            for t in np.arange(0, 12., dt):
                signal = zero.clone()
                signal[0, 0] = math.sin(2 * math.pi * freq * t)
                q.update(signal, zero, torch.zeros(1, dtype=torch.float64), zero, zero)
                if t > 6:
                    samples.append(float(q.velocity_band[0, 0]))
            return np.std(samples) * math.sqrt(2)
        mid = gain(2.)
        self.assertGreater(mid, .6)
        self.assertLess(gain(.03), .1)
        self.assertLess(gain(20.), mid * .6)
        self.assertAlmostEqual(gain(2., .01), mid, delta=.06)

    def test_nonfinite_sample_does_not_poison_history_and_cost_is_capped(self):
        q = self.tracker()
        zero = torch.zeros(1, 3, dtype=torch.float64)
        yaw = torch.zeros(1, dtype=torch.float64)
        q.update(zero, zero, yaw, zero, zero)
        q.update(zero + float('nan'), zero, yaw, zero, zero)
        self.assertFalse(bool(q.ready[0]))
        q.update(zero, zero, yaw + float('nan'), zero, zero)
        self.assertTrue(torch.isfinite(q.velocity_band).all())
        q.update(zero + 1e5, zero, yaw, zero + 1e5, zero + 1e5)
        self.assertTrue(torch.isfinite(q.velocity_band).all())
        self.assertLessEqual(float(q.ripple_cost[0]), q.cfg.cost_cap)
        self.assertLessEqual(float(q.wrench_cost[0]), q.cfg.cost_cap)

    def test_invalid_filter_fails_early(self):
        cfg = PushCartTaskCfg().quality
        cfg.high_hz = 25.
        with self.assertRaises(ValueError):
            CartQualityTracker(cfg, .02, 1)


class QualityIntegrationTests(unittest.TestCase):
    def test_profiles_are_opt_in_and_old_layout_is_unchanged(self):
        old = PushCartTaskCfg()
        new = apply_quality_profile(PushCartTaskCfg(), "combined")
        self.assertFalse(old.quality.enabled)
        self.assertEqual(old.rewards.scales.cart_ripple, 0.)
        self.assertLess(new.rewards.scales.cart_ripple, 0.)
        self.assertEqual((num_actor_obs(new), num_critic_obs(new)), (num_actor_obs(old), num_critic_obs(old)))
        self.assertEqual(new.env.num_hl_actions, 18)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'params'
            p.mkdir()
            (p / 'env.yaml').write_text('task:\n  actions:\n    clip: 1.0\n  rewards:\n    scales:\n      force_rate: -0.2\n', encoding='utf-8')
            restored = task_cfg_from_run(td, verbose=False)
            self.assertFalse(restored.quality.enabled)
            self.assertEqual(restored.rewards.scales.cart_wrench_osc, 0.)

    def test_training_profile_is_forwarded_without_changing_task_rewards(self):
        from homie_pushcart import training
        with patch.object(training, '_lab_scripts', return_value=Path('official')), patch.object(training, '_run', return_value=0) as run:
            training.train(['--quality_profile', 'ripple', '--num_envs', '16', '--headless'])
        args = run.call_args.args[1]
        self.assertIn('env.task.quality.enabled=true', args)
        self.assertIn('env.task.rewards.scales.cart_ripple=-0.05', args)
        self.assertIn('env.task.rewards.scales.cart_wrench_osc=0.0', args)
        self.assertFalse(any('track_cart' in a for a in args))

    def test_metrics_exclude_transient_but_keep_failure_peak(self):
        trace = Trace()
        for i in range(10):
            trace.append(i * .02, [0, 0, 0], [.3, 0, .2], [.3, 0, .2], telemetry={
                'quality_ready': i >= 2, 'cart_vx_band': .02 if i >= 5 else 10.,
                'cart_fx': -4., 'cart_mz': 3., 'cart_tilt_rad': .5 if i == 0 else .01,
                'cart_ref_vx': .3, 'cart_ref_vy': 0., 'cart_ref_wz': .2})
        trace.failure = 'cart_tilt'
        m = cart_quality_metrics(trace, start_s=.10)
        self.assertEqual(m['quality_samples'], 5)
        self.assertAlmostEqual(m['cart_vx_band_rms'], .02)
        self.assertEqual(m['cart_tilt_rad_max'], .5)
        self.assertEqual(m['cart_fx_mean'], -4.)  # signed braking remains visible
        self.assertEqual(m['cart_vx_raw_ref_rmse'], 0.)
        self.assertEqual(len(trace.decimated(1)['cart_mz']), 10)
        self.assertEqual(cart_quality_metrics(Trace()), {})

    def test_recovery_dwell_and_unrecovered_events(self):
        trace = Trace()
        for i in range(40):
            t = i * .1
            vx = .5 if 1. <= t < 2. or t >= 3. else .3
            trace.append(t, [0, 0, 0], [vx, 0, 0], [.3, 0, 0])
        result = disturbance_recovery_metrics(trace, [{"kind": "kick", "t": 1., "end_t": 1.},
                                                     {"kind": "kick", "t": 3., "end_t": 3.}],
                                              {"filter_tau_s": 0., "hold_s": .3})
        self.assertAlmostEqual(result['events'][0]['recovery_s'], 1.)
        self.assertAlmostEqual(result['events'][0]['confirmed_at_s'], 2.3)
        self.assertEqual(result['events'][0]['status'], 'recovered')
        self.assertIsNone(result['events'][1]['recovery_s'])
        self.assertEqual(result['metrics']['disturbance_recovered_frac'], .5)
        self.assertAlmostEqual(result['metrics']['disturbance_recovery_mean_s'], 1.)
        stable = disturbance_recovery_metrics(trace, [{"kind": "kick", "t": .1, "end_t": .1}],
                                              {"filter_tau_s": 0., "hold_s": .3})
        self.assertEqual(stable['events'][0]['status'], 'within_tolerance')


if __name__ == '__main__':
    unittest.main()
