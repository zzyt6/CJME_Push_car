"""V5.2 upper-body torque-sensor model: HOMIE-ratio bias/noise/delay, observation-only, curriculum-scaled."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v52.task_config import PushCartTaskCfg


class TorqueSensorConfigTests(unittest.TestCase):
    def test_homie_ratios(self):
        from homie_isaaclab.config import G1RoughCfg
        homie = G1RoughCfg().domain_rand
        so = PushCartTaskCfg().observations
        self.assertEqual(so.upper_torque_bias_frac_of_limit, max(abs(v) for v in homie.actuation_offset_range))
        self.assertEqual(so.upper_torque_step_noise_frac_of_limit, max(abs(v) for v in homie.joint_injection_range))
        self.assertGreaterEqual(so.upper_torque_delay_steps_max, 1)


class TorqueSensorModelTests(unittest.TestCase):
    """Replays the exact delay/bias/noise arithmetic of PushCartEnv._update_hl_observations."""

    def reading(self, hist, delay, episode_length, tau, bias, noise, limits, so, seed=0):
        torch.manual_seed(seed)
        n = tau.shape[0]
        hist[:, 1:] = hist[:, :-1].clone()
        hist[:, 0] = tau
        d = torch.minimum(delay, episode_length)
        out = hist.gather(1, d.view(n, 1, 1).expand(-1, 1, 15))[:, 0]
        if noise > 0:
            out = out * (1. + so.upper_torque_noise_frac * noise * (2 * torch.rand_like(out) - 1))
            out = out + bias + (2 * torch.rand_like(out) - 1) * so.upper_torque_step_noise_frac_of_limit * noise * limits
        return out

    def test_delay_returns_older_sample_and_is_clamped_after_reset(self):
        so = PushCartTaskCfg().observations
        hist = torch.zeros(1, so.upper_torque_delay_steps_max + 1, 15)
        delay = torch.tensor([2])
        limits = torch.full((15,), 25.)
        outputs = []
        for t in range(4):
            outputs.append(self.reading(hist, delay, torch.tensor([t]), torch.full((1, 15), float(t + 1)),
                                        torch.zeros(1, 15), 0., limits, so))
        self.assertEqual([o[0, 0].item() for o in outputs], [1., 1., 1., 2.])

    def test_bias_and_noise_bounds_scale_with_curriculum(self):
        so = PushCartTaskCfg().observations
        limits = torch.tensor([88.] + [25.] * 5 + [5.] * 2 + [25.] * 5 + [5.] * 2)
        tau = torch.zeros(4096, 15)
        for noise in (0., 0.5, 1.):
            bias = (2 * torch.rand(4096, 15) - 1) * so.upper_torque_bias_frac_of_limit * noise * limits
            hist = torch.zeros(4096, 3, 15)
            out = self.reading(hist, torch.zeros(4096, dtype=torch.long), torch.full((4096,), 5), tau, bias, noise, limits, so)
            bound = (so.upper_torque_bias_frac_of_limit + so.upper_torque_step_noise_frac_of_limit) * noise * limits
            self.assertTrue((out.abs() <= bound + 1e-6).all())
            if noise == 1.:
                # At full level the error reaches most of the +/-10 % of limit range.
                self.assertGreater((out.abs() / limits).max().item(), 0.08)
            if noise == 0.:
                self.assertEqual(out.abs().max().item(), 0.)


if __name__ == "__main__":
    unittest.main()
