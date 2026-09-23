"""V4 command sequences respect curriculum limits and asynchronous resets."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart.command_schedule import V4CommandSchedule


def config(**overrides):
    values = dict(sequence_enabled=True, stop_go_prob=.25, turn_reversal_prob=.25,
                  move_time_s=(2., 4.), stop_hold_time_s=(2., 3.), turn_time_s=(2., 4.),
                  structured_speed_min_frac=.35)
    values.update(overrides)
    return SimpleNamespace(**values)


def level(vx=(-.6, .9), vy=.25, wz=.4):
    return SimpleNamespace(vx_range=vx, vy_max=vy, wz_max=wz)


class V4CommandScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)

    def test_disabled_preserves_legacy_output_and_random_stream(self):
        sampler = V4CommandSchedule(config(sequence_enabled=False), .02, 4)
        ids = torch.arange(4)
        commands, steps = torch.randn(4, 3), torch.full((4,), 250)
        rng = torch.random.get_rng_state().clone()
        sampler.reset(ids, level())
        actual_cmd, actual_steps = sampler.sample(ids, level(), commands, steps)
        self.assertIs(actual_cmd, commands)
        self.assertIs(actual_steps, steps)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_stop_go_has_exact_stop_and_nonzero_restart(self):
        sampler = V4CommandSchedule(config(stop_go_prob=1., turn_reversal_prob=0.), .02, 8)
        ids = torch.arange(8)
        sampler.reset(ids, level())
        dummy, timers = torch.ones(8, 3), torch.ones(8, dtype=torch.long)
        moving, move_steps = sampler.sample(ids, level(), dummy, timers)
        stopped, stop_steps = sampler.sample(ids, level(), dummy, timers)
        restarted, restart_steps = sampler.sample(ids, level(), dummy, timers)
        self.assertTrue(torch.all(moving[:, 0] >= .9 * .35))
        torch.testing.assert_close(moving[:, 1:], torch.zeros(8, 2))
        torch.testing.assert_close(stopped, torch.zeros(8, 3), atol=0, rtol=0)
        torch.testing.assert_close(restarted, moving)
        self.assertTrue(torch.all((move_steps >= 100) & (move_steps <= 200)))
        self.assertTrue(torch.all((stop_steps >= 100) & (stop_steps <= 150)))
        self.assertTrue(torch.all((restart_steps >= 100) & (restart_steps <= 200)))
        torch.testing.assert_close(dummy, torch.ones(8, 3))

    def test_turns_keep_vx_and_reverse_yaw_in_both_directions(self):
        sampler = V4CommandSchedule(config(stop_go_prob=0., turn_reversal_prob=1.), .02, 64)
        ids = torch.arange(64)
        sampler.reset(ids, level())
        dummy, timers = torch.zeros(64, 3), torch.ones(64, dtype=torch.long)
        first, _ = sampler.sample(ids, level(), dummy, timers)
        second, _ = sampler.sample(ids, level(), dummy, timers)
        torch.testing.assert_close(first[:, 0], second[:, 0])
        torch.testing.assert_close(first[:, 2], -second[:, 2])
        torch.testing.assert_close(first[:, 1], torch.zeros(64))
        self.assertTrue(torch.any(first[:, 2] > 0) and torch.any(first[:, 2] < 0))
        self.assertTrue(torch.all(first[:, 2].abs() >= .4 * .35))

    def test_reset_and_resampling_touch_only_requested_environments(self):
        sampler = V4CommandSchedule(config(stop_go_prob=1., turn_reversal_prob=0.), .02, 3)
        ids = torch.arange(3)
        sampler.reset(ids, level())
        sampler.sample(ids, level(), torch.zeros(3, 3), torch.ones(3, dtype=torch.long))
        held_targets = sampler.targets[[0, 2]].clone()
        sampler.reset(torch.tensor([1], dtype=torch.int32), level())
        torch.testing.assert_close(sampler.targets[[0, 2]], held_targets)
        torch.testing.assert_close(sampler.phase, torch.tensor([1, 0, 1]))
        subset = torch.tensor([1, 2])
        commands, _ = sampler.sample(subset, level(), torch.zeros(2, 3), torch.ones(2, dtype=torch.long))
        self.assertGreater(float(commands[0, 0]), 0.)
        torch.testing.assert_close(commands[1], torch.zeros(3))
        torch.testing.assert_close(sampler.phase, torch.tensor([1, 1, 2]))

    def test_early_levels_keep_locked_axes_and_original_random_episodes(self):
        sampler = V4CommandSchedule(config(), .02, 4096)
        ids = torch.arange(4096)
        for lp in (level((0., 0.), 0., 0.), level((-.2, .2), 0., 0.), level((-.4, .4), 0., .1)):
            with self.subTest(vx=lp.vx_range, wz=lp.wz_max):
                sampler.reset(ids, lp)
                dummy = torch.zeros(4096, 3)
                dummy[:, 0] = lp.vx_range[1] * .5
                timers = torch.full((4096,), 250)
                result, steps = sampler.sample(ids, lp, dummy, timers)
                self.assertTrue(torch.all(result[:, 0] >= lp.vx_range[0]))
                self.assertTrue(torch.all(result[:, 0] <= lp.vx_range[1]))
                self.assertTrue(torch.all(result[:, 1].abs() <= lp.vy_max))
                self.assertTrue(torch.all(result[:, 2].abs() <= lp.wz_max))
                random = sampler.mode == sampler.RANDOM
                torch.testing.assert_close(result[random], dummy[random])
                torch.testing.assert_close(steps[random], timers[random])
                if lp.vx_range == (0., 0.):
                    self.assertTrue(random.all())
                elif lp.wz_max == 0:
                    self.assertFalse(torch.any(sampler.mode == sampler.TURN_REVERSAL))
                else:
                    for mode, target_fraction in ((0, .5), (1, .25), (2, .25)):
                        self.assertAlmostEqual(float((sampler.mode == mode).float().mean()), target_fraction, delta=.03)

    def test_demotion_reclips_seeded_targets_to_current_limits(self):
        sampler = V4CommandSchedule(config(stop_go_prob=0., turn_reversal_prob=1.), .02, 32)
        ids = torch.arange(32)
        sampler.reset(ids, level())
        restricted = level((-.1, .1), 0., 0.)
        command, _ = sampler.sample(ids, restricted, torch.zeros(32, 3), torch.ones(32, dtype=torch.long))
        self.assertTrue(torch.all(command[:, 0].abs() <= .1))
        torch.testing.assert_close(command[:, 1:], torch.zeros(32, 2))

    def test_invalid_probabilities_and_durations_rejected(self):
        for invalid in ({"stop_go_prob": .8, "turn_reversal_prob": .3}, {"stop_go_prob": float("nan")},
                        {"turn_reversal_prob": -.1}, {"move_time_s": (0., 1.)},
                        {"stop_hold_time_s": (3., 2.)}, {"turn_time_s": (1., float("inf"))},
                        {"structured_speed_min_frac": 0.}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                V4CommandSchedule(config(**invalid), .02, 1)


if __name__ == "__main__":
    unittest.main()
