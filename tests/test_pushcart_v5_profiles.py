"""Profiles must isolate the claimed V5 ablation regardless of previous profile."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v5.cart_quality import QUALITY_PROFILES, apply_quality_profile, quality_profile_overrides
from homie_pushcart_v5.task_config import PushCartTaskCfg, num_actor_obs


def at(cfg, path):
    for name in path.split("."):
        cfg = getattr(cfg, name)
    return cfg


class V5QualityProfileTests(unittest.TestCase):
    def test_non_v5_profiles_remove_new_reaction_channels_and_rewards(self):
        for profile in QUALITY_PROFILES:
            if profile == "v5":
                continue
            with self.subTest(profile=profile):
                cfg = apply_quality_profile(PushCartTaskCfg(), profile)
                self.assertFalse(cfg.reaction.enabled)
                self.assertFalse(cfg.reaction.compensation_enabled)
                for name in ("task_wrench", "reaction_motion", "centroidal_recovery", "support_feasibility"):
                    self.assertEqual(getattr(cfg.rewards.scales, name), 0.)
                self.assertEqual(cfg.rewards.scales.force_rate, -.2)
                self.assertTrue(cfg.randomization.cart_force_pulse)
                self.assertEqual(num_actor_obs(cfg), 705)

    def test_profile_order_does_not_leak_reward_or_controller_switches(self):
        paths = set().union(*(quality_profile_overrides(name) for name in QUALITY_PROFILES))
        for previous in QUALITY_PROFILES:
            for current in QUALITY_PROFILES:
                with self.subTest(previous=previous, current=current):
                    direct = apply_quality_profile(PushCartTaskCfg(), current)
                    switched = apply_quality_profile(apply_quality_profile(PushCartTaskCfg(), previous), current)
                    for path in paths:
                        self.assertEqual(at(switched, path), at(direct, path), path)

    def test_diagnostic_has_no_quality_or_reaction_costs(self):
        cfg = apply_quality_profile(PushCartTaskCfg(), "diagnostic")
        self.assertTrue(cfg.quality.enabled)
        for name in ("cart_ripple", "cart_lin_ripple", "cart_yaw_ripple", "cart_response_error", "cart_wrench_osc",
                     "task_wrench", "reaction_motion", "centroidal_recovery", "support_feasibility"):
            self.assertEqual(getattr(cfg.rewards.scales, name), 0.)


if __name__ == "__main__":
    unittest.main()
