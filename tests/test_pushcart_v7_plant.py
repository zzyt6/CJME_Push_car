"""V7 plant regressions found 2026-10-08: caster swivel scrub, bench plant selection, revision-8 migration."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "scripts")]


class SwivelScrubLaw(unittest.TestCase):
    def test_standstill_scrub_and_rolling_bearing(self):
        import torch
        from homie_pushcart_v7.physics import swivel_torque
        self.assertAlmostEqual(float(swivel_torque(0.225, 0.01, 0.0, 0.05)), 0.225)
        self.assertAlmostEqual(float(swivel_torque(0.225, 0.01, 0.4, 0.05)), 0.01, places=3)
        t = swivel_torque(torch.tensor([0.4]), torch.tensor([0.02]), torch.tensor([-0.05]), 0.05)
        self.assertAlmostEqual(float(t), 0.02 + 0.38 / np.e, places=5)

    def test_bare_cart_turns_without_dead_band_moment(self):
        """The constant 0.225 N.m Coulomb scrub needed ~13.6 N.m to hold a 0.4 rad/s turn in MuJoCo (dead band side
        force tau / trail on every caster); with the rolling law it must stay near the V3.2 casters (~1.2 N.m)."""
        import caster_steer_test as cst
        cst.VARIANTS = {"rolling_0.225": (0.225, 0.025, True), "coulomb_0.225": (0.225, 0.025, False)}
        res = cst.run_mujoco(0.0)
        self.assertLess(abs(res["rolling_0.225"]["steady_turn_Mz"]), 3.0)
        self.assertGreater(res["coulomb_0.225"]["steady_turn_Mz"], 8.0)          # the bug the test guards against


class BenchPlantSelection(unittest.TestCase):
    def test_spec(self):
        from pushcart_bench.policies import parse_policy_spec
        self.assertEqual(parse_policy_spec("v7@legacy:a.pt"),
                         {"kind": "cross", "version": 7, "plant": "legacy", "checkpoint": "a.pt"})
        self.assertEqual(parse_policy_spec("passthrough@v7")["checkpoint"], None)
        self.assertEqual(parse_policy_spec("v6@v7")["checkpoint"], "latest")
        self.assertEqual(parse_policy_spec("v7:latest"), {"kind": "v7", "checkpoint": "latest"})
        with self.assertRaises(ValueError):
            parse_policy_spec("two_stage@v7:x.pt")


class Revision8Migration(unittest.TestCase):
    def test_constant_scrub_kept_for_old_runs(self):
        from homie_pushcart_v7.task_config import migrate_saved_task
        old = {"implementation_revision": 8, "cart_physics": {"swivel_friction_range": [0.05, 0.4]}}
        new = migrate_saved_task(old)
        self.assertEqual(new["cart_physics"]["swivel_bearing_range"], [0.0, 0.0])
        self.assertGreater(new["cart_physics"]["swivel_scrub_speed_m_s"], 1e6)
        self.assertNotIn("swivel_bearing_range", old["cart_physics"])        # input untouched
        self.assertIs(migrate_saved_task({"implementation_revision": 10}).get("cart_physics"), None)
        rev9 = migrate_saved_task({"implementation_revision": 9, "cart_physics": {"swivel_bearing_range": [0.002, 0.02]}})
        self.assertEqual(rev9["cart_physics"]["swivel_bearing_range"], [0.002, 0.02])     # rev 9 physics kept
        self.assertEqual(rev9["cart_physics"]["yaw_damping_range"], [0.0, 0.0])
        self.assertEqual(rev9["release"], "7.0")

    def test_current_tracking_matches_v6(self):
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        cfg = PushCartTaskCfg()
        sc = cfg.rewards.scales
        self.assertEqual((sc.track_cart_vx, sc.track_cart_vy, sc.track_cart_wz), (1.5, 0.75, 3.0))
        np.testing.assert_allclose(np.square(cfg.rewards.tracking_sigma_axes), [0.1, 0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
