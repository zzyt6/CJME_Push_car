"""Model selection checks, including real legacy/V6 inference and safe failed switches."""
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart.mujoco_viewer import (
    MANUAL, TwoStageViewer, checkpoint_family, find_hl_checkpoint,
    find_viewer_checkpoint, list_viewer_runs,
)


class ViewerDiscoveryTests(unittest.TestCase):
    def test_viewer_searches_both_roots_without_changing_legacy_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_root, v6_root = Path(tmp) / "homie_pushcart", Path(tmp) / "homie_pushcart_v6"
            old, new = legacy_root / "2026-09-21_v32", v6_root / "2026-09-29_v6"
            for run in (old, new):
                run.mkdir(parents=True)
                for iteration in (200, 1000):
                    (run / f"model_{iteration}.pt").touch()
            (new / "model_9999.pt").mkdir()  # A directory is not a checkpoint.
            (v6_root / "2026-09-30_empty").mkdir()
            roots = (legacy_root, v6_root)
            self.assertEqual(list_viewer_runs(roots), [old, new])
            self.assertEqual(find_hl_checkpoint(log_dir=legacy_root), old / "model_1000.pt")
            self.assertEqual(find_viewer_checkpoint(log_dirs=roots), new / "model_1000.pt")
            self.assertEqual(find_viewer_checkpoint(run=Path(old.name), log_dirs=roots), old / "model_1000.pt")
            self.assertEqual(find_viewer_checkpoint(run=new, log_dirs=roots), new / "model_1000.pt")
            self.assertEqual(find_viewer_checkpoint(checkpoint=old / "model_200.pt", log_dirs=roots), old / "model_200.pt")
            with self.assertRaises(FileNotFoundError):
                find_viewer_checkpoint(run=Path("missing"), log_dirs=roots)

    def test_ambiguous_run_name_requires_full_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = (Path(tmp) / "a", Path(tmp) / "b")
            for root in roots:
                (root / "same_run").mkdir(parents=True)
                (root / "same_run/model_1.pt").touch()
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                find_viewer_checkpoint(run=Path("same_run"), log_dirs=roots)
            self.assertEqual(find_viewer_checkpoint(run=roots[1] / "same_run", log_dirs=roots),
                             roots[1] / "same_run/model_1.pt")

    def test_family_uses_top_task_version_and_does_not_construct_yaml_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "params").mkdir()
            env = run / "params/env.yaml"
            env.write_text("task:\n  baseline_source:\n    version: V3.2\n  version: 6\n"
                           "unused: !!python/object:unknown.NotConstructed {}\n", encoding="utf-8")
            self.assertEqual(checkpoint_family(run / "model_0.pt"), "v6")
            env.write_text("task:\n  curriculum: {}\n", encoding="utf-8")
            self.assertEqual(checkpoint_family(run), "legacy")
            env.write_text("task:\n  version: 5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                checkpoint_family(run)


LEGACY_CHECKPOINT = (ROOT / "logs/rsl_rl/homie_pushcart/"
                     "2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_12000.pt")
V6_CHECKPOINT = (ROOT / "logs/rsl_rl/homie_pushcart_v6/"
                 "2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_12000.pt")


@unittest.skipUnless(LEGACY_CHECKPOINT.is_file() and V6_CHECKPOINT.is_file(),
                     "real V3.2 and V6 test checkpoints are not installed")
class ViewerModelSwitchTests(unittest.TestCase):
    def test_real_v3_v6_v3_switch_preserves_controls_and_failed_load_keeps_running(self):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = SimpleNamespace(checkpoint=LEGACY_CHECKPOINT, run=None, level=17,
                               homie_checkpoint=None, no_invert=False, smoothing=0.1,
                               port=port, payload=7.0, scenario=None, auto=False,
                               transient_s=1.5, render_hz=25.0)
        viewer = None
        try:
            viewer = TwoStageViewer(args)
            base_mass = float(viewer.env.cart_params["mass"]["base_kg"])
            self.assertAlmostEqual(viewer.runner.cart_mass_total, base_mass + 7.0)
            self.assertEqual(viewer.family, "legacy")
            self.assertEqual(viewer.runner.hl_obs.shape, (620,))
            original_runner = viewer.runner
            original_renderer = viewer.renderer
            viewer.hist["t"].append(999.0)
            viewer.scenario_stats["old_model"] = {}
            viewer.scenario, viewer._acc = "hold", {"old": True}
            viewer.sl_payload.value = 15.0
            viewer.sl_level.value = 11
            viewer.cb_hl.value = False
            viewer.dd_run.value = viewer._run_label(V6_CHECKPOINT.parent)
            # GUI events only queue changes, including the high-level switch.
            self.assertIs(viewer.runner, original_runner)
            self.assertIsNotNone(viewer.runner.hl)
            self.assertTrue(viewer._process_requests())
            self.assertEqual(viewer.family, "v6")
            self.assertIsNot(viewer.runner, original_runner)
            self.assertIsNot(viewer.renderer, original_renderer)
            self.assertIs(viewer.policy.runner, viewer.runner)
            self.assertIs(viewer.policy.policy, viewer.runner.v6_policy)
            self.assertIsNone(viewer.runner.hl)
            self.assertEqual(viewer.runner.level, 11)
            self.assertAlmostEqual(viewer.runner.cart_mass_total, base_mass + 15.0)
            self.assertEqual(viewer.runner.hl_obs.shape, (605,))
            self.assertEqual(tuple(viewer.runner.estimator_obs.shape), (1, 462))
            self.assertTrue(all(len(values) == 0 for values in viewer.hist.values()))
            self.assertFalse(viewer.scenario_stats)
            self.assertIsNone(viewer.scenario)
            self.assertIsNone(viewer._acc)
            self.assertEqual(viewer.dd_scn.value, MANUAL)
            self.assertEqual(viewer.counts["resets"], 1)
            self.assertFalse(viewer.flags["reset"])
            self.assertIn("V6", viewer.md_header.content)
            self.assertIn(V6_CHECKPOINT.parent.name, viewer.md_header.content)

            # Selecting an earlier snapshot within the same run also recreates
            # the complete runner instead of retaining stale estimator history.
            latest_runner = viewer.runner
            viewer.dd_ckpt.value = V6_CHECKPOINT.name
            self.assertIs(viewer.runner, latest_runner)
            self.assertTrue(viewer._process_requests())
            self.assertIsNot(viewer.runner, latest_runner)
            self.assertEqual(viewer.hl_checkpoint, V6_CHECKPOINT)
            viewer.cb_hl.value = True
            self.assertIsNone(viewer.runner.hl)
            viewer._process_requests()
            self.assertIs(viewer.runner.hl, viewer.policy)
            for _ in range(32):
                viewer.runner.step(np.array([0.3, 0.0, 0.1]))
            self.assertTrue(np.isfinite(viewer.runner.hl_actions).all())
            self.assertTrue(np.isfinite(viewer.runner.pred_hands).all())
            self.assertTrue(np.isfinite(viewer.runner.pred_cart_velocity).all())

            # Bad checkpoint in a different run must not disturb the active model.
            with tempfile.TemporaryDirectory() as tmp:
                bad_run = Path(tmp) / "new_v6_run"
                (bad_run / "params").mkdir(parents=True)
                for name in ("env.yaml", "agent.yaml"):
                    shutil.copy2(V6_CHECKPOINT.parent / "params" / name, bad_run / "params" / name)
                bad = bad_run / "model_0.pt"
                bad.write_bytes(b"invalid checkpoint")
                stable_runner, stable_renderer = viewer.runner, viewer.renderer
                stable_qpos = viewer.env.data.qpos.copy()
                viewer._queue_request("checkpoint", bad)
                self.assertFalse(viewer._process_requests())
                self.assertIs(viewer.runner, stable_runner)
                self.assertIs(viewer.renderer, stable_renderer)
                np.testing.assert_array_equal(viewer.env.data.qpos, stable_qpos)
                self.assertEqual(viewer.dd_ckpt.value, V6_CHECKPOINT.name)
                self.assertIn("加载失败", viewer.md_load.content)
                viewer.runner.step(np.zeros(3))
                self.assertTrue(np.isfinite(viewer.env.data.qpos).all())

            viewer.dd_run.value = viewer._run_label(LEGACY_CHECKPOINT.parent)
            self.assertTrue(viewer._process_requests())
            viewer.dd_ckpt.value = LEGACY_CHECKPOINT.name
            self.assertTrue(viewer._process_requests())
            self.assertEqual(viewer.family, "legacy")
            self.assertEqual(viewer.hl_checkpoint, LEGACY_CHECKPOINT)
            self.assertEqual(viewer.runner.level, 11)
            self.assertAlmostEqual(viewer.runner.cart_mass_total, base_mass + 15.0)
            self.assertEqual(viewer.runner.hl_obs.shape, (620,))
            for _ in range(32):
                viewer.runner.step(np.array([0.3, 0.0, 0.1]))
            self.assertTrue(np.isfinite(viewer.runner.hl_actions).all())
        finally:
            if viewer is not None:
                viewer.server.stop()
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
