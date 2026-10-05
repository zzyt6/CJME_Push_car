"""Bench viewer V3/V6 selection keeps each trained observation adapter and rolls back bad loads."""
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

from pushcart_bench.view import BenchViewer, checkpoint_policy_spec
from pushcart_bench.runner import Condition
from scripts.view_v6_bench import main, resolve_checkpoint

V3 = (ROOT / "logs/rsl_rl/homie_pushcart/"
      "2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_12000.pt")
V6 = (ROOT / "logs/rsl_rl/homie_pushcart_v6/"
      "2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_12000.pt")


class BenchCheckpointRoutingTests(unittest.TestCase):
    def test_saved_family_routes_to_its_original_bench_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "params").mkdir()
            env = run / "params/env.yaml"
            checkpoint = run / "model_10.pt"
            checkpoint.touch()
            env.write_text("task:\n  version: V3.2\n", encoding="utf-8")
            self.assertEqual(checkpoint_policy_spec(checkpoint), f"two_stage:{checkpoint.resolve()}")
            env.write_text("task:\n  version: 6\n  baseline_source:\n    version: V3.2\n", encoding="utf-8")
            self.assertEqual(checkpoint_policy_spec(checkpoint), f"v6:{checkpoint.resolve()}")
            self.assertEqual(resolve_checkpoint(checkpoint), checkpoint.resolve())
            self.assertEqual(resolve_checkpoint(run=run), checkpoint.resolve())

    def test_default_latest_stays_v6(self):
        with patch("pushcart_bench.policies_v6.resolve_v6_checkpoint", return_value=V6) as resolve:
            self.assertEqual(resolve_checkpoint(), V6.resolve())
        resolve.assert_called_once_with("latest")

    def test_cli_accepts_explicit_v3_checkpoint_and_enables_browser_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "params").mkdir()
            (run / "params/env.yaml").write_text("task:\n  version: V3.2\n", encoding="utf-8")
            checkpoint = run / "model_10.pt"
            checkpoint.touch()
            with patch("pushcart_bench.experiments.exp_path") as experiment:
                main(["--checkpoint", str(checkpoint), "--path", "s_large"])
            args, _ = experiment.call_args.args
            self.assertEqual(args.policies, [f"two_stage:{checkpoint.resolve()}"])
            self.assertTrue(args.checkpoint_selection)
            self.assertEqual(args.scene_version, "original")


class BenchSelectionRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = {}
        for name in ("old", "new", "newer"):
            run = Path(self.temp.name) / name
            run.mkdir()
            for iteration in (1, 10):
                (run / f"model_{iteration}.pt").touch()
            self.runs[name] = run.resolve()
        v = self.viewer = BenchViewer.__new__(BenchViewer)
        v._selection_lock = threading.RLock()
        v.live_policy = object()
        v.run_choices = self.runs
        v.dd_run = SimpleNamespace(value="old")
        v.dd_ckpt = SimpleNamespace(value="model_10.pt", options=("model_1.pt", "model_10.pt"))
        v._checkpoint_run = self.runs["old"]
        v._checkpoint_names = v.dd_ckpt.options
        v._syncing_checkpoints = False
        v.selected_checkpoint = v.requested_checkpoint = self.runs["old"] / "model_10.pt"
        v.md_load = SimpleNamespace(content="")
        v.want_rerun = False

    def test_start_while_run_discovery_pending_resolves_current_gui_not_stale_selected_value(self):
        from homie_pushcart.mujoco_viewer import list_checkpoints
        v = self.viewer
        entered, release = threading.Event(), threading.Event()
        def slow_list(run):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test discovery gate")
            return list_checkpoints(run)
        v.dd_run.value = "new"
        with patch("homie_pushcart.mujoco_viewer.list_checkpoints", slow_list):
            select = threading.Thread(target=v._select_run)
            select.start()
            self.assertTrue(entered.wait(3))
            start = threading.Thread(target=v.request_start)
            start.start()
            release.set()
            select.join(3)
            start.join(3)
            self.assertFalse(select.is_alive() or start.is_alive())
        self.assertEqual(v.requested_checkpoint, self.runs["new"] / "model_10.pt")
        self.assertTrue(v.want_rerun)

    def test_newer_run_wins_during_discovery_and_late_run_callback_keeps_explicit_checkpoint(self):
        from homie_pushcart.mujoco_viewer import list_checkpoints
        v = self.viewer
        entered, release = threading.Event(), threading.Event()
        def slow_list(run):
            if run == self.runs["new"]:
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test discovery gate")
            return list_checkpoints(run)
        v.dd_run.value = "new"
        with patch("homie_pushcart.mujoco_viewer.list_checkpoints", slow_list):
            select = threading.Thread(target=v._select_run)
            select.start()
            self.assertTrue(entered.wait(3))
            # Viser updates handle values before dispatching callbacks to workers.
            v.dd_run.value = "newer"
            release.set()
            select.join(3)
            self.assertFalse(select.is_alive())
        self.assertEqual(v.selected_checkpoint, self.runs["newer"] / "model_10.pt")
        v.dd_ckpt.value = "model_1.pt"
        v._select_checkpoint()
        v._select_run()  # Earlier browser event reaches its worker late.
        v.request_start()
        self.assertEqual(v.requested_checkpoint, self.runs["newer"] / "model_1.pt")

    def test_start_validates_checkpoint_belongs_to_selected_run(self):
        v = self.viewer
        v.dd_ckpt.value = "../new/model_10.pt"
        v.request_start()
        self.assertFalse(v.want_rerun)
        self.assertIn("无法开始", v.md_load.content)

    def test_manual_restart_is_not_listed_as_completed_timeout(self):
        v = self.viewer
        v.results_rows = []
        v.results_md = SimpleNamespace(content="unchanged")
        result = {"kind": "path", "label": "interrupted", "policy": {"name": "old"},
                  "metrics": {"success": False, "time_to_goal": None}}
        v.want_rerun = True
        v.end_run(result)
        self.assertEqual(v.results_rows, [])
        self.assertEqual(v.results_md.content, "unchanged")
        v.want_rerun = False
        v.end_run(result)
        self.assertEqual(len(v.results_rows), 1)


@unittest.skipUnless(V3.is_file() and V6.is_file(), "real V3/V6 checkpoints are unavailable")
class BenchViewerSwitchTests(unittest.TestCase):
    def test_actual_v6_v3_passthrough_v6_switch_and_failed_checkpoint_preserves_live_scene(self):
        import torch
        from pushcart_bench.policies import make_policy
        from pushcart_bench.paths import build_path
        from pushcart_bench.config_io import load_config
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        viewer = None
        try:
            initial = make_policy(checkpoint_policy_spec(V6))
            condition = Condition(payload_kg=15.0, seed=1)
            condition.apply(initial)
            viewer = BenchViewer(port=port, speed=0.0)
            viewer.setup_live(["s_large"], [initial.name], payload=15.0)
            viewer.setup_checkpoint_selection(initial, level=17)
            path = build_path("s_large", load_config()["paths"]["s_large"])
            viewer.begin_run(path, initial, "initial")
            original_renderer = viewer.renderer
            for checkpoint, expected_shape, expected_kind in (
                    (V3, (620,), "two_stage"), (None, (620,), "passthrough"), (V6, (605,), "v6")):
                before = viewer.live_policy
                viewer.hist["t"].append(999.0)
                viewer.trail.append((9., 9., 0.))
                run_dir = checkpoint.parent if checkpoint is not None else None
                label = next(label for label, run in viewer.run_choices.items() if run == run_dir)
                # Browsing choices never loads another network or mutates the simulation.
                with patch("pushcart_bench.policies.make_policy", side_effect=AssertionError("eager load")):
                    viewer.dd_run.value = label
                    if checkpoint is not None:
                        viewer.dd_ckpt.value = checkpoint.name
                self.assertEqual(viewer.dd_ckpt.disabled, checkpoint is None)
                self.assertIs(viewer.live_policy, before)
                self.assertEqual(viewer.selected_checkpoint, checkpoint)
                viewer.request_start()
                candidate = viewer.prepare_selected_policy(condition)
                self.assertIsNotNone(candidate)
                self.assertIsNot(candidate, before)
                self.assertEqual(candidate.describe()["kind"], expected_kind)
                self.assertEqual(checkpoint_policy_spec(viewer.loaded_checkpoint).split(":")[0], expected_kind)
                if checkpoint is None:
                    self.assertIsNone(candidate.runner.hl)
                    self.assertEqual(viewer.dd_ckpt.value, "无需检查点")
                    self.assertIs(viewer.prepare_selected_policy(condition), candidate)
                self.assertEqual(candidate.runner.hl_obs.shape, expected_shape)
                self.assertEqual(candidate.env.xml_path, initial.env.xml_path)
                self.assertAlmostEqual(candidate.cart_mass_total(), initial.cart_mass_total())
                np.testing.assert_array_equal(candidate.env.model.dof_damping, initial.env.model.dof_damping)
                np.testing.assert_array_equal(candidate.env.model.dof_armature, initial.env.model.dof_armature)
                viewer.begin_run(path, candidate, "switched")
                self.assertIsNot(viewer.renderer, original_renderer)
                self.assertFalse(viewer.trail)
                self.assertTrue(all(not history for history in viewer.hist.values()))
                self.assertEqual(len(viewer.plot_v.figure.data), 0)
                self.assertFalse(any(candidate.env.model.geom(g).name.startswith(("cart_", "wheel_", "caster_"))
                                     for g in viewer.renderer.collision_handles))
                for _ in range(32):
                    candidate.step(np.array([0.3, 0.0, 0.1]))
                self.assertTrue(np.isfinite(candidate.env.data.qpos).all())
                self.assertTrue(np.isfinite(candidate.runner.hl_actions).all())
                if expected_kind == "v6":
                    self.assertEqual(tuple(candidate.runner.estimator_obs.shape), (1, 462))
                    self.assertTrue(np.isfinite(candidate.runner.pred_cart_velocity).all())
                    self.assertTrue(np.isfinite(candidate.runner.pred_hands).all())
                original_renderer = viewer.renderer

            with tempfile.TemporaryDirectory() as tmp:
                bad_run = Path(tmp)
                shutil.copytree(V6.parent / "params", bad_run / "params")
                bad = bad_run / "model_0.pt"
                bad.write_bytes(b"invalid checkpoint")
                policy, renderer = viewer.live_policy, viewer.renderer
                qpos = policy.env.data.qpos.copy()
                history = list(viewer.hist["t"])
                viewer.run_choices["invalid"] = bad_run.resolve()
                viewer.dd_run.options = (*viewer.dd_run.options, "invalid")
                viewer.dd_run.value = "invalid"
                viewer.request_start()
                self.assertIsNone(viewer.prepare_selected_policy(condition))
                self.assertIs(viewer.live_policy, policy)
                self.assertIs(viewer.renderer, renderer)
                np.testing.assert_array_equal(policy.env.data.qpos, qpos)
                self.assertEqual(list(viewer.hist["t"]), history)
                self.assertIn("加载失败", viewer.md_load.content)
                self.assertEqual(viewer.loaded_checkpoint, V6)
        finally:
            if viewer is not None:
                viewer.server.stop()
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
