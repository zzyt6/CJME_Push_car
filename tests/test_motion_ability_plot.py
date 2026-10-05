"""Synthetic schema/metric tests only; fixtures are not experiment measurements."""
import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_motion_ability", ROOT / "scripts" / "plot_motion_ability.py")
plotter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plotter
SPEC.loader.exec_module(plotter)


def save_fixture(root, axis="vx", method="CWC", values=(0.0, 0.3), *, stop_after=None):
    """Artificial instantaneous steps deliberately expose cross-stage smearing."""
    dt = 0.02
    stages = [{"start_s": i * 4.0, "end_s": (i + 1) * 4.0, "value": value}
              for i, value in enumerate(values)]
    n = len(stages) * 200 if stop_after is None else stop_after
    starts = np.arange(n, dtype=np.float64) * dt
    stage_ids = (np.arange(n) // 200).astype(int)
    command = np.zeros((n, 3))
    command[:, plotter.AXES[axis][0]] = np.asarray(values)[stage_ids]
    terminated = np.zeros(n, dtype=bool)
    if stop_after is not None:
        terminated[-1] = True
    metadata = {"axis": axis, "method": method, "completed": stop_after is None, "dt": dt,
                "payload_kg": 15, "duration_s": len(stages) * 4.0, "stages": stages}
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez(raw_dir / f"{axis}_{method}.npz", t=starts + dt, interval_start=starts,
             command=command, reference=command, cart_velocity=command, terminated=terminated)
    (raw_dir / f"{axis}_{method}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return plotter.load_trial(root, axis, method)


def attach_reset_state(root, axis="vx", method="CWC", velocity=(0.0, 0.0, 0.0)):
    name = f"raw/{axis}_{method}_initial_state.npz"
    np.savez(root / name, t=np.asarray(0.0), cart_velocity=np.asarray(velocity))
    path = root / "raw" / f"{axis}_{method}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(initial_state_file=name, initial_state_sha256=plotter.sha256(root / name))
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return plotter.load_trial(root, axis, method)


class PlotSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_plot_test_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_display_window_does_not_mix_command_stages(self):
        trial = save_fixture(self.root)
        display = plotter.local_display(trial, 0.2)
        np.testing.assert_allclose(display["mean"], trial.signal, atol=1e-14)
        np.testing.assert_allclose(display["min"], trial.signal, atol=1e-14)
        np.testing.assert_allclose(display["max"], trial.signal, atol=1e-14)
        np.testing.assert_allclose(display["std"], 0.0, atol=1e-14)
        self.assertEqual(display["n"][100], 11)
        self.assertEqual(display["n"][199], 6)
        self.assertEqual(display["n"][200], 6)
        self.assertEqual(trial.stage_index[199], 0)  # Interval [3.98, 4.00] uses the old command.
        self.assertEqual(trial.stage_index[200], 1)  # Interval [4.00, 4.02] uses the new command.

    def test_half_second_std_is_population_and_truncated_at_boundaries(self):
        trial = save_fixture(self.root)
        trial.velocity[:, 0] += np.tile([-0.1, 0.1], len(trial.t) // 2)
        display = plotter.local_display(trial, 0.5)
        self.assertEqual(display["n"][100], 25)
        self.assertEqual(display["n"][199], 13)
        self.assertEqual(display["n"][200], 13)
        for pos, expected in ((100, trial.signal[88:113]), (199, trial.signal[187:200]),
                              (200, trial.signal[200:213])):
            self.assertAlmostEqual(display["mean"][pos], expected.mean())
            self.assertAlmostEqual(display["std"][pos], expected.std(ddof=0))
            self.assertAlmostEqual(display["lower"][pos], expected.mean() - expected.std(ddof=0))
            self.assertAlmostEqual(display["upper"][pos], expected.mean() + expected.std(ddof=0))

    def test_raw_metrics_keep_alternating_ripple(self):
        trial = save_fixture(self.root)
        trial.velocity[:, 0] += np.tile([-0.1, 0.1], len(trial.t) // 2)
        rows = plotter.stage_metrics(trial)
        for row in rows:
            self.assertEqual(row["samples"], 100)
            self.assertAlmostEqual(row["raw_rmse"], 0.1)
            self.assertAlmostEqual(row["raw_ripple_rms_demeaned"], 0.1)
            self.assertAlmostEqual(row["raw_bias"], 0.0)
        display = plotter.local_display(trial, 0.2)
        self.assertLess(np.std(display["mean"][:200]), 0.015)
        np.testing.assert_allclose(trial.signal[:2], [-0.1, 0.1])

    def test_early_termination_not_filled_or_declared_complete(self):
        trial = save_fixture(self.root, stop_after=150)
        self.assertEqual(len(plotter.local_display(trial, 0.2)["mean"]), 150)
        rows = plotter.stage_metrics(trial)
        self.assertFalse(rows[0]["stage_completed"])
        self.assertEqual(rows[0]["samples"], 50)
        self.assertEqual(rows[1]["samples"], 0)
        self.assertIsNone(rows[1]["raw_rmse"])

    def test_reject_reset_splicing(self):
        trial = save_fixture(self.root)
        path = trial.source_paths[0]
        with np.load(path) as npz:
            arrays = dict(npz)
        arrays["terminated"][20] = True
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "continue after termination"):
            plotter.load_trial(self.root, "vx", "CWC")

    def test_reject_command_shift(self):
        trial = save_fixture(self.root)
        path = trial.source_paths[0]
        with np.load(path) as npz:
            arrays = dict(npz)
        arrays["command"] = np.roll(arrays["command"], 1, axis=0)
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "commands differ"):
            plotter.load_trial(self.root, "vx", "CWC")

    def test_pair_allows_truncation_but_rejects_different_payload(self):
        cwc = save_fixture(self.root)
        baseline = save_fixture(self.root, method="baseline", stop_after=150)
        plotter.validate_pair(cwc, baseline)
        baseline.metadata["payload_kg"] = 25
        with self.assertRaisesRegex(ValueError, "different payload_kg"):
            plotter.validate_pair(cwc, baseline)

    def test_five_stage_wz_protocol(self):
        trial = save_fixture(self.root, axis="wz", values=(0.0, 0.1, 0.2, 0.3, 0.4))
        self.assertEqual(len(trial.stages), 5)
        self.assertEqual(len(plotter.stage_metrics(trial)), 5)
        self.assertEqual(trial.stage_index[-1], 4)

    def test_render_preserves_sources_and_exports_raw_and_band_variants(self):
        (self.root / "manifest.json").write_text(json.dumps({"fixture": "synthetic, not experiment data",
                                                            "engine": "Isaac Sim / PhysX", "payload_kg": 15}),
                                                encoding="utf-8")
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method)
        source_paths = list((self.root / "raw").iterdir()) + [self.root / "manifest.json"]
        original_hashes = {path: plotter.sha256(path) for path in source_paths}
        manifest = plotter.render(self.root, dpi=72)
        for extension in ("png", "svg", "pdf"):
            self.assertEqual(len(list((self.root / "figures").glob(f"*.{extension}"))), 8)
        shapes = [plotter.plt.imread(self.root / "figures" / f"{axis}.png").shape[:2]
                  for axis in plotter.AXES]
        self.assertEqual(len(set(shapes)), 1)
        self.assertEqual(shapes[0], (int(58 / 25.4 * 72), int(60 / 25.4 * 72)))
        combined_shape = plotter.plt.imread(self.root / "figures" / "motion_ability.png").shape[:2]
        self.assertEqual(combined_shape, (int(58 / 25.4 * 72), int(180 / 25.4 * 72)))
        self.assertEqual(manifest["layout"]["legend"], ["vel cmd", "CWC", "baseline"])
        self.assertFalse(manifest["layout"]["shared_legend"])
        self.assertEqual(manifest["layout"]["combined_legend_panels"], ["vx"])
        self.assertEqual(manifest["layout"]["standalone_legend_panels"], ["vx"])
        for axis in plotter.AXES:
            svg = (self.root / "figures" / f"{axis}.svg").read_text(encoding="utf-8")
            self.assertEqual("vel cmd" in svg, axis == "vx")
        self.assertEqual(manifest["display"]["standard_deviation_ddof"], 0)
        self.assertEqual(manifest["simulation_engine"], "Isaac Sim / PhysX")
        self.assertIn("Cart velocity tracking in Isaac Sim / PhysX with a 15 kg payload",
                      (self.root / "FIGURES_README.md").read_text(encoding="utf-8"))
        for path, digest in original_hashes.items():
            self.assertEqual(plotter.sha256(path), digest)
        self.assertEqual(len(manifest["sources"]), 13)
        self.assertEqual(len(manifest["trial_status"]), 6)
        for entry in manifest["outputs"]:
            self.assertEqual(plotter.sha256(self.root / entry["path"]), entry["sha256"])

    def test_mujoco_metadata_controls_caption_and_reproduction_directory(self):
        (self.root / "manifest.json").write_text(json.dumps({"fixture": "synthetic, not experiment data",
                                                            "engine": "MuJoCo", "payload_kg": 15}),
                                                encoding="utf-8")
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method)
        manifest = plotter.render(self.root, dpi=72)
        readme = (self.root / "FIGURES_README.md").read_text(encoding="utf-8")
        self.assertEqual(manifest["simulation_engine"], "MuJoCo")
        self.assertIn("Cart velocity tracking in MuJoCo with a 15 kg payload", readme)
        self.assertIn(self.root.as_posix(), readme)
        self.assertNotIn("Isaac", readme)

    def test_reject_missing_engine_or_manifest_payload_mismatch(self):
        (self.root / "manifest.json").write_text('{"payload_kg":15}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "simulation engine"):
            plotter.render(self.root, dpi=72)
        (self.root / "manifest.json").write_text('{"engine":"MuJoCo","payload_kg":25}', encoding="utf-8")
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method)
        with self.assertRaisesRegex(ValueError, "different payload_kg"):
            plotter.render(self.root, dpi=72)

    def test_real_reset_record_is_display_only_not_a_smoothing_sample(self):
        plain = save_fixture(self.root, values=(0.3, 0.5))
        original_metrics = plotter.stage_metrics(plain)
        trial = attach_reset_state(self.root)
        display = plotter.local_display(trial, 0.5)
        self.assertEqual(trial.initial_t, 0.0)
        np.testing.assert_array_equal(trial.initial_velocity, np.zeros(3))
        self.assertEqual(len(trial.t), 400)
        self.assertEqual(display["n"][0], 13)
        self.assertAlmostEqual(display["mean"][0], 0.3)
        self.assertAlmostEqual(display["std"][0], 0.0)
        self.assertEqual(plotter.stage_metrics(trial), original_metrics)
        self.assertEqual(len(trial.source_paths), 3)

    def test_reset_state_hash_is_verified(self):
        save_fixture(self.root, values=(0.3, 0.5))
        attach_reset_state(self.root)
        np.savez(self.root / "raw/vx_CWC_initial_state.npz", t=0.0, cart_velocity=np.ones(3))
        with self.assertRaisesRegex(ValueError, "initial state hash"):
            plotter.load_trial(self.root, "vx", "CWC")

    def test_reset_row_export_and_only_left_combined_legend(self):
        (self.root / "manifest.json").write_text('{"engine":"Isaac Sim / PhysX","payload_kg":15}',
                                                encoding="utf-8")
        trials = {}
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method, values=(0.3, 0.5))
                trials[(axis, method)] = attach_reset_state(self.root, axis, method)
        manifest = plotter.render(self.root, dpi=72)
        self.assertEqual(len(manifest["sources"]), 19)
        self.assertEqual(manifest["display"]["reset_initial_records"], 6)
        with (self.root / "derived/display_vx_CWC.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 401)
        self.assertEqual(rows[0]["sample_kind"], "reset_initial")
        self.assertEqual(rows[0]["window_samples"], "0")
        self.assertEqual(rows[0]["interval_start_s"], "")
        self.assertEqual(rows[0]["internal_reference"], "")
        self.assertEqual(float(rows[0]["actual_raw"]), 0.0)
        self.assertEqual(float(rows[0]["display_lower"]), float(rows[0]["display_upper"]))
        self.assertEqual(rows[1]["sample_kind"], "control_end")
        self.assertAlmostEqual(float(rows[1]["display_mean"]), 0.3)
        display = {key: plotter.local_display(trial, 0.5) for key, trial in trials.items()}
        fig, axes = plotter.panel_figure(3)
        for i, (ax, axis) in enumerate(zip(axes, plotter.AXES)):
            handles = plotter.plot_panel(ax, axis, trials, display, raw=False, show_legend=i == 0)
            for line in handles[1:]:
                self.assertEqual(line.get_xdata()[0], 0.0)
                self.assertEqual(line.get_ydata()[0], 0.0)
                self.assertEqual(len(line.get_xdata()), 401)
        self.assertIsNotNone(axes[0].get_legend())
        self.assertIsNone(axes[1].get_legend())
        self.assertIsNone(axes[2].get_legend())
        plotter.plt.close(fig)

    def test_display_initialization_hold_shifts_only_plot_clock(self):
        (self.root / "manifest.json").write_text('{"engine":"Isaac Sim / PhysX","payload_kg":15}',
                                                encoding="utf-8")
        trials = {}
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method, values=(0.3, 0.5))
                trials[(axis, method)] = attach_reset_state(self.root, axis, method)
        original = plotter.render(self.root, dpi=72)
        metrics_hash = plotter.sha256(self.root / "derived/stage_metrics.csv")
        updated = plotter.render(self.root, dpi=72, initialization_hold_s=4.0)
        self.assertEqual(original["sources"], updated["sources"])
        self.assertEqual(metrics_hash, plotter.sha256(self.root / "derived/stage_metrics.csv"))
        self.assertEqual(updated["display"]["initialization_hold_s"], 4.0)
        self.assertFalse(updated["metrics"]["initialization_hold_in_metrics"])
        with (self.root / "derived/display_vx_CWC.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 402)
        for row, t_display in zip(rows[:2], [0.0, 4.0]):
            self.assertEqual(row["sample_kind"], "initialization_hold")
            self.assertEqual(float(row["simulation_time_s"]), 0.0)
            self.assertEqual(float(row["plot_time_s"]), t_display)
            self.assertEqual(row["window_samples"], "0")
            self.assertEqual(float(row["display_lower"]), float(row["display_upper"]))
        self.assertAlmostEqual(float(rows[2]["simulation_time_s"]), 0.02)
        self.assertAlmostEqual(float(rows[2]["plot_time_s"]), 4.02)
        self.assertAlmostEqual(float(rows[2]["display_mean"]), 0.3)
        display = {key: plotter.local_display(trial, 0.5) for key, trial in trials.items()}
        fig, (ax,) = plotter.panel_figure(1)
        handles = plotter.plot_panel(ax, "vx", trials, display, raw=True, initialization_hold_s=4.0)
        np.testing.assert_allclose(handles[0].get_data().edges, [0.0, 4.0, 8.0, 12.0])
        np.testing.assert_allclose(handles[0].get_data().values, [0.0, 0.3, 0.5])
        for line in handles[1:]:
            np.testing.assert_allclose(line.get_xdata()[:3], [0.0, 4.0, 4.02])
            np.testing.assert_allclose(line.get_ydata()[:2], 0.0)
            np.testing.assert_allclose(line.get_ydata()[2:], trials[("vx", line.get_label())].signal)
        self.assertEqual(ax.get_xlim(), (0.0, 12.0))
        self.assertNotIn("Init.", [text.get_text() for text in ax.texts])
        plotter.plt.close(fig)
        readme = (self.root / "FIGURES_README.md").read_text(encoding="utf-8")
        self.assertIn("simulation is not advanced", readme)
        self.assertIn("--initialization-hold-s 4", readme)

    def test_display_hold_requires_measured_initial_states(self):
        (self.root / "manifest.json").write_text('{"engine":"MuJoCo","payload_kg":15}', encoding="utf-8")
        for axis in plotter.AXES:
            for method in plotter.METHODS:
                save_fixture(self.root, axis=axis, method=method)
        with self.assertRaisesRegex(ValueError, "verified reset record"):
            plotter.render(self.root, dpi=72, initialization_hold_s=4.0)

    def test_axes_have_identical_physical_sizes_single_and_horizontal(self):
        single, (single_ax,) = plotter.panel_figure(1)
        combined, axes = plotter.panel_figure(3)
        for fig, ax in [(single, single_ax)] + [(combined, ax) for ax in axes]:
            rectangle = ax.get_position()
            width_mm = rectangle.width * fig.get_figwidth() * 25.4
            height_mm = rectangle.height * fig.get_figheight() * 25.4
            self.assertAlmostEqual(width_mm, 45.0)
            self.assertAlmostEqual(height_mm, 38.0)
        plotter.plt.close(single)
        plotter.plt.close(combined)

    def test_single_wz_units_and_legend_fit_fixed_canvas(self):
        with plotter.plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                                    "xtick.labelsize": 7, "ytick.labelsize": 7}):
            fig, (ax,) = plotter.panel_figure(1)
            plotter.style_axis(ax, "wz", 20.0)
            ax.set_ylim(-0.21, 0.56)
            handles = [ax.plot([], [], label=name)[0] for name in ("vel cmd", "CWC", "baseline")]
            plotter.panel_legend(ax, handles)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            self.assertEqual(len(fig.legends), 0)
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                             ["vel cmd", "CWC", "baseline"])
            for artist in (ax.yaxis.label, ax.xaxis.label, ax.get_legend()):
                box = artist.get_window_extent(renderer)
                self.assertGreaterEqual(box.x0, 0)
                self.assertGreaterEqual(box.y0, 0)
                self.assertLessEqual(box.x1, fig.bbox.width)
                self.assertLessEqual(box.y1, fig.bbox.height)
            plotter.plt.close(fig)

    def test_old_publication_archive_is_created_once_per_manifest(self):
        old = self.root / "plots_manifest.json"
        old.write_text('{"old":"original publication"}', encoding="utf-8")
        (self.root / "figures").mkdir()
        figure = self.root / "figures" / "vx.svg"
        figure.write_text("old original figure", encoding="utf-8")
        first = plotter.archive_existing_artifacts(self.root)
        second = plotter.archive_existing_artifacts(self.root)
        self.assertEqual(first, second)
        self.assertEqual(plotter.sha256(first / "plots_manifest.json"), plotter.sha256(old))
        self.assertEqual((first / "figures" / "vx.svg").read_text(), "old original figure")
        self.assertEqual(len(list((self.root / "archive").iterdir())), 1)
        figure.write_text("new figure", encoding="utf-8")
        self.assertEqual((first / "figures" / "vx.svg").read_text(), "old original figure")


if __name__ == "__main__":
    unittest.main()
