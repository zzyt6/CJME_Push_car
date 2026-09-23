"""V4 launcher, saved configuration and official train/play argument regression checks."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart import training


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pushcart_entrypoint_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "logs/rsl_rl/homie_pushcart/2026-09-22_v4"
        (self.run / "params").mkdir(parents=True)
        self.checkpoint = self.run / "model_20.pt"
        self.checkpoint.touch()
        self.root_patch = patch.object(training, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def call_entrypoint(self, name, args):
        with patch.object(training, "_lab_scripts", return_value=Path("official")), \
                patch.object(training, "_run", return_value=0) as run:
            self.assertEqual(getattr(training, name)(args), 0)
        return run.call_args

    def test_v4_scratch_and_explicit_override_precedence(self):
        call = self.call_entrypoint("train", ["--quality_profile", "v4", "--num_envs", "8", "--headless",
                                               "--level", "0", "--run_name", "v4_scratch",
                                               "env.task.rewards.scales.cart_yaw_ripple=-0.2"])
        args = call.args[1]
        self.assertNotIn("--resume", args)
        self.assertNotIn("--checkpoint", args)
        self.assertNotIn("restore_checkpoint", call.kwargs)
        self.assertIn("env.task.rewards.force_rate_seed_on_reset=true", args)
        self.assertIn("env.curriculum_level=0", args)
        self.assertLess(args.index("env.task.rewards.scales.cart_yaw_ripple=-0.15"),
                        args.index("env.task.rewards.scales.cart_yaw_ripple=-0.2"))
        self.assertEqual(args[args.index("--experiment_name") + 1], "homie_pushcart")

    def test_explicit_play_keeps_the_absolute_file_path(self):
        call = self.call_entrypoint("play", ["--checkpoint", str(self.checkpoint), "--level", "17"])
        args = call.args[1]
        self.assertEqual(args[args.index("--checkpoint") + 1], str(self.checkpoint.resolve()))
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())
        self.assertNotIn("--resume", args)
        self.assertIn("env.curriculum_level=17", args)

    def test_default_play_selects_latest_numbered_model_and_matching_config(self):
        (self.run / "model_9.pt").touch()
        call = self.call_entrypoint("play", [])
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())

    def test_checkpoint_filename_regex_can_contain_backslash_escapes(self):
        call = self.call_entrypoint("play", ["--checkpoint", r"model_\d+\.pt$"])
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())

    def test_hydra_selectors_restore_the_run_that_will_be_loaded(self):
        later = self.root / "logs/rsl_rl/homie_pushcart/2026-09-23_other"
        later.mkdir()
        (later / "model_40.pt").touch()
        call = self.call_entrypoint("play", ["agent.load_run='.*v4'", r"agent.load_checkpoint='model_\d+\.pt$'"])
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())

    def test_published_checkpoint_mode_does_not_restore_an_unrelated_local_run(self):
        call = self.call_entrypoint("play", ["--use_pretrained_checkpoint"])
        self.assertIn("--use_pretrained_checkpoint", call.args[1])
        self.assertNotIn("restore_checkpoint", call.kwargs)

    def test_play_accepts_a_checkpoint_outside_training_log_root(self):
        checkpoint = self.root / "exported run" / "model_4.pt"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        call = self.call_entrypoint("play", ["--checkpoint", str(checkpoint)])
        self.assertEqual(call.kwargs["restore_checkpoint"], checkpoint.resolve())

    def test_resume_resolves_exact_run_and_checkpoint_with_saved_settings(self):
        call = self.call_entrypoint("train", ["--checkpoint", str(self.checkpoint), "--quality_profile", "v4"])
        args = call.args[1]
        self.assertIn("--resume", args)
        self.assertEqual(args[args.index("--load_run") + 1], "2026\\-09\\-22_v4$")
        self.assertEqual(args[args.index("--checkpoint") + 1], "model_20\\.pt$")
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())

    def test_resume_run_selection_and_custom_experiment(self):
        run = self.root / "logs/rsl_rl/custom/2026-09-23_v4"
        run.mkdir(parents=True)
        checkpoint = run / "model_8.pt"
        checkpoint.touch()
        call = self.call_entrypoint("train", ["--resume", "--load_run", ".*v4", "--experiment_name", "custom"])
        self.assertEqual(call.kwargs["restore_checkpoint"], checkpoint.resolve())

    def test_missing_explicit_path_fails_before_launch(self):
        with patch.object(training, "_run") as run, self.assertRaises(FileNotFoundError):
            training.train(["--checkpoint", str(self.run / "missing.pt")])
        run.assert_not_called()

    def test_scratch_does_not_inherit_restore_environment_variable(self):
        with patch.dict(os.environ, {training._RESTORE_RUN_ENV: "stale"}), \
                patch.object(training.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            training._run(Path("official/train.py"), ["--task", training.TASK_ID])
        self.assertNotIn(training._RESTORE_RUN_ENV, run.call_args.kwargs["env"])

    def test_top_level_cli_forwards_v4_and_unknown_official_arguments(self):
        spec = importlib.util.spec_from_file_location("pushcart_entrypoint_test", ROOT / "scripts/pushcart.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(training, "train", return_value=0) as train:
            self.assertEqual(module.main(["train", "--quality_profile", "v4", "--num_envs", "8", "--headless"]), 0)
        self.assertEqual(train.call_args.args[0], ["--quality_profile", "v4", "--num_envs", "8", "--headless"])

    def test_saved_env_restores_control_settings_and_legacy_behavior(self):
        from homie_isaaclab.math_utils import class_to_dict
        from homie_pushcart.task_config import PushCartTaskCfg
        defaults = dict(task=class_to_dict(PushCartTaskCfg()), homie={"control": {"decimation": 4}},
                        checkpoint=None, sim=SimpleNamespace(dt=.005, gravity=[0., 0., -9.81], device="cuda:0"),
                        scene=SimpleNamespace(num_envs=4096), log_dir=None, episode_length_s=20.)
        factory = lambda: SimpleNamespace(**copy.deepcopy(defaults))
        (self.run / "params/env.yaml").write_text(
            "task:\n  actions:\n    clip: 3.0\n  commands:\n    ref_tau_s: 0.25\n"
            "  rewards:\n    scales:\n      cart_yaw_ripple: -0.15\n"
            "homie:\n  control:\n    decimation: 2\n"
            "checkpoint: null\ncurriculum_level: 6\nepisode_length_s: 12.0\n"
            "sim:\n  dt: 0.01\n  device: cpu\nscene:\n  num_envs: 8\nlog_dir: old_run\n",
            encoding="utf-8")
        (self.run / "pushcart_task.json").write_text(json.dumps({"homie_checkpoint": "frozen.pt"}), encoding="utf-8")
        with patch.dict(os.environ, {training._RESTORE_RUN_ENV: str(self.checkpoint)}), \
                patch.dict(sys.modules, {"homie_pushcart.lab_config": SimpleNamespace(PushCartEnvCfg=factory)}):
            cfg = training.restored_env_cfg()
        self.assertEqual(cfg.task["actions"]["clip"], 3.)
        self.assertEqual(cfg.task["actions"]["idle_gate_speed"], 0.)
        self.assertEqual(cfg.task["commands"]["ref_tau_s"], .25)
        self.assertFalse(cfg.task["rewards"]["force_rate_seed_on_reset"])
        self.assertEqual(cfg.homie["control"]["decimation"], 2)
        self.assertEqual(cfg.checkpoint, "frozen.pt")
        self.assertEqual(cfg.sim.dt, .01)
        self.assertEqual(cfg.sim.device, "cuda:0")
        self.assertEqual(cfg.scene.num_envs, 4096)
        self.assertIsNone(cfg.log_dir)
        self.assertNotIn("source_yaml", cfg.task)
        self.assertEqual((cfg.observation_space, cfg.state_space, cfg.action_space), (620, 652, 18))

    def test_runner_restores_network_and_ppo_but_not_old_run_name(self):
        (self.run / "params/env.yaml").write_text("task: {}\n", encoding="utf-8")
        (self.run / "params/agent.yaml").write_text(
            "actor:\n  hidden_dims: [64, 32]\nalgorithm:\n  gamma: 0.95\nrun_name: old_run\ndevice: cpu\n",
            encoding="utf-8")
        class FakeRunner:
            def __init__(self):
                self.run_name = ""
                self.device = "cuda:0"
            def from_dict(self, values):
                self.__dict__.update(values)
        with patch.dict(os.environ, {training._RESTORE_RUN_ENV: str(self.checkpoint)}), \
                patch.dict(sys.modules, {"homie_pushcart.lab_config": SimpleNamespace(PushCartRunnerCfg=FakeRunner)}):
            cfg = training.restored_runner_cfg()
        self.assertEqual(cfg.actor["hidden_dims"], [64, 32])
        self.assertEqual(cfg.algorithm["gamma"], .95)
        self.assertEqual(cfg.run_name, "")
        self.assertEqual(cfg.device, "cuda:0")


@unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell is unavailable")
class PowerShellEntryPointTests(unittest.TestCase):
    def test_actual_parameter_forwarding_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="pushcart_v4_cli_") as td:
            root = Path(td)
            for filename in ("run_v4_pushcart_train.ps1", "run_one_stage_pushcart.ps1"):
                shutil.copy2(ROOT / filename, root / filename)
            (root / "scripts").mkdir()
            (root / "scripts/pushcart.py").write_text(
                "import sys,json; print(json.dumps(sys.argv[1:])); sys.exit(7)", encoding="utf-8")
            env = dict(os.environ, HOMIE_PYTHON=sys.executable)
            command = ["powershell.exe", "-NoProfile", "-File", str(root / "run_v4_pushcart_train.ps1")]
            for arguments, expected in (([], ["train", "--quality_profile", "v4", "--level", "0", "--num_envs", "4096",
                                              "--max_iterations", "20000", "--run_name", "v4_speed_quality_scratch", "--headless"]),
                                        (["-NumEnvs", "8", "-MaxIterations", "2", "-Level", "6", "-RunName", "v4 audit", "--seed", "7"],
                                         ["train", "--quality_profile", "v4", "--level", "6", "--num_envs", "8", "--max_iterations", "2",
                                          "--run_name", "v4 audit", "--headless", "--seed", "7"])):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([*command, *arguments], env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 7, result.stderr)
                    self.assertEqual(json.loads(result.stdout.strip()), expected)


if __name__ == "__main__":
    unittest.main()
