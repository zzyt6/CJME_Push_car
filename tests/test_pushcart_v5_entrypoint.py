"""V5 isolation, launch selection and actor interface checks; no simulator launch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart_v5 import paths, training
from homie_pushcart_v5.task_config import PushCartTaskCfg, num_actor_obs, num_critic_obs, task_cfg_from_run


class IsolationTests(unittest.TestCase):
    def test_legacy_snapshot_is_unchanged(self):
        snapshot = json.loads((paths.PACKAGE_DIR / "legacy_snapshot.json").read_text(encoding="utf-8"))
        for name, expected in snapshot["sha256"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_generated_destinations_and_config_are_isolated(self):
        for generated in (paths.G1_GRASP_URDF, paths.CART_DIR, paths.CART_URDF,
                          paths.PACKED_USD, paths.ASSET_REPORT):
            self.assertTrue(generated.is_relative_to(ROOT / "assets/g1_pushcart_v5"))
        self.assertEqual(paths.CONFIG_DIR, paths.PACKAGE_DIR / "config")
        self.assertEqual(paths.LOG_DIR, ROOT / "logs/rsl_rl/homie_pushcart_v5")
        self.assertNotEqual(paths.G1_URDF, paths.G1_GRASP_URDF)

    def test_registers_both_tasks_without_replacing_legacy(self):
        import gymnasium as gym
        import homie_pushcart
        import homie_pushcart_v5
        homie_pushcart.register_tasks()
        original = gym.spec("Homie-G1-PushCart-Direct-v0").entry_point
        homie_pushcart_v5.register_tasks()
        self.assertEqual(gym.spec("Homie-G1-PushCart-Direct-v0").entry_point, original)
        self.assertEqual(gym.spec(training.TASK_ID).entry_point, "homie_pushcart_v5.direct_env:PushCartDirectEnv")

    def test_export_layout_covers_new_observations(self):
        from homie_pushcart_v5.export import frame_layout
        cfg = PushCartTaskCfg()
        layout = frame_layout(cfg)
        self.assertEqual((num_actor_obs(cfg), num_critic_obs(cfg)), (705, 753))
        self.assertEqual(sum(item["dim"] for item in layout), 141)
        reaction = [item for item in layout if item["name"].startswith("reaction_")]
        self.assertEqual(sum(item["dim"] for item in reaction), 17)
        self.assertEqual(reaction[0]["start"], 124)
        self.assertEqual(reaction[-1]["start"], 140)

    def test_legacy_saved_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "params").mkdir()
            (run / "params/env.yaml").write_text("task:\n  env:\n    actor_history: 5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "V5 checkpoint"):
                task_cfg_from_run(run, verbose=False)

    def test_legacy_checkpoint_without_metadata_has_clear_dimension_error(self):
        import torch
        from homie_pushcart_v5.hl_policy import HighLevelPolicy
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_1.pt"
            torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros(32, 620)}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "620 observations; V5 needs 705"):
                HighLevelPolicy(checkpoint, 705, 753, 18)

    def test_grasp_meshes_still_resolve_after_relocation(self):
        import xml.etree.ElementTree as ET
        from homie_pushcart_v5.assets.g1_grasp_urdf import build_g1_grasp_urdf, verify_g1_grasp_urdf
        from homie_pushcart_v5.pose import load_initial_pose
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested/g1_grasp.urdf"
            build_g1_grasp_urdf(load_initial_pose()["robot"]["finger_pos"], destination=target)
            verify_g1_grasp_urdf(destination=target)
            for mesh in ET.parse(target).getroot().findall(".//mesh"):
                self.assertTrue((target.parent / mesh.get("filename")).resolve().is_file())

    def test_eval_restores_saved_task_and_control_before_explicit_overrides(self):
        from homie_pushcart_v5.env_smoke import make_configs
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "params").mkdir()
            (run / "params/env.yaml").write_text(
                "task:\n  version: 5\n  reaction:\n    compensation_enabled: false\n"
                "  commands:\n    ref_tau_s: 0.37\nhomie:\n  control:\n    decimation: 3\n"
                "decimation: 2\nsim:\n  dt: 0.01\ncurriculum_level: 7\ncheckpoint: frozen_saved.pt\n",
                encoding="utf-8")
            cfg, task = make_configs(8, False, level=3, episode_length_s=9., checkpoint=run / "model_1.pt")
            self.assertFalse(task.reaction.compensation_enabled)
            self.assertEqual(task.commands.ref_tau_s, .37)
            self.assertEqual(cfg.control.decimation, 2)
            self.assertEqual(cfg.sim.dt, .01)
            self.assertEqual(cfg.env.num_envs, 8)
            self.assertEqual(task.curriculum.init_level, 3)
            self.assertEqual(task.env.episode_length_s, 9.)
            self.assertEqual(cfg.pushcart_saved_checkpoint, "frozen_saved.pt")
            self.assertFalse(cfg.noise.add_noise)

    def test_saved_actor_activation_and_shape_used_for_inference_and_export(self):
        import torch
        import yaml
        from homie_pushcart_v5.hl_policy import HighLevelPolicy, import_rsl_rl5
        from homie_pushcart_v5.export import _PlainActor
        from homie_isaaclab.math_utils import class_to_dict
        import_rsl_rl5()
        from tensordict import TensorDict
        from rsl_rl.models import MLPModel
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "params").mkdir()
            (run / "params/env.yaml").write_text(yaml.safe_dump({"task": class_to_dict(PushCartTaskCfg())}), encoding="utf-8")
            config = {"class_name": "MLPModel", "hidden_dims": [32, 16], "activation": "tanh", "obs_normalization": False,
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": .4, "std_type": "scalar"}}
            (run / "params/agent.yaml").write_text(yaml.safe_dump({"actor": config, "obs_groups": {"actor": ["policy"], "critic": ["critic"]}}), encoding="utf-8")
            td = TensorDict({"policy": torch.randn(3, 705), "critic": torch.zeros(3, 753)}, batch_size=[3])
            kwargs = {key: value for key, value in config.items() if key != "class_name"}
            actor = MLPModel(td, {"actor": ["policy"]}, "actor", 18, **kwargs).eval()
            checkpoint = run / "model_1.pt"
            torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
            restored = HighLevelPolicy(checkpoint, 705, 753, 18)
            torch.testing.assert_close(restored(td["policy"]), actor(td))
            plain = _PlainActor(restored.actor).eval()
            torch.testing.assert_close(torch.jit.script(plain)(td["policy"]), actor(td))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pushcart_v5_entry_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "logs/rsl_rl/homie_pushcart_v5/2026-09-24_v5"
        self.run.mkdir(parents=True)
        self.checkpoint = self.run / "model_20.pt"
        self.checkpoint.touch()
        self.root_patch = patch.object(training, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def launch(self, command, args):
        with patch.object(training, "_lab_scripts", return_value=Path("official")), \
                patch.object(training, "_run", return_value=0) as run:
            self.assertEqual(getattr(training, command)(args), 0)
        return run.call_args

    def test_default_scratch_is_v5_and_explicit_overrides_win(self):
        call = self.launch("train", ["--num_envs", "8", "--headless", "env.task.reaction.compensation_enabled=false"])
        args = call.args[1]
        self.assertEqual(args[args.index("--task") + 1], training.TASK_ID)
        self.assertEqual(args[args.index("--experiment_name") + 1], "homie_pushcart_v5")
        self.assertLess(args.index("env.task.reaction.compensation_enabled=true"),
                        args.index("env.task.reaction.compensation_enabled=false"))
        self.assertIn("env.task.rewards.scales.cart_wrench_osc=0.0", args)
        self.assertNotIn("--resume", args)

    def test_resume_does_not_reapply_scratch_profile(self):
        call = self.launch("train", ["--resume"])
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())
        self.assertFalse(any(arg.startswith("env.task.") for arg in call.args[1]))
        self.assertIn("--resume", call.args[1])

    def test_play_selects_only_v5_latest_numbered_checkpoint(self):
        (self.run / "model_9.pt").touch()
        legacy = self.root / "logs/rsl_rl/homie_pushcart/2099_newer"
        legacy.mkdir(parents=True)
        (legacy / "model_999.pt").touch()
        call = self.launch("play", [])
        self.assertEqual(call.kwargs["restore_checkpoint"], self.checkpoint.resolve())

    def test_legacy_log_destination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "separate experiment"):
            self.launch("train", ["--experiment_name", "homie_pushcart"])

    def test_cli_routes_to_v5_training(self):
        spec = importlib.util.spec_from_file_location("pushcart_v5_cli_test", ROOT / "scripts/pushcart_v5.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(training, "train", return_value=0) as train:
            self.assertEqual(module.main(["train", "--quality_profile", "v5", "--num_envs", "8", "--headless"]), 0)
        self.assertEqual(train.call_args.args[0], ["--quality_profile", "v5", "--num_envs", "8", "--headless"])


@unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell is unavailable")
class PowerShellTests(unittest.TestCase):
    def test_shell_routes_to_v5_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="pushcart_v5_shell_") as directory:
            root = Path(directory)
            for filename in ("run_v5_pushcart.ps1", "run_v5_pushcart_train.ps1"):
                shutil.copy2(ROOT / filename, root / filename)
            (root / "scripts").mkdir()
            (root / "scripts/pushcart_v5.py").write_text(
                "import sys,json; print(json.dumps(sys.argv[1:])); sys.exit(7)", encoding="utf-8")
            result = subprocess.run(["powershell.exe", "-NoProfile", "-File", str(root / "run_v5_pushcart_train.ps1"),
                                     "-NumEnvs", "8", "-MaxIterations", "2", "-RunName", "v5 test", "--seed", "7"],
                                    env=dict(os.environ, HOMIE_PYTHON=sys.executable), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(json.loads(result.stdout.strip()), ["train", "--quality_profile", "v5", "--level", "0",
                             "--num_envs", "8", "--max_iterations", "2", "--run_name", "v5 test", "--headless", "--seed", "7"])


if __name__ == "__main__":
    unittest.main()
