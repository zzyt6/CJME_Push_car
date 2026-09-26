"""V5.1 checkpoint metadata remains isolated when V5 hooks coexist in-process."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart_v5 import checkpointing as v5
from homie_pushcart_v51 import checkpointing as v51


def runner_class():
    class RecorderRunner:
        def __init__(self, env):
            self.env = SimpleNamespace(unwrapped=env)

        def save(self, path, infos=None):
            self.saved_infos = infos
            return infos

        def load(self, path, *args, **kwargs):
            return self.saved_infos
    return RecorderRunner


def versioned_env(version):
    env = SimpleNamespace(restored=[])
    setattr(env, f"{version}_curriculum_state_dict", lambda: {"version": version})
    setattr(env, f"{version}_load_curriculum_state_dict",
            lambda state, explicit_level=None: env.restored.append((state, explicit_level)))
    return env


class CheckpointNamespaceTests(unittest.TestCase):
    def test_v5_and_v51_hooks_coexist_in_both_installation_orders(self):
        for modules in ((v5, v51), (v51, v5)):
            with self.subTest(order=[module.__name__ for module in modules]):
                runner_type = runner_class()
                for module in modules:
                    module.install_runner_checkpoint_hooks(runner_type)
                for version, selected, unrelated in (("v51", v51, v5), ("v5", v5, v51)):
                    env = versioned_env(version)
                    runner = runner_type(env)
                    original_infos = {"user": "preserve"}
                    runner.save("unused.pt", original_infos)
                    self.assertEqual(original_infos, {"user": "preserve"})
                    self.assertEqual(runner.saved_infos["user"], "preserve")
                    self.assertEqual(runner.saved_infos[selected.STATE_KEY], {"version": version})
                    self.assertNotIn(unrelated.STATE_KEY, runner.saved_infos)
                    with patch.dict(os.environ, {selected.LEVEL_OVERRIDE_ENV: "7",
                                                 unrelated.LEVEL_OVERRIDE_ENV: "4"}):
                        runner.load("unused.pt")
                    self.assertEqual(env.restored, [({"version": version}, 7)])

    def test_install_is_idempotent_and_unrelated_environment_is_untouched(self):
        runner_type = runner_class()
        v51.install_runner_checkpoint_hooks(runner_type)
        first_save = runner_type.save
        first_load = runner_type.load
        v51.install_runner_checkpoint_hooks(runner_type)
        self.assertIs(runner_type.save, first_save)
        self.assertIs(runner_type.load, first_load)
        runner = runner_type(SimpleNamespace())
        infos = {"existing": 1}
        self.assertIs(runner.save("unused.pt", infos), infos)
        self.assertIs(runner.load("unused.pt"), infos)

    def test_v5_metadata_never_restores_v51_environment(self):
        runner_type = runner_class()
        v51.install_runner_checkpoint_hooks(runner_type)
        env = versioned_env("v51")
        runner = runner_type(env)
        runner.saved_infos = {v5.STATE_KEY: {"version": "v5"}}
        with patch("builtins.print") as printer:
            runner.load("old.pt")
        self.assertEqual(env.restored, [])
        self.assertTrue(any("no V5.1 curriculum state" in str(call) for call in printer.call_args_list))

    def test_wrapper_clears_only_v51_restore_environment_on_new_run(self):
        from homie_pushcart_v51 import training
        values = {v51.LEVEL_OVERRIDE_ENV: "stale", training._RESTORE_RUN_ENV: "stale.pt",
                  v5.LEVEL_OVERRIDE_ENV: "5"}
        with patch.dict(os.environ, values), patch.object(training.subprocess, "run",
                return_value=SimpleNamespace(returncode=0)) as run:
            training._run(Path("official/train.py"), [])
        child = run.call_args.kwargs["env"]
        self.assertNotIn(v51.LEVEL_OVERRIDE_ENV, child)
        self.assertNotIn(training._RESTORE_RUN_ENV, child)
        self.assertEqual(child[v5.LEVEL_OVERRIDE_ENV], "5")


if __name__ == "__main__":
    unittest.main()
