"""Real checkpoint serialization for live V5 curriculum progress, without simulation."""
from __future__ import annotations

import copy
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart_v5.checkpointing import (STATE_KEY, LEVEL_OVERRIDE_ENV, WINDOW_TENSORS,
    curriculum_state, restore_curriculum, install_runner_checkpoint_hooks)
from homie_pushcart_v5.curriculum import CurriculumTracker
from homie_pushcart_v5.task_config import PushCartTaskCfg


def environment():
    cfg = PushCartTaskCfg()
    env = SimpleNamespace(task_cfg=cfg, curriculum=CurriculumTracker(cfg, 12), dt=.02,
                          common_step_counter=200, _next_curriculum_eval=500, episodes_finished=912,
                          term_counts=torch.tensor([3., 4., 5.]), episode_length_buf=torch.ones(3))
    values = {"_track_sum": [.5, .7, .9], "_fail_sum": 2., "_count": 6.,
              "_gain_vr": [.4, -.1, .2], "_gain_rr": [.6, .3, .4], "_gain_steps": 120.}
    for key, value in values.items():
        setattr(env, key, torch.tensor(value))
    env.reset_calls = 0
    env._apply_level = lambda params: setattr(env, "level_params", params)
    def reset():
        env.reset_calls += 1
        for name in WINDOW_TENSORS:
            getattr(env, name).zero_()
    env.reset = reset
    env.curriculum.record([[.1, .2, .3], [.4, .5, .6]], [0, 1])
    env.curriculum._failed_windows = 2
    env.curriculum.changes = 8
    return env


class CurriculumStateTests(unittest.TestCase):
    def test_full_tracker_roundtrip_preserves_partial_window_and_hysteresis(self):
        before = environment().curriculum
        state = before.state_dict()
        after = CurriculumTracker(PushCartTaskCfg(), 0)
        after.load_state_dict(state)
        self.assertEqual(after.level, 12)
        self.assertEqual(after._failed_windows, 2)
        self.assertEqual(after.changes, 8)
        self.assertEqual(after._count, 2)
        for actual, expected in zip(after._tracking_sum, [.5, .7, .9]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(after._fail_sum, 1.)
        # Both trackers must make the same next promotion/demotion decision.
        self.assertEqual(before.record_window([0., 0., 0.], 1.), after.record_window([0., 0., 0.], 1.))
        self.assertEqual(after.level, 11)

    def test_completed_gpu_window_and_remaining_timer_restore_after_reset(self):
        source = environment()
        state = curriculum_state(source)
        target = environment()
        target.common_step_counter = 0
        target.curriculum.level = 0
        result = restore_curriculum(target, state)
        self.assertTrue(result["completed_window_restored"])
        self.assertEqual(target.reset_calls, 1)
        self.assertEqual(target.curriculum.level, 12)
        self.assertEqual(target.level_params.level, 12)
        self.assertEqual(target._next_curriculum_eval, 300)
        self.assertEqual(target.episodes_finished, 912)
        self.assertTrue((target.episode_length_buf == 0).all())
        for name in WINDOW_TENSORS:
            torch.testing.assert_close(getattr(source, name), getattr(target, name))

    def test_explicit_level_discards_prior_window(self):
        target = environment()
        result = restore_curriculum(target, curriculum_state(environment()), explicit_level=4)
        self.assertEqual(target.curriculum.level, 4)
        self.assertFalse(result["completed_window_restored"])
        self.assertEqual(target.curriculum._failed_windows, 0)
        for name in WINDOW_TENSORS:
            self.assertTrue((getattr(target, name) == 0).all())
        self.assertEqual(target.episodes_finished, 912)

    def test_schedule_change_keeps_live_level_but_restarts_statistics(self):
        state = curriculum_state(environment())
        target = environment()
        target.task_cfg.curriculum.window_episodes = 900
        result = restore_curriculum(target, state)
        self.assertTrue(result["schedule_changed"])
        self.assertEqual(target.curriculum.level, 12)
        self.assertEqual(target.curriculum.changes, 8)
        self.assertEqual(target._count.item(), 0.)

    def test_invalid_state_does_not_mutate_environment(self):
        state = curriculum_state(environment())
        state["completed_window"]["_gain_rr"][0] = float("nan")
        target = environment()
        with self.assertRaisesRegex(ValueError, "_gain_rr"):
            restore_curriculum(target, state)
        self.assertEqual(target.reset_calls, 0)
        self.assertEqual(target._count.item(), 6.)

    def test_snapshot_does_not_alias_running_tensors(self):
        env = environment()
        state = curriculum_state(env)
        env._track_sum.zero_()
        self.assertGreater(state["completed_window"]["_track_sum"].sum(), 0)


class RunnerHookTests(unittest.TestCase):
    def make_runner(self, env):
        # Exercise the actual installed OnPolicyRunner save/load methods, its
        # standard torch checkpoint file and public infos payload.
        from homie_pushcart_v5.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from rsl_rl.runners import OnPolicyRunner
        class TestRunner(OnPolicyRunner):
            pass
        install_runner_checkpoint_hooks(TestRunner)
        runner = object.__new__(TestRunner)
        runner.current_learning_iteration = 17
        runner.alg = SimpleNamespace(save=lambda: {"actor_state_dict": {"test": torch.tensor([2.])}},
                                     load=lambda loaded, cfg, strict: True)
        runner.logger = SimpleNamespace(save_model=lambda *args: None)
        wrapper = SimpleNamespace(v5_curriculum_state_dict=lambda: curriculum_state(env),
                                  v5_load_curriculum_state_dict=lambda state, explicit_level=None:
                                      restore_curriculum(env, state, explicit_level))
        runner.env = SimpleNamespace(unwrapped=wrapper)
        return runner

    def test_actual_torch_checkpoint_embeds_and_restores_state_and_user_infos(self):
        with tempfile.TemporaryDirectory() as directory:
            source = environment()
            saver = self.make_runner(source)
            checkpoint = Path(directory) / "model_17.pt"
            saver.save(str(checkpoint), {"user_tag": "preserve"})
            loaded = torch.load(checkpoint, weights_only=False)
            self.assertEqual(loaded["infos"][STATE_KEY]["tracker"]["level"], 12)
            target = environment()
            target.curriculum.level = 0
            target._count.zero_()
            loader = self.make_runner(target)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(LEVEL_OVERRIDE_ENV, None)
                infos = loader.load(str(checkpoint), map_location="cpu")
            self.assertEqual(infos["user_tag"], "preserve")
            self.assertEqual(loader.current_learning_iteration, 17)
            self.assertEqual(target.curriculum.level, 12)
            self.assertEqual(target._count.item(), 6.)

    def test_actual_load_honors_explicit_level_and_old_checkpoint_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            saver = self.make_runner(environment())
            saver.save(path)
            target = environment()
            loader = self.make_runner(target)
            with patch.dict(os.environ, {LEVEL_OVERRIDE_ENV: "5"}):
                loader.load(path, map_location="cpu")
            self.assertEqual(target.curriculum.level, 5)
            self.assertEqual(target._count.item(), 0.)
            torch.save({"iter": 9, "infos": None}, path)
            with patch("builtins.print") as printer:
                loader.load(path, map_location="cpu")
            self.assertTrue(any("predates live curriculum" in str(c) for c in printer.call_args_list))
            self.assertEqual(target.curriculum.level, 5)

    def test_hydra_level_override_is_parsed_and_stale_env_is_cleared(self):
        from homie_pushcart_v5 import training
        self.assertEqual(training._explicit_level(["env.task.curriculum.init_level=7"]), 7)
        self.assertEqual(training._explicit_level(["env.curriculum_level=8", "env.task.curriculum.init_level=3"]), 3)
        with patch.dict(os.environ, {LEVEL_OVERRIDE_ENV: "stale"}), \
                patch.object(training.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            training._run(Path("official/train.py"), [])
        self.assertNotIn(LEVEL_OVERRIDE_ENV, run.call_args.kwargs["env"])

    def test_non_v5_runner_save_and_load_are_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(environment())
            runner.env = SimpleNamespace(unwrapped=SimpleNamespace())
            path = str(Path(directory) / "legacy_model.pt")
            original_infos = {"existing": "value"}
            runner.save(path, original_infos)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved["infos"], original_infos)
            self.assertEqual(runner.load(path, map_location="cpu"), original_infos)

    def test_wrapper_propagates_explicit_hydra_level_to_child(self):
        from homie_pushcart_v5 import training
        with patch.object(training.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            training._run(Path("official/train.py"), ["env.task.curriculum.init_level=6"],
                          restore_checkpoint=Path("model_17.pt"))
        self.assertEqual(run.call_args.kwargs["env"][LEVEL_OVERRIDE_ENV], "6")


if __name__ == "__main__":
    unittest.main()
