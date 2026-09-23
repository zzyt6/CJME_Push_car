"""V4 motion profile integration through real task methods, without a simulator."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from homie_pushcart.cart_quality import apply_quality_profile
from homie_pushcart.curriculum import level_params
from homie_pushcart.sim_env import PushCartSimEnv
from homie_pushcart.task_config import PushCartTaskCfg, task_cfg_from_run
from homie_pushcart.task_env import PushCartEnv


def fixture(n=3, cfg=None):
    """Use actual buffer allocation and task methods with only scene inputs stubbed."""
    env = PushCartEnv.__new__(PushCartEnv)
    env.task_cfg = cfg if cfg is not None else apply_quality_profile(PushCartTaskCfg(), "v4")
    env.num_envs, env.device, env.dt = n, "cpu", .02
    env.cart_base_kg = 12.
    env.cart_params = {"wheels": {"friction": .7, "wheel_damping": .1, "swivel_damping": .1}}
    env.robot = SimpleNamespace(body_names=["pelvis"])
    env._init_hl_buffers()
    env.level_params = level_params(17, env.task_cfg)
    return env


class V4MotionProfileIntegrationTests(unittest.TestCase):
    def test_v4_constructs_both_motion_features_but_defaults_do_not(self):
        legacy = fixture(cfg=PushCartTaskCfg())
        self.assertIsNone(legacy.command_schedule)
        self.assertIsNone(legacy.idle_correction)
        upgraded = fixture()
        self.assertIsNotNone(upgraded.command_schedule)
        self.assertIsNotNone(upgraded.idle_correction)
        self.assertTrue(upgraded.task_cfg.rewards.force_rate_seed_on_reset)

    def test_legacy_checkpoint_cannot_silently_enable_new_motion_features(self):
        # Simulate a saved V3.2 config with quality rewards but no V4 keys.
        saved = {"task": {"quality": {"enabled": True}, "commands": {"ref_tau_s": .4},
                          "actions": {"idle_gate_speed": .05},
                          "rewards": {"scales": {"cart_lin_ripple": -.1}}}}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "params").mkdir()
            (run / "params" / "env.yaml").write_text(json.dumps(saved), encoding="utf-8")
            cfg = task_cfg_from_run(run / "model_19999.pt", verbose=False)
            self.assertFalse(cfg.commands.sequence_enabled)
            self.assertFalse(cfg.idle_correction.enabled)
            self.assertFalse(cfg.rewards.force_rate_seed_on_reset)
            self.assertEqual(cfg.rewards.scales.cart_lin_ripple, -.1)
            # Restore explicitly saved V4 switches through the same loader.
            saved["task"]["commands"]["sequence_enabled"] = True
            saved["task"]["idle_correction"] = {"enabled": True}
            saved["task"]["rewards"]["force_rate_seed_on_reset"] = True
            (run / "params" / "env.yaml").write_text(json.dumps(saved), encoding="utf-8")
            cfg = task_cfg_from_run(run, verbose=False)
            self.assertTrue(cfg.commands.sequence_enabled)
            self.assertTrue(cfg.idle_correction.enabled)
            self.assertTrue(cfg.rewards.force_rate_seed_on_reset)

    def test_switching_back_to_combined_disables_v4_motion_features(self):
        cfg = apply_quality_profile(apply_quality_profile(PushCartTaskCfg(), "v4"), "combined")
        self.assertFalse(cfg.commands.sequence_enabled)
        self.assertFalse(cfg.idle_correction.enabled)
        self.assertFalse(cfg.rewards.force_rate_seed_on_reset)
        self.assertEqual(cfg.rewards.scales.cart_lin_ripple, -.1)


class V4CommandTaskIntegrationTests(unittest.TestCase):
    def stop_go_env(self, n=3):
        cfg = apply_quality_profile(PushCartTaskCfg(), "v4")
        cfg.commands.stop_go_prob, cfg.commands.turn_reversal_prob = 1., 0.
        cfg.commands.zero_prob = 1.  # The legacy samples are guaranteed to be zero.
        env = fixture(n=n, cfg=cfg)
        env._reset_hl_state(torch.arange(n))
        return env

    def test_real_sampler_overrides_only_requested_rows_and_retains_exact_stop(self):
        env = self.stop_go_env()
        env.cmd_task.fill_(.123)
        env.cmd_timer.fill_(777)
        ids = torch.tensor([0, 2], dtype=torch.int32)
        env._sample_task_command(ids)
        first = env.cmd_task.clone()
        self.assertTrue(torch.all(first[ids, 0] > 0))
        torch.testing.assert_close(first[ids, 1:], torch.zeros(2, 2))
        torch.testing.assert_close(first[1], torch.full((3,), .123))
        self.assertEqual(int(env.cmd_timer[1]), 777)
        self.assertTrue(torch.all((env.cmd_timer[ids] >= 100) & (env.cmd_timer[ids] <= 200)))
        env._sample_task_command(ids)
        torch.testing.assert_close(env.cmd_task[ids], torch.zeros(2, 3), atol=0, rtol=0)
        env._sample_task_command(ids)
        torch.testing.assert_close(env.cmd_task[ids], first[ids])

    def test_fixed_evaluation_command_bypasses_training_schedule(self):
        env = self.stop_go_env()
        env.fixed_command = [.2, 0., -.1]
        env._sample_task_command(torch.arange(3))
        torch.testing.assert_close(env.cmd_task, torch.tensor([[.2, 0., -.1]]).expand(3, -1))
        torch.testing.assert_close(env.cmd_timer, torch.full((3,), 10 ** 9))
        torch.testing.assert_close(env.command_schedule.phase, torch.zeros(3, dtype=torch.long))

    def test_actual_partial_reset_restarts_only_selected_episode(self):
        env = self.stop_go_env()
        env._sample_task_command(torch.arange(3))
        env._sample_task_command(torch.tensor([0, 1]))
        # Its previous episode can have a different mode from the newly sampled one.
        env.command_schedule.mode[1] = env.command_schedule.TURN_REVERSAL
        previous_target = env.command_schedule.targets[0].clone()
        previous_mode = env.command_schedule.mode[[0, 2]].clone()
        env.idle_correction.gate_level.fill_(.2)
        env.idle_gate_sum.fill_(5.)
        env._reset_hl_state(torch.tensor([1], dtype=torch.int32))
        torch.testing.assert_close(env.command_schedule.phase, torch.tensor([2, 0, 1]))
        torch.testing.assert_close(env.command_schedule.targets[0], previous_target)
        torch.testing.assert_close(env.command_schedule.mode[[0, 2]], previous_mode)
        self.assertEqual(int(env.command_schedule.mode[1]), env.command_schedule.STOP_GO)
        torch.testing.assert_close(env.idle_correction.gate_level, torch.tensor([.2, 0., .2]))
        torch.testing.assert_close(env.idle_gate_sum, torch.tensor([5., 0., 5.]))
        env._sample_task_command(torch.tensor([1, 2]))
        self.assertGreater(float(env.cmd_task[1, 0]), 0.)
        torch.testing.assert_close(env.cmd_task[2], torch.zeros(3))

    def test_reset_first_segment_and_later_segments_last_exact_control_ticks(self):
        cfg = apply_quality_profile(PushCartTaskCfg(), "v4")
        cfg.commands.stop_go_prob, cfg.commands.turn_reversal_prob = 1., 0.
        cfg.commands.move_time_s = [.04, .04]  # Two control steps moving.
        cfg.commands.stop_hold_time_s = [.06, .06]  # Three control steps stopped.
        env = fixture(n=1, cfg=cfg)
        env._record_episode_stats = lambda ids: None
        env._randomize_cart = lambda ids: None
        env._log_episode_diagnostics = lambda ids: None
        env.command_buffer = torch.zeros(1, 3)
        env.upper_delta = torch.zeros(1, 15)
        # Let the real task reset allocate/seed its command state. Only the base
        # class's simulator work is skipped.
        with patch.object(PushCartSimEnv, "reset_idx", return_value=None):
            env.reset_idx(torch.tensor([0]))
        env.ref_alpha, env.cmd_alpha = .05, .1
        env.residual_scale = torch.tensor(cfg.actions.cmd_residual_scale)
        limits = torch.tensor(cfg.actions.homie_cmd_limits)
        env.cmd_lo, env.cmd_hi = limits[:, 0], limits[:, 1]
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(base_height_target=.72),
                                  control=SimpleNamespace(action_scale=.25))
        env.commands = torch.zeros(1, 5)
        env.num_one_step_obs, env.obs_buf = 76, torch.zeros(1, 76)
        env.commands_scale = torch.ones(3)
        env.upper_scale = torch.ones(1, 15)
        env.hold_target_q = torch.zeros(1, 15)
        env.upper_lo, env.upper_hi = torch.full((1, 15), -1.), torch.ones(1, 15)

        class BeforePhysics(Exception):
            pass

        def capture_before_physics():
            raise BeforePhysics

        env._apply_robot_push = capture_before_physics
        moving = []
        for _ in range(10):
            # Exercise the actual decrement/resampling/reference/action order;
            # stop immediately before the first simulator operation.
            with self.assertRaises(BeforePhysics):
                env.step(torch.zeros(1, 18))
            moving.append(float(env.cmd_task[0, 0]) > 0.)
        self.assertEqual(moving, [True, True, False, False, False] * 2)


if __name__ == "__main__":
    unittest.main()
