"""V4 exceptional-sample boundaries and per-environment reset regression checks.

The actual task methods run on CPU fixtures; no simulator or PPO rollout is launched.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from homie_isaaclab.task_core import HomieTaskCore
from homie_pushcart.cart_quality import CartQualityTracker, apply_quality_profile
from homie_pushcart.task_config import HL_FRAME, PushCartTaskCfg
from homie_pushcart.task_env import PushCartEnv


class V4QualityValidityTests(unittest.TestCase):
    def test_every_wrench_component_invalidates_all_quality_costs(self):
        for field in ("force", "moment"):
            for axis in range(3):
                with self.subTest(field=field, axis=axis):
                    cfg = PushCartTaskCfg().quality
                    cfg.warmup_s = 0.
                    q = CartQualityTracker(cfg, .02, 2)
                    zero = torch.zeros(2, 3)
                    yaw = torch.zeros(2)
                    q.update(zero, zero + .3, yaw, zero, zero)
                    previous = [v.clone() for v in (q.fast, q.slow, q.response_error_w)]
                    force, moment = zero.clone(), zero.clone()
                    (force if field == "force" else moment)[0, axis] = float("nan")
                    q.update(zero, zero + .3, yaw, force, moment)
                    self.assertEqual(q.ready.tolist(), [False, True])
                    self.assertEqual(q.response_ready.tolist(), [False, True])
                    self.assertEqual(q.steps.tolist(), [1, 2])
                    for actual, before in zip((q.fast, q.slow, q.response_error_w), previous):
                        torch.testing.assert_close(actual[0], before[0])
                    for cost in (q.ripple_cost, q.lin_ripple_cost, q.yaw_ripple_cost,
                                 q.response_cost, q.wrench_cost):
                        self.assertEqual(float(cost[0]), 0.)
                    self.assertGreater(float(q.response_cost[1]), 0.)
                    q.update(zero, zero + .3, yaw, zero, zero)
                    self.assertEqual(q.response_ready.tolist(), [True, True])
                    self.assertTrue(torch.isfinite(q.response_cost).all())


class V4ForceRateTests(unittest.TestCase):
    def fixture(self, n=2):
        return SimpleNamespace(
            ts=SimpleNamespace(force_c=torch.full((n, 2, 3), 30.), budget=torch.full((n,), 100.)),
            force_c_prev=torch.zeros(n, 2, 3), force_c_prev_valid=torch.zeros(n, dtype=torch.bool),
            task_cfg=apply_quality_profile(PushCartTaskCfg(), "v4"),
        )

    def test_partial_reset_reseeds_only_reset_row(self):
        env = self.fixture()
        torch.testing.assert_close(PushCartEnv._reward_force_rate(env), torch.zeros(2))
        # Give _reset_hl_state the normal independent buffers; it must not reset row 1.
        for name in ("hl_actions", "hl_actions_prev", "hl_actions_prev2", "hl_action_raw_prev",
                     "hl_raw", "hl_raw_prev", "hl_raw_prev2", "upper_delta_q", "vel_est",
                     "actor_hist", "cart_pose_hist", "track_err2", "track_steps", "track_vr",
                     "track_rr", "turn_dq_sum", "turn_steps", "rel_yaw_abs_sum", "sat_steps",
                     "overload_counter", "wheel_lift_counter", "overload_dir", "stall_timer",
                     "push_timer", "push_force", "quality_steps", "response_steps", "response_error2_sum",
                     "cart_pose_delay", "hl_delay", "cart_push_timer", "idle_gate_applied", "idle_gate_sum"):
            setattr(env, name, torch.ones(2))
        env.level_params = SimpleNamespace(obs_noise_frac=0.)
        for name in ("cart_pose_delay", "hl_delay", "cart_push_timer"):
            setattr(env, name, torch.ones(2, dtype=torch.long))
        env.device, env.dt = "cpu", .02
        env.cart_quality = None
        env.idle_correction = None
        env.command_schedule = None
        env.quality_sums, env.quality_peaks = {}, {}
        PushCartEnv._reset_hl_state(env, torch.tensor([0]))
        self.assertEqual(env.force_c_prev_valid.tolist(), [False, True])
        env.ts.force_c[:] = 40.
        rate = PushCartEnv._reward_force_rate(env)
        self.assertEqual(float(rate[0]), 0.)
        self.assertAlmostEqual(float(rate[1]), .06)
        self.assertEqual(env.force_c_prev_valid.tolist(), [True, True])

    def test_invalid_first_sample_does_not_seed_or_silence_reward_failure(self):
        env = self.fixture(1)
        env.ts.force_c[0, 0, 0] = float("nan")
        env.rew_buf = torch.zeros(1)
        env.reward_names = ["force_rate"]
        env.reward_functions = [lambda: PushCartEnv._reward_force_rate(env)]
        env.reward_scales = {"force_rate": -.2 * .02}
        env.episode_sums = {"force_rate": torch.zeros(1)}
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(only_positive_rewards=False))
        with self.assertRaisesRegex(FloatingPointError, "Non-finite reward: force_rate"):
            HomieTaskCore.compute_reward(env)
        self.assertFalse(bool(env.force_c_prev_valid[0]))
        self.assertTrue(torch.isfinite(env.force_c_prev).all())
        self.assertEqual(float(env.episode_sums["force_rate"][0]), 0.)
        env.ts.force_c[:] = 30.
        self.assertEqual(float(PushCartEnv._reward_force_rate(env)[0]), 0.)


def state_fixture():
    n = 2
    z = lambda *shape: torch.zeros(*shape)
    quat = torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1)
    wrench_force, wrench_torque = z(n, 2, 3), z(n, 2, 3)
    cart_data = SimpleNamespace(
        body_link_pose_w=SimpleNamespace(torch=z(n, 6, 7)),
        joint_pos=SimpleNamespace(torch=z(n, 4)),
    )
    env = SimpleNamespace(
        task_cfg=apply_quality_profile(PushCartTaskCfg(), "v4"), ts=SimpleNamespace(), num_envs=n,
        cart_state=lambda: (z(n, 3), quat, z(n, 3), z(n, 3)),
        gravity_vec=torch.tensor([[0., 0., -1.]]).repeat(n, 1), bar_local=z(3),
        root_states=z(n, 13), base_quat=quat, yaw=z(n, 1),
        grasp_wrench=lambda: (wrench_force, wrench_torque, wrench_force),
        episode_length_buf=torch.tensor([0, 1]), cart_quality=None, cart_mass_total=torch.full((n,), 25.),
        cart=SimpleNamespace(data=cart_data), wheel_body_ids=torch.arange(4), wheel_radius=.1,
        swivel_joint_ids=torch.arange(4), torques=z(n, 27), contact_forces=z(n, 2, 3),
        feet_indices=torch.arange(2), near_caster_body_ids=torch.tensor([4, 5]), feet_pos=z(n, 2, 3),
    )
    return env, wrench_force, wrench_torque


class V4ResetObservationTests(unittest.TestCase):
    def test_stale_nonfinite_wrench_is_masked_only_for_reset_rows(self):
        env, force, torque = state_fixture()
        force[0] = float("nan")
        torque[0] = float("inf")
        force[1, :, 0] = 30.
        torque[1, :, 0] = 2.
        PushCartEnv._compute_task_state(env)
        torch.testing.assert_close(env.ts.force_w[0], torch.zeros(2, 3))
        torch.testing.assert_close(env.ts.torque_w[0], torch.zeros(2, 3))
        torch.testing.assert_close(env.ts.force_w[1], force[1])
        torch.testing.assert_close(env.ts.torque_w[1], torque[1])

    def observation_fixture(self):
        env, force, torque = state_fixture()
        n = env.num_envs
        z = lambda *shape: torch.zeros(*shape)
        env._compute_task_state = lambda: PushCartEnv._compute_task_state(env)
        env.device = "cpu"
        env.level_params = SimpleNamespace(obs_noise_frac=0., arm_scale=.4)
        env.num_one_step_obs = 76
        env.obs_buf = z(n, 380)
        env.cmd_ref = z(n, 3)
        env.commands_scale3 = torch.tensor([2., 2., .5])
        env.vel_est = z(n, 3)
        env.obs_scales = SimpleNamespace(lin_vel=2., ang_vel=.25)
        env.cmd_homie = z(n, 3)
        env.base_ang_vel = z(n, 3)
        env.cart_pose_hist = z(n, 3, 4)
        env.cart_pose_delay = torch.zeros(n, dtype=torch.long)
        env.hl_actions = z(n, 18)
        env.actor_hist = z(n, 5, HL_FRAME)
        env.cfg = SimpleNamespace(normalization=SimpleNamespace(clip_observations=100.))
        env.payload_com = z(n, 2)
        env.wheel_friction = torch.ones(n)
        env.wheel_damping = torch.full((n,), .01)
        env.swivel_damping = torch.full((n,), .1)
        env.level_tensor, env.level_max = torch.tensor(6.), 17
        env.base_lin_vel = z(n, 3)
        return env, force, torque

    def test_finite_observation_shapes_and_reset_nan_mask(self):
        env, force, torque = self.observation_fixture()
        force[0] = float("nan")
        torque[0] = float("nan")
        PushCartEnv._update_hl_observations(env)
        self.assertEqual(tuple(env.hl_obs.shape), (2, 620))
        self.assertEqual(tuple(env.hl_critic_obs.shape), (2, 652))
        self.assertTrue(torch.isfinite(env.hl_critic_obs).all())

    def test_active_torque_only_nan_or_inf_cannot_enter_critic_rollout(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                env, _, torque = self.observation_fixture()
                torque[1, 0, 0] = bad
                with self.assertRaisesRegex(FloatingPointError, r"observation in environments \[1\]"):
                    PushCartEnv._update_hl_observations(env)

    def test_nonfinite_prior_actor_history_cannot_enter_rollout(self):
        env, _, _ = self.observation_fixture()
        env.actor_hist[1, 3, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, r"observation in environments \[1\]"):
            PushCartEnv._update_hl_observations(env)


if __name__ == "__main__":
    unittest.main()
