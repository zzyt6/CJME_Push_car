"""V5.1 real observation assembler with mock scene queries, no simulator launch."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "HomieRL", ROOT / "HomieRL/rsl_rl"):
    sys.path.insert(0, str(path))

from homie_pushcart_v51.fixed_grasp import FixedGraspObserver
from homie_pushcart_v51.observation import RecoveryAuthority, frame_layout, WRENCH_SCALE, make_proprioceptive_sample
from homie_pushcart_v51.task_config import PushCartTaskCfg, HL_FRAME, PRIVILEGED_FRAME
from homie_pushcart_v51.task_env import PushCartEnv
from homie_pushcart_v51.sim_env import PushCartSimEnv
from homie_pushcart_v51.pose import homie_default_joint_angles
from homie_pushcart_v51.paths import HOMIE_JOINTS


def mock_environment(n=2):
    cfg = PushCartTaskCfg()
    ob = FixedGraspObserver(n, position_tolerance_m=cfg.interaction.fk_position_tolerance_m,
                           orientation_tolerance_rad=cfg.interaction.fk_orientation_tolerance_rad)
    defaults = homie_default_joint_angles()
    z = lambda *shape: torch.zeros(*shape)
    env = SimpleNamespace(task_cfg=cfg, level_params=SimpleNamespace(obs_noise_frac=0.), num_envs=n, device="cpu")
    env.num_one_step_obs = 76
    env.obs_scales = SimpleNamespace(dof_pos=1.7, dof_vel=.07, ang_vel=.5, lin_vel=2.)
    env.default_dof_pos = torch.tensor([[defaults[j] for j in HOMIE_JOINTS]])
    q = ob.initial_joint_pos.expand(n, -1).clone()
    dq = torch.arange(27).float()[None].expand(n, -1) * .0001
    env.dof_pos, env.dof_vel = q.clone(), dq.clone()
    env.base_ang_vel = torch.tensor([.01, -.02, .03]).expand(n, -1).clone()
    env.projected_gravity = torch.tensor([0., 0., -1.]).expand(n, -1).clone()
    env.obs_buf = z(n, 3 * 76)
    env.obs_buf[:, :-76] = 77.  # Only the newest low-level sample may enter FK.
    f = env.obs_buf[:, -76:]
    f[:, 4:7] = torch.tensor([.01, -.02, .03]) * env.obs_scales.ang_vel
    f[:, 7:10] = torch.tensor([0., 0., -1.])
    f[:, 10:37] = (q - env.default_dof_pos) * env.obs_scales.dof_pos
    f[:, 37:64] = dq * env.obs_scales.dof_vel
    env.fixed_grasp, env.kinematic_valid = ob, torch.zeros(n, dtype=torch.bool)
    env.recovery_authority = RecoveryAuthority(cfg.recovery, .02, n, "cpu")
    env.vel_est = torch.tensor([.1, -.02, .03]).expand(n, -1).clone()
    env.episode_length_buf = torch.ones(n, dtype=torch.long)
    env.hold_target_q = q[:1, 12:] + .01
    env.upper_delta_q = torch.full((n, 15), .02)
    env.cmd_ref = torch.tensor([.2, .01, -.1]).expand(n, -1).clone()
    env.cmd_homie = torch.tensor([.18, .02, -.08]).expand(n, -1).clone()
    env.commands_scale3 = torch.tensor([2., 2., .5])
    env.hl_actions = torch.linspace(-.5, .5, 18)[None].expand(n, -1).clone()
    env.actor_hist = z(n, cfg.env.actor_history, HL_FRAME)
    env.cfg = SimpleNamespace(normalization=SimpleNamespace(clip_observations=100.), noise=SimpleNamespace(add_noise=False))
    env.ts = SimpleNamespace(upper_tau=torch.full((n, 15), 2.), cart_yaw=z(n),
        cart_v=z(n, 2), cart_wz=z(n), force_w_raw=z(n, 2, 3), torque_w_raw=z(n, 2, 3),
        cart_task_wrench=torch.tensor([8., -4., 1.2]).expand(n, -1).clone(), caster_angles=z(n, 4))
    # These actor-forbidden quantities differ independently of sensor data.
    env.ts.bar_b, env.ts.rel_yaw = torch.full((n, 3), 99.), torch.full((n,), 2.)
    env._compute_task_state = lambda: None
    env.cart_pose_truth = z(n, 3)
    quat = z(n, 4)
    quat[:, 3] = 1.
    env.cart_state = lambda: (env.cart_pose_truth, quat, z(n, 3), z(n, 3))
    env.bar_local = torch.tensor([-.49, 0., .917])
    env.cart_applied_wrench, env.applied_robot_wrench = z(n, 6), z(n, 6)
    env._cart_applied_reference = z(n, 3)
    env.dynamics_wrench_valid = torch.ones(n, dtype=torch.bool)
    env.cart_mass_total, env.payload_com = torch.full((n,), 26.9), z(n, 2)
    env.wheel_friction, env.wheel_damping, env.swivel_damping = torch.ones(n), torch.full((n,), .005), torch.full((n,), .05)
    env.base_lin_vel = z(n, 3)
    return env


class V51ObservationTests(unittest.TestCase):
    def run_update(self, env):
        PushCartEnv._update_hl_observations(env)

    def test_full_frame_schema_uses_independent_joint_imu_measurements(self):
        env = mock_environment()
        recorded = {}
        original = env.fixed_grasp.update
        def observe(q, dq, base_v, gyro, gravity, **kwargs):
            recorded.update(q=q.clone(), dq=dq.clone(), base_v=base_v.clone(), gyro=gyro.clone(), gravity=gravity.clone())
            return original(q, dq, base_v, gyro, gravity, **kwargs)
        env.fixed_grasp.update = observe
        self.run_update(env)
        self.assertEqual(env.hl_obs.shape, (2, 20 * 124))
        self.assertEqual(env.hl_critic_obs.shape, (2, HL_FRAME + PRIVILEGED_FRAME))
        low, fk = env.obs_buf[:, -76:], env.fixed_grasp
        torch.testing.assert_close(recorded["q"], fk.initial_joint_pos.expand(2, -1))
        torch.testing.assert_close(recorded["dq"], low[:, 37:64] / env.obs_scales.dof_vel)
        torch.testing.assert_close(recorded["gyro"], low[:, 4:7] / env.obs_scales.ang_vel)
        torch.testing.assert_close(recorded["gravity"], low[:, 7:10])
        torch.testing.assert_close(recorded["base_v"], env.vel_est)
        expected = {
            "joint_position": low[:, 10:37], "joint_velocity": low[:, 37:64],
            "upper_torque": env.ts.upper_tau * env.task_cfg.observations.torque_scale,
            "imu_gyro": low[:, 4:7], "projected_gravity": low[:, 7:10],
            "him_velocity": env.vel_est * env.obs_scales.lin_vel,
            "handle_position": fk.handle_pos_b, "cart_relative_yaw": fk.rel_yaw[:, None],
            "cart_twist_estimate": fk.cart_twist_heading * env.commands_scale3,
            "task_reference": env.cmd_ref * env.commands_scale3,
            "homie_command_applied": env.cmd_homie * env.commands_scale3,
            "upper_target_applied": (env.hold_target_q + env.upper_delta_q - env.default_dof_pos[:, 12:]) * env.obs_scales.dof_pos,
            "previous_high_level_action": env.hl_actions,
        }
        for field in frame_layout():
            with self.subTest(field=field["name"]):
                start = field["start"]
                torch.testing.assert_close(env.actor_hist[:, -1, start:start+field["dim"]], expected[field["name"]])
        torch.testing.assert_close(env.hl_critic_obs[:, :HL_FRAME], env.actor_hist[:, -1])

    def test_actor_and_recovery_do_not_depend_on_privileged_truth_or_label(self):
        a, b = mock_environment(), mock_environment()
        b.ts.bar_b.fill_(-33.)
        b.ts.rel_yaw.fill_(-2.)
        b.ts.cart_yaw.fill_(.7)
        b.ts.cart_v.fill_(3.)
        b.ts.cart_wz.fill_(2.)
        b.ts.force_w_raw.fill_(30.)
        b.ts.torque_w_raw.fill_(5.)
        b.ts.cart_task_wrench.fill_(60.)
        b.ts.caster_angles.fill_(.5)
        b.cart_pose_truth.fill_(1.)
        b.cart_applied_wrench.fill_(2.)
        b.applied_robot_wrench.fill_(3.)
        b.cart_mass_total.fill_(50.)
        b.payload_com.fill_(.07)
        b.base_lin_vel.fill_(5.)
        b.dynamics_wrench_valid[0] = False
        for _ in range(7):
            self.run_update(a)
            self.run_update(b)
        torch.testing.assert_close(a.hl_obs, b.hl_obs)
        torch.testing.assert_close(a.fixed_grasp.cart_twist_heading, b.fixed_grasp.cart_twist_heading)
        torch.testing.assert_close(a.recovery_authority.gate, b.recovery_authority.gate)
        self.assertFalse(torch.equal(a.hl_critic_obs, b.hl_critic_obs))
        self.assertFalse(torch.equal(a.interaction_target, b.interaction_target))

    def test_fresh_reset_clears_only_own_history_and_masks_stale_label_torque(self):
        env = mock_environment()
        env.actor_hist.fill_(.7)
        env.episode_length_buf[0] = 0
        env.ts.upper_tau[0] = float("nan")
        env.ts.cart_task_wrench[0] = float("nan")
        self.run_update(env)
        self.assertEqual(env.actor_hist[0, :-1].count_nonzero(), 0)
        torch.testing.assert_close(env.actor_hist[1, :-1], torch.full_like(env.actor_hist[1, :-1], .7))
        self.assertEqual(env.actor_hist[0, -1, 54:69].count_nonzero(), 0)
        self.assertEqual(env.interaction_target[0].count_nonzero(), 0)
        torch.testing.assert_close(env.interaction_target[1, :3], env.ts.cart_task_wrench[1] / torch.tensor(WRENCH_SCALE))
        self.assertEqual(env.recovery_authority.gate[0].item(), 0.)

    def test_real_reset_keeps_pre_action_targets_until_ppo_storage_copies_them(self):
        from homie_pushcart_v51.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from homie_pushcart_v51.interaction_policy import InteractionActor
        from homie_pushcart_v51.interaction_ppo import InteractionPPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        class ResetEnvironment(SimpleNamespace):
            def __getattr__(self, name):
                # Extra episode accumulators only need indexed clearing. All
                # observation-bearing buffers come from mock_environment.
                value = torch.ones(self.num_envs)
                setattr(self, name, value)
                return value

        env = ResetEnvironment(**mock_environment().__dict__)
        env.dt = .02
        for name in ("cart_pose_delay", "hl_delay", "cart_push_timer"):
            setattr(env, name, torch.zeros(env.num_envs, dtype=torch.long))
        env.quality_sums, env.quality_peaks = {}, {}
        env.cart_quality = env.idle_correction = env.command_schedule = None
        self.run_update(env)
        before = TensorDict({"policy": env.hl_obs, "critic": env.hl_critic_obs,
                             "interaction_target": env.interaction_target}, batch_size=[env.num_envs])
        expected = before.clone()
        groups = {"actor": ["policy"], "critic": ["critic"]}
        actor = InteractionActor(before, groups, "actor", 18, hidden_dims=[16],
                                 encoder_hidden_dims=[16], interaction_hidden_dims=[8],
                                 distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution"})
        critic = MLPModel(before, groups, "critic", 1, hidden_dims=[16])
        storage = RolloutStorage("rl", env.num_envs, 1, before, [18], "cpu")
        algorithm = InteractionPPO(actor, critic, storage)
        with torch.inference_mode():
            algorithm.act(before)
            reset_ids = torch.tensor([0])
            env.episode_length_buf[reset_ids] = 0
            PushCartEnv._reset_hl_state(env, reset_ids)
            torch.testing.assert_close(before["interaction_target"], expected["interaction_target"])
            self.assertEqual(env.interaction_target[0].count_nonzero(), 0)
            self.assertEqual(env.interaction_target[1, 3].item(), 1.)
            self.run_update(env)
            after = TensorDict({"policy": env.hl_obs, "critic": env.hl_critic_obs,
                                "interaction_target": env.interaction_target}, batch_size=[env.num_envs])
            algorithm.process_env_step(after, torch.zeros(env.num_envs), torch.tensor([1, 0]), {})
        for key in expected.keys():
            torch.testing.assert_close(before[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(storage.observations[0][key], expected[key], rtol=0, atol=0)
        self.assertEqual(after["interaction_target"][0].count_nonzero(), 0)
        self.assertEqual(after["interaction_target"][1, 3].item(), 1.)

    def test_bad_fk_zeros_derived_actor_fields_and_recovery_authority(self):
        env = mock_environment()
        env.dof_pos[0, 20] += .4
        self.run_update(env)
        self.assertEqual(env.kinematic_valid.tolist(), [False, True])
        self.assertEqual(env.actor_hist[0, -1, 78:85].count_nonzero(), 0)
        self.assertGreater(env.actor_hist[1, -1, 78:85].count_nonzero(), 0)
        self.assertEqual(env.recovery_authority.gate[0].item(), 0.)

    def test_homie_augmented_proprioception_cannot_directly_enter_fk_or_actor(self):
        a, b = mock_environment(), mock_environment()
        b.obs_buf.fill_(-50.)
        self.run_update(a)
        self.run_update(b)
        torch.testing.assert_close(a.hl_obs, b.hl_obs)
        torch.testing.assert_close(a.fixed_grasp.cart_twist_heading, b.fixed_grasp.cart_twist_heading)

    def test_actor_and_fk_share_exact_same_noisy_encoder_sample(self):
        env = mock_environment()
        env.cfg.noise.add_noise = True
        seed = 293
        torch.manual_seed(seed)
        q, dq, gyro, gravity = make_proprioceptive_sample(env.dof_pos, env.dof_vel, env.base_ang_vel,
            env.projected_gravity, env.task_cfg.observations, noise_enabled=True)
        original = env.fixed_grasp.update
        observed = {}
        def capture(*args, **kwargs):
            observed["values"] = args
            return original(*args, **kwargs)
        env.fixed_grasp.update = capture
        torch.manual_seed(seed)
        self.run_update(env)
        for actual, expected in zip(observed["values"], (q, dq, env.vel_est, gyro, gravity)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(env.actor_hist[:, -1, :27], (q-env.default_dof_pos)*env.obs_scales.dof_pos)
        torch.testing.assert_close(env.actor_hist[:, -1, 27:54], dq*env.obs_scales.dof_vel)
        torch.testing.assert_close(env.actor_hist[:, -1, 69:72], gyro*env.obs_scales.ang_vel)
        torch.testing.assert_close(env.actor_hist[:, -1, 72:75], gravity)

    def test_step_uses_applied_delayed_action_and_post_physics_him_measurement(self):
        env = PushCartEnv.__new__(PushCartEnv)
        env.__dict__.update(mock_environment().__dict__)
        n, a = env.num_envs, 18
        env.num_hl_actions, env.dt = a, .02
        z = lambda *shape: torch.zeros(*shape)
        for name in ("hl_raw", "hl_raw_prev", "hl_raw_prev2", "hl_action_raw_prev",
                     "hl_actions_prev", "hl_actions_prev2"):
            setattr(env, name, z(n, a))
        env.hl_action_raw_prev[1] = .3
        env.hl_delay = torch.tensor([0, 1])
        env.cmd_timer = torch.full((n,), 100, dtype=torch.long)
        env.cmd_task = env.cmd_ref.clone()
        env.ref_alpha, env.cmd_alpha = 1., 1.
        env.idle_gate_applied, env.idle_gate_sum = z(n), z(n)
        env.residual_scale = torch.tensor(env.task_cfg.actions.cmd_residual_scale)
        limits = torch.tensor(env.task_cfg.actions.homie_cmd_limits)
        env.cmd_lo, env.cmd_hi = limits[:, 0], limits[:, 1]
        env.cmd_homie_prev, env.command_buffer, env.commands = z(n, 3), z(n, 3), z(n, 5)
        env.commands_scale = env.commands_scale3
        env.cfg.rewards = SimpleNamespace(base_height_target=.76)
        env.cfg.control = SimpleNamespace(action_scale=.25)
        env.upper_delta_q.zero_()
        env.upper_delta = z(n, 15)
        env.upper_scale = torch.full((1, 15), .4)
        env.upper_lo, env.upper_hi = env.hold_target_q - .05, env.hold_target_q + .05
        env._apply_robot_push, env._apply_cart_disturbances = lambda: None, lambda: None
        events = []
        env.policy = SimpleNamespace(
            act_and_estimate=lambda obs: (events.append("action") or z(n, 12), torch.full((n, 3), -9.)),
            estimate_velocity=lambda obs: events.append("post_estimate") or torch.tensor([[.62, .04, 0.]]).expand(n, -1))
        self.run_update(env)
        env.kinematic_valid[1] = False
        before_reference = env.fixed_grasp.reference_base(env.cmd_ref).clone()
        def advance(self, legs):
            events.append("physics")
            torch.testing.assert_close(self.upper_delta_q[0], torch.full((15,), .05), atol=1e-7, rtol=1e-6)
            self.dof_pos[:, 12] += .01
            self.episode_length_buf += 1
            self.rew_buf, self.reset_buf, self.time_out_buf, self.extras = z(n), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool), {}
        with patch.object(PushCartSimEnv, "step", advance):
            result = env.step(torch.ones(n, a))
        self.assertEqual(events, ["action", "physics", "post_estimate"])
        torch.testing.assert_close(env.hl_actions[0], torch.ones(a))
        torch.testing.assert_close(env.hl_actions[1], torch.full((a,), .3))
        torch.testing.assert_close(env.cmd_homie[0], (before_reference[0] + env.residual_scale).clamp(env.cmd_lo, env.cmd_hi))
        torch.testing.assert_close(env.cmd_homie[1], torch.zeros(3))
        torch.testing.assert_close(env.vel_est, torch.tensor([[.31, .02, 0.]]).expand(n, -1))
        torch.testing.assert_close(env.actor_hist[:, -1, 75:78], env.vel_est * env.obs_scales.lin_vel)
        torch.testing.assert_close(env.actor_hist[:, -1, 106:124], env.hl_actions)
        self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
