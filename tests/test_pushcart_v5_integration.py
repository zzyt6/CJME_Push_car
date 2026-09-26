"""V5 observation parity and actual MuJoCo force/mass integration contracts."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "HomieRL", ROOT / "HomieRL/rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from homie_pushcart_v5.task_config import PushCartTaskCfg, HL_FRAME, num_actor_obs, num_critic_obs
from homie_pushcart_v5.task_env import PushCartEnv
from homie_pushcart_v5.sim_env import PushCartSimEnv
from homie_pushcart_v5.mujoco_env import MujocoHomieEnv
from homie_pushcart_v5.mujoco_hl import MujocoHighLevelRunner
from homie_pushcart_v5.reaction_observation import ReactionObservation
from homie_pushcart_v5.reaction import ReactionController


def config():
    cfg = PushCartTaskCfg()
    for name in ("sensor_force_noise_std_n", "sensor_moment_noise_std_nm",
                 "sensor_force_bias_std_n", "sensor_moment_bias_std_nm"):
        setattr(cfg.reaction, name, 0.)
    return cfg


def synthetic_runners():
    """Exercise actual observation assemblers with shared sensor samples.

    Only the scene queries are replaced: both real runtime adapters, histories,
    frame transforms, normalizations and critic concatenation still execute.
    """
    cfg = config()
    lp = SimpleNamespace(obs_noise_frac=0., arm_scale=.4)
    force, moment = np.array([30., -10., 5.]), np.array([2., 0., -4.])
    yaw = .4
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    fw, mw = rotation @ force, rotation @ moment
    frame = np.zeros((1, 76), dtype=np.float32)
    frame[:, 6] = .05  # gyro wz with scale=.5
    frame[:, 9] = -1.
    robot = dict(mass=torch.tensor([35.]), linear_momentum=torch.zeros(1, 3),
                 angular_momentum=torch.zeros(1, 3), velocity=torch.zeros(1, 3))
    ground = dict(feet=torch.zeros(1, 2, 6), valid=torch.ones(1, dtype=torch.bool))

    isaac = PushCartEnv.__new__(PushCartEnv)
    isaac.task_cfg, isaac.level_params = cfg, lp
    isaac.num_envs, isaac.device, isaac.dt = 1, "cpu", .02
    isaac.num_one_step_obs, isaac.obs_buf = 76, torch.tensor(frame)
    isaac.cmd_ref = torch.tensor([[.2, .01, .1]])
    isaac.commands_scale3 = torch.tensor([2., 2., .5])
    isaac.cmd_homie = torch.tensor([[.21, .02, .12]])
    isaac.obs_scales = SimpleNamespace(lin_vel=2., ang_vel=.5)
    isaac.cart_mass_total, isaac.cart_mass_estimate = torch.tensor([26.9]), torch.tensor([26.9])
    isaac.robot_mass_estimate = torch.tensor([35.])
    isaac.vel_est = torch.tensor([[.15, .02, 0.]])
    isaac.base_ang_vel = torch.tensor([[0., 0., .1]])
    isaac.base_lin_vel = torch.zeros(1, 3)
    isaac.episode_length_buf = torch.ones(1, dtype=torch.long)
    isaac.cart_pose_hist = torch.zeros(1, 3, 4)
    isaac.cart_pose_delay = torch.zeros(1, dtype=torch.long)
    isaac.hl_actions = torch.zeros(1, 18)
    isaac.actor_hist = torch.zeros(1, cfg.env.actor_history, HL_FRAME)
    isaac.cfg = SimpleNamespace(normalization=SimpleNamespace(clip_observations=100.))
    isaac.robot = object()
    isaac.centroidal_state = lambda _robot: robot
    isaac.ground_contact_wrenches = lambda: ground
    isaac.dynamics_wrench_valid = torch.ones(1, dtype=torch.bool)
    isaac.payload_com = torch.zeros(1, 2)
    isaac.wheel_friction, isaac.wheel_damping, isaac.swivel_damping = (torch.ones(1) for _ in range(3))
    isaac.level_tensor, isaac.level_max = torch.tensor(0.), 17
    isaac.idle_correction = None
    isaac._compute_task_state = lambda: None
    isaac.ts = SimpleNamespace(upper_tau=torch.zeros(1, 15), bar_b=torch.tensor([[.4, .03, .2]]),
        rel_yaw=torch.tensor([.1]), cart_yaw=torch.tensor([yaw]), cart_force_w=torch.tensor(fw[None], dtype=torch.float32),
        cart_moment_w=torch.tensor(mw[None], dtype=torch.float32), cart_v=torch.zeros(1, 2), cart_wz=torch.zeros(1),
        force_w=torch.zeros(1, 2, 3), torque_w=torch.zeros(1, 2, 3), caster_angles=torch.zeros(1, 4))
    isaac.reaction_observer = ReactionObservation(cfg.reaction, .02, 1)
    isaac.reaction = isaac.reaction_observer.controller

    mj = MujocoHighLevelRunner.__new__(MujocoHighLevelRunner)
    mj.env = SimpleNamespace(obs_hist=frame.copy(), model=SimpleNamespace(body_mass=np.array([26.9])),
                             _cart_bodies=np.array([0]), obs_scales={"lin_vel": 2., "ang_vel": .5},
                             base_ang_vel=np.array([0., 0., .1]), clip_obs=100.)
    mj.task_cfg, mj.lp = config(), lp
    mj.commands_scale = isaac.commands_scale3.numpy()
    mj.cmd_ref = isaac.cmd_ref[0].numpy().copy()
    mj.cmd_homie = isaac.cmd_homie[0].numpy().copy()
    mj.vel_est = isaac.vel_est[0].numpy().copy()
    mj.hl_actions = np.zeros(18)
    mj.actor_hist = np.zeros((cfg.env.actor_history, HL_FRAME), dtype=np.float32)
    mj.idle_correction = None
    mj.ts = dict(upper_tau=np.zeros(15), bar_b=isaac.ts.bar_b[0].numpy().copy(), rel_yaw=.1,
                 cart_yaw=yaw, cart_force_w=fw.copy(), cart_moment_w=mw.copy(), robot_centroidal={"mass": 35.})
    mj.reaction_observer = ReactionObservation(mj.task_cfg.reaction, .02, 1)
    mj.reaction = mj.reaction_observer.controller
    return isaac, mj


class ReactionAssemblerIntegrationTests(unittest.TestCase):
    def test_real_isaac_and_mujoco_observation_assemblers_match(self):
        isaac, mj = synthetic_runners()
        for _ in range(65):
            isaac._update_hl_observations()
            mj._update_hl_obs(fresh=False)
        self.assertEqual(isaac.hl_obs.shape, (1, num_actor_obs(isaac.task_cfg)))
        self.assertEqual(isaac.hl_critic_obs.shape, (1, num_critic_obs(isaac.task_cfg)))
        np.testing.assert_allclose(isaac.hl_obs[0].numpy(), mj.hl_obs, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(isaac.reaction.actor_features, mj.reaction.actor_features)
        torch.testing.assert_close(isaac.reaction_observer.command_correction_base,
                                   mj.reaction_observer.command_correction_base)

    def test_privileged_velocity_and_momentum_changes_do_not_enter_actor(self):
        left, _ = synthetic_runners()
        right, _ = synthetic_runners()
        right.ts.cart_v.fill_(3.)
        right.ts.cart_wz.fill_(4.)
        right.base_lin_vel.fill_(5.)
        right.centroidal_state = lambda _robot: dict(mass=torch.tensor([35.]),
            linear_momentum=torch.full((1, 3), 100.), angular_momentum=torch.full((1, 3), 200.),
            velocity=torch.full((1, 3), 3.))
        for _ in range(45):
            left._update_hl_observations()
            right._update_hl_observations()
        torch.testing.assert_close(left.hl_obs, right.hl_obs)
        self.assertFalse(torch.equal(left.hl_critic_obs, right.hl_critic_obs))

    def test_global_yaw_rotation_leaves_actor_and_critic_vectors_invariant(self):
        left, _ = synthetic_runners()
        right, _ = synthetic_runners()
        angle = .9
        from homie_pushcart_v5.cart_quality import rotate_planar
        yaw = torch.tensor([angle])
        robot = left.centroidal_state(left.robot)
        robot["linear_momentum"][:] = torch.tensor([[2., 3., 1.]])
        robot["angular_momentum"][:] = torch.tensor([[.4, -.2, .1]])
        robot["velocity"][:] = robot["linear_momentum"] / 35.
        rotated_robot = {key: (rotate_planar(value, yaw) if key in
                         ("linear_momentum", "angular_momentum", "velocity") else value.clone())
                         for key, value in robot.items()}
        right.centroidal_state = lambda _: rotated_robot
        ground = left.ground_contact_wrenches()
        ground["feet"][..., :3] = torch.tensor([3., 4., 170.])
        ground["feet"][..., 3:] = torch.tensor([2., -1., .5])
        rotated_ground = dict(feet=torch.cat((
            rotate_planar(ground["feet"][..., :3].reshape(-1, 3), yaw.expand(2)),
            rotate_planar(ground["feet"][..., 3:].reshape(-1, 3), yaw.expand(2))), -1).reshape(1, 2, 6),
            valid=ground["valid"].clone())
        right.ground_contact_wrenches = lambda: rotated_ground
        right.ts.cart_yaw += angle
        for key in ("cart_force_w", "cart_moment_w"):
            setattr(right.ts, key, rotate_planar(getattr(left.ts, key), yaw))
        for _ in range(45):
            left._update_hl_observations()
            right._update_hl_observations()
        torch.testing.assert_close(left.hl_obs, right.hl_obs, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(left.hl_critic_obs, right.hl_critic_obs, atol=2e-6, rtol=2e-5)

    def test_measured_relative_yaw_controls_virtual_wrist_axes_and_public_transform(self):
        q, _ = synthetic_runners()
        q.cart_pose_delay.fill_(1)
        q.cart_pose_hist[:, 0] = torch.tensor([[.4, .03, .2, .5]])
        q._update_hl_observations()
        self.assertAlmostEqual(q.reaction_observer.relative_yaw.item(), .5, places=5)
        # The first update is estimator warmup, but actuation already has an
        # observed orientation and must not fall back to current true relYaw=.1.
        self.assertFalse(q.reaction_observer.ready.item())
        self.assertNotEqual(q.reaction_observer.relative_yaw.item(), q.ts.rel_yaw.item())

    def test_disabling_reaction_preserves_frame_layout_with_zero_feature_block(self):
        isaac, mj = synthetic_runners()
        isaac.task_cfg.reaction.enabled = False
        mj.task_cfg.reaction.enabled = False
        for _ in range(45):
            isaac._update_hl_observations()
            mj._update_hl_obs(fresh=False)
        self.assertEqual(isaac.hl_obs.shape[-1], 705)
        self.assertEqual(isaac.actor_hist[0, -1, -17:].abs().max().item(), 0.)
        self.assertEqual(abs(mj.actor_hist[-1, -17:]).max(), 0.)


class RewardFrameIntegrationTests(unittest.TestCase):
    def test_yaw_lead_does_not_reward_robot_tilt_or_heading_difference(self):
        # The cart and robot have identical physical angular velocity about
        # world vertical. Robot tilt mixes horizontal rotation into body-z,
        # which must not be mistaken for the cart tracking yaw better.
        orientations = ((0., 0., .7), (.5, .6, .9), (-.3, .7, -1.4))
        world_omega = np.array([.4, -.2, .2])
        body_omega = []
        for roll, pitch, yaw in orientations:
            cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll),
                                      math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw))
            rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
            ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
            rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
            body_omega.append((rz @ ry @ rx).T @ world_omega)
        env = PushCartEnv.__new__(PushCartEnv)
        env.task_cfg = config()
        env.meas_alpha = .1
        env.cmd_ref = torch.tensor([[0., 0., .2]]).repeat(3, 1)
        env.cmd_ref_f = torch.zeros(3, 3)
        env.cart_vel_f = torch.zeros(3, 3)
        env.base_wz_f = torch.zeros(3)
        env.base_ang_vel = torch.tensor(np.array(body_omega), dtype=torch.float32)
        env.root_states = torch.zeros(3, 13)
        env.root_states[:, 10:13] = torch.tensor(world_omega, dtype=torch.float32)
        env.ts = SimpleNamespace(cart_v=torch.zeros(3, 2), cart_wz=torch.full((3,), .2),
                                 cart_yaw=torch.tensor([.2, -.4, .8]))
        env.cart_quality = None
        # Bypass scene/other reward queries, retaining the real post-physics
        # tracking filters and the actual yaw-lead reward function.
        for name in ("_compute_task_state", "_update_reaction_diagnostics",
                     "_update_persistence_counters", "_accumulate_diagnostics"):
            setattr(env, name, lambda: None)
        self.assertTrue(torch.all((env.base_ang_vel[1:, 2] - .2).abs() > .04))
        self.assertTrue(torch.all((torch.tensor([q[2] for q in orientations]) - env.ts.cart_yaw).abs() > .1))
        with patch.object(PushCartSimEnv, "_post_physics_step_callback", return_value=None):
            for _ in range(80):
                env._post_physics_step_callback()
        torch.testing.assert_close(env._reward_cart_yaw_lead(), torch.zeros(3), atol=1e-7, rtol=0.)

    def fixture(self, command=(0., 0., 0.), force=(0., 0., 200.), cop=(0., 0.)):
        env = PushCartEnv.__new__(PushCartEnv)
        env.num_envs, env.dt = 1, .02
        env.task_cfg = config()
        env.cfg = SimpleNamespace(rewards=SimpleNamespace(base_height_target=.74))
        env.cmd_ref = torch.tensor([command])
        env.projected_gravity = torch.tensor([[0., 0., -1.]])
        env.root_states = torch.zeros(1, 13)
        env.root_states[:, 12] = command[2]
        env.dynamics_wrench_valid = torch.ones(1, dtype=torch.bool)
        env.episode_length_buf = torch.ones(1, dtype=torch.long)
        env.robot, env.cart = object(), object()
        robot_com, cart_com = torch.tensor([[0., 0., .74]]), torch.tensor([[.8, 0., .3]])
        inertia = torch.tensor([[[3., 0., .4], [0., 2., .2], [.4, .2, 1.5]]])
        robot_velocity = torch.tensor([[command[0], command[1] - .8 * command[2], 0.]])
        robot = dict(com=robot_com, mass=torch.tensor([35.]), velocity=robot_velocity,
                     inertia_world=inertia, angular_momentum=inertia[:, :, 2] * command[2])
        cart = dict(com=cart_com, mass=torch.tensor([26.9]), inertia_world=torch.eye(3)[None])
        env.centroidal_state = lambda body: robot if body is env.robot else cart
        env.ts = SimpleNamespace(cart_force_w=torch.zeros(1, 3), cart_moment_w=torch.zeros(1, 3),
                                 cart_yaw=torch.zeros(1), rel_yaw=torch.zeros(1),
                                 cart_v=env.cmd_ref[:, :2], cart_wz=env.cmd_ref[:, 2])
        env.reaction_teacher = ReactionController(env.task_cfg.reaction, .02, 1)
        env.centroidal_filtered = torch.zeros(1, 6)
        env.centroidal_cost, env.support_cost = torch.zeros(1), torch.zeros(1)
        env.reaction_episode_sum, env.reaction_episode_samples = torch.zeros(1, 4), torch.zeros(1)
        env.feet_pos = torch.tensor([[[0., .1, .035], [0., -.1, .035]]])
        env.feet_quat = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])
        contact_force = torch.tensor(force)[None, None].expand(1, 2, 3)
        contact_position = env.feet_pos + torch.tensor([.035 + cop[0], cop[1], -.035])
        contact_moment = torch.linalg.cross(contact_position - robot_com[:, None], contact_force)
        ground = dict(feet=torch.cat((contact_force, contact_moment), -1), valid=torch.ones(1, dtype=torch.bool))
        env.ground_contact_wrenches = lambda: ground
        return env

    def test_command_residual_uses_same_observed_heading_as_actuation(self):
        env = PushCartEnv.__new__(PushCartEnv)
        env.cmd_ref = torch.tensor([[.3, 0., .2]])
        env.cmd_homie = torch.tensor([[0., .3, .2]])
        env.reaction_observer = ReactionObservation(config().reaction, .02, 1, handle_offset_cart=(0., 0., 0.))
        env.reaction_observer._relative_yaw.fill_(math.pi / 2.)
        self.assertLess(env._reward_homie_cmd_residual().item(), 1e-12)

    def test_nominal_rigid_turn_has_no_spurious_robot_momentum_or_motion_cost(self):
        env = self.fixture(command=(.3, 0., .4))
        for _ in range(45):
            env._update_reaction_diagnostics()
        self.assertEqual(env.centroidal_cost.item(), 0.)
        self.assertEqual(env.reaction_teacher.costs["reaction_motion"].item(), 0.)

    def test_cop_is_about_shifted_sole_plane_with_tangential_force(self):
        # Forward foot pressure at x=.105 from ankle is inside the nominal
        # forward contact x=.12. Tangential force must not invent a lever error.
        env = self.fixture(force=(60., 0., 200.), cop=(.07, 0.))
        env._update_reaction_diagnostics()
        self.assertEqual(env.support_cost.item(), 0.)
        outside = self.fixture(cop=(-.10, 0.))
        outside._update_reaction_diagnostics()
        self.assertGreater(outside.support_cost.item(), .1)

    def test_centroidal_filter_memory_stays_in_fixed_world_axes_during_turn(self):
        env = self.fixture()
        env.centroidal_filtered[0, :3] = torch.tensor([.4, 0., 0.])
        env.ts.cart_yaw.fill_(math.pi / 2.)
        env._update_reaction_diagnostics()
        # World-x memory stays world-x even as cart axes turn 90 degrees.
        self.assertGreater(env.centroidal_filtered[0, 0].item(), .3)
        self.assertEqual(env.centroidal_filtered[0, 1].item(), 0.)


class MujocoPhysicsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MujocoHomieEnv(with_cart=True)

    def setUp(self):
        self.env.reset()

    def test_force_pulse_delivers_final_partial_control_interval_and_exact_impulse(self):
        env = self.env
        dt = env.model.opt.timestep
        env.inject_cart_wrench([10., 0., 0.], [0., 0., 0.], 5 * dt, offset=[0., .2, 0.])
        samples = []
        for _ in range(2):
            env.step(np.zeros(env.num_lower))
            samples.extend(env.physics_samples)
        applied = np.asarray([sample["cart_applied_wrench"] for sample in samples])
        np.testing.assert_allclose(applied[:5], np.tile([10., 0., 0., 0., 0., -2.], (5, 1)))
        np.testing.assert_allclose(applied[5:], 0.)
        np.testing.assert_allclose(env.cart_pulse_impulse, np.array([10., 0., 0., 0., 0., -2.]) * 5 * dt)
        self.assertEqual(env._pulse_steps, 0)
        np.testing.assert_allclose(np.diff([x["time_s"] for x in samples]), dt)

    def test_reset_clears_active_external_force_and_impulse_history(self):
        env = self.env
        env.inject_cart_wrench([10., 0., 0.], [0., 0., 1.], 1.)
        env.step(np.zeros(env.num_lower))
        self.assertGreater(env.cart_pulse_impulse[0], 0.)
        env.reset()
        self.assertEqual(env._pulse_steps, 0)
        np.testing.assert_allclose(env.cart_pulse_impulse, 0.)
        np.testing.assert_allclose(env.data.xfrc_applied, 0.)
        env.step(np.zeros(env.num_lower))
        self.assertTrue(all(np.max(np.abs(x["cart_applied_wrench"])) == 0. for x in env.physics_samples))

    def test_nonfinite_or_wrong_shape_pulse_offset_rejected_before_physics(self):
        for offset in ([0., float("nan"), 0.], [0., 0.], [0., 0., float("inf")]):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.env.inject_cart_wrench([10., 0., 0.], [0., 0., 0.], .1, offset=offset)
        self.assertEqual(self.env._pulse_steps, 0)

    def test_payload_updates_whole_cart_mass_and_physical_inertia_without_accumulation(self):
        runner = MujocoHighLevelRunner(self.env, SimpleNamespace(), None, config(), level=0)
        original_inertia = self.env.model.body_inertia[runner.cart_body].copy()
        runner.set_payload(25.)
        self.assertAlmostEqual(runner.cart_mass_total, 39.4, places=6)
        self.assertAlmostEqual(self.env.centroidal_state(self.env.cart_id)["mass"], 39.4, places=6)
        expected = original_inertia + 25. * np.array([.3**2 + .2**2, .4**2 + .2**2, .4**2 + .3**2]) / 12.
        np.testing.assert_allclose(self.env.model.body_inertia[runner.cart_body], expected)
        runner.set_payload(0.)
        self.assertAlmostEqual(runner.cart_mass_total, 14.4, places=6)
        np.testing.assert_allclose(self.env.model.body_inertia[runner.cart_body], original_inertia)


if __name__ == "__main__":
    unittest.main()
