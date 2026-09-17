"""CPU parity checks for the push-cart low-level path (no Isaac Sim needed).

1. The MuJoCo sim2sim env assembles HOMIE's 76-dim observation frame exactly like ``HomieTaskCore``
   (same function, fed with the MuJoCo state) and shares its PD parameters.
2. ``PushCartSimEnv`` inherits observation / torque / stepping from ``HomieIsaacLabEnv`` unchanged
   (checked on the source, so it holds without launching Kit).
3. The upper-body hold target (preset + measured sag) lies inside HOMIE's action limits, i.e. the arms
   are commanded exactly to the preset and never clipped.
4. The frozen HOMIE checkpoint loads with strict=True and is deterministic on CPU.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "HomieRL", ROOT / "HomieRL" / "rsl_rl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from homie_isaaclab.config import G1RoughCfg  # noqa: E402
from homie_isaaclab.task_core import HomieTaskCore  # noqa: E402
from homie_pushcart.paths import G1_GRASP_URDF, G1_URDF, HOMIE_JOINTS, HOMIE_RUN_DIR  # noqa: E402
from homie_pushcart.pose import load_initial_pose  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _mujoco_env():
    from homie_pushcart.assets.mjcf_scene import MJCF_SCENE
    from homie_pushcart.mujoco_env import MujocoHomieEnv
    if not MJCF_SCENE.is_file():
        raise unittest.SkipTest("run `pushcart.py mj build` first")
    return MujocoHomieEnv(with_cart=True)


def _core_like(env):
    """A stand-in with exactly the attributes HomieTaskCore.compute_observations reads, from MuJoCo state."""
    import mujoco
    cfg = env.cfg
    w, x, y, z = env.base_quat_wxyz
    quat_xyzw = torch.tensor([[x, y, z, w]], dtype=torch.float32)
    # HOMIE rotates the IMU's *world* angular velocity into the IMU frame (imu_in_pelvis has identity rotation)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, env.base_quat_wxyz)
    omega_world = torch.tensor(rot.reshape(3, 3) @ env.base_ang_vel, dtype=torch.float32)
    imu_index = 0
    rigid = torch.zeros(1, 1, 13)
    rigid[0, imu_index, 3:7] = quat_xyzw[0]
    rigid[0, imu_index, 10:13] = omega_world
    scales = cfg.normalization.obs_scales
    ns = SimpleNamespace(
        rigid_body_states=rigid, imu_index=imu_index,
        gravity_vec=torch.tensor([[0., 0., -1.]]),
        commands=torch.tensor(env.commands, dtype=torch.float32).unsqueeze(0),
        commands_scale=torch.tensor([scales.lin_vel, scales.lin_vel, scales.ang_vel]),
        obs_scales=scales,
        dof_pos=torch.tensor(env.dof_pos, dtype=torch.float32).unsqueeze(0),
        dof_vel=torch.tensor(env.dof_vel, dtype=torch.float32).unsqueeze(0),
        default_dof_pos=torch.tensor(env.default_dof_pos, dtype=torch.float32).unsqueeze(0),
        actions=torch.tensor(np.concatenate([env.actions, np.zeros(15)]), dtype=torch.float32).unsqueeze(0),
        add_noise=False,
        num_one_step_obs=cfg.env.num_one_step_observations,
        actor_proprioceptive_obs_length=cfg.env.num_observations,
        num_one_step_privileged_obs=cfg.env.num_one_step_privileged_obs,
        critic_proprioceptive_obs_length=cfg.env.num_privileged_obs,
        obs_buf=torch.zeros(1, cfg.env.num_observations),
        privileged_obs_buf=torch.zeros(1, cfg.env.num_privileged_obs),
        base_lin_vel=torch.tensor(env.base_lin_vel, dtype=torch.float32).unsqueeze(0),
    )
    return ns


@unittest.skipIf(torch is None, "torch not available")
class MujocoObservationParity(unittest.TestCase):
    def test_observation_frame_matches_task_core(self):
        env = _mujoco_env()
        env.reset()
        env.set_command(0.3, -0.1, 0.2)
        rng = np.random.default_rng(0)
        frame = env.one_step_obs
        prev = None
        for _ in range(6):
            obs = env.step(0.3 * rng.standard_normal(12))[0]
            ns = _core_like(env)
            HomieTaskCore.compute_observations(ns)
            np.testing.assert_allclose(obs[-frame:], ns.obs_buf[0, -frame:].numpy(), rtol=0, atol=2e-6)
            # critic frame = actor frame + base linear velocity, same as HOMIE
            np.testing.assert_allclose(ns.privileged_obs_buf[0, -frame - 3:-3].numpy(), ns.obs_buf[0, -frame:].numpy(), atol=0)
            if prev is not None:      # history is a pure shift: the previous frame moved one slot back
                np.testing.assert_array_equal(obs[-2 * frame:-frame], prev[-frame:])
            prev = obs

    def test_history_is_zero_after_reset_then_shifts(self):
        env = _mujoco_env()
        obs0 = env.reset()
        frame = env.one_step_obs
        self.assertEqual(obs0.shape, (env.history_len * frame,))
        self.assertTrue(np.all(obs0[:-frame] == 0.0))
        env.set_command(0.2, 0., 0.)
        obs1 = env.step(np.zeros(12))[0]
        np.testing.assert_array_equal(obs1[-2 * frame:-frame], obs0[-frame:])

    def test_pd_parameters_match_homie(self):
        env = _mujoco_env()
        cfg = G1RoughCfg()
        p, d = np.zeros(27), np.zeros(27)
        for i, name in enumerate(HOMIE_JOINTS):              # task_core._init_task_buffers: last matching key wins
            for key in cfg.control.stiffness:
                if key in name:
                    p[i], d[i] = cfg.control.stiffness[key], cfg.control.damping[key]
        np.testing.assert_array_equal(env.p_gains, p)
        np.testing.assert_array_equal(env.d_gains, d)
        np.testing.assert_array_equal(env.default_dof_pos, [cfg.init_state.default_joint_angles[n] for n in HOMIE_JOINTS])
        self.assertEqual(env.action_scale, cfg.control.action_scale)
        self.assertEqual(env.decimation, cfg.control.decimation)
        self.assertAlmostEqual(env.dt, cfg.control.decimation * cfg.sim.dt)
        efforts = {j.get("name"): float(j.find("limit").get("effort")) for j in ET.parse(G1_URDF).getroot().findall("joint")
                   if j.get("type") != "fixed"}
        np.testing.assert_allclose(env.torque_limits, [efforts[n] for n in HOMIE_JOINTS])
        limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                  for j in ET.parse(G1_URDF).getroot().findall("joint") if j.get("type") != "fixed"}
        lo = np.array([limits[n][0] for n in HOMIE_JOINTS])
        hi = np.array([limits[n][1] for n in HOMIE_JOINTS])
        np.testing.assert_allclose(env.action_min, (lo - env.default_dof_pos) / env.action_scale, atol=1e-6)
        np.testing.assert_allclose(env.action_max, (hi - env.default_dof_pos) / env.action_scale, atol=1e-6)


class SimEnvInheritsHomieCore(unittest.TestCase):
    INHERITED = ("compute_observations", "compute_termination_observations", "compute_reward", "check_termination",
                 "_apply_action", "step", "_sync_states", "_get_noise_scale_vec", "_post_physics_step_callback",
                 "_init_task_buffers", "_prepare_reward_function")
    OVERRIDDEN = ("_pre_physics_step", "_resample_commands", "_reset_dofs", "_reset_root_states")

    def test_pushcart_sim_env_overrides_only_the_scene_and_command_hooks(self):
        source = (ROOT / "HomieRL/homie_pushcart/sim_env.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PushCartSimEnv")
        self.assertEqual([b.id for b in cls.bases], ["HomieIsaacLabEnv"])
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name in self.INHERITED:
            self.assertNotIn(name, methods, f"{name} must stay inherited from HOMIE")
        for name in self.OVERRIDDEN:
            self.assertIn(name, methods)

    def test_homie_isaaclab_package_untouched_by_pushcart(self):
        pkg = ROOT / "HomieRL/homie_pushcart"
        for path in pkg.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("homie_isaaclab/", text.replace("HomieRL/homie_isaaclab", ""),
                             f"{path.name} must not write into homie_isaaclab")


class UpperHoldTarget(unittest.TestCase):
    def test_hold_target_inside_action_limits(self):
        pose = load_initial_pose()
        cfg = G1RoughCfg()
        limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                  for j in ET.parse(G1_GRASP_URDF).getroot().findall("joint") if j.get("type") != "fixed"}
        offset = pose["robot"].get("upper_hold_offset") or {}
        for name in HOMIE_JOINTS[12:]:
            target = pose["robot"]["joint_pos"][name] + float(offset.get(name, 0.))
            lo, hi = limits[name]
            self.assertGreaterEqual(target, lo, name)
            self.assertLessEqual(target, hi, name)
            action = (target - cfg.init_state.default_joint_angles[name]) / cfg.control.action_scale
            self.assertLessEqual(abs(action), cfg.normalization.clip_actions)
        # wrist pitch is the tightest: report the remaining margin so a future pose change is noticed
        margin = limits["left_wrist_pitch_joint"][1] - pose["robot"]["joint_pos"]["left_wrist_pitch_joint"]
        self.assertGreater(margin, 0.2)


class TaskConfigAndCurriculum(unittest.TestCase):
    def test_observation_layout_constants(self):
        from homie_pushcart.task_config import HL_FRAME, HOMIE_FRAME, PRIVILEGED_FRAME, PushCartTaskCfg, num_actor_obs, num_critic_obs
        cfg = PushCartTaskCfg()
        self.assertEqual(HOMIE_FRAME, G1RoughCfg.env.num_one_step_observations)
        self.assertEqual(HL_FRAME, HOMIE_FRAME + 3 + 1 + 15 + 3 + 3 + 4 + cfg.env.num_hl_actions + 1)
        self.assertEqual(num_actor_obs(cfg), HL_FRAME * cfg.env.actor_history)
        self.assertEqual(num_critic_obs(cfg), num_actor_obs(cfg) + PRIVILEGED_FRAME)
        from homie_isaaclab.math_utils import class_to_dict
        scales = class_to_dict(cfg.rewards.scales)
        self.assertGreater(scales["track_cart_lin_vel"], 0)
        self.assertLess(scales["termination"], 0)
        self.assertFalse(cfg.env.enable_height_action)
        self.assertEqual(cfg.env.leg_residual_scale, 0.0)

    def test_curriculum_schedule_is_monotone_and_bounded(self):
        from homie_pushcart.curriculum import level_params
        from homie_pushcart.task_config import PushCartTaskCfg
        cfg = PushCartTaskCfg()
        prev = None
        for level in range(cfg.curriculum.num_levels):
            p = level_params(level, cfg)
            self.assertLessEqual(p.vx_range[1], cfg.commands.vx_range_max[1])
            self.assertGreaterEqual(p.vx_range[0], cfg.commands.vx_range_max[0])
            self.assertLessEqual(p.wz_max, cfg.commands.wz_max)
            self.assertLessEqual(p.arm_scale, cfg.actions.arm_scale_range[1])
            self.assertLessEqual(p.payload_max_kg, 25.0)
            self.assertGreaterEqual(p.grasp_threshold_mult, 1.0)
            if prev is not None:
                self.assertGreaterEqual(p.vx_range[1], prev.vx_range[1])
                self.assertGreaterEqual(p.payload_max_kg, prev.payload_max_kg)
                self.assertLessEqual(p.grasp_threshold_mult, prev.grasp_threshold_mult)
                self.assertGreaterEqual(p.obs_noise_frac, prev.obs_noise_frac)
            prev = p
        first, last = level_params(0, cfg), level_params(cfg.curriculum.num_levels - 1, cfg)
        self.assertEqual(first.vx_range, (0.0, 0.0))
        self.assertEqual(first.wz_max, 0.0)
        self.assertEqual(last.vx_range, tuple(cfg.commands.vx_range_max))
        self.assertEqual(last.wz_max, cfg.commands.wz_max)
        self.assertEqual(last.payload_max_kg, 25.0)
        self.assertTrue(last.robot_push)

    def test_curriculum_tracker_hysteresis(self):
        from homie_pushcart.curriculum import CurriculumTracker
        from homie_pushcart.task_config import PushCartTaskCfg
        cfg = PushCartTaskCfg()
        t = CurriculumTracker(cfg)
        self.assertTrue(t.record_window(0.8, 0.0)); self.assertEqual(t.level, 1)
        self.assertTrue(t.record_window(0.9, 0.0)); self.assertEqual(t.level, 2)
        self.assertFalse(t.record_window(0.8, 0.2)); self.assertEqual(t.level, 2)      # fail rate too high: no promotion (1st miss)
        self.assertTrue(t.record_window(0.5, 0.0)); self.assertEqual(t.level, 1)       # 2nd consecutive miss: demote
        self.assertFalse(t.record_window(0.5, 0.0)); self.assertEqual(t.level, 1)      # counter restarted after the demotion
        self.assertTrue(t.record_window(0.5, 0.0)); self.assertEqual(t.level, 0)
        self.assertFalse(t.record_window(0.5, 0.0)); self.assertFalse(t.record_window(0.5, 0.0)); self.assertEqual(t.level, 0)   # floor
        # per-episode interface reaches the same decision once a window is full
        t2 = CurriculumTracker(cfg)
        self.assertTrue(t2.record([0.9] * cfg.curriculum.window_episodes, [0] * cfg.curriculum.window_episodes))
        self.assertEqual(t2.level, 1)


@unittest.skipIf(torch is None, "torch not available")
class MujocoHighLevelFrame(unittest.TestCase):
    def test_runner_frame_layout_matches_isaac_definition(self):
        from types import SimpleNamespace
        from homie_pushcart.export import frame_layout
        from homie_pushcart.lowlevel import latest_checkpoint
        from homie_pushcart.task_config import HL_FRAME, PushCartTaskCfg, num_actor_obs
        try:
            latest_checkpoint(HOMIE_RUN_DIR)
        except FileNotFoundError:
            raise unittest.SkipTest("no HOMIE checkpoint")
        from homie_pushcart.mujoco_hl import make_runner
        runner, _ = make_runner(SimpleNamespace(checkpoint=None, homie_checkpoint=None, level=6))
        cfg = PushCartTaskCfg()
        obs = runner.reset()
        self.assertEqual(obs.shape, (num_actor_obs(cfg),))
        self.assertTrue(np.all(obs[:-HL_FRAME] == 0.0))           # fresh history
        for _ in range(5):
            obs, fallen = runner.step((0.3, 0.0, 0.1))
            self.assertFalse(fallen)
        self.assertTrue(np.isfinite(obs).all())
        frame = obs[-HL_FRAME:]
        slices = {item["name"]: slice(item["start"], item["start"] + item["dim"]) for item in frame_layout(cfg)}
        np.testing.assert_array_equal(frame[slices["homie_cmd"]].astype(np.float64).round(6),
                                      (runner.cmd_homie * runner.commands_scale).round(6))
        np.testing.assert_allclose(frame[slices["task_cmd"]], np.array([0.3, 0.0, 0.1]) * runner.commands_scale, atol=1e-6)
        self.assertAlmostEqual(float(frame[slices["cart_mass"]][0]), (12.0 - 12.0) / 25.0, places=6)
        np.testing.assert_array_equal(frame[slices["last_hl_action"]], np.zeros(18))     # passthrough
        self.assertAlmostEqual(float(frame[slices["correction_scale"]][0]), 0.45 / 0.5, places=5)
        pose = frame[slices["handle_pose_in_base"]]
        self.assertGreater(pose[0], 0.25); self.assertLess(pose[0], 0.4)                # handle in front of the pelvis
        self.assertLess(abs(pose[3]), 0.1)                                              # cart and robot aligned
        # the whole history shifted by one frame per step
        obs2, _ = runner.step((0.3, 0.0, 0.1))
        np.testing.assert_array_equal(obs2[-2 * HL_FRAME:-HL_FRAME], obs[-HL_FRAME:])


@unittest.skipIf(torch is None, "torch not available")
class HighLevelExport(unittest.TestCase):
    def test_export_matches_actor_and_mujoco_frame_layout(self):
        import tempfile
        runs = sorted((ROOT / "logs/rsl_rl/homie_pushcart").glob("*/model_*.pt")) if (ROOT / "logs/rsl_rl/homie_pushcart").is_dir() else []
        if not runs:
            raise unittest.SkipTest("no high-level checkpoint yet")
        from homie_pushcart.export import export_checkpoint, frame_layout
        from homie_pushcart.hl_policy import HighLevelPolicy
        from homie_pushcart.task_config import HL_FRAME, PushCartTaskCfg, num_actor_obs, num_critic_obs
        cfg = PushCartTaskCfg()
        layout = frame_layout(cfg)
        self.assertEqual(layout[-1]["start"] + layout[-1]["dim"], HL_FRAME)
        with tempfile.TemporaryDirectory() as tmp:
            info = export_checkpoint(runs[-1], Path(tmp))
            hl = HighLevelPolicy(runs[-1], num_actor_obs(cfg), num_critic_obs(cfg), cfg.env.num_hl_actions, "cpu")
            obs = torch.randn(4, num_actor_obs(cfg)) * 0.5
            jit = torch.jit.load(info["files"]["torchscript"])
            torch.testing.assert_close(jit(obs), hl(obs), atol=1e-5, rtol=1e-5)
            try:
                import onnxruntime as ort
            except ImportError:
                return
            sess = ort.InferenceSession(info["files"]["onnx"], providers=["CPUExecutionProvider"])
            out = sess.run(None, {"actor_obs": obs.numpy()})[0]
            np.testing.assert_allclose(out, hl(obs).numpy(), atol=1e-4, rtol=1e-4)


@unittest.skipIf(torch is None, "torch not available")
class LowLevelPolicy(unittest.TestCase):
    def test_checkpoint_loads_strict_and_is_deterministic(self):
        from homie_pushcart.lowlevel import HimLowLevelPolicy, latest_checkpoint
        try:
            checkpoint = latest_checkpoint(HOMIE_RUN_DIR)
        except FileNotFoundError:
            raise unittest.SkipTest("no HOMIE checkpoint")
        policy = HimLowLevelPolicy(checkpoint, device="cpu")
        obs = torch.zeros(3, policy.num_obs)
        obs[:, -76:-12] = torch.linspace(-0.5, 0.5, 64)
        a, b = policy(obs), policy(obs)
        self.assertEqual(tuple(a.shape), (3, 12))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()
