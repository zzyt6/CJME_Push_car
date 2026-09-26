"""Physical regression checks for the V5.x Isaac-parameter MuJoCo profile."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
from pushcart_bench.policies_v5x import InstrumentedEnv
from pushcart_bench.sim2sim_scene import payload_properties, set_payload
from homie_pushcart_v52.paths import G1_GRASP_URDF, HOMIE_JOINTS


class AlignedSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = InstrumentedEnv(True, scene_profile="isaac")
        cls.urdf = ET.parse(G1_GRASP_URDF).getroot()

    def test_robot_collision_set_and_meshes_match_training_urdf(self):
        m = self.env.model
        expected = {l.get("name") for l in self.urdf.findall("link") if l.find("collision") is not None}
        actual = {m.body(int(m.geom_bodyid[g])).name for g in range(m.ngeom)
                  if (m.geom_contype[g] or m.geom_conaffinity[g]) and
                  not m.body(int(m.geom_bodyid[g])).name.startswith(("world", "cart", "wheel", "caster"))}
        self.assertEqual(actual, expected)
        for body in expected:
            geom = m.geom(f"isaac_{body}_0").id
            self.assertEqual(m.geom_type[geom], mujoco.mjtGeom.mjGEOM_MESH)

    def test_joint_ranges_and_passive_dynamics_match_training(self):
        e, m = self.env, self.env.model
        for name in HOMIE_JOINTS:
            limit = self.urdf.find(f"joint[@name='{name}']/limit")
            np.testing.assert_allclose(m.jnt_range[m.joint(name).id], [float(limit.get("lower")), float(limit.get("upper"))])
        np.testing.assert_array_equal(m.dof_frictionloss[e.qvel_adr], 0.)
        np.testing.assert_array_equal(m.dof_damping[e.qvel_adr], 0.)
        np.testing.assert_allclose(m.dof_armature[e.qvel_adr], e.cfg.asset.armature)
        for i in range(4):
            for name in (f"wheel_{i}_joint", f"caster_{i}_swivel_joint"):
                dof = m.joint(name).dofadr[0]
                self.assertEqual(m.dof_armature[dof], 0.)
                self.assertEqual(m.dof_frictionloss[dof], 0.)

    def test_robot_mass_is_urdf_mass_including_fixed_palms(self):
        mass = sum(float(x.get("value")) for x in self.urdf.findall("link/inertial/mass"))
        self.assertAlmostEqual(self.env.model.body_subtreemass[self.env.pelvis_id], mass, places=8)

    def test_contact_friction_matches_physx_runtime_materials(self):
        m = self.env.model
        for side in ("left", "right"):
            np.testing.assert_allclose(m.pair_friction[m.pair(f"isaac_{side}_ankle_roll_link_0_floor").id], [.75, .75, 0., 0., 0.])
        for i in range(4):
            np.testing.assert_allclose(m.pair_friction[m.pair(f"wheel_floor_{i}").id], [.9, .9, 0., 0., 0.])

    def test_independent_newton_euler_balance_matches_weld_force(self):
        e = InstrumentedEnv(True, scene_profile="isaac")
        e.audit_newton_euler = True
        e.reset()
        errors = []
        for _ in range(20):
            e.step(np.zeros(12))
            errors.extend((np.array(e.sub_newton_euler)-e.sub_hands).reshape(-1, 6))
        rms = np.sqrt((np.array(errors)**2).mean(0))
        self.assertLess(rms[:3].max(), 1e-6)
        self.assertLess(rms[3:].max(), .05)  # finite soft-weld anchor displacement

    def test_upper_actuator_observation_is_pd_proxy_not_position_control(self):
        e = InstrumentedEnv(True, decimation=1, scene_profile="isaac")
        e.reset()
        e.upper_delta[:] = .02
        target = e.default_dof_pos[12:] + e.action_scale * np.clip(
            e.upper_hold_actions + e.upper_delta, e.action_min[12:], e.action_max[12:])
        expected = np.clip(e.p_gains[12:] * (target - e.dof_pos[12:]) - e.d_gains[12:] * e.dof_vel[12:],
                           -e.torque_limits[12:], e.torque_limits[12:])
        e.step(np.zeros(12))
        np.testing.assert_allclose(e.upper_torque_proxy, expected, atol=1e-10)
        np.testing.assert_allclose(e.data.ctrl[e.act_idx[12:]], target, atol=1e-10)
        self.assertGreater(np.max(abs(e.upper_torque_proxy - e.data.ctrl[e.act_idx[12:]])), 1.)

    def test_integrated_pose_and_reported_body_pose_are_same_sample(self):
        e = self.env
        e.reset()
        for _ in range(10):
            e.step(np.zeros(12))
        np.testing.assert_allclose(e.data.xpos[e.pelvis_id], e.data.qpos[e.root_qpos:e.root_qpos+3], atol=1e-10)
        cart_adr = e.model.joint("cart_free").qposadr[0]
        np.testing.assert_allclose(e.data.xpos[e.cart_id], e.data.qpos[cart_adr:cart_adr+3], atol=1e-10)

    def test_payload_parallel_axis_and_reset_do_not_accumulate(self):
        e = InstrumentedEnv(True, scene_profile="isaac")
        b = e.cart_id
        set_payload(e, 25., (.1, -.08))
        mass, com, tensor = payload_properties(12., np.array([0., 0., .3]),
                                              np.diag([.6516, 1.36, 1.2916]), 25., (.1, -.08))
        rot = Rotation.from_quat(np.roll(e.model.body_iquat[b], -1)).as_matrix()
        np.testing.assert_allclose(e.model.body_ipos[b], com)
        np.testing.assert_allclose(rot @ np.diag(e.model.body_inertia[b]) @ rot.T, tensor, atol=1e-10)
        self.assertAlmostEqual(e.model.body_subtreemass[b], 39.4)
        self.assertAlmostEqual(e.model.body_mass[b], mass)
        set_payload(e, 0.)
        np.testing.assert_allclose(e.model.body_ipos[b], [0., 0., .3])
        self.assertAlmostEqual(e.model.body_subtreemass[b], 14.4)
        with self.assertRaises(ValueError):
            set_payload(e, -1.)


class CheckpointSensorCompatibility(unittest.TestCase):
    def test_old_checkpoint_does_not_inherit_new_sensor_delay(self):
        from homie_pushcart_v52.task_config import task_cfg_from_run, PushCartTaskCfg
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "params").mkdir()
            path = run / "params/env.yaml"
            saved = {"task": {"version": 52, "observations": {"upper_torque_noise_frac": .05}}}
            path.write_text(yaml.safe_dump(saved), encoding="utf-8")
            cfg = task_cfg_from_run(run, verbose=False)
            self.assertEqual(cfg.observations.upper_torque_delay_steps_max, 0)
            self.assertEqual(cfg.observations.upper_torque_bias_frac_of_limit, 0.)
            self.assertEqual(cfg.observations.upper_torque_noise_frac, .05)
            saved['task']['observations']['upper_torque_delay_steps_max'] = 2
            path.write_text(yaml.safe_dump(saved), encoding="utf-8")
            self.assertEqual(task_cfg_from_run(run, verbose=False).observations.upper_torque_delay_steps_max, 2)
        self.assertEqual(PushCartTaskCfg().observations.upper_torque_delay_steps_max, 2)

    def test_actual_isaac_and_mujoco_frame_builders_match_on_identical_samples(self):
        # Reuse a scene-query fixture, then execute BOTH production assemblers.
        # This detects field order, scales, history/reset and prior bypasses.
        from types import SimpleNamespace
        import torch
        from test_pushcart_v51_observation import mock_environment
        from homie_pushcart_v52.task_config import PushCartTaskCfg
        from homie_pushcart_v52.task_env import PushCartEnv
        from homie_pushcart_v52.fixed_grasp import FixedGraspObserver
        from homie_pushcart_v52 import observation as ob
        from pushcart_bench.policies_v5x import V5xRunner
        e = mock_environment(1)
        e.task_cfg = cfg = PushCartTaskCfg()
        cfg.interaction.cuda_graph_fk = False
        e.fixed_grasp = FixedGraspObserver(1)
        e.recovery_authority = ob.RecoveryAuthority(cfg.recovery, .02, 1, 'cpu')
        e.actor_hist = torch.zeros(1, 20, 136)
        e.actions = torch.zeros(1, 27)
        e.upper_torque_hist = torch.zeros(1, 3, 15)
        e.upper_torque_delay = torch.zeros(1, dtype=torch.long)
        e.hand_wrench_scale = torch.tensor(ob.HAND_WRENCH_SCALE).repeat(2)
        e.grasp_wrench_interval_mean = lambda: (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3),
                                                torch.zeros(1, 2, 3), torch.ones(1))
        r = V5xRunner.__new__(V5xRunner)
        r.task_cfg, r.obm = cfg, ob
        r.observer = FixedGraspObserver(1)
        r.recovery = ob.RecoveryAuthority(cfg.recovery, .02, 1, 'cpu')
        r.frame_dim, r.has_prior = 136, True
        r.hand_scale = e.hand_wrench_scale.numpy()
        r.actor_hist = np.zeros((20, 136), dtype=np.float32)
        r._predict = lambda: None
        r.commands_scale = e.commands_scale3.numpy()
        r.env = SimpleNamespace(obs_scales=vars(e.obs_scales), default_dof_pos=e.default_dof_pos[0].numpy(),
                                num_lower=12, clip_obs=100.)
        for step in range(4):
            e.episode_length_buf[:] = step
            e.actions[:, :12] = torch.arange(12) * .01 + step * .1
            r.env.actions = e.actions[0, :12].numpy()
            e.dof_pos[:, 14] += .001
            e.dof_vel[:, 13] += .02
            for key, source in (('dof_pos', e.dof_pos), ('dof_vel', e.dof_vel),
                                ('base_ang_vel', e.base_ang_vel), ('projected_gravity', e.projected_gravity)):
                setattr(r.env, key, source[0].numpy())
            for key in ('vel_est', 'hold_target_q', 'upper_delta_q', 'cmd_ref', 'cmd_homie', 'hl_actions'):
                setattr(r, key, getattr(e, key)[0].numpy())
            r.ts = {'upper_tau': e.ts.upper_tau[0].numpy()}
            PushCartEnv._update_hl_observations(e)
            r._update_hl_obs(fresh=step == 0)
            np.testing.assert_allclose(r.hl_obs, e.hl_obs[0].numpy(), atol=2e-6, rtol=0.)
            np.testing.assert_allclose(r.falcon_obs.numpy(), e.falcon_obs.numpy(), atol=2e-6, rtol=0.)


if __name__ == "__main__":
    unittest.main()
