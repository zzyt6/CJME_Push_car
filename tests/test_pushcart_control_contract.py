"""End-to-end control-interface checks on identical sensor samples (no Kit)."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
from homie_isaaclab.task_core import HomieTaskCore
from homie_pushcart_v52.control_contract import pd_effort
from homie_pushcart_v52.env_smoke import make_configs
from homie_pushcart_v52.lowlevel import HimLowLevelPolicy, frozen_checkpoint
from homie_pushcart_v52.sim_env import PushCartSimEnv
from homie_pushcart_v52.task_config import task_cfg_from_run
from pushcart_bench.policies_v5x import InstrumentedEnv
from test_pushcart_parity import _core_like


class SharedControlTests(unittest.TestCase):
    def test_full_homie_history_actions_and_velocity_on_same_samples(self):
        e = InstrumentedEnv(True, scene_profile='isaac')
        e.reset()
        low = HimLowLevelPolicy(frozen_checkpoint(), 'cpu')
        history = torch.zeros(1, 456)
        # Reset plus seven transitions exercises the complete six-frame buffer.
        for step in range(8):
            if step:
                e.set_command(.1 * step, -.02, .03)
                e.step(np.full(12, .02 * step))
            core = _core_like(e)
            core.obs_buf = history
            HomieTaskCore.compute_observations(core)
            history = core.obs_buf
            mj_obs = torch.from_numpy(e.obs_hist.reshape(1, -1))
            torch.testing.assert_close(mj_obs, history, atol=2e-6, rtol=0.)
            a, v = low.act_and_estimate(mj_obs)
            b, w = low.act_and_estimate(history)
            torch.testing.assert_close(a, b, atol=2e-5, rtol=0.)
            torch.testing.assert_close(v, w, atol=2e-5, rtol=0.)

    def test_actual_isaac_and_mujoco_explicit_efforts_and_targets_agree(self):
        e = InstrumentedEnv(True, decimation=1, scene_profile='isaac', upper_drive='explicit_pd')
        e.reset()
        e.data.qvel[e.qvel_adr] = np.linspace(-.4, .4, 27)
        e.upper_delta[:] = np.linspace(-.1, .1, 15)
        legs = np.linspace(-1., 1., 12)
        upper = np.clip(e.upper_hold_actions + e.upper_delta, e.action_min[12:], e.action_max[12:])
        isaac = PushCartSimEnv.__new__(PushCartSimEnv)
        isaac.cfg = SimpleNamespace(domain_rand=SimpleNamespace(delay=False),
                                    control=SimpleNamespace(action_scale=e.action_scale))
        tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).clone()
        isaac.upper_drive, isaac.num_lower_dof = 'explicit_pd', 12
        isaac.actions = tensor(np.r_[legs, upper])[None]
        isaac.default_dof_pos = tensor(e.default_dof_pos)[None]
        isaac.dof_pos, isaac.dof_vel = tensor(e.dof_pos)[None], tensor(e.dof_vel)[None]
        isaac.p_gains, isaac.d_gains = tensor(e.p_gains), tensor(e.d_gains)
        isaac.torque_limits, isaac.joint_ids = tensor(e.torque_limits), torch.arange(27)
        isaac.Kp_factors, isaac.Kd_factors = torch.ones(1, 27), torch.ones(1, 27)
        isaac.actuation_offset, isaac.joint_injection = torch.zeros(1, 27), torch.zeros(1, 27)
        isaac._physics_substep = 0
        applied, targets = np.zeros(27), np.zeros(27)
        isaac.robot = SimpleNamespace(
            set_joint_effort_target_index=lambda target, joint_ids: applied.__setitem__(joint_ids.numpy(), target[0].numpy()),
            set_joint_position_target_index=lambda target, joint_ids: targets.__setitem__(joint_ids.numpy(), target[0].numpy()))
        PushCartSimEnv._apply_action(isaac)
        e.step(legs)
        np.testing.assert_allclose(e.data.ctrl[e.act_idx], applied, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(e.joint_pos_target[12:], targets[12:], atol=1e-10, rtol=0.)
        # Native servo bias must be disabled to prevent a second PD loop.
        np.testing.assert_array_equal(e.model.actuator_biasprm[e.act_idx], 0.)
        np.testing.assert_array_equal(e.model.actuator_gainprm[e.act_idx, 0], 1.)
        # Exercise randomized upper gains/injection through the actual new path.
        isaac.Kp_factors[:, 12:] = 1.1
        isaac.Kd_factors[:, 12:] = .9
        isaac.joint_injection[:, 12:] = .2
        PushCartSimEnv._apply_action(isaac)
        expected = pd_effort(isaac.dof_pos, isaac.dof_vel, isaac.joint_pos_target,
                             isaac.p_gains * isaac.Kp_factors, isaac.d_gains * isaac.Kd_factors,
                             isaac.torque_limits, isaac.joint_injection)
        np.testing.assert_allclose(applied, expected[0].numpy(), atol=1e-10, rtol=0.)

    def test_numpy_torch_pd_saturation_and_sign_match(self):
        q = np.array([.1, -.2, .3])
        dq = np.array([.4, -.5, .6])
        target, kp, kd, limits = np.array([2., -2., .2]), np.array([20., 30., 4.]), np.array([1., 2., .2]), np.array([3., 4., 5.])
        a = pd_effort(q, dq, target, kp, kd, limits)
        b = pd_effort(*[torch.from_numpy(x) for x in (q, dq, target, kp, kd, limits)])
        np.testing.assert_array_equal(a, b.numpy())
        np.testing.assert_allclose(a, [3., -4., -.52])

    def test_saved_config_controls_mj_model_and_observation_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'params').mkdir()
            saved = {'task': {'version': 52}, 'homie': {'control': {'action_scale': .19},
                     'normalization': {'obs_scales': {'ang_vel': .37}}},
                     'sim': {'dt': .004, 'gravity': [0., 0., -9.7]}, 'decimation': 5}
            (run / 'params/env.yaml').write_text(yaml.safe_dump(saved), encoding='utf-8')
            # Preserve class defaults while probing non-default saved parameters.
            from homie_isaaclab.config import G1RoughCfg
            original = (G1RoughCfg.control.action_scale, G1RoughCfg.normalization.obs_scales.ang_vel,
                        G1RoughCfg.sim.dt, copy.copy(G1RoughCfg.sim.gravity), G1RoughCfg.control.decimation)
            try:
                cfg, _ = make_configs(1, False, checkpoint=run)
                e = InstrumentedEnv(True, scene_profile='isaac', cfg=cfg, decimation=cfg.control.decimation)
                self.assertEqual(e.action_scale, .19)
                self.assertEqual(e.obs_scales['ang_vel'], .37)
                self.assertEqual(e.decimation, 5)
                self.assertAlmostEqual(e.model.opt.timestep, .004)
                np.testing.assert_allclose(e.model.opt.gravity, [0., 0., -9.7])
                self.assertAlmostEqual(e.dt, .02)
            finally:
                (G1RoughCfg.control.action_scale, G1RoughCfg.normalization.obs_scales.ang_vel,
                 G1RoughCfg.sim.dt, G1RoughCfg.sim.gravity, G1RoughCfg.control.decimation) = original

    def test_old_run_keeps_training_drive_and_explicit_setting_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'params').mkdir()
            f = p / 'params/env.yaml'
            f.write_text(yaml.safe_dump({'task': {'version': 52}}), encoding='utf-8')
            self.assertEqual(task_cfg_from_run(p, verbose=False).actions.upper_drive, 'training')
            f.write_text(yaml.safe_dump({'task': {'version': 52, 'actions': {'upper_drive': 'explicit_pd'}}}), encoding='utf-8')
            self.assertEqual(task_cfg_from_run(p, verbose=False).actions.upper_drive, 'explicit_pd')


if __name__ == '__main__':
    unittest.main()
