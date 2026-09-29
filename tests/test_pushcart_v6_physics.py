"""V6 must retain the complete V3.2 physical model and execution trajectory."""
from pathlib import Path
import inspect
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

from homie_isaaclab.config import G1RoughCfg
from homie_pushcart.mujoco_env import MujocoHomieEnv
from homie_pushcart.mujoco_hl import MujocoHighLevelRunner
from homie_pushcart_v6 import physics


class V6PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1_pushcart/mujoco/g1_pushcart.xml"))
        cls.v6 = physics.make_mujoco_env()

    def test_single_scene_and_all_usd_dependencies_are_locked(self):
        manifest = physics.physics_manifest()
        self.assertEqual(manifest["physics_id"], "v6_v32_legacy_four_spheres")
        self.assertEqual(len(manifest["asset_sha256"]), 22)
        self.assertEqual(physics.MJCF_SCENE, self.v6.xml_path)
        self.assertEqual(type(self.v6), MujocoHomieEnv)
        self.assertEqual(list(inspect.signature(physics.make_mujoco_env).parameters), ["cfg"])
        self.assertEqual(self.v6.dt, .02)
        # Detect drift before loading either simulator; no user asset is edited.
        with patch.object(physics, "_INPUT_SHA256", {physics.MJCF_SCENE: "0" * 64}):
            with self.assertRaisesRegex(ValueError, "baseline asset changed"):
                physics.validate_assets()

    def test_every_compiled_model_array_and_solver_setting_match_legacy(self):
        a, b = self.original, self.v6.model
        arrays = 0
        for name in dir(a):
            if name.startswith("_"):
                continue  # Native object addresses are not physical parameters.
            value = getattr(a, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(b, name), err_msg=name)
                arrays += 1
            elif isinstance(value, (int, float, str, bytes)):
                self.assertEqual(value, getattr(b, name), name)
        self.assertGreater(arrays, 150)
        for name in dir(a.opt):
            if name.startswith("_"):
                continue
            value = getattr(a.opt, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(b.opt, name), err_msg="opt." + name)
            elif isinstance(value, (int, float)):
                self.assertEqual(value, getattr(b.opt, name), "opt." + name)

    def test_legacy_four_spheres_and_passive_joints_are_preserved(self):
        m = self.v6.model
        for side in ("left", "right"):
            body = m.body(f"{side}_ankle_roll_link").id
            active = np.flatnonzero((m.geom_bodyid == body) & ((m.geom_contype | m.geom_conaffinity) != 0))
            self.assertEqual(len(active), 4)
            np.testing.assert_array_equal(m.geom_type[active], np.full(4, mujoco.mjtGeom.mjGEOM_SPHERE))
            np.testing.assert_array_equal(m.geom_pos[active], physics.FOOT_CONTACT_POINTS)
            np.testing.assert_array_equal(m.geom_size[active, 0], np.full(4, .005))
        self.assertEqual(m.npair, 0)
        for name, expected in {
            "left_hip_pitch_joint": (.001, .1, .01),
            "floating_base_joint": (.001, .1, .01),
            "caster_0_swivel_joint": (.05, .01, .0005),
            "wheel_0_joint": (.005, .02, .0005),
        }.items():
            adr = int(m.joint(name).dofadr[0])
            np.testing.assert_array_equal(
                [m.dof_damping[adr], m.dof_frictionloss[adr], m.dof_armature[adr]], expected)

    def test_original_action_sequence_produces_identical_states_and_torques(self):
        old, new = MujocoHomieEnv(True), physics.make_mujoco_env()
        np.testing.assert_array_equal(old.reset(), new.reset())
        hold = (old.preset_dof_pos[:12] - old.default_dof_pos[:12]) / old.action_scale
        for step in range(60):
            actions = hold + .015 * np.sin(step * .17 + np.arange(12))
            upper = .01 * np.cos(step * .11 + np.arange(15))
            old.upper_delta[:] = new.upper_delta[:] = upper
            command = (.3, .05 * np.sin(step * .2), .2 * np.cos(step * .13))
            old.set_command(*command)
            new.set_command(*command)
            obs_old, fallen_old = old.step(actions)
            obs_new, fallen_new = new.step(actions)
            for field in ("qpos", "qvel", "ctrl", "qacc", "actuator_force"):
                np.testing.assert_array_equal(getattr(old.data, field), getattr(new.data, field),
                                              err_msg=f"step {step}: {field}")
            np.testing.assert_array_equal(obs_old, obs_new)
            self.assertEqual(fallen_old, fallen_new)
            self.assertTrue(np.isfinite(new.data.qpos).all())

    def test_original_payload_algorithm_and_repeated_reset_do_not_accumulate_inertia(self):
        old, new = MujocoHomieEnv(True), physics.make_mujoco_env()
        runners = [MujocoHighLevelRunner(env, lowlevel=None, level=6) for env in (old, new)]
        base_mass = float(new.cart_params["mass"]["base_kg"])
        base_inertia = new.model.body_inertia[new.cart_id].copy()
        base_com = new.model.body_ipos.copy()
        for payload in (29., 0., 15., 29., 0.):
            for runner in runners:
                runner.set_payload(payload)
                runner.reset(caster_angle=.2)
            self.assertEqual(new.model.body_mass[new.cart_id], base_mass + payload)
            np.testing.assert_array_equal(new.model.body_inertia[new.cart_id],
                                          base_inertia * ((base_mass + payload) / base_mass))
            np.testing.assert_array_equal(new.model.body_ipos, base_com)
            for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "body_subtreemass"):
                np.testing.assert_array_equal(getattr(old.model, field), getattr(new.model, field), err_msg=field)
            for field in ("qpos", "qvel", "ctrl"):
                np.testing.assert_array_equal(getattr(old.data, field), getattr(new.data, field), err_msg=field)

    def test_physical_overrides_are_rejected(self):
        for section, field, replacement in (("control", "decimation", 2),
                                             ("control", "action_scale", .5),
                                             ("sim", "dt", .0025)):
            cfg = G1RoughCfg()
            setattr(getattr(cfg, section), field, replacement)
            with self.assertRaisesRegex(ValueError, f"{section}.{field}"):
                physics.make_mujoco_env(cfg=cfg)
        cfg = G1RoughCfg()
        cfg.control.stiffness = dict(cfg.control.stiffness, waist=100.)
        with self.assertRaisesRegex(ValueError, "control.stiffness"):
            physics.make_mujoco_env(cfg=cfg)


if __name__ == "__main__":
    unittest.main()
