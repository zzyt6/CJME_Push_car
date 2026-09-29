"""The alternate foot contacts must not change the robot, drives or cart grasp."""
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
from homie_isaaclab.config import G1RoughCfg
from pushcart_bench.sim2sim_scene import build_isaac_scene
from pushcart_bench.policies_v5x import InstrumentedEnv


class SceneVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = G1RoughCfg()
        cls.original_path = build_isaac_scene(cfg, 'explicit_pd', 'urdf')
        cls.new_path = build_isaac_scene(cfg, 'explicit_pd', 'urdf', 'four_spheres')
        cls.original = mujoco.MjModel.from_xml_path(str(cls.original_path))
        cls.new = mujoco.MjModel.from_xml_path(str(cls.new_path))

    def test_original_is_default_and_both_versions_keep_the_same_dynamics_and_grasp(self):
        self.assertEqual(self.original_path, build_isaac_scene(G1RoughCfg(), 'explicit_pd', 'urdf', 'original'))
        self.assertNotEqual(self.original_path, self.new_path)
        self.assertEqual(self.new_path, build_isaac_scene(G1RoughCfg(), 'explicit_pd', 'urdf', 'new'))
        a, b = self.original, self.new
        self.assertEqual((a.nq, a.nv, a.nu, a.neq), (b.nq, b.nv, b.nu, b.neq))
        for field in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'jnt_range', 'jnt_actfrcrange',
                      'dof_damping', 'dof_frictionloss', 'dof_armature', 'actuator_gainprm', 'actuator_biasprm',
                      'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_data', 'eq_solref', 'eq_solimp'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
        self.assertEqual(a.opt.integrator, b.opt.integrator)
        self.assertEqual(a.opt.timestep, b.opt.timestep)

    def test_contact_support_changes_but_floor_friction_and_cart_pairs_do_not(self):
        a, b = self.original, self.new
        for side in ('left', 'right'):
            foot = f'{side}_ankle_roll_link'
            active = lambda m: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(foot).id
                                and (m.geom_contype[g] or m.geom_conaffinity[g])]
            self.assertEqual(len(active(a)), 1)
            self.assertEqual(int(a.geom_type[active(a)[0]]), int(mujoco.mjtGeom.mjGEOM_MESH))
            self.assertEqual(len(active(b)), 4)
            for g in active(b):
                self.assertEqual(int(b.geom_type[g]), int(mujoco.mjtGeom.mjGEOM_SPHERE))
                self.assertEqual(b.geom_size[g, 0], .005)
                pair = b.pair(b.geom(g).name + '_floor').id
                np.testing.assert_array_equal(b.pair_friction[pair], [.75, .75, 0., 0., 0.])
        for i in range(4):
            name = f'wheel_floor_{i}'
            np.testing.assert_array_equal(a.pair_friction[a.pair(name).id], b.pair_friction[b.pair(name).id])
        env = InstrumentedEnv(True, scene_profile='isaac', upper_drive='explicit_pd', scene_version='new')
        self.assertEqual(env.scene_version, 'four_spheres')
        self.assertEqual(env.xml_path, self.new_path)


if __name__ == '__main__': unittest.main()
