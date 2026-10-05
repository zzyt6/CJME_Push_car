"""Appearance changes must not alter the fixed V3/V6 MuJoCo physics."""
from contextlib import nullcontext
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

from homie_pushcart.mujoco_teleop import MujocoViserRenderer


class FakeHandle:
    def __init__(self, name, **kwargs):
        self.name = name
        self.visible = True
        self.removed = False
        self.remove_count = 0
        self.__dict__.update(kwargs)

    def remove(self):
        self.removed = True
        self.remove_count += 1


class FakeScene:
    def __init__(self):
        self.nodes = {}

    def _add(self, name, **kwargs):
        handle = FakeHandle(name, **kwargs)
        self.nodes[name] = handle
        return handle

    def add_grid(self, name, **kwargs):
        return self._add(name, **kwargs)

    def add_frame(self, name, **kwargs):
        return self._add(name, **kwargs)

    def add_mesh_simple(self, name, vertices, faces, **kwargs):
        return self._add(name, vertices=vertices, faces=faces, **kwargs)


class FakeServer:
    def __init__(self):
        self.scene = FakeScene()

    def atomic(self):
        return nullcontext()


class MujocoVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/g1_pushcart/mujoco/g1_pushcart.xml"))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)
        cls.cart_geoms = {
            g for g in range(cls.model.ngeom)
            if cls.model.body(int(cls.model.geom_bodyid[g])).name.startswith(("cart_", "wheel_", "caster_"))
        }

    def setUp(self):
        self.server = FakeServer()
        self.renderer = MujocoViserRenderer(self.server, self.model)

    def test_cart_uses_only_the_original_complete_visual_mesh_and_is_opaque(self):
        cart_visual = self.model.geom("cart_visual").id
        self.assertEqual(set(self.renderer.handles) & self.cart_geoms, {cart_visual})
        self.assertFalse(set(self.renderer.collision_handles) & self.cart_geoms)
        handle = self.renderer.handles[cart_visual]
        self.assertEqual(handle.name, "/mj/cart_visual")
        self.assertIsNone(handle.opacity)  # Viser's fully opaque material.
        self.assertAlmostEqual(float(self.model.geom_rgba[cart_visual, 3]), .45, places=6)
        mid = int(self.model.geom_dataid[cart_visual])
        self.assertEqual(len(handle.faces), self.model.mesh_facenum[mid])
        # The original OBJ has all four wheels; no primitive replacement is needed.
        xyz = (handle.vertices @ self.data.geom_xmat[cart_visual].reshape(3, 3).T
               + self.data.geom_xpos[cart_visual] - self.data.xpos[self.model.body("cart_base").id])
        for sign_x in (-1, 1):
            for sign_y in (-1, 1):
                wheel = xyz[(xyz[:, 2] < .11) & (xyz[:, 0] * sign_x > .35)
                            & (xyz[:, 1] * sign_y > .15)]
                self.assertGreater(len(wheel), 100)
                self.assertLess(wheel[:, 2].min(), .01)

    def test_collision_toggle_only_affects_robot_collision_geoms(self):
        self.assertGreater(len(self.renderer.collision_handles), 0)
        for visible in (True, False, True):
            self.renderer.set_collision_visible(visible)
            self.assertTrue(all(h.visible == visible for h in self.renderer.collision_handles.values()))
            self.assertTrue(all(h.visible for h in self.renderer.handles.values()))
            self.assertFalse(set(self.renderer.collision_handles) & self.cart_geoms)
        with_collisions = MujocoViserRenderer(FakeServer(), self.model, show_robot_collision=True)
        self.assertTrue(all(h.visible for h in with_collisions.collision_handles.values()))
        self.assertFalse(set(with_collisions.collision_handles) & self.cart_geoms)

    def test_render_update_and_remove_leave_every_model_array_unchanged(self):
        arrays = {name: value.copy() for name in dir(self.model) if not name.startswith("_")
                  and isinstance(value := getattr(self.model, name), np.ndarray)}
        self.assertGreater(len(arrays), 150)
        renderer = MujocoViserRenderer(FakeServer(), self.model, show_robot_collision=True)
        renderer.update(self.data)
        for g, handle in renderer.handles.items():
            np.testing.assert_array_equal(handle.position, self.data.geom_xpos[g])
        renderer.set_collision_visible(False)
        renderer.remove()
        for name, original in arrays.items():
            np.testing.assert_array_equal(original, getattr(self.model, name), err_msg=name)

    def test_remove_cleans_all_nodes_once_for_model_switching(self):
        handles = list(self.server.scene.nodes.values())
        self.renderer.remove()
        self.renderer.remove()
        self.assertTrue(all(handle.removed and handle.remove_count == 1 for handle in handles))
        self.assertEqual(self.renderer.handles, {})
        self.assertEqual(self.renderer.collision_handles, {})
        self.assertEqual(self.renderer._scene_handles, [])


if __name__ == "__main__":
    unittest.main()
