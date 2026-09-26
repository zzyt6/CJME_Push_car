"""Sensor-only V5.1 fixed-grasp FK: calibration, velocity and frame contracts."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v51.fixed_grasp import BatchedPalmKinematics, FixedGraspObserver
from homie_pushcart_v51.paths import HOMIE_JOINTS, PALM_LINKS
from homie_pushcart_v51.pose import Kinematics, load_initial_pose
from homie_pushcart_v51.assets.cart_urdf import derive_cart_pose, load_cart_params


class FixedGraspTests(unittest.TestCase):
    def observer(self, n=1, **kwargs):
        return FixedGraspObserver(n, dtype=torch.float64, **kwargs)

    def inputs(self, observer):
        q = observer.initial_joint_pos[None].expand(observer.num_envs, -1).clone()
        return q, torch.zeros_like(q), torch.zeros(observer.num_envs, 3, dtype=observer.dtype, device=observer.device)

    def test_random_fk_matches_yourdfpy_all_27_angles(self):
        fk, kin = BatchedPalmKinematics(dtype=torch.float64), Kinematics(load_meshes=False)
        rng = np.random.default_rng(4)
        q = rng.uniform(-.8, .8, (8, 27))
        p, r, jac = fk.forward(torch.tensor(q))
        for row in range(len(q)):
            kin.set_configuration(dict(zip(HOMIE_JOINTS, q[row])), (0., 0., 0.))
            for side_id, side in enumerate(("left", "right")):
                expected = kin.base_T(PALM_LINKS[side])
                np.testing.assert_allclose(p[row, side_id], expected[:3, 3], atol=1e-12)
                np.testing.assert_allclose(r[row, side_id], expected[:3, :3], atol=1e-12)
        torch.testing.assert_close(jac[..., :12], torch.zeros_like(jac[..., :12]), rtol=0, atol=0)

    def test_initial_calibration_matches_pack_usd_cart_and_handle(self):
        ob = self.observer()
        q, dq, z = self.inputs(ob)
        pose = load_initial_pose()
        rot = Rotation.from_quat(pose["robot"]["base_quat_xyzw"]).as_matrix()
        gravity = torch.tensor((rot.T @ np.array([0., 0., -1.]))[None])
        ob.update(q, dq, z, z, gravity)
        derived = derive_cart_pose(load_cart_params(), pose)
        expected_pos = rot.T @ (np.array(derived["base_pos_world"]) - pose["robot"]["base_pos"])
        expected_rot = rot.T @ Rotation.from_euler("z", derived["base_yaw"]).as_matrix()
        expected_handle = rot.T @ (np.array(derived["bar_center_world"]) - pose["robot"]["base_pos"])
        np.testing.assert_allclose(ob.cart_origin_pos_b[0], expected_pos, atol=1e-12)
        np.testing.assert_allclose(ob.cart_orientation_b[0], expected_rot, atol=1e-12)
        np.testing.assert_allclose(ob.handle_pos_b[0], expected_handle, atol=1e-12)
        self.assertTrue(ob.valid.item())
        self.assertLess(ob.position_mismatch_m.item(), 1e-12)
        self.assertLess(ob.orientation_mismatch_rad.item(), 1e-12)
        self.assertAlmostEqual(ob.rel_yaw.item(), derived["base_yaw"], places=12)

    def test_jacobian_cart_velocity_matches_joint_finite_difference(self):
        ob = self.observer(5)
        q, _, z = self.inputs(ob)
        gen = torch.Generator().manual_seed(7)
        q += .2 * torch.randn(q.shape, generator=gen, dtype=q.dtype)
        dq = torch.randn(q.shape, generator=gen, dtype=q.dtype)
        ob.update(q, dq, z, z)
        velocity, angular = ob.cart_linear_velocity_b.clone(), ob.cart_angular_velocity_b.clone()
        eps = 1e-6
        ob.update(q + eps * dq, dq, z, z)
        p_plus, r_plus = ob.cart_origin_pos_b.clone(), ob.cart_orientation_b.clone()
        ob.update(q - eps * dq, dq, z, z)
        numerical_v = (p_plus - ob.cart_origin_pos_b) / (2 * eps)
        numerical_w = np.stack([Rotation.from_matrix(m).as_rotvec() / (2 * eps)
                                for m in (r_plus @ ob.cart_orientation_b.transpose(-1, -2)).numpy()])
        torch.testing.assert_close(velocity, numerical_v, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(angular, numerical_w, atol=2e-9, rtol=2e-9)

    def test_base_angular_velocity_transports_cart_origin(self):
        ob = self.observer(3)
        q, dq, z = self.inputs(ob)
        omega = torch.tensor([[0., 0., .8], [.2, -.3, .5], [-.7, .1, 0.]], dtype=q.dtype)
        base_v = torch.tensor([[.4, -.2, .1]], dtype=q.dtype).expand(3, -1)
        ob.update(q, dq, base_v, omega)
        expected = base_v + torch.linalg.cross(omega, ob.cart_origin_pos_b)
        torch.testing.assert_close(ob.cart_linear_velocity_b, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ob.cart_angular_velocity_b, omega, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(ob.cart_twist_base_heading[0, 2].item(), .8, places=12)

    def test_tilted_cart_yaw_rate_is_heading_derivative(self):
        ob = self.observer()
        q, dq, z = self.inputs(ob)
        q[:, 13] += .4
        ob.update(q, dq, z, torch.tensor([[.5, -.3, .2]], dtype=q.dtype))
        omega = ob.cart_angular_velocity_b[0].numpy()
        r = ob.cart_orientation_b[0].numpy()
        eps = 1e-6
        after = Rotation.from_rotvec(eps * omega).as_matrix() @ r
        before = Rotation.from_rotvec(-eps * omega).as_matrix() @ r
        expected = (np.arctan2(after[1, 0], after[0, 0]) - np.arctan2(before[1, 0], before[0, 0])) / (2 * eps)
        self.assertAlmostEqual(ob.cart_twist_heading[0, 2].item(), expected, places=8)

    def test_world_yaw_invariance_and_gravity_leveling(self):
        ob = self.observer(4)
        q, dq, z = self.inputs(ob)
        roll_pitch = Rotation.from_euler("xyz", [.2, -.25, 0.]).as_matrix()
        v0, omega0 = np.array([.3, -.2, .1]), np.array([.2, .1, .5])
        rb, vb, wb, gb = [], [], [], []
        for yaw in (0., 1., -2., 2.8):
            rz = Rotation.from_euler("z", yaw).as_matrix()
            rotation = rz @ roll_pitch
            rb.append(rotation)
            vb.append(rotation.T @ (rz @ v0))
            wb.append(rotation.T @ (rz @ omega0))
            gb.append(rotation.T @ np.array([0., 0., -1.]))
        ob.update(q, dq, torch.tensor(np.array(vb)), torch.tensor(np.array(wb)), torch.tensor(np.array(gb)))
        torch.testing.assert_close(ob.cart_twist_heading, ob.cart_twist_heading[:1].expand(4, -1), atol=1e-12, rtol=1e-12)
        expected = roll_pitch @ ob.cart_linear_velocity_b[0].numpy()
        np.testing.assert_allclose(ob.cart_twist_base_heading[0, :2], expected[:2], atol=1e-12)
        np.testing.assert_allclose(ob.body_to_heading[0], roll_pitch, atol=1e-12)

    def test_dual_chain_mismatch_invalidates_without_changing_primary(self):
        ob = self.observer(2)
        q, dq, z = self.inputs(ob)
        q[1, 20] += .25
        ob.update(q, dq, z, z)
        self.assertEqual(ob.valid.tolist(), [True, False])
        torch.testing.assert_close(ob.cart_origin_pos_b[0], ob.cart_origin_pos_b[1], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ob.cart_orientation_b[0], ob.cart_orientation_b[1], atol=1e-12, rtol=1e-12)
        self.assertGreater(ob.orientation_mismatch_rad[1], .15)

    def test_reference_base_rigid_velocity_transport_sign(self):
        ob = self.observer()
        q, dq, z = self.inputs(ob)
        ob.update(q, dq, z, z)
        command = torch.tensor([[.2, -.1, .7]], dtype=q.dtype)
        base_ref = ob.reference_base(command)
        r = (ob.body_to_heading @ ob.cart_origin_pos_b[:, :, None]).squeeze(-1)
        omega = torch.stack((z[:, 0], z[:, 0], command[:, 2]), -1)
        recovered = torch.cat((base_ref[:, :2], z[:, :1]), -1) + torch.linalg.cross(omega, r)
        yaw = ob.rel_yaw.item()
        expected = Rotation.from_euler("z", yaw).as_matrix() @ command[0].numpy() * np.array([1., 1., 0.])
        np.testing.assert_allclose(recovered[0, :2], expected[:2], atol=1e-12)
        self.assertEqual(base_ref[0, 2], command[0, 2])

    def test_invalid_input_and_partial_reset_then_immediate_readiness(self):
        ob = self.observer(3)
        q, dq, z = self.inputs(ob)
        ob.update(q, dq, z, z, timestamp_s=2.)
        original = ob.cart_origin_pos_b.clone()
        ob.reset(torch.tensor([1]))
        self.assertEqual(ob.valid.tolist(), [True, False, True])
        torch.testing.assert_close(ob.cart_origin_pos_b[[0, 2]], original[[0, 2]])
        ob.update(q, dq, z, z, timestamp_s=torch.tensor([2., 3., 4.]))
        self.assertTrue(ob.ready.all())
        q[1, 14] = float("nan")
        ob.update(q, dq, z, z, valid=torch.tensor([True, True, False]))
        self.assertEqual(ob.valid.tolist(), [True, False, False])
        self.assertTrue(torch.isfinite(ob.cart_twist_heading).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_batch_matches_cpu(self):
        cpu, gpu = self.observer(64), self.observer(64, device="cuda")
        q, dq, z = self.inputs(cpu)
        q[:, 12] = torch.linspace(-.4, .4, 64, dtype=q.dtype)
        dq[:, 12] = .6
        cpu.update(q, dq, z, z)
        gpu.update(q.cuda(), dq.cuda(), z.cuda(), z.cuda())
        for name in ("palm_pos_b", "cart_orientation_b", "cart_twist_heading", "rel_yaw"):
            torch.testing.assert_close(getattr(cpu, name), getattr(gpu, name).cpu(), atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
