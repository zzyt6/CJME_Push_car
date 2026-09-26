"""Physical identities which must hold independently of Isaac/PhysX."""
from pathlib import Path
import math
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))
from homie_pushcart_v5.dynamics import (
    centroidal_quantities, compensate_subtree_wrench, contact_buffer_wrench,
    moving_origin_wrench, rotate_inertia, rotation_matrix_xyzw, shift_wrench,
)


class DynamicsTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=1.e-8, rtol=1.e-8)

    def test_whole_cart_keeps_wheel_mass_orbital_and_spin_momentum(self):
        mass = torch.tensor([[12., 1., 1.]], dtype=torch.float64)
        position = torch.tensor([[[0., 0., 1.], [1., 0., 0.], [-1., 0., 0.]]], dtype=torch.float64)
        velocity = torch.tensor([[[2., 0., 0.], [2., 1., 0.], [2., -1., 0.]]], dtype=torch.float64)
        omega = torch.tensor([[[0., 0., 0.], [0., 3., 0.], [0., 3., 0.]]], dtype=torch.float64)
        inertia = torch.diag_embed(torch.tensor([[[1., 2., 3.], [.1, .2, .1], [.1, .2, .1]]], dtype=torch.float64))
        state = centroidal_quantities(mass, position, velocity, omega, inertia)
        self.assertClose(state['mass'], torch.tensor([14.], dtype=torch.float64))
        self.assertClose(state['com'], torch.tensor([[0., 0., 12./14]], dtype=torch.float64))
        self.assertClose(state['linear_momentum'], torch.tensor([[28., 0., 0.]], dtype=torch.float64))
        self.assertClose(state['angular_momentum'], torch.tensor([[0., 1.2, 2.]], dtype=torch.float64))
        self.assertClose(state['inertia_world'][..., 2, 2], torch.tensor([5.2], dtype=torch.float64))

    def test_centroidal_quantities_transform_under_translation_and_rotation(self):
        torch.manual_seed(7)
        mass = torch.tensor([[1., 2., 4.]], dtype=torch.float64)
        pos, vel, omega = [torch.randn(1, 3, 3, dtype=torch.float64) for _ in range(3)]
        inertia = torch.diag_embed(torch.rand(1, 3, 3, dtype=torch.float64)+.1)
        quat = torch.tensor([.3, -.2, .1, .7], dtype=torch.float64)
        rotation = rotation_matrix_xyzw(quat)
        offset = torch.tensor([4., -3., 2.], dtype=torch.float64)
        old = centroidal_quantities(mass, pos, vel, omega, inertia)
        new = centroidal_quantities(mass, pos @ rotation.T+offset, vel @ rotation.T,
                                   omega @ rotation.T, rotation @ inertia @ rotation.T)
        for key in ('velocity', 'linear_momentum', 'angular_momentum'):
            self.assertClose(new[key], old[key] @ rotation.T)
        self.assertClose(new['com'], old['com'] @ rotation.T+offset)
        self.assertClose(new['inertia_world'], rotation @ old['inertia_world'] @ rotation.T)

    def test_inertia_uses_prim_rotation_not_com_position(self):
        inertia = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64))
        quat = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)], dtype=torch.float64)
        self.assertClose(rotate_inertia(inertia, quat), torch.diag(torch.tensor([2., 1., 4.], dtype=torch.float64)))

    def test_accelerating_unloaded_hand_removes_gravity_moment_and_dynamic_load(self):
        # A translating, offset hand needs a joint force AND moment even in free air.
        mass = torch.tensor([[2.]], dtype=torch.float64)
        origin = torch.tensor([[1., 3., 0.]], dtype=torch.float64)
        position = torch.tensor([[[1.5, 3., 0.]]], dtype=torch.float64)
        acceleration = torch.tensor([[[1., 0., 2.]]], dtype=torch.float64)
        gravity = torch.tensor([0., 0., -10.], dtype=torch.float64)
        zero = torch.zeros(1, 1, 3, dtype=torch.float64)
        inertia = torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3)
        incoming = torch.tensor([[2., 0., 24., 0., -12., 0.]], dtype=torch.float64)
        output = compensate_subtree_wrench(incoming, origin, mass, position, zero,
                                          acceleration, zero, inertia, gravity)
        self.assertClose(output, torch.zeros_like(output))
        cart_load = torch.tensor([[9., 4., 0., 1., 2., 3.]], dtype=torch.float64)
        output = compensate_subtree_wrench(incoming+cart_load, origin, mass, position, zero,
                                          acceleration, zero, inertia, gravity)
        self.assertClose(output, cart_load)

    def test_angular_acceleration_and_gyroscopic_moment_are_compensated(self):
        mass = torch.ones(1, 1, dtype=torch.float64)
        zero = torch.zeros(1, 1, 3, dtype=torch.float64)
        omega = torch.tensor([[[1., 2., 0.]]], dtype=torch.float64)
        alpha = torch.tensor([[[0., 0., 3.]]], dtype=torch.float64)
        inertia = torch.diag_embed(torch.tensor([[[1., 2., 4.]]], dtype=torch.float64))
        # I*alpha=(0,0,12); omega x I*omega=(0,0,2).
        incoming = torch.tensor([[0., 0., 0., 0., 0., 14.]], dtype=torch.float64)
        result = compensate_subtree_wrench(incoming, zero[:, 0], mass, zero, omega, zero, alpha,
                                          inertia, torch.zeros(3, dtype=torch.float64))
        self.assertClose(result, torch.zeros_like(result))

    def test_subtree_compensation_is_frame_and_reference_equivariant(self):
        mass = torch.tensor([[1., 2.]], dtype=torch.float64)
        origin = torch.tensor([[.2, .3, -.1]], dtype=torch.float64)
        position = torch.tensor([[[0., 1., 0.], [1., 0., 1.]]], dtype=torch.float64)
        omega = torch.tensor([[[1., 2., 3.], [3., 2., 1.]]], dtype=torch.float64)
        acc = torch.tensor([[[3., 0., 2.], [2., -1., 1.]]], dtype=torch.float64)
        alpha = omega*.3
        inertia = torch.diag_embed(torch.tensor([[[.2, .4, .5], [.1, .3, .4]]], dtype=torch.float64))
        gravity = torch.tensor([0., 0., -9.81], dtype=torch.float64)
        incoming = torch.tensor([[2., 5., 30., 4., -3., 2.]], dtype=torch.float64)
        base = compensate_subtree_wrench(incoming, origin, mass, position, omega, acc, alpha, inertia, gravity)
        q = torch.tensor([.4, -.2, .1, .8], dtype=torch.float64)
        r = rotation_matrix_xyzw(q)
        t = torch.tensor([7., 3., -2.], dtype=torch.float64)
        rotate_wrench = lambda w: torch.cat((w[..., :3] @ r.T, w[..., 3:] @ r.T), -1)
        changed = compensate_subtree_wrench(rotate_wrench(incoming), origin @ r.T+t, mass,
                                            position @ r.T+t, omega @ r.T, acc @ r.T, alpha @ r.T,
                                            r @ inertia @ r.T, gravity @ r.T)
        self.assertClose(changed, rotate_wrench(base))
        new_origin = origin+torch.tensor([[.5, -.1, .2]], dtype=torch.float64)
        changed = compensate_subtree_wrench(shift_wrench(incoming, origin, new_origin), new_origin,
                                            mass, position, omega, acc, alpha, inertia, gravity)
        self.assertClose(changed, shift_wrench(base, origin, new_origin))

    def test_palm_anchor_cart_reference_shifts_are_identical(self):
        wrench = torch.tensor([[10., 2., 0., 1., 2., 3.]], dtype=torch.float64)
        palm = torch.tensor([[0., .2, 1.]], dtype=torch.float64)
        anchor = palm+torch.tensor([[.0585, .03, 0.]], dtype=torch.float64)
        cart = torch.tensor([[.5, 0., .4]], dtype=torch.float64)
        self.assertClose(shift_wrench(shift_wrench(wrench, palm, anchor), anchor, cart),
                         shift_wrench(wrench, palm, cart))

    def test_moving_origin_transport_term_restores_zero_external_moment(self):
        p = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
        origin_velocity = torch.tensor([[0., 3., 0.]], dtype=torch.float64)
        d_angular = torch.tensor([[0., 0., 6.]], dtype=torch.float64)
        wrench = moving_origin_wrench(torch.zeros_like(p), d_angular, origin_velocity, p)
        self.assertClose(wrench, torch.zeros_like(wrench))

    def test_contact_moment_uses_each_force_point_not_average_point(self):
        forces = torch.tensor([[0., 0., 1.], [0., 0., 3.], [0., 0., 0.]], dtype=torch.float64)
        points = torch.tensor([[-1., 0., 0.], [1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
        result, valid = contact_buffer_wrench(forces, points, torch.tensor([[2]]), torch.tensor([[0]]),
                                              torch.zeros(1, 3, dtype=torch.float64))
        self.assertTrue(valid.item())
        self.assertClose(result, torch.tensor([[0., 0., 4., 0., -2., 0.]], dtype=torch.float64))

    def test_contact_overflow_is_invalid_and_empty_stale_nan_does_not_leak(self):
        forces = torch.full((3, 3), float('nan'), dtype=torch.float64)
        points = forces.clone()
        result, valid = contact_buffer_wrench(forces, points, torch.tensor([[0]]), torch.tensor([[0]]),
                                              torch.zeros(1, 3, dtype=torch.float64))
        self.assertTrue(valid.item())
        self.assertClose(result, torch.zeros_like(result))
        _, valid = contact_buffer_wrench(torch.ones_like(forces), torch.zeros_like(points),
                                         torch.tensor([[3]]), torch.tensor([[0]]),
                                         torch.zeros(1, 3, dtype=torch.float64))
        self.assertFalse(valid.item())

    def test_malformed_contact_ranges_and_absent_capacity_cannot_be_valid(self):
        f = torch.zeros(4, 3)
        ref = torch.zeros(1, 3)
        for count, start in ((1, -1), (-1, 0), (3, 2)):
            with self.subTest(count=count, start=start):
                _, valid = contact_buffer_wrench(f, f, torch.tensor([[count]]), torch.tensor([[start]]), ref)
                self.assertFalse(valid.item())
        result, valid = contact_buffer_wrench(f[:0], f[:0], torch.tensor([[0]]), torch.tensor([[0]]), ref)
        self.assertFalse(valid.item())
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_one_bad_contact_invalidates_only_its_environment_before_global_overflow(self):
        force = torch.tensor([[0., 0., float('nan')], [0., 0., 5.], [0., 0., 0.], [0., 0., 0.]])
        result, valid = contact_buffer_wrench(force, torch.zeros_like(force), torch.tensor([[1], [1]]),
                                              torch.tensor([[0], [1]]), torch.zeros(2, 3))
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(torch.isfinite(result[1]).all())


if __name__ == '__main__':
    unittest.main()
