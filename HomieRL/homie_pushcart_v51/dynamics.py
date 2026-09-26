"""World-frame rigid-body mechanics for V5 (torch only; no simulator import).

Wrenches are [force, moment]. Positions are actual rigid-body COM positions;
inertias are COM-centered tensors. Quaternion convention is xyzw.
"""
from __future__ import annotations

import torch


def rotation_matrix_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotate_inertia(inertia_body: torch.Tensor, link_quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate an inertia expressed in link/prim axes, NOT principal COM axes."""
    rotation = rotation_matrix_xyzw(link_quaternion)
    return rotation @ inertia_body @ rotation.transpose(-1, -2)


def shift_wrench(wrench: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.cat((wrench[..., :3], wrench[..., 3:] + torch.cross(
        source-target, wrench[..., :3], dim=-1)), -1)


def centroidal_quantities(mass, position, velocity, angular_velocity, inertia_world):
    """Sum every body, including spinning wheels, about the assembly COM.

    Inputs [N,B], [N,B,3], [N,B,3,3]. The returned assembly inertia includes
    parallel-axis terms, but angular momentum also includes relative body motion
    and therefore is generally NOT assembly inertia times root angular velocity.
    """
    total_mass = mass.sum(-1)
    com = (position * mass[..., None]).sum(-2) / total_mass[..., None].clamp_min(1.e-12)
    body_momentum = mass[..., None] * velocity
    momentum = body_momentum.sum(-2)
    offset = position-com.unsqueeze(-2)
    spin = (inertia_world @ angular_velocity.unsqueeze(-1)).squeeze(-1)
    angular = (spin + torch.cross(offset, body_momentum, dim=-1)).sum(-2)
    identity = torch.eye(3, dtype=position.dtype, device=position.device)
    parallel = mass[..., None, None] * (
        offset.square().sum(-1)[..., None, None]*identity - offset[..., :, None]*offset[..., None, :])
    return {"mass": total_mass, "com": com, "velocity": momentum/total_mass[..., None].clamp_min(1.e-12),
            "linear_momentum": momentum, "angular_momentum": angular,
            "inertia_world": (inertia_world+parallel).sum(-3)}


def compensate_subtree_wrench(incoming_world, origin_world, mass, position, angular_velocity,
                              acceleration, angular_acceleration, inertia_world, gravity):
    """Recover subtree->environment wrench from parent->subtree joint wrench.

    The incoming moment and output moment are both about ``origin_world``.
    All rigid bodies distal to that joint must be included. This acceleration
    form of Newton-Euler includes gravity moments, orbital inertia, and spin
    inertia. It uses no derivative about a moving origin, so no origin-velocity
    term is missing. Other external loads on the hand are indistinguishable from
    contact loads and remain in the output with the subtree->environment sign.
    """
    effective = mass[..., None] * (acceleration-gravity)
    spin = (inertia_world @ angular_velocity.unsqueeze(-1)).squeeze(-1)
    spin_rate = ((inertia_world @ angular_acceleration.unsqueeze(-1)).squeeze(-1)
                 + torch.cross(angular_velocity, spin, dim=-1))
    moment = (spin_rate+torch.cross(position-origin_world.unsqueeze(-2), effective, dim=-1)).sum(-2)
    return incoming_world-torch.cat((effective.sum(-2), moment), -1)


def moving_origin_wrench(momentum_rate, angular_momentum_rate, origin_velocity, linear_momentum):
    """External wrench from d(momentum about a moving origin)/dt in world axes."""
    return torch.cat((momentum_rate, angular_momentum_rate+torch.cross(
        origin_velocity, linear_momentum, dim=-1)), -1)


def contact_buffer_wrench(forces, points, counts, starts, reference, slots=32):
    """Sum ragged point-force buffers without confusing mean contact with CoP.

    ``counts``/``starts`` have [N,filters]; ``reference`` has [N,3]. Returns
    [N,6] and [N] validity. A capped/overflowed contact list is marked invalid.
    All filters are summed. The caller must select only ground partners.
    """
    if slots < 1:
        raise ValueError('contact slots must be positive')
    if counts.ndim != 2 or starts.shape != counts.shape or reference.shape != (counts.shape[0], 3):
        raise ValueError('contact counts/starts need [N,filters] and references [N,3]')
    if forces.ndim != 2 or forces.shape[-1] != 3 or points.shape != forces.shape:
        raise ValueError('contact forces and points need identical [capacity,3] shapes')
    counts, starts = counts.to(torch.long), starts.to(torch.long)
    if len(forces) == 0:
        return reference.new_zeros(len(reference), 6), torch.zeros(len(reference), device=reference.device, dtype=torch.bool)
    index = starts[..., None] + torch.arange(slots, device=starts.device)
    used = torch.arange(slots, device=starts.device) < counts[..., None]
    in_bounds = (index >= 0) & (index < len(forces))
    safe_index = index.clamp(0, max(0, len(forces)-1))
    selected_force = torch.where((used & in_bounds)[..., None], forces[safe_index], 0.)
    offset = points[safe_index]-reference[:, None, None, :]
    moment = torch.where((used & in_bounds)[..., None], torch.cross(offset, selected_force, dim=-1), 0.)
    wrench = torch.cat((selected_force.sum((1, 2)), moment.sum((1, 2))), -1)
    valid = ((counts >= 0) & (counts <= slots) &
             (((starts >= 0) & (starts+counts <= len(forces))) | (counts == 0))).all(-1)
    valid &= torch.isfinite(wrench).all(-1)
    # Filling the global buffer can indicate truncation by the PhysX reporter.
    valid &= counts.sum() < len(forces)
    return wrench, valid
