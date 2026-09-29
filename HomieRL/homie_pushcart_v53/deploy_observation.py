"""Simulator-free V5.3 input builder. All arguments are deployment measurements.

Visual yaw = cart heading minus upright robot heading, after camera extrinsic
calibration. No handle position, cart pose/velocity, or fixed-grasp transform is
accepted. Robot FK is used ONLY for the optional arm-torque wrench hint.
"""
from __future__ import annotations

import torch

from homie_pushcart_v52.fixed_grasp import BatchedPalmKinematics
from .observation import FRAME_DIM, HAND_WRENCH_SCALE, body_to_cart_heading, static_hand_wrench


def level_rotation(gravity):
    g = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1.e-9)
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    pitch = torch.atan2(g[:, 0], g[:, 1:].norm(dim=-1))
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    z = torch.zeros_like(cr)
    return torch.stack((cp, sp*sr, sp*cr, z, cr, -sr, -sp, cp*sr, cp*cr), -1).reshape(-1, 3, 3)


def visual_heading(yaw, valid, age_s, max_age_s=.12):
    """Discard missing/nonfinite/stale vision packets; never reconstruct from hands."""
    good = valid.bool() & torch.isfinite(yaw) & torch.isfinite(age_s) & (age_s >= 0) & (age_s <= max_age_s)
    angle = torch.where(good, torch.atan2(yaw.sin(), yaw.cos()), 0.)
    return angle, good


def command_in_base(command, yaw, valid):
    c, s = yaw.cos(), yaw.sin()
    value = torch.stack((c*command[:, 0]-s*command[:, 1], s*command[:, 0]+c*command[:, 1], command[:, 2]), -1)
    return torch.where(valid[:, None], value, 0.)


class ObservableHistory:
    def __init__(self, num_envs, history=20, device='cpu', *, q_scale=1., dq_scale=.05,
                 gyro_scale=.5, velocity_scale=2., torque_scale=.02, clip=100., max_heading_age_s=.12):
        self.buffer = torch.zeros(num_envs, history, FRAME_DIM, device=device)
        self.kinematics = BatchedPalmKinematics(device=device)
        self.q_scale, self.dq_scale, self.gyro_scale = q_scale, dq_scale, gyro_scale
        self.velocity_scale, self.torque_scale = velocity_scale, torque_scale
        self.clip, self.max_heading_age_s = clip, max_heading_age_s
        self.hand_scale = torch.tensor(HAND_WRENCH_SCALE, device=device).repeat(2)
        self.command_scale = torch.tensor([2., 2., .5], device=device)
        self.yaw = torch.zeros(num_envs, device=device)
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, ids):
        self.buffer[ids] = 0.
        self.yaw[ids] = 0.
        self.valid[ids] = False

    def push(self, *, q, dq, torque, gyro, gravity, him_velocity, relative_yaw,
             heading_valid, heading_age_s, default_q, task_reference, homie_command,
             upper_target, previous_high_action, previous_lower_action, fresh):
        n = q.shape[0]
        for name, value, width in [('q', q, 27), ('dq', dq, 27), ('torque', torque, 27),
                                   ('gyro', gyro, 3), ('gravity', gravity, 3), ('him_velocity', him_velocity, 3),
                                   ('upper_target', upper_target, 15), ('previous_high_action', previous_high_action, 18),
                                   ('previous_lower_action', previous_lower_action, 12)]:
            if value.shape != (n, width):
                raise ValueError(f'{name} must be [{n},{width}], got {tuple(value.shape)}')
        self.yaw, self.valid = visual_heading(relative_yaw, heading_valid, heading_age_s, self.max_heading_age_s)
        self.valid = self.valid & torch.isfinite(gravity).all(-1) & (gravity.norm(dim=-1) > 1.e-6)
        level = level_rotation(gravity)
        _, _, jac = self.kinematics.forward(q)
        tau = torch.where(fresh[:, None], 0., torque)
        prior = static_hand_wrench(jac, tau[:, 12:], body_to_cart_heading(level, self.yaw))
        prior = torch.where(self.valid[:, None], prior / self.hand_scale, 0.)
        heading = torch.stack((self.yaw.sin(), self.yaw.cos()), -1)
        heading = torch.where(self.valid[:, None], heading, 0.)
        frame = torch.cat(((q-default_q)*self.q_scale, dq*self.dq_scale, tau*self.torque_scale,
                           gyro*self.gyro_scale, gravity, him_velocity*self.velocity_scale,
                           heading, self.valid[:, None].float(), task_reference*self.command_scale,
                           homie_command*self.command_scale, (upper_target-default_q[..., 12:])*self.q_scale,
                           torch.where(fresh[:, None], 0., previous_high_action),
                           torch.where(fresh[:, None], 0., previous_lower_action), prior), -1)
        if frame.shape != (n, FRAME_DIM):
            raise ValueError(f'V5.3 frame must have {FRAME_DIM} channels')
        # New storage each update: an RSL rollout may still hold the previous view.
        history = torch.cat((self.buffer[:, 1:], frame[:, None]), 1)
        history[fresh, :-1] = 0.
        self.buffer = history.clamp(-self.clip, self.clip)
        return self.buffer.flatten(1)
