"""V5.1 observable frame contract and bounded, measurement-only recovery authority."""
from __future__ import annotations

import math
import torch

FRAME_FIELDS = (
    ('joint_position', 27), ('joint_velocity', 27), ('upper_torque', 15),
    ('imu_gyro', 3), ('projected_gravity', 3), ('him_velocity', 3),
    ('handle_position', 3), ('cart_relative_yaw', 1), ('cart_twist_estimate', 3),
    ('task_reference', 3), ('homie_command_applied', 3), ('upper_target_applied', 15),
    ('previous_high_level_action', 18),
)
FRAME_DIM = sum(n for _, n in FRAME_FIELDS)
# These fields are NEVER routed to the Actor or its encoder.
PRIVILEGED_FIELDS = (
    ('cart_geometric_twist', 3), ('hand_force', 6), ('hand_moment_at_palms', 6),
    ('cart_external_wrench_at_handle', 6), ('robot_external_wrench_at_pelvis_com', 6),
    ('cart_total_mass', 1), ('payload_offset', 2), ('wheel_friction', 1),
    ('wheel_damping', 1), ('swivel_damping', 1), ('caster_sin_cos', 8),
    ('base_velocity', 3), ('hand_numerical_valid', 1), ('kinematic_valid', 1),
)
PRIVILEGED_DIM = sum(n for _, n in PRIVILEGED_FIELDS)
WRENCH_SCALE = (80., 80., 12.)


def make_proprioceptive_sample(q, dq, gyro, gravity, cfg, noise_enabled=True):
    """Simulated encoders/IMU, independent of HOMIE's large training augmentation.

    Bounds are experimental sensor assumptions, not hardware calibration. Only
    directly measurable joints/IMU are sampled; cart/root velocity truth is absent.
    """
    values = (q, dq, gyro, gravity)
    bounds = (cfg.joint_position_noise_rad, cfg.joint_velocity_noise_rad_s,
              cfg.gyro_noise_rad_s, cfg.gravity_noise)
    return tuple(value + (2. * torch.rand_like(value) - 1.) * bound
                 if noise_enabled and bound else value.clone()
                 for value, bound in zip(values, bounds))


def frame_layout():
    offset, result = 0, []
    for name, width in FRAME_FIELDS:
        result.append(dict(name=name, start=offset, dim=width))
        offset += width
    return result


class RecoveryAuthority:
    """Open bounded residual authority on observable deviations, including initial idle.

    No external-force truth, desired correction direction, or elapsed stop window
    enters this gate. Persistent observable motion keeps it active; hysteresis and
    a calm dwell suppress switching. The policy learns the correction direction.
    """
    def __init__(self, cfg, dt, num_envs, device):
        self.cfg, self.dt = cfg, float(dt)
        self.gate = torch.zeros(num_envs, 1, device=device)
        self.activation = torch.zeros(num_envs, device=device)
        self.calm = torch.zeros(num_envs, device=device)
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, ids):
        for buf in (self.gate, self.activation, self.calm, self.active):
            buf[ids] = 0

    def update(self, twist, reference, gyro, gravity, valid):
        cfg = self.cfg
        error = twist - reference
        motion = error[:, :2].norm(dim=-1)
        yaw = error[:, 2].abs()
        tilt = gravity[:, :2].norm(dim=-1) / gravity.norm(dim=-1).clamp_min(1.e-6)
        angular = gyro[:, :2].norm(dim=-1)
        triggered = ((motion > cfg.motion_on) | (yaw > cfg.yaw_on)
                     | (tilt > math.sin(cfg.tilt_on)) | (angular > cfg.gyro_on)) & valid
        calm = ((motion < cfg.motion_off) & (yaw < cfg.yaw_off)
                & (tilt < math.sin(cfg.tilt_off)) & (angular < cfg.gyro_off)) & valid
        self.activation.copy_(torch.where(triggered, self.activation + self.dt, 0.))
        self.calm.copy_(torch.where(calm, self.calm + self.dt, 0.))
        self.active |= self.activation >= cfg.activation_s
        self.active &= (self.calm < cfg.calm_hold_s) & valid
        target = self.active[:, None].float() * cfg.max_gate
        alpha = torch.where(target > self.gate, 1. - math.exp(-self.dt / cfg.rise_s),
                            1. - math.exp(-self.dt / cfg.fall_s))
        self.gate.add_(alpha * (target - self.gate))
        self.gate.masked_fill_(~valid[:, None], 0.)
        return self.gate
