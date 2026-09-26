"""V5.2 observable frame contract and bounded, measurement-only recovery authority."""
from __future__ import annotations

import math
import torch

FRAME_FIELDS = (
    ('joint_position', 27), ('joint_velocity', 27), ('upper_torque', 15),
    ('imu_gyro', 3), ('projected_gravity', 3), ('him_velocity', 3),
    ('handle_position', 3), ('cart_relative_yaw', 1), ('cart_twist_estimate', 3),
    ('task_reference', 3), ('homie_command_applied', 3), ('upper_target_applied', 15),
    ('previous_high_level_action', 18),
    # V5.2 physics prior: per-hand arm-torque-equivalent wrench (J^T)^+ tau in cart axes.
    ('static_hand_wrench', 12),
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
# Handle-centre resultant [Fx, Fy, Mz]; logging/evaluation scale.
WRENCH_SCALE = (80., 80., 12.)
# One hand's [Fx, Fy, Fz, Mx, My, Mz] at its palm origin, cart-heading axes.
# Labels, estimator outputs and the static prior are [left(6), right(6)] / this.
HAND_WRENCH_SCALE = (50., 50., 50., 5., 5., 5.)
# HOMIE joint order: legs 0-11, waist 12, left arm 13-19, right arm 20-26.
ARM_JOINT_SLICES = (slice(13, 20), slice(20, 27))


def body_to_cart_heading(body_to_heading, rel_yaw):
    """[N,3,3] map pelvis-body axes -> upright cart-heading axes (the label axes)."""
    c, s = torch.cos(rel_yaw), torch.sin(rel_yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    yaw_inverse = torch.stack((c, s, zero, -s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)
    return yaw_inverse @ body_to_heading


def static_hand_wrench(jacobian_b, tau_upper, body_to_cart, damping=1.e-3, clip=(300., 30.)):
    """Per-hand wrench whose joint-space image best matches the measured arm torques.

    Solves tau_arm = J^T w by damped least squares, w = (J J^T + damping I)^-1 J tau,
    with J the [6,7] palm-origin Jacobian (linear, angular; pelvis axes) and tau
    the 15 measured upper-body torques (waist first). In static equilibrium the
    motor torque balances J^T of the hand-on-cart wrench plus arm gravity, so w
    approximates the label up to arm gravity, inertia and PD transients; the
    estimator learns that remainder. Rotated by ``body_to_cart`` and clipped.
    Returns [N,12] = [left F, left M, right F, right M], physical units.
    """
    eye = torch.eye(6, device=jacobian_b.device, dtype=jacobian_b.dtype)
    out = []
    for side, cols in enumerate(ARM_JOINT_SLICES):
        jac = jacobian_b[:, side, :, cols]                                          # [N,6,7]
        tau = tau_upper[:, 1 + 7 * side:8 + 7 * side, None]                         # [N,7,1]
        # solve_ex: no host synchronization for the (never expected) singular case.
        wrench = torch.linalg.solve_ex(jac @ jac.transpose(-1, -2) + damping * eye, jac @ tau)[0][..., 0]
        force = (body_to_cart @ wrench[:, :3, None])[..., 0].clamp(-clip[0], clip[0])
        moment = (body_to_cart @ wrench[:, 3:, None])[..., 0].clamp(-clip[1], clip[1])
        out += [force, moment]
    return torch.nan_to_num(torch.cat(out, -1), nan=0., posinf=0., neginf=0.)


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
