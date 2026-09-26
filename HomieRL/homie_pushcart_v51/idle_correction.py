"""Causal idle-motion gate from the same signals available to the actor.

This is a motion detector, not a cart COM velocity sensor or a controller.  A
windowed least-squares slope suppresses pose noise before allowing a bounded
policy correction for a bounded interval after a commanded stop.  It never
uses privileged cart state, and initial idle cannot activate this path.
"""
from __future__ import annotations

import math

import torch


class IdleCorrectionTracker:
    """Update once after observing a control step; ``gate`` is read-only.

    ``handle_pos_b`` and ``rel_yaw`` must be the delayed/noisy actor values,
    ``base_vel_b`` the HOMIE estimate, and ``base_gyro_b`` the observed IMU in
    physical units.  The optional gravity is the observed body-frame down
    direction.  None of the inputs is a privileged cart velocity.

    Body-frame kinematics give ``v_handle = dr/dt + v_base + omega x r``.
    We integrate the last two terms and fit the slope of the resulting pseudo
    position over a window.  This averages velocities expressed in the moving
    body frame; it is sufficient to detect residual motion, not precise COM
    tracking.  Yaw uses unwrapped relative yaw plus integrated observed base
    Euler yaw rate.  No accumulated global position is needed by the actor.
    """

    def __init__(self, cfg, dt: float, num_envs: int, device="cpu", dtype=torch.float32):
        self.cfg, self.dt = cfg, float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("idle correction requires finite dt > 0")
        defaults = dict(window_s=.8, motion_on_m_s=.10, motion_off_m_s=.04,
                        yaw_on_rad_s=.22, yaw_off_rad_s=.09, activation_s=.20,
                        calm_hold_s=.4, max_gate=.25, near_zero_command=.025,
                        gate_rise_tau_s=.15, gate_fall_tau_s=.25, recovery_window_s=5.)
        for name, default in defaults.items():
            value = float(getattr(cfg, name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"idle_correction.{name} must be finite and positive")
            setattr(self, name, value)
        if not self.motion_off_m_s < self.motion_on_m_s or not self.yaw_off_rad_s < self.yaw_on_rad_s:
            raise ValueError("idle correction off thresholds must be below on thresholds")
        if self.max_gate > 1:
            raise ValueError("idle correction max_gate must not exceed 1")
        self.window_steps = max(3, int(math.ceil(self.window_s / self.dt)) + 1)
        t = torch.arange(self.window_steps, device=device, dtype=dtype) * self.dt
        centered = t - t.mean()
        self.slope_weights = centered / centered.square().sum()
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        self.history = z(num_envs, self.window_steps, 4)
        self.previous_kinematic = z(num_envs, 4)
        self.previous_yaw = z(num_envs)
        self.estimated_twist = z(num_envs, 3)
        self.on_time, self.calm_time, self.gate_level = z(num_envs), z(num_envs), z(num_envs)
        self.recovery_remaining = z(num_envs)
        self.samples = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.ready = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.active = self.ready.clone()

    def reset(self, env_ids=None):
        """Clear selected rows, including nonfinite history, without touching others."""
        if env_ids is None:
            env_ids = slice(None)
        for tensor in (self.history, self.previous_kinematic,
                       self.previous_yaw, self.estimated_twist,
                       self.on_time, self.calm_time, self.gate_level, self.samples,
                       self.ready, self.active, self.recovery_remaining):
            tensor[env_ids] = 0

    def update(self, handle_pos_b, rel_yaw, base_vel_b, base_gyro_b, projected_gravity_b=None,
               *, task_command=None):
        """Advance once from actor observations and the current task command.

        Only a commanded movement arms stop recovery.  At initial zero command,
        even a persistent biased velocity estimate cannot enable exploration.
        Omitting ``task_command`` therefore cannot arm recovery.
        """
        valid = (torch.isfinite(handle_pos_b).all(-1) & torch.isfinite(rel_yaw)
                 & torch.isfinite(base_vel_b).all(-1) & torch.isfinite(base_gyro_b).all(-1))
        self.recovery_remaining.sub_(self.dt).clamp_(min=0.)
        if task_command is not None:
            valid &= torch.isfinite(task_command).all(-1)
            commanded = torch.nan_to_num(task_command).norm(dim=-1) > self.near_zero_command
            self.recovery_remaining.masked_fill_(commanded, self.recovery_window_s)
        gyro = torch.nan_to_num(base_gyro_b)
        if projected_gravity_b is None:
            yaw_rate = gyro[:, 2]
        else:
            valid &= torch.isfinite(projected_gravity_b).all(-1)
            g = torch.nan_to_num(projected_gravity_b)
            # Euler yaw derivative; gravity fixes roll/pitch, no world orientation.
            g_norm = g.norm(dim=-1).clamp_min(1e-6)
            g = g / g_norm.unsqueeze(-1)
            denom = g[:, 1:].square().sum(-1)
            valid &= (g_norm > .5) & (denom > .25)
            yaw_rate = -(g[:, 1] * gyro[:, 1] + g[:, 2] * gyro[:, 2]) / denom.clamp_min(.25)
        pose = torch.nan_to_num(handle_pos_b)
        yaw = torch.nan_to_num(rel_yaw)
        kinematic = torch.cat((torch.nan_to_num(base_vel_b) + torch.linalg.cross(gyro, pose),
                               yaw_rate.unsqueeze(-1)), dim=-1)
        valid &= torch.isfinite(kinematic).all(-1)
        fresh = self.samples == 0
        delta_yaw = torch.atan2(torch.sin(yaw - self.previous_yaw), torch.cos(yaw - self.previous_yaw))
        increment = torch.where(fresh.unsqueeze(-1), torch.zeros_like(kinematic),
                                .5 * self.dt * (kinematic + self.previous_kinematic))
        # Rebase the previous pseudo positions into the current sample's origin
        # every step, avoiding an unbounded float32 integration over long play.
        # Translation retains r coordinates; yaw is always relative to its latest
        # wrapped measurement, so a +/-pi crossing is just one wrapped delta.
        increment[:, 3] += torch.where(fresh, 0., delta_yaw)
        self.history -= increment.unsqueeze(1)
        sample = torch.cat((pose, torch.zeros_like(yaw).unsqueeze(-1)), dim=-1)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = sample
        self.history[fresh] = sample[fresh].unsqueeze(1)
        self.previous_kinematic.copy_(kinematic)
        self.previous_yaw.copy_(yaw)
        self.samples.add_(1).clamp_(max=self.window_steps)
        self.ready.copy_(self.samples >= self.window_steps)
        # Subtract the latest point to avoid cancellation from integrated offsets.
        slope = ((self.history - self.history[:, -1:]) * self.slope_weights[None, :, None]).sum(1)
        valid &= torch.isfinite(slope).all(-1)
        self.estimated_twist.copy_(torch.stack((slope[:, 0], slope[:, 1], slope[:, 3]), -1))
        speed, yaw_speed = self.estimated_twist[:, :2].norm(dim=-1), self.estimated_twist[:, 2].abs()
        above = self.ready & ((speed >= self.motion_on_m_s) | (yaw_speed >= self.yaw_on_rad_s))
        calm = self.ready & (speed <= self.motion_off_m_s) & (yaw_speed <= self.yaw_off_rad_s)
        self.on_time = torch.where(above, self.on_time + self.dt, 0.)
        self.calm_time = torch.where(calm, self.calm_time + self.dt, 0.)
        self.active |= self.on_time >= self.activation_s - 1e-7
        self.active &= ~(self.calm_time >= self.calm_hold_s - 1e-7)
        target = self.active.to(self.gate_level.dtype) * self.max_gate
        tau = torch.where(self.active, self.gate_level.new_tensor(self.gate_rise_tau_s),
                          self.gate_level.new_tensor(self.gate_fall_tau_s))
        self.gate_level += -torch.expm1(-self.dt / tau) * (target - self.gate_level)
        self.gate_level.masked_fill_(self.gate_level < 1e-5, 0.)
        # A missing/invalid observation breaks temporal continuity and closes the gate.
        self.reset((~valid).nonzero(as_tuple=False).flatten())

    def gate(self, cmd_ref, cmd_task):
        """Return the bounded correction gate; caller takes max with its usual gate.

        Only a near-zero *task* within the commanded-stop window enables this path.
        The finite window prevents bias or action noise from sustaining idle
        exploration indefinitely.  During reference decay the
        existing command gate dominates automatically.  This read does not
        advance filters and can safely be called for telemetry more than once.
        """
        finite = torch.isfinite(cmd_task).all(-1) & torch.isfinite(cmd_ref).all(-1)
        eligibility = (1. - torch.nan_to_num(cmd_task).norm(dim=-1) / self.near_zero_command).clamp(0., 1.)
        # Fade over the end of the recovery window rather than abruptly closing.
        window = (self.recovery_remaining / self.gate_fall_tau_s).clamp(0., 1.)
        return (self.gate_level * eligibility * window * finite).unsqueeze(-1)
