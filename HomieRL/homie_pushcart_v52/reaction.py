"""Causal V5 task-wrench model and bounded reaction recovery at the HOMIE port.

Forces/velocities use common **cart-heading** axes (world vertical, cart yaw);
hand wrenches are ROBOT-ON-CART, summed about the whole-cart COM. A positive hand
Fx therefore exerts a negative Fx on the robot.  This module is a planar model
and a velocity-reference compensator, not a whole-body force/ZMP controller.

The force input is a virtual wrist measurement.  Delay, noise, saturation and a
causal low-pass are applied before it enters either the controller or actor.
Mass/inertia inputs are configured/identified estimates, not an external-force
oracle.  No wheel force, disturbance schedule, future sample, robot momentum,
or simulator contact truth is required.  Cart velocity must be an observable
estimate if this interface is deployed on hardware.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch


ACTOR_FEATURE_DIM = 17
DEFAULT_REACTION_CFG = {
    # Virtual wrist: seconds, samples, N and Nm. Zero noise is useful for QA.
    "sensor_delay_steps": 1,
    "sensor_filter_tau_s": .06,
    "sensor_force_noise_std_n": .7,
    "sensor_moment_noise_std_nm": .07,
    "sensor_force_bias_std_n": .3,
    "sensor_moment_bias_std_nm": .03,
    "sensor_force_clip_n": 500.,
    "sensor_moment_clip_nm": 100.,
    # Approximate planar dynamics: [x, y, yaw]. Resistance opposes motion,
    # so the hand target contains +b*v + r*tanh(v/epsilon).
    "reference_accel_tau_s": .10,
    "velocity_gain": (1.5, 1.5, 1.5),
    "acceleration_limit": (1.2, .8, 1.5),
    "viscous_resistance": (2., 4., .3),
    "rolling_resistance": (2., 3., .25),
    "resistance_velocity_scale": (.05, .05, .10),
    "uncertainty_abs": (8., 8., 1.5),
    "uncertainty_fraction": .25,
    # Motion-confirmed recovery at the available velocity-command interface.
    # Force alone cannot drive into a stiff object: only amplify a measured
    # restoring velocity correction with matching sign and bounded magnitude.
    "force_admittance_gain": (.004, .004, .015),
    "force_motion_cap_gain": (.5, .5, .5),
    "velocity_recovery_gain": (.25, .25, .2),
    "correction_limit": (.12, .08, .15),
    "correction_tau_s": .18,
    "correction_rate_limit": (.5, .4, .6),
    "compensation_enabled": True,
    # Finite static recovery: persistent sensor bias cannot keep walking on.
    "near_zero_command": .025,
    "gate_force_on": (12., 12., 2.),
    "gate_velocity_on": (.12, .12, .30),
    "gate_tilt_on_rad": .12,
    "gate_gyro_on_rad_s": .35,
    "gate_max": .25,
    "gate_activation_s": .06,
    "gate_window_s": 2.5,
    "gate_calm_rearm_s": .40,
    "gate_rise_tau_s": .10,
    "gate_fall_tau_s": .25,
    # Costs are proxies; support is an estimated friction demand + tilt cost.
    "motion_velocity_deadzone": (.12, .12, .20),
    "motion_velocity_scale": (.30, .30, .40),
    "motion_gyro_deadzone_rad_s": .35,
    "motion_gyro_scale_rad_s": .8,
    "support_friction_estimate": .6,
    "support_tilt_deadzone_rad": .15,
    "wrench_scale": (80., 80., 12.),
    "sensor_scale": (80., 80., 80., 12., 12., 12.),
    "feature_clip": 5.,
    "cost_clip": 25.,
}


class ReactionController:
    """One vectorized update per high-level control step, CPU or CUDA.

    ``update`` returns ``self``. Public tensors retain storage across updates:
    required_wrench/aligned_required_wrench/excess_wrench [N,3],
    filtered_wrench [N,6], command_correction [N,3], recovery_gate/sensor_valid
    [N,1], actor_features [N,17], support_utilization [N], and ``costs`` with
    task_wrench/reaction_motion/support [N]. ``required_wrench`` is the current
    model target; ``aligned_required_wrench`` has exactly the sensor's delay
    and filter. Residuals compare measurements and targets at the SAME time.

    Features: [filtered wrench / sensor_scale (6), aligned target / wrench_scale
    (3), excess / wrench_scale (3), correction / limit (3), gate / gate_max (1),
    validity (1)]. No reward-only truth is appended to actor features.

    All numeric defaults can be overridden by a mapping, dataclass or namespace.
    Constant drift is bounded by a finite static event window, and the gate
    rearms only after a calm interval. Holding a nominal moving command permits
    continuous bounded compensation. There is no accumulating position integral.
    """

    def __init__(self, cfg, dt: float, num_envs: int, device="cpu", dtype=torch.float32):
        self.dt, self.num_envs = float(dt), int(num_envs)
        if not math.isfinite(self.dt) or self.dt <= 0 or self.num_envs <= 0:
            raise ValueError("reaction controller requires finite dt > 0 and num_envs > 0")
        if not dtype.is_floating_point:
            raise ValueError("reaction controller requires a floating point dtype")
        self.device, self.dtype = torch.device(device), dtype
        get = cfg.get if isinstance(cfg, Mapping) else lambda k, d: getattr(cfg, k, d)
        nonnegative = {"sensor_force_noise_std_n", "sensor_moment_noise_std_nm",
                       "sensor_force_bias_std_n", "sensor_moment_bias_std_nm",
                       "uncertainty_fraction", "viscous_resistance", "rolling_resistance",
                       "force_admittance_gain", "force_motion_cap_gain", "velocity_recovery_gain", "velocity_gain",
                       "motion_velocity_deadzone", "motion_gyro_deadzone_rad_s",
                       "support_tilt_deadzone_rad"}
        for name, default in DEFAULT_REACTION_CFG.items():
            value = get(name, default)
            if name == "compensation_enabled":
                if not isinstance(value, bool):
                    raise ValueError(f"reaction.{name} must be boolean")
            elif name == "sensor_delay_steps":
                if isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 100:
                    raise ValueError("reaction.sensor_delay_steps must be an integer in [0,100]")
                value = int(value)
            elif isinstance(default, tuple):
                try:
                    values = tuple(float(x) for x in value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reaction.{name} requires {len(default)} numeric values") from exc
                if len(values) != len(default) or any(not math.isfinite(x) or
                        (x < 0 if name in nonnegative else x <= 0) for x in values):
                    raise ValueError(f"reaction.{name} has invalid values")
                value = torch.tensor(values, device=self.device, dtype=dtype)
            else:
                value = float(value)
                if not math.isfinite(value) or (value < 0 if name in nonnegative else value <= 0):
                    raise ValueError(f"reaction.{name} has invalid value")
            setattr(self, name, value)
        if self.gate_max > 1:
            raise ValueError("reaction.gate_max must not exceed one")
        z = lambda *shape: torch.zeros(*shape, device=self.device, dtype=dtype)
        n = self.num_envs
        self.required_wrench = z(n, 3)
        self.aligned_required_wrench = z(n, 3)
        self.excess_wrench = z(n, 3)
        self.filtered_wrench = z(n, 6)
        self.command_correction = z(n, 3)
        self.recovery_gate = z(n, 1)
        self.sensor_valid = z(n, 1)
        self.actor_features = z(n, ACTOR_FEATURE_DIM)
        self.support_utilization = z(n)
        self.costs = {name: z(n) for name in ("task_wrench", "reaction_motion", "support")}
        self._history = z(n, self.sensor_delay_steps + 1, 9)
        self._previous_reference = z(n, 3)
        self._reference_accel = z(n, 3)
        self._sensor_bias = z(n, 6)
        self._activation_time = z(n)
        self._calm_time = z(n)
        self._recovery_remaining = z(n)
        self._samples = torch.zeros(n, device=self.device, dtype=torch.long)
        self._armed = torch.ones(n, device=self.device, dtype=torch.bool)
        self._zero_state = (self.required_wrench, self.aligned_required_wrench,
                            self.excess_wrench, self.filtered_wrench, self.command_correction,
                            self.recovery_gate, self.sensor_valid, self.actor_features,
                            self.support_utilization, self._history, self._previous_reference,
                            self._reference_accel, self._activation_time, self._calm_time,
                            self._recovery_remaining, self._samples, *self.costs.values())
        self.reset()

    def reset(self, ids=None):
        """Clear all temporal state for selected environments and resample bias."""
        ids = slice(None) if ids is None else ids
        for tensor in self._zero_state:
            tensor[ids] = 0
        self._armed[ids] = True
        selected = self._sensor_bias[ids]
        std = selected.new_tensor([self.sensor_force_bias_std_n] * 3 +
                                  [self.sensor_moment_bias_std_nm] * 3)
        self._sensor_bias[ids] = torch.randn_like(selected) * std

    def _matrix(self, tensor, name, width):
        tensor = torch.as_tensor(tensor, device=self.device, dtype=self.dtype)
        if tensor.shape != (self.num_envs, width):
            raise ValueError(f"{name} must have shape [{self.num_envs},{width}]")
        return tensor

    def _vector(self, tensor, name):
        tensor = torch.as_tensor(tensor, device=self.device, dtype=self.dtype)
        if tensor.shape == (self.num_envs, 1):
            tensor = tensor[:, 0]
        if tensor.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape [{self.num_envs}]")
        return tensor

    @staticmethod
    def _deadzone(value, width):
        return value.sign() * (value.abs() - width).clamp_min(0.)

    @staticmethod
    def _rotate_xy_(value, delta):
        """Re-express vector history from previous into current heading axes."""
        c, s = torch.cos(delta), torch.sin(delta)
        while c.ndim < value.ndim - 1:
            c, s = c.unsqueeze(-1), s.unsqueeze(-1)
        x, y = value[..., 0].clone(), value[..., 1].clone()
        value[..., 0] = c * x + s * y
        value[..., 1] = -s * x + c * y

    @torch.no_grad()
    def update(self, hand_force_b, hand_moment_b, cart_velocity_b, command_ref,
               robot_velocity_b, robot_omega_b, gravity_b, cart_mass,
               cart_yaw_inertia, robot_mass, valid, *, heading_delta_rad=None):
        """Advance from a sample in cart-heading axes; return this controller.

        Velocity layouts: cart [vx,vy,wz], robot [vx,vy,vz], gyro [wx,wy,wz].
        ``robot_velocity_b`` is the cart-equivalent robot velocity: subtract
        omega_ref x (robot_point-cart_COM) before calling, so nominal rigid
        turning motion at different points is not mistaken for task error.
        ReactionObservation applies this correction from observed geometry;
        the privileged reward path uses actual COM geometry independently.
        ``gravity_b`` is an IMU tilt cue: the projected body down direction
        (unit or m/s^2), with its horizontal components rotated by the robot/cart
        relative yaw. This retains body tilt relative to desired upright; it is
        NOT actual world gravity expressed in the level heading frame (which
        would always be [0,0,-g]). Only tilt detection uses this cue. ``valid``
        marks the complete sample, not force magnitude.
        Nonfinite/invalid rows are cleared immediately, including delay history.

        Required planar acceleration includes reference slew, rotating-frame
        transport, AND recovery feedback K*(v_ref-v_cart). Thus useful recovery
        force is not automatically classified as excess. Resistance is a
        configured independent model; it is never reconstructed from hand force.
        ``heading_delta_rad`` is the observed change of the measurement heading
        since the previous sample. All vector histories are transported into
        current axes before any derivative, delay or filter. If omitted, a
        first-order yaw-rate estimate cart_velocity_b[:,2]*dt is used.
        """
        names = ("hand_force_b", "hand_moment_b", "cart_velocity_b", "command_ref",
                 "robot_velocity_b", "robot_omega_b", "gravity_b")
        matrices = [self._matrix(value, name, 3) for value, name in zip(
            (hand_force_b, hand_moment_b, cart_velocity_b, command_ref,
             robot_velocity_b, robot_omega_b, gravity_b), names)]
        force, moment, cart, ref, robot, gyro, gravity = matrices
        mass = self._vector(cart_mass, "cart_mass")
        inertia = self._vector(cart_yaw_inertia, "cart_yaw_inertia")
        rmass = self._vector(robot_mass, "robot_mass")
        valid = self._vector(valid, "valid")
        good = torch.isfinite(valid) & (valid > .5)
        for value in matrices:
            good &= torch.isfinite(value).all(-1)
        for value in (mass, inertia, rmass):
            good &= torch.isfinite(value) & (value > 0.)
        gravity_norm = gravity.norm(dim=-1)
        good &= (gravity_norm > .5) & (gravity_norm < 15.)
        # Sanitise before arithmetic; a broken row must never contaminate a batch.
        force, moment, cart, ref, robot, gyro, gravity = [torch.nan_to_num(x) for x in matrices]
        mass, inertia, rmass = [torch.nan_to_num(x).clamp(1e-3, 1e5) for x in (mass, inertia, rmass)]
        fresh = self._samples == 0
        heading_delta = (cart[:, 2] * self.dt if heading_delta_rad is None else
                         self._vector(heading_delta_rad, "heading_delta_rad"))
        good &= torch.isfinite(heading_delta)
        heading_delta = torch.where(fresh, 0., torch.nan_to_num(heading_delta))
        for value in (self.filtered_wrench[:, :3], self.filtered_wrench[:, 3:],
                      self.aligned_required_wrench, self.command_correction,
                      self._previous_reference, self._reference_accel,
                      self._history[:, :, :3], self._history[:, :, 3:6], self._history[:, :, 6:]):
            self._rotate_xy_(value, heading_delta)
        transport = torch.stack((-cart[:, 2] * ref[:, 1], cart[:, 2] * ref[:, 0],
                                 torch.zeros_like(cart[:, 2])), dim=-1)
        # Covariant finite difference already includes omega x v. Adding an
        # extra transport term after filtering would double-count frame motion
        # and create fictitious acceleration for a world-fixed reference.
        reference_derivative = torch.where(fresh[:, None], transport,
                                          (ref - self._previous_reference) / self.dt)
        derivative_alpha = -math.expm1(-self.dt / self.reference_accel_tau_s)
        self._reference_accel.copy_(torch.where(fresh[:, None], transport, self._reference_accel))
        self._reference_accel.lerp_(reference_derivative, derivative_alpha)
        self._previous_reference.copy_(ref)
        accel = self._reference_accel + self.velocity_gain * (ref - cart)
        accel = accel.clamp(-self.acceleration_limit, self.acceleration_limit)
        generalized_mass = torch.stack((mass, mass, inertia), -1)
        resistance = self.viscous_resistance * cart + self.rolling_resistance * torch.tanh(
            cart / self.resistance_velocity_scale)
        self.required_wrench.copy_(generalized_mass * accel + resistance)

        sensor = torch.cat((force, moment), -1)
        std = sensor.new_tensor([self.sensor_force_noise_std_n] * 3 +
                                [self.sensor_moment_noise_std_nm] * 3)
        sensor = sensor + self._sensor_bias + torch.randn_like(sensor) * std
        limits = sensor.new_tensor([self.sensor_force_clip_n] * 3 + [self.sensor_moment_clip_nm] * 3)
        sensor = sensor.clamp(-limits, limits)
        if self.sensor_delay_steps:
            self._history[:, :-1].copy_(self._history[:, 1:].clone())
        self._history[:, -1] = torch.cat((sensor, self.required_wrench), -1)
        self._samples.add_(1).clamp_(max=self.sensor_delay_steps + 2)
        ready = good & (self._samples > self.sensor_delay_steps)
        delayed = self._history[:, 0]
        alpha = -math.expm1(-self.dt / self.sensor_filter_tau_s)
        self.filtered_wrench.lerp_(delayed[:, :6], alpha)
        self.aligned_required_wrench.lerp_(delayed[:, 6:], alpha)
        planar_sensor = self.filtered_wrench[:, [0, 1, 5]]
        self.excess_wrench.copy_(planar_sensor - self.aligned_required_wrench)
        uncertainty = self.uncertainty_abs + self.uncertainty_fraction * self.aligned_required_wrench.abs()
        error = self._deadzone(self.excess_wrench, uncertainty)

        robot_twist = torch.stack((robot[:, 0], robot[:, 1], gyro[:, 2]), dim=-1)
        velocity_error = ref - robot_twist
        g = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # atan2 remains well-defined across large tilt (unlike asin(gx)).
        tilt = torch.atan2(g[:, :2].norm(dim=-1), -g[:, 2])
        force_score = (error.abs() / self.gate_force_on).amax(-1)
        velocity_score = (velocity_error.abs() / self.gate_velocity_on).amax(-1)
        gyro_score = gyro[:, :2].norm(dim=-1) / self.gate_gyro_on_rad_s
        score = torch.stack((force_score, velocity_score, gyro_score,
                             tilt / self.gate_tilt_on_rad), -1).amax(-1)
        above, calm = ready & (score >= 1.), ready & (score < .5)
        self._activation_time.copy_(torch.where(above, self._activation_time + self.dt, 0.))
        self._calm_time.copy_(torch.where(calm, self._calm_time + self.dt, 0.))
        self._recovery_remaining.sub_(self.dt).clamp_(min=0.)
        # Rearming requires both window expiry and an actually quiet measurement.
        self._armed |= ((self._calm_time >= self.gate_calm_rearm_s) &
                        (self._recovery_remaining <= 0.))
        trigger = self._armed & (self._activation_time >= self.gate_activation_s - 1e-7)
        self._recovery_remaining.masked_fill_(trigger, self.gate_window_s)
        self._armed &= ~trigger
        window = (self._recovery_remaining / self.gate_fall_tau_s).clamp(0., 1.)
        target_gate = self.gate_max * window * (~calm).to(self.dtype)
        tau = torch.where(target_gate > self.recovery_gate[:, 0],
                          self.recovery_gate.new_tensor(self.gate_rise_tau_s),
                          self.recovery_gate.new_tensor(self.gate_fall_tau_s))
        self.recovery_gate[:, 0] += -torch.expm1(-self.dt / tau) * (
            target_gate - self.recovery_gate[:, 0])
        self.recovery_gate.masked_fill_(self.recovery_gate < 1e-5, 0.)

        moving = ref.norm(dim=-1) > self.near_zero_command
        authority = torch.where(moving, 1., self.recovery_gate[:, 0] / self.gate_max)
        force_correction = self.force_admittance_gain * error
        force_cap = self.force_motion_cap_gain * velocity_error.abs()
        force_correction = force_correction.clamp(-force_cap, force_cap)
        force_correction = torch.where(force_correction * velocity_error > 0., force_correction, 0.)
        target_correction = force_correction + self.velocity_recovery_gain * velocity_error
        target_correction = target_correction.clamp(-self.correction_limit, self.correction_limit)
        target_correction *= (authority * ready.to(self.dtype))[:, None]
        if not self.compensation_enabled:
            target_correction.zero_()
        delta = -math.expm1(-self.dt / self.correction_tau_s) * (
            target_correction - self.command_correction)
        max_step = self.correction_rate_limit * self.dt
        self.command_correction.add_(delta.clamp(-max_step, max_step))
        self.command_correction.clamp_(-self.correction_limit, self.correction_limit)
        self.command_correction.masked_fill_(self.command_correction.abs() < 1e-6, 0.)

        self.costs["task_wrench"].copy_((error / self.wrench_scale).square().sum(-1))
        robot_reference = torch.cat((ref[:, :2], torch.zeros_like(ref[:, :1])), -1)
        linear_error = self._deadzone(robot - robot_reference, self.motion_velocity_deadzone)
        angular_error = self._deadzone(gyro[:, :2], self.motion_gyro_deadzone_rad_s)
        self.costs["reaction_motion"].copy_(
            (linear_error / self.motion_velocity_scale).square().sum(-1) +
            (angular_error / self.motion_gyro_scale_rad_s).square().sum(-1))
        # Quasi-static required support force, ignoring feet CoP/contact changes.
        # This is an estimated friction-demand diagnostic, NOT a support polygon.
        support_accel = torch.cat((self.velocity_gain[:2] * (ref[:, :2] - robot[:, :2]),
                                   torch.zeros_like(ref[:, :1])), -1)
        # Actual gravity in the level heading frame is exactly vertical; the
        # separate IMU tilt cue must not be treated as a fictitious lateral force.
        ground_force = rmass[:, None] * support_accel + self.filtered_wrench[:, :3]
        ground_force[:, 2] += rmass * 9.81
        normal = ground_force[:, 2]
        self.support_utilization.copy_(ground_force[:, :2].norm(dim=-1) / (
            self.support_friction_estimate * normal.clamp_min(1.)))
        # A tensile required ground normal is infeasible even when horizontal
        # demand is zero (a pure upward hand load must not look safely supported).
        normal_deficit = (-normal).clamp_min(0.) / (rmass * 9.81).clamp_min(1.)
        support_tilt = (tilt - self.support_tilt_deadzone_rad).clamp_min(0.)
        self.costs["support"].copy_((self.support_utilization - 1.).clamp_min(0.).square() +
                                   normal_deficit.square() + (support_tilt / self.gate_tilt_on_rad).square())
        for cost in self.costs.values():
            cost.clamp_(max=self.cost_clip)
            cost.mul_(ready)
        self.sensor_valid.copy_(ready[:, None])
        features = torch.cat((self.filtered_wrench / self.sensor_scale,
                              self.aligned_required_wrench / self.wrench_scale,
                              self.excess_wrench / self.wrench_scale,
                              self.command_correction / self.correction_limit,
                              self.recovery_gate / self.gate_max, self.sensor_valid), -1)
        self.actor_features.copy_(features.clamp(-self.feature_clip, self.feature_clip))
        # Arithmetic overflow also invalidates the whole row; no stale force survives.
        good &= torch.isfinite(self.actor_features).all(-1) & torch.isfinite(self.required_wrench).all(-1)
        self.reset((~good).nonzero(as_tuple=False).flatten())
        return self
