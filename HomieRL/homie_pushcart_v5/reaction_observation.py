"""Shared V5 reaction adapter from actor-visible kinematics and wrist sensing.

This estimates planar cart COM motion from observed relative handle pose, HOMIE
base-velocity estimate and IMU. It deliberately does not accept simulator cart
velocity, wheel forces, disturbance labels, or full-body momentum. The existing
windowed handle-motion estimator is approximate under changing body attitude;
the planar geometry, nominal COM lever arm and yaw inertia are model estimates.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from types import SimpleNamespace

import torch

from .idle_correction import IdleCorrectionTracker
from .reaction import ReactionController


def rotate_planar(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate XY by ``yaw``; keep the third linear/angular component unchanged.

    For observed relYaw = cart_yaw - base_yaw: base -> cart uses -relYaw,
    and cart -> base uses +relYaw. Works for either xyz or [vx,vy,wz].
    """
    if vector.ndim != 2 or vector.shape[-1] != 3 or yaw.shape != vector.shape[:1]:
        raise ValueError("rotate_planar requires vector [N,3] and yaw [N]")
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * vector[:, 0] - s * vector[:, 1],
                        s * vector[:, 0] + c * vector[:, 1], vector[:, 2]), -1)


class ReactionObservation:
    """One identical sensor-to-controller path for Isaac and MuJoCo.

    ``pose_base`` is actor-visible [handle_x,handle_y,handle_z,cart_rel_yaw],
    in metres/radians; base velocity and gyro are observed body-frame values.
    ``wrench_cart`` is ROBOT-ON-CART [Fx,Fy,Fz,Mx,My,Mz] already transformed to
    level cart-heading axes, about estimated whole-cart COM. The controller
    adds virtual wrist latency/noise. ``command_ref`` is already in CART axes.

    ``handle_offset_cart`` is the nominal handle-center minus cart COM vector.
    Only XY participates in planar yaw correction. Its default x=-.49m matches
    the existing trolley's centered-load geometry; payload COM shifts introduce
    model uncertainty. Estimated yaw inertia is mass*r_gyration^2 with a .33m
    default radius. Override both from identified geometry when available.

    ``ready`` indicates a complete motion-estimator window plus valid inputs;
    controller.sensor_valid additionally includes the virtual wrist warmup.
    ``command_correction_base`` returns the ready controller correction rotated
    back to the HOMIE base-heading interface. No transform advances state.
    """

    def __init__(self, cfg, dt: float, num_envs: int, device="cpu", dtype=torch.float32,
                 handle_offset_cart=(-.49, 0., 0.)):
        self.controller = ReactionController(cfg, dt, num_envs, device, dtype)
        self.num_envs = self.controller.num_envs
        self.device, self.dtype = self.controller.device, dtype
        get = cfg.get if isinstance(cfg, Mapping) else lambda k, d: getattr(cfg, k, d)
        self.motion_window_s = float(get("motion_window_s", .4))
        self.yaw_radius_gyration_m = float(get("yaw_radius_gyration_m", .33))
        if not math.isfinite(self.motion_window_s) or self.motion_window_s <= 0:
            raise ValueError("reaction.motion_window_s must be finite and positive")
        if not math.isfinite(self.yaw_radius_gyration_m) or self.yaw_radius_gyration_m <= 0:
            raise ValueError("reaction.yaw_radius_gyration_m must be finite and positive")
        self.handle_offset_cart = torch.as_tensor(handle_offset_cart, device=self.device, dtype=dtype)
        if self.handle_offset_cart.shape != (3,) or not torch.isfinite(self.handle_offset_cart).all():
            raise ValueError("handle_offset_cart must be three finite values")
        self.motion = IdleCorrectionTracker(SimpleNamespace(window_s=self.motion_window_s),
                                             dt, num_envs, device, dtype)
        self.estimated_cart_twist = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        self.command_correction_base = torch.zeros_like(self.estimated_cart_twist)
        self.ready = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._relative_yaw = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self._cart_to_base = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        self._previous_base_yaw_rate = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self._pose_valid = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    @property
    def relative_yaw(self):
        """Latest observed cart-minus-base yaw; available before motion warmup."""
        return self._relative_yaw

    def reference_base(self, command_ref):
        """Map a cart-COM task twist to the observed base point and heading.

        Includes omega x (base-cart_COM), then rotates cart -> base axes. Uses
        only stored measured pose/nominal geometry and does not advance state.
        Before the first valid pose after reset the lever is zero.
        """
        reference = self.controller._matrix(command_ref, "command_ref", 3)
        lever = self._cart_to_base * self._pose_valid[:, None]
        relative = torch.stack((-reference[:, 2] * lever[:, 1],
                                 reference[:, 2] * lever[:, 0],
                                 torch.zeros_like(reference[:, 2])), -1)
        return rotate_planar(reference + relative, self._relative_yaw)

    def reset(self, ids=None):
        """Flush every kinematic and sensor history for selected episodes."""
        self.motion.reset(ids)
        self.controller.reset(ids)
        ids = slice(None) if ids is None else ids
        for value in (self.estimated_cart_twist, self.command_correction_base,
                      self.ready, self._relative_yaw, self._cart_to_base,
                      self._previous_base_yaw_rate, self._pose_valid):
            value[ids] = 0

    @torch.no_grad()
    def update(self, wrench_cart, pose_base, base_velocity_est, imu_gyro_b, gravity_b,
               command_ref, cart_mass_est, robot_mass_est, valid):
        """Update once after observing a control step; return ``controller``.

        Handle motion follows dr/dt + v_base + omega x r. Its windowed twist is
        rotated to cart axes, then omega_cart x (handle-COM) is subtracted.
        Body-frame IMU gravity is only yaw-rotated, preserving the tilt cue; it
        must not be replaced by world gravity, which has no attitude information.
        The yaw-only body/cart transform assumes near-upright locomotion; neither
        estimated COM twist nor support cost is claimed as full dynamics truth.
        """
        q = self.controller
        wrench = q._matrix(wrench_cart, "wrench_cart", 6)
        pose = q._matrix(pose_base, "pose_base", 4)
        velocity = q._matrix(base_velocity_est, "base_velocity_est", 3)
        gyro = q._matrix(imu_gyro_b, "imu_gyro_b", 3)
        gravity = q._matrix(gravity_b, "gravity_b", 3)
        reference = q._matrix(command_ref, "command_ref", 3)
        cart_mass = q._vector(cart_mass_est, "cart_mass_est")
        robot_mass = q._vector(robot_mass_est, "robot_mass_est")
        validity = q._vector(valid, "valid")
        good = torch.isfinite(validity) & (validity > .5)
        for value in (wrench, pose, velocity, gyro, gravity, reference):
            good &= torch.isfinite(value).all(-1)
        good &= torch.isfinite(cart_mass) & (cart_mass > 0.)
        good &= torch.isfinite(robot_mass) & (robot_mass > 0.)

        # Existing motion tracker owns its finite/IMU-attitude validation. Other
        # invalid channels also break its history, even when kinematics are finite.
        self.motion.update(pose[:, :3], pose[:, 3], velocity, gyro, gravity)
        self.motion.reset((~good).nonzero(as_tuple=False).flatten())
        self.ready.copy_(good & self.motion.ready)
        observed_yaw = torch.nan_to_num(pose[:, 3])
        g = torch.nan_to_num(gravity)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        denom = g[:, 1:].square().sum(-1).clamp_min(.25)
        base_yaw_rate = -(g[:, 1] * gyro[:, 1] + g[:, 2] * gyro[:, 2]) / denom
        delta_relative = torch.atan2(torch.sin(observed_yaw - self._relative_yaw),
                                     torch.cos(observed_yaw - self._relative_yaw))
        heading_delta = delta_relative + .5 * q.dt * (base_yaw_rate + self._previous_base_yaw_rate)
        heading_delta = torch.where(self._pose_valid, heading_delta, 0.)
        self._relative_yaw.copy_(observed_yaw)
        self._previous_base_yaw_rate.copy_(torch.nan_to_num(base_yaw_rate))
        self._pose_valid.copy_(good & (self.motion.samples > 0))
        handle_twist_cart = rotate_planar(self.motion.estimated_twist, -self._relative_yaw)
        omega = handle_twist_cart[:, 2]
        lever_velocity = torch.stack((-omega * self.handle_offset_cart[1],
                                       omega * self.handle_offset_cart[0],
                                       torch.zeros_like(omega)), -1)
        self.estimated_cart_twist.copy_(handle_twist_cart - lever_velocity)
        self.estimated_cart_twist.masked_fill_(~self.ready[:, None], 0.)
        robot_cart = rotate_planar(torch.nan_to_num(velocity), -self._relative_yaw)
        # In a steady turn, the base and cart COM do not have equal velocities.
        # Map the base velocity to an equivalent cart-COM point using only the
        # observed handle position and nominal cart geometry. The controller's
        # v_ref-v_robot term then measures task-relative motion at ONE point.
        base_to_handle_cart = rotate_planar(torch.nan_to_num(pose[:, :3]), -self._relative_yaw)
        cart_to_base = self.handle_offset_cart - base_to_handle_cart
        self._cart_to_base.copy_(cart_to_base)
        nominal_relative_velocity = torch.stack((-reference[:, 2] * cart_to_base[:, 1],
                                                 reference[:, 2] * cart_to_base[:, 0],
                                                 torch.zeros_like(reference[:, 2])), -1)
        robot_cart = robot_cart - nominal_relative_velocity
        gyro_cart = rotate_planar(torch.nan_to_num(gyro), -self._relative_yaw)
        gravity_cue_cart = rotate_planar(torch.nan_to_num(gravity), -self._relative_yaw)
        q.update(wrench[:, :3], wrench[:, 3:], self.estimated_cart_twist, reference,
                 robot_cart, gyro_cart, gravity_cue_cart, cart_mass,
                 cart_mass * self.yaw_radius_gyration_m ** 2, robot_mass, self.ready,
                 heading_delta_rad=heading_delta)
        self.command_correction_base.copy_(self.reference_base(q.command_correction))
        self.command_correction_base.masked_fill_(~self.ready[:, None], 0.)
        return q
