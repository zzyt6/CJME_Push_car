"""Cart-centred wrench geometry and causal quality signals, shared by Isaac and MuJoCo.

No simulator imports. Filters advance once per *control* step, never while reading observations.
Forces act on the cart; moments use the caller's declared origin (Isaac: handle centre),
expressed in world coordinates. No learned estimator output is used for rewards.
The band is a difference of two one-pole low-passes, not an ideal brick-wall band-pass.
"""
from __future__ import annotations

import math

import torch


QUALITY_PROFILES = ("v53", "v53_wrench_guarded")
QUALITY_REWARD_NAMES = ("cart_ripple", "cart_lin_ripple", "cart_vx_ripple", "cart_vy_ripple",
                        "cart_yaw_ripple", "cart_response_error", "cart_wrench_osc",
                        "cart_wrench_guarded")


def quality_profile_overrides(profile: str) -> dict:
    """Motion-first default, plus an explicitly opt-in guarded force ablation.

    Specify all changed terms so a resumed run can deliberately adopt this revision.
    Applying the default after the force ablation switches its force cost off again.
    """
    if profile not in QUALITY_PROFILES:
        raise ValueError(f"V5.3 quality profile must be one of {QUALITY_PROFILES}")
    return {"quality.enabled": True, "commands.sequence_enabled": True,
            "idle_correction.enabled": False, "reaction.enabled": False,
            "reaction.compensation_enabled": False,
            "rewards.scales.task_wrench": 0., "rewards.scales.reaction_motion": 0.,
            "rewards.scales.centroidal_recovery": 0., "rewards.scales.support_feasibility": 0.,
            "rewards.scales.force_waste": 0., "rewards.scales.force_rate": 0.,
            "rewards.scales.cart_yaw_lead": 0., "rewards.scales.cart_wrench_osc": 0.,
            "rewards.scales.track_cart_vx": 2., "rewards.scales.track_progress": 1.25,
            "rewards.scales.cart_ripple": 0., "rewards.scales.cart_lin_ripple": 0.,
            "rewards.scales.cart_vx_ripple": -.25, "rewards.scales.cart_vy_ripple": -.10,
            "rewards.scales.cart_yaw_ripple": -.15, "rewards.scales.cart_response_error": -.20,
            "rewards.scales.cart_wrench_guarded": -.005 if profile == "v53_wrench_guarded" else 0.,
            "quality.velocity_scale": [.03, .05, .10], "quality.wrench_interval_mean": True}


def apply_quality_profile(cfg, profile: str):
    for path, value in quality_profile_overrides(profile).items():
        target = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def resultant_wrench(force_w, torque_w, points_w, origin_w):
    """Sum point-referenced hand wrenches about one origin; shapes (..., hands, 3)."""
    moment = torque_w + torch.linalg.cross(points_w - origin_w.unsqueeze(-2), force_w, dim=-1)
    return force_w.sum(dim=-2), moment.sum(dim=-2)


def rotate_planar(value, yaw, inverse=False):
    """Rotate first two entries; third entry is the world vertical angular component."""
    angle = -yaw if inverse else yaw
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * value[..., 0] - s * value[..., 1],
                        s * value[..., 0] + c * value[..., 1], value[..., 2]), dim=-1)


class CartQualityTracker:
    def __init__(self, cfg, dt: float, num_envs: int, device="cpu", dtype=torch.float32):
        self.cfg, self.dt = cfg, float(dt)
        lo, hi = float(cfg.low_hz), float(cfg.high_hz)
        if not (math.isfinite(dt) and dt > 0 and 0 < lo < hi < 0.5 / dt):
            raise ValueError("quality filter requires 0 < low_hz < high_hz < control Nyquist")
        for name in ("warmup_s", "reward_ramp_s", "cost_cap"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0 or (name == "cost_cap" and value == 0):
                raise ValueError(f"quality.{name} must be finite and nonnegative (cap > 0)")
        self.alpha_fast = -math.expm1(-2 * math.pi * hi * dt)
        self.alpha_slow = -math.expm1(-2 * math.pi * lo * dt)
        response_tau = float(cfg.response_filter_tau_s)
        if not math.isfinite(response_tau) or response_tau <= 0:
            raise ValueError("quality.response_filter_tau_s must be finite and positive")
        self.alpha_response = -math.expm1(-dt / response_tau)
        def vector(name, positive):
            values = list(getattr(cfg, name))
            if len(values) != 3 or any(not math.isfinite(v) or (v <= 0 if positive else v < 0) for v in values):
                raise ValueError(f"quality.{name} requires three finite {'positive' if positive else 'nonnegative'} values")
            return torch.tensor(values, device=device, dtype=dtype)
        self.velocity_scale = vector("velocity_scale", True)
        self.response_scale = vector("response_scale", True)
        self.response_deadband = torch.zeros_like(self.response_scale)
        self.wrench_scale = vector("wrench_scale", True)
        self.velocity_deadband = vector("velocity_deadband", False)
        self.wrench_deadband = vector("wrench_deadband", False)
        self.wrench_clip = vector("wrench_input_clip", True)
        self.guard_acceleration = vector("wrench_guard_ref_accel_max", True)
        for name in ("wrench_guard_hold_s", "wrench_guard_rise_s", "wrench_guard_min_speed",
                     "wrench_guard_lateral_max", "wrench_guard_yaw_max", "wrench_guard_command_delta"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"quality.{name} must be finite and positive")
        self.fast = torch.zeros(num_envs, 6, device=device, dtype=dtype)
        self.slow = torch.zeros_like(self.fast)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.ready = self.initialized.clone()
        self.velocity_band = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self.wrench_band = torch.zeros_like(self.velocity_band)
        self.ripple_cost = torch.zeros(num_envs, device=device, dtype=dtype)
        self.wrench_cost = torch.zeros_like(self.ripple_cost)
        self.lin_ripple_cost = torch.zeros_like(self.ripple_cost)
        self.yaw_ripple_cost = torch.zeros_like(self.ripple_cost)
        self.axis_ripple_cost = torch.zeros_like(self.velocity_band)
        self.guarded_wrench_cost = torch.zeros_like(self.ripple_cost)
        self.wrench_gate = torch.zeros_like(self.ripple_cost)
        self.guard_elapsed = torch.zeros_like(self.ripple_cost)
        self.guard_previous_ref = torch.zeros_like(self.velocity_band)
        self.guard_previous_command = torch.zeros_like(self.velocity_band)
        self.guard_initialized = self.initialized.clone()
        self.guard_fast = torch.zeros_like(self.velocity_band)
        self.guard_slow = torch.zeros_like(self.velocity_band)
        self.guarded_wrench_band = torch.zeros_like(self.velocity_band)
        self.guard_eligible = self.initialized.clone()
        self.response_cost = torch.zeros_like(self.ripple_cost)
        self.response_error_w = torch.zeros_like(self.velocity_band)
        self.response_error = torch.zeros_like(self.velocity_band)
        self.response_ready = self.initialized.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.fast, self.slow, self.steps, self.velocity_band, self.wrench_band,
                      self.ripple_cost, self.wrench_cost, self.lin_ripple_cost, self.yaw_ripple_cost,
                      self.response_cost, self.response_error_w, self.response_error,
                      self.axis_ripple_cost, self.guarded_wrench_cost, self.wrench_gate,
                      self.guard_elapsed, self.guard_previous_ref, self.guard_previous_command,
                      self.guard_fast, self.guard_slow, self.guarded_wrench_band):
            value[ids] = 0
        self.initialized[ids] = False
        self.ready[ids] = False
        self.response_ready[ids] = False
        self.guard_initialized[ids] = False
        self.guard_eligible[ids] = False

    def _huber(self, signal, scale, deadband):
        z = (signal.abs() - deadband).clamp_min(0) / scale
        return torch.where(z <= 1, 0.5 * z.square(), z - 0.5)

    def _cost(self, signal, scale, deadband, ready=None):
        mask = self.ready if ready is None else ready
        return self._huber(signal, scale, deadband).mean(dim=-1).clamp_max(float(self.cfg.cost_cap)) * mask

    def update(self, velocity_c, reference_c, yaw, force_w, moment_w, valid=None, *,
               command_c=None, disturbance_active=None, wrench_valid=None):
        """One post-physics sample. Seed each reset row from its first valid sample (zero initial band).

        Temporal filtering is in a fixed world frame. Only the resulting band is projected into the
        current cart heading, so a coordinate rotation alone cannot create a spurious force jump.
        Raw input telemetry is retained by callers; clipping here only bounds the reward filter.
        """
        error_w = rotate_planar(velocity_c - reference_c, yaw)
        wrench_w = torch.stack((force_w[:, 0], force_w[:, 1], moment_w[:, 2]), dim=-1)
        # A complete wrench is one physical sample, including unscored vertical/roll/pitch
        # components. A NaN in Fz or Mx/My must not silently leave the speed costs enabled.
        finite = torch.isfinite(torch.cat((error_w, force_w, moment_w), dim=-1)).all(dim=-1)
        # Quality is assessed for a complete physical sample. A bad wrench invalidates its
        # speed costs too; this local mask does not itself remove a transition from PPO storage.
        valid = finite if valid is None else valid & finite
        wrench_w = torch.maximum(torch.minimum(wrench_w, self.wrench_clip), -self.wrench_clip)
        sample = torch.cat((error_w, wrench_w), dim=-1)
        sample = torch.where(valid[:, None], sample, self.fast)
        seed = valid & ~self.initialized
        # Filtering the world-frame difference equals filtering velocity/reference identically.
        # Seed to the actual error: a persistent initial tracking error must not disappear.
        response_sample = torch.where(valid[:, None], error_w, self.response_error_w)
        self.response_error_w = torch.where(seed[:, None], response_sample, self.response_error_w)
        self.response_error_w += self.alpha_response * (response_sample - self.response_error_w)
        self.fast = torch.where(seed[:, None], sample, self.fast)
        self.slow = torch.where(seed[:, None], sample, self.slow)
        self.fast += self.alpha_fast * (sample - self.fast) * valid[:, None]
        self.slow += self.alpha_slow * (sample - self.slow) * valid[:, None]
        self.initialized |= valid
        self.steps += valid.long()
        self.ready = valid & (self.steps * self.dt >= float(self.cfg.warmup_s))
        band = torch.where(valid[:, None], self.fast - self.slow, torch.zeros_like(self.fast))
        safe_yaw = torch.where(torch.isfinite(yaw), yaw, torch.zeros_like(yaw))
        self.velocity_band = rotate_planar(band[:, :3], safe_yaw, inverse=True)
        self.wrench_band = rotate_planar(band[:, 3:], safe_yaw, inverse=True)
        self.ripple_cost = self._cost(self.velocity_band, self.velocity_scale, self.velocity_deadband)
        self.lin_ripple_cost = self._cost(self.velocity_band[:, :2], self.velocity_scale[:2], self.velocity_deadband[:2])
        self.yaw_ripple_cost = self._cost(self.velocity_band[:, 2:], self.velocity_scale[2:], self.velocity_deadband[2:])
        self.axis_ripple_cost = self._huber(self.velocity_band, self.velocity_scale, self.velocity_deadband
                                          ).clamp_max(float(self.cfg.cost_cap)) * self.ready[:, None]
        self.wrench_cost = self._cost(self.wrench_band, self.wrench_scale, self.wrench_deadband)
        self._update_wrench_guard(reference_c, safe_yaw, wrench_w, valid,
                                  command_c, disturbance_active, wrench_valid)
        # Response accounting starts on the first valid sample, not after ripple's 1 s warmup.
        self.response_ready = valid
        self.response_error = torch.where(
            valid[:, None], rotate_planar(self.response_error_w, safe_yaw, inverse=True),
            torch.zeros_like(self.response_error_w))
        self.response_cost = self._cost(self.response_error, self.response_scale, self.response_deadband, valid)

    def _update_wrench_guard(self, reference, yaw, wrench_w, valid, command, disturbance, wrench_valid):
        """Only steady straight transport; missing context disables this optional cost.

        Eligibility depends on the requested motion and injected disturbances, never on the
        policy's tracking success. Large tracking error cannot switch off the main motion costs.
        The additional co-ripple factor permits varying useful force with ripple-free motion;
        it detects coexistence, not physical causality. The default reward weight stays zero.
        """
        self.guarded_wrench_cost.zero_()
        self.wrench_gate.zero_()
        if command is None or disturbance is None or wrench_valid is None:
            self.guard_elapsed.zero_()
            self.guard_initialized.zero_()
            self.guard_eligible.zero_()
            self.guarded_wrench_band.zero_()
            return
        cfg = self.cfg
        context_valid = valid & wrench_valid.bool() & torch.isfinite(command).all(-1)
        safe_ref = torch.where(context_valid[:, None], reference, self.guard_previous_ref)
        safe_command = torch.where(context_valid[:, None], command, self.guard_previous_command)
        acceleration = (safe_ref - self.guard_previous_ref).abs() / self.dt
        changed = (safe_command - self.guard_previous_command).abs().amax(-1) > cfg.wrench_guard_command_delta
        # Exclude all commanded lateral motion / turning, not just sharp turns. A constant
        # body-frame command in a turn still needs changing world-frame centripetal force.
        straight = ((safe_ref[:, 1].abs() <= cfg.wrench_guard_lateral_max)
                    & (safe_command[:, 1].abs() <= cfg.wrench_guard_lateral_max)
                    & (safe_ref[:, 2].abs() <= cfg.wrench_guard_yaw_max)
                    & (safe_command[:, 2].abs() <= cfg.wrench_guard_yaw_max))
        moving = (safe_ref[:, 0].abs() >= cfg.wrench_guard_min_speed) & (
            safe_command[:, 0].abs() >= cfg.wrench_guard_min_speed)
        self.guard_eligible = (context_valid & self.guard_initialized & ~changed & ~disturbance.bool()
                               & (acceleration <= self.guard_acceleration).all(-1) & straight & moving)
        self.guard_elapsed = torch.where(self.guard_eligible, self.guard_elapsed + self.dt, 0.)
        self.wrench_gate = ((self.guard_elapsed - cfg.wrench_guard_hold_s) / cfg.wrench_guard_rise_s
                           ).clamp(0., 1.) * self.ready
        # Erase transient force history while excluded. No retroactive force cost when the
        # hold/rise finishes, including after invalid intervals, resets or repeated pulses.
        sample = torch.where(context_valid[:, None], wrench_w, self.guard_fast)
        self.guard_fast = torch.where(self.guard_eligible[:, None], self.guard_fast, sample)
        self.guard_slow = torch.where(self.guard_eligible[:, None], self.guard_slow, sample)
        self.guard_fast += self.alpha_fast * (sample - self.guard_fast)
        self.guard_slow += self.alpha_slow * (sample - self.guard_slow)
        self.guarded_wrench_band = rotate_planar(self.guard_fast - self.guard_slow, yaw, inverse=True)
        coupling = ((self.velocity_band.abs() - self.velocity_deadband).clamp_min(0)
                    / self.velocity_scale).clamp_max(1.)
        # Cap each axis before combining; no one axis can dominate the auxiliary term.
        axis_cost = self._huber(self.guarded_wrench_band, self.wrench_scale, self.wrench_deadband
                               ).clamp_max(float(cfg.cost_cap))
        self.guarded_wrench_cost = (axis_cost * coupling).mean(-1) * self.wrench_gate
        self.guard_previous_ref.copy_(safe_ref)
        self.guard_previous_command.copy_(safe_command)
        self.guard_initialized.copy_(context_valid)

    def metadata(self):
        return {"filter": "difference_of_one_pole_lowpasses_world_frame", "low_hz": self.cfg.low_hz,
                "high_hz": self.cfg.high_hz, "control_dt_s": self.dt, "warmup_s": self.cfg.warmup_s,
                "velocity_scale": list(self.cfg.velocity_scale), "wrench_scale": list(self.cfg.wrench_scale),
                "wrench_input_clip": list(self.cfg.wrench_input_clip), "cost_cap": self.cfg.cost_cap,
                "response_filter_tau_s": self.cfg.response_filter_tau_s,
                "response_scale": list(self.cfg.response_scale), "response_warmup_s": 0.0,
                "velocity_deadband": list(self.cfg.velocity_deadband),
                "wrench_deadband": list(self.cfg.wrench_deadband),
                "guarded_wrench": {"mode": "steady_straight_transport_with_coincident_motion_ripple",
                    "missing_context": "disabled", "hold_s": self.cfg.wrench_guard_hold_s,
                    "rise_s": self.cfg.wrench_guard_rise_s,
                    "ref_accel_max": list(self.cfg.wrench_guard_ref_accel_max),
                    "min_speed": self.cfg.wrench_guard_min_speed,
                    "lateral_max": self.cfg.wrench_guard_lateral_max,
                    "yaw_max": self.cfg.wrench_guard_yaw_max,
                    "command_delta": self.cfg.wrench_guard_command_delta}}
