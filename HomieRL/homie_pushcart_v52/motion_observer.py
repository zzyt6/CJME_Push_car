"""Optional, offline-first cart-motion estimator with consistent inertial history.

This candidate is deliberately not wired into V5 training or deployment. It
estimates planar cart COM motion from actor-visible relative handle pose, HIM
body velocity and IMU alone. Known integer pose latency aligns each pose with
the stored robot pose/attitude at capture time. No cart velocity, force, external
disturbance label or simulator momentum enters the observer.

All positions in a fitting window share one local inertial frame. On every
update that frame is rigidly rebased to the current base origin and heading;
rebasing all historical samples together changes neither their relative motion
nor fitted slopes. This avoids accumulating world positions or mixing vectors
from different historical headings. Gravity levels observed body vectors.

Remaining approximations: cart motion is planar; handle-to-COM geometry is
nominal; HIM velocity and IMU bias integrate over the short window; linear
regression returns a window-averaged velocity, not instantaneous truth. The
latest cart heading is predicted at constant fitted yaw rate during pose delay.
Unknown sensor delays and dynamically corrupted gravity are not identified.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from types import SimpleNamespace

import torch

from .idle_correction import IdleCorrectionTracker


def _rotate_xy(value, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.ndim < value.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    return torch.stack((c * value[..., 0] - s * value[..., 1],
                        s * value[..., 0] + c * value[..., 1]), -1)


def _level_body(vector, roll, pitch):
    """R_y(pitch) R_x(roll): body axes -> upright axes with the same yaw."""
    cr, sr, cp, sp = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
    x = vector[:, 0]
    y = cr * vector[:, 1] - sr * vector[:, 2]
    z = sr * vector[:, 1] + cr * vector[:, 2]
    return torch.stack((cp * x + sp * z, y, -sp * x + cp * z), -1)


class InertialCartMotionObserver:
    """Causal batched candidate; ``update`` returns self, outputs retain storage.

    Config keys: ``window_s=.4``, ``max_pose_delay_steps=4``. Constructor accepts
    a mapping/namespace or None. ``handle_offset_cart`` is handle minus whole-
    cart COM in cart-heading axes; its XY defines the planar lever correction.

    Outputs [N,3]: ``estimated_twist_base`` (level current-base heading) and
    ``estimated_twist_cart`` (predicted current cart heading), both [vx,vy,wz].
    Also ``ready`` [N] bool, ``estimated_cart_relative_yaw``, ``sample_age_s``,
    ``fit_center_age_s``, ``fit_span_s`` [N]. Output velocity is zero until ready.
    The nominal effective measurement age is fit_center_age_s; do not compare
    this window estimate to raw instantaneous truth without reporting latency.

    Delay is the *known age of the handle pose*, in integer control steps. HIM
    and IMU inputs must refer to the current step. Repeated/out-of-order pose
    capture times are ignored for fitting, while the inertial history advances.
    Each episode must call reset; invalid inputs break only their own history.
    """

    def __init__(self, cfg=None, dt=.02, num_envs=1, device="cpu", dtype=torch.float32,
                 handle_offset_cart=(-.49, 0., 0.)):
        get = cfg.get if isinstance(cfg, Mapping) else lambda k, d: getattr(cfg, k, d)
        self.dt, self.num_envs = float(dt), int(num_envs)
        self.device, self.dtype = torch.device(device), dtype
        self.window_s = float(get("window_s", .4))
        delay = get("max_pose_delay_steps", 4)
        if (not math.isfinite(self.dt) or self.dt <= 0 or self.num_envs <= 0 or
                not math.isfinite(self.window_s) or self.window_s < 2 * self.dt):
            raise ValueError("observer requires dt>0, num_envs>0 and window_s>=2*dt")
        if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay <= 100:
            raise ValueError("max_pose_delay_steps must be an integer in [0,100]")
        self.max_pose_delay_steps = delay
        self.window_steps = int(math.ceil(self.window_s / self.dt)) + 1
        self.handle_offset_cart = torch.as_tensor(handle_offset_cart, device=device, dtype=dtype)
        if self.handle_offset_cart.shape != (3,) or not torch.isfinite(self.handle_offset_cart).all():
            raise ValueError("handle_offset_cart must contain three finite values")
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)
        n, d, w = self.num_envs, delay + 1, self.window_steps
        # Base history: XY in current local frame, yaw relative current heading,
        # and roll/pitch at capture time. Index zero is current sample.
        self._base_history = z(n, d, 5)
        self._base_samples = torch.zeros(n, device=device, dtype=torch.long)
        self._previous_velocity = z(n, 2)
        self._previous_yaw_rate = z(n)
        self._positions = z(n, w, 2)
        self._headings = z(n, w)
        self._times = z(n, w)
        self._present = torch.zeros(n, w, device=device, dtype=torch.bool)
        self.estimated_twist_base = z(n, 3)
        self.estimated_twist_cart = z(n, 3)
        self.estimated_cart_relative_yaw = z(n)
        self.sample_age_s, self.fit_center_age_s, self.fit_span_s = z(n), z(n), z(n)
        self.ready = torch.zeros(n, device=device, dtype=torch.bool)
        self._all_state = (self._base_history, self._base_samples, self._previous_velocity,
                           self._previous_yaw_rate, self._positions, self._headings, self._times,
                           self._present, self.estimated_twist_base, self.estimated_twist_cart,
                           self.estimated_cart_relative_yaw, self.sample_age_s,
                           self.fit_center_age_s, self.fit_span_s, self.ready)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for value in self._all_state:
            value[ids] = 0

    def _matrix(self, value, name, width):
        value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if value.shape != (self.num_envs, width):
            raise ValueError(f"{name} must have shape [{self.num_envs},{width}]")
        return value

    @torch.no_grad()
    def update(self, handle_pose_b, base_velocity_b, gyro_b, gravity_b, *,
               valid=None, pose_delay_steps=0):
        pose = self._matrix(handle_pose_b, "handle_pose_b", 4)
        velocity = self._matrix(base_velocity_b, "base_velocity_b", 3)
        gyro = self._matrix(gyro_b, "gyro_b", 3)
        gravity = self._matrix(gravity_b, "gravity_b", 3)
        valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool) if valid is None else torch.as_tensor(valid, device=self.device)
        if valid.shape != (self.num_envs,):
            raise ValueError("valid must have shape [N]")
        good = torch.isfinite(valid) & (valid > .5)
        for value in (pose, velocity, gyro, gravity):
            good &= torch.isfinite(value).all(-1)
        norm = gravity.norm(dim=-1)
        good &= (norm > .5) & (norm < 15.)
        g = torch.nan_to_num(gravity) / torch.nan_to_num(norm).clamp_min(1e-6)[:, None]
        cp2 = g[:, 1:].square().sum(-1)
        good &= cp2 > .25  # avoid yaw singularity beyond +/-60deg pitch.
        roll, pitch = torch.atan2(-g[:, 1], -g[:, 2]), torch.atan2(g[:, 0], cp2.sqrt())
        yaw_rate = -(g[:, 1] * torch.nan_to_num(gyro[:, 1]) + g[:, 2] * torch.nan_to_num(gyro[:, 2])) / cp2.clamp_min(.25)
        level_velocity = _level_body(torch.nan_to_num(velocity), roll, pitch)[:, :2]
        delay = torch.as_tensor(pose_delay_steps, device=self.device)
        if delay.ndim == 0:
            delay = delay.expand(self.num_envs)
        if delay.shape != (self.num_envs,) or not torch.isfinite(delay).all() or not torch.equal(delay, delay.round()):
            raise ValueError("pose_delay_steps must be integer scalar or [N]")
        if torch.any((delay < 0) | (delay > self.max_pose_delay_steps)):
            raise ValueError("pose_delay_steps exceeds retained base history")
        delay = delay.long()

        fresh = self._base_samples == 0
        delta_yaw = torch.where(fresh, 0., .5 * self.dt * (yaw_rate + self._previous_yaw_rate))
        old_velocity_current = _rotate_xy(self._previous_velocity, -delta_yaw)
        displacement = torch.where(fresh[:, None], 0., .5 * self.dt * (old_velocity_current + level_velocity))
        self._base_history[:, :, :2] = _rotate_xy(self._base_history[:, :, :2], -delta_yaw) - displacement[:, None]
        self._base_history[:, :, 2] -= delta_yaw[:, None]
        self._positions.copy_(_rotate_xy(self._positions, -delta_yaw) - displacement[:, None])
        self._headings.sub_(delta_yaw[:, None])
        self._times.sub_(self.dt)
        if self.max_pose_delay_steps:
            self._base_history[:, 1:] = self._base_history[:, :-1].clone()
        self._base_history[:, 0] = torch.stack((torch.zeros_like(roll), torch.zeros_like(roll),
                                               torch.zeros_like(roll), roll, pitch), -1)
        self._base_samples.add_(1).clamp_(max=self.max_pose_delay_steps + 1)
        self._previous_velocity.copy_(level_velocity)
        self._previous_yaw_rate.copy_(yaw_rate)

        capture = self._base_history[torch.arange(self.num_envs, device=self.device), delay]
        level_handle = _level_body(torch.nan_to_num(pose[:, :3]), capture[:, 3], capture[:, 4])
        observed_heading = capture[:, 2] + torch.nan_to_num(pose[:, 3])
        # Unwrap in the SAME local frame as existing heading samples.
        delta = torch.atan2(torch.sin(observed_heading - self._headings[:, -1]),
                            torch.cos(observed_heading - self._headings[:, -1]))
        unwrapped_heading = torch.where(self._present[:, -1], self._headings[:, -1] + delta, observed_heading)
        cart_position = (capture[:, :2] + _rotate_xy(level_handle[:, :2], capture[:, 2]) -
                         _rotate_xy(self.handle_offset_cart[None, :2].expand(self.num_envs, -1), observed_heading))
        capture_time = -delay.to(self.dtype) * self.dt
        accept = good & (self._base_samples > delay) & (
            ~self._present[:, -1] | (capture_time > self._times[:, -1] + self.dt * 1e-4))
        for buffer, new in ((self._positions, cart_position), (self._headings, unwrapped_heading),
                            (self._times, capture_time), (self._present, torch.ones_like(accept))):
            shifted = torch.cat((buffer[:, 1:], new[:, None]), dim=1)
            buffer.copy_(torch.where(accept.reshape(self.num_envs, *([1] * (buffer.ndim - 1))), shifted, buffer))
        # Keep angular coordinates bounded without changing any fit differences.
        self._headings.sub_((2 * math.pi * torch.round(self._headings[:, -1] / (2 * math.pi)))[:, None])

        in_window = self._present & (self._times >= self._times[:, -1:] - self.window_s - self.dt * 1e-4)
        count = in_window.sum(-1).clamp_min(1)
        mean_time = (self._times * in_window).sum(-1) / count
        centered = (self._times - mean_time[:, None]) * in_window
        variance = centered.square().sum(-1).clamp_min(1e-10)
        # Subtract newest position first to avoid cancellation over long runs.
        slope_xy = ((self._positions - self._positions[:, -1:]) * centered[:, :, None]).sum(1) / variance[:, None]
        slope_yaw = ((self._headings - self._headings[:, -1:]) * centered).sum(-1) / variance
        oldest = torch.where(in_window, self._times, torch.full_like(self._times, float("inf"))).amin(-1)
        span = self._times[:, -1] - oldest
        self.fit_span_s.copy_(torch.where(in_window.any(-1), span, 0.))
        self.sample_age_s.copy_(torch.where(self._present[:, -1], -self._times[:, -1], 0.))
        self.fit_center_age_s.copy_(torch.where(in_window.any(-1), -mean_time, 0.))
        self.ready.copy_(good & (count >= 3) & (span >= self.window_s - self.dt * .5) &
                         (self.sample_age_s <= self.max_pose_delay_steps * self.dt + self.dt * .5))
        predicted_heading = self._headings[:, -1] + slope_yaw * self.sample_age_s
        self.estimated_cart_relative_yaw.copy_(torch.atan2(torch.sin(predicted_heading), torch.cos(predicted_heading)))
        self.estimated_twist_base.copy_(torch.cat((slope_xy, slope_yaw[:, None]), -1))
        self.estimated_twist_cart.copy_(torch.cat((_rotate_xy(slope_xy, -predicted_heading), slope_yaw[:, None]), -1))
        self.estimated_twist_base.masked_fill_(~self.ready[:, None], 0.)
        self.estimated_twist_cart.masked_fill_(~self.ready[:, None], 0.)
        self.reset((~good).nonzero(as_tuple=False).flatten())
        return self


def compare_motion_estimators(trace, *, dt=.02, window_s=.4, max_pose_delay_steps=4,
                              handle_offset_cart=(-.49, 0., 0.)):
    """Replay actor samples offline through candidate and current V5 estimator.

    Trace keys [T,N,D] (N may be omitted): ``handle_pose_b`` D4,
    ``base_velocity_b``/``gyro_b``/``gravity_b`` D3. Optional [T,N] ``valid``,
    ``reset`` (before update), ``pose_delay_steps``. Optional evaluation-only
    ``truth_cart_twist_base`` D3 uses current upright base-heading axes. Truth
    is NEVER passed into either estimator. This contract records sensor values
    before scaling, their known latency, and reset boundaries explicitly.

    Returns CPU tensors for each estimate/validity/age, and optional RMSE/mean
    error versus CURRENT truth on the intersection of ready samples. The RMSE
    is deliberately labeled instantaneous: it includes regression-window delay,
    so it alone cannot prove an estimator with more smoothing is more accurate.
    """
    def matrix(name, width):
        value = torch.as_tensor(trace[name], dtype=torch.float64)
        if value.ndim == 2:
            value = value[:, None]
        if value.ndim != 3 or value.shape[-1] != width:
            raise ValueError(f"{name} must be [T,N,{width}] or [T,{width}]")
        return value
    pose, velocity, gyro, gravity = (matrix(name, width) for name, width in (
        ("handle_pose_b", 4), ("base_velocity_b", 3), ("gyro_b", 3), ("gravity_b", 3)))
    steps, n, _ = pose.shape
    if any(value.shape[:2] != (steps, n) for value in (velocity, gyro, gravity)):
        raise ValueError("all sensor traces must share [T,N]")
    def optional(name, default):
        value = torch.as_tensor(trace.get(name, default))
        if value.ndim == 0:
            return value.expand(steps, n)
        if value.shape == (steps,):
            return value[:, None].expand(-1, n)
        if value.shape != (steps, n):
            raise ValueError(f"{name} must be scalar, [T], or [T,N]")
        return value
    validity, reset, delay = optional("valid", True), optional("reset", False), optional("pose_delay_steps", 0)
    candidate = InertialCartMotionObserver(dict(window_s=window_s, max_pose_delay_steps=max_pose_delay_steps),
                                          dt, n, dtype=torch.float64, handle_offset_cart=handle_offset_cart)
    baseline = IdleCorrectionTracker(SimpleNamespace(window_s=window_s), dt, n, dtype=torch.float64)
    offset = torch.tensor(handle_offset_cart, dtype=torch.float64)
    out = {name: [] for name in ("candidate_twist_base", "baseline_twist_base", "candidate_ready", "baseline_ready",
                                 "candidate_sample_age_s", "candidate_fit_center_age_s", "candidate_fit_span_s")}
    for i in range(steps):
        ids = reset[i].bool().nonzero(as_tuple=False).flatten()
        candidate.reset(ids)
        baseline.reset(ids)
        candidate.update(pose[i], velocity[i], gyro[i], gravity[i], valid=validity[i], pose_delay_steps=delay[i])
        baseline.update(pose[i, :, :3], pose[i, :, 3], velocity[i], gyro[i], gravity[i])
        baseline.reset((~validity[i].bool()).nonzero(as_tuple=False).flatten())
        observed_lever = _rotate_xy(offset[None, :2].expand(n, -1), pose[i, :, 3])
        omega = baseline.estimated_twist[:, 2]
        lever_velocity = torch.stack((-omega * observed_lever[:, 1], omega * observed_lever[:, 0]), -1)
        base_twist = baseline.estimated_twist.clone()
        base_twist[:, :2] -= lever_velocity
        base_twist.masked_fill_(~baseline.ready[:, None], 0.)
        values = (candidate.estimated_twist_base, base_twist, candidate.ready, baseline.ready,
                  candidate.sample_age_s, candidate.fit_center_age_s, candidate.fit_span_s)
        for key, value in zip(out, values):
            out[key].append(value.clone())
    out = {key: torch.stack(value) for key, value in out.items()}
    if "truth_cart_twist_base" in trace:
        truth = matrix("truth_cart_twist_base", 3)
        if truth.shape != out["candidate_twist_base"].shape:
            raise ValueError("truth_cart_twist_base must share sensor [T,N]")
        common = out["candidate_ready"] & out["baseline_ready"] & torch.isfinite(truth).all(-1)
        metrics = {"sample_count": int(common.sum()), "comparison": "current truth; includes regression-window latency",
                   "candidate_mean_fit_center_age_s": out["candidate_fit_center_age_s"][common].mean().item() if common.any() else None,
                   "candidate_mean_sample_age_s": out["candidate_sample_age_s"][common].mean().item() if common.any() else None}
        for label in ("candidate", "baseline"):
            error = (out[label + "_twist_base"] - truth)[common]
            metrics[label + "_rmse_instantaneous"] = error.square().mean(0).sqrt().tolist() if len(error) else None
            metrics[label + "_mean_error_instantaneous"] = error.mean(0).tolist() if len(error) else None
        out["metrics"] = metrics
    return out
