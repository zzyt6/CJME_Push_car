"""Per-environment continuous curriculum for V7 (replaces V3.2's 18 global levels).

V3.2 / V6 moved every environment together through 18 discrete levels; in the V6 run a level passed every ~42
iterations (the evaluation interval), every level change shifted the whole population at once (1-3 % reward dips),
and several items switched on abruptly (robot pushes at level 12 -> the longest stall, 458 iterations).

Here each environment carries its own difficulty d in [0, 1] (as legged_gym's per-environment terrain curriculum,
but continuous). At the end of an episode d rises after a success (no termination and every tracking axis at
``success_tracking_frac`` of its maximum reward) and falls after a termination; the population therefore spreads
over a range of difficulties instead of sitting on one level. Every randomised quantity is a linear ramp of d over
its own window, so nothing switches on abruptly; the windows reproduce the V3.2 level schedule (level L <-> d = L/17)
so the overall pacing is unchanged. V7's physical ranges widen around their nominal values (automatic-domain-
randomisation style): at d = 0 an environment samples ``physics_width_start`` of the full width.
"""
from __future__ import annotations

import torch

from homie_pushcart.curriculum import LevelParams

LEVELS = 17                     # V3.2 level L corresponds to d = L / 17


def ramp(d: torch.Tensor, window) -> torch.Tensor:
    lo, hi = float(window[0]), float(window[1])
    return ((d - lo) / max(hi - lo, 1e-6)).clamp(0., 1.)


class ContinuousCurriculum:
    def __init__(self, task_cfg, num_envs: int, device, init_difficulty: float = 0.0):
        self.cfg = task_cfg.curriculum_v7
        self.tc = task_cfg
        self.d = torch.full((num_envs,), float(init_difficulty), device=device)
        self.device = device

    # ------------------------------------------------------------------ update
    def update(self, env_ids, success, failed):
        if not self.cfg.enabled or len(env_ids) == 0:
            return
        step = success.float() * self.cfg.step_up - failed.float() * self.cfg.step_down
        self.d[env_ids] = (self.d[env_ids] + step).clamp(0., 1.)

    # ------------------------------------------------------------------ per-environment schedules
    def commands(self, ids):
        c, d = self.cfg, self.d[ids]
        vx_lo, vx_hi = self.tc.commands.vx_range_max
        return (vx_lo * ramp(d, c.vx_backward_ramp), vx_hi * ramp(d, c.vx_forward_ramp),
                self.tc.commands.vy_max * ramp(d, c.vy_ramp), self.tc.commands.wz_max * ramp(d, c.wz_ramp))

    def payload_max(self, ids):
        c = self.cfg
        return c.payload_start_kg + (c.payload_max_kg - c.payload_start_kg) * ramp(self.d[ids], c.payload_ramp)

    def capacity_mult(self, ids):
        g = self.tc.grasp
        return g.capacity_mult_start - (g.capacity_mult_start - g.capacity_mult_floor) * ramp(self.d[ids], self.cfg.capacity_ramp)

    def disturbance(self, ids):
        return ramp(self.d[ids], self.cfg.disturbance_ramp)

    def robot_push(self, ids):
        return ramp(self.d[ids], self.cfg.robot_push_ramp)

    def physics_width(self, ids):
        w0 = self.cfg.physics_width_start
        return w0 + (1. - w0) * ramp(self.d[ids], self.cfg.physics_ramp)

    def sample(self, value_range, ids, shape_tail=()):
        """U over the range shrunk to physics_width around its centre, per environment."""
        lo, hi = float(value_range[0]), float(value_range[1])
        w = self.physics_width(ids).view(-1, *([1] * len(shape_tail)))
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * w
        u = torch.rand(len(ids), *shape_tail, device=self.device)
        return mid - half + 2. * half * u

    # ------------------------------------------------------------------ population-level values
    def global_params(self) -> LevelParams:
        """Continuous V3.2 LevelParams at the population's mean difficulty, for the remaining global uses
        (observation noise / delay, wheel-stall rate) and for logging."""
        c, tc = self.cfg, self.tc
        d = self.d.mean().view(1)
        vx_lo, vx_hi = tc.commands.vx_range_max
        return LevelParams(
            level=int(round(float(d) * LEVELS)),
            vx_range=(float(vx_lo * ramp(d, c.vx_backward_ramp)), float(vx_hi * ramp(d, c.vx_forward_ramp))),
            vy_max=float(tc.commands.vy_max * ramp(d, c.vy_ramp)), wz_max=float(tc.commands.wz_max * ramp(d, c.wz_ramp)),
            arm_scale=float(tc.actions.arm_scale_range[1]), waist_scale=float(tc.actions.waist_scale_range[1]),
            payload_max_kg=float(c.payload_start_kg + (c.payload_max_kg - c.payload_start_kg) * ramp(d, c.payload_ramp)),
            caster_angle_range=float(tc.cart_physics.caster_heading_range_rad),
            grasp_threshold_mult=float(tc.grasp.capacity_mult_start - (tc.grasp.capacity_mult_start - tc.grasp.capacity_mult_floor)
                                       * ramp(d, c.capacity_ramp)),
            obs_noise_frac=float(ramp(d, c.noise_ramp)), disturbance_frac=float(ramp(d, c.disturbance_ramp)),
            robot_push=True)

    def stats(self) -> dict:
        q = torch.quantile(self.d, torch.tensor([0.1, 0.5, 0.9], device=self.device))
        return {"mean": float(self.d.mean()), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]),
                "at_max_frac": float((self.d >= 0.999).float().mean())}
