"""Batched V4 command sequences without simulator dependencies.

The caller retains the legacy random-command sampler. ``sample`` only replaces
the rows assigned a structured episode on ``reset``; command filtering still
belongs to the environment. State advances on resampling, not on observation.
"""
from __future__ import annotations

import math

import torch


class V4CommandSchedule:
    RANDOM = 0
    STOP_GO = 1
    TURN_REVERSAL = 2

    def __init__(self, cfg, dt: float, num_envs: int, device="cpu"):
        self.enabled = bool(getattr(cfg, "sequence_enabled", False))
        self.dt = float(dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("command schedule dt must be finite and positive")
        self.stop_go_prob = float(getattr(cfg, "stop_go_prob", 0.25))
        self.turn_reversal_prob = float(getattr(cfg, "turn_reversal_prob", 0.25))
        probabilities = (self.stop_go_prob, self.turn_reversal_prob)
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("command sequence probabilities must be finite, nonnegative and sum to <= 1")
        self.min_frac = float(getattr(cfg, "structured_speed_min_frac", 0.35))
        if not math.isfinite(self.min_frac) or not 0 < self.min_frac <= 1:
            raise ValueError("commands.structured_speed_min_frac must lie in (0, 1]")
        self.device = torch.device(device)
        self._duration_steps = {}
        for name, default in (("move_time_s", (2.0, 4.0)), ("stop_hold_time_s", (3.0, 5.0)),
                              ("turn_time_s", (2.0, 4.0))):
            bounds = tuple(float(v) for v in getattr(cfg, name, default))
            if len(bounds) != 2 or any(not math.isfinite(v) or v <= 0 for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"commands.{name} requires two finite positive ordered durations")
            # Quantize to whole control steps; a sub-step interval lasts one step.
            lo = max(1, int(math.ceil(bounds[0] / self.dt - 1e-9)))
            hi = max(lo, int(math.floor(bounds[1] / self.dt + 1e-9)))
            self._duration_steps[name] = (lo, hi)
        self.mode = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        # phase counts emitted structured targets, so phase 0 is moving / first turn.
        self.phase = torch.zeros_like(self.mode)
        self.targets = torch.zeros(num_envs, 3, device=self.device)

    @staticmethod
    def _bounds(level_params):
        vx_lo, vx_hi = (float(v) for v in level_params.vx_range)
        vy, wz = float(level_params.vy_max), float(level_params.wz_max)
        if not all(math.isfinite(v) for v in (vx_lo, vx_hi, vy, wz)) or vx_lo > vx_hi or min(vy, wz) < 0:
            raise ValueError("command curriculum ranges must be finite and ordered, with nonnegative axis maxima")
        return vx_lo, vx_hi, vy, wz

    def reset(self, env_ids, level_params):
        """Choose each reset episode's mode; leave other environments untouched.

        Unsupported modes fall back to legacy random commands: level zero stays
        still, and turn reversal is unavailable until yaw commands are unlocked.
        """
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.mode[ids] = self.RANDOM
        self.phase[ids] = 0
        self.targets[ids] = 0
        if not self.enabled or ids.numel() == 0:
            return
        vx_lo, vx_hi, _, wz = self._bounds(level_params)
        # Exact stop must be admitted by the curriculum; do not invent a stop
        # outside a caller's strictly positive/negative command interval.
        can_stop_go = vx_lo <= 0 <= vx_hi and (vx_lo < 0 or vx_hi > 0)
        if not can_stop_go:
            return
        draw = torch.rand(ids.numel(), device=self.device)
        modes = torch.where(draw < self.stop_go_prob, self.STOP_GO, self.RANDOM)
        if wz > 0:
            turn = (draw >= self.stop_go_prob) & (draw < self.stop_go_prob + self.turn_reversal_prob)
            modes = torch.where(turn, self.TURN_REVERSAL, modes)
        self.mode[ids] = modes
        # Forward push whenever available; a reverse-only curriculum is supported.
        speed_bound = vx_hi if vx_hi > 0 else vx_lo
        amplitude = self.min_frac + (1 - self.min_frac) * torch.rand(ids.numel(), 2, device=self.device)
        self.targets[ids, 0] = speed_bound * amplitude[:, 0]
        signs = torch.where(torch.rand(ids.numel(), device=self.device) < 0.5, -1., 1.)
        self.targets[ids, 2] = signs * wz * amplitude[:, 1]

    def _durations(self, count, name):
        lo, hi = self._duration_steps[name]
        return torch.randint(lo, hi + 1, (count,), device=self.device)

    def sample(self, env_ids, level_params, random_commands, random_steps):
        """Return targets/timers for requested rows, advancing only their phases.

        ``random_commands`` is shaped (len(env_ids), 3), ``random_steps`` (len(env_ids),).
        They are unchanged for random episodes. Seed ``reset`` before the first
        call in each episode. Current curriculum limits apply even if another
        episode caused a demotion since this episode's targets were seeded.
        """
        if not self.enabled or env_ids.numel() == 0:
            return random_commands, random_steps
        ids = env_ids.to(device=self.device, dtype=torch.long)
        vx_lo, vx_hi, vy, wz = self._bounds(level_params)
        commands, steps = random_commands.clone(), random_steps.clone()
        modes = self.mode[ids]
        phase = self.phase[ids]
        stop_go = modes == self.STOP_GO
        turn = modes == self.TURN_REVERSAL
        moving = stop_go & ((phase % 2) == 0)
        stopped = stop_go & ~moving
        targets = self.targets[ids].clone()
        targets[:, 0].clamp_(vx_lo, vx_hi)
        targets[:, 1].clamp_(-vy, vy)
        targets[:, 2].clamp_(-wz, wz)
        commands[stop_go] = 0
        commands[moving, 0] = targets[moving, 0]
        # Draw all requested rows so no GPU-to-host reduction is needed to learn
        # the size of each mode subset during training.
        steps[moving] = self._durations(ids.numel(), "move_time_s")[moving]
        steps[stopped] = self._durations(ids.numel(), "stop_hold_time_s")[stopped]
        commands[turn] = targets[turn]
        yaw_sign = torch.where((phase[turn] % 2) == 0, 1., -1.)
        commands[turn, 2] *= yaw_sign
        steps[turn] = self._durations(ids.numel(), "turn_time_s")[turn]
        self.phase[ids] += (stop_go | turn).to(self.phase.dtype)
        return commands, steps
