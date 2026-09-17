"""Fine-grained multi-scale curriculum for the push-cart task (about 18 levels, HOMIE style).

Each level opens every scale by one small notch; speed first, then turning, then payload, then disturbances
and observation noise. Promotion needs the last ``window_episodes`` finished episodes to reach
``advance_tracking_frac`` of the maximum tracking reward with a non-timeout termination rate below
``advance_max_fail_rate``; ``demote_after_windows`` consecutive failing windows drop one level (hysteresis).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class LevelParams:
    level: int
    vx_range: tuple[float, float]
    vy_max: float
    wz_max: float
    arm_scale: float
    waist_scale: float
    payload_max_kg: float
    caster_angle_range: float          # initial swivel angle ~ U[-r, r]
    grasp_threshold_mult: float
    obs_noise_frac: float              # 0 = clean, 1 = full noise / delay
    disturbance_frac: float            # 0 = none, 1 = full cart impulse / wheel stall
    robot_push: bool

    def as_dict(self) -> dict:
        return asdict(self)


def level_params(level: int, cfg) -> LevelParams:
    """Schedule per level. ``cfg`` is ``PushCartTaskCfg`` (needs .commands / .actions / .curriculum)."""
    L = max(0, int(level))
    n_last = cfg.curriculum.num_levels - 1
    vx_lo, vx_hi = cfg.commands.vx_range_max
    if L == 0:
        vx = (0.0, 0.0)
    else:
        vx = (max(vx_lo, -0.1 * L), min(vx_hi, 0.1 * L))
    wz = 0.0 if L < 4 else min(cfg.commands.wz_max, 0.1 * (L - 3))
    vy = 0.0 if L < 8 else min(cfg.commands.vy_max, 0.05 * (L - 7))
    arm0, arm1 = cfg.actions.arm_scale_range
    waist0, waist1 = cfg.actions.waist_scale_range
    arm = min(arm1, arm0 + 0.05 * L)
    waist = min(waist1, waist0 + 0.04 * L)
    payload = 5.0 if L < 6 else min(25.0, 5.0 + 2.5 * (L - 5))
    caster = 0.0 if L < 4 else min(math.pi, math.pi / 8 * (L - 3))
    grasp = 2.0 if L < 3 else max(1.0, 2.0 - 0.1 * (L - 2))
    noise = 0.0 if L < 8 else min(1.0, (L - 7) / 6.0)
    disturb = 0.0 if L < 10 else min(1.0, (L - 9) / 6.0)
    push = L >= 12
    return LevelParams(min(L, n_last), vx, vy, wz, arm, waist, payload, caster, grasp, noise, disturb, push)


class CurriculumTracker:
    """Windowed promotion / demotion bookkeeping shared by all envs (episode statistics come in batches)."""

    def __init__(self, cfg, level: int | None = None):
        self.cfg = cfg
        self.level = cfg.curriculum.init_level if level is None else int(level)
        self.enabled = bool(cfg.curriculum.enabled)
        self.window = int(cfg.curriculum.window_episodes)
        self._tracking_sum = 0.0
        self._fail_sum = 0.0
        self._count = 0
        self._failed_windows = 0
        self.last_window = {"tracking_frac": float("nan"), "fail_rate": float("nan")}
        self.changes = 0

    @property
    def params(self) -> LevelParams:
        return level_params(self.level, self.cfg)

    def record(self, tracking_frac, failed) -> bool:
        """Add finished episodes (sequences of per-episode tracking fraction and 0/1 failure). Returns True if
        the level changed."""
        tracking_frac = list(tracking_frac)
        failed = list(failed)
        changed = False
        for t, f in zip(tracking_frac, failed):
            self._tracking_sum += float(t)
            self._fail_sum += float(f)
            self._count += 1
            if self._count >= self.window:
                changed |= self._evaluate_window()
        return changed

    def record_window(self, tracking_mean: float, fail_rate: float) -> bool:
        """Evaluate one already-aggregated window (used by the env, which accumulates on the GPU)."""
        self._tracking_sum, self._fail_sum, self._count = float(tracking_mean), float(fail_rate), 1
        return self._evaluate_window()

    def _evaluate_window(self) -> bool:
        tracking = self._tracking_sum / self._count
        fail_rate = self._fail_sum / self._count
        self.last_window = {"tracking_frac": tracking, "fail_rate": fail_rate}
        self._tracking_sum = self._fail_sum = 0.0
        self._count = 0
        if not self.enabled:
            return False
        max_level = self.cfg.curriculum.num_levels - 1
        passed = tracking >= self.cfg.curriculum.advance_tracking_frac and fail_rate < self.cfg.curriculum.advance_max_fail_rate
        if passed:
            self._failed_windows = 0
            if self.level < max_level:
                self.level += 1
                self.changes += 1
                return True
            return False
        self._failed_windows += 1
        if self._failed_windows >= self.cfg.curriculum.demote_after_windows and self.level > 0:
            self.level -= 1
            self._failed_windows = 0
            self.changes += 1
            return True
        return False

    def state_dict(self) -> dict:
        return {"level": self.level, "failed_windows": self._failed_windows, "changes": self.changes}
