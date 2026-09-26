"""Portable FALCON-style PD command and Unitree field adapter (no transport).

One command describes q_des, dq_des, kp, kd and tau_ff. Simulators evaluate it
explicitly; the robot motor controller can evaluate the same hybrid command.
The adapter never initializes DDS, calculates a CRC, or publishes a command.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .control_contract import pd_effort

# Verified against HomieDeploy/unitree_sdk2/g1_control.cpp:G1JointIndex.
# Our 27-DOF model omits motor 13/14 (waist roll/pitch). Their holding controller
# is owned by the robot integration; this adapter MUST NOT silently command them.
HOMIE_TO_UNITREE_G1 = tuple(range(13)) + tuple(range(15, 29))


@dataclass(frozen=True)
class PDCommand:
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray
    torque_limits: np.ndarray
    timestamp_s: float

    def __post_init__(self):
        n = np.asarray(self.q_des).size
        if n == 0 or not math.isfinite(self.timestamp_s):
            raise ValueError('Empty command or invalid monotonic timestamp')
        for key in ('q_des', 'dq_des', 'kp', 'kd', 'tau_ff', 'torque_limits'):
            value = np.array(getattr(self, key), dtype=np.float64, copy=True)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f'{key} must be a finite joint vector of width {n}')
            if key in ('kp', 'kd') and np.any(value < 0):
                raise ValueError('PD gains must be nonnegative')
            if key == 'torque_limits' and np.any(value <= 0):
                raise ValueError('Torque limits must be positive')
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def check_age(self, now_s, max_age_s):
        age = float(now_s) - self.timestamp_s
        if not math.isfinite(age) or not math.isfinite(max_age_s) or max_age_s <= 0 or not 0 <= age <= max_age_s:
            raise ValueError('Stale or future PD command; caller must enter its hardware stop state')

    def effort(self, q, dq):
        q, dq = np.asarray(q), np.asarray(dq)
        if q.shape != self.q_des.shape or dq.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError('Joint state shape/finite check failed')
        return pd_effort(q, dq, self.q_des, self.kp, self.kd, self.torque_limits, self.tau_ff, self.dq_des)

    def fill_unitree(self, low_cmd, *, now_s, max_age_s=.06, execution='motor_pd', q=None, dq=None,
                     motor_indices=HOMIE_TO_UNITREE_G1):
        """Populate existing SDK2 LowCmd motor fields after validating the full packet.

        motor_pd: FALCON's q/dq/kp/kd/tau fields; total saturation is implemented
        by the physical drive and must be checked against its configured limits.
        host_effort: software-clipped total tau, kp=kd=0 (no duplicate PD);
        caller supplies fresh q/dq on EVERY 200 Hz servo iteration.
        Unmapped motors and packet-level mode/CRC fields stay caller-owned.
        """
        self.check_age(now_s, max_age_s)
        if execution not in ('motor_pd', 'host_effort'):
            raise ValueError('Unknown PD execution backend')
        ids = tuple(motor_indices)
        if len(ids) != len(self.q_des) or len(set(ids)) != len(ids):
            raise ValueError('Motor mapping must be one-to-one and cover every model joint')
        if any(not isinstance(i, int) or i < 0 or i >= len(low_cmd.motor_cmd) for i in ids):
            raise ValueError('Motor mapping is outside the SDK packet')
        if execution == 'host_effort':
            tau, kp, kd = self.effort(q, dq), np.zeros_like(self.kp), np.zeros_like(self.kd)
        else:
            if np.any(abs(self.tau_ff) > self.torque_limits):
                raise ValueError('Feed-forward effort exceeds limits')
            tau, kp, kd = self.tau_ff, self.kp, self.kd
        for j, i in enumerate(ids):
            m = low_cmd.motor_cmd[i]
            m.mode = 1
            m.q, m.dq = float(self.q_des[j]), float(self.dq_des[j])
            m.kp, m.kd, m.tau = float(kp[j]), float(kd[j]), float(tau[j])


def command_from_contract(contract, target, timestamp_s, *, target_velocity=None, feedforward=None):
    """Build a nominal command from the exported training contract, no simulator."""
    if contract['upper_drive'] != 'explicit_pd':
        raise ValueError('This deploy adapter requires the explicit-PD training contract')
    q = np.asarray(target, dtype=float)
    if q.shape != (len(contract['joint_names']),):
        raise ValueError('Target joint order/width differs from the control contract')
    if 'position_lower' in contract and (np.any(q < contract['position_lower']) or np.any(q > contract['position_upper'])):
        raise ValueError('Target exceeds the exported joint position limits')
    return PDCommand(q, np.zeros_like(q) if target_velocity is None else target_velocity,
                     contract['kp'], contract['kd'], np.zeros_like(q) if feedforward is None else feedforward,
                     contract['torque_limits'], timestamp_s)
