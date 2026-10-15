"""LCM link to the C++ bridge (unitree_sdk2/g1_control_v81, which talks DDS to the robot at 200 Hz).

Channels (types from HomieDeploy, unchanged):
  in   body_control_data     body_control_data_lcmt   q[29], qd[29]           motor order (G1 29-dof)
  in   state_estimator_data  state_estimator_lcmt     rpy, omegaBody (pelvis IMU)
  in   rc_command            rc_command_lcmt          sticks; mode = raw key bitmask (g1_control_v81 only)
  out  pd_plustau_targets    pd_tau_targets_lcmt      q_des[0:15] = 12 legs, waist yaw, roll, pitch
  out  arm_action            arm_action_lcmt          act[14] = motors 15..28 (left arm 7, right arm 7)
  out  hand_action           hand_action_lcmt         act[14] = left Dex-3 7, right Dex-3 7 (hand_control)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .lcm_types.arm_action_lcmt import arm_action_lcmt
from .lcm_types.body_control_data_lcmt import body_control_data_lcmt
from .lcm_types.hand_action_lcmt import hand_action_lcmt
from .lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from .lcm_types.rc_command_lcmt import rc_command_lcmt
from .lcm_types.state_estimator_lcmt import state_estimator_lcmt

NUM_MOTORS = 29
# HOMIE's 27 joints (12 legs, waist yaw, 7 + 7 arms) in G1 motor order; motors 13, 14 = waist roll / pitch (not in
# HOMIE's model: held at 0)
HOMIE_TO_MOTOR = list(range(13)) + list(range(15, 29))
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14

# unitree WirelessController_.keys bit layout (unitree_sdk2 xKeySwitchUnion)
KEYS = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "up", "right", "down", "left")


def rpy_to_matrix(rpy) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), the Unitree IMU convention (HomieDeploy cheetah_state_estimator)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class RobotState:
    q29: np.ndarray
    dq29: np.ndarray
    q29_prev: np.ndarray    # the previous joint message (5 ms older at the bridge's 200 Hz)
    dq29_prev: np.ndarray
    rpy: np.ndarray
    omega: np.ndarray
    t_body: float           # receive time (time.time()) of the last joint message
    t_imu: float
    lx: float
    ly: float
    rx: float
    ry: float
    keys: int
    t_rc: float

    @property
    def q(self) -> np.ndarray:                 # HOMIE order (27)
        return self.q29[HOMIE_TO_MOTOR]

    @property
    def dq(self) -> np.ndarray:
        return self.dq29[HOMIE_TO_MOTOR]

    @property
    def q_prev(self) -> np.ndarray:
        return self.q29_prev[HOMIE_TO_MOTOR]

    @property
    def dq_prev(self) -> np.ndarray:
        return self.dq29_prev[HOMIE_TO_MOTOR]

    @property
    def gravity(self) -> np.ndarray:           # projected gravity in the pelvis frame
        return rpy_to_matrix(self.rpy).T @ np.array([0.0, 0.0, -1.0])

    @property
    def waist(self) -> np.ndarray:             # yaw, roll, pitch
        return self.q29[[WAIST_YAW, WAIST_ROLL, WAIST_PITCH]]

    def pressed(self, name: str) -> bool:
        return bool(self.keys >> KEYS.index(name) & 1)


class RobotIO:
    def __init__(self, url: str = "udpm://239.255.76.67:7667?ttl=255", publish: bool = True):
        import lcm
        self.lc = lcm.LCM(url)
        self.publish_enabled = bool(publish)
        self._lock = threading.Lock()
        self._q = np.zeros(NUM_MOTORS)
        self._dq = np.zeros(NUM_MOTORS)
        self._q_prev = np.zeros(NUM_MOTORS)
        self._dq_prev = np.zeros(NUM_MOTORS)
        self._rpy = np.zeros(3)
        self._omega = np.zeros(3)
        self._t_body = self._t_imu = self._t_rc = 0.0
        self._sticks = np.zeros(4)
        self._keys = 0
        self.lc.subscribe("body_control_data", self._on_body)
        self.lc.subscribe("state_estimator_data", self._on_imu)
        self.lc.subscribe("rc_command", self._on_rc)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ receive
    def _spin(self):
        while self._running:
            self.lc.handle_timeout(10)

    def _on_body(self, _chan, data):
        msg = body_control_data_lcmt.decode(data)
        with self._lock:
            self._q_prev[:], self._dq_prev[:] = self._q, self._dq
            self._q[:] = msg.q
            self._dq[:] = msg.qd
            self._t_body = time.time()

    def _on_imu(self, _chan, data):
        msg = state_estimator_lcmt.decode(data)
        with self._lock:
            self._rpy[:] = msg.rpy
            self._omega[:] = msg.omegaBody
            self._t_imu = time.time()

    def _on_rc(self, _chan, data):
        msg = rc_command_lcmt.decode(data)
        with self._lock:
            self._sticks[:] = (*msg.left_stick, *msg.right_stick)
            self._keys = int(msg.mode) & 0xFFFF
            self._t_rc = time.time()

    def snapshot(self) -> RobotState:
        with self._lock:
            return RobotState(self._q.copy(), self._dq.copy(), self._q_prev.copy(), self._dq_prev.copy(),
                              self._rpy.copy(), self._omega.copy(), self._t_body,
                              self._t_imu, *map(float, self._sticks), self._keys, self._t_rc)

    def wait_for_state(self, timeout_s: float = 10.0) -> RobotState:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            s = self.snapshot()
            if s.t_body > 0 and s.t_imu > 0:
                return s
            time.sleep(0.01)
        raise TimeoutError("no body_control_data / state_estimator_data on LCM: is g1_control_v81 running?")

    # ------------------------------------------------------------------ send
    def send_joints(self, q_homie27: np.ndarray, waist_roll: float = 0.0, waist_pitch: float = 0.0):
        """Joint position targets in HOMIE order -> legs + waist (pd_plustau_targets) and arms (arm_action)."""
        if not self.publish_enabled:
            return
        q = np.asarray(q_homie27, dtype=float)
        msg = pd_tau_targets_lcmt()
        q_des = [0.0] * NUM_MOTORS
        q_des[:13] = q[:13].tolist()
        q_des[WAIST_ROLL], q_des[WAIST_PITCH] = float(waist_roll), float(waist_pitch)
        msg.q_des = q_des
        msg.tau_ff = [0.0] * NUM_MOTORS
        msg.timestamp_us = int(time.time() * 1e6)
        arm = arm_action_lcmt()
        arm.act = q[13:27].tolist()
        self.lc.publish("arm_action", arm.encode())          # arms first: the bridge's watchdog keys on the legs
        self.lc.publish("pd_plustau_targets", msg.encode())

    def send_hands(self, act14):
        if not self.publish_enabled:
            return
        msg = hand_action_lcmt()
        msg.act = [float(x) for x in act14]
        self.lc.publish("hand_action", msg.encode())

    def close(self):
        self._running = False
