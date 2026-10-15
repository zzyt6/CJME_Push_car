"""Relative yaw of the cart (the only exteroceptive V8.1 input), from an AprilTag seen by the head camera.

V8.1 observes rel_yaw = wrap(yaw(cart) - yaw(pelvis)), both world yaws (homie_pushcart.mujoco_hl.task_state). On the
robot the chain is

    R_world_cart = R_world_pelvis(IMU rpy) . R_pelvis_torso(waist yaw, roll, pitch) . R_torso_cam . R_cam_tag . R_tag_cart

and rel_yaw = wrap(yaw(R_world_cart) - IMU yaw - offset). The IMU yaw drift cancels (both yaws come from the same IMU).
The IMU / waist state is taken at the image's capture time (ring buffer), so the observation is the true relative yaw
delayed by the camera latency -- the training camera model (latency 40-100 ms, 25-50 Hz, 1.5 deg noise, 3 % dropout;
values in v81_config.json "camera_training_model"). The last measurement is held between updates and on dropouts, as
in training.

tag_camera.py sends one UDP datagram per detection to 127.0.0.1:<port>:
    struct '<dd9d3di' = t_capture (time.time() clock), t_sent, R_cam_tag (row-major, OpenCV optical frame), t_cam_tag, id
"""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import socket
import struct
import time

import numpy as np

from .robot_io import rpy_to_matrix

PACKET = struct.Struct("<dd9d3di")

# optical frame (x right, y down, z forward) expressed in the camera link frame (x forward, y left, z up)
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])

# tag axes (x right, y up as printed, z out of the printed face) expressed in the cart frame (x forward = pushing
# direction, z up). Columns = x_tag, y_tag, z_tag.
TAG_MOUNTS = {
    # tag on the cart's rear face (the face towards the robot), upright, readable from the robot
    "vertical_rear": np.array([[0.0, 0.0, -1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0]]),
    # tag lying flat on the cart deck, its top edge pointing away from the robot
    "horizontal_top": np.array([[0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]]),
}


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def pelvis_to_torso(waist) -> np.ndarray:
    """G1 waist chain pelvis -> yaw (z) -> roll (x) -> pitch (y) -> torso; joint origins have zero rotation (g1.urdf)."""
    qy, qr, qp = waist
    return rot_z(qy) @ rot_x(qr) @ rot_y(qp)


def load_calibration(path) -> float:
    p = Path(path)
    if not p.is_file():
        return 0.0
    return float(json.loads(p.read_text(encoding="utf-8"))["offset_rad"])


class CartYaw:
    def __init__(self, cam_rpy_in_torso, tag_mount: str, offset_rad: float = 0.0, port: int = 9870,
                 tag_id: int | None = None, outlier_rad: float = 0.8, buffer_s: float = 1.0, dt: float = 0.02):
        self.R_torso_cam = rpy_to_matrix(cam_rpy_in_torso) @ R_LINK_OPTICAL
        self.R_tag_cart = TAG_MOUNTS[tag_mount].T
        self.offset = float(offset_rad)
        self.tag_id, self.outlier = tag_id, float(outlier_rad)
        self.buf = deque(maxlen=max(2, int(buffer_s / dt)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", int(port)))
        self.sock.setblocking(False)
        self.value = 0.0               # held measurement (rad)
        self.raw = 0.0                 # without offset
        self.t_capture = 0.0
        self.t_recv = 0.0
        self.n_ok = self.n_outlier = 0

    def push_state(self, t: float, rpy, waist):
        """Called every control tick with the robot state at time t."""
        self.buf.append((float(t), np.array(rpy, float), np.array(waist, float)))

    def _state_at(self, t: float):
        if not self.buf:
            return None
        return min(self.buf, key=lambda s: abs(s[0] - t))

    def measure(self, R_cam_tag, rpy, waist) -> float:
        R_world_cart = rpy_to_matrix(rpy) @ pelvis_to_torso(waist) @ self.R_torso_cam @ R_cam_tag @ self.R_tag_cart
        return wrap(yaw_of(R_world_cart) - float(rpy[2]))

    def poll(self) -> bool:
        """Read every pending datagram; True when a new measurement was accepted."""
        new = False
        while True:
            try:
                data = self.sock.recv(256)
            except (BlockingIOError, InterruptedError):
                break
            if len(data) != PACKET.size:
                continue
            v = PACKET.unpack(data)
            t_cap, R, tag = v[0], np.array(v[2:11]).reshape(3, 3), v[14]
            if self.tag_id is not None and tag != self.tag_id:
                continue
            st = self._state_at(t_cap)
            if st is None:
                continue
            raw = self.measure(R, st[1], st[2])
            val = wrap(raw - self.offset)
            if abs(val) > self.outlier:                        # tag pose flip / wrong tag: keep the held value
                self.n_outlier += 1
                continue
            self.raw, self.value, self.t_capture, self.t_recv = raw, val, t_cap, time.time()
            self.n_ok += 1
            new = True
        return new

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.t_capture if self.t_capture > 0 else float("inf")
