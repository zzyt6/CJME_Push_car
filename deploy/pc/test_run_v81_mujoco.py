"""Rehearse deploy/robot/run_v81.py end to end on the PC: the LCM link (RobotIO) is replaced by a MuJoCo robot (the
V8.1 bench plant), the Unitree remote by a scripted button / stick sequence, and the head camera by UDP datagrams in
tag_camera.py's format built from the MuJoCo poses (60 ms old, 30 Hz). Everything else -- state machine, safety
checks, controller, cart-yaw composition, logging, 50 Hz real-time loop -- is the robot code unchanged.

    python deploy/pc/test_run_v81_mujoco.py

Script: START (ramp, robot hanging) -> R2 (HOMIE; the plant is put in the grasp stance) -> X (hands) -> A (PUSH) ->
hold L1 with the left stick forward for 10 s (vx ~0.29 m/s; right stick turns left in the middle) -> release -> B ->
L2+B (E-STOP). Passes if the run reaches the E-STOP by L2+B without a fall and the cart moved forward.
"""
from __future__ import annotations

import math
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "deploy" / "robot")]

import run_v81                                                               # noqa: E402
from verify_sim2sim import step_recording                                    # noqa: E402
from pushcart_v81.cart_yaw import PACKET, TAG_MOUNTS, pelvis_to_torso       # noqa: E402
from pushcart_v81.robot_io import KEYS, RobotState, rpy_to_matrix           # noqa: E402

SCRIPT = [  # (t_start, t_end, keys, (lx, ly, rx))
    (0.5, 0.6, ("start",), None), (4.0, 4.1, ("R2",), None), (5.0, 5.1, ("X",), None), (6.0, 6.1, ("A",), None),
    (7.0, 17.0, ("L1",), (0.0, 0.75, 0.0)), (11.0, 14.0, ("L1",), (0.0, 0.75, -0.5)),
    (20.0, 20.1, ("B",), None), (21.0, 21.2, ("L2", "B"), None),
]


def quat_to_rpy(q):
    w, x, y, z = q
    return np.array([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
                     math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


class FakeIO:
    """RobotIO over the MuJoCo V8.1 bench plant."""

    instance = None

    def __init__(self, url=None, publish=True):
        from pushcart_bench.policies_v8 import V8Policy
        import json
        FakeIO.instance = self
        self.pol = V8Policy("final")
        self.pol.reset(payload_kg=12.5, caster_angle=0.0, seed=0)
        self.r, self.env = self.pol.runner, self.pol.env
        rc = json.loads((ROOT / "deploy/robot/config/robot_v81.json").read_text(encoding="utf-8"))
        cc = rc["camera"]
        self.R_torso_cam = rpy_to_matrix(cc["cam_rpy_in_torso"]) @ np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]])
        self.R_cart_tag = TAG_MOUNTS[cc["tag_mount"]]
        self.sock, self.port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), cc["udp_port"]
        self.t0 = None                                       # script clock starts at the first state read
        self.hung, self.n_steps, self.hands = True, 0, []
        self.prev = None                                     # joint state before the last 5 ms substep
        self.cam_queue, self.last_cam = [], 0.0
        self.t_body = 0.0
        self.cart_start = None
        self.rel_yaw_true = []

    # ------------------------------------------------------------------ remote script
    def _remote(self, t):
        keys, sticks = 0, (0.0, 0.0, 0.0)
        for a, b, names, st in SCRIPT:
            if a <= t < b:
                for n in names:
                    keys |= 1 << KEYS.index(n)
                if st is not None:
                    sticks = st
        if keys >> KEYS.index("R2") & 1 and self.hung:       # robot put on the ground in the V8.1 grasp stance
            self.hung = False
            self.pol.reset(payload_kg=12.5, caster_angle=0.0, seed=0)
            self.cart_start = self.pol.pose()
        return keys, sticks

    # ------------------------------------------------------------------ RobotIO interface
    def _state(self):
        env = self.env
        idx = list(range(13)) + list(range(15, 29))
        q29, dq29, q29p, dq29p = np.zeros(29), np.zeros(29), np.zeros(29), np.zeros(29)
        q29[idx], dq29[idx] = env.dof_pos, env.dof_vel
        q29p[idx], dq29p[idx] = self.prev if self.prev is not None else (env.dof_pos, env.dof_vel)
        return q29, dq29, q29p, dq29p, quat_to_rpy(env.base_quat_wxyz), env.base_ang_vel.copy()

    def snapshot(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        keys, (lx, ly, rx) = self._remote(now - self.t0)
        q29, dq29, q29p, dq29p, rpy, om = self._state()
        self._camera(now, q29, rpy)
        return RobotState(q29, dq29, q29p, dq29p, rpy, om, self.t_body or now, now, lx, ly, rx, 0.0, keys, now)

    def wait_for_state(self, timeout_s=10.0):
        return self.snapshot()

    def _camera(self, now, q29, rpy):
        """Queue the cart's pose in the camera now; deliver it 60 ms later at 30 Hz (capture time = now)."""
        if self.hung or now - self.last_cam < 1 / 30:
            pass
        else:
            self.last_cam = now
            d = self.env.data
            R_w_cam = rpy_to_matrix(rpy) @ pelvis_to_torso(q29[[12, 13, 14]]) @ self.R_torso_cam
            Rc = np.zeros(9)
            import mujoco
            mujoco.mju_quat2Mat(Rc, d.xquat[self.r.cart_body])
            R_cam_tag = R_w_cam.T @ Rc.reshape(3, 3) @ self.R_cart_tag
            self.cam_queue.append((now, PACKET.pack(now, now + 0.06, *R_cam_tag.ravel(), 0.0, 0.0, 0.6, 0)))
            self.rel_yaw_true.append((now, self.r.ts["rel_yaw"]))
        while self.cam_queue and now - self.cam_queue[0][0] >= 0.06:
            self.sock.sendto(self.cam_queue.pop(0)[1], ("127.0.0.1", self.port))

    def send_joints(self, q, waist_roll=0.0, waist_pitch=0.0):
        if self.hung:
            self.t_body = time.time()
            return
        env, r = self.env, self.r
        n = env.num_lower
        applied = (q[:n] - env.default_dof_pos[:n]) / env.action_scale
        env.upper_delta[:] = (q[n:] - env.default_dof_pos[n:]) / env.action_scale - env.upper_hold_actions
        (_, fallen), self.prev = step_recording(env, env.step, applied)
        r.grasp.update(r.grasp.hand_loads())
        r.casters.update()
        r.task_state()
        self.n_steps += 1
        self.t_body = time.time()

    def send_hands(self, act14):
        self.hands.append(np.array(act14))

    def close(self):
        pass


def main():
    run_v81.RobotIO = FakeIO
    log_dir = ROOT / "logs/validation/v81_run_rehearsal"
    sys.argv = ["run_v81.py", "--log_dir", str(log_dir)]
    t0 = time.time()
    run_v81.main()
    io = FakeIO.instance
    logs = sorted(log_dir.glob("v81_*.npz"))
    z = np.load(logs[-1])
    states = [str(s) for s in z["state"]]
    x0, y0, yaw0 = io.cart_start
    x1, y1, yaw1 = io.pol.pose()
    fwd = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
    push = np.array([s == "PUSH" for s in states])
    tc = z["t_compute"] * 1e3
    dt = np.diff(z["t"])
    yaw_err = []
    tt = np.array([a for a, _ in io.rel_yaw_true])
    for t, v, age in zip(z["t"][push] + float(z["t0"]), z["cam_yaw"][push], z["cam_age"][push]):
        k = int(np.argmin(np.abs(tt - (t - age))))                     # truth at the capture time
        yaw_err.append(abs(v - io.rel_yaw_true[k][1]))
    print(f"\n[rehearsal] {time.time() - t0:.0f} s wall, {io.n_steps} physics ticks, states seen: "
          f"{list(dict.fromkeys(states))}")
    print(f"[rehearsal] cart moved {fwd:+.2f} m along its start heading, yaw {math.degrees(yaw1 - yaw0):+.1f} deg; "
          f"final robot tilt |g_xy| {np.linalg.norm(io.env.projected_gravity[:2]):.3f}")
    print(f"[rehearsal] loop period mean {1e3 * dt.mean():.2f} ms (p99 {1e3 * np.percentile(dt, 99):.2f}), "
          f"compute mean {tc.mean():.2f} ms (p99 {np.percentile(tc, 99):.2f})")
    print(f"[rehearsal] camera yaw vs MuJoCo truth at capture time: max {math.degrees(max(yaw_err)):.3f} deg; "
          f"age in PUSH mean {1e3 * z['cam_age'][push].mean():.0f} ms")
    print(f"[rehearsal] task command max {np.abs(z['task_cmd']).max(0).round(3)}; log {logs[-1]}")
    ok = "PUSH" in states and fwd > 1.0 and np.linalg.norm(io.env.projected_gravity[:2]) < 0.3
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
