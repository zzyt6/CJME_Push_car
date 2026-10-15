"""V8.1 push-cart controller on the real G1 (runs on the robot's Orin, next to g1_control_v81 and hand_control).

    python3 run_v81.py                       # normal run (tag_camera.py must be running for PUSH)
    python3 run_v81.py --dry_run             # everything except publishing (check sensors, camera, timing)
    python3 run_v81.py --print_buttons       # show the remote's keys / sticks only

State machine (Unitree remote):
    WAIT   --start-->  RAMP (stand_ramp_s, robot hanging)  -->  STAND (holds the stand / grasp-hold pose)
    STAND  --R2----->  HOMIE   plain HOMIE at zero command, arms at the V8.1 grasp-hold pose (lower the gantry)
    HOMIE  --X / Y-->  close / open the Dex-3 hands on the handle
    HOMIE  --A------>  PUSH    V8.1 full stack (needs: hands closed, fresh cart yaw)
    PUSH   --hold L1->  task command from the sticks (left: vx / vy, right x: wz); released -> 0
    PUSH   --B------>  HOMIE   (only once the reference has decayed to standstill)
    any    --L2+B--->  E-STOP  (stop publishing; g1_control_v81 latches damping itself as well)
Automatic E-STOP: robot tilt |g_xy| > tilt_limit, joint data older than state_timeout_s, Ctrl-C.
E-STOP = damping: the robot goes limp. Always run with the gantry / a spotter.

Every tick is logged to logs/v81_<time>.npz (sensors, camera, commands, frames, actions, targets, timing).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pushcart_v81.controller import MODE_HOMIE, MODE_PUSH, OnnxModels, Sensors, V81Controller, load_config  # noqa: E402
from pushcart_v81.robot_io import KEYS, WAIST_PITCH, WAIST_ROLL, RobotIO                               # noqa: E402


def deadzone(x: float, dz: float) -> float:
    return 0.0 if abs(x) < dz else math.copysign((abs(x) - dz) / (1.0 - dz), x)


def stick_command(s, js) -> np.ndarray:
    """Cart-frame task command (vx, vy, wz) from the sticks: left up = forward, left = +vy, right stick left = +wz."""
    ly, lx, rx = deadzone(s.ly, js["deadzone"]), deadzone(s.lx, js["deadzone"]), deadzone(s.rx, js["deadzone"])
    vx = ly * (js["vx_max"] if ly >= 0 else -js["vx_min"])
    return np.array([vx, -lx * js["vy_max"], -rx * js["wz_max"]])


class Log:
    """Per-tick arrays (float32); "t" is seconds since t0 (float64, saved as "t0" = time.time() at the start)."""

    def __init__(self, path: Path):
        self.path, self.rows, self.t0 = path, {}, time.time()

    def add(self, t, **kw):
        self.rows.setdefault("t", []).append(float(t - self.t0))
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float32) if not isinstance(v, str) else v)

    def save(self):
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, t0=self.t0, **{k: np.array(v) for k, v in self.rows.items()})
        print(f"[v81] log: {self.path} ({len(self.rows.get('t', []))} ticks)", flush=True)


def print_buttons(io: RobotIO):
    print("Press keys on the Unitree remote (Ctrl-C to quit). Expected names:", ", ".join(KEYS))
    while True:
        s = io.snapshot()
        down = [k for k in KEYS if s.pressed(k)]
        print(f"\rkeys {s.keys:016b} {' '.join(down):30s} lx {s.lx:+.2f} ly {s.ly:+.2f} rx {s.rx:+.2f} ry {s.ry:+.2f}",
              end="", flush=True)
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(HERE / "models"))
    ap.add_argument("--config", default=str(HERE / "config" / "robot_v81.json"))
    ap.add_argument("--dry_run", action="store_true", help="do not publish any command")
    ap.add_argument("--print_buttons", action="store_true")
    ap.add_argument("--no_camera", action="store_true", help="no tag camera (HOMIE mode only)")
    ap.add_argument("--allow_push_without_camera", action="store_true",
                    help="hanging pipeline test only: PUSH with cart yaw = 0")
    ap.add_argument("--log_dir", default=str(HERE / "logs"))
    args = ap.parse_args()

    rc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    io = RobotIO(rc["lcm_url"], publish=not args.dry_run)
    if args.print_buttons:
        try:
            print_buttons(io)
        except KeyboardInterrupt:
            return 0
    cfg = load_config(args.models)
    ctl = V81Controller(cfg, OnnxModels(args.models, threads=rc["onnx_threads"]))
    for _ in range(20):                                  # warm up onnxruntime (first calls are slow)
        ctl.models.run_student(np.zeros(630, np.float32), np.zeros(4032, np.float32))
        ctl.models.run_homie(np.zeros(456, np.float32), True)
    ctl.reset_homie()
    ctl.reset_task()

    cam = None
    if not args.no_camera:
        from pushcart_v81.cart_yaw import CartYaw, load_calibration
        cc = rc["camera"]
        offset = load_calibration(HERE / cc["calibration_file"])
        cam = CartYaw(cc["cam_rpy_in_torso"], cc["tag_mount"], offset, cc["udp_port"], cc["tag_id"], cc["outlier_rad"],
                      dt=cfg["dt"])
        print(f"[v81] camera: UDP {cc['udp_port']}, tag {cc['tag_id']} ({cc['tag_mount']}), offset {offset:+.4f} rad")
        if offset == 0.0:
            print("[v81] WARNING: no cart-yaw calibration file; run calibrate_cart_yaw.py first")

    js, hands_cfg, safety = rc["joystick"], rc["hands"], rc["safety"]
    tr = cfg["task_command_range"]
    js["vx_max"], js["vx_min"] = min(js["vx_max"], tr["vx"][1]), max(js["vx_min"], tr["vx"][0])
    js["vy_max"], js["wz_max"] = min(js["vy_max"], tr["vy_max"]), min(js["wz_max"], tr["wz_max"])
    hand_open = np.array(hands_cfg["open"], float)
    hand_grasp = (np.array(cfg["finger_grasp_fit"], float) if hands_cfg["grasp"] == "fit"
                  else np.array(hands_cfg[hands_cfg["grasp"]], float))
    lo, hi = np.array(cfg["joint_lower"]), np.array(cfg["joint_upper"])
    default = np.array(cfg["default_dof_pos"])
    q_stand = np.concatenate([default[:cfg["num_lower"]], cfg["hold_target_q"]])
    dt = float(cfg["dt"])

    print("[v81] waiting for g1_control_v81 ...", flush=True)
    s = io.wait_for_state()
    print(f"[v81] state ok: rpy {np.round(s.rpy, 3)}, models {cfg['provenance']['student_checkpoint']}")
    print("[v81] WAIT: hang the robot, press START to ramp to the stand pose", flush=True)

    log = Log(Path(args.log_dir) / time.strftime("v81_%Y%m%d_%H%M%S.npz"))
    state, prev_keys = "WAIT", s.keys
    ramp_q0 = ramp_t = None
    hands_target, hands_cmd, hands_active, hands_closed = hand_open.copy(), None, False, False
    estop_reason = None
    next_t = time.perf_counter()
    last_print = 0.0
    try:
        while True:
            t_loop = time.perf_counter()
            now = time.time()
            s = io.snapshot()
            prev, prev_keys = prev_keys, s.keys
            edge = lambda k, p=prev: s.pressed(k) and not (p >> KEYS.index(k) & 1)       # noqa: E731

            # ---------------------------------------------------------------- safety
            if s.pressed("L2") and s.pressed("B"):
                estop_reason = "L2+B"
            elif now - s.t_body > safety["state_timeout_s"] and state != "WAIT":
                estop_reason = f"joint data {1e3 * (now - s.t_body):.0f} ms old"
            elif state in ("HOMIE", "PUSH") and np.linalg.norm(s.gravity[:2]) > safety["tilt_limit"]:
                estop_reason = f"tilt |g_xy| = {np.linalg.norm(s.gravity[:2]):.2f}"
            if estop_reason:
                break

            if cam is not None:
                cam.push_state(now, s.rpy, s.waist)
                cam.poll()
            cam_age = cam.age(now) if cam is not None else float("inf")
            cam_ok = cam is not None and cam_age < rc["camera"]["timeout_s"]

            # ---------------------------------------------------------------- hands
            if edge("X") and state in ("STAND", "HOMIE"):
                hands_target, hands_active, hands_closed = hand_grasp.copy(), True, True
                print("[v81] hands: close", flush=True)
            if edge("Y") and state in ("STAND", "HOMIE"):
                hands_target, hands_active, hands_closed = hand_open.copy(), True, False
                print("[v81] hands: open", flush=True)
            if hands_active:
                if hands_cmd is None:                    # hand_control starts in its closed default ("fist")
                    hands_cmd = np.array(hands_cfg["fist"], float)
                step = np.maximum(np.abs(hand_grasp - hand_open), 0.5) * dt / max(hands_cfg["ramp_s"], dt)
                hands_cmd = hands_cmd + np.clip(hands_target - hands_cmd, -step, step)
                io.send_hands(hands_cmd)

            # ---------------------------------------------------------------- state machine
            q_target, out, task_cmd, waist_rp = None, None, np.zeros(3), np.zeros(2)
            if state == "WAIT":
                if edge("start"):
                    ramp_q0, ramp_t = s.q.copy(), 0.0
                    ramp_w0 = s.q29[[WAIST_ROLL, WAIST_PITCH]].copy()
                    state = "RAMP"
                    print("[v81] RAMP to the stand pose", flush=True)
            elif state == "RAMP":
                ramp_t += dt
                a = min(1.0, ramp_t / rc["stand_ramp_s"])
                q_target = ramp_q0 + a * (q_stand - ramp_q0)
                waist_rp = (1.0 - a) * ramp_w0
                if a >= 1.0:
                    state = "STAND"
                    print("[v81] STAND: lower the robot onto its feet, press R2 to start HOMIE", flush=True)
            elif state == "STAND":
                q_target = q_stand
                if edge("R2"):
                    ctl.reset_homie()
                    ctl.reset_task()
                    state = "HOMIE"
                    print("[v81] HOMIE (zero command). X/Y close/open hands, A = PUSH", flush=True)
            if state in ("HOMIE", "PUSH"):
                if state == "HOMIE" and edge("A"):
                    if not hands_closed:
                        print("[v81] PUSH refused: close the hands first (X)", flush=True)
                    elif not (cam_ok or args.allow_push_without_camera):
                        print(f"[v81] PUSH refused: no fresh cart yaw (age {cam_age:.2f} s)", flush=True)
                    else:
                        ctl.reset_task()
                        state = "PUSH"
                        print("[v81] PUSH: hold L1 and use the sticks; B = back to HOMIE when stopped", flush=True)
                elif state == "PUSH" and edge("B"):
                    if np.linalg.norm(ctl.cmd_ref) < 0.05 and not s.pressed("L1"):
                        state = "HOMIE"
                        print("[v81] HOMIE", flush=True)
                    else:
                        print("[v81] release L1 and wait for the cart to stop before B", flush=True)
                if state == "PUSH":
                    if s.pressed("L1") and (cam_ok or args.allow_push_without_camera):
                        task_cmd = stick_command(s, js)
                    yaw = cam.value if cam is not None else 0.0
                    out = ctl.step(Sensors(s.q, s.dq, s.omega, s.gravity, s.q_prev, s.dq_prev), task_cmd, yaw, MODE_PUSH)
                else:
                    out = ctl.step(Sensors(s.q, s.dq, s.omega, s.gravity, s.q_prev, s.dq_prev), np.zeros(3), 0.0, MODE_HOMIE)
                q_target = out.q_target

            if q_target is not None:
                q_send = np.clip(q_target, lo, hi)
                io.send_joints(q_send, *waist_rp)

            # ---------------------------------------------------------------- log, status, timing
            t_compute = time.perf_counter() - t_loop
            if state != "WAIT":
                log.add(t=now, state=state, q29=s.q29, dq29=s.dq29, rpy=s.rpy, omega=s.omega, keys=s.keys,
                        sticks=[s.lx, s.ly, s.rx, s.ry], task_cmd=task_cmd,
                        cam_yaw=cam.value if cam is not None else 0.0, cam_raw=cam.raw if cam is not None else 0.0,
                        cam_age=min(cam_age, 99.0), q_target=q_target if q_target is not None else np.zeros(27),
                        cmd_homie=out.cmd_homie if out else np.zeros(3), cmd_ref=out.cmd_ref if out else np.zeros(3),
                        hl_actions=out.hl_actions if out else np.zeros(18),
                        leg_residual=out.leg_residual if out else np.zeros(12),
                        homie_legs=out.homie_legs if out else np.zeros(12), vel_used=out.vel_used if out else np.zeros(3),
                        est=out.est if out else np.zeros(19),
                        frame=ctl.frames[-1] if state == "PUSH" else np.zeros(ctl.fd),
                        hands=hands_cmd if hands_cmd is not None else np.zeros(14), t_compute=t_compute)
            if now - last_print > 0.5:
                last_print = now
                extra = ""
                if out is not None:
                    extra = (f" task {np.round(task_cmd, 2)} ref {np.round(out.cmd_ref, 2)} homie {np.round(out.cmd_homie, 2)}"
                             f" | est cart vx {out.est[0] / 2:+.2f} wz {out.est[2] / 0.5:+.2f}")
                camtxt = f" | yaw {math.degrees(cam.value):+5.1f} deg age {1e3 * min(cam_age, 9.99):4.0f} ms" if cam else ""
                print(f"[v81] {state:5s}{extra}{camtxt} | compute {1e3 * t_compute:4.1f} ms", flush=True)

            next_t += dt
            lag = time.perf_counter() - next_t
            if lag > safety["max_overrun_ticks"] * dt:
                print(f"[v81] WARNING: loop {1e3 * lag:.0f} ms late, schedule reset", flush=True)
                next_t = time.perf_counter()
            while True:
                remaining = next_t - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(remaining - 0.0005 if remaining > 0.001 else 0)
    except KeyboardInterrupt:
        estop_reason = "Ctrl-C"
    finally:
        # stop publishing: once it has received a command, g1_control_v81's watchdog enters damping after 100 ms
        print(f"\n[v81] E-STOP ({estop_reason}): publishing stopped"
              + (" -> damping" if state != "WAIT" else " (nothing was sent)"), flush=True)
        log.save()
        io.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
