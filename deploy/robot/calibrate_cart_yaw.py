"""Calibrate the cart-yaw offset: with the cart held straight in front of the robot (handle square to the robot, the
V8.1 nominal grasp, where the simulated rel_yaw is v81_config.json "rel_yaw_at_nominal_grasp_rad" = 0), record the
measured relative yaw and store offset = mean - nominal in calib/cart_yaw.json.

    python3 calibrate_cart_yaw.py --seconds 3          # g1_control_v81 + tag_camera.py running; read-only (no commands)

It also prints the noise (std) of the measurement; V8.1 was trained with 1.5 deg per measurement.
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

from pushcart_v81.cart_yaw import CartYaw          # noqa: E402
from pushcart_v81.robot_io import RobotIO          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--config", default=str(HERE / "config" / "robot_v81.json"))
    ap.add_argument("--models", default=str(HERE / "models"))
    a = ap.parse_args()
    rc = json.loads(Path(a.config).read_text(encoding="utf-8"))
    nominal = json.loads((Path(a.models) / "v81_config.json").read_text(encoding="utf-8"))["rel_yaw_at_nominal_grasp_rad"]
    cc = rc["camera"]
    io = RobotIO(rc["lcm_url"], publish=False)
    io.wait_for_state()
    cam = CartYaw(cc["cam_rpy_in_torso"], cc["tag_mount"], 0.0, cc["udp_port"], cc["tag_id"], outlier_rad=math.pi)
    raw, t_end = [], time.time() + a.seconds
    while time.time() < t_end:
        s = io.snapshot()
        cam.push_state(time.time(), s.rpy, s.waist)
        if cam.poll():
            raw.append(cam.raw)
        time.sleep(0.005)
    io.close()
    if len(raw) < 10:
        raise SystemExit(f"only {len(raw)} measurements: is tag_camera.py running and the tag visible?")
    raw = np.unwrap(np.array(raw))
    offset = float(np.mean(raw) - nominal)
    out = HERE / cc["calibration_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"offset_rad": offset, "offset_deg": math.degrees(offset), "n": len(raw),
                               "std_deg": math.degrees(float(np.std(raw))), "rate_hz": len(raw) / a.seconds,
                               "tag_mount": cc["tag_mount"], "cam_rpy_in_torso": cc["cam_rpy_in_torso"],
                               "date": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
    print(f"offset {math.degrees(offset):+.2f} deg, noise std {math.degrees(np.std(raw)):.2f} deg (training: 1.5), "
          f"{len(raw) / a.seconds:.1f} Hz (training: 25-50) -> {out}")
    if abs(offset) > math.radians(15):
        print("WARNING: offset > 15 deg: check tag_mount / cam_rpy_in_torso / that the cart is square to the robot")


if __name__ == "__main__":
    main()
