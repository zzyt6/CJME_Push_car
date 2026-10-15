"""Synthetic end-to-end check of the cart-yaw chain (no robot): render an AprilTag 36h11 into the head camera's image
for known robot / waist / cart poses, detect it with tag_camera.py's detector, solve its pose as tag_camera.py does,
and compare CartYaw.measure (the robot-side composition) with the training definition
rel_yaw = wrap(yaw(cart) - yaw(pelvis)).

    python deploy/pc/test_cart_yaw.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "robot"))

from pushcart_v81.cart_yaw import TAG_MOUNTS, CartYaw, pelvis_to_torso, wrap      # noqa: E402
from pushcart_v81.robot_io import rpy_to_matrix                                  # noqa: E402
from tag_camera import make_detector                                             # noqa: E402

W, H, F = 640, 480, 610.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
TAG = 0.12
CAM_RPY = [0.0, 0.8307767239493009, 0.0]
CAM_POS_TORSO = np.array([0.0576235, 0.01753, 0.42987])          # g1.urdf d435_joint
TORSO_POS_PELVIS = np.array([-0.0039635, 0.0, 0.044])            # g1.urdf waist_roll_joint origin


def render(R_cam_tag, t_cam_tag, marker_px=200):
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0,
                                           marker_px)
    # marker pixel (u, v) -> tag plane (x right, y up), metres
    A = np.array([[TAG / marker_px, 0, -TAG / 2], [0, -TAG / marker_px, TAG / 2], [0, 0, 1]])
    Hm = K @ np.column_stack((R_cam_tag[:, 0], R_cam_tag[:, 1], t_cam_tag)) @ A
    img = cv2.warpPerspective(marker, Hm, (W, H), borderValue=255)
    # warpPerspective fills outside the marker with white: the marker's own quiet zone is the white background
    return img


def main(trials=200, seed=0):
    rng = np.random.default_rng(seed)
    yaw = CartYaw(CAM_RPY, "vertical_rear", 0.0, port=0)
    detect = make_detector()
    s = TAG / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    errs, missed = [], 0
    R_cart_tag = TAG_MOUNTS["vertical_rear"]
    for _ in range(trials):
        rpy = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.12), rng.uniform(-math.pi, math.pi)])
        waist = np.array([rng.uniform(-0.3, 0.3), 0.0, 0.0])
        rel_true = rng.uniform(-0.5, 0.5)
        R_w_pelvis = rpy_to_matrix(rpy)
        p_pelvis = np.array([0.0, 0.0, 0.76])
        R_w_torso = R_w_pelvis @ pelvis_to_torso(waist)
        p_torso = p_pelvis + R_w_pelvis @ TORSO_POS_PELVIS
        R_w_cam = R_w_torso @ yaw.R_torso_cam                        # optical frame
        p_cam = p_torso + R_w_torso @ CAM_POS_TORSO
        # cart: level, yaw = pelvis yaw + rel_true; tag on its rear face, 0.55 m ahead of the pelvis, 0.2 m below the
        # camera (in view of the 47.6 deg pitched-down camera)
        cy = rpy[2] + rel_true
        R_w_cart = rpy_to_matrix([0.0, 0.0, cy])
        fwd = R_w_pelvis[:, 0] * np.array([1, 1, 0])
        fwd /= np.linalg.norm(fwd)
        p_tag = p_pelvis + 0.55 * fwd + np.array([0, 0, 0.12]) + rng.normal(0, 0.03, 3)
        R_w_tag = R_w_cart @ R_cart_tag
        R_cam_tag = R_w_cam.T @ R_w_tag
        t_cam_tag = R_w_cam.T @ (p_tag - p_cam)
        if t_cam_tag[2] < 0.2:
            continue
        img = render(R_cam_tag, t_cam_tag)
        corners, ids = detect(img)
        if ids is None or 0 not in ids.ravel():
            missed += 1
            continue
        c = corners[list(ids.ravel()).index(0)]
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, np.zeros(5),
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        R_est, _ = cv2.Rodrigues(rvec)
        meas = yaw.measure(R_est, rpy, waist)
        errs.append(math.degrees(wrap(meas - rel_true)))
        # the composition alone (exact R_cam_tag) must be exact
        assert abs(wrap(yaw.measure(R_cam_tag, rpy, waist) - rel_true)) < 1e-9
    e = np.abs(np.array(errs))
    print(f"{len(errs)} detections ({missed} missed): |measured - true rel_yaw| mean {e.mean():.2f} deg, "
          f"p95 {np.percentile(e, 95):.2f} deg, max {e.max():.2f} deg  (training noise model: 1.5 deg)")
    print("composition with the exact tag rotation: exact (|err| < 1e-9 rad) on every trial")
    return 0 if np.percentile(e, 95) < 3.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
