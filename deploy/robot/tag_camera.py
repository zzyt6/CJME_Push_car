"""Head camera -> AprilTag pose on the cart -> UDP to run_v81.py (one datagram per detection).

    python3 tag_camera.py --tag_size 0.12            # 36h11 tag, id 0, side length of the black square in metres
    python3 tag_camera.py --tag_size 0.12 --show     # with a preview window (needs a display)

RealSense colour stream (the head D455 of HomieDeploy/d455.py; stop d455.py first, the device can only be opened
once). Detection: OpenCV's ArUco module with the AprilTag 36h11 dictionary; pose: solvePnP IPPE_SQUARE with the
camera's factory intrinsics. Packet: struct '<dd9d3di' = t_capture, t_sent (time.time() clock), R_cam_tag row-major
(OpenCV optical frame), t_cam_tag (m), tag id; see pushcart_v81/cart_yaw.py.

V8.1 was trained with a camera latency of 40-100 ms and 25-50 Hz updates; the printed latency statistics must stay in
that range (the capture time is the RealSense frame time when it is in the host clock domain).
"""
from __future__ import annotations

import argparse
import socket
import struct
import time

import numpy as np

PACKET = struct.Struct("<dd9d3di")


def make_detector():
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):                                  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: det.detectMarkers(gray)[:2]
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def main():
    import cv2
    import pyrealsense2 as rs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag_size", type=float, required=True, help="black square side length (m)")
    ap.add_argument("--tag_id", type=int, default=0)
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    pipe, conf = rs.pipeline(), rs.config()
    conf.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, a.fps)
    profile = pipe.start(conf)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs[:5], dtype=float)
    s = a.tag_size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)   # TL, TR, BR, BL
    detect = make_detector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", a.port)
    print(f"[tag] {a.width}x{a.height}@{a.fps} fx {intr.fx:.1f}, tag 36h11 id {a.tag_id} size {a.tag_size} m -> UDP {a.port}")

    lat, n_frames, n_det, t_report = [], 0, 0, time.time()
    try:
        while True:
            frames = pipe.wait_for_frames()
            t_arrive = time.time()
            color = frames.get_color_frame()
            if not color:
                continue
            t_cap = t_arrive
            if color.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                t_cap = color.get_timestamp() / 1000.0                  # host-synchronised capture time
            img = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray)
            n_frames += 1
            if ids is not None:
                for c, i in zip(corners, ids.ravel()):
                    if int(i) != a.tag_id:
                        continue
                    ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, dist,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok:
                        continue
                    R, _ = cv2.Rodrigues(rvec)
                    t_sent = time.time()
                    sock.sendto(PACKET.pack(t_cap, t_sent, *R.ravel(), *tvec.ravel(), int(i)), dest)
                    lat.append(t_sent - t_cap)
                    n_det += 1
                    if a.show:
                        cv2.drawFrameAxes(img, K, dist, rvec, tvec, a.tag_size)
            if a.show:
                cv2.imshow("tag", img)
                cv2.waitKey(1)
            if time.time() - t_report > 2.0:
                dtr = time.time() - t_report
                l = np.array(lat) * 1e3 if lat else np.array([np.nan])
                print(f"[tag] {n_frames / dtr:4.1f} fps, detections {n_det / dtr:4.1f} Hz, "
                      f"capture->send {np.nanmean(l):5.1f} ms (max {np.nanmax(l):5.1f})", flush=True)
                lat, n_frames, n_det, t_report = [], 0, 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
