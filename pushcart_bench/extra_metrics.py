"""Interaction, balance and force-estimation metrics from full-rate (200 Hz) grasp-wrench recordings.

Used on top of the standard bench metrics. ``Recorder`` wraps a cart policy whose
environment is ``instrumentation.InstrumentedEnv`` and stores, per control step, the four physics-step
weld solves. Everything is expressed in upright cart-heading axes unless stated otherwise.

Definitions (all window statistics over the caller's index window):
  handle wrench       hand-on-cart resultant [Fx, Fy, Mz] at the handle centre; 50 Hz = mean of 4 substeps
  band RMS            zero-phase 2nd-order Butterworth 0.5-5 Hz on the 50 Hz signal (training ripple band)
  HF RMS              zero-phase high-pass 5 Hz on the 200 Hz substep signal (content above the ripple band)
  gait fraction       Welch power share of 1-3 Hz within 0.3-25 Hz (stepping frequency band)
  squeeze / diff push 0.5 (F_left - F_right) along cart y / x (internal lateral force / yaw-producing push)
  ZMP offset          quasi-static required foot-ZMP shift caused by the hand reactions (zero CoM acceleration
                      and angular-momentum rate): (z - c)_x = sum(p_z f_x - (p_x - c_x) f_z + n_y) / (m g - sum f_z),
                      (z - c)_y = sum(p_z f_y - (p_y - c_y) f_z - n_x) / (m g - sum f_z), with f, n the loads on the
                      robot (= minus hand-on-cart) at palm origins; robot heading axes.
  efficiency          mean cart work rate (handle force . cart velocity) / mean joint |tau . qdot|
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

G = 9.81
SIDES = ("left", "right")


def rotate_xy(values, yaw, inverse=True):
    """Rotate [..., 3] (x, y rotated; z kept) by -yaw (inverse) or +yaw; yaw broadcast over leading dims."""
    angle = -yaw if inverse else yaw
    c, s = np.cos(angle), np.sin(angle)
    out = np.array(values, dtype=float, copy=True)
    out[..., 0] = c * values[..., 0] - s * values[..., 1]
    out[..., 1] = s * values[..., 0] + c * values[..., 1]
    return out


class Recorder:
    """Per-control-step capture after every ``policy.step``; cleared on ``policy.reset``."""

    def __init__(self, policy):
        self.policy = policy
        self.clear()
        step, reset = policy.step, policy.reset

        def wrapped_step(cmd):
            step(cmd)
            self.capture()

        def wrapped_reset(*args, **kwargs):
            reset(*args, **kwargs)
            self.clear()

        policy.step, policy.reset = wrapped_step, wrapped_reset

    def clear(self):
        self.rows = {k: [] for k in ("hands", "palm", "handle", "power", "yaw", "vel", "cmd_ref", "com", "pelvis_yaw",
                                     "tilt", "gyro", "cart_vel_w", "pred_resultant", "pred_hands", "prior")}

    def capture(self):
        p = self.policy
        env, runner = p.env, p.runner
        d = env.data
        r = self.rows
        r["hands"].append(np.asarray(env.sub_hands))         # [4, 2, 6] world, palm-origin moments
        r["palm"].append(np.asarray(env.sub_palm))           # [4, 2, 3]
        r["handle"].append(np.asarray(env.sub_handle))       # [4, 6] world, handle-centre moment
        r["power"].append(np.asarray(env.sub_power))         # [4]
        x, y, yaw = p.pose()
        vel = p.velocity()
        r["yaw"].append(yaw)
        r["vel"].append(vel)
        r["cart_vel_w"].append(rotate_xy(np.array([vel[0], vel[1], 0.]), np.float64(yaw), inverse=False))
        r["cmd_ref"].append(np.asarray(runner.cmd_ref, dtype=float).copy())
        r["com"].append(d.subtree_com[env.pelvis_id].copy())
        q = env.base_quat_wxyz
        r["pelvis_yaw"].append(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        r["tilt"].append(float(np.arcsin(np.clip(np.linalg.norm(env.projected_gravity[:2]), 0, 1))))
        r["gyro"].append(float(np.linalg.norm(env.base_ang_vel[:2])))
        r["pred_resultant"].append(np.asarray(getattr(runner, "pred_resultant", np.full(3, np.nan)), dtype=float).copy())
        r["pred_hands"].append(np.asarray(getattr(runner, "pred_hands", np.full(12, np.nan)), dtype=float).copy())
        r["prior"].append(np.asarray(getattr(runner, "prior", np.full(12, np.nan)), dtype=float).copy())

    def arrays(self) -> dict:
        out = {k: np.asarray(v, dtype=float) for k, v in self.rows.items()}
        if not len(out["yaw"]):
            return out
        yaw = out["yaw"]
        # Cart-heading-axis signals.
        handle = out["handle"]                                             # [T, 4, 6]
        h = np.stack((handle[..., 0], handle[..., 1], handle[..., 5]), -1)
        out["F_sub"] = rotate_xy(h, yaw[:, None])                          # [T, 4, 3] Fx, Fy, Mz
        out["F"] = out["F_sub"].mean(1)                                    # [T, 3]
        hands = out["hands"]                                               # [T, 4, 2, 6]
        force = rotate_xy(hands[..., :3], yaw[:, None, None])
        moment = rotate_xy(hands[..., 3:], yaw[:, None, None])
        out["hands_c_sub"] = np.concatenate((force, moment), -1)           # [T, 4, 2, 6]
        out["hands_c"] = out["hands_c_sub"].mean(1)                        # [T, 2, 6]
        out["zmp"] = zmp_offset(hands, out["palm"], out["com"], out["pelvis_yaw"], self.policy.env.robot_mass)
        return out


def zmp_offset(hands, palm, com, pelvis_yaw, robot_mass):
    """Quasi-static hand-reaction ZMP offset [T, 2] (robot heading axes), interval-mean over substeps."""
    f, n = -hands[..., :3], -hands[..., 3:]                               # loads on the robot
    p = palm                                                               # [T, 4, 2, 3]
    c = com[:, None, None, :]
    num_x = (p[..., 2] * f[..., 0] - (p[..., 0] - c[..., 0]) * f[..., 2] + n[..., 1]).sum(-1)
    num_y = (p[..., 2] * f[..., 1] - (p[..., 1] - c[..., 1]) * f[..., 2] - n[..., 0]).sum(-1)
    den = robot_mass * G - f[..., 2].sum(-1)
    world = np.stack((num_x / den, num_y / den, np.zeros_like(den)), -1).mean(1)
    return rotate_xy(world, np.asarray(pelvis_yaw))[:, :2]


def _bandpass(x, fs, lo, hi):
    if len(x) < 30:
        return np.full_like(x, np.nan)
    sos = signal.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _highpass(x, fs, lo):
    if len(x) < 30:
        return np.full_like(x, np.nan)
    sos = signal.butter(2, lo, btype="high", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x ** 2))) if x.size else None


def _band_fraction(x, fs, band=(1.0, 3.0), total=(0.3, 25.0)):
    if len(x) < 64:
        return None
    f, p = signal.welch(x - np.mean(x), fs=fs, nperseg=min(256, len(x)))
    sel_t = (f >= total[0]) & (f < total[1])
    sel_b = (f >= band[0]) & (f < band[1])
    denom = p[sel_t].sum()
    return float(p[sel_b].sum() / denom) if denom > 0 else None


def window_metrics(a: dict, i0: int, i1: int, dt: float, cart_mass: float, robot_mass: float) -> dict:
    """Interaction ripple, internal force, ZMP offset, balance and efficiency on control steps [i0, i1)."""
    T = len(a.get("yaw", []))
    i0, i1 = max(0, i0), min(T, i1)
    if i1 - i0 < 30:
        return {}
    fs, fs_sub = 1. / dt, 4. / dt
    F = a["F"][i0:i1]
    F_sub = a["F_sub"][i0:i1].reshape(-1, 3)
    hands = a["hands_c"][i0:i1]
    out = {}
    band = _bandpass(F, fs, 0.5, 5.0)
    hf = _highpass(F_sub, fs_sub, 5.0)
    for j, name in enumerate(("Fx", "Fy", "Mz")):
        out[f"handle_{name}_mean"] = float(F[:, j].mean())
        out[f"handle_{name}_band_rms"] = _rms(band[:, j])
        out[f"handle_{name}_hf_rms"] = _rms(hf[:, j])
        out[f"handle_{name}_rate_rms"] = _rms(np.diff(F[:, j]) / dt)
        std = float(F[:, j].std())
        out[f"handle_{name}_crest"] = float(np.max(np.abs(F[:, j] - np.median(F[:, j]))) / std) if std > 1e-9 else None
    out["handle_Fx_gait_frac"] = _band_fraction(F[:, 0], fs)
    out["handle_Mz_gait_frac"] = _band_fraction(F[:, 2], fs)
    squeeze = 0.5 * (hands[:, 0, 1] - hands[:, 1, 1])
    diff_push = 0.5 * (hands[:, 0, 0] - hands[:, 1, 0])
    out.update(squeeze_y_mean=float(squeeze.mean()), squeeze_y_std=float(squeeze.std()),
               diff_push_x_std=float(diff_push.std()))
    norm = np.linalg.norm(a["hands_c_sub"][i0:i1, ..., :3], axis=-1)      # [T, 4, 2]
    out["hand_force_p95"] = float(np.percentile(norm.mean(1).max(-1), 95))
    out["hand_force_peak_200hz"] = float(norm.max())
    zmp = a["zmp"][i0:i1]
    zband = _bandpass(zmp, fs, 0.5, 5.0)
    out.update(zmp_offset_x_mean=float(zmp[:, 0].mean()), zmp_offset_y_mean=float(zmp[:, 1].mean()),
               zmp_offset_band_rms=_rms(np.linalg.norm(zband, axis=-1)),
               zmp_offset_abs_p95=float(np.percentile(np.linalg.norm(zmp, axis=-1), 95)))
    out.update(robot_tilt_max=float(a["tilt"][i0:i1].max()), robot_tilt_p95=float(np.percentile(a["tilt"][i0:i1], 95)),
               robot_gyro_xy_rms=_rms(a["gyro"][i0:i1]))
    power = float(a["power"][i0:i1].mean())
    handle_w = a["handle"][i0:i1, :, :3].mean(1)
    cart_power = float(np.mean(np.sum(handle_w * a["cart_vel_w"][i0:i1], -1)))
    speed = float(np.mean(np.linalg.norm(a["vel"][i0:i1, :2], axis=-1)))
    out.update(mech_power_mean_W=power, cart_power_mean_W=cart_power,
               push_efficiency=cart_power / power if power > 1e-6 and speed > 0.05 else None,
               cost_of_transport=power / ((robot_mass + cart_mass) * G * speed) if speed > 0.05 else None,
               cart_speed_mean=speed)
    return out


def estimator_series(a: dict, i0: int, i1: int) -> dict | None:
    """Prediction / truth pairs for pooled accuracy statistics (None for policies without an estimator)."""
    pred = a["pred_resultant"][i0:i1]
    if not np.isfinite(pred).any():
        return None
    out = {"pred_resultant": pred, "true_resultant_mean": a["F"][i0:i1],
           "true_resultant_last": a["F_sub"][i0:i1, -1]}
    hands = a["pred_hands"][i0:i1]
    if np.isfinite(hands).any():
        truth = a["hands_c"][i0:i1]                                         # [T, 2, 6]
        out["pred_hands"] = hands
        out["true_hands"] = truth.reshape(len(truth), 12)
        out["prior"] = a["prior"][i0:i1]
    return out


def estimator_accuracy(series: list[dict]) -> dict:
    """Pooled R2 / RMSE / zero-lag correlation / best lag over all supplied runs."""
    series = [s for s in series if s is not None]
    if not series:
        return {}
    out = {}

    def stats(pred, true, names, prefix):
        m = np.isfinite(pred).all(-1) & np.isfinite(true).all(-1)
        p, t = pred[m], true[m]
        err = ((p - t) ** 2).mean(0)
        var = t.var(0)
        for j, n in enumerate(names):
            out[f"{prefix}_{n}_r2"] = float(1 - err[j] / var[j]) if var[j] > 1e-12 else None
            out[f"{prefix}_{n}_rmse"] = float(np.sqrt(err[j]))
            out[f"{prefix}_{n}_true_std"] = float(np.sqrt(var[j]))
            out[f"{prefix}_{n}_gain"] = float(np.polyfit(t[:, j], p[:, j], 1)[0]) if var[j] > 1e-12 else None

    cat = lambda k: np.concatenate([s[k] for s in series if k in s])  # noqa: E731
    stats(cat("pred_resultant"), cat("true_resultant_mean"), ("Fx", "Fy", "Mz"), "est_handle_vs_mean")
    stats(cat("pred_resultant"), cat("true_resultant_last"), ("Fx", "Fy", "Mz"), "est_handle_vs_last")
    lags = {}
    for j, n in enumerate(("Fx", "Fy", "Mz")):
        best, best_c = 0, -2.
        for k in range(-5, 6):
            ps, ts = [], []
            for s in series:
                p, t = s["pred_resultant"][:, j], s["true_resultant_mean"][:, j]
                if k >= 0:
                    ps.append(p[k:]); ts.append(t[:len(t) - k])
                else:
                    ps.append(p[:len(p) + k]); ts.append(t[-k:])
            c = float(np.corrcoef(np.concatenate(ps), np.concatenate(ts))[0, 1])
            if c > best_c:
                best, best_c = k, c
        lags[n] = {"best_lag_steps": best, "corr": best_c}
    out["est_handle_lag"] = lags
    if any("pred_hands" in s for s in series):
        names = [f"{side}_{a}" for side in ("L", "R") for a in ("fx", "fy", "fz", "mx", "my", "mz")]
        stats(cat("pred_hands"), cat("true_hands"), names, "est_hand")
        stats(cat("prior"), cat("true_hands"), names, "prior_hand")
    return out


def pulse_metrics(a: dict, onset: int, end: int, dt: float, tol=(0.08, 0.06, 0.10), hold_s=0.5,
                  axis: int = 0, window_s: float = 4.0) -> dict:
    """Disturbance response around a known cart-side wrench [onset, end) (control-step indices)."""
    T = len(a["yaw"])
    pre = slice(max(0, onset - int(1.0 / dt)), onset)
    post = slice(onset, min(T, end + int(window_s / dt)))
    vel, ref = a["vel"], a["cmd_ref"]
    err = vel - ref
    filt = err.copy()
    alpha = -math.expm1(-dt / 0.2)
    for i in range(1, T):
        filt[i] = filt[i - 1] + alpha * (err[i] - filt[i - 1])
    dev = vel[post] - vel[pre].mean(0)
    out = {f"peak_dev_{n}": float(np.max(np.abs(dev[:, j]))) for j, n in enumerate(("vx", "vy", "wz"))}
    within = (np.abs(filt) <= np.asarray(tol)).all(-1)

    def recovery(start):
        begin = None
        left = False
        for i in range(start, T):
            if not within[i]:
                begin, left = None, True
                continue
            begin = i if begin is None else begin
            if (i - begin) * dt + 1e-9 >= hold_s:
                return (begin - start) * dt, left
        return None, True

    rec, left = recovery(end)
    out.update(recovery_s=rec, recovered=rec is not None, left_tolerance=left)
    F = a["F"]
    out["handle_peak_dev"] = float(np.max(np.abs(F[post, axis] - F[pre, axis].mean())))
    norm = np.linalg.norm(a["hands_c_sub"][post][..., :3], axis=-1)
    out["hand_force_peak_200hz"] = float(norm.max())
    z = np.linalg.norm(a["zmp"][post] - a["zmp"][pre].mean(0), axis=-1)
    out["zmp_offset_peak_dev"] = float(z.max())
    out["robot_tilt_peak"] = float(a["tilt"][post].max())
    out["robot_gyro_peak"] = float(a["gyro"][post].max())
    pred = a["pred_resultant"]
    if np.isfinite(pred[post]).any():
        dp = pred[post, axis] - pred[pre, axis].mean()
        dt_ = F[post, axis] - F[pre, axis].mean()
        ip, it = int(np.argmax(np.abs(dp))), int(np.argmax(np.abs(dt_)))
        out["est_peak_ratio"] = float(dp[ip] / dt_[it]) if abs(dt_[it]) > 1e-6 else None
        out["est_sign_correct"] = bool(np.sign(dp[ip]) == np.sign(dt_[it]))
        thr = 0.5 * dt_[it]
        cross = lambda s: next((i for i in range(len(s)) if abs(s[i]) >= abs(thr) and np.sign(s[i]) == np.sign(thr)), None)  # noqa: E731
        ci, cp = cross(dt_), cross(dp)
        out["est_half_rise_lag_ms"] = None if ci is None or cp is None else (cp - ci) * dt * 1000
        out["est_trace_rmse"] = float(np.sqrt(np.mean((dp - dt_) ** 2)))
    return out
