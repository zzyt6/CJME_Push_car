"""V8.1 inference core: sensors -> joint targets, one 50 Hz tick. numpy + onnxruntime only (no LCM, no torch).

This is a line-by-line port of HomieRL/homie_pushcart_v8/mujoco_runner.py (MujocoV8Runner.step / _update_hl_obs) and
homie_pushcart/mujoco_env.py (compute_observations / step), the pipeline validated in the MuJoCo sim2sim bench. The
same class runs on the robot (run_v81.py) and inside MuJoCo on the PC (deploy/pc/verify_sim2sim.py), which checks it
step by step against the runner.

Timing of one tick k (the runner's step k-1 tail + step k head):
  1. HOMIE frame from the sensors S_k, with the command and the applied leg action of tick k-1  -> 6-frame history
  2. V8 frame (126) from that HOMIE frame + task reference, upper torque, camera yaw, last actions -> 5 / 32 frames
  3. task reference filter, idle gate
  4. V8.1 student (policy 630, history 4032) -> 18 high-level + 12 leg-residual actions
  5. HOMIE command = filtered (reference + gated residual), written into HOMIE's newest frame
  6. upper-body target = grasp hold pose + gated, slew-limited correction
  7. HOMIE actor (fine-tuned velocity estimate, own latent) -> legs; + leg residual -> leg targets
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

MODE_HOMIE = "homie"   # plain HOMIE: high level off, HOMIE's own velocity estimate, task command must be 0
MODE_PUSH = "push"     # V8.1: full stack


@dataclass
class Sensors:
    q: np.ndarray            # (27,) HOMIE joint order, rad
    dq: np.ndarray           # (27,) rad/s
    ang_vel: np.ndarray      # (3,) pelvis IMU gyro, pelvis frame, rad/s
    gravity: np.ndarray      # (3,) projected gravity in the pelvis frame (unit vector)
    # joint state one 5 ms sample earlier (the bridge's previous 200 Hz message), for the upper-body torque
    # observation: in training it is the PD torque of the last 5 ms physics substep, i.e. computed from the state
    # 5 ms before the observation (deploy/pc/verify_sim2sim.py: using q, dq instead costs ~4 mm at 0 kg)
    q_tau: np.ndarray | None = None
    dq_tau: np.ndarray | None = None


@dataclass
class Output:
    q_target: np.ndarray                  # (27,) joint position targets, HOMIE joint order
    cmd_homie: np.ndarray                 # (3,) command HOMIE follows (vx, vy, wz)
    cmd_ref: np.ndarray                   # (3,) filtered task reference
    hl_actions: np.ndarray                # (18,)
    leg_residual: np.ndarray              # (12,) applied (HOMIE action units)
    homie_legs: np.ndarray                # (12,) HOMIE's own output
    vel_used: np.ndarray                  # (3,) m/s, velocity estimate fed to HOMIE's actor
    est: np.ndarray = field(default_factory=lambda: np.zeros(19, np.float32))   # student velocity head (diagnostics)
    gate: float = 0.0


def load_config(models_dir) -> dict:
    cfg = json.loads((Path(models_dir) / "v81_config.json").read_text(encoding="utf-8"))
    if cfg.get("schema") != "pushcart_v81_deploy_v1":
        raise ValueError(f"unexpected config schema {cfg.get('schema')}")
    return cfg


class OnnxModels:
    def __init__(self, models_dir, threads: int = 2):
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = int(threads)
        opt.inter_op_num_threads = 1
        d = Path(models_dir)
        prov = ["CPUExecutionProvider"]
        self.student = ort.InferenceSession(str(d / "v81_student.onnx"), opt, providers=prov)
        self.homie = ort.InferenceSession(str(d / "homie_v81.onnx"), opt, providers=prov)

    def run_student(self, policy, history):
        a, est = self.student.run(None, {"policy": policy[None], "history": history[None]})
        return a[0], est[0]

    def run_homie(self, obs_history, use_finetuned: bool):
        legs, vel, vel_h = self.homie.run(None, {"obs_history": obs_history[None],
                                                 "use_finetuned": np.array([[1.0 if use_finetuned else 0.0]],
                                                                           np.float32)})
        return legs[0], vel[0], vel_h[0]


class V81Controller:
    def __init__(self, cfg: dict, models: OnnxModels):
        self.cfg, self.models = cfg, models
        a = lambda k: np.asarray(cfg[k], dtype=np.float64)                  # noqa: E731
        self.n_low = int(cfg["num_lower"])
        self.default = a("default_dof_pos")
        self.kp, self.kd, self.tau_lim = a("kp"), a("kd"), a("torque_limits")
        self.action_scale = float(cfg["action_scale"])
        self.clip_actions, self.clip_obs = float(cfg["clip_actions"]), float(cfg["clip_obs"])
        s = cfg["obs_scales"]
        self.s_ang, self.s_dof, self.s_vel, self.s_lin = s["ang_vel"], s["dof_pos"], s["dof_vel"], s["lin_vel"]
        self.cmd_scale = a("commands_scale")
        self.height = float(cfg["height_command"])
        self.hf, self.hh = int(cfg["homie_frame"]), int(cfg["homie_history"])
        self.fd, self.ph, self.lh = int(cfg["frame_dim"]), int(cfg["policy_history"]), int(cfg["long_history"])
        self.n_hl = int(cfg["hl_actions"])
        self.hl_clip = float(cfg["hl_clip"])
        self.ref_alpha, self.cmd_alpha, self.gate_speed = cfg["ref_alpha"], cfg["cmd_alpha"], cfg["gate_speed"]
        self.res_scale = a("cmd_residual_scale")
        self.cmd_lo, self.cmd_hi = a("homie_cmd_lo"), a("homie_cmd_hi")
        self.upper_scale, self.step_lim = a("upper_scale"), float(cfg["upper_step_limit"])
        self.hold = a("hold_target_q")
        self.upper_lo, self.upper_hi = a("upper_lo"), a("upper_hi")
        self.tau_scale, self.scale_obs = float(cfg["torque_obs_scale"]), float(cfg["scale_obs"])
        lr = cfg["leg_residual"]
        self.lr_enabled, self.lr_mode = bool(lr["enabled"]), lr["mode"]
        self.lr_range, self.lr_clip, self.lr_alpha = float(lr["range"]), float(lr["clip"]), float(lr["alpha"])
        self.reset_homie()
        self.reset_task()

    # ------------------------------------------------------------------ resets
    def reset_homie(self):
        """Fresh HOMIE state (= MujocoHomieEnv.reset): empty history, zero command and action."""
        self.homie_hist = np.zeros((self.hh, self.hf), np.float32)
        self.cmd_homie = np.zeros(3)
        self.applied = np.zeros(self.n_low)              # last applied leg action (HOMIE + residual)
        self.q_target_prev = None

    def reset_task(self):
        """Fresh V8 episode state (= MujocoV8Runner.reset_buffers). HOMIE's own state is kept."""
        self.cmd_ref = np.zeros(3)
        self.hl_actions = np.zeros(self.n_hl)
        self.leg_residual = np.zeros(self.n_low)
        self.upper_delta = np.zeros(len(self.hold))
        self.frames = np.zeros((self.ph, self.fd), np.float32)
        self.frames_long = np.zeros((self.lh, self.fd), np.float32)
        self.fresh = True                                 # next V8 frame starts a zero-padded history

    # ------------------------------------------------------------------ observation pieces
    def homie_frame(self, s: Sensors) -> np.ndarray:
        return np.concatenate([
            self.cmd_homie * self.cmd_scale, [self.height],
            np.asarray(s.ang_vel) * self.s_ang, np.asarray(s.gravity),
            (np.asarray(s.q) - self.default) * self.s_dof, np.asarray(s.dq) * self.s_vel,
            self.applied]).astype(np.float32)

    def upper_torque(self, s: Sensors) -> np.ndarray:
        """PD torque of waist + arms (what MuJoCo's explicit PD / the motor driver applies), as observed in training."""
        if self.q_target_prev is None:
            return np.zeros(len(self.hold))
        q = s.q if s.q_tau is None else s.q_tau
        dq = s.dq if s.dq_tau is None else s.dq_tau
        tau = self.kp * (self.q_target_prev - np.asarray(q)) - self.kd * np.asarray(dq)
        return np.clip(tau, -self.tau_lim, self.tau_lim)[self.n_low:]

    def leg_residual_target(self, u):
        if self.lr_mode == "tanh":
            return self.lr_range * np.tanh(u)
        return np.clip(u, -self.lr_clip, self.lr_clip)

    # ------------------------------------------------------------------ one tick
    def step(self, s: Sensors, task_cmd, cam_yaw: float, mode: str, upper_tau=None) -> Output:
        push = mode == MODE_PUSH
        task_cmd = np.asarray(task_cmd, dtype=np.float64) if push else np.zeros(3)
        # 1. HOMIE frame (state S_k, command / applied action of the last tick)
        hf = self.homie_frame(s)
        self.homie_hist = np.roll(self.homie_hist, -1, axis=0)
        self.homie_hist[-1] = hf
        # 2. V8 frame
        if push:
            tau = self.upper_torque(s) if upper_tau is None else np.asarray(upper_tau)
            frame = np.concatenate([hf, self.cmd_ref * self.cmd_scale, tau * self.tau_scale, [cam_yaw],
                                    self.hl_actions, self.leg_residual, [self.scale_obs]]).astype(np.float32)
            if self.fresh:
                self.frames[:] = 0.
                self.frames_long[:] = 0.
                self.fresh = False
            else:
                self.frames = np.roll(self.frames, -1, axis=0)
                self.frames_long = np.roll(self.frames_long, -1, axis=0)
            self.frames[-1] = frame
            self.frames_long[-1] = frame
        # 3. task reference, idle gate
        self.cmd_ref = self.cmd_ref + self.ref_alpha * (task_cmd - self.cmd_ref)
        gate = min(1.0, float(np.linalg.norm(self.cmd_ref)) / self.gate_speed) if self.gate_speed > 0 else 1.0
        # 4. high level
        est = np.zeros(19, np.float32)
        if push:
            c = self.clip_obs
            out, est = self.models.run_student(np.clip(self.frames.reshape(-1), -c, c),
                                               np.clip(self.frames_long.reshape(-1), -c, c))
            out = out.astype(np.float64)
        else:
            out = np.zeros(self.n_hl + self.n_low)
        a = np.clip(out[:self.n_hl], -self.hl_clip, self.hl_clip)
        target = self.leg_residual_target(out[self.n_hl:])
        self.leg_residual = (self.leg_residual + self.lr_alpha * (target - self.leg_residual)
                             if self.lr_enabled else np.zeros(self.n_low))
        self.hl_actions = a
        # 5. HOMIE command
        tgt = np.clip(self.cmd_ref + gate * a[:3] * self.res_scale, self.cmd_lo, self.cmd_hi)
        self.cmd_homie = self.cmd_homie + self.cmd_alpha * (tgt - self.cmd_homie)
        self.homie_hist[-1, :3] = self.cmd_homie * self.cmd_scale
        # 6. upper body
        delta_target = gate * a[3:] * self.upper_scale
        delta = self.upper_delta + np.clip(delta_target - self.upper_delta, -self.step_lim, self.step_lim)
        q_upper = np.clip(self.hold + delta, self.upper_lo, self.upper_hi)
        self.upper_delta = q_upper - self.hold
        # 7. HOMIE legs (+ residual)
        c = self.clip_obs
        legs, vel, _ = self.models.run_homie(np.clip(self.homie_hist.reshape(-1), -c, c), use_finetuned=push)
        self.applied = np.clip(legs.astype(np.float64) + self.leg_residual, -self.clip_actions, self.clip_actions)
        q_legs = self.default[:self.n_low] + self.action_scale * self.applied
        q_target = np.concatenate([q_legs, q_upper])
        self.q_target_prev = q_target
        return Output(q_target=q_target, cmd_homie=self.cmd_homie.copy(), cmd_ref=self.cmd_ref.copy(),
                      hl_actions=a.copy(), leg_residual=self.leg_residual.copy(), homie_legs=legs.copy(),
                      vel_used=vel / self.s_lin, est=est, gate=gate)
