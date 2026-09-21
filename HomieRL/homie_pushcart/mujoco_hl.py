"""MuJoCo sim2sim of the stage-one stack: high-level policy -> frozen HOMIE -> PD, mirroring ``PushCartEnv``.

``MujocoHighLevelRunner`` reproduces the Isaac-side action pipeline (first-order command reference, idle gate,
command residual + low-pass + clipping, slew-limited upper-body corrections inside the hard limits) and the
124-dim actor frame (HOMIE's own frame, task command reference, cart mass, upper torques, HIM velocity
estimate, command-estimate difference, handle pose in the base frame, last action, correction scale) so a
checkpoint trained in Isaac Lab can be evaluated or teleoperated in MuJoCo without any Isaac dependency.
The task configuration comes from the checkpoint's own run (``task_cfg_from_run``), so checkpoints trained
under different residual scales / clips / references keep their behaviour. Observation noise / delay are off.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .assets.mjcf_scene import CART_BODY
from .curriculum import level_params
from .cart_quality import CartQualityTracker
from .mujoco_env import MujocoHomieEnv, quat_rotate_inverse_wxyz
from .paths import HOMIE_RUN_DIR, ROOT
from .task_config import HL_FRAME, PushCartTaskCfg, num_actor_obs, num_critic_obs, task_cfg_from_run


def _yaw(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_to_mat(quat_wxyz) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, dtype=float))
    return m.reshape(3, 3)


class MujocoHighLevelRunner:
    def __init__(self, env: MujocoHomieEnv, lowlevel, hl_policy=None, task_cfg: PushCartTaskCfg | None = None, level: int = 6):
        if not env.with_cart:
            raise ValueError("the high-level policy needs the cart scene (mode=cart)")
        self.env, self.lowlevel, self.hl = env, lowlevel, hl_policy
        self.task_cfg = task_cfg or PushCartTaskCfg()
        tc = self.task_cfg
        self.dt = env.dt
        self.num_actions = tc.env.num_hl_actions
        self.history = tc.env.actor_history
        self.residual_scale = np.array(tc.actions.cmd_residual_scale)
        lim = np.array(tc.actions.homie_cmd_limits)
        self.cmd_lo, self.cmd_hi = lim[:, 0], lim[:, 1]
        self.cmd_alpha = min(1.0, self.dt / tc.actions.cmd_filter_tau_s) if tc.actions.cmd_filter_tau_s > 0 else 1.0
        ref_tau = float(getattr(tc.commands, "ref_tau_s", 0.0))            # 0 (v2 runs): the reference is the raw command
        self.ref_alpha = min(1.0, self.dt / ref_tau) if ref_tau > 0 else 1.0
        self.gate_speed = float(getattr(tc.actions, "idle_gate_speed", 0.0))   # 0 (v2 runs): no idle gate
        self.set_level(level)
        self.step_lim = tc.actions.upper_rate_limit_rad_s * self.dt
        n_low = env.num_lower
        self.hold_target_q = env.preset_dof_pos[n_low:] + env.upper_hold_offset
        self.upper_lo = env.default_dof_pos[n_low:] + env.action_min[n_low:] * env.action_scale
        self.upper_hi = env.default_dof_pos[n_low:] + env.action_max[n_low:] * env.action_scale
        self.commands_scale = env.commands_scale
        m = env.model
        self.cart_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CART_BODY)
        self.pelvis = env.pelvis_id
        derived = env.derived
        yaw0 = float(derived["base_yaw"])
        d = np.array(derived["bar_center_world"]) - np.array(derived["base_pos_world"])
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        self.bar_local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        self.swivel_qpos = [m.joint(f"caster_{i}_swivel_joint").qposadr[0] for i in range(4)]
        self.cart_free = m.joint("cart_free")
        self.upper_act = env.act_idx[n_low:]
        self.quality = CartQualityTracker(tc.quality, self.dt, 1, dtype=torch.float64)
        self.wheel_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{i}") for i in range(4)]
        self.reset_buffers()

    # ------------------------------------------------------------------ state
    def set_level(self, level: int):
        """Curriculum level that defines the upper-body correction scale (and the ``scale`` observation)."""
        self.level = int(level)
        self.lp = level_params(self.level, self.task_cfg)
        self.upper_scale = np.array([self.lp.waist_scale] + [self.lp.arm_scale] * 14)

    @property
    def cart_mass_total(self) -> float:
        return float(self.env.model.body_mass[self.cart_body])

    def set_payload(self, payload_kg: float):
        m = self.env.model
        base = float(self.env.cart_params["mass"]["base_kg"])
        ratio = (base + payload_kg) / base
        m.body_mass[self.cart_body] = base + payload_kg
        m.body_inertia[self.cart_body] = self._default_inertia * ratio
        mujoco.mj_setConst(m, self.env.data)

    def reset_buffers(self):
        self.quality.reset()
        self.cmd_task = np.zeros(3)
        self.cmd_ref = np.zeros(3)
        self.cmd_homie = np.zeros(3)
        self.hl_actions = np.zeros(self.num_actions)
        self.upper_delta_q = np.zeros(15)
        self.vel_est = np.zeros(3)
        self.actor_hist = np.zeros((self.history, HL_FRAME), dtype=np.float32)
        self.env.upper_delta[:] = 0.
        if not hasattr(self, "_default_inertia"):
            self._default_inertia = self.env.model.body_inertia[self.cart_body].copy()

    def reset(self, caster_angle: float | None = None) -> np.ndarray:
        obs = self.env.reset()
        if caster_angle is not None:
            for adr in self.swivel_qpos:
                self.env.data.qpos[adr] = caster_angle
            mujoco.mj_forward(self.env.model, self.env.data)
        self.reset_buffers()
        self._obs = obs
        self.task_state()
        self._update_hl_obs(fresh=True)
        return self.hl_obs

    def task_state(self) -> dict:
        env, d = self.env, self.env.data
        cart_pos = d.xpos[self.cart_body].copy()
        cart_quat = d.xquat[self.cart_body].copy()
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_BODY, self.cart_body, vel, 0)
        lin, ang = vel[3:6], vel[:3]
        yaw = _yaw(cart_quat)
        c, s = math.cos(yaw), math.sin(yaw)
        cart_v = np.array([c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1]])
        bar_w = cart_pos + _quat_to_mat(cart_quat) @ self.bar_local
        pelvis_pos = d.xpos[self.pelvis].copy()
        base_quat = env.base_quat_wxyz.copy()
        rel = bar_w - pelvis_pos
        bar_b = quat_rotate_inverse_wxyz(base_quat, rel)
        rel_yaw = math.atan2(math.sin(yaw - _yaw(base_quat)), math.cos(yaw - _yaw(base_quat)))
        tilt = float(np.linalg.norm(quat_rotate_inverse_wxyz(cart_quat, np.array([0., 0., -1.]))[:2]))
        self.ts = {
            "cart_v": cart_v, "cart_wz": float(ang[2]), "cart_speed": float(np.hypot(lin[0], lin[1])),
            "bar_b": bar_b, "rel_yaw": rel_yaw, "dist": float(np.hypot(rel[0], rel[1])), "cart_tilt": tilt,
            "upper_tau": d.ctrl[self.upper_act].copy(), "grasp": env.grasp_forces(),
        }
        hands = env.grasp_cart_wrenches()
        total = sum(hands.values(), np.zeros(6))
        self.ts.update({"cart_yaw": yaw, "cart_com_w": d.subtree_com[self.cart_body].copy(),
                        "cart_force_w": total[:3], "cart_moment_w": total[3:], "hand_cart_wrenches": hands,
                        "cart_task_wrench": np.array([c * total[0] + s * total[1], -s * total[0] + c * total[1], total[5]]),
                        "robot_tilt_rad": float(np.arcsin(np.clip(np.linalg.norm(env.projected_gravity[:2]), 0, 1))),
                        "wheel_lift_m": float(np.maximum(d.xpos[self.wheel_body_ids, 2] - env.cart_params["wheels"]["radius"], 0).max())})
        return self.ts

    # ------------------------------------------------------------------ stepping
    def step(self, cmd_task) -> tuple[np.ndarray, bool]:
        tc, env = self.task_cfg, self.env
        self.cmd_task = np.asarray(cmd_task, dtype=float)
        self.cmd_ref = self.cmd_ref + self.ref_alpha * (self.cmd_task - self.cmd_ref)
        gate = min(1.0, float(np.linalg.norm(self.cmd_ref)) / self.gate_speed) if self.gate_speed > 0 else 1.0
        if self.hl is not None:
            a = self.hl(torch.from_numpy(self.hl_obs).unsqueeze(0)).squeeze(0).cpu().numpy()
            a = np.clip(a, -tc.actions.clip, tc.actions.clip)
        else:
            a = np.zeros(self.num_actions)
        self.hl_actions = a
        target = np.clip(self.cmd_ref + gate * a[:3] * self.residual_scale, self.cmd_lo, self.cmd_hi)
        self.cmd_homie = self.cmd_homie + self.cmd_alpha * (target - self.cmd_homie)
        env.set_command(*self.cmd_homie)
        # HOMIE observes the command it is about to follow (same in-place patch as the Isaac env)
        env.obs_hist[-1, :3] = env.commands[:3] * self.commands_scale
        delta_target = gate * a[3:] * self.upper_scale
        delta = self.upper_delta_q + np.clip(delta_target - self.upper_delta_q, -self.step_lim, self.step_lim)
        q_target = np.clip(self.hold_target_q + delta, self.upper_lo, self.upper_hi)
        self.upper_delta_q = q_target - self.hold_target_q
        env.upper_delta[:] = self.upper_delta_q / env.action_scale
        obs_t = torch.from_numpy(np.clip(env.obs_hist.reshape(-1), -env.clip_obs, env.clip_obs)).unsqueeze(0)
        legs, vel_est = self.lowlevel.act_and_estimate(obs_t)
        self.vel_est = vel_est.squeeze(0).cpu().numpy() / env.obs_scales["lin_vel"]
        self._obs, fallen = env.step(legs.squeeze(0).cpu().numpy())
        self.task_state()
        self.quality.update(torch.tensor([[*self.ts["cart_v"], self.ts["cart_wz"]]], dtype=torch.float64),
                            torch.from_numpy(self.cmd_ref[None]), torch.tensor([self.ts["cart_yaw"]], dtype=torch.float64),
                            torch.from_numpy(self.ts["cart_force_w"][None]), torch.from_numpy(self.ts["cart_moment_w"][None]))
        self._update_hl_obs(fresh=False)
        return self.hl_obs, fallen

    def _update_hl_obs(self, fresh: bool):
        env, tc, ts = self.env, self.task_cfg, self.ts
        homie_frame = env.obs_hist[-1]
        task_cmd = self.cmd_ref * self.commands_scale
        mass = np.array([(self.cart_mass_total - tc.observations.cart_mass_norm[0]) / tc.observations.cart_mass_norm[1]])
        tau = ts["upper_tau"] * tc.observations.torque_scale
        est_scaled = self.vel_est * env.obs_scales["lin_vel"]
        diff = np.array([self.cmd_homie[0] - self.vel_est[0], self.cmd_homie[1] - self.vel_est[1],
                         self.cmd_homie[2] - env.base_ang_vel[2]]) * self.commands_scale
        pose = np.concatenate([ts["bar_b"], [ts["rel_yaw"]]])
        scale = np.array([self.lp.arm_scale / tc.actions.arm_scale_range[1]])
        frame = np.concatenate([homie_frame, task_cmd, mass, tau, est_scaled, diff, pose, self.hl_actions, scale]).astype(np.float32)
        assert frame.shape[0] == HL_FRAME, frame.shape
        if fresh:
            self.actor_hist[:] = 0.
        else:
            self.actor_hist = np.roll(self.actor_hist, -1, axis=0)
        self.actor_hist[-1] = frame
        self.hl_obs = np.clip(self.actor_hist.reshape(-1), -env.clip_obs, env.clip_obs)

    def geometry_failure(self) -> str | None:
        tc, ts = self.task_cfg.termination, self.ts
        if abs(ts["rel_yaw"]) > tc.yaw_diff_rad:
            return "cart_yaw"
        if not (tc.distance_m[0] <= ts["dist"] <= tc.distance_m[1]):
            return "cart_distance"
        if ts["cart_tilt"] > math.sin(tc.cart_tilt_rad):
            return "cart_tilt"
        return None


def make_runner(args):
    from .hl_policy import HighLevelPolicy
    from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
    env = MujocoHomieEnv(with_cart=True)
    homie_ckpt = Path(args.homie_checkpoint) if getattr(args, "homie_checkpoint", None) else frozen_checkpoint()
    lowlevel = HimLowLevelPolicy(homie_ckpt, device="cpu")
    checkpoint = getattr(args, "checkpoint", None)
    task_cfg = task_cfg_from_run(checkpoint) if checkpoint else PushCartTaskCfg()
    hl = None
    if getattr(args, "checkpoint", None):
        hl = HighLevelPolicy(Path(args.checkpoint), num_actor_obs(task_cfg), num_critic_obs(task_cfg), task_cfg.env.num_hl_actions, "cpu")
    return MujocoHighLevelRunner(env, lowlevel, hl, task_cfg, level=args.level), homie_ckpt


def run_mujoco_eval(args) -> dict:
    """Same scenario suite as the Isaac `eval` command, in MuJoCo (single env, sequential)."""
    from .evaluate import SCENARIOS
    runner, homie_ckpt = make_runner(args)
    names = args.scenarios or list(SCENARIOS)
    results, start = {}, time.perf_counter()
    for name in names:
        cmd, payload, caster, seconds, switch = SCENARIOS[name]
        runner.set_payload(payload)
        runner.reset(caster_angle=caster)
        steps, skip = int(seconds / runner.dt), int(args.transient_s / runner.dt)
        cmd_now = np.array(cmd)
        vx_err, wz_err, dist, grasp, fails = [], [], [], [], {}
        for step in range(steps):
            if switch is not None and step == int(switch[0] / runner.dt):
                cmd_now = np.array(switch[1])
            _, fallen = runner.step(cmd_now)
            failure = "fall" if fallen else runner.geometry_failure()
            if failure:
                fails[failure] = fails.get(failure, 0) + 1
                runner.reset(caster_angle=caster)
                continue
            if step < skip or (switch is not None and abs(step - int(switch[0] / runner.dt)) < skip):
                continue
            ts = runner.ts
            vx_err.append(cmd_now[0] - ts["cart_v"][0])
            wz_err.append(cmd_now[2] - ts["cart_wz"])
            dist.append(ts["dist"])
            grasp.append(max(ts["grasp"].values(), default=0.0))
        vx_err, wz_err = np.array(vx_err), np.array(wz_err)
        res = {
            "command": list(cmd), "payload_kg": payload, "seconds": seconds,
            "cart_vx_rmse": float(np.sqrt(np.mean(vx_err ** 2))) if len(vx_err) else None,
            "cart_vx_bias": float(np.mean(vx_err)) if len(vx_err) else None,
            "cart_wz_rmse": float(np.sqrt(np.mean(wz_err ** 2))) if len(wz_err) else None,
            "robot_bar_distance_mean": float(np.mean(dist)) if dist else None,
            "grasp_force_p95_N": float(np.percentile(grasp, 95)) if grasp else None,
            "failures": fails,
        }
        results[name] = res
        print(f"[mj-eval] {name:15s} vx rmse {res['cart_vx_rmse']} (bias {res['cart_vx_bias']}) wz rmse {res['cart_wz_rmse']} "
              f"| dist {res['robot_bar_distance_mean']} | grasp p95 {res['grasp_force_p95_N']} N | failures {fails}", flush=True)
    tag = Path(args.checkpoint).stem if getattr(args, "checkpoint", None) else "passthrough"
    out = {"simulator": "mujoco", "checkpoint": str(args.checkpoint) if getattr(args, "checkpoint", None) else None,
           "homie_checkpoint": str(homie_ckpt), "level": args.level, "scenarios": results, "elapsed_s": time.perf_counter() - start}
    path = ROOT / f"logs/pushcart_mj_eval_{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[mj-eval] wrote {path}", flush=True)
    return out
