"""V8 on the bench: the V8 deployable student (or a ROA teacher, privileged, simulation only) on the V7 MuJoCo plant,
with the fine-tuned HOMIE velocity estimator and the leg residual (homie_pushcart_v8.mujoco_runner). Grip loss is not a
stop; a fall is."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .policies import CartPolicyBase


V8_FINAL_STUDENT = "logs/rsl_rl/homie_pushcart_v8/2026-10-12_07-56-35_v8_fp_rl_armdr800/model_200.pt"


def resolve_v8_checkpoint(checkpoint="latest"):
    """A V8 checkpoint path; "latest" / "final" = the released V8 student (docs/push_cart_v8_teacher_student_plan.md)."""
    from homie_pushcart.paths import ROOT
    if str(checkpoint) not in ("latest", "final"):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    final = ROOT / V8_FINAL_STUDENT
    if final.is_file():
        return final
    log_dir = ROOT / "logs/rsl_rl/homie_pushcart_v8"
    runs = sorted((d for d in log_dir.glob("*") if d.is_dir() and any(d.glob("model_*.pt"))), key=lambda d: d.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No trained V8 checkpoint under {log_dir}")
    return max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


class V8Policy(CartPolicyBase):
    def __init__(self, checkpoint="latest", level=17, homie_checkpoint=None, *, name=None, zero_leg_residual=False,
                 camera_noise=1.0):
        from homie_pushcart_v8.mujoco_runner import MujocoV8Runner
        checkpoint = resolve_v8_checkpoint(checkpoint)
        self.runner = MujocoV8Runner(checkpoint, level, homie_checkpoint, zero_leg_residual=zero_leg_residual,
                                     camera_noise=camera_noise)
        self.env = self.runner.env
        self.homie_checkpoint = self.runner.homie_checkpoint
        self.dt, self.level, self.hl_checkpoint = self.env.dt, level, checkpoint
        tag = "V8 学生" if self.runner.student else "V8"
        self.name = name or (f"{tag} · {checkpoint.parent.name}/{checkpoint.stem}" + (" · legres0" if zero_leg_residual else ""))
        self.est_err = []
        m = self.env.model
        self.cart_body = self.runner.cart_body
        self.cart_qvel = int(m.joint("cart_free").dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._fallen, self._payload = False, 0.
        self.feet = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.loads, self.gait = [], []

    def reset(self, payload_kg=0.0, caster_angle=0.0, seed=0, **kw):
        self.runner.seed = int(seed)
        super().reset(payload_kg, caster_angle, seed, **kw)
        self.loads, self.gait = [], []
        self.est_err = []

    def step(self, cmd):
        super().step(cmd)
        r, g = self.runner, self.runner.grasp
        st = g.state()
        self.loads.append(np.concatenate([g.hand_loads().ravel(), st["rot_q"].ravel(), st["slide_q"] - g.anchor]))
        env, d = self.env, self.env.data
        foot_v = [np.linalg.norm(d.cvel[b][3:5]) for b in self.feet]       # (approx.) linear speed of the feet
        if r.student:                                   # |estimate - truth| of the 19 explicit quantities
            self.est_err.append(np.abs(r.explicit_est - r.privileged[:19]))
        self.gait.append(np.concatenate([r.leg_residual, r.homie_leg_actions, r.vel_est, r.vel_est_homie,
                                         env.base_lin_vel, env.projected_gravity, env.base_ang_vel, foot_v]))

    def failure(self):
        return "fall" if self._fallen else None

    def describe(self):
        from homie_pushcart_v7.physics import physics_manifest
        out = super().describe()
        out.update(kind="v8_student" if self.runner.student else "v8_teacher", version=8, physics=physics_manifest(),
                   policy_input=630, privileged_input=0 if self.runner.student else 62,
                   history_input=4032 if self.runner.student else 0,
                   homie_velocity_estimator=str(self.runner.hv_path), leg_residual=not self.runner.zero_leg_residual,
                   scene_selection="v7_square_tube_elastoplastic_grasp")
        return out

    def telemetry(self):
        from homie_pushcart_v8.task_config import leg_residual_bound
        out = super().telemetry()
        r = self.runner
        out.update({"leg_residual_abs": float(np.abs(r.leg_residual).mean()),
                    "leg_residual_at_bound": float(np.mean(np.abs(r.leg_residual) >= 0.95 * leg_residual_bound(r.task_cfg))),
                    "leg_raw_abs": float(np.abs(r.leg_raw).mean()),
                    "homie_vel_tuned": r.vel_est.tolist(), "homie_vel_original": r.vel_est_homie.tolist(),
                    "camera_yaw": float(r.frames[-1, 94])})
        return out
