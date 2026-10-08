"""V7 on the bench: the V7 MuJoCo plant (square handle tube, elasto-plastic slide, elastic-yield pivots, Coulomb cart)
with the V7 high-level policy. Grip loss is only recorded (events in the telemetry): for watching a run end to end,
the run is never ended by it -- only a fall ends it."""
from pathlib import Path

import mujoco
import numpy as np

from .policies import CartPolicyBase


def resolve_v7_checkpoint(checkpoint="latest"):
    from homie_pushcart.paths import ROOT
    if str(checkpoint) != "latest":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    log_dir = ROOT / "logs/rsl_rl/homie_pushcart_v7"
    runs = sorted((d for d in log_dir.glob("*") if d.is_dir() and any(d.glob("model_*.pt"))), key=lambda d: d.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No trained V7 checkpoint under {log_dir}")
    return max(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


class V7Policy(CartPolicyBase):
    def __init__(self, checkpoint="latest", level=17, homie_checkpoint=None, *, name=None):
        from homie_pushcart_v7.assets.build import load_v7_params, nominal_grasp
        from homie_pushcart_v7.mujoco_runner import MujocoV7Runner
        checkpoint = resolve_v7_checkpoint(checkpoint)
        self.runner = MujocoV7Runner(checkpoint, level, homie_checkpoint)
        self.env = self.runner.env
        self.homie_checkpoint = self.runner.homie_checkpoint
        self.dt, self.level, self.hl_checkpoint = self.env.dt, level, checkpoint
        self.name = name or f"V7 · {checkpoint.parent.name}/{checkpoint.stem}"
        m = self.env.model
        self.cart_body = self.runner.cart_body
        self.cart_qvel = int(m.joint("cart_free").dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._fallen, self._payload = False, 0.
        # grip-loss bookkeeping with the nominal (mid-range) grasp the MuJoCo plant is configured with
        params = load_v7_params()
        g, nom = params["grasp"], nominal_grasp(params)
        mean = lambda r: 0.5 * (float(r[0]) + float(r[1]))                                    # noqa: E731
        self.cap_pull, self.cap_lift, self.cap_push = mean(g["pull_capacity_range"]), mean(g["lift_capacity_range"]), float(g["push_capacity_n"])
        self.cap_roll, self.cap_moment = float(nom["roll_capacity"]), float(nom["moment_capacity"])
        self.roll_break, self.tilt_break = float(g["roll_break_rad"]), float(g["tilt_break_rad"])
        self.slide_break, self.persist = float(g["slide_drift_break_m"]), int(g["overload_persist_steps"])
        self._reset_grip_log()

    def _reset_grip_log(self):
        self.events, self._over, self._active = [], 0, {"overload": False, "pivot": False, "slide": False}
        self._t, self._anchor0, self.ratios = 0.0, None, np.zeros((2, 5))
        self.pivot_frac, self.drift_m = 0.0, 0.0

    def reset(self, *a, **k):
        super().reset(*a, **k)
        self._reset_grip_log()
        self._anchor0 = self.runner.grasp.anchor.copy()

    def step(self, cmd) -> None:
        super().step(cmd)
        self._t += self.dt
        gr = self.runner.grasp
        f = gr.hand_loads()                                    # (2, 6) hand -> cart at the grasp point, cart axes
        st = gr.state()
        self.ratios = np.stack((np.maximum(-f[:, 0], 0) / self.cap_pull, np.maximum(f[:, 2], 0) / self.cap_lift,
                                np.abs(f[:, 4]) / self.cap_roll, np.abs(f[:, 3]) / self.cap_moment, np.abs(f[:, 5]) / self.cap_moment), -1)
        over = bool((self.ratios[:, :2] > 1).any() or (np.maximum(f[:, 0], 0) > self.cap_push).any())
        self._over = self._over + 1 if over else 0
        pivot = max(float(np.abs(st["rot_q"][0]).max()) / self.roll_break, float(np.abs(st["rot_q"][1:]).max()) / self.tilt_break)
        drift = float(np.abs(gr.anchor - self._anchor0).max()) if self._anchor0 is not None else 0.0
        self.pivot_frac, self.drift_m = pivot, drift
        for key, now, what in (("overload", self._over >= self.persist, "回拉/上提超限 0.4 s"), ("pivot", pivot > 1.0, "撬过断开角"),
                               ("slide", drift > self.slide_break, "滑走 8 cm")):
            if now and not self._active[key]:
                self.events.append(f"{self._t:.1f} s {what}")
            self._active[key] = now
        if self._over >= self.persist:
            self._over = 0

    def failure(self):
        return "fall" if self._fallen else None                # grip loss is recorded, not a stop

    def describe(self):
        from homie_pushcart_v7.physics import physics_manifest
        out = super().describe()
        out.update(kind="v7", version=7, physics=physics_manifest(), policy_input=605, critic_input=695,
                   scene_selection="v7_square_tube_elastoplastic_grasp")
        return out

    def telemetry(self):
        out = super().telemetry()
        out["cart_est_horizon_s"] = self.dt
        out.update(dict(zip(("cart_est_vx", "cart_est_vy", "cart_est_wz"), map(float, self.runner.pred_cart_velocity))))
        r = self.ratios
        ev = f"{len(self.events)} 次" + ("：" + "；".join(self.events[-3:]) if self.events else "")
        out["status_line"] = (f"- V7 握持（MuJoCo 标称参数）载荷/能力：回拉 {r[0, 0]:.2f}/{r[1, 0]:.2f} · 绕管 {r[:, 2].max():.2f} · "
                              f"前后倾 {r[:, 3].max():.2f} · 偏转 {r[:, 4].max():.2f} · 转角/断开角 {self.pivot_frac:.2f} · "
                              f"沿杆滑移 {1e3 * self.drift_m:.1f} mm\n- 脱手事件（不中止）{ev}")
        return out
