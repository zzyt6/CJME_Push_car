"""Official V6 bench: one frozen V3.2/legacy four-sphere backend."""
from pathlib import Path
import mujoco

from .policies import CartPolicyBase


def resolve_v6_checkpoint(checkpoint="latest"):
    from homie_pushcart_v6.paths import LOG_DIR
    if str(checkpoint) != "latest":
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    runs = sorted((d for d in LOG_DIR.glob("*") if d.is_dir() and any(d.glob("model_*.pt"))), key=lambda d:d.name)
    if not runs:
        raise FileNotFoundError(f"No trained V6 checkpoint under {LOG_DIR}")
    return max(runs[-1].glob("model_*.pt"), key=lambda p:int(p.stem.rsplit("_",1)[1]))


class V6Policy(CartPolicyBase):
    def __init__(self, checkpoint="latest", level=17, homie_checkpoint=None, *, name=None):
        from homie_pushcart_v6.mujoco_runner import MujocoV6Runner
        checkpoint = resolve_v6_checkpoint(checkpoint)
        self.runner = MujocoV6Runner(checkpoint, level, homie_checkpoint)
        self.env = self.runner.env
        self.homie_checkpoint = self.runner.homie_checkpoint
        self.dt, self.level, self.hl_checkpoint = self.env.dt, level, checkpoint
        self.name = name or f"V6 · {checkpoint.parent.name}/{checkpoint.stem}"
        m = self.env.model
        self.cart_body = self.runner.cart_body
        self.cart_qvel = int(m.joint("cart_free").dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._fallen, self._payload = False, 0.

    def describe(self):
        from homie_pushcart_v6.physics import physics_manifest
        out = super().describe()
        out.update(kind="v6", version=6, frame_dim=121, actor_history=5, estimator_history=6,
                   physics=physics_manifest(), policy_input=605, actor_mlp_input=652, critic_input=665,
                   cart_prediction_horizon_s=self.dt,
                   speed_reference="V3.2 cart COM", scene_selection="fixed_v32_legacy_four_spheres")
        return out

    def telemetry(self):
        out = super().telemetry()
        out["cart_est_horizon_s"] = self.dt
        out.update(dict(zip(("cart_est_vx","cart_est_vy","cart_est_wz"),map(float,self.runner.pred_cart_velocity))))
        for side, hand in zip(("left","right"), self.runner.pred_hands.reshape(2,6)):
            out.update({f"est_{side}_{axis}":float(value) for axis,value in zip(("fx","fy","fz","mx","my","mz"),hand)})
        return out
