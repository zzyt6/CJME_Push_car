"""Policies under test: a uniform interface over "something that takes a velocity command and moves the cart".

    passthrough            task command straight into the frozen HOMIE, upper body held at the grasp pose
    two_stage:<ckpt|latest> trained high-level checkpoint (deterministic mean) on top of the frozen HOMIE
    robot_only             HOMIE without the cart, tracked body = pelvis (upper bound, quantifies the cart's cost)
    plugin:<module>:<attr> any class implementing ``PolicyUnderTest`` (future RL stages)

All MuJoCo, CPU. The command is expressed in the tracked body's frame: cart base for cart policies, pelvis for
robot_only. Cart policies also expose the physical knobs used by the robustness sweep (payload, COM offset,
wheel friction, velocity kicks, wheel stalls); robot_only ignores them.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np
import torch


class PolicyUnderTest(Protocol):
    name: str
    dt: float
    tracks: str                                     # "cart" | "robot"

    def reset(self, payload_kg: float = 0.0, caster_angle: float = 0.0, seed: int = 0, *,
              wheel_friction: float | None = None, com_offset_m: float = 0.0) -> None: ...
    def step(self, cmd) -> None: ...                # cmd (3,) in the tracked body frame
    def pose(self) -> tuple[float, float, float]: ...
    def velocity(self) -> np.ndarray: ...           # (vx, vy, wz) in the tracked body frame
    def failure(self) -> str | None: ...
    def telemetry(self) -> dict: ...


def _yaw(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------------------- cart policies

class CartPolicyBase:
    """Frozen HOMIE + optional high-level policy through ``MujocoHighLevelRunner``; tracked body = cart base."""

    tracks = "cart"

    def __init__(self, hl_checkpoint: Path | None, level: int = 17, homie_checkpoint: Path | None = None, name: str | None = None):
        from types import SimpleNamespace
        from homie_pushcart.mujoco_hl import make_runner
        args = SimpleNamespace(checkpoint=hl_checkpoint, homie_checkpoint=homie_checkpoint, level=level)
        self.runner, self.homie_checkpoint = make_runner(args)
        self.env = self.runner.env
        self.dt = self.env.dt
        self.level = level
        self.hl_checkpoint = Path(hl_checkpoint) if hl_checkpoint else None
        self.name = name or ("two_stage" if hl_checkpoint else "passthrough")
        m = self.env.model
        self.cart_body = self.runner.cart_body
        free = m.joint("cart_free")
        self.cart_qvel = int(free.dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._fallen = False
        self._payload = 0.0

    # -- interface
    def describe(self) -> dict:
        return {"name": self.name, "kind": "two_stage" if self.hl_checkpoint else "passthrough", "tracks": self.tracks,
                "hl_checkpoint": str(self.hl_checkpoint) if self.hl_checkpoint else None,
                "hl_iteration": getattr(self.runner.hl, "iteration", None) if self.runner.hl is not None else None,
                "homie_checkpoint": str(self.homie_checkpoint), "level": self.level}

    def reset(self, payload_kg: float = 0.0, caster_angle: float = 0.0, seed: int = 0, *,
              wheel_friction: float | None = None, com_offset_m: float = 0.0) -> None:
        """Model changes (payload, friction, COM) go first: ``mj_setConst`` leaves the data at qpos0, the reset then
        restores the preset configuration."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.restore_physics()
        if wheel_friction is not None:
            self.set_wheel_friction(wheel_friction)
        if abs(com_offset_m) > 0:
            self.set_payload_com_offset(com_offset_m, 0.0)
        self._payload = float(payload_kg)
        self.runner.set_payload(float(payload_kg))
        mujoco.mj_resetData(self.env.model, self.env.data)      # no solver warm-start carried over between runs
        self.env.set_command(0.0, 0.0, 0.0)                     # env.reset keeps the last command otherwise (stale HOMIE frame)
        self.runner.reset(caster_angle=float(caster_angle))
        self._fallen = False

    def step(self, cmd) -> None:
        _, fallen = self.runner.step(np.asarray(cmd, dtype=float))
        self._fallen = bool(fallen)

    def pose(self) -> tuple[float, float, float]:
        d = self.env.data
        p = d.xpos[self.cart_body]
        return float(p[0]), float(p[1]), _yaw(d.xquat[self.cart_body])

    def velocity(self) -> np.ndarray:
        ts = self.runner.ts
        return np.array([ts["cart_v"][0], ts["cart_v"][1], ts["cart_wz"]], dtype=float)

    def failure(self) -> str | None:
        return "fall" if self._fallen else self.runner.geometry_failure()

    def telemetry(self) -> dict:
        ts = self.runner.ts
        yaw = self.pose()[2]
        c, s = math.cos(yaw), math.sin(yaw)
        a = self.runner.hl_actions
        out = {"homie_cmd": self.runner.cmd_homie.tolist(), "upper_dq_max": float(np.abs(self.runner.upper_delta_q).max()),
               "upper_dq": self.runner.upper_delta_q.tolist(), "payload_kg": self._payload,
               # is the residual network actually modulating, or pinned at the clip? (fraction of saturated outputs)
               "cmd_clip_frac": float(np.mean(np.abs(a[:3]) >= 0.99)) if self.runner.hl is not None else 0.0,
               "arm_clip_frac": float(np.mean(np.abs(a[3:]) >= 0.99)) if self.runner.hl is not None else 0.0}
        for side, f in self.env.grasp_force_vectors().items():
            out[f"grasp_{side}"] = float(np.linalg.norm(f))
            out[f"push_{side}"] = float(c * f[0] + s * f[1])          # cart-frame x (hand -> cart)
            out[f"lat_{side}"] = float(-s * f[0] + c * f[1])          # cart-frame y
            out[f"up_{side}"] = float(f[2])
        return out

    # -- physical knobs (robustness sweep)
    def restore_physics(self):
        m = self.env.model
        m.geom_friction[self.wheel_geoms] = self._wheel_friction0
        m.dof_damping[self.wheel_dofs] = self._wheel_damping0
        m.body_ipos[self.cart_body] = self._ipos0

    def set_wheel_friction(self, mu: float):
        self.env.model.geom_friction[self.wheel_geoms, 0] = float(mu)

    def set_payload_com_offset(self, dx: float, dy: float = 0.0):
        """Shift the cart's centre of mass in its own frame (payload placed off-centre). Call before ``reset``."""
        m = self.env.model
        m.body_ipos[self.cart_body] = self._ipos0 + np.array([dx, dy, 0.0])
        mujoco.mj_setConst(m, mujoco.MjData(m))                 # scratch data: keeps the live state untouched

    def kick(self, dv_xy):
        """Instantaneous planar velocity impulse on the cart (world frame), like the training disturbance."""
        self.env.data.qvel[self.cart_qvel:self.cart_qvel + 2] += np.asarray(dv_xy, dtype=float)

    def set_wheel_damping_factor(self, wheel: int, factor: float):
        self.env.model.dof_damping[self.wheel_dofs[wheel]] = self._wheel_damping0[wheel] * float(factor)

    def cart_mass_total(self) -> float:
        return float(self.runner.cart_mass_total)


class PassthroughPolicy(CartPolicyBase):
    def __init__(self, level: int = 17, homie_checkpoint: Path | None = None):
        super().__init__(None, level, homie_checkpoint, name="passthrough")


class TwoStagePolicy(CartPolicyBase):
    def __init__(self, checkpoint: str | Path, level: int = 17, homie_checkpoint: Path | None = None, name: str | None = None):
        from homie_pushcart.mujoco_viewer import find_hl_checkpoint
        ckpt = find_hl_checkpoint(None if str(checkpoint) == "latest" else Path(checkpoint))
        super().__init__(ckpt, level, homie_checkpoint, name=name or f"two_stage:{ckpt.parent.name}/{ckpt.stem}")


# ---------------------------------------------------------------------------- robot without cart

class RobotOnlyPolicy:
    """Frozen HOMIE alone (g1_grasp.xml); the follower drives the pelvis. No cart, so no interaction telemetry."""

    tracks = "robot"

    def __init__(self, homie_checkpoint: Path | None = None):
        from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
        from homie_pushcart.mujoco_env import MujocoHomieEnv
        self.env = MujocoHomieEnv(with_cart=False)
        self.homie_checkpoint = Path(homie_checkpoint) if homie_checkpoint else frozen_checkpoint()
        self.policy = HimLowLevelPolicy(self.homie_checkpoint, device="cpu")
        self.dt = self.env.dt
        self.name = "robot_only"
        self._obs = None
        self._fallen = False

    def describe(self) -> dict:
        return {"name": self.name, "kind": "robot_only", "tracks": self.tracks, "homie_checkpoint": str(self.homie_checkpoint)}

    def reset(self, payload_kg: float = 0.0, caster_angle: float = 0.0, seed: int = 0, *,
              wheel_friction: float | None = None, com_offset_m: float = 0.0) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        mujoco.mj_resetData(self.env.model, self.env.data)
        self.env.set_command(0.0, 0.0, 0.0)
        self._obs = self.env.reset()
        self._fallen = False

    def step(self, cmd) -> None:
        cmd = np.asarray(cmd, dtype=float)
        self.env.set_command(float(cmd[0]), float(cmd[1]), float(cmd[2]))
        self.env.obs_hist[-1, :3] = self.env.commands[:3] * self.env.commands_scale        # HOMIE sees the new command
        obs_t = torch.from_numpy(np.clip(self.env.obs_hist.reshape(-1), -self.env.clip_obs, self.env.clip_obs)).unsqueeze(0)
        actions = self.policy(obs_t).squeeze(0).numpy()
        self._obs, fallen = self.env.step(actions)
        self._fallen = bool(fallen)

    def pose(self) -> tuple[float, float, float]:
        p = self.env.data.xpos[self.env.pelvis_id]
        return float(p[0]), float(p[1]), _yaw(self.env.base_quat_wxyz)

    def velocity(self) -> np.ndarray:
        v = self.env.base_lin_vel
        return np.array([v[0], v[1], self.env.base_ang_vel[2]], dtype=float)

    def failure(self) -> str | None:
        return "fall" if self._fallen else None

    def telemetry(self) -> dict:
        return {}

    # knobs are no-ops without a cart
    def restore_physics(self): ...
    def set_wheel_friction(self, mu): ...
    def set_payload_com_offset(self, dx, dy=0.0): ...
    def kick(self, dv_xy): ...
    def set_wheel_damping_factor(self, wheel, factor): ...
    def cart_mass_total(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------- registry

def parse_policy_spec(spec: str) -> dict:
    """``passthrough`` | ``robot_only`` | ``two_stage:<ckpt|latest>`` | ``plugin:<module>:<attr>[:<arg>]``."""
    spec = spec.strip()
    if spec in ("passthrough", "robot_only"):
        return {"kind": spec}
    if spec.startswith("two_stage"):
        rest = spec[len("two_stage"):]
        ckpt = rest[1:] if rest.startswith(":") and len(rest) > 1 else "latest"
        return {"kind": "two_stage", "checkpoint": ckpt}
    if spec.startswith("plugin:"):
        parts = spec.split(":", 3)
        if len(parts) < 3:
            raise ValueError(f"plugin spec needs plugin:<module>:<attr>[:<arg>], got {spec!r}")
        out = {"kind": "plugin", "module": parts[1], "attr": parts[2]}
        if len(parts) == 4:
            out["arg"] = parts[3]
        return out
    raise ValueError(f"unknown policy spec {spec!r}; use passthrough | robot_only | two_stage[:ckpt|latest] | plugin:module:Class")


def make_policy(spec: str, level: int = 17, homie_checkpoint: Path | None = None):
    p = parse_policy_spec(spec)
    if p["kind"] == "passthrough":
        return PassthroughPolicy(level, homie_checkpoint)
    if p["kind"] == "robot_only":
        return RobotOnlyPolicy(homie_checkpoint)
    if p["kind"] == "two_stage":
        return TwoStagePolicy(p["checkpoint"], level, homie_checkpoint)
    module = importlib.import_module(p["module"])
    cls = getattr(module, p["attr"])
    return cls(p["arg"]) if "arg" in p else cls()
