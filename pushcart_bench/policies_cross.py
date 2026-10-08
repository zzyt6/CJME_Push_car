"""Any high-level policy on any MuJoCo plant: the bench's way to compare versions on one common plant.

Spec ``<kind>@<plant>[:<checkpoint>]`` (see ``policies.parse_policy_spec``):

    kind   passthrough | v6 | v7
    plant  legacy   the frozen V3.2 four-sphere plant every bench before V7 used (palms welded to the cart)
           v7       the V7 plant: square-tube grasp chain (elasto-plastic slide, elastic-yield pivots), Coulomb
                    rolling resistance, swivel scrub fading with rolling speed (2026-10-08)
           v7_constswivel / v7_noswivel / v7_noroll / v7_viscous / v7_strong   diagnostic variants of the V7 plant

V6 and V7 share the Actor interface (policy 605 + HIM 462 -> 18 actions) and the V3.2 action pipeline, so a
checkpoint of either runs unchanged on either plant.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from .policies import CartPolicyBase

PLANTS = ("legacy", "v7", "v7_constswivel", "v7_noswivel", "v7_noroll", "v7_viscous", "v7_strong")


def _make_runner_cls():
    from homie_pushcart.mujoco_hl import MujocoHighLevelRunner

    class CrossRunner(MujocoHighLevelRunner):
        """V3.2 action pipeline; policy from ``version`` (6 / 7 / None = passthrough) on ``plant``."""

        def __init__(self, checkpoint, version, plant, level=17, homie_checkpoint=None):
            from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
            from homie_isaaclab.config import G1RoughCfg
            if plant not in PLANTS:
                raise ValueError(f"unknown plant {plant!r}; one of {PLANTS}")
            if version == 6 or (version is None and plant == "legacy"):
                from homie_pushcart_v6 import task_config as tcm, hl_policy as hlm, baseline as blm, observation as obm
            else:
                from homie_pushcart_v7 import task_config as tcm, hl_policy as hlm, baseline as blm, observation as obm
            self._obm = obm
            task = tcm.task_cfg_from_run(checkpoint, verbose=False) if checkpoint else tcm.PushCartTaskCfg()
            cfg = G1RoughCfg()
            blm.apply_config(cfg, blm.homie_baseline())
            if plant == "legacy":
                from homie_pushcart_v6.physics import make_mujoco_env
            else:
                from homie_pushcart_v7.physics import make_mujoco_env
            env = make_mujoco_env(cfg=cfg)
            self.homie_checkpoint = Path(homie_checkpoint or frozen_checkpoint())
            low = HimLowLevelPolicy(self.homie_checkpoint, "cpu")
            self.xpolicy = hlm.HighLevelPolicy(Path(checkpoint), device="cpu") if checkpoint else None
            self.plant = plant
            self.grasp = self.casters = None
            super().__init__(env, low, None, task, level)
            if self.xpolicy is not None:
                runner, pol = self, self.xpolicy

                class _Bridge:
                    iteration = pol.iteration

                    def __call__(self, obs):
                        return pol(obs, runner.estimator_obs)
                self.hl = _Bridge()
            if plant.startswith("v7"):
                from homie_pushcart_v7.assets.build import load_v7_params
                from homie_pushcart_v7.physics import MujocoCasters, MujocoGrasp
                self.grasp = MujocoGrasp(env, load_v7_params())
                self.grasp.reset_anchor()
                fixed_swivel = any(k in plant for k in ("constswivel", "noswivel", "viscous"))
                self.casters = None if fixed_swivel else MujocoCasters(env, load_v7_params())
            apply_plant_variant(self, plant)

        def set_payload(self, payload_kg: float):
            super().set_payload(payload_kg)
            if self.casters is not None and "noroll" not in self.plant:
                self.casters.set_cart_mass(float(self.env.cart_params["mass"]["base_kg"]) + float(payload_kg))

        def step(self, cmd_task):
            out = super().step(cmd_task)
            if self.grasp is not None:
                self.grasp.update(self.grasp.hand_loads())
            if self.casters is not None:
                self.casters.update()
            return out

        def reset_buffers(self):
            super().reset_buffers()
            if getattr(self, "grasp", None) is not None:
                self.grasp.reset_anchor()
            if getattr(self, "casters", None) is not None:
                self.casters.update()
            self.estimator_history = self._obm.HIMHistory(1)
            self.estimator_obs = self.estimator_history.buffer.flatten(1)
            self.pred_cart_velocity = np.zeros(3)

        def _update_hl_obs(self, fresh):
            super()._update_hl_obs(fresh)
            if getattr(self, "xpolicy", None) is None:
                return
            obm = self._obm
            legacy = torch.from_numpy(self.hl_obs.copy()).unsqueeze(0)
            self.estimator_obs = self.estimator_history.push(obm.him_frame(legacy.reshape(1, 5, 124)[:, -1]),
                                                             torch.tensor([fresh]))
            rv = getattr(self.task_cfg, "robot_velocity", None)
            if rv is not None and rv.replace_homie_estimate:         # V7.1 checkpoints: same masking as training
                legacy = obm.mask_homie_velocity(legacy)
            self.hl_obs = obm.without_handle_position(legacy)[0].numpy()
            with torch.inference_mode():
                speed, _ = self.xpolicy.physical_estimates(torch.from_numpy(self.hl_obs).unsqueeze(0), self.estimator_obs)
            self.pred_cart_velocity = speed[0].cpu().numpy()

    return CrossRunner


def apply_plant_variant(runner, variant: str):
    """Diagnostic variants of the V7 plant, each isolating one physical change (mutates the MuJoCo model)."""
    if variant in ("legacy", "v7", "v7_constswivel"):
        return
    m = runner.env.model
    if "strong" in variant:
        # nominal compliance, capacities that never bind (demand measurement)
        from homie_pushcart_v7.assets.build import load_v7_params, nominal_grasp
        nom = nominal_grasp(load_v7_params())
        runner.grasp.configure(nom["mu"], nom["normal"], nom["stiffness"], 100.0, 100.0)
    if any(k in variant for k in ("viscous", "noswivel", "noroll")):
        from homie_pushcart.assets.cart_urdf import load_cart_params
        wheels = load_cart_params()["wheels"]
        targets = []
        if "viscous" in variant or "noswivel" in variant:
            targets += [(f"caster_{i}_swivel_joint", "swivel_damping") for i in range(4)]
        if "viscous" in variant or "noroll" in variant:
            targets += [(f"wheel_{i}_joint", "wheel_damping") for i in range(4)]
        for name, key in targets:                       # V3.2 viscous drag, V3.2 (soft) friction-loss constraint
            j = m.joint(name)
            m.dof_frictionloss[j.dofadr[0]] = 0.0
            m.dof_damping[j.dofadr[0]] = float(wheels[key])
            m.dof_solref[j.dofadr[0]] = [0.02, 1.0]
            m.dof_solimp[j.dofadr[0]] = [0.9, 0.95, 0.001, 0.5, 2.0]


class CrossPolicy(CartPolicyBase):
    """``version`` (None = passthrough HOMIE, 6, 7) on ``plant``; records per-step hand loads for diagnostics."""

    def __init__(self, version, plant, checkpoint=None, level=17, homie_checkpoint=None, *, name=None):
        if version is not None and checkpoint is None:
            raise ValueError("a V6 / V7 policy needs a checkpoint")
        self.version, self.plant = version, plant
        self.runner = _make_runner_cls()(checkpoint, version, plant, level, homie_checkpoint)
        self.env = self.runner.env
        self.homie_checkpoint = self.runner.homie_checkpoint
        self.dt, self.level, self.hl_checkpoint = self.env.dt, level, Path(checkpoint) if checkpoint else None
        tag = "passthrough" if version is None else f"V{version}"
        self.name = name or (f"{tag}@{plant}" + (f" · {Path(checkpoint).parent.name}/{Path(checkpoint).stem}" if checkpoint else ""))
        m = self.env.model
        self.cart_body = self.runner.cart_body
        self.cart_qvel = int(m.joint("cart_free").dofadr[0])
        self.wheel_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{i}_geom") for i in range(4)]
        self.wheel_dofs = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        self._wheel_friction0 = m.geom_friction[self.wheel_geoms].copy()
        self._wheel_damping0 = m.dof_damping[self.wheel_dofs].copy()
        self._ipos0 = m.body_ipos[self.cart_body].copy()
        self._fallen, self._payload = False, 0.
        self.loads = []

    def reset(self, *a, **k):
        super().reset(*a, **k)
        self.loads = []

    def step(self, cmd):
        super().step(cmd)
        g = self.runner.grasp
        if g is not None:
            st = g.state()
            self.loads.append(np.concatenate([g.hand_loads().ravel(), st["rot_q"].ravel(), st["slide_q"] - g.anchor]))
        else:
            W = self.env.grasp_cart_wrenches()
            yaw = self.pose()[2]
            c, s = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            row = []
            for side in ("left", "right"):
                row += list(Rz.T @ W[side][:3]) + list(Rz.T @ W[side][3:])
            self.loads.append(np.array(row + [0.0] * 8))

    def describe(self):
        out = super().describe()
        out.update(kind=f"{'passthrough' if self.version is None else f'v{self.version}'}@{self.plant}",
                   version=self.version, plant=self.plant,
                   scene_selection="fixed_v32_legacy_four_spheres" if self.plant == "legacy" else f"v7_plant:{self.plant}")
        if self.plant.startswith("v7"):
            from homie_pushcart_v7.physics import physics_manifest
            out["physics"] = physics_manifest(require_usd=False)
        return out

    def telemetry(self):
        out = super().telemetry()
        out.update(dict(zip(("cart_est_vx", "cart_est_vy", "cart_est_wz"), map(float, self.runner.pred_cart_velocity))))
        return out
