"""Cart alone (no robot): planar wrench needed to push at 0.4 m/s and steer +-0.4 rad/s, per caster model, in
MuJoCo and in Isaac/PhysX (the same cart, the same friction writers as the V7 task).

Physics expectation for a rolling cart with free-swivel casters: once the casters trail their wheel velocity, a steady
turn needs only the yaw component of the rolling resistance (~0) -- swivel friction acts only while a caster turns
relative to the cart. A constant Coulomb swivel torque tau leaves every caster stuck inside a dead band where its
wheel carries a side force up to tau / trail, which resists the turn by sum(tau / trail * lever).

    E:\\envs\\RLisaac601\\python.exe scripts/caster_steer_test.py mujoco
    E:\\envs\\RLisaac601\\python.exe scripts/caster_steer_test.py physx --headless
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
OUT = ROOT / "logs/validation/v7_caster"
STIFF = {"solreffriction": "0.005 1", "solimpfriction": "0.999 0.9999 0.001"}
# name: (swivel Coulomb torque N.m at standstill, swivel viscous N.m.s/rad, rolling-dependent scrub)
VARIANTS = {"legacy_viscous": (0.0, 0.05, False), "coulomb_0.05": (0.05, 0.025, False),
            "coulomb_0.225": (0.225, 0.025, False), "coulomb_0.4": (0.4, 0.025, False),
            "rolling_0.225": (0.225, 0.025, True), "rolling_0.4": (0.4, 0.025, True)}
DECIMATION = 4                       # the law is re-evaluated once per 20 ms control step, as in the task


def law():
    from homie_pushcart_v7.assets.build import load_v7_params
    w = load_v7_params()
    ww = w["wheels_v7"]
    return 0.5 * sum(ww["swivel_bearing_range"]), float(ww["swivel_scrub_speed_m_s"]), float(w["wheels"]["radius"])
DT, T_END = 0.005, 18.0


def reference(t):
    return 0.4, (0.0 if t < 6 else (0.4 if t < 12 else -0.4))


class PI:
    def __init__(self):
        self.iv = self.iw = 0.0

    def __call__(self, vx, wz, t):
        v_ref, w_ref = reference(t)
        ev, ew = v_ref - vx, w_ref - wz
        self.iv += ev * DT
        self.iw += ew * DT
        return 150 * ev + 150 * self.iv, 60 * ew + 120 * self.iw


def summarize(rows) -> dict:
    a = np.array(rows)               # t, vx, wz, Fx, Mz

    def win(t0, t1):
        return a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    st, tr, ss, rev = win(3, 6), win(6, 8), win(9, 12), win(12, 14)
    return {"push_force_N": float(st[:, 3].mean()), "turn_start_Mz_p95": float(np.percentile(np.abs(tr[:, 4]), 95)),
            "steady_turn_Mz": float(ss[:, 4].mean()), "steady_turn_wz": float(ss[:, 2].mean()),
            "reversal_Mz_p95": float(np.percentile(np.abs(rev[:, 4]), 95))}


# ---------------------------------------------------------------------------------------------------- MuJoCo

def mujoco_cart(swivel_tau, swivel_visc, payload=0.0, scrub_model=False):
    import mujoco
    from homie_pushcart_v7.paths import MJCF_SCENE
    root = ET.parse(MJCF_SCENE).getroot()
    for wb in root.findall("worldbody"):
        for ch in list(wb):
            if not (ch.tag in ("geom", "light") or (ch.tag == "body" and ch.get("name") == "cart_base")):
                wb.remove(ch)
    cart = next(b for b in root.iter("body") if b.get("name") == "cart_base")
    for b in list(cart.findall("body")):
        if not b.get("name", "").startswith("caster"):
            cart.remove(b)
    for tag in ("equality", "actuator", "contact", "sensor", "keyframe"):
        for e in root.findall(tag):
            root.remove(e)
    cart.find("inertial").set("mass", str(12.0 + payload))
    for j in cart.iter("joint"):
        if j.get("name", "").endswith("_swivel_joint"):
            j.set("frictionloss", f"{swivel_tau:g}")
            j.set("damping", f"{swivel_visc:g}")
            j.attrib.update(STIFF)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def run_mujoco(payload=0.0) -> dict:
    import mujoco
    out = {}
    from homie_pushcart_v7.physics import swivel_torque
    bearing, vs, radius = law()
    for name, (tau, visc, rolling) in VARIANTS.items():
        m = mujoco_cart(tau, visc, payload)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cb = m.body("cart_base").id
        sw = [int(m.joint(f"caster_{i}_swivel_joint").dofadr[0]) for i in range(4)]
        wh = [int(m.joint(f"wheel_{i}_joint").dofadr[0]) for i in range(4)]
        pi, rows = PI(), []
        for k in range(int(T_END / DT)):
            t = k * DT
            if rolling and k % DECIMATION == 0:
                m.dof_frictionloss[sw] = swivel_torque(tau, bearing, d.qvel[wh] * radius, vs)
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, cb, v, 0)
            R = d.xmat[cb].reshape(3, 3)
            vx, wz = float(R[:, 0] @ v[3:]), float(v[2])
            Fx, Mz = pi(vx, wz, t)
            d.xfrc_applied[cb, :3] = R[:, 0] * Fx
            d.xfrc_applied[cb, 3:] = [0, 0, Mz]
            mujoco.mj_step(m, d)
            rows.append((t, vx, wz, Fx, Mz))
        out[name] = summarize(rows)
        print("[mujoco]", name, out[name], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- PhysX

def run_physx(device: str, payload=0.0) -> dict:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab_physx.physics import PhysxCfg
    from homie_pushcart_v7.paths import CART_URDF

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=device, physics=PhysxCfg()))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    arts = []
    for k, name in enumerate(VARIANTS):
        cfg = ArticulationCfg(
            prim_path=f"/World/cart_{k}",
            spawn=sim_utils.UrdfFileCfg(asset_path=str(CART_URDF), fix_base=False, merge_fixed_joints=False,
                                        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                                            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 4. * k, 0.)),
            actuators={"casters": ImplicitActuatorCfg(joint_names_expr=["caster_.*_swivel_joint", "wheel_.*_joint"],
                                                      stiffness=0., damping=0.),
                       "grasp": ImplicitActuatorCfg(joint_names_expr=["grasp_.*"], stiffness=1e4, damping=50.)})
        arts.append(Articulation(cfg))
    sim.reset()
    from homie_pushcart_v7.assets.build import load_v7_params
    params = load_v7_params()
    w7 = params["wheels_v7"]
    mean = lambda r: 0.5 * (r[0] + r[1])                                                     # noqa: E731
    mass = 12.0 + payload
    from homie_pushcart_v7.physics import swivel_torque
    bearing, vs, radius = law()
    for art, (name, (tau, visc, _)) in zip(arts, VARIANTS.items()):
        nj = art.num_joints
        sw = art.find_joints("caster_.*_swivel_joint")[0]
        wh = art.find_joints("wheel_.*_joint")[0]
        damping = torch.zeros(1, nj, device=device)
        damping[:, sw] = visc
        damping[:, wh] = mean(w7["wheel_viscous_range"])
        art.write_joint_damping_to_sim_index(damping=damping)
        static, dynamic = torch.zeros(1, nj, device=device), torch.zeros(1, nj, device=device)
        rolling = mean(w7["rolling_coeff_range"]) * (mass + 5.0) * 9.81 / 4 * params["wheels"]["radius"]
        static[:, wh], dynamic[:, wh] = rolling * mean(w7["breakaway_over_rolling"]), rolling
        static[:, sw], dynamic[:, sw] = tau, 0.9 * tau
        art.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=static, joint_dynamic_friction_coeff=dynamic,
                                                          joint_viscous_friction_coeff=torch.zeros_like(static))
    base = [art.find_bodies("base_link")[0] for art in arts]
    for _ in range(100):                     # settle on the wheels
        sim.step()
        for art in arts:
            art.update(DT)
    pis, rows = [PI() for _ in arts], [[] for _ in arts]
    for k in range(int(T_END / DT)):
        t = k * DT
        if k % DECIMATION == 0:          # the task's writer: static / dynamic swivel friction from the wheel speed
            for art, (name, (tau, visc, rolling)) in zip(arts, VARIANTS.items()):
                if not rolling:
                    continue
                sw, wh = art.find_joints("caster_.*_swivel_joint")[0], art.find_joints("wheel_.*_joint")[0]
                jv = art.data.joint_vel.torch if hasattr(art.data.joint_vel, "torch") else art.data.joint_vel
                t_sw = swivel_torque(tau, bearing, jv[:, wh] * radius, vs)
                st = torch.zeros(1, art.num_joints, device=device)
                dy = torch.zeros_like(st)
                rolling_t = 0.5 * sum(w7["rolling_coeff_range"]) * (mass + 5.0) * 9.81 / 4 * params["wheels"]["radius"]
                st[:, wh], dy[:, wh] = rolling_t * 0.5 * sum(w7["breakaway_over_rolling"]), rolling_t
                st[:, sw], dy[:, sw] = t_sw, 0.9 * t_sw
                art.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=st, joint_dynamic_friction_coeff=dy,
                                                                  joint_viscous_friction_coeff=torch.zeros_like(st))
        for i, art in enumerate(arts):
            q = art.data.root_quat_w.torch[0] if hasattr(art.data.root_quat_w, "torch") else art.data.root_quat_w[0]
            v = art.data.root_lin_vel_w.torch[0] if hasattr(art.data.root_lin_vel_w, "torch") else art.data.root_lin_vel_w[0]
            w = art.data.root_ang_vel_w.torch[0] if hasattr(art.data.root_ang_vel_w, "torch") else art.data.root_ang_vel_w[0]
            qw, qx, qy, qz = [float(x) for x in q]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            hx = (math.cos(yaw), math.sin(yaw))
            vx, wz = float(v[0]) * hx[0] + float(v[1]) * hx[1], float(w[2])
            Fx, Mz = pis[i](vx, wz, t)
            art.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[Fx * hx[0], Fx * hx[1], 0.]]], device=device),
                torques=torch.tensor([[[0., 0., Mz]]], device=device), body_ids=base[i], is_global=True)
            art.write_data_to_sim()
            rows[i].append((t, vx, wz, Fx, Mz))
        sim.step()
        for art in arts:
            art.update(DT)
    out = {}
    for (name, _), r in zip(VARIANTS.items(), rows):
        out[name] = summarize(r)
        print("[physx]", name, out[name], flush=True)
    sim.clear_instance()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backend", choices=("mujoco", "physx"))
    p.add_argument("--payload", type=float, default=0.0)
    if "physx" in (argv or sys.argv[1:]):
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.backend == "mujoco":
        res = run_mujoco(a.payload)
    else:
        from isaaclab.app import AppLauncher
        app = AppLauncher(a)
        try:
            res = run_physx(a.device, a.payload)
        finally:
            (OUT / f"steer_{a.backend}_{a.payload:g}kg.json").write_text(json.dumps(locals().get("res", {}), indent=1), encoding="utf-8")
            app.app.close()
    (OUT / f"steer_{a.backend}_{a.payload:g}kg.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
