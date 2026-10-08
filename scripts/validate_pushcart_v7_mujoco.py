"""V7 plant validation on CPU (MuJoCo). Writes logs/validation/v7_mujoco/report*.json.

1. clutch    The two joint laws of the grasp chain with the exact MJCF parameters:
             slide  -- elasto-plastic friction (anchor spring capped at the limit, anchor dragged past it):
                       elastic below the limit, slides at it, stays put (elastic recovery only) after release;
             pivot  -- general actuator -k q - d qdot clamped to +-capacity: elastic (q = tau / k) below the
                       capacity, yields at it, returns to the seat when the load is removed.
2. geometry  MuJoCo's own collision distance between every hand geom and the square handle (no penetration,
             fingers resting on it).
3. demand    The trained V6 policy (rigid V3.2 grasp) and a no-policy passthrough driving the V7 plant: the
             transmitted along-bar force stays under the load-dependent friction limit, the pivot moments under
             the capacities (net of chain inertia), and how often the hand yields / breaks.
4. casters   Start-up response to the initial caster headings with V3.2-style (viscous only) and V7 (Coulomb)
             wheel/swivel resistance, same policy and command.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


# ----------------------------------------------------------------------------------------------- 1. clutch

SLIDE_XML = """
<mujoco><compiler angle="radian"/><option timestep="0.005" integrator="implicitfast" gravity="0 0 0"/>
<worldbody><body name="bar">
  <body name="slider"><joint name="slide" type="slide" axis="0 1 0" armature="0.01" range="-0.5 0.5" frictionloss="0"/>
    <inertial pos="0 0 0" mass="0.05" diaginertia="1.5e-5 1.5e-5 1.5e-5"/>
    <body name="hand"><inertial pos="0 0 0" mass="0.7" diaginertia="2e-3 2e-3 2e-3"/></body></body></body></worldbody>
<actuator><general name="servo" joint="slide" gaintype="fixed" biastype="affine" gainprm="{k}" biasprm="0 {mk} {md}"
  ctrllimited="true" ctrlrange="-0.5 0.5" forcelimited="true" forcerange="{mfs} {fs}"/></actuator></mujoco>"""

PIVOT_XML = """
<mujoco><compiler angle="radian"/><option timestep="0.005" integrator="implicitfast" gravity="0 0 0"/>
<worldbody><body name="bar">
  <body name="hand"><joint name="pivot" type="hinge" axis="1 0 0" armature="{arm}" range="-0.6 0.6"/>
    <inertial pos="0 0 0" mass="0.7" diaginertia="2e-3 2e-3 2e-3"/></body></body></worldbody>
<actuator><general name="servo" joint="pivot" gaintype="fixed" biastype="affine" gainprm="0" biasprm="0 {mk} {md}"
  ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="{mcap} {cap}"/></actuator></mujoco>"""


def slide_case(fs: float, k: float, damping_ratio: float, damping_mass: float, decimation: int = 4,
               peak: float = 2.0) -> dict:
    """Force ramp 0 -> peak x limit over 3 s along the tube, then release for 3 s; the anchor is updated once per
    control step (decimation physics steps), as in both tasks. peak < 1: purely elastic, the hand must return
    by the elastic deflection peak * limit / k on release; peak > 1: it slides, then stops and stays."""
    import mujoco
    from homie_pushcart_v7.physics import drag_anchor
    c = 2 * damping_ratio * math.sqrt(k * damping_mass)
    m = mujoco.MjModel.from_xml_string(SLIDE_XML.format(fs=fs, mfs=-fs, k=k, mk=-k, md=-c))
    d = mujoco.MjData(m)
    hand, dt = m.body("hand").id, m.opt.timestep
    anchor, rows = 0.0, []
    for i in range(int(6.0 / dt)):
        t = i * dt
        s = min(t / 3.0, 1.0) if t < 3.0 else 0.0
        d.xfrc_applied[hand] = [0, peak * fs * s, 0, 0, 0, 0]
        d.ctrl[0] = anchor
        mujoco.mj_step(m, d)
        if (i + 1) % decimation == 0:
            anchor = float(drag_anchor(anchor, d.qpos[0], fs, k)[0])
        rows.append((t, peak * fs * s, d.qpos[0], d.qvel[0], anchor, d.actuator_force[0]))
    a = np.array(rows)
    elastic = (a[:, 0] < 3.0) & (a[:, 1] > 0.1 * fs) & (a[:, 1] < 0.9 * fs)
    lin_err = float(np.max(np.abs(a[elastic, 2] - a[elastic, 1] / k)) / (0.9 * fs / k))
    moved = np.abs(a[:, 4]) > 1e-7                                          # plastic slip: the anchor moved at all
    onset = float(a[np.argmax(moved), 1] / fs) if moved.any() else float("inf")
    loaded_end = a[a[:, 0] < 3.0][-1, 2]
    settled = a[a[:, 0] >= 4.0]
    return {"slide_limit_N": fs, "stiffness": k, "slide_onset_over_limit": onset,
            "elastic_linearity_error": lin_err,
            "anchor_motion_below_0.9_limit_m": float(np.abs(a[elastic, 4]).max()),
            "force_peak_over_limit": float(np.abs(a[:, 5]).max() / fs),
            "release_displacement_m": float(abs(settled[-1, 2] - loaded_end)), "elastic_deflection_m": peak * fs / k,
            "slide_speed_1s_after_release": float(abs(settled[0, 3])),
            "drift_last_2s_m": float(abs(settled[-1, 2] - settled[0, 2]))}


def pivot_case(k: float, cap: float, arm: float, ratio: float) -> dict:
    import mujoco
    d_ = 2 * ratio * math.sqrt(k * arm)
    m = mujoco.MjModel.from_xml_string(PIVOT_XML.format(arm=arm, mk=-k, md=-d_, cap=cap, mcap=-cap))
    d = mujoco.MjData(m)
    hand, dt = m.body("hand").id, m.opt.timestep
    rows = []
    for i in range(int(6.0 / dt)):                      # torque ramp 0 -> 1.5 x capacity over 3 s, then release
        t = i * dt
        s = min(t / 3.0, 1.0) if t < 3.0 else 0.0
        tau = 1.5 * cap * s
        d.xfrc_applied[hand] = [0, 0, 0, tau, 0, 0]
        mujoco.mj_step(m, d)
        rows.append((t, tau, d.qpos[0], d.qvel[0], d.actuator_force[0]))
    a = np.array(rows)
    elastic = (a[:, 0] < 3.0) & (a[:, 1] < 0.8 * cap) & (a[:, 1] > 0.1 * cap)
    lin_err = float(np.max(np.abs(a[elastic, 2] - a[elastic, 1] / k)) / (0.8 * cap / k))
    yield_at = float(a[np.argmax(np.abs(a[:, 3]) > 0.05), 1] / cap)            # runaway rate > 0.05 rad/s
    peak_force = float(np.abs(a[:, 4]).max() / cap)
    rest = a[a[:, 0] >= 5.5]
    return {"stiffness": k, "capacity_Nm": cap, "elastic_linearity_error": lin_err, "yield_onset_over_capacity": yield_at,
            "servo_force_peak_over_capacity": peak_force, "angle_at_yield_rad": float(cap / k),
            "residual_angle_2.5s_after_release_rad": float(abs(rest[-1, 2]))}


def test_clutch() -> dict:
    from homie_pushcart_v7.assets.build import FRICTION_SOLIMP, FRICTION_SOLREF, load_v7_params
    g = load_v7_params()["grasp"]
    ks = g["slide_stiffness_range"]
    slide = {f"{fs:g}N_k{k:g}": slide_case(fs, k, g["slide_damping_ratio"], g["slide_damping_mass_kg"])
             for fs in (10., 36.25, 75.) for k in (ks[0], ks[1])}
    unload = {f"{fs:g}N_k{k:g}_0.9": slide_case(fs, k, g["slide_damping_ratio"], g["slide_damping_mass_kg"], peak=0.9)
              for fs in (10., 75.) for k in (ks[0], ks[1])}
    pivot = {f"k{k:g}_cap{c:g}": pivot_case(k, c, g["rotation_armature"], g["rotation_damping_ratio"])
             for k, c in ((80., 1.0), (190., 1.75), (300., 2.5), (80., 3.), (300., 6.),      # tilt / yaw
                          (15., 1.0), (60., 2.5))}                                          # roll
    ok_slide = all(abs(c["slide_onset_over_limit"] - 1) < 0.05 and c["elastic_linearity_error"] < 0.05
                   and c["anchor_motion_below_0.9_limit_m"] < 1e-6 and c["force_peak_over_limit"] <= 1.0 + 1e-6
                   and c["slide_speed_1s_after_release"] < 1e-3 and c["drift_last_2s_m"] < 1e-4
                   for c in slide.values())
    ok_unload = all(c["slide_onset_over_limit"] == float("inf") and c["drift_last_2s_m"] < 1e-4
                    and abs(c["release_displacement_m"] - c["elastic_deflection_m"]) < 0.05 * c["elastic_deflection_m"]
                    for c in unload.values())
    ok_pivot = all(c["elastic_linearity_error"] < 0.05 and abs(c["yield_onset_over_capacity"] - 1) < 0.1
                   and c["servo_force_peak_over_capacity"] <= 1.0 + 1e-6 and c["residual_angle_2.5s_after_release_rad"] < 0.01
                   for c in pivot.values())
    return {"pass": bool(ok_slide and ok_unload and ok_pivot), "slide": slide, "slide_unload": unload, "pivot": pivot,
            "caster_friction_constraint": {"solreffriction": FRICTION_SOLREF, "solimpfriction": FRICTION_SOLIMP}}


# ----------------------------------------------------------------------------------------------- 2. geometry

def test_geometry() -> dict:
    import mujoco
    from homie_pushcart_v7 import physics
    env = physics.make_mujoco_env()
    m, d = env.model, env.data
    d.qpos[:] = env.preset
    mujoco.mj_forward(m, d)
    bar = m.geom("cart_bar_0").id
    ft = np.zeros(6)
    out, ok = {"bar_geom_type": int(m.geom_type[bar]), "bar_size_m": m.geom_size[bar].tolist()}, True
    for side in ("left", "right"):
        rec = {}
        for i in range(m.ngeom):
            b = m.body(m.geom_bodyid[i]).name
            if (side + "_hand") in b or (b == f"{side}_wrist_yaw_link" and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH):
                name = b.replace(f"{side}_hand_", "").replace(f"{side}_wrist_yaw_link", "palm(merged)")
                dist = float(mujoco.mj_geomDistance(m, d, i, bar, 0.05, ft))
                rec[name] = min(rec.get(name, 1.), dist)
        touching = [k for k, v in rec.items() if v < 0.0015]
        out[side] = {"distance_m": rec, "touching_within_1.5mm": touching, "min_m": min(rec.values())}
        ok &= min(rec.values()) > -1e-4 and len(touching) >= 5
        print("[geometry]", side, {k: round(v * 1e3, 2) for k, v in rec.items()}, flush=True)
    return {"pass": bool(ok), **out}


# ----------------------------------------------------------------------------------------------- 3/4. full plant

SCENARIOS = {  # (seconds, task command vx vy wz)
    "go_stop": [(1.0, (0, 0, 0)), (4.0, (0.6, 0, 0)), (3.0, (0, 0, 0)), (4.0, (0.9, 0, 0)), (3.0, (0, 0, 0))],
    "turn": [(1.0, (0, 0, 0)), (3.0, (0.3, 0, 0)), (4.0, (0.3, 0, 0.4)), (4.0, (0.3, 0, -0.4)), (2.0, (0, 0, 0))],
    "reverse": [(1.0, (0, 0, 0)), (4.0, (-0.4, 0, 0)), (3.0, (0, 0, 0)), (3.0, (0.4, 0, 0)), (2.0, (0, 0, 0))],
}


class Plant:
    """The V6 policy (trained on the rigid V3.2 grasp) -- or no high-level policy -- driving the V7 MuJoCo plant."""

    def __init__(self, checkpoint: str):
        import mujoco
        import homie_pushcart_v6.mujoco_runner as v6_runner
        from homie_pushcart_v7 import physics as v7_physics
        from homie_pushcart_v7.assets.build import load_v7_params
        from pushcart_bench.policies_v6 import resolve_v6_checkpoint
        v6_runner.make_mujoco_env = v7_physics.make_mujoco_env          # same policy, V7 plant
        self.checkpoint = resolve_v6_checkpoint(checkpoint)
        self.runner = v6_runner.MujocoV6Runner(self.checkpoint, 17)
        self.policy_hl = self.runner.hl
        self.env, self.m, self.d = self.runner.env, self.runner.env.model, self.runner.env.data
        self.mj = mujoco
        self.params = load_v7_params()
        self.grasp = v7_physics.MujocoGrasp(self.env, self.params)
        self.swivel = [self.m.joint(f"caster_{i}_swivel_joint") for i in range(4)]
        self.wheel = [self.m.joint(f"wheel_{i}_joint") for i in range(4)]
        self.default_dof = (self.m.dof_frictionloss.copy(), self.m.dof_damping.copy())
        self.cart = self.runner.cart_body
        self.casters = None

    def configure(self, *, mu, normal, stiffness, roll_cap, moment_cap, payload, driver="v6_policy",
                  cart_resistance="v7", rolling_coeff=0.025, swivel_nm=0.225):
        m = self.m
        m.dof_frictionloss[:], m.dof_damping[:] = self.default_dof[0], self.default_dof[1]
        total = self.params["mass"]["base_kg"] + payload
        for sw, wh in zip(self.swivel, self.wheel):
            if cart_resistance == "v32":            # V3.2 Isaac: viscous only
                m.dof_frictionloss[sw.dofadr[0]] = 0.0
                m.dof_frictionloss[wh.dofadr[0]] = 0.0
                m.dof_damping[sw.dofadr[0]] = self.params["wheels"]["swivel_damping"]
                m.dof_damping[wh.dofadr[0]] = self.params["wheels"]["wheel_damping"]
            else:
                m.dof_frictionloss[sw.dofadr[0]] = swivel_nm
                m.dof_frictionloss[wh.dofadr[0]] = rolling_coeff * total * 9.81 / 4 * self.params["wheels"]["radius"]
        # "v7": swivel scrub fading with the wheel's rolling speed (physics.MujocoCasters, 2026-10-08);
        # "v7_const": the earlier constant scrub
        self.casters = (v7_physics_casters(self.env, self.params, swivel_nm) if cart_resistance == "v7" else None)
        self.grasp.configure(mu, normal, stiffness, roll_cap, moment_cap)
        self.runner.set_payload(payload)
        self.runner.hl = self.policy_hl if driver == "v6_policy" else None
        self.caps = np.array([roll_cap, moment_cap, moment_cap])

    def reset(self, casters):
        self.runner.reset(caster_angle=0.0)
        for joint, angle in zip(self.swivel, casters):
            self.d.qpos[joint.qposadr[0]] = angle
        self.mj.mj_forward(self.m, self.d)
        self.grasp.reset_anchor()
        self.grasp.update(np.zeros((2, 6)))

    def rollout(self, seq, settle=25):
        rec = {k: [] for k in ("loads", "pivot_m", "limit", "slide_q", "slide_v", "slip", "slide_a", "rot_q", "rot_v",
                               "rot_a", "cart")}
        fallen = False
        for secs, cmd in seq:
            for _ in range(int(round(secs / self.runner.dt))):
                limit = self.grasp.slide_limit_now.copy()       # the limit in force during this step
                _, fell = self.runner.step(np.asarray(cmd, dtype=float))
                fallen |= bool(fell)
                if self.casters is not None:
                    self.casters.update()
                loads = self.grasp.hand_loads()
                self.grasp.update(loads)
                st = self.grasp.state()
                qacc = self.d.qacc
                rec["loads"].append(loads)
                rec["pivot_m"].append(self.grasp.pivot_moments())
                rec["limit"].append(limit)
                rec["slide_q"].append(st["slide_q"])
                rec["slide_v"].append(st["slide_v"])
                rec["slip"].append(self.grasp.slip.copy())
                rec["slide_a"].append(np.array([qacc[a] for a in self.grasp.slide_dof]))
                rec["rot_q"].append(st["rot_q"])
                rec["rot_v"].append(st["rot_v"])
                rec["rot_a"].append(np.array([[qacc[a] for a in self.grasp.rot_dof[r]] for r in ("roll", "tilt", "yaw")]))
                vel = np.zeros(6)
                self.mj.mj_objectVelocity(self.m, self.d, self.mj.mjtObj.mjOBJ_BODY, self.cart, vel, 1)
                rec["cart"].append(np.array([vel[3], vel[4], vel[2]]))
                if fallen:
                    break
            if fallen:
                break
        out = {k: np.array(v)[settle:] if k != "cart" else np.array(v) for k, v in rec.items()}
        out["fallen"] = fallen
        return out


def v7_physics_casters(env, params, scrub):
    from homie_pushcart_v7.physics import MujocoCasters
    return MujocoCasters(env, params, scrub=scrub)


def chain_inertia(plant):
    g = plant.params["grasp"]
    return 4 * float(g["link_mass_kg"]) + float(g["slide_armature"]), float(g["rotation_armature"])


GRIPS = {"weak": dict(mu=0.3, normal=50., stiffness=80., roll_cap=1.0, moment_cap=3.0),
         "nominal": dict(mu=0.55, normal=75., stiffness=190., roll_cap=1.75, moment_cap=4.5),
         "strong": dict(mu=0.8, normal=100., stiffness=300., roll_cap=2.5, moment_cap=6.0)}


def test_demand(plant: Plant) -> dict:
    g = plant.params["grasp"]
    m_slide, i_rot = chain_inertia(plant)
    lo = np.array([plant.m.jnt_range[plant.m.joint(f"grasp_{s}_slide_joint").id][0] for s in ("left", "right")])
    hi = np.array([plant.m.jnt_range[plant.m.joint(f"grasp_{s}_slide_joint").id][1] for s in ("left", "right")])
    pull_cap, lift_cap = np.mean(g["pull_capacity_range"]), np.mean(g["lift_capacity_range"])
    results, ok = {}, True
    for driver in ("v6_policy", "passthrough"):
        for label, grip in GRIPS.items():
            for payload in (0.0, 25.0):
                for name, seq in SCENARIOS.items():
                    plant.configure(payload=payload, driver=driver, **grip)
                    plant.reset([0.0] * 4)
                    r = plant.rollout(seq)
                    loads = r["loads"]
                    finite = bool(np.isfinite(loads).all() and np.isfinite(r["rot_q"]).all())
                    free = (r["slide_q"] > lo + 2e-3) & (r["slide_q"] < hi - 2e-3)
                    along_net = (np.abs(loads[..., 1]) - m_slide * np.abs(r["slide_a"])) / r["limit"]
                    moments = r["pivot_m"]                                    # (T, 3 roll/tilt/yaw, 2), joint axes
                    moment_net = (moments - i_rot * np.abs(r["rot_a"])) / plant.caps[None, :, None]
                    pivot = np.maximum(np.abs(r["rot_q"][:, 0]) / g["roll_break_rad"],
                                       np.abs(r["rot_q"][:, 1:]).max(1) / g["tilt_break_rad"])     # (T, 2)
                    elastic_limit = plant.caps / grip["stiffness"]
                    yielding = np.abs(r["rot_q"]) > 1.2 * elastic_limit[None, :, None]
                    rec = {
                        "fallen": bool(r["fallen"]), "finite": finite,
                        "along_bar_net_p999_over_limit": float(np.percentile(along_net[free], 99.9)),
                        "pivot_moment_net_p999_over_capacity": float(np.percentile(moment_net, 99.9)),
                        "slide_limit_mean_N": float(r["limit"].mean()), "slide_limit_max_N": float(r["limit"].max()),
                        "slide_slipping_frac": float((np.abs(r["slip"]) > 1e-4).mean()),          # anchor moved
                        "slide_travel_max_m": float(np.abs(r["slide_q"]).max()),
                        "yielding_frac": float(yielding.any(axis=(1, 2)).mean()),
                        "pivot_max_over_break": float(pivot.max()),
                        "pivot_break_steps": int((pivot > 1.0).any(1).sum()),
                        "rot_q_abs_max_rad": np.abs(r["rot_q"]).max(axis=(0, 2)).round(4).tolist(),
                        "pull_over_capacity_frac": float((-loads[..., 0] > pull_cap).mean()),
                        "lift_over_capacity_frac": float((loads[..., 2] > lift_cap).mean()),
                    }
                    # along-bar load net of the chain's inertia vs the limit (spring + damping capped by forcerange)
                    rec["pass"] = (finite and not r["fallen"] and rec["along_bar_net_p999_over_limit"] < 1.08
                                   and rec["pivot_moment_net_p999_over_capacity"] < 1.05)
                    ok &= rec["pass"]
                    key = f"{driver}/{label}/{payload:.0f}kg/{name}"
                    results[key] = rec
                    print("[demand]", key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    return {"pass": bool(ok), "runs": results}


def test_casters(plant: Plant, seeds: int = 6) -> dict:
    rng = np.random.default_rng(0)
    headings = {"aligned_0": [[0.0] * 4], "all_90deg": [[math.pi / 2] * 4], "all_180deg": [[math.pi] * 4],
                "per_wheel_random": [list(rng.uniform(-math.pi, math.pi, 4)) for _ in range(seeds)]}
    seq = [(0.5, (0, 0, 0)), (2.5, (0.5, 0, 0))]
    out = {}
    for model in ("v32", "v7_const", "v7"):
        for label, sets in headings.items():
            recs = []
            for casters in sets:
                plant.configure(payload=15.0, cart_resistance=model, **GRIPS["nominal"])
                plant.reset(casters)
                r = plant.rollout(seq, settle=0)
                cart = r["cart"]
                recs.append({"lateral_speed_peak_m_s": float(np.abs(cart[:, 1]).max()),
                             "yaw_rate_peak_rad_s": float(np.abs(cart[:, 2]).max()),
                             "yaw_drift_rad": float(abs(np.sum(cart[:, 2]) * plant.runner.dt)),
                             "forward_speed_at_1s": float(cart[min(len(cart) - 1, int(1.5 / plant.runner.dt)), 0]),
                             "fallen": bool(r["fallen"])})
            out[f"{model}/{label}"] = {k: float(np.mean([x[k] for x in recs])) for k in recs[0]}
            print(f"[casters] {model:4s} {label:16s}", {k: round(v, 3) for k, v in out[f'{model}/{label}'].items()}, flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="latest", help="V6 checkpoint used as the zero-shot driver")
    parser.add_argument("--only", choices=("clutch", "geometry", "demand", "casters"))
    parser.add_argument("--output", type=Path, default=ROOT / "logs/validation/v7_mujoco")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    report = {}
    if args.only in (None, "clutch"):
        report["clutch"] = test_clutch()
        print("[clutch]", json.dumps(report["clutch"], indent=1), flush=True)
    if args.only in (None, "geometry"):
        report["geometry"] = test_geometry()
    if args.only in (None, "demand", "casters"):
        plant = Plant(args.checkpoint)
        report["driver_checkpoint"] = str(plant.checkpoint)
        if args.only in (None, "demand"):
            report["demand"] = test_demand(plant)
        if args.only in (None, "casters"):
            report["casters"] = test_casters(plant)
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"report_{args.only}.json" if args.only else "report.json"
    (args.output / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = all(v.get("pass", True) for v in report.values() if isinstance(v, dict))
    print("V7 MUJOCO VALIDATION", "PASS" if passed else "FAIL", "->", args.output / name)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
