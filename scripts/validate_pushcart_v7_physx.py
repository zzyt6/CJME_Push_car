"""V7 plant validation in Isaac Lab / PhysX (needs the GPU and the packed V7 USD).

1. clutch    The two joint laws of the V7 grasp chain as fixed-base articulations without gravity, with the
             parameters written through the same runtime APIs as the task:
             slide -- elasto-plastic friction: implicit drive to an anchor with stiffness k and maxForce = limit,
                      the anchor dragged once per control step (physics.drag_anchor): elastic (q = F / k) below the
                      limit, slides at it, stops and stays after release, springs back by F / k when unloaded
                      below the limit;
             pivot -- implicit drive to 0 with stiffness k and maxForce = capacity: elastic (q = tau / k) below the
                      capacity, yields at it, returns to the seat after release.
2. geometry  The packed USD as Isaac composes it: every Dex3 link pose relative to its palm equals the fitted grasp,
             and the hand surface rests on the square handle around the grasp chain axis (no penetration).
3. env       The V7 task environment with a fixed command and zero high-level action: observation sizes, finite
             values, along-bar load vs the per-step (load dependent) slide limit, pivots, terminations, step time,
             and the closed-loop check that failed with PhysX joint friction: below the limit the grip must not
             creep (plastic slip of the anchor ~0 in the welded robot-cart scene).

    E:\\envs\\RLisaac601\\python.exe scripts/validate_pushcart_v7_physx.py --only clutch --headless
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
OUT = ROOT / "logs/validation/v7_physx"

SLIDE_URDF = """<?xml version="1.0"?>
<robot name="slide_clutch">
  <link name="bar"><inertial><mass value="1"/><inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial></link>
  <link name="hand"><inertial><mass value="0.75"/><inertia ixx="2e-3" ixy="0" ixz="0" iyy="2e-3" iyz="0" izz="2e-3"/></inertial></link>
  <joint name="slide" type="prismatic"><parent link="bar"/><child link="hand"/><axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="1000" velocity="5"/></joint>
</robot>
"""
PIVOT_URDF = """<?xml version="1.0"?>
<robot name="pivot_servo">
  <link name="bar"><inertial><mass value="1"/><inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial></link>
  <link name="hand"><inertial><mass value="0.7"/><inertia ixx="2e-3" ixy="0" ixz="0" iyy="2e-3" iyz="0" izz="2e-3"/></inertial></link>
  <joint name="pivot" type="revolute"><parent link="bar"/><child link="hand"/><axis xyz="1 0 0"/>
    <limit lower="-0.6" upper="0.6" effort="1000" velocity="50"/></joint>
</robot>
"""


def _articulation(sim_utils, ArticulationCfg, ImplicitActuatorCfg, urdf, prim, pos, armature):
    from isaaclab.assets import Articulation
    cfg = ArticulationCfg(
        prim_path=prim,
        spawn=sim_utils.UrdfFileCfg(asset_path=str(urdf), fix_base=True, merge_fixed_joints=False,
                                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0., armature=armature,
                                              effort_limit_sim=1000.)})
    return Articulation(cfg)


def test_clutch(device: str) -> dict:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_physx.physics import PhysxCfg
    from homie_pushcart_v7.assets.build import load_v7_params

    g = load_v7_params()["grasp"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slide.urdf").write_text(SLIDE_URDF, encoding="utf-8")
    (OUT / "pivot.urdf").write_text(PIVOT_URDF, encoding="utf-8")
    ks = g["slide_stiffness_range"]
    slides = ((10., ks[1], 2.), (36.25, 0.5 * (ks[0] + ks[1]), 2.), (75., ks[0], 2.), (10., ks[1], 0.9), (75., ks[0], 0.9))
    s_ratio, s_mass = float(g["slide_damping_ratio"]), float(g["slide_damping_mass_kg"])
    decimation = 4
    pivots = ((80., 1.0), (190., 1.75), (300., 2.5), (80., 3.), (300., 6.),          # tilt / yaw
              (15., 1.0), (60., 2.5))                                                # roll
    arm, ratio = float(g["rotation_armature"]), float(g["rotation_damping_ratio"])
    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device, gravity=(0., 0., 0.), physics=PhysxCfg()))
    s_arts = [_articulation(sim_utils, ArticulationCfg, ImplicitActuatorCfg, OUT / "slide.urdf", f"/World/slide_{k}",
                            (0., 2. * k, 1.), 0.01) for k in range(len(slides))]
    p_arts = [_articulation(sim_utils, ArticulationCfg, ImplicitActuatorCfg, OUT / "pivot.urdf", f"/World/pivot_{k}",
                            (3., 2. * k, 1.), arm) for k in range(len(pivots))]
    sim.reset()
    from homie_pushcart_v7.physics import drag_anchor
    for art, (fs, k, _) in zip(s_arts, slides):          # the same writers the task uses
        art.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[k]], device=device))
        art.write_joint_damping_to_sim_index(damping=torch.tensor([[2 * s_ratio * math.sqrt(k * s_mass)]], device=device))
        art.write_joint_effort_limit_to_sim_index(limits=torch.tensor([[fs]], device=device))
        art.set_joint_position_target_index(target=torch.zeros(1, 1, device=device))
    anchors = [0.0] * len(slides)
    for art, (k, cap) in zip(p_arts, pivots):
        art.write_joint_stiffness_to_sim_index(stiffness=torch.tensor([[k]], device=device))
        art.write_joint_damping_to_sim_index(damping=torch.tensor([[2 * ratio * math.sqrt(k * arm)]], device=device))
        art.write_joint_effort_limit_to_sim_index(limits=torch.tensor([[cap]], device=device))
        art.set_joint_position_target_index(target=torch.zeros(1, 1, device=device))
    s_rows, p_rows = [[] for _ in s_arts], [[] for _ in p_arts]
    for step in range(int(6.0 / dt)):
        t = step * dt
        s = min(t / 3.0, 1.0) if t < 3.0 else 0.0
        for art, (fs, k, peak) in zip(s_arts, slides):
            art.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[0., peak * fs * s, 0.]]], device=device), body_ids=art.find_bodies("hand")[0], is_global=True)
            art.write_data_to_sim()
        for art, (k, cap) in zip(p_arts, pivots):
            art.permanent_wrench_composer.set_forces_and_torques_index(
                torques=torch.tensor([[[1.5 * cap * s, 0., 0.]]], device=device), body_ids=art.find_bodies("hand")[0], is_global=True)
            art.write_data_to_sim()
        sim.step()
        for i, art in enumerate(s_arts):
            art.update(dt)
            fs, k, peak = slides[i]
            q = art.data.joint_pos.torch[0, 0].item()
            if (step + 1) % decimation == 0:
                anchors[i] = float(drag_anchor(anchors[i], q, fs, k)[0])
                art.set_joint_position_target_index(target=torch.tensor([[anchors[i]]], device=device))
            s_rows[i].append((t, peak * fs * s, q, anchors[i]))
        for i, art in enumerate(p_arts):
            art.update(dt)
            p_rows[i].append((t, 1.5 * pivots[i][1] * s, art.data.joint_pos.torch[0, 0].item(), art.data.joint_vel.torch[0, 0].item()))
    out, ok = {"slide": {}, "pivot": {}}, True
    for (fs, k, peak), r in zip(slides, s_rows):
        a = np.array(r)                                     # t, applied force, q, anchor
        elastic = (a[:, 0] < 3.0) & (a[:, 1] > 0.1 * fs) & (a[:, 1] < 0.9 * fs)
        moved = np.abs(a[:, 3]) > 1e-7                      # plastic slip: the anchor moved at all
        loaded_end = a[a[:, 0] < 3.0][-1, 2]
        settled = a[a[:, 0] >= 4.0]
        rec = {"stiffness": k, "peak_over_limit": peak,
               "slide_onset_over_limit": float(a[np.argmax(moved), 1] / fs) if moved.any() else float("inf"),
               "elastic_linearity_error": float(np.max(np.abs(a[elastic, 2] - a[elastic, 1] / k)) / (0.9 * fs / k)),
               "release_displacement_m": float(abs(settled[-1, 2] - loaded_end)), "elastic_deflection_m": peak * fs / k,
               "drift_last_2s_m": float(abs(settled[-1, 2] - settled[0, 2]))}
        if peak > 1:
            # PhysX realises a stiff drive ~10 % stiffer at k 4e4 (k dt^2 / m ~ 1.3; pivots: 5 % at k 300); the stiffness
            # itself is an estimate spanning 1e4..4e4, so 15 % is a fidelity check, the friction limit is held to 5 %
            rec["pass"] = bool(abs(rec["slide_onset_over_limit"] - 1) < 0.05 and rec["elastic_linearity_error"] < 0.15
                               and rec["drift_last_2s_m"] < 1e-4)
        else:
            rec["pass"] = bool(rec["slide_onset_over_limit"] == float("inf") and rec["drift_last_2s_m"] < 1e-4
                               and abs(rec["release_displacement_m"] - rec["elastic_deflection_m"]) < 0.15 * rec["elastic_deflection_m"])
        ok &= rec["pass"]
        out["slide"][f"{fs:g}N_k{k:g}_x{peak:g}"] = rec
        print("[physx slide]", f"{fs:g}N k{k:g} x{peak:g}", rec, flush=True)
    for (k, cap), r in zip(pivots, p_rows):
        a = np.array(r)
        elastic = (a[:, 0] < 3.0) & (a[:, 1] < 0.8 * cap) & (a[:, 1] > 0.1 * cap)
        rest = a[a[:, 0] >= 5.5]
        # yield: the angle leaves the elastic line q = tau / k by more than 20 % (PhysX's reported velocity of a
        # driven joint is biased, so the test is position based)
        off = (a[:, 0] < 3.0) & (np.abs(a[:, 2]) > 1.2 * a[:, 1] / k + 1e-4)
        rec = {"elastic_linearity_error": float(np.max(np.abs(a[elastic, 2] - a[elastic, 1] / k)) / (0.8 * cap / k)),
               "yield_onset_over_capacity": float(a[np.argmax(off), 1] / cap) if off.any() else float("nan"),
               "angle_at_yield_rad": cap / k, "residual_angle_2.5s_after_release_rad": float(abs(rest[-1, 2]))}
        # stiffness is a wide estimate (80-300); PhysX realises it within ~5 %, MuJoCo within 1 %
        rec["pass"] = bool(rec["elastic_linearity_error"] < 0.10 and abs(rec["yield_onset_over_capacity"] - 1) < 0.1
                           and rec["residual_angle_2.5s_after_release_rad"] < 0.01)
        ok &= rec["pass"]
        out["pivot"][f"k{k:g}_cap{cap:g}"] = rec
        print("[physx pivot]", f"k{k:g} cap{cap:g}", rec, flush=True)
    sim.clear_instance()
    return {"pass": bool(ok), **out}


def test_geometry() -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    from homie_pushcart_v7.assets.build import load_v7_pose
    from homie_pushcart_v7.assets.grasp_fit import HandModel, Profile
    from homie_pushcart_v7.paths import GRASP_LINKS, PACKED_USD
    pose = load_v7_pose()
    fit = pose["grasp"]["v7_fit"]
    stage = Usd.Stage.Open(str(PACKED_USD))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    # rigid-body prims only: the importer also names visual sub-prims after their links
    prims = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath("/Scene"), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            if p.GetName() in prims:
                raise ValueError(f"duplicate rigid body name {p.GetName()}")
            prims[p.GetName()] = p
    world = lambda name: np.array(cache.GetLocalToWorldTransform(prims[name])).T          # noqa: E731
    out, ok = {}, True
    for side in ("left", "right"):
        hand = HandModel(side, samples_per_mesh=8000)
        angles = {j: v for j, v in pose["robot"]["finger_pos"].items() if j.startswith(side)}
        fk = hand.link_T(angles)
        palm = world(hand.palm)
        bar = world(GRASP_LINKS[side])                     # origin on the tube axis; x = cart forward, y = tube axis
        profile = Profile(fit["profile"], fit["size_m"], bar[:3, 1], bar[:3, 0], bar[:3, 2])
        pose_err, gaps = 0.0, {}
        for link, pts in hand.points.items():
            T = world(link)
            pose_err = max(pose_err, float(np.abs(np.linalg.inv(palm) @ T - fk[link]).max()))
            w = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            gaps[link.replace(f"{side}_hand_", "")] = float(profile.gap(w, bar[:3, 3]).min())
        touching = [k for k, g in gaps.items() if g < 0.0015]
        rec = {"link_pose_max_abs_error": pose_err, "surface_gap_m": gaps, "links_touching_within_1.5mm": touching,
               "min_gap_m": min(gaps.values())}
        rec["pass"] = bool(pose_err < 1e-5 and rec["min_gap_m"] > 0.0 and len(touching) >= 5)
        ok &= rec["pass"]
        out[side] = rec
        print("[physx geometry]", side, json.dumps(rec), flush=True)
    return {"pass": bool(ok), "profile": fit, **out}


def test_env(device: str, num_envs: int, steps: int, level: int = 0, load_friction: bool = True) -> dict:
    import torch
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v7.baseline import apply_config, homie_baseline
    from homie_pushcart_v7.task_config import PushCartTaskCfg, num_critic_obs
    from homie_pushcart_v7.task_env import PushCartEnv

    cfg = G1RoughCfg()
    apply_config(cfg, homie_baseline())
    cfg.env.num_envs = num_envs
    task = PushCartTaskCfg()
    task.curriculum.enabled = False
    task.curriculum_v7.enabled = False           # fixed difficulty d = level / 17 for a reproducible check
    task.curriculum.init_level = level
    task.grasp.load_dependent_friction = load_friction
    env = PushCartEnv(cfg, task, ROOT, device, True, level=level)
    env.fixed_command, env.fixed_payload_kg = [.5, 0., .2], 25.
    env.reset()
    sizes = {k: int(v.shape[-1]) for k, v in env.observation_dict().items()}
    zero = torch.zeros(num_envs, 18, device=device)
    ratio, limits, pivot, travel, finite = [], [], [], [], True
    creep, elastic = [], []
    terms = {k: 0 for k in env.term_reason_buf}
    kinds = np.zeros(4)
    for i in range(steps):
        limit = env.slide_limit_now.clone()                 # the limit in force during this control step
        anchor0 = env.slide_anchor.clone()
        if i == 20:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        env.step(zero)
        obs = env.observation_dict()
        finite &= all(bool(torch.isfinite(v).all()) for v in obs.values())
        valid = env._hand_valid
        ratio.append((env.ts.grasp_force_c[valid][..., 1].abs() / limit[valid]).cpu().numpy())
        limits.append(limit[valid].cpu().numpy())
        pivot.append(env.ts.pivot.max().item())
        travel.append((env.slide_anchor - env.slide_init).abs().max().item())
        # plastic slip of hands loaded below 80 % of their limit (must be ~0: the creep PhysX joint friction showed)
        r_now = env.ts.grasp_force_c[..., 1].abs() / limit
        held = valid.unsqueeze(1) & (r_now < 0.8) & (env.episode_length_buf > 1).unsqueeze(1)
        creep.append((env.slide_anchor - anchor0).abs()[held].cpu().numpy())
        elastic.append((env.ts.slide_q - env.slide_anchor).abs()[valid].cpu().numpy())
        for k, v in env.term_reason_buf.items():
            terms[k] += int(v.sum().item())
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / (steps - 20) * 1e3
    # rolling-dependent swivel scrub (revision 9): the per-step CPU mirror must equal what PhysX holds (also checks
    # the (static, dynamic, viscous) layout the mirror assumes), and rolling casters must be at bearing friction
    import warp as wp
    swivel = {}
    if getattr(env, "_fric_cpu", None) is not None:
        held = wp.to_torch(env.cart.root_view.get_dof_friction_properties())
        rim = (env.cart.data.joint_vel.torch[:, env.caster_wheel_ids] * env.wheel_radius_t).abs()
        tau, scrub, bearing = env.swivel_now, env.swivel_static[:, None].expand(-1, 4), env.swivel_bearing[:, None].expand(-1, 4)
        rolling, still = rim > 0.2, rim < 0.005
        swivel = {"mirror_vs_physx_max_abs": float((held - env._fric_cpu).abs().max()),
                  "physx_swivel_static_vs_law_max_abs": float((held[:, env.caster_swivel_ids.cpu(), 0] - tau.cpu()).abs().max()),
                  "rolling_frac": float(rolling.float().mean()),
                  "rolling_tau_over_bearing_p50_p99": (torch.quantile((tau / bearing)[rolling], torch.tensor([.5, .99], device=device)).tolist()
                                                       if rolling.any() else None),
                  "still_tau_over_scrub_p50": float((tau / scrub)[still].median()) if still.any() else None}
    ratio = np.concatenate([r.ravel() for r in ratio if r.size])
    lim = np.concatenate([x.ravel() for x in limits if x.size])
    creep = np.concatenate([x.ravel() for x in creep if x.size])
    elastic = np.concatenate([x.ravel() for x in elastic if x.size])
    rec = {"observation_sizes": sizes, "critic_expected": num_critic_obs(task), "finite": bool(finite),
           "load_dependent_friction": load_friction, "step_ms": step_ms, "num_envs": num_envs, "steps": steps,
           "along_bar_p99_over_current_limit": float(np.percentile(ratio, 99)),
           "slide_limit_N_p5_p50_p95": np.percentile(lim, [5, 50, 95]).round(1).tolist(),
           "slide_travel_max_m": float(max(travel)), "pivot_max_over_break": float(max(pivot)),
           "plastic_slip_per_step_below_0.8_limit_m_p99_max": [float(np.percentile(creep, 99)), float(creep.max())],
           "elastic_deflection_m_p50_p99": [float(np.percentile(elastic, 50)), float(np.percentile(elastic, 99))],
           "terminations": terms, "grasp_kinds_pull_lift_push_pivot": env.grasp_kind_counts.cpu().tolist(),
           "capacity_mult_mean": float(env.capacity_mult.mean()), "swivel_scrub": swivel}
    rec["pass"] = bool(finite and sizes["critic"] == num_critic_obs(task) and sizes["policy"] == 605
                       and sizes["estimator"] == 462 and rec["along_bar_p99_over_current_limit"] < 1.3
                       and terms.get("blowup", 0) == 0 and float(np.percentile(creep, 99)) < 1e-5
                       and (not swivel or (swivel["mirror_vs_physx_max_abs"] < 1e-6
                                           and swivel["physx_swivel_static_vs_law_max_abs"] < 1e-5)))
    print("[physx env]", level, json.dumps(rec), flush=True)
    env.sim.clear_instance()
    return {"pass": rec["pass"], f"level_{level}": rec}


def test_curriculum(device: str, num_envs: int, steps: int, level: int) -> dict:
    """Continuous per-environment curriculum: episode outcomes move each environment's difficulty, and the
    per-episode physics / commands of each environment follow its own difficulty."""
    import torch
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v7.baseline import apply_config, homie_baseline
    from homie_pushcart_v7.task_config import PushCartTaskCfg
    from homie_pushcart_v7.task_env import PushCartEnv
    from homie_pushcart_v7.curriculum import LEVELS

    cfg = G1RoughCfg()
    apply_config(cfg, homie_baseline())
    cfg.env.num_envs = num_envs
    task = PushCartTaskCfg()
    task.curriculum.init_level = level
    task.env.episode_length_s = 5.0                       # short episodes: several difficulty updates per env
    env = PushCartEnv(cfg, task, ROOT, device, True, level=level)
    env.reset()
    d0 = env.cur.d.clone()
    zero = torch.zeros(num_envs, 18, device=device)
    checks = {"payload_within_own_max": True, "critic_slot_is_difficulty": True,
              "commands_within_own_range": True}
    for i in range(steps):
        env.step(zero)
        if i % 50 == 0:
            ids = torch.arange(num_envs, device=device)
            pay = env.cart_mass_total - env.cart_base_kg
            # an env's payload was drawn at its last reset; its difficulty may have changed since (only upwards by
            # step_up per success or downwards), so compare with the max over a +-1 step band
            band = task.curriculum_v7.step_down
            dd = env.cur.d.clone()
            env.cur.d = (dd + band).clamp(0, 1); hi = env.cur.payload_max(ids)
            env.cur.d = dd
            checks["payload_within_own_max"] &= bool((pay <= hi + 1e-4).all())
            critic_level = env.hl_critic_obs[:, 5 * 124 + 3 + 12 + 2 + 3 + 8]
            checks["critic_slot_is_difficulty"] &= bool(torch.allclose(critic_level, env.level_tensor / LEVELS, atol=1e-5)
                                                       or (critic_level - env.cur.d).abs().max() < task.curriculum_v7.step_down + 1e-5)
            vlo, vhi, vy, wz = env.cur.commands(ids)
            env.cur.d = (dd + band).clamp(0, 1); vlo2, vhi2, vy2, wz2 = env.cur.commands(ids); env.cur.d = dd
            c = env.cmd_task
            checks["commands_within_own_range"] &= bool(((c[:, 0] <= vhi2 + 1e-5) & (c[:, 0] >= vlo2 - 1e-5)
                                                         & (c[:, 1].abs() <= vy2 + 1e-5) & (c[:, 2].abs() <= wz2 + 1e-5)).all())
    d1 = env.cur.d
    moved = (d1 - d0)
    st = env.cur.stats()
    rec = {"init_difficulty": level / LEVELS, "after": st, "up_frac": float((moved > 1e-6).float().mean()),
           "down_frac": float((moved < -1e-6).float().mean()), "global_level_params": env.level_params.as_dict(),
           "checks": checks}
    rec["pass"] = bool(all(checks.values()) and rec["up_frac"] + rec["down_frac"] > 0.2 and st["p90"] > st["p10"])
    print("[physx curriculum]", json.dumps(rec), flush=True)
    env.sim.clear_instance()
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("geometry", "clutch", "env", "curriculum"), required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--constant_slide_friction", action="store_true", help="env: disable load-dependent friction")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    launcher = AppLauncher(args)
    code = 1
    try:
        result = {"geometry": test_geometry, "clutch": lambda: test_clutch(args.device),
                  "env": lambda: test_env(args.device, args.num_envs, args.steps, args.level,
                                          not args.constant_slide_friction),
                  "curriculum": lambda: test_curriculum(args.device, args.num_envs, args.steps, args.level)}[args.only]()
        OUT.mkdir(parents=True, exist_ok=True)
        tag = (f"_level{args.level}_n{args.num_envs}" + ("_const" if args.constant_slide_friction else "")) if args.only == "env" else ""
        (OUT / f"report_{args.only}{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("V7 PHYSX VALIDATION", args.only, "PASS" if result["pass"] else "FAIL", flush=True)
        code = 0 if result["pass"] else 1
    except BaseException:
        import traceback
        traceback.print_exc()        # Kit's close() ends the process: print before it swallows the exception
        sys.stdout.flush()
        raise
    finally:
        launcher.app.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
