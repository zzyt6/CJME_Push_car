"""Why is HOMIE's HIM velocity estimate poor in MuJoCo? Scene (robot model / contact) vs estimator (2026-10-08).

Robot alone (no cart), the frozen HOMIE (or any 456->12 HIM checkpoint), the original 24 s / 6-program audit protocol
(scripts/eval_homie_body_velocity.py). The robot MJCF is varied one factor at a time against what Isaac trains with:

    base        legacy robot MJCF: 4 spheres r 5 mm per foot (g1.xml), joint frictionloss 0.1 (g1.xml default)
    footmesh    foot collider = the ankle_roll_link mesh (its convex hull), as the URDF that Isaac / PhysX uses
    nofric      joint frictionloss 0 (Isaac articulation joints carry no friction in the HOMIE config)
    footmesh_nofric   both

Reported per variant: HIM estimate error vs the pelvis COM velocity (body frame), achieved velocity vs command in the
steady phases, and stance-foot slip (horizontal speed of a foot body while it touches the floor).

    E:\\envs\\RLisaac601\\python.exe scripts/homie_mujoco_contact_audit.py [--variants base footmesh ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "scripts")]
OUT = ROOT / "logs/validation/homie_mujoco_contact"
SCENARIOS = ("straight", "left", "right", "reverse", "lateral_left", "lateral_right")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def variant_xml(variant: str) -> Path:
    from homie_pushcart.assets.mjcf_scene import MJCF_ROBOT
    tree = ET.parse(MJCF_ROBOT)
    root = tree.getroot()
    if "nofric" in variant:
        for j in root.iter("joint"):
            if j.get("frictionloss") is not None:
                j.set("frictionloss", "0")
        for d in root.iter("default"):
            for j in d.findall("joint"):
                j.set("frictionloss", "0")
    if "footmesh" in variant:
        for body in root.iter("body"):
            if body.get("name") in FEET:
                for g in list(body.findall("geom")):
                    if g.get("type") in (None, "sphere") and g.get("size") == "0.005":
                        body.remove(g)
                ET.SubElement(body, "geom", type="mesh", mesh=body.get("name"), rgba="0.2 0.2 0.2 1")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"g1_grasp_{variant}.xml"
    tree.write(path, encoding="unicode")
    return path


def run(variant: str, checkpoint: Path | None, steps: int) -> dict:
    import mujoco
    import torch
    from homie_pushcart.mujoco_env import MujocoHomieEnv
    from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from eval_homie_body_velocity import command, config_free
    torch.set_num_threads(1)
    cfg = config_free(1)
    env = MujocoHomieEnv(False, cfg=cfg, xml_path=variant_xml(variant))
    env.upper_hold_actions[:] = 0.
    env.upper_hold_offset[:] = 0.
    env.preset_dof_pos[:] = env.default_dof_pos
    env.preset[env.qpos_adr] = env.default_dof_pos
    env.preset[env.root_qpos:env.root_qpos + 3] = cfg.init_state.pos
    env.preset[env.root_qpos + 3:env.root_qpos + 7] = [1., 0., 0., 0.]
    low = HimLowLevelPolicy(Path(checkpoint) if checkpoint else frozen_checkpoint(), "cpu")
    m, d = env.model, env.data
    scale = env.obs_scales["lin_vel"]
    floor = [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0]
    foot_ids = [m.body(f).id for f in FEET]
    out = {}
    for sc in SCENARIOS:
        mujoco.mj_resetData(m, d)
        env.reset(); env.set_command(0., 0., 0.)
        filtered = np.zeros(3)
        err, track, slip, fallen = [], [], [], False
        for step in range(steps):
            t = step * env.dt
            cmd = command(t, sc)
            filtered += min(1., env.dt / .4) * (cmd - filtered)
            env.set_command(*filtered)
            env.obs_hist[-1, :3] = filtered * env.commands_scale
            with torch.inference_mode():
                act, est = low.act_and_estimate(torch.from_numpy(env.obs_hist.reshape(1, -1)))
            _, fell = env.step(act[0].numpy())
            if fell:
                fallen = True
                break
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, env.pelvis_id, v, 0)
            from homie_pushcart.mujoco_env import quat_rotate_inverse_wxyz
            truth = quat_rotate_inverse_wxyz(env.base_quat_wxyz, v[3:])
            pred = est[0].numpy() / scale
            if t >= 2.0:
                err.append(pred - truth)
            if 8.0 <= t < 12.0:                       # steady main phase (command filter settled)
                track.append(np.r_[filtered, truth[:2], env.base_ang_vel[2]])
            touching = set()
            for c in d.contact[:d.ncon]:
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if o in floor:
                        touching.add(int(m.geom_bodyid[g]))
            for fid in foot_ids:
                if fid in touching:
                    fv = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fid, fv, 0)
                    slip.append(np.hypot(fv[3], fv[4]))
        e = np.array(err) if err else np.zeros((1, 3))
        tr = np.array(track) if track else np.full((1, 6), np.nan)
        out[sc] = {"fallen": fallen, "est_mae": np.abs(e).mean(0).tolist(), "est_bias": e.mean(0).tolist(),
                   "cmd_vx_vy_wz": tr[:, :3].mean(0).tolist(), "achieved_vx_vy_wz": tr[:, 3:].mean(0).tolist(),
                   "stance_foot_speed_mean_p90": [float(np.mean(slip)) if slip else None,
                                                  float(np.percentile(slip, 90)) if slip else None]}
    all_e = np.array([out[s]["est_mae"] for s in SCENARIOS if not out[s]["fallen"]])
    out["mean_est_mae"] = all_e.mean(0).tolist() if len(all_e) else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=["base", "footmesh", "nofric", "footmesh_nofric"])
    p.add_argument("--checkpoint", default=None, help="HOMIE HIM checkpoint (default: frozen model_60000)")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    res = {}
    for v in a.variants:
        r = run(v, a.checkpoint, a.steps)
        res[v] = r
        print(f"== {v:16s} mean est MAE vx/vy/vz {np.round(r['mean_est_mae'], 3) if r['mean_est_mae'] else 'n/a'}", flush=True)
        for sc in SCENARIOS:
            s = r[sc]
            print(f"   {sc:13s} fell={s['fallen']!s:5s} MAE {np.round(s['est_mae'], 3)} bias {np.round(s['est_bias'], 3)} "
                  f"cmd {np.round(s['cmd_vx_vy_wz'], 2)} got {np.round(s['achieved_vx_vy_wz'], 2)} "
                  f"foot slip mean/p90 {np.round([x if x is not None else np.nan for x in s['stance_foot_speed_mean_p90']], 3)}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"audit{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
