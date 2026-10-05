"""Fresh-reset MuJoCo zero-command control diagnostic; never paper step data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/Picture/motion_ability/diagnostics/mujoco_zero20")
    parser.add_argument("--ablation", choices=["none", "robot_passive_damping0", "dry_friction0", "cart_armature0", "all_three"], default="none")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    if args.ablation != "none" and args.output == ROOT / "logs/Picture/motion_ability/diagnostics/mujoco_zero20":
        parser.error("an ablation requires a distinct --output path")
    import mujoco
    import numpy as np
    import torch
    from homie_isaaclab.math_utils import class_to_dict
    from pushcart_bench.policies_v6 import V6Policy

    paper = json.loads((ROOT / "logs/Picture/motion_ability/manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(paper["checkpoint"])
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    policy = V6Policy(checkpoint, level=17)
    policy.reset(payload_kg=float(paper["payload_kg"]), caster_angle=0., seed=int(paper["seed"]))
    runner, env = policy.runner, policy.env
    m, d = env.model, env.data
    original_passive = {key: getattr(m, key).copy() for key in ["dof_damping", "dof_frictionloss", "dof_armature"]}
    if args.ablation in ["robot_passive_damping0", "all_three"]:
        m.dof_damping[env.qvel_adr] = 0.
    if args.ablation in ["dry_friction0", "all_three"]:
        m.dof_frictionloss[:] = 0.
    if args.ablation in ["cart_armature0", "all_three"]:
        for i in range(4):
            for name in [f"caster_{i}_swivel_joint", f"wheel_{i}_joint"]:
                m.dof_armature[int(m.joint(name).dofadr[0])] = 0.
    if args.ablation != "none":
        mujoco.mj_setConst(m, mujoco.MjData(m))
        mujoco.mj_forward(m, d)
    changes = {}
    for key, original in original_passive.items():
        changed = np.flatnonzero(getattr(m, key) != original)
        changes[key] = dict(dof_indices=changed.tolist(), before=original[changed].tolist(),
                            after=getattr(m, key)[changed].tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    initial = {
        "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy(),
        "homie_history": env.obs_hist.copy(), "actor_history": runner.actor_hist.copy(),
        "estimator_history": runner.estimator_obs.numpy().copy(),
        "command": runner.cmd_task.copy(), "reference": runner.cmd_ref.copy(),
        "homie_command": runner.cmd_homie.copy(), "pose": np.asarray(policy.pose()),
        "body_mass": m.body_mass.copy(), "body_inertia": m.body_inertia.copy(),
        "geom_friction": m.geom_friction.copy(), "dof_damping": m.dof_damping.copy(),
        "dof_armature": m.dof_armature.copy(), "dof_frictionloss": m.dof_frictionloss.copy(),
        "p_gains": env.p_gains.copy(), "d_gains": env.d_gains.copy(),
    }
    np.savez_compressed(args.output / "initial_state.npz", **initial)
    assert np.count_nonzero(d.qvel) == 0
    assert not np.any(runner.cmd_ref) and not np.any(runner.cmd_homie)
    samples = {key: [] for key in ["t", "command", "reference", "homie_command", "cart_velocity", "pose",
                                  "robot_pose", "high_action", "upper_delta_q", "lower_action", "joint_pos",
                                  "joint_vel", "cart_force_w", "cart_moment_w"]}
    wall = time.perf_counter()
    steps = round(args.duration / policy.dt)
    for k in range(steps):
        with torch.inference_mode():
            policy.step(np.zeros(3))
        values = dict(t=(k + 1) * policy.dt, command=runner.cmd_task, reference=runner.cmd_ref,
                      homie_command=runner.cmd_homie, cart_velocity=policy.velocity(), pose=policy.pose(),
                      robot_pose=np.r_[d.xpos[env.pelvis_id], d.xquat[env.pelvis_id]],
                      high_action=runner.hl_actions, upper_delta_q=runner.upper_delta_q,
                      lower_action=env.actions, joint_pos=env.dof_pos, joint_vel=env.dof_vel,
                      cart_force_w=runner.ts["cart_force_w"], cart_moment_w=runner.ts["cart_moment_w"])
        for key, value in values.items():
            samples[key].append(np.asarray(value).copy())
        if policy.failure():
            break
    data = {key: np.asarray(value) for key, value in samples.items()}
    assert all(np.isfinite(value).all() for value in data.values())
    np.savez_compressed(args.output / "trace.npz", **data)
    np.savetxt(args.output / "trace.csv", np.column_stack([data["t"], data["cart_velocity"], data["pose"]]),
               delimiter=",", header="t_s,cart_vx_m_s,cart_vy_m_s,cart_wz_rad_s,cart_x_m,cart_y_m,cart_yaw_rad", comments="")
    metrics = {}
    for start, end in [(0., 4.), (4., 20.), (10., 20.)]:
        selected = (data["t"] > start) & (data["t"] <= end)
        if not np.any(selected):
            continue
        v = data["cart_velocity"][selected]
        poses = data["pose"][selected]
        metrics[f"{start:g}_{end:g}s"] = dict(mean=v.mean(0).tolist(), rms=np.sqrt((v ** 2).mean(0)).tolist(),
                                                  std=v.std(0).tolist(), minimum=v.min(0).tolist(), maximum=v.max(0).tolist(),
                                                  yaw_peak_to_peak_deg=float(np.rad2deg(np.ptp(np.unwrap(poses[:, 2])))),
                                                  position_range_m=np.ptp(poses[:, :2], axis=0).tolist())
    report = dict(simulator="MuJoCo", mujoco_version=mujoco.__version__, device="cpu", checkpoint=str(checkpoint),
                  checkpoint_sha256=checkpoint_sha256, iteration=paper["iteration"], seed=paper["seed"],
                  method="CWC", duration_s=args.duration, samples=len(data["t"]), dt_s=policy.dt,
                  payload_kg=paper["payload_kg"], caster_angle_rad=0., command=[0., 0., 0.],
                  reset="fresh process -> policy.reset -> mj_resetData -> env.reset -> reset_buffers -> fresh history frame",
                  warmup_s=0., automatic_resets=0, randomization="not implemented in official MuJoCo runner; observation noise/delay off",
                  failure=policy.failure(), policy=policy.describe(), task_cfg=class_to_dict(runner.task_cfg),
                  nominal_cfg=class_to_dict(env.cfg), model=env.info(), initial_physical_velocity_zero=True,
                  all_homie_commands_zero=bool(not np.any(data["homie_command"])),
                  upper_correction_max_abs=float(np.abs(data["upper_delta_q"]).max()),
                  ablation=args.ablation, live_model_changes=changes, asset_files_modified=False,
                  wall_s=time.perf_counter() - wall, metrics=metrics,
                  purpose="Independent zero-command simulator diagnostic; never replaces Isaac paper data")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha256
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["simulator", "ablation", "samples", "failure", "wall_s", "all_homie_commands_zero", "upper_correction_max_abs", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
