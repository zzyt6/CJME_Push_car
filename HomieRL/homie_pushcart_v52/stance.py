"""Measure HOMIE's real zero-command standing pose (Isaac Lab, robot only) and bake it into the preset pose.

The trajectory JSON gave the arms; the legs of ``initial_pose.yaml`` were HOMIE's *default* joint angles
(-0.1 / 0.3 / -0.2), which is not where the policy actually stands: it settles ~2.5 cm lower with more
knee bend. With the hands constrained to the cart handle, that height mismatch appeared as a permanent
~43 N/hand preload. This tool records the policy's standing leg angles and pelvis height and rewrites the
preset (``--apply``), after which ``cart build``, ``pack`` and ``mj build`` must be re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .paths import CONFIG_DIR, HOMIE_RUN_DIR, LEG_JOINTS, ROOT

STANCE_YAML = CONFIG_DIR / "homie_stance.yaml"


def run_stance(args, launcher, root: Path = ROOT) -> dict:
    from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from .sim_env import PushCartSimEnv
    from .teleop import play_config

    device = args.device
    headless = bool(getattr(launcher, "_headless", True))
    checkpoint = Path(args.checkpoint) if args.checkpoint else frozen_checkpoint()
    cfg = play_config(num_envs=args.num_envs, episode_length_s=60.0)
    env = PushCartSimEnv(cfg, root, device, headless, with_cart=False)
    policy = HimLowLevelPolicy(checkpoint, device)

    settle = int(round(args.settle_s / env.dt))
    average = int(round(args.average_s / env.dt))
    obs, _ = env.reset()
    env.set_command(0., 0., 0.)
    legs, heights, pitches, grav, feet, upper = [], [], [], [], [], []
    falls = 0
    for step in range(settle + average):
        if not launcher.app.is_running():
            raise RuntimeError("Simulation closed before the stance measurement completed")
        obs, _, _, _, info, _, _ = env.step(policy(obs))
        falls += int(info["terminated"].sum().item())
        if step >= settle:
            legs.append(env.dof_pos[:, :12].cpu().numpy().copy())
            heights.append(env.root_states[:, 2].cpu().numpy().copy())
            pitches.append(env.pitch.cpu().numpy().copy())
            grav.append(env.projected_gravity.cpu().numpy().copy())
            feet.append(env.feet_pos[:, :, 2].cpu().numpy().copy())
            upper.append(env.dof_pos[:, 12:].cpu().numpy().copy())
    if falls:
        raise RuntimeError(f"{falls} falls during the stance measurement; the checkpoint does not stand")
    legs = np.concatenate(legs)              # (steps*envs, 12)
    heights = np.concatenate(heights)
    pitches = np.concatenate(pitches)
    grav = np.concatenate(grav)
    feet = np.concatenate(feet)
    upper = np.concatenate(upper)            # (steps*envs, 15): gravity-loaded PD equilibrium of the held arms
    preset_upper = env.preset_dof_pos[0, 12:].cpu().numpy()
    # idempotent: the run already used the current hold offset, so the new offset adds the remaining sag to it
    current_offset = env.upper_hold_offset[0].cpu().numpy()
    stance = {
        "checkpoint": str(checkpoint),
        "iteration": policy.iteration,
        "num_envs": args.num_envs, "settle_s": args.settle_s, "average_s": args.average_s,
        "root_height_mean": float(heights.mean()), "root_height_std": float(heights.std()),
        "pelvis_pitch_mean": float(pitches.mean()), "pelvis_pitch_std": float(pitches.std()),
        "projected_gravity_mean": grav.mean(0).round(5).tolist(),
        # ankle_roll link height while standing: calibrates the FK "lowest vertex on z=0" pelvis height
        # against where PhysX actually rests the foot (contact offset / sole penetration)
        "feet_link_z_stance": float(np.median(feet.min(axis=1))),   # lower (stance) foot, median: ignores shuffles
        "feet_link_z_mean": float(feet.mean()), "feet_link_z_std": float(feet.std()),
        "leg_joint_mean": {n: float(v) for n, v in zip(env.dof_names[:12], legs.mean(0))},
        "leg_joint_std": {n: float(v) for n, v in zip(env.dof_names[:12], legs.std(0))},
        # arm sag under gravity: preset (= PD target) minus where the joints actually rest without the cart
        "upper_joint_mean": {n: float(v) for n, v in zip(env.dof_names[12:], upper.mean(0))},
        "upper_joint_std": {n: float(v) for n, v in zip(env.dof_names[12:], upper.std(0))},
        "upper_sag_rad": {n: float(o + p - v) for n, o, p, v in zip(env.dof_names[12:], current_offset, preset_upper, upper.mean(0))},
        "upper_hold_offset_before": {n: float(o) for n, o in zip(env.dof_names[12:], current_offset)},
        "preset_root_height_before": float(env.pose["robot"]["base_pos"][2]),
    }
    STANCE_YAML.write_text(yaml.safe_dump(stance, sort_keys=False), encoding="utf-8")
    print("[stance] " + json.dumps({k: v for k, v in stance.items() if k not in ("leg_joint_mean", "leg_joint_std")}), flush=True)
    for n in LEG_JOINTS:
        print(f"[stance]   {n:26s} {stance['leg_joint_mean'][n]:+.4f} ± {stance['leg_joint_std'][n]:.4f}", flush=True)
    sag = stance["upper_sag_rad"]
    print("[stance] upper-body sag (preset - rest) [rad]: " +
          ", ".join(f"{n.replace('_joint', '')}={sag[n]:+.4f}" for n in sag if abs(sag[n]) > 1e-3), flush=True)
    print(f"[stance] wrote {STANCE_YAML}", flush=True)
    if getattr(args, "apply", False):
        from .assets.cart_urdf import build_cart
        from .pose import apply_stance
        pose = apply_stance(stance, bake_pitch=not getattr(args, "upright_base", False))
        report = build_cart()
        print(f"[stance] initial_pose.yaml updated: pelvis z {stance['preset_root_height_before']:.4f} -> "
              f"{pose['robot']['base_pos'][2]:.4f} m (feet float {pose['stance']['feet_float_at_reset_m'] * 1000:.1f} mm); "
              f"ankle leveling {pose['stance']['ankle_leveling_delta_rad']}; "
              f"upper hold offset max {max(abs(v) for v in pose['robot']['upper_hold_offset'].values()):.4f} rad")
        print(f"[stance] cart.urdf rebuilt; handle height now {report['derived']['handle_height']:.4f} m")
        print("[stance] next: `pack --viz none` then `mj build`", flush=True)
    return stance
