"""Headless smoke test of ``PushCartEnv`` (phase B2): reset invariants, random / zero high-level actions,
per-term reward statistics and termination-reason accounting."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .paths import ASSET_REPORT, ROOT


def make_configs(num_envs: int, randomization: bool, level: int | None = None, episode_length_s: float | None = None,
                 checkpoint=None):
    from homie_isaaclab.config import G1RoughCfg
    from .task_config import task_cfg_from_run, run_config_path, _apply_lenient
    cfg = G1RoughCfg()
    task_cfg = task_cfg_from_run(checkpoint)
    source = run_config_path(checkpoint) if checkpoint else None
    if source:
        import yaml
        saved = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
        _apply_lenient(cfg, saved.get("homie", {}), [], {"ignored": []})
        if "decimation" in saved:
            cfg.control.decimation = int(saved["decimation"])
        for key in ("dt", "gravity"):
            if key in saved.get("sim", {}):
                setattr(cfg.sim, key, saved["sim"][key])
        if "episode_length_s" in saved:
            task_cfg.env.episode_length_s = float(saved["episode_length_s"])
        if saved.get("curriculum_level") is not None:
            task_cfg.curriculum.init_level = int(saved["curriculum_level"])
        cfg.pushcart_saved_checkpoint = saved.get("checkpoint")
        metadata = source.parent.parent / "pushcart_task.json"
        if not cfg.pushcart_saved_checkpoint and metadata.is_file():
            cfg.pushcart_saved_checkpoint = json.loads(metadata.read_text(encoding="utf-8")).get("homie_checkpoint")
    cfg.env.num_envs = num_envs
    if not randomization:
        for key in dir(cfg.domain_rand):
            if key.startswith("randomize_") or key in ("use_random", "push_robots", "delay"):
                setattr(cfg.domain_rand, key, False)
        cfg.noise.add_noise = False
    if level is not None:
        task_cfg.curriculum.init_level = int(level)
    if episode_length_s is not None:
        task_cfg.env.episode_length_s = float(episode_length_s)
    return cfg, task_cfg


def reset_invariants(env, report: dict) -> dict:
    """Checks that every reset leaves a level cart with all wheels on the ground and the grasp satisfied.

    Only the environments that were just reset (episode length 0) are inspected; the others are mid-episode.
    """
    from .smoke import grasp_errors
    ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
        return {"pass": True, "n": 0}
    cart_pos, cart_quat, _, _ = env.cart_state()
    tilt = torch.asin(torch.clamp(quat_gravity_xy(cart_quat[ids]), -1., 1.))
    wheel_ids = [i for i, n in enumerate(env.cart.body_names) if n.startswith("wheel_")]
    wheel_bottom = env.cart.data.body_link_pose_w.torch[ids][:, wheel_ids, 2] - env.cart_params["wheels"]["radius"]
    errs = grasp_errors(env, env.pose, report, ids)
    bar_b = env.ts.bar_b[ids]
    out = {
        "n": int(len(ids)),
        "cart_tilt_max_deg": float(math.degrees(tilt.abs().max().item())),
        "wheel_bottom_abs_max_mm": float(wheel_bottom.abs().max().item() * 1000),
        "grasp_error_max_mm": float(max(errs.values()) * 1000) if errs else None,
        "root_height_range": [float(env.root_states[ids, 2].min()), float(env.root_states[ids, 2].max())],
        "preset_root_height": float(env.pose["robot"]["base_pos"][2]),
        "robot_yaw_std_rad": float(env.yaw[ids].std().item()) if len(ids) > 1 else 0.0,
        "bar_in_base_std_m": float(bar_b.std(dim=0).max().item()) if len(ids) > 1 else 0.0,
        "bar_in_base_mean": bar_b.mean(0).tolist(),
        "rel_yaw_abs_max_rad": float(env.ts.rel_yaw[ids].abs().max().item()),
        "caster_angle_abs_max_rad": float(env.ts.caster_angles[ids].abs().max().item()),
        "cart_mass_range_kg": [float(env.cart_mass_total[ids].min()), float(env.cart_mass_total[ids].max())],
    }
    out["pass"] = (out["cart_tilt_max_deg"] < 0.5 and out["wheel_bottom_abs_max_mm"] < 1.0
                   and (out["grasp_error_max_mm"] is None or out["grasp_error_max_mm"] < 1.0)
                   and out["bar_in_base_std_m"] < 1e-3 and out["rel_yaw_abs_max_rad"] < 1e-3)
    return out


def quat_gravity_xy(q_xyzw: torch.Tensor) -> torch.Tensor:
    from homie_isaaclab.math_utils import quat_rotate_inverse
    g = torch.tensor([0., 0., -1.], device=q_xyzw.device).expand(q_xyzw.shape[0], -1)
    return quat_rotate_inverse(q_xyzw, g)[:, :2].norm(dim=-1)


def run_env_smoke(args, launcher, root: Path = ROOT) -> dict:
    from .task_env import PushCartEnv, TERMINATION_REASONS

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    headless = bool(getattr(launcher, "_headless", True))
    cfg, task_cfg = make_configs(args.num_envs, not args.no_randomization, args.level, args.episode_s,
                                checkpoint=getattr(args, "hl_checkpoint", None))
    if getattr(args, "quality_profile", None):
        from .cart_quality import apply_quality_profile
        apply_quality_profile(task_cfg, args.quality_profile)
    # diagnostic knobs (attribution of numerical blow-ups): each one restores a pre-fix behaviour
    cart_params = None
    if getattr(args, "cart_iters", None):
        from .assets.cart_urdf import load_cart_params
        cart_params = load_cart_params()
        cart_params["solver"] = {"position_iterations": int(args.cart_iters[0]), "velocity_iterations": int(args.cart_iters[1])}
    if getattr(args, "legacy_push", False):
        task_cfg.randomization.robot_push_duration_s = 0.0
    if getattr(args, "no_force_clip", False):
        task_cfg.rewards.force_clip_n = 1.0e9
    if getattr(args, "no_stall", False):
        task_cfg.randomization.wheel_stall_prob_per_s = 0.0
    if getattr(args, "no_kick", False):
        task_cfg.randomization.cart_push_vel_max = 0.0
        task_cfg.randomization.cart_force_pulse = False
    if getattr(args, "no_robot_push", False):
        task_cfg.randomization.robot_push = False
    if getattr(args, "no_idle_correction", False):
        task_cfg.idle_correction.enabled = False
    lowlevel_checkpoint = args.checkpoint or getattr(cfg, "pushcart_saved_checkpoint", None)
    env = PushCartEnv(cfg, task_cfg, root, device, headless,
                      checkpoint=Path(lowlevel_checkpoint) if lowlevel_checkpoint else None, cart_params=cart_params,
                      allow_robot_cart_contact=bool(getattr(args, "allow_contact", False)))
    report = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
    if getattr(args, "fixed_command", None) is not None:
        env.fixed_command = list(args.fixed_command)
    if getattr(args, "payload", None) is not None:
        if not math.isfinite(args.payload) or args.payload < 0:
            raise ValueError("payload must be finite and nonnegative")
        env.fixed_payload_kg = float(args.payload)

    obs, critic = env.reset()
    initial_cart_xy = env.cart_state()[0][:, :2].clone()
    assert obs.shape == (args.num_envs, env.num_hl_obs), obs.shape
    assert critic.shape == (args.num_envs, env.num_hl_critic_obs), critic.shape
    invariants = [reset_invariants(env, report)]
    print("[env-smoke] reset:", json.dumps(invariants[0]), flush=True)

    # per-term reward accounting through HOMIE's reward machinery
    term_sums = {name: 0.0 for name in env.reward_names}
    wrapped = []
    for name, fn in zip(env.reward_names, env.reward_functions):
        def make(fn=fn, name=name):
            def call():
                r = fn()
                term_sums[name] += float((r * env.reward_scales[name]).mean().item())
                return r
            return call
        wrapped.append(make())
    env.reward_functions = wrapped

    gen = torch.Generator(device=device).manual_seed(args.seed)
    hl = None
    if getattr(args, "hl_checkpoint", None):
        from .hl_policy import HighLevelPolicy
        hl = HighLevelPolicy(Path(args.hl_checkpoint), env.num_hl_obs, env.num_hl_critic_obs, env.num_hl_actions, device)
        print(f"[env-smoke] driving with {args.hl_checkpoint} (iter {hl.iteration}), exploration std {args.hl_noise_std}", flush=True)
    rew_hist, grasp_hist, push_hist, track_err, cart_speed, n_resets = [], [], [], [], [], 0
    min_dist, max_force, global_max_force, big_events = 1e9, 0.0, 0.0, 0
    # Disturbance bookkeeping: wheel stalls, physical cart pulses and robot pushes.
    stall_len = max(2, int(task_cfg.randomization.wheel_stall_duration_s / env.dt))
    n_stalls, n_kicks, n_pushes = 0, 0, 0
    kick_timer_prev = env.cart_push_timer.clone()
    last_push_prev = env.last_push_step
    start = time.perf_counter()
    quality_costs = []
    idle_gate_max, idle_active_samples = 0., 0
    command_mode_counts = [0, 0, 0]
    sequence_phase_max = 0
    max_cart_displacement = 0.
    for step in range(args.steps):
        if not launcher.app.is_running():
            raise RuntimeError("Simulation closed before the env smoke completed")
        if hl is not None:
            actions = hl(obs, critic)
            if args.hl_noise_std > 0:
                actions = actions + args.hl_noise_std * torch.randn(actions.shape, device=device, generator=gen)
        elif args.policy == "zero":
            actions = torch.zeros(args.num_envs, env.num_hl_actions, device=device)
        elif args.policy == "random":
            actions = (args.action_std * torch.randn(args.num_envs, env.num_hl_actions, device=device, generator=gen)).clamp(-1, 1)
        else:   # sine: slow coordinated sweep, exercises the slew limiter and joint clamps
            t = step * env.dt
            actions = torch.full((args.num_envs, env.num_hl_actions), math.sin(2 * math.pi * t / 4.0), device=device)
        obs, critic, rew, done, extras = env.step(actions)
        if not done.any():
            max_cart_displacement = max(max_cart_displacement,
                float((env.cart_state()[0][:, :2] - initial_cart_xy).norm(dim=1).max()))
        if env.idle_correction is not None:
            tracker = env.idle_correction
            for value in (tracker.estimated_twist, tracker.gate_level, env.idle_gate_applied):
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"idle correction has non-finite values at step {step}")
            idle_gate_max = max(idle_gate_max, float(env.idle_gate_applied.max()))
            idle_active_samples += int((env.idle_gate_applied > 1e-4).sum())
            assert idle_gate_max <= task_cfg.idle_correction.max_gate + 1e-6
        if env.command_schedule is not None:
            for mode in range(3):
                command_mode_counts[mode] += int((env.command_schedule.mode == mode).sum())
            sequence_phase_max = max(sequence_phase_max, int(env.command_schedule.phase.max()))
        for name, tensor in (("obs", obs), ("critic", critic), ("rew", rew)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} has non-finite values at step {step}")
        if env.cart_quality is not None:
            q = env.cart_quality
            for value in (q.velocity_band, q.wrench_band, q.ripple_cost, q.wrench_cost,
                          q.lin_ripple_cost, q.yaw_ripple_cost, q.response_error, q.response_cost,
                          env.ts.cart_force_w, env.ts.cart_moment_w, env.ts.cart_com_w):
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"cart quality has non-finite values at step {step}")
            quality_costs.append([float(cost.mean()) for cost in
                                  (q.ripple_cost, q.lin_ripple_cost, q.yaw_ripple_cost, q.response_cost, q.wrench_cost)])
        fast_cart = float(env.ts.cart_speed.max()) > 1.5
        if getattr(args, "force_trace", 0) and (float(env.ts.force_norm_raw.max()) > args.force_trace or fast_cart):
            worst = int(env.ts.force_norm_raw.max(dim=1).values.argmax()) if not fast_cart else int(env.ts.cart_speed.argmax())
            q = env.dof_pos[worst, 12:]
            wheel_w = env.cart.data.joint_vel.torch[worst, env.wheel_joint_ids]
            print(f"[env-smoke]   env {worst} cart lin_w {env.cart_state()[2][worst].tolist()} ang_w {env.cart_state()[3][worst].tolist()} "
                  f"wheel_w {[round(v, 1) for v in wheel_w.tolist()]} stall_timer {int(env.stall_timer[worst])} stall_wheel {int(env.stall_wheel[worst])} "
                  f"cart_push_timer {int(env.cart_push_timer[worst])} casters {[round(v, 2) for v in env.ts.caster_angles[worst].tolist()]} "
                  f"cmd_homie {[round(v, 2) for v in env.cmd_homie[worst].tolist()]} ep_len {int(env.episode_length_buf[worst])}", flush=True)
            at_limit = int(((q - env.upper_lo[0]).abs() < 0.02).sum() + ((env.upper_hi[0] - q).abs() < 0.02).sum())
            print(f"[env-smoke] step {step} env {worst}: raw |F| L/R = {[round(v) for v in env.ts.force_norm_raw[worst].tolist()]} N, "
                  f"since_push {env.common_step_counter - env.last_push_step} steps, foot-caster {float(env.ts.feet_caster_dist[worst].min()):.3f} m, "
                  f"upper joints at limit {at_limit}, dist {float(env.ts.dist[worst]):.2f} tilt {float(env.ts.cart_tilt[worst]):.3f} "
                  f"wheel_lift {float(env.ts.wheel_lift[worst].max()):.3f} base_v {env.base_lin_vel[worst, :2].norm().item():.2f} "
                  f"cart_v {float(env.ts.cart_speed[worst]):.2f} feet_contact {[round(v) for v in env.contact_forces[worst, env.feet_indices].norm(dim=-1).tolist()]} "
                  f"payload {float(env.cart_mass_total[worst]):.1f} kg", flush=True)
        n_stalls += int((env.stall_timer == stall_len).sum().item())          # freshly started this step
        # A reset also rearms this timer; count only surviving rows with enabled pulses.
        if task_cfg.randomization.cart_force_pulse:
            n_kicks += int(((env.cart_push_timer > kick_timer_prev) & ~done.bool()).sum().item())
        kick_timer_prev.copy_(env.cart_push_timer)
        n_pushes += int(env.last_push_step != last_push_prev)                  # HOMIE push schedule fired (all envs)
        last_push_prev = env.last_push_step
        rew_hist.append(float(rew.mean().item()))
        grasp_hist.append(env.ts.force_norm.mean(0).tolist())
        push_hist.append(env.ts.force_c.mean(0).flatten().tolist())      # cart-frame [Lx, Ly, Lz, Rx, Ry, Rz]
        track_err.append(float((env.cmd_task[:, 0] - env.ts.cart_v[:, 0]).abs().mean().item()))
        cart_speed.append(float(env.ts.cart_speed.mean().item()))
        if done.any():
            n_resets += int(done.sum().item())
            invariants.append(reset_invariants(env, report))
        min_dist = min(min_dist, float(env.ts.feet_caster_dist.min()))
        step_max_force = float(env.ts.force_norm_raw.max())
        max_force = max(max_force, step_max_force)
        global_max_force = max(global_max_force, step_max_force)
        big_events += int((env.ts.force_norm_raw.max(dim=1).values > 1000.0).sum().item())
        if (step + 1) % 100 == 0:
            print(f"[env-smoke] step {step + 1}: reward {np.mean(rew_hist[-100:]):+.4f}/step, grasp F "
                  f"{np.mean(grasp_hist[-100:], axis=0).round(1).tolist()} N, |vx err| {np.mean(track_err[-100:]):.3f}, "
                  f"cart v {np.mean(cart_speed[-100:]):.3f}, resets {n_resets}, level {env.curriculum.level}, "
                  f"foot-caster min {min_dist:.3f} m, raw |F| max {max_force:.0f} N", flush=True)
            min_dist, max_force = 1e9, 0.0
    elapsed = time.perf_counter() - start

    # the timeout path and the terminal contract
    env.episode_length_buf[:] = env.max_episode_length - 1
    obs, critic, rew, done, extras = env.step(torch.zeros(args.num_envs, env.num_hl_actions, device=device))
    assert done.all() and extras["time_outs"].all(), "timeout should reset every env"
    if env.cart_quality is not None:
        assert not env.cart_quality.initialized.any(), "quality history must reset with each environment"
    if env.idle_correction is not None:
        assert not env.idle_correction.ready.any(), "idle estimate must warm up again after reset"
        assert not env.idle_correction.gate_level.any(), "idle gate must close on reset"
    invariants.append(reset_invariants(env, report))

    summary = env.termination_summary()
    result = {
        "result": "PASS" if all(i["pass"] for i in invariants) else "RESET_INVARIANT_FAILED",
        "version": 5, "dynamics": getattr(env, "dynamics_metadata", None),
        "policy": args.policy, "num_envs": args.num_envs, "steps": args.steps, "level": env.curriculum.level,
        "motion": {"command_sequence_enabled": env.command_schedule is not None,
                      "command_mode_sample_counts": dict(zip(("random", "stop_go", "turn_reversal"), command_mode_counts)),
                      "sequence_phase_max": sequence_phase_max,
                      "idle_correction_enabled": env.idle_correction is not None,
                      "idle_gate_max": idle_gate_max, "idle_gate_active_samples": idle_active_samples,
                      "max_cart_displacement_m": max_cart_displacement if n_resets == 0 else None,
                      "fixed_command": getattr(args, "fixed_command", None),
                      "fixed_payload_kg": getattr(args, "payload", None)},
        "randomization": not args.no_randomization,
        "cart_quality": ({**env.cart_quality.metadata(), "profile": getattr(args, "quality_profile", None),
                          "cost_names": ["cart_ripple", "cart_lin_ripple", "cart_yaw_ripple",
                                         "cart_response_error", "cart_wrench_osc"],
                          "cost_means_unweighted": np.mean(quality_costs, axis=0).tolist(),
                          "wrench_source": "physx_incoming_joint_hand_subtree_newton_euler_proxy"}
                         if env.cart_quality is not None else None),
        "obs_dims": {"actor": env.num_hl_obs, "critic": env.num_hl_critic_obs, "actions": env.num_hl_actions},
        "reward_per_step_mean": float(np.mean(rew_hist)),
        "reward_terms_per_step": {k: v / args.steps for k, v in term_sums.items()},
        "grasp_force_mean_N": np.mean(grasp_hist, axis=0).round(2).tolist(),
        "grasp_force_max_N": np.max(grasp_hist, axis=0).round(2).tolist(),
        "grasp_force_cart_frame_mean_N": np.mean(push_hist, axis=0).round(2).tolist(),
        "vx_tracking_abs_err_mean": float(np.mean(track_err)),
        "cart_speed_mean": float(np.mean(cart_speed)),
        "resets": n_resets, "termination": summary,
        "raw_grasp_force_max_N": global_max_force, "env_steps_with_force_gt_1kN": big_events,
        "env_steps_total": args.steps * args.num_envs,
        "disturbance_events": {"wheel_stalls": n_stalls, "cart_force_pulses": n_kicks, "robot_push_rounds": n_pushes},
        "diagnostic_knobs": {"allow_contact": bool(getattr(args, "allow_contact", False)), "cart_iters": getattr(args, "cart_iters", None),
                             "legacy_push": bool(getattr(args, "legacy_push", False)), "no_force_clip": bool(getattr(args, "no_force_clip", False)),
                             "no_stall": bool(getattr(args, "no_stall", False)), "no_kick": bool(getattr(args, "no_kick", False)),
                             "no_robot_push": bool(getattr(args, "no_robot_push", False))},
        "reset_invariants_all_pass": all(i["pass"] for i in invariants), "reset_checks": len(invariants),
        "worst_reset": {k: max(i.get(k, 0.0) for i in invariants) for k in ("cart_tilt_max_deg", "wheel_bottom_abs_max_mm", "bar_in_base_std_m", "rel_yaw_abs_max_rad", "grasp_error_max_mm")},
        "realtime_factor": args.steps * env.dt / elapsed,
        "checkpoint": str(env.policy.checkpoint),
        "level_params": env.level_params.as_dict(),
    }
    path = root / "logs/pushcart_v5_env_smoke.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[env-smoke] " + json.dumps({k: v for k, v in result.items() if k not in ("level_params",)}), flush=True)
    return result
