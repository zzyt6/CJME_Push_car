"""Batched undisturbed Isaac acceptance, with replayable actor-visible observer inputs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "HomieRL"), str(ROOT / "HomieRL/rsl_rl")]
from homie_pushcart_v5.baseline_acceptance import TASKS, PROTOCOL, task_command, summarize_trial
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--zero-policy", action="store_true", help="Keep checkpoint's environment, evaluate zero high-level action")
parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
parser.add_argument("--payloads", nargs="+", type=float, default=[0., 12.5, 25.])
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--level", type=int, default=7)
parser.add_argument("--noisy", action="store_true", help="Enable HOMIE noise/randomization and requested level's observation noise")
parser.add_argument("--output", type=Path, default=ROOT / "logs/bench" / ("v5_baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if any(x < 0 for x in args.payloads):
    parser.error("payloads must be nonnegative")
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
launcher = AppLauncher(args)
error = None
try:
    import numpy as np
    import torch
    from homie_isaaclab.math_utils import class_to_dict
    from homie_pushcart_v5.env_smoke import make_configs
    from homie_pushcart_v5.task_env import PushCartEnv
    from homie_pushcart_v5.hl_policy import HighLevelPolicy

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cases = [(name, payload) for payload in args.payloads for name in args.tasks]
    cfg, task = make_configs(len(cases), args.noisy, args.level, 60., args.checkpoint)
    task.curriculum.enabled = False
    task.randomization.robot_push = False
    task.randomization.cart_force_pulse = False
    task.randomization.wheel_stall_prob_per_s = 0.
    task.randomization.init_xy_m = 0.
    task.randomization.init_yaw = False
    task.randomization.payload_com_offset_m = 0.
    task.randomization.wheel_friction_range = [.8, .8]
    task.randomization.wheel_damping_range = [.005, .005]
    task.randomization.swivel_damping_range = [.05, .05]
    task.randomization.grasp_threshold_jitter = 0.
    task.reaction.mass_estimate_error_fraction = 0.
    if not args.noisy:
        task.actions.delay_steps_max = 0
        for key in ("cart_pose_noise_pos_m", "cart_pose_noise_yaw_rad", "cart_pose_delay_steps_max",
                    "upper_torque_noise_frac", "est_vel_noise_m_s"):
            setattr(task.observations, key, 0)
        for key in ("sensor_force_noise_std_n", "sensor_moment_noise_std_nm", "sensor_force_bias_std_n", "sensor_moment_bias_std_nm"):
            if hasattr(task.reaction, key):
                setattr(task.reaction, key, 0.)
    env = PushCartEnv(cfg, task, ROOT, args.device, True,
                     checkpoint=getattr(cfg, "pushcart_saved_checkpoint", None))
    env.fixed_command, env.fixed_caster_angle = [0., 0., 0.], 0.
    env.reset()
    for payload in args.payloads:
        ids = torch.tensor([i for i, case in enumerate(cases) if case[1] == payload], device=args.device)
        env.fixed_payload_kg = payload
        env.reset_idx(ids)
    # Refresh observations once after batch payload resets; no truth is fed to policy.
    env._sync_states()
    env._get_observations()
    env.vel_est = env.policy.estimate_velocity(env.obs_buf) / env.obs_scales.lin_vel
    env._update_hl_observations()
    obs, critic = env.hl_obs, env.hl_critic_obs
    env.term_counts.zero_()
    env.episodes_finished = 0
    policy = (HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
              if args.checkpoint and not args.zero_policy else None)
    latest = {}
    original_update = env.reaction_observer.update
    observer_names = ("wrench_cart", "pose_base", "base_velocity_est", "imu_gyro_b", "gravity_b",
                      "command_ref", "cart_mass_est", "robot_mass_est", "valid")

    def capture_observer(*values, **kwargs):
        latest.clear()
        latest.update({name: value.detach().cpu().numpy().copy() for name, value in zip(observer_names, values)})
        return original_update(*values, **kwargs)

    env.reaction_observer.update = capture_observer
    records, physics = [], []
    failed = torch.zeros(len(cases), dtype=torch.bool, device=args.device)
    failure_time = torch.full((len(cases),), float("nan"), device=args.device)
    with torch.inference_mode():
        for step in range(round(PROTOCOL["duration_s"] / env.dt)):
            t = step * env.dt
            env.cmd_task[:] = torch.tensor([task_command(name, t) for name, _ in cases], device=args.device)
            env.cmd_timer[:] = 10**9
            action = policy(obs, critic) if policy else torch.zeros(len(cases), 18, device=args.device)
            obs, critic, _, done, extras = env.step(action)
            newly_failed = done.bool() & ~failed
            failure_time[newly_failed] = (step+1)*env.dt
            failed |= done.bool()
            record = {**latest,
                      "time_s": np.full(len(cases), (step+1)*env.dt),
                      "actual": torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1).cpu().numpy().copy(),
                      "reference": env.cmd_ref.cpu().numpy().copy(),
                      "cart_yaw": env.ts.cart_yaw.cpu().numpy().copy(),
                      "base_yaw": env.yaw.squeeze(1).cpu().numpy().copy(),
                      "cart_com": env.ts.cart_com_w.cpu().numpy().copy(),
                      "cart_momentum": torch.cat((env.ts.cart_centroidal["linear_momentum"],
                                                   env.ts.cart_centroidal["angular_momentum"]), -1).cpu().numpy().copy(),
                      "cart_mass": env.ts.cart_centroidal["mass"].cpu().numpy().copy(),
                      "cart_inertia_world": env.ts.cart_centroidal["inertia_world"].cpu().numpy().copy(),
                      "estimated_cart_twist": env.reaction_observer.estimated_cart_twist.cpu().numpy().copy(),
                      "observer_ready": env.reaction.sensor_valid[:, 0].cpu().numpy().copy(),
                      "pose_delay_steps": env.cart_pose_delay.cpu().numpy().copy(),
                      "failed": failed.cpu().numpy().copy(),
                      "sensor_wrench": env.reaction.filtered_wrench.cpu().numpy().copy(),
                      "aligned_required_wrench": env.reaction.aligned_required_wrench.cpu().numpy().copy(),
                      "hand_wrench_world": torch.cat((env.ts.cart_force_w, env.ts.cart_moment_w), -1).cpu().numpy().copy()}
            records.append(record)
            samples = {key: value.clone() for key, value in env.physics_samples.items()}
            terminal = extras.get("terminal_physics")
            if terminal is not None:
                for key in samples:
                    samples[key][:, terminal["env_ids"]] = terminal["samples"][key]
            physics.append({key: value.cpu().numpy() for key, value in samples.items()})
            if (step+1) % 100 == 0:
                print(f"[baseline] {step+1}/600 steps, terminated cases {int(failed.sum())}/{len(cases)}", flush=True)
    traces = {key: np.stack([x[key] for x in records]) for key in records[0]}
    traces.update({"physics_"+key: np.concatenate([x[key] for x in physics]) for key in physics[0]})
    results = []
    for i, (name, payload) in enumerate(cases):
        uninterrupted = ~traces["failed"][:, i]
        one = {key: traces[key][uninterrupted, i] for key in ("time_s", "actual", "reference", "cart_yaw", "cart_com")}
        results.append({"task": name, "payload_kg": payload, "failure_time_s":
                        float(failure_time[i]) if failed[i] else None,
                        **summarize_trial(one, name, bool(failed[i]))})
    report = {"protocol": PROTOCOL, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              "policy": "loaded_actor" if policy else "zero_high_level_action", "seed": args.seed,
              "noisy": args.noisy, "level": args.level, "exogenous_disturbances": False,
              "dt": env.dt, "physics_dt": cfg.sim.dt, "task_config": class_to_dict(task),
              "observer_geometry": env.reaction_observer.handle_offset_cart.cpu().tolist(),
              "cases": results, "passed_cases": sum(x["pass"] for x in results),
              "num_cases": len(results), "all_pass": all(x["pass"] for x in results),
              "termination": env.termination_summary(),
              "limitations": ["One seed, aligned casters and nominal friction/damping; not a robustness claim.",
                              "Failed cases end at first termination; post-reset samples excluded from acceptance.",
                              "Post-failure reset rows may use the last configured payload; exclude them from all offline comparisons.",
                              "Candidate estimators are evaluated offline only; truth columns never enter control."]}
    np.savez_compressed(args.output/"traces.npz", **traces)
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[baseline] gate {report['passed_cases']}/{len(results)}, wrote {args.output}", flush=True)
except BaseException as exc:
    error = traceback.format_exc()
    (args.output/"error.txt").write_text(error, encoding="utf-8")
    print(error, file=sys.stderr, flush=True)
finally:
    launcher.app.close(exit_code=1 if error is not None else 0)
if error is not None:
    raise SystemExit(1)
