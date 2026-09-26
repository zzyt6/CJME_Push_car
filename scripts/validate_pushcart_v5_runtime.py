"""Bounded Isaac V5 audit: noisy turns, stop/reversal, physical events and reset."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "HomieRL"), str(ROOT / "HomieRL/rsl_rl")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=350)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--output", type=Path, default=ROOT / "logs/bench" / ("v5_runtime_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import numpy as np
    import torch
    from homie_pushcart_v5.env_smoke import make_configs
    from homie_pushcart_v5.task_env import PushCartEnv
    torch.manual_seed(24)
    cfg, task = make_configs(args.num_envs, True, 17, 4.)
    task.curriculum.enabled = False
    task.randomization.robot_push = False
    task.randomization.cart_force_pulse = False
    task.randomization.wheel_stall_prob_per_s = 0.
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=17)
    env.fixed_command, env.fixed_payload_kg = [.25, 0., .2], 12.5
    env.reset()
    records, physical_pulses = [], []
    resets, terminal_snapshots = 0, 0
    minimum_age = max(35, env.reaction_observer.motion.window_steps + env.reaction.sensor_delay_steps)
    with torch.inference_mode():
        for step in range(args.steps):
            if step in (150, 225):
                command = [0., 0., 0.] if step == 150 else [.25, 0., -.2]
                env.fixed_command = command
                env.cmd_task[:] = torch.tensor(command, device=args.device)
            if step == 70:
                env.inject_cart_wrench([15., 0., 0.], [0., 0., 0.], .025, env_ids=[0], offset_w=[0., .2, 0.])
            obs, critic, reward, done, extras = env.step(torch.zeros(args.num_envs, 18, device=args.device))
            assert obs.shape == (args.num_envs, 705) and critic.shape == (args.num_envs, 753)
            assert torch.isfinite(obs).all() and torch.isfinite(critic).all() and torch.isfinite(reward).all()
            terminal = extras.get("terminal_physics")
            terminal_snapshots += int(terminal is not None)
            resets += int(done.sum())
            good = (env.episode_length_buf > minimum_age) & env.reaction.sensor_valid[:, 0].bool()
            actual = torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1)
            error = env.reaction_observer.estimated_cart_twist - actual
            costs = torch.stack((env._reward_task_wrench(), env._reward_reaction_motion(),
                                 env._reward_centroidal_recovery(), env._reward_support_feasibility()), -1)
            assert torch.isfinite(costs).all()
            records.append(dict(valid=good.cpu().numpy(), error=error.cpu().numpy(),
                                actual=actual.cpu().numpy(), estimated=env.reaction_observer.estimated_cart_twist.cpu().numpy(),
                                reference=env.cmd_ref.cpu().numpy(), age=env.episode_length_buf.cpu().numpy(),
                                costs=costs.cpu().numpy(), ground_valid=env.ground_wrench_valid.cpu().numpy(),
                                correction=env.reaction.command_correction.cpu().numpy()))
            pulse_samples = env.physics_samples["cart_applied_wrench"].clone()
            if terminal is not None:
                pulse_samples[:, terminal["env_ids"]] = terminal["samples"]["cart_applied_wrench"]
            physical_pulses.append(pulse_samples.cpu())
    valid = np.stack([x["valid"] for x in records])
    errors = np.stack([x["error"] for x in records])
    costs = np.stack([x["costs"] for x in records])
    actual = np.stack([x["actual"] for x in records])
    estimated = np.stack([x["estimated"] for x in records])
    # Sensitivity diagnostic only: this component-wise box mean is NOT the
    # observer's LS slope kernel and does not transport older cart-frame vectors
    # or match pose latency. A lower error here is not calibrated sensor accuracy.
    # No truth enters control.
    window = int(round(task.reaction.motion_window_s / env.dt)) + 1
    averaged_actual = np.stack([actual[max(0, i-window+1):i+1].mean(0) for i in range(len(actual))])
    pulses = torch.cat(physical_pulses).sum(0).numpy() * cfg.sim.dt
    assert valid.any(), "No ready reaction sensor samples after warmup"
    assert np.allclose(pulses[0], [.375, 0., 0., 0., 0., -.075], atol=1e-6), pulses[0]
    report = dict(result="PASS_EXECUTION", steps=args.steps, num_envs=args.num_envs,
                  policy="zero_high_level_action", domain_randomization=True, observation_level=17,
                  finite_observations_rewards=True, ready_samples=int(valid.sum()), resets=resets,
                  terminal_snapshot_steps=terminal_snapshots, termination=env.termination_summary(),
                  estimated_cart_twist_rmse_vs_truth=np.sqrt(np.mean(errors[valid]**2, axis=0)).tolist(),
                  estimated_cart_twist_rmse_vs_window_mean_truth=np.sqrt(np.mean((estimated[valid]-averaged_actual[valid])**2,axis=0)).tolist(),
                  ground_measurement_valid_fraction=float(np.stack([x["ground_valid"] for x in records]).mean()),
                  unweighted_cost_mean=costs[valid].mean(0).tolist(),
                  unweighted_cost_max=costs[valid].max(0).tolist(),
                  cost_names=["task_wrench", "reaction_motion", "centroidal_recovery", "support_feasibility"],
                  prescribed_pulse_impulse=pulses.tolist(),
                  limitations=["No trained V5 high-level actor; execution/estimator diagnostic only.",
                               "Noise and domain randomization are enabled; task random events replaced by one known pulse.",
                               "Window mean is a component-wise diagnostic in changing cart axes, not an observer-kernel/frame/latency matched truth reference."])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "traces.npz", **{key:np.stack([x[key] for x in records]) for key in records[0]})
    print("[v5-runtime-audit]", json.dumps(report), flush=True)
finally:
    launcher.app.close()
