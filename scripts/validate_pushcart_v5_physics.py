"""Bounded Isaac contact/momentum/reset diagnostic; requires the project Isaac runtime."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'HomieRL'), str(ROOT/'HomieRL/rsl_rl')]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path,
                    default=ROOT / 'logs/bench' / ('v5_physics_ground_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import torch
    from homie_pushcart_v5.sim_env import PushCartSimEnv
    from homie_pushcart_v5.lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from homie_pushcart_v5.teleop import play_config
    cfg = play_config(num_envs=2, episode_length_s=20.)
    env = PushCartSimEnv(cfg, ROOT, args.device, True, with_cart=True)
    policy = HimLowLevelPolicy(frozen_checkpoint(), args.device)
    obs, _ = env.reset()
    snapshots = []
    reset_checks = {}
    with torch.inference_mode():
        for step in range(80):
            env.set_command(.2 if step >= 40 else 0., 0., 0.)
            obs, _, _, _, info, ids, _ = env.step(policy(obs))
            if len(ids):
                raise RuntimeError('Unexpected reset in bounded physics diagnostic')
            assert torch.isfinite(env._hand_wrench).all()
            assert torch.isfinite(env._feet_wrench).all()
            assert torch.isfinite(env._wheel_wrench).all()
            if step >= 10:
                snapshots.append({key: env.physics_samples[key].cpu().clone()
                                  for key in ('hand_wrench', 'feet_wrench', 'wheel_wrench', 'valid', 'ground_valid',
                                              'robot_momentum', 'cart_momentum', 'robot_reaction_wrench',
                                              'cart_hand_wrench')})
        mass_robot = env.centroidal_state(env.robot)['mass']
        mass_cart = env.centroidal_state(env.cart)['mass']
        saved_other = env._hand_wrench[1].clone()
        env.reset_idx(torch.tensor([0], device=args.device, dtype=torch.long))
        reset_checks['selected_invalid'] = bool(not env.dynamics_wrench_valid[0] and not env.ground_wrench_valid[0])
        reset_checks['selected_zero'] = bool(env._hand_wrench[0].count_nonzero() == 0 and env.physics_samples['valid'][:, 0].count_nonzero() == 0)
        reset_checks['other_environment_unchanged'] = bool(torch.equal(saved_other, env._hand_wrench[1]))
        obs = env.get_observations()
        env.step(policy(obs))
        reset_checks['first_physics_sample_invalid'] = bool(not env.physics_samples['valid'][0, 0])
        reset_checks['second_physics_sample_valid'] = bool(env.physics_samples['valid'][1, 0])
        assert all(reset_checks.values()), reset_checks
    trace = {key: torch.cat([row[key] for row in snapshots], 0) for key in snapshots[0]}
    feet = trace['feet_wrench'][..., :3].sum(2)
    wheels = trace['wheel_wrench'][..., :3].sum(2)
    hands = trace['hand_wrench'][..., :3].sum(2)
    dp_robot = trace['robot_momentum'][1:, :, :3]-trace['robot_momentum'][:-1, :, :3]
    dp_cart = trace['cart_momentum'][1:, :, :3]-trace['cart_momentum'][:-1, :, :3]
    gravity = torch.tensor(cfg.sim.gravity)
    residual_robot = dp_robot/cfg.sim.dt-feet[1:]+hands[1:]-mass_robot.cpu()[:, None]*gravity
    residual_cart = dp_cart/cfg.sim.dt-wheels[1:]-hands[1:]-mass_cart.cpu()[:, None]*gravity
    dl_robot = trace['robot_momentum'][1:, :, 3:]-trace['robot_momentum'][:-1, :, 3:]
    dl_cart = trace['cart_momentum'][1:, :, 3:]-trace['cart_momentum'][:-1, :, 3:]
    moment_residual_robot = dl_robot/cfg.sim.dt-trace['feet_wrench'][1:, ..., 3:].sum(2)-trace['robot_reaction_wrench'][1:, ..., 3:]
    moment_residual_cart = dl_cart/cfg.sim.dt-trace['wheel_wrench'][1:, ..., 3:].sum(2)-trace['cart_hand_wrench'][1:, ..., 3:]
    ratio = (feet+wheels)[..., 2].mean(0)/((mass_robot+mass_cart).cpu()*9.81)
    report = {
        'result': 'PASS' if bool(trace['valid'].all() and trace['ground_valid'].all() and ((ratio>.8)&(ratio<1.2)).all()) else 'FAIL',
        'num_envs': 2, 'steps': 80, 'physics_dt_s': cfg.sim.dt, 'physical_samples': len(feet),
        'robot_mass_kg': mass_robot.cpu().tolist(), 'cart_mass_kg': mass_cart.cpu().tolist(),
        'feet_force_mean_world_N': feet.mean(0).tolist(), 'wheels_force_mean_world_N': wheels.mean(0).tolist(),
        'total_ground_support_to_weight_ratio': ratio.tolist(),
        'valid_fraction': float(trace['valid'].float().mean()), 'ground_valid_fraction': float(trace['ground_valid'].float().mean()),
        'robot_linear_balance_residual_rms_N': residual_robot.square().sum(-1).mean(0).sqrt().tolist(),
        'cart_linear_balance_residual_rms_N': residual_cart.square().sum(-1).mean(0).sqrt().tolist(),
        'robot_angular_balance_residual_rms_Nm': moment_residual_robot.square().sum(-1).mean(0).sqrt().tolist(),
        'cart_angular_balance_residual_rms_Nm': moment_residual_cart.square().sum(-1).mean(0).sqrt().tolist(),
        'reset_checks': reset_checks, 'source': env.dynamics_metadata,
    }
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[v5-physics]', json.dumps(report), flush=True)
finally:
    launcher.app.close()
