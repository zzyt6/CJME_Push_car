"""Evaluate a policy on the stand-still task: walk-to-stop switches and the stage-2 grasp stance.

Every environment starts at t = 0 (from a recorded switch state or the grasp stance) and runs for
`--seconds` with a zero command. Reported per start type and condition:
- fall rate;
- steps: a foot unloaded (normal force < 1 N) for >= 3 control steps (60 ms) counts as one;
- foot drift: horizontal ankle displacement from t = 0 (max over the two feet);
- pelvis height: at t = 0, mean over the last 2 s, and their difference (the drop after a switch);
- pelvis pitch over the last 2 s (the grasp stance was measured at 0.019 rad).
Conditions: `nominal` (no randomization, arms hold the grasp) and `disturbed` (training randomization:
upper-body curriculum, 0.2 m/s pushes, delay, gains, friction).
"""
import json
import math

import torch

from .config import G1RoughCfg
from .env import HomieIsaacLabEnv
from .math_utils import euler_from_quaternion

LIFT_N, LIFT_STEPS, SETTLE_S = 1., 3, 2.


def _stats(x):
    x = x.float()
    return {'mean': round(float(x.mean()), 4), 'p95': round(float(x.quantile(.95)), 4), 'max': round(float(x.max()), 4)}


def _run(env, policy, seconds):
    n, dt = env.num_envs, env.dt
    steps = int(math.ceil(seconds / dt))
    settle = int(SETTLE_S / dt)
    obs, _ = env.reset()
    feet0 = env.feet_pos[:, :, :2].clone()
    z0 = env.root_states[:, 2].clone()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    unloaded = torch.zeros(n, 2, dtype=torch.long, device=env.device)
    lifts = torch.zeros(n, device=env.device)
    z_sum = torch.zeros(n, device=env.device)
    pitch_sum = torch.zeros(n, device=env.device)
    for step in range(steps):
        obs, _, _, _, _, reset_ids, _ = env.step(policy(obs))
        alive[reset_ids] = False
        off = env.contact_forces[:, env.feet_indices, 2] < LIFT_N
        unloaded = torch.where(off, unloaded + 1, 0)
        lifts += ((unloaded == LIFT_STEPS).sum(1) * alive).float()
        if step >= steps - settle:
            z_sum += env.root_states[:, 2]
            pitch_sum += euler_from_quaternion(env.base_quat)[1][:, 0]
    drift = (env.feet_pos[:, :, :2] - feet0).norm(dim=-1).max(dim=1).values
    z_end = z_sum / settle
    ok = alive
    return {'envs': n, 'fall_rate': round(float(1 - alive.float().mean()), 4),
            'steps_per_episode': _stats(lifts[ok]), 'episodes_with_a_step': round(float((lifts[ok] > 0).float().mean()), 4),
            'foot_drift_m': _stats(drift[ok]), 'pelvis_z_start_m': _stats(z0[ok]), 'pelvis_z_end_m': _stats(z_end[ok]),
            'pelvis_z_end_std_m': round(float(z_end[ok].std()), 4),
            'pelvis_drop_m': _stats((z0 - z_end)[ok]), 'pelvis_pitch_end_rad': _stats(pitch_sum[ok] / settle)}


def evaluate(args, app, root):
    """One condition per process (one simulation scene); both start types run in the same scene."""
    from homie_pushcart.lowlevel import HimLowLevelPolicy

    torch.manual_seed(args.seed)
    cfg = G1RoughCfg()
    cfg.env.num_envs = args.num_envs
    cfg.env.episode_length_s = args.seconds + 5.
    cfg.balance.enabled = True
    cfg.balance.bank_ratio = 1.
    if args.condition == 'nominal':
        for key in dir(cfg.domain_rand):
            if key.startswith('randomize_') or key in ('use_random', 'push_robots', 'delay'):
                setattr(cfg.domain_rand, key, False)
        cfg.noise.add_noise = False
    env = HomieIsaacLabEnv(cfg, root, args.device, True)
    policy = HimLowLevelPolicy(args.checkpoint, args.device)
    report = {'checkpoint': str(args.checkpoint), 'condition': args.condition, 'seconds': args.seconds}
    for start, ratio in (('switch', 1.), ('stance', 0.)):
        if not app.is_running():
            raise RuntimeError('Simulation closed during evaluation')
        cfg.balance.bank_ratio = ratio
        report[start] = _run(env, policy, args.seconds)
        print(f'[balance-eval] {args.condition}/{start}: ' + json.dumps(report[start]), flush=True)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('BALANCE EVAL PASS ' + str(out), flush=True)
