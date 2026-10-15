"""Record walk-to-stop switch states of the walking policy for balance-model training.

Each environment walks a random command (HOMIE's ranges, height 0.74) for 1.5-4 s, then the command
decays to zero through the 0.4 s first-order filter used by the deployment benchmarks. The moment
|command| < 0.05 (the official GR00T Balance/Walk switch threshold) the full state is stored: root pose
and velocity, joints, the 6-frame observation history, the last two actions and the upper-body
curriculum state, so a balance episode starts exactly where the walking policy hands over.
"""
from datetime import datetime
import hashlib
import json
import math

import torch

from .config import G1RoughCfg
from .env import HomieIsaacLabEnv

SWITCH_NORM = .05
FILTER_TAU_S = .4


def _sample_commands(n, ranges, device):
    lo = torch.tensor([ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_yaw[0]], device=device)
    hi = torch.tensor([ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_yaw[1]], device=device)
    cmd = lo + (hi - lo) * torch.rand(n, 3, device=device)
    # Single-axis and two-axis walks as well as full combinations.
    return cmd * (torch.rand(n, 3, device=device) > .3)


def record(args, app, root):
    from homie_pushcart.lowlevel import HimLowLevelPolicy

    torch.manual_seed(args.seed)
    cfg = G1RoughCfg()
    cfg.env.num_envs = args.num_envs
    cfg.domain_rand.push_robots = False          # a normal stop; balance training adds its own pushes
    cfg.domain_rand.init_upper_ratio = 1.        # the walking model's final curriculum
    cfg.commands.resampling_time = 1e6           # commands are driven here
    env = HomieIsaacLabEnv(cfg, root, args.device, True)
    policy = HimLowLevelPolicy(args.checkpoint, args.device)
    n, dev, dt = env.num_envs, env.device, env.dt
    alpha = min(1., dt / FILTER_TAU_S)
    height = cfg.rewards.base_height_target
    last_frame = slice(env.num_obs - env.num_one_step_obs, env.num_obs)

    target = _sample_commands(n, cfg.commands.ranges, dev)
    filtered = torch.zeros(n, 3, device=dev)
    walk_left = torch.empty(n, device=dev).uniform_(1.5, 4.)
    stopping = torch.zeros(n, dtype=torch.bool, device=dev)
    peak = torch.zeros(n, device=dev)
    states = {k: [] for k in ('root_pose', 'root_vel', 'joint_pos', 'joint_vel', 'obs', 'critic_obs', 'actions',
                              'last_actions', 'current_upper_actions', 'delta_upper_actions', 'peak_command')}
    obs, _ = env.reset()
    steps = int(math.ceil(args.seconds / dt))
    for step in range(steps):
        if not app.is_running():
            raise RuntimeError('Simulation closed while recording')
        filtered += alpha * (torch.where(stopping[:, None], 0., target) - filtered)
        peak = torch.maximum(peak, filtered.norm(dim=-1))
        env.commands[:, :3] = filtered
        env.commands[:, 4] = height
        # The policy sees the current command, as in deployment.
        obs = obs.clone()
        obs[:, last_frame.start:last_frame.start + 3] = filtered * env.commands_scale
        obs, _, _, _, _, reset_ids, _ = env.step(policy(obs))
        walk_left -= dt
        stopping |= walk_left <= 0
        restarted = torch.zeros(n, dtype=torch.bool, device=dev)
        restarted[reset_ids] = True
        switch = stopping & (filtered.norm(dim=-1) < SWITCH_NORM) & ~restarted
        if switch.any():
            ids = switch.nonzero().flatten()
            pose = env.robot.data.root_link_pose_w.torch[ids].clone()
            pose[:, :3] -= env.env_origins[ids]
            values = dict(root_pose=pose, root_vel=env.robot.data.root_com_vel_w.torch[ids],
                          joint_pos=env.dof_pos[ids], joint_vel=env.dof_vel[ids], obs=env.obs_buf[ids],
                          critic_obs=env.privileged_obs_buf[ids], actions=env.actions[ids],
                          last_actions=env.last_last_actions[ids],
                          current_upper_actions=env.current_upper_actions[ids],
                          delta_upper_actions=env.delta_upper_actions[ids], peak_command=peak[ids])
            for key, value in values.items():
                states[key].append(value.detach().clone().cpu())
        # New walk after a switch or a reset (a fall or timeout breaks the segment).
        fresh = switch | restarted
        if fresh.any():
            m = int(fresh.sum())
            target[fresh] = _sample_commands(m, cfg.commands.ranges, dev)
            walk_left[fresh] = torch.empty(m, device=dev).uniform_(1.5, 4.)
            stopping[fresh] = False
            peak[fresh] = 0.
        filtered[restarted] = 0.
        if step % 500 == 0:
            print(f'[bank] step {step}/{steps}: {sum(len(v) for v in states["obs"])} switch states', flush=True)

    states = {k: torch.cat(v) for k, v in states.items()}
    count = len(states['obs'])
    speed = states['root_vel'][:, :2].norm(dim=-1)
    meta = {'created': datetime.now().isoformat(timespec='seconds'), 'checkpoint': str(args.checkpoint),
            'checkpoint_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
            'num_envs': n, 'seconds': args.seconds, 'seed': args.seed, 'count': count,
            'switch_norm': SWITCH_NORM, 'filter_tau_s': FILTER_TAU_S, 'height_command': height,
            'pelvis_height_m': [float(states['root_pose'][:, 2].min()), float(states['root_pose'][:, 2].mean()),
                                float(states['root_pose'][:, 2].max())],
            'pelvis_speed_m_s': [float(speed.mean()), float(speed.quantile(.95)), float(speed.max())],
            'peak_command_mean': float(states['peak_command'].mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'states': states, 'meta': meta}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('BANK PASS ' + json.dumps(meta), flush=True)
