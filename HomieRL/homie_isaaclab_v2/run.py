"""Smoke checks, HIM training, and checkpoint playback."""
from datetime import datetime
import json
import math
import random

import numpy as np
import torch

from .config import G1RoughCfg, G1RoughCfgPPO
from .env import HomieIsaacLabEnv
from .math_utils import class_to_dict


def run(args, app, root):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = G1RoughCfg()
    cfg.env.num_envs = args.num_envs
    cfg.balance.enabled = getattr(args, 'balance', False)
    # Playback is deterministic by default.  A separate opt-in mode enables
    # only the upper-body pose curriculum used by HOMIE training; it does not
    # turn on mass, friction, pushes, delays, or observation noise.
    play_upper_randomization = args.mode == 'play' and getattr(args, 'play_upper_randomization', False)
    if args.no_randomization or (args.mode == 'play' and not play_upper_randomization):
        for key in dir(cfg.domain_rand):
            if key.startswith('randomize_') or key in ('use_random', 'push_robots', 'delay'):
                setattr(cfg.domain_rand, key, False)
        cfg.noise.add_noise = False
    elif play_upper_randomization:
        for key in dir(cfg.domain_rand):
            if key.startswith('randomize_') or key in ('push_robots', 'delay'):
                setattr(cfg.domain_rand, key, False)
        cfg.domain_rand.use_random = False
        cfg.domain_rand.upper_body_randomization = True
        cfg.domain_rand.init_upper_ratio = 1.0
        cfg.noise.add_noise = False
    env = HomieIsaacLabEnv(cfg, root, args.device, args.headless)
    report_dir = root / 'logs/homie_isaaclab_v2'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'physics_config.json').write_text(json.dumps(env.physics_report, indent=2), encoding='utf-8')
    if args.mode == 'smoke':
        physics = env.physics_report
        assert abs(physics['physics_dt_s'] - .005) < 1e-8
        assert abs(physics['control_dt_s'] - .02) < 1e-8
        assert physics['decimation'] == 4 and physics['max_episode_steps'] == 1000
        assert all(abs(v - cfg.asset.armature) < 1e-7 for v in physics['armature'])
        assert all(abs(body['linear_damping'] - cfg.asset.linear_damping) < 1e-7 and
                   abs(body['angular_damping'] - cfg.asset.angular_damping) < 1e-7
                   for body in physics['rigid_body_properties'])
        assert all(abs(body['max_angular_velocity_deg_s'] - math.degrees(cfg.asset.max_angular_velocity)) < 1e-2
                   for body in physics['rigid_body_properties'])
        assert len(physics['collision_shapes']) == 2
        assert all(abs(shape['contact_offset'] - cfg.sim.physx.contact_offset) < 1e-7 and
                   abs(shape['rest_offset'] - cfg.sim.physx.rest_offset) < 1e-7
                   for shape in physics['collision_shapes'])
        obs, critic = env.reset()
        resets = 0
        max_contact = 0.
        trace = {'actions': [], 'root_states': [], 'joint_positions': [], 'foot_contacts': [], 'commands': [], 'dones': []}
        for step in range(args.steps):
            if not app.is_running():
                raise RuntimeError('Simulation closed before smoke test completed')
            # Small nonzero actions exercise both command mapping and state updates.
            actions = .05 * torch.randn(env.num_envs, 12, device=env.device)
            obs, critic, reward, done, info, ids, terminal = env.step(actions)
            for name, tensor in [('obs', obs), ('critic', critic), ('reward', reward),
                                 ('terminal', terminal), ('joints', env.dof_pos)]:
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(f'{name} has non-finite values at step {step}')
            assert obs.shape == (args.num_envs, 456)
            assert critic.shape == (args.num_envs, 79)
            assert terminal.shape == (len(ids), 79)
            resets += len(ids)
            max_contact = max(max_contact, env.contact_forces.norm(dim=-1).max().item())
            if args.record_ovd is not None:
                for name, tensor in [('actions', actions), ('root_states', env.root_states),
                                     ('joint_positions', env.dof_pos), ('commands', env.commands), ('dones', done),
                                     ('foot_contacts', env.contact_forces[:, env.feet_indices])]:
                    trace[name].append(tensor.detach().cpu().numpy().copy())
        # Exercise the actual timeout path and the terminal-observation contract.
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, _, done, info, ids, terminal = env.step(torch.zeros(env.num_envs, 12, device=env.device))
        assert done.all() and info['time_outs'].all() and len(ids) == env.num_envs
        assert torch.isfinite(terminal).all()
        print(f'[HOMIE] Root height after reset: {env.root_states[:, 2].tolist()}; contact bodies: {env._contact_names}')
        assert max_contact > 0., 'No contacts were reported by the physics sensor'
        result = {'result': 'PASS', 'num_envs': args.num_envs, 'steps': args.steps,
                  'resets': resets, 'max_contact_force': max_contact,
                  'timeout_reset': True, 'physics_parameters': 'PASS', 'actor_shape': list(obs.shape),
                  'critic_shape': list(critic.shape), 'joint_names': env.dof_names,
                  'lab_joint_ids': env.joint_ids.tolist(), 'randomization': cfg.domain_rand.use_random}
        if cfg.balance.enabled:
            assert not env.commands[:, :4].any(), 'balance commands must stay zero'
            assert torch.all(env.commands[:, 4] == cfg.rewards.base_height_target)
            result['balance'] = env.physics_report['balance']
        path = root / f'logs/homie_isaaclab_v2/smoke_result{"_balance" if cfg.balance.enabled else ""}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        if args.record_ovd is not None:
            np.savez_compressed(args.record_ovd / 'rollout.npz',
                                **{key: np.stack(value) for key, value in trace.items()},
                                control_dt_s=env.dt, joint_names=np.asarray(env.dof_names))
            (args.record_ovd / 'physics_config.json').write_text(json.dumps(env.physics_report, indent=2), encoding='utf-8')
            (args.record_ovd / 'smoke_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('SMOKE PASS ' + json.dumps(result))
        return

    from rsl_rl.runners import HIMOnPolicyRunner
    train_cfg = class_to_dict(G1RoughCfgPPO())
    train_cfg['runner'].update(logger='tensorboard', experiment_name='homie_g1_isaaclab_v2', run_name=args.run_name)
    log_dir = root / 'logs/homie_isaaclab_v2' / f'{datetime.now():%Y%m%d_%H%M%S}_{args.run_name}'
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / 'config.json').write_text(json.dumps({'env': class_to_dict(cfg), 'train': train_cfg}, indent=2), encoding='utf-8')
    (log_dir / 'physics_config.json').write_text(json.dumps(env.physics_report, indent=2), encoding='utf-8')
    runner = HIMOnPolicyRunner(env, train_cfg, str(log_dir), args.device)
    if args.checkpoint is not None:
        runner.load(str(args.checkpoint), load_optimizer=args.mode == 'train')
    if args.mode == 'train':
        runner.learn(args.max_iterations, init_at_random_ep_len=True)
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        if not all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters()):
            raise FloatingPointError('Non-finite policy parameters after training')
        print(f'TRAIN PASS: {args.max_iterations} iterations; checkpoints in {log_dir}')
    else:
        policy = runner.get_inference_policy(device=args.device)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running():
                    break
                obs, *_ = env.step(policy(obs))
        print(f'PLAY PASS: {args.checkpoint}')
