"""Matched nominal vx/wz rollout; evaluation-only, never modifies training files.

The command program is identical to audit_pushcart_v53_trained_grasp.py.
V5.2 full/aligned uses the V5.3 nominal cart inertia setter order so the plant
is shared. V5.2 original/native is a separate historical-system reference.
Raw physical samples, including terminal frames, are recorded before reset.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
V52 = ROOT / 'logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt'
V53 = ROOT / 'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-37-10_v53_final_20k/model_15000.pt'
FULL = ROOT / 'logs/bench/v52_falcon_explicit/model_explicit_pd_full_22/estimator.pt'
TOKEN = 'V52_V53_TRACKING_CHILD'

if not os.environ.get(TOKEN):
    token = uuid.uuid4().hex
    child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=dict(os.environ, **{TOKEN: token}))
    if '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(child.returncode)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--output', type=Path, required=True)
    known, _ = probe.parse_known_args()
    result = json.loads(known.output.read_text(encoding='utf8')) if known.output.is_file() else {}
    passed = child.returncode == 0 and result.get('run_id') == token and result.get('result') == 'PASS_EXECUTION'
    if result.get('run_id') == token:
        result.update(child_exit_code=child.returncode, parent_verified=passed)
        known.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(f'[tracking parent] exit={child.returncode} verified={passed}', flush=True)
    raise SystemExit(0 if passed else 1)

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--variant', choices=('v52_original', 'v52_full', 'v53'), required=True)
p.add_argument('--cart-inertia', choices=('native', 'aligned'), default='aligned')
p.add_argument('--num_envs', type=int, default=24)
p.add_argument('--steps', type=int, default=1200)
p.add_argument('--level', type=int, default=13)
p.add_argument('--seed', type=int, default=20260928)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.num_envs % 12:
    raise ValueError('Expected a multiple of 12 cases')
launcher = AppLauncher(args)

try:
    import numpy as np
    import torch
    import warp as wp
    from homie_isaaclab.math_utils import class_to_dict
    from homie_pushcart_v53.domain_randomization import loaded_cart_properties

    old = args.variant.startswith('v52')
    package = 'homie_pushcart_v52' if old else 'homie_pushcart_v53'
    module = lambda name: importlib.import_module(package + '.' + name)
    checkpoint = V52 if old else V53
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if old:
        drive = 'training' if args.variant == 'v52_original' else 'explicit_pd'
        cfg, task = module('control_contract').nominal_configs(checkpoint, drive, 30.)
        cfg.env.num_envs = args.num_envs
        task.curriculum.init_level = args.level
        # Old nominal_configs does not disable these two random effects.
        task.randomization.grasp_threshold_jitter = 0.
        task.observations.cart_pose_delay_steps_max = 0
        task.observations.upper_torque_bias_frac_of_limit = 0.
    else:
        cfg, task = module('env_smoke').make_configs(args.num_envs, False, args.level, 30., checkpoint=checkpoint)
    task.curriculum.enabled = False
    task.diagnostics.finite_check_interval_steps = 100
    low = Path(cfg.pushcart_saved_checkpoint)
    if not low.is_absolute():
        low = ROOT / low
    env = module('task_env').PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level, checkpoint=low)
    reasons = module('task_env').TERMINATION_REASONS
    policy = module('hl_policy').HighLevelPolicy(checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    if args.variant == 'v52_full':
        module('transfer_estimator').install_estimator(policy.actor, FULL)
    cases = [dict(env=i, payload_kg=[5., 15., 25.][(i // 4) % 3],
                  scenario=['straight', 'left', 'right', 'reverse'][i % 4], replicate=i // 12)
             for i in range(args.num_envs)]
    payloads = torch.tensor([c['payload_kg'] for c in cases], device=args.device)
    original_randomize = env._randomize_cart

    def intended_properties(ids):
        return loaded_cart_properties(env.cart_default_mass[ids], env.cart_default_inertia[ids],
            env.cart_default_com[ids], env.cart_base_body, torch.ones_like(env.cart_default_mass[ids]),
            payloads[ids.long(), None], torch.zeros(len(ids), 2, device=args.device))

    def fixed_payload_groups(ids):
        previous = env.fixed_payload_kg
        try:
            for payload in (5., 15., 25.):
                selected = ids[payloads[ids.long()] == payload]
                if len(selected):
                    env.fixed_payload_kg = payload
                    original_randomize(selected)
                    if old and args.cart_inertia == 'aligned':
                        masses, inertias, coms = intended_properties(selected)
                        # Only this evaluation instance: use the same physical cart
                        # tensor as V5.3, rather than the old reset rotation bug.
                        env.cart.set_coms_index(coms=coms, env_ids=selected)
                        env.cart.set_inertias_index(inertias=inertias, env_ids=selected)
                        env.refresh_dynamics_properties(env.cart, selected, masses=masses, inertias=inertias, coms=coms)
        finally:
            env.fixed_payload_kg = previous

    env._randomize_cart = fixed_payload_groups
    env.fixed_command = [0., 0., 0.]
    env.fixed_caster_angle = 0.
    obs, critic = env.reset()
    assert not cfg.domain_rand.use_random and not cfg.noise.add_noise
    assert not env.hl_delay.any() and torch.all(env.grasp_jitter == 1)
    assert torch.all(env.Kp_factors == 1) and torch.all(env.Kd_factors == 1)
    torque_delay = env.upper_torque_delay if old else env.joint_torque_delay
    pose_delay = env.cart_pose_delay if old else env.heading_sensor.delay
    assert not torque_delay.any() and not pose_delay.any()
    torch.testing.assert_close(env.cart_mass_total, payloads + 14.4)
    assert not env.payload_com.any()
    raw_cart_inertia = wp.to_torch(env.cart.root_view.get_inertias()).clone().reshape(args.num_envs, -1, 3, 3).transpose(-1, -2)
    expected = intended_properties(torch.arange(args.num_envs, device=args.device))[1].reshape_as(raw_cart_inertia)
    inertia_difference = (raw_cart_inertia.to(expected.device) - expected).abs().max().item()
    if args.cart_inertia == 'aligned' or not old:
        assert inertia_difference < 5e-6, inertia_difference
    cpu = lambda v: v.detach().cpu().tolist()
    snapshots = dict(mass_kg=cpu(env.cart_mass_total), wheel_friction=cpu(env.wheel_friction),
        wheel_damping=cpu(env.wheel_damping), swivel_damping=cpu(env.swivel_damping),
        kp_factors=cpu(env.Kp_factors), kd_factors=cpu(env.Kd_factors),
        p_gains=cpu(env.p_gains), d_gains=cpu(env.d_gains), thresholds=cpu(env._grasp_thresholds()),
        torque_delay_steps=cpu(torque_delay), pose_delay_steps=cpu(pose_delay),
        cart_body_names=list(env.cart.body_names),
        cart_body_mass=cpu(wp.to_torch(env.cart.root_view.get_masses())),
        cart_inertia_link_frame=cpu(raw_cart_inertia),
        cart_com_pose=cpu(wp.to_torch(env.cart.root_view.get_coms())),
        robot_body_mass=cpu(wp.to_torch(env.robot.root_view.get_masses())),
        robot_inertia=cpu(wp.to_torch(env.robot.root_view.get_inertias())),
        cart_materials=cpu(wp.to_torch(env.cart.root_view.get_material_properties())),
        robot_materials=cpu(wp.to_torch(env.robot.root_view.get_material_properties())),
        intended_inertia_max_difference=inertia_difference)
    records = []
    original_reward = env.compute_reward
    step_index = 0
    episode_ids = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)

    def record_reward():
        original_reward()
        values = dict(time=torch.full((args.num_envs,), (step_index+1)*env.dt, device=args.device),
            episode_id=episode_ids, episode_step=env.episode_length_buf,
            command=env.cmd_task, reference=env.cmd_ref,
            velocity=torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1),
            thresholds=env._grasp_thresholds(), high_action=env.hl_raw,
            force=env.ts.force_c, moment_norm=env.ts.torque_norm, torques=env.torques,
            robot_wrench=env.applied_robot_wrench, cart_wrench=env.cart_applied_wrench,
            stall=env.stall_timer > 0, terminated=env.terminated_buf, timeout=env.time_out_buf,
            reasons=torch.stack([env.term_reason_buf[k] for k in reasons], -1))
        copied = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}
        for k in ('velocity', 'reference', 'torques'):
            if not np.isfinite(copied[k]).all():
                raise FloatingPointError(k)
        records.append(copied)
    env.compute_reward = record_reward

    def protocol(seconds):
        command = torch.zeros(args.num_envs, 3, device=args.device)
        case = torch.arange(args.num_envs, device=args.device) % 4
        if seconds < 2.: return 'settle', command
        if seconds < 6.:
            command[:, 0] = torch.where(case == 3, -.2, .3)
            return 'start', command
        if seconds < 12. or 16. <= seconds < 21.:
            command[:, 0] = torch.where(case == 0, .6, torch.where(case == 3, -.35, .35))
            command[:, 2] = torch.where(case == 1, .25, torch.where(case == 2, -.25, 0.))
            if seconds >= 16.: command[:, 2] *= -1.
            return ('main' if seconds < 12. else 'restart_turn_reverse'), command
        return ('brake' if seconds < 16. else 'final_stop'), command

    with torch.inference_mode():
        for step_index in range(args.steps):
            if not launcher.app.is_running():
                raise RuntimeError('Simulator closed early')
            stage, command = protocol(step_index*env.dt)
            env.cmd_task.copy_(command)
            env.cmd_timer.fill_(1000000)
            actions = policy(obs, critic, estimator_obs=env.falcon_obs) if old else policy(obs, critic)
            obs, critic, reward, done, extras = env.step(actions)
            episode_ids += done.long()
            if step_index % 200 == 0:
                print(f'[tracking] {args.variant}/{args.cart_inertia} {step_index}/{args.steps} '
                      f'{stage} terminations={int(episode_ids.sum())}', flush=True)

    assert len(records) == args.steps
    data = {k: np.stack([r[k] for r in records]) for k in records[0]}
    assert not data['robot_wrench'].any() and not data['cart_wrench'].any() and not data['stall'].any()
    stages = np.array([protocol(i*env.dt)[0] for i in range(args.steps)])
    source_files = [Path(__file__).resolve(), checkpoint, low, ROOT/'assets/g1_pushcart_v5/g1_pushcart.usd',
        Path(module('task_env').__file__), Path(module('interaction_policy').__file__),
        checkpoint.parent/'params/env.yaml', checkpoint.parent/'params/agent.yaml']
    if args.variant == 'v52_full': source_files.append(FULL)
    result = dict(result='PASS_EXECUTION', run_id=os.environ[TOKEN], variant=args.variant,
        cart_inertia=args.cart_inertia, upper_drive=task.actions.upper_drive,
        checkpoint=str(checkpoint), iteration=policy.iteration, force_head=str(FULL) if args.variant == 'v52_full' else None,
        lowlevel_checkpoint=str(low), seed=args.seed, level=args.level, num_envs=args.num_envs, steps=args.steps,
        control_dt=env.dt, physics_dt=cfg.sim.dt, cases=cases, initial_parameters=snapshots,
        randomization_effective=False, noise_effective=False, action_exploration=False,
        control_contract=module('control_contract').control_contract(cfg, env.dof_names, cpu(env.torque_limits), task.actions.upper_drive),
        termination_config=class_to_dict(task.termination), level_params=env.level_params.as_dict(),
        reference_tau_s=task.commands.ref_tau_s,
        terminations=int(data['terminated'].sum()), timeouts=int(data['timeout'].sum()),
        termination_reasons=dict(zip(reasons, data['reasons'].sum(axis=(0, 1)).tolist())),
        overall_tracking_mae=np.abs(data['velocity']-data['reference']).mean(axis=(0,1)).tolist(),
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        interpretation=['Pre-reset physical velocity, not the additional reward low-pass velocity.',
            'Global command program resumes after automatic resets; failures must not be omitted.',
            'Nominal duplicates are not independent stochastic trials.',
            'V52 aligned cart inertia is an explicit evaluation-only plant alignment.',
            'Original/native is historical-system context; it is not an isolated network comparison.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **data, stages=stages)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print('[tracking] RESULT', json.dumps({k: result[k] for k in ('variant','cart_inertia','terminations','overall_tracking_mae')}), flush=True)
except BaseException:
    import traceback
    error = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(error=error)), encoding='utf8')
    print(error, flush=True)
    raise
finally:
    launcher.app.close()
