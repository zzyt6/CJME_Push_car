"""Fresh Isaac evaluation of a saved V5.3 policy, preserving grasp termination rules.

Physical samples are captured before automatic resets. Estimator pairs are
captured before stepping, using the exact observation and target at that time.
No training, optimizer updates, or changes to overload thresholds are performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))
DEFAULT_OUTPUT = ROOT / 'logs/bench/v53_trained_grasp_20260928/nominal.json'
if not os.environ.get('V53_GRASP_AUDIT_CHILD'):
    token = uuid.uuid4().hex
    child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=dict(os.environ, V53_GRASP_AUDIT_CHILD=token))
    if '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(child.returncode)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    known, _ = probe.parse_known_args()
    result = json.loads(known.output.read_text(encoding='utf8')) if known.output.is_file() else {}
    passed = child.returncode == 0 and result.get('run_id') == token and result.get('result') == 'PASS_EXECUTION'
    if result.get('run_id') == token:
        result.update(child_exit_code=child.returncode, parent_verified=passed)
        known.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(f'[grasp audit parent] exit={child.returncode} verified={passed}', flush=True)
    raise SystemExit(0 if passed else 1)

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--mode', choices=('nominal', 'full', 'sampled', 'noise'), default='nominal')
p.add_argument('--num_envs', type=int, default=24)
p.add_argument('--steps', type=int, default=1200)
p.add_argument('--level', type=int, default=13)
p.add_argument('--seed', type=int, default=20260928)
p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.num_envs % 12:
    raise ValueError('num_envs must be a multiple of 12 (3 payloads x 4 protocols)')
launcher = AppLauncher(args)

try:
    import numpy as np
    import torch
    from homie_isaaclab.math_utils import class_to_dict
    from homie_pushcart_v53.env_smoke import make_configs
    from homie_pushcart_v53.task_env import PushCartEnv, TERMINATION_REASONS
    from homie_pushcart_v53.hl_policy import HighLevelPolicy
    from homie_pushcart_v53.cart_quality import rotate_planar

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    randomized = args.mode in ('full', 'sampled')
    cfg, task = make_configs(args.num_envs, randomized, args.level, 30., checkpoint=args.checkpoint)
    task.curriculum.enabled = False
    task.diagnostics.finite_check_interval_steps = 100
    if args.mode == 'noise':
        cfg.noise.add_noise = True
    lowlevel = getattr(cfg, 'pushcart_saved_checkpoint', None)
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level,
                     checkpoint=Path(lowlevel) if lowlevel else None)
    policy = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    cases = [dict(env=i, payload_kg=[5., 15., 25.][(i // 4) % 3],
                  scenario=['straight', 'left', 'right', 'reverse'][i % 4],
                  replicate=i // 12) for i in range(args.num_envs)]
    payloads = torch.tensor([c['payload_kg'] for c in cases], device=args.device)
    original_randomize = env._randomize_cart

    def fixed_payload_groups(ids):
        previous = env.fixed_payload_kg
        for payload in (5., 15., 25.):
            selected = ids[payloads[ids.long()] == payload]
            if len(selected):
                env.fixed_payload_kg = payload
                original_randomize(selected)
        env.fixed_payload_kg = previous

    env._randomize_cart = fixed_payload_groups
    env.fixed_command = [0., 0., 0.]
    if not randomized:
        env.fixed_caster_angle = 0.
    obs, critic = env.reset()
    if args.mode == 'nominal':
        assert not env.task_randomization_enabled and not cfg.noise.add_noise
        assert not env.hl_delay.any() and not env.joint_torque_delay.any()
        assert not env.heading_sensor.delay.any()
        torch.testing.assert_close(env.cart_mass_total, payloads + 14.4)
    torch.testing.assert_close(env.cart_payload_kg, payloads)
    snapshots = dict(mass_kg=env.cart_mass_total.cpu().tolist(),
        wheel_friction=env.wheel_friction.cpu().tolist(),
        wheel_damping=env.wheel_damping_per_wheel.cpu().tolist(),
        swivel_damping=env.swivel_damping_per_wheel.cpu().tolist(),
        heading_delay_steps=env.heading_sensor.delay.cpu().tolist(),
        torque_delay_steps=env.joint_torque_delay.cpu().tolist(),
        kp_factors=env.Kp_factors.cpu().tolist(), kd_factors=env.Kd_factors.cpu().tolist())

    records, estimator_records = [], []
    original_reward = env.compute_reward
    step_index = 0
    stage = 'settle'
    episode_ids = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    # All statistics below use pre-reset states, including the actual terminal frame.
    def record_reward():
        original_reward()
        yaw = env.ts.cart_yaw[:, None].expand(-1, 2).reshape(-1)
        rot = lambda a: rotate_planar(a.reshape(-1, 3), yaw, inverse=True).reshape(-1, 2, 3)
        mean_f, mean_m, _, count = env.grasp_wrench_interval_mean()
        palm = env.robot.data.body_link_pose_w.torch[:, env.palm_body_ids, :3]
        span = palm[:, 1] - palm[:, 0]
        unit = span / span.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # This specific equal/opposite component has zero net force AND moment.
        # General left-right force differences can instead be useful yaw moments.
        internal = .5 * ((env.ts.force_w_raw[:, 0]-env.ts.force_w_raw[:, 1])*unit).sum(-1)
        values = dict(time=torch.full((args.num_envs,), (step_index+1)*env.dt, device=args.device),
            episode_id=episode_ids.clone(), episode_step=env.episode_length_buf.clone(),
            command=env.cmd_task, reference=env.cmd_ref,
            velocity=torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1),
            force=rot(env.ts.force_w_raw), moment=rot(env.ts.torque_w_raw),
            force_mean=rot(mean_f), moment_mean=rot(mean_m),
            force_used=env.ts.force_c, moment_norm_used=env.ts.torque_norm,
            label_valid=env.dynamics_wrench_valid & (count == cfg.control.decimation),
            thresholds=env._grasp_thresholds(), over_directions=env.overload_dir,
            overload_counter=env.overload_counter, internal_axial_force=internal,
            hand_separation=span.norm(dim=-1),
            net_force=rotate_planar(env.ts.cart_force_w, env.ts.cart_yaw, inverse=True),
            net_moment_handle=rotate_planar(env.ts.cart_moment_w, env.ts.cart_yaw, inverse=True),
            torques=env.torques, q=env.dof_pos, dq=env.dof_vel,
            high_action=env.hl_raw, total_mass=env.cart_mass_total,
            robot_wrench=env.applied_robot_wrench, cart_wrench=env.cart_applied_wrench,
            stall=env.stall_timer > 0, terminated=env.terminated_buf, timeout=env.time_out_buf,
            reasons=torch.stack([env.term_reason_buf[k] for k in TERMINATION_REASONS], -1))
        copied = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}
        for k in ('force', 'moment', 'velocity', 'torques', 'thresholds'):
            if not np.isfinite(copied[k]).all():
                raise FloatingPointError(f'Nonfinite audit field {k}')
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
            if seconds >= 16.:
                command[:, 2] *= -1.
            return ('main' if seconds < 12. else 'restart_turn_reverse'), command
        return ('brake' if seconds < 16. else 'final_stop'), command

    # Separate RNG prevents action sampling from directly consuming domain RNG.
    action_rng = torch.Generator(device=args.device).manual_seed(args.seed + 100)
    with torch.inference_mode():
        for step_index in range(args.steps):
            if not launcher.app.is_running():
                raise RuntimeError('Simulator closed before audit completion')
            stage, command = protocol(step_index*env.dt)
            env.cmd_task.copy_(command)
            env.cmd_timer.fill_(1000000)
            hands = policy.actor.predict_hand_wrench_history(obs) * policy.actor.hand_scale
            speed = policy.actor.predict_cart_velocity_history(obs) / policy.actor.cart_velocity_scale
            estimator_records.append(dict(
                prediction=hands.cpu().numpy().copy(),
                label=(env.interaction_target[:, :12]*policy.actor.hand_scale).cpu().numpy().copy(),
                valid=(env.interaction_target[:, 12] > .5).cpu().numpy().copy(),
                speed_prediction=speed.cpu().numpy().copy(),
                speed_label=(env.cart_velocity_target[:, :3]/policy.actor.cart_velocity_scale).cpu().numpy().copy(),
                speed_valid=(env.cart_velocity_target[:, 3] > .5).cpu().numpy().copy()))
            actions = policy(obs, critic)
            if args.mode == 'sampled':
                dist = policy.actor.distribution
                std = dist.std_param if dist.std_type == 'scalar' else dist.log_std_param.exp()
                actions = actions + std*torch.randn(actions.shape, device=args.device, generator=action_rng)
            obs, critic, reward, done, extras = env.step(actions)
            episode_ids += done.long()
            if step_index % 200 == 0:
                print(f'[trained grasp] mode={args.mode} step={step_index}/{args.steps} stage={stage} '
                      f'terminations={int(episode_ids.sum())}', flush=True)

    assert len(records) == args.steps
    data = {k: np.stack([r[k] for r in records]) for k in records[0]}
    estimates = {k: np.stack([r[k] for r in estimator_records]) for k in estimator_records[0]}
    stage_names = np.array([protocol(i*env.dt)[0] for i in range(args.steps)])
    limits = env.torque_limits.cpu().numpy()
    valid = data['label_valid']
    def summarize(mask):
        mask = mask & valid
        n = int(mask.sum())
        if not n: return {'samples': 0}
        f, m = data['force'][mask], data['moment'][mask]
        ratio = np.abs(data['torques'][mask]) / limits
        o = data['over_directions'][mask]
        return dict(samples=n,
            force_abs_mean=np.abs(f).mean(0).tolist(), force_abs_p95=np.quantile(np.abs(f), .95, axis=0).tolist(),
            force_abs_max=np.abs(f).max(0).tolist(), moment_abs_p95=np.quantile(np.abs(m), .95, axis=0).tolist(),
            tracking_abs_mean=np.abs(data['velocity'][mask]-data['reference'][mask]).mean(0).tolist(),
            mean_velocity=data['velocity'][mask].mean(0).tolist(), mean_reference=data['reference'][mask].mean(0).tolist(),
            over_fraction=o.mean(0).tolist(), over_any_fraction=float(o.any(-1).mean()),
            terminations=int(data['terminated'][mask].sum()),
            termination_reasons=dict(zip(TERMINATION_REASONS, data['reasons'][mask].sum(0).tolist())),
            internal_axial_abs_mean=float(np.abs(data['internal_axial_force'][mask]).mean()),
            internal_axial_abs_p95=float(np.quantile(np.abs(data['internal_axial_force'][mask]), .95)),
            torque_saturation_fraction=(ratio >= .99).mean(0).tolist(),
            torque_ratio_p95=np.quantile(ratio, .95, axis=0).tolist())
    overall = summarize(np.ones_like(valid))
    stage_stats = {name: summarize(np.broadcast_to((stage_names == name)[:, None], valid.shape))
                   for name in dict.fromkeys(stage_names)}
    case_stats = []
    for payload in (5., 15., 25.):
        for scenario in ('straight', 'left', 'right', 'reverse'):
            ids = np.array([c['payload_kg'] == payload and c['scenario'] == scenario for c in cases])
            mask = np.broadcast_to(ids, valid.shape)
            case_stats.append(dict(payload=payload, scenario=scenario,
                **summarize(mask), environments=int(ids.sum()),
                environments_with_failure=int(data['terminated'][:, ids].any(0).sum()),
                main=summarize(mask & (stage_names == 'main')[:, None]),
                brake=summarize(mask & (stage_names == 'brake')[:, None])))
    evalid = estimates['valid']
    absolute = np.abs(estimates['prediction']-estimates['label'])[evalid]
    speed_error = np.abs(estimates['speed_prediction']-estimates['speed_label'])[estimates['speed_valid']]
    event_rows = []
    for t, i in np.argwhere(data['terminated']):
        event_rows.append(dict(step=int(t), env=int(i), time=float(data['time'][t, i]), stage=str(stage_names[t]),
            case=cases[i], episode_id=int(data['episode_id'][t,i]),
            reason=[name for j,name in enumerate(TERMINATION_REASONS) if data['reasons'][t,i,j]],
            force=data['force'][t,i].tolist(), moment=data['moment'][t,i].tolist(),
            force_mean=data['force_mean'][t,i].tolist(), moment_norm=data['moment_norm_used'][t,i].tolist(),
            thresholds=data['thresholds'][t,i].tolist(), over_directions=data['over_directions'][t,i].tolist(),
            velocity=data['velocity'][t,i].tolist(), reference=data['reference'][t,i].tolist(),
            internal_axial_force=float(data['internal_axial_force'][t,i]),
            torques=data['torques'][t,i].tolist(), robot_wrench=data['robot_wrench'][t,i].tolist(),
            cart_wrench=data['cart_wrench'][t,i].tolist(), stall=bool(data['stall'][t,i])))
    assert len(event_rows) == int(data['terminated'].sum())
    result = dict(result='PASS_EXECUTION', run_id=os.environ['V53_GRASP_AUDIT_CHILD'],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        iteration=policy.iteration, mode=args.mode, level=args.level, seed=args.seed, steps=args.steps,
        num_envs=args.num_envs, control_dt=env.dt, physics_dt=cfg.sim.dt, cases=cases,
        initial_parameters=snapshots, overall=overall, stages=stage_stats, case_statistics=case_stats,
        estimator=dict(valid_samples=int(evalid.sum()), mae=absolute.mean(0).tolist(),
            absolute_p95=np.quantile(absolute,.95,axis=0).tolist(),
            within_2_units=(absolute<=2).mean(0).tolist(), within_5_units=(absolute<=5).mean(0).tolist(),
            speed_mae=speed_error.mean(0).tolist()), events=event_rows,
        joint_names=env.dof_names, torque_limits=limits.tolist(),
        termination_config=class_to_dict(task.termination), level_params=env.level_params.as_dict(),
        randomization_effective=env.task_randomization_enabled, noise_effective=cfg.noise.add_noise,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__).resolve(), ROOT/'HomieRL/homie_pushcart_v53/task_env.py',
             ROOT/'HomieRL/homie_pushcart_v53/interaction_policy.py']},
        interpretation=['Execution success is not policy success; inspect termination and tracking statistics.',
            'Fresh 24 s global protocol; after failures the environment resets and rejoins current protocol phase.',
            'Payloads are fixed operating points; full mode retains other L13 training randomization.',
            'Control-rate physical samples are pre-reset; torques are last-substep drive efforts, not all-substep maxima.',
            'Estimator pairs are at pre-step time; physical traces are at post-step time, offset by 20 ms.',
            'Nominal replicas are not independent stochastic trials.',
            'Hand labels are model-based subtree reaction proxies; not independently calibrated contact sensors.',
            'Internal axial force is the equal/opposite force along the palm-to-palm line; other differences can produce useful yaw.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **data, stages=stage_names,
                        **{'est_'+k:v for k,v in estimates.items()})
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print('[trained grasp] RESULT', json.dumps(dict(terminations=overall['terminations'],
          tracking=overall['tracking_abs_mean'], estimator_mae=result['estimator']['mae'])), flush=True)
except BaseException:
    import traceback
    error = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(error=error)), encoding='utf8')
    print(error, flush=True)
    raise
finally:
    launcher.app.close()
