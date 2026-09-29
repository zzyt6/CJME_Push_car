"""Fresh Isaac trajectories: reward magnitudes and force-guard coverage, not a training comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))
DEFAULT_OUTPUT = ROOT/'logs/bench/v53_reward_audit/report.json'

if not os.environ.get('PUSHCART_V53_REWARD_CHILD'):
    run_id = uuid.uuid4().hex
    child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=dict(os.environ, PUSHCART_V53_REWARD_CHILD=run_id))
    if '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(child.returncode)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    known, _ = probe.parse_known_args()
    report = json.loads(known.output.read_text(encoding='utf-8')) if known.output.is_file() else {}
    success = child.returncode == 0 and report.get('run_id') == run_id and report.get('result') == 'PASS_EXECUTION'
    if report.get('run_id') == run_id:
        report['child_exit_code'] = child.returncode
        report['parent_verified'] = success
        known.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[reward audit parent] child_exit={child.returncode}, verified={success}', flush=True)
    raise SystemExit(0 if success else 1)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8)
parser.add_argument('--steps', type=int, default=1100)
parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

try:
    import numpy as np
    import torch
    from homie_isaaclab.math_utils import class_to_dict
    from homie_pushcart_v53.cart_quality import apply_quality_profile
    from homie_pushcart_v53.env_smoke import make_configs
    from homie_pushcart_v53.task_env import PushCartEnv

    torch.manual_seed(53)
    cfg, task = make_configs(args.num_envs, False, level=8, episode_length_s=30.)
    apply_quality_profile(task, 'v53_wrench_guarded')
    task.curriculum.enabled = False
    task.quality.reward_ramp_s = 0.  # Audit final strength, never hide contributions behind the 60 s ramp.
    task.diagnostics.finite_check_interval_steps = 1
    task.randomization.cart_force_pulse = False
    task.randomization.cart_push_vel_max = 0.
    task.randomization.wheel_stall_prob_per_s = 0.
    task.randomization.robot_push = False
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=8)
    env.fixed_command = [.3, 0., 0.]
    env.fixed_caster_angle = 0.
    env.reset()
    records, term_values, weighted = [], {}, {}
    # Record actual reward-function return values at their existing invocation; never
    # evaluate stateful rewards a second time and never sample after automatic reset.
    def record_term(name, fn):
        def wrapped():
            value = fn()
            term_values[name] = value.detach().clone()
            return value
        return wrapped
    env.reward_functions = [record_term(n, fn) for n, fn in zip(env.reward_names, env.reward_functions)]
    compute_reward = env.compute_reward
    stage = 'start_cruise'
    step_index = 0

    def audited_compute_reward():
        compute_reward()
        q = env.cart_quality
        if 'termination' in env.reward_scales:
            term_values['termination'] = env.terminated_buf.float()/env.dt
        reconstructed = sum(v*env.reward_scales[n] for n, v in term_values.items())
        torch.testing.assert_close(env.rew_buf, reconstructed, atol=1e-6, rtol=1e-5)
        values = {n: (v*env.reward_scales[n]/env.dt).cpu().numpy() for n, v in term_values.items()}
        for n, v in values.items():
            weighted.setdefault(n, []).append(v)
        records.append(dict(stage=stage,
            velocity=torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1).cpu().numpy(),
            reference=env.cmd_ref.cpu().numpy().copy(), command=env.cmd_task.cpu().numpy().copy(),
            error_band=q.velocity_band.cpu().numpy().copy(), axis_cost=q.axis_ripple_cost.cpu().numpy().copy(),
            response_cost=q.response_cost.cpu().numpy().copy(), wrench_band=q.wrench_band.cpu().numpy().copy(),
            wrench_cost=q.wrench_cost.cpu().numpy().copy(), guarded_cost=q.guarded_wrench_cost.cpu().numpy().copy(),
            eligible=q.guard_eligible.cpu().numpy().copy(), gate=q.wrench_gate.cpu().numpy().copy(),
            ready=q.ready.cpu().numpy().copy(), filtered_velocity=env.cart_vel_f.cpu().numpy().copy(),
            filtered_reference=env.cmd_ref_f.cpu().numpy().copy(), done=env.reset_buf.cpu().numpy().copy(),
            total_reward_per_step=env.rew_buf.cpu().numpy().copy()))
        assert torch.isfinite(env.rew_buf).all()
        assert (q.guarded_wrench_cost >= 0).all() and (q.guarded_wrench_cost <= task.quality.cost_cap).all()
        if stage in ('stop', 'turn', 'turn_reverse'):
            assert not q.wrench_gate.any(), f'force penalty active during {stage}'
        if step_index == 250:  # First explicitly injected pulse step.
            assert not q.wrench_gate.any(), 'force penalty active during external pulse'

    env.compute_reward = audited_compute_reward
    protocol = [(0., 'start_cruise', [.3, 0., 0.]), (5., 'pulse_recovery', [.3, 0., 0.]),
                (8., 'stop', [0., 0., 0.]), (10., 'restart', [.3, 0., 0.]),
                (13., 'turn', [.3, 0., .2]), (16., 'turn_reverse', [.3, 0., -.2]),
                (19., 'reverse_translation', [-.2, 0., 0.])]
    for step_index in range(args.steps):
        seconds = step_index*env.dt
        _, stage, command = next(item for item in reversed(protocol) if seconds >= item[0])
        env.fixed_command = command
        env.cmd_task[:] = env.cmd_task.new_tensor(command)
        env.cmd_timer.fill_(1000000)
        if step_index == 250:
            force = torch.zeros(args.num_envs, 3, device=args.device)
            force[:, 0] = -8.
            env.inject_cart_wrench(force, torch.zeros_like(force), .2)
        env.step(torch.zeros(args.num_envs, 18, device=args.device))
        if step_index % 200 == 0:
            print(f'[reward audit] {step_index}/{args.steps}: {stage}', flush=True)

    data = {key: np.stack([r[key] for r in records]) for key in records[0] if key != 'stage'}
    stages = np.array([r['stage'] for r in records])
    weighted = {n: np.stack(v) for n, v in weighted.items()}
    assert data['gate'].max() > 0, 'No eligible cruise sampled; guard not exercised'
    assert data['ready'].any(), 'No post-warmup quality samples'
    positive = sum(weighted.get(n, 0.) for n in ('track_cart_vx', 'track_cart_vy', 'track_cart_wz', 'track_progress'))
    motion_cost = -sum(weighted.get(n, 0.) for n in ('cart_vx_ripple', 'cart_vy_ripple', 'cart_yaw_ripple', 'cart_response_error'))
    stage_stats = {}
    for name in dict.fromkeys(stages):
        mask = stages == name
        ready = data['ready'][mask]
        cap = data['axis_cost'][mask] >= task.quality.cost_cap - 1e-6
        stage_stats[name] = dict(samples=int(mask.sum()*args.num_envs),
            tracking_error_abs_mean=np.abs(data['velocity'][mask]-data['reference'][mask]).mean((0, 1)).tolist(),
            error_band_abs_mean=np.abs(data['error_band'][mask])[ready].mean(0).tolist() if ready.any() else None,
            eligible_fraction=float(data['eligible'][mask].mean()), gate_mean=float(data['gate'][mask].mean()),
            gate_max=float(data['gate'][mask].max()),
            axis_cap_fraction=cap[ready].mean(0).tolist() if ready.any() else None,
            reward_per_s={n: float(v[mask].mean()) for n, v in weighted.items()},
            resets=int(data['done'][mask].sum()))
    # Counterfactual revision-1 vs revision-2 motion shaping on the SAME physical traces.
    # These scores are not a before/after policy performance comparison.
    band = data['error_band']
    z = np.maximum(np.abs(band)-np.array([.005, .005, .01]), 0)/np.array([.05, .05, .10])
    huber = np.where(z <= 1, .5*z*z, z-.5)
    old_planar = np.minimum(huber[..., :2].mean(-1), 4)*data['ready']
    old_motion_cost = (.1*old_planar + .15*data['axis_cost'][..., 2] + .1*data['response_cost'])
    report = dict(result='PASS_EXECUTION', run_id=os.environ['PUSHCART_V53_REWARD_CHILD'],
        version=53, implementation_revision=task.implementation_revision, seed=53,
        num_envs=args.num_envs, steps=args.steps, dt=env.dt,
        policy='frozen HOMIE, zero high-level residual; not a trained V5.3 policy',
        randomization=False, quality_reward_ramp_s=0., profile='v53_wrench_guarded',
        protocol=protocol, quality=q.metadata() if (q := env.cart_quality) else None,
        reward_scales=class_to_dict(task.rewards.scales), stages=stage_stats,
        overall=dict(reward_per_s={n: float(v.mean()) for n, v in weighted.items()},
            tracking_plus_progress_mean=float(positive.mean()), motion_penalty_mean=float(motion_cost.mean()),
            motion_to_tracking_ratio=float(motion_cost.mean()/positive.mean()),
            revision1_motion_penalty_same_trace=float(old_motion_cost.mean()),
            guarded_force_penalty_mean=float(-weighted['cart_wrench_guarded'].mean()),
            unguarded_force_penalty_same_weight=float(.005*data['wrench_cost'].mean()),
            reset_events=int(data['done'].sum())),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'HomieRL/homie_pushcart_v53'/f for f in ('task_config.py', 'task_env.py', 'cart_quality.py')]},
        interpretation='Execution and reward-contribution audit only. No learning improvement, optimal weights, force accuracy or sim2sim success claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **data, stages=stages,
                        **{'reward_'+k:v for k,v in weighted.items()})
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['overall'], indent=2), flush=True)
except BaseException:
    import traceback
    error = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(result='FAIL', error=error), indent=2), encoding='utf-8')
    print(error, flush=True)
    raise
finally:
    launcher.app.close()
