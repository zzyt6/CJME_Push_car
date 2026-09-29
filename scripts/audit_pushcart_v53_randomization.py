"""Read back actual PhysX domain properties, then run a bounded frozen-HOMIE rollout."""
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
DEFAULT_OUTPUT = ROOT/'logs/bench/v53_domain_randomization/full.json'
if not os.environ.get('PUSHCART_V53_DR_CHILD'):
    token = uuid.uuid4().hex
    child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=dict(os.environ, PUSHCART_V53_DR_CHILD=token))
    if '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(child.returncode)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    a, _ = p.parse_known_args()
    r = json.loads(a.output.read_text(encoding='utf-8')) if a.output.is_file() else {}
    success = child.returncode == 0 and r.get('run_id') == token and r.get('result') == 'PASS_EXECUTION'
    if r.get('run_id') == token:
        r.update(child_exit_code=child.returncode, parent_verified=success)
        a.output.write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(f'[DR audit parent] child_exit={child.returncode}, verified={success}', flush=True)
    raise SystemExit(0 if success else 1)

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=('full', 'initial', 'nominal'), default='full')
p.add_argument('--num_envs', type=int, default=32)
p.add_argument('--steps', type=int, default=400)
p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)

try:
    import torch
    import warp as wp
    from homie_isaaclab.math_utils import class_to_dict
    from homie_pushcart_v53.env_smoke import make_configs
    from homie_pushcart_v53.task_env import PushCartEnv
    from homie_pushcart_v53.domain_randomization import loaded_cart_properties

    torch.manual_seed(5303)
    level = 0 if args.mode == 'initial' else 17
    enabled = args.mode != 'nominal'
    cfg, task = make_configs(args.num_envs, enabled, level=level, episode_length_s=20.)
    task.curriculum.enabled = False
    task.diagnostics.finite_check_interval_steps = 1
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=level)
    env.fixed_command = [.25, 0., 0.]
    ids = torch.arange(args.num_envs, device=args.device, dtype=torch.int32)
    snapshots = []
    max_errors = dict(mass=0., inertia=0., com_position=0., damping=0., material=0., cache_inertia=0.)

    def readback():
        actual_m = wp.to_torch(env.cart.root_view.get_masses()).to(args.device).clone()
        actual_i = wp.to_torch(env.cart.root_view.get_inertias()).to(args.device).clone()
        actual_c = wp.to_torch(env.cart.root_view.get_coms()).to(args.device).clone()
        expected_m, expected_i, expected_c = loaded_cart_properties(
            env.cart_default_mass, env.cart_default_inertia, env.cart_default_com, env.cart_base_body,
            env.cart_mass_scale, env.cart_payload_kg[:, None], env.payload_com)
        damping = wp.to_torch(env.cart.root_view.get_dof_dampings()).to(args.device).clone()
        expected_d = torch.zeros_like(damping)
        expected_d[:, env.wheel_joint_ids] = env.wheel_damping_per_wheel
        expected_d[:, env.swivel_joint_ids] = env.swivel_damping_per_wheel
        materials = wp.to_torch(env.cart.root_view.get_material_properties()).to(args.device).clone()
        pairs = dict(mass=(actual_m, expected_m), inertia=(actual_i, expected_i),
            com_position=(actual_c[..., :3], expected_c[..., :3]), damping=(damping, expected_d),
            material=(materials[..., :2], env.wheel_friction[:, None, None].expand_as(materials[..., :2])),
            cache_inertia=(env._dynamics_property_cache[id(env.cart)]['inertia'],
                           actual_i.reshape(args.num_envs, -1, 3, 3).transpose(-1, -2)))
        for key, (actual, expected) in pairs.items():
            max_errors[key] = max(max_errors[key], float((actual-expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6)
        eig = torch.linalg.eigvalsh(actual_i.reshape(args.num_envs, -1, 3, 3))
        assert (actual_m > 0).all() and (eig > 0).all()
        assert (eig[..., 2] <= eig[..., :2].sum(-1)+1e-5).all()
        if not enabled:
            assert (env.cart_mass_scale == 1).all() and not env.cart_payload_kg.any()
            assert not env.payload_com.any() and not env.hl_delay.any() and not env.joint_torque_delay.any()
            assert (env.Kp_factors == 1).all() and (env.Kd_factors == 1).all()
            assert not env.heading_sensor.delay.any() and not env.heading_sensor.bias.any()
            assert not cfg.domain_rand.push_robots and not env.task_randomization_enabled
        else:
            assert .9 <= env.Kp_factors.min() and env.Kp_factors.max() <= 1.1
            assert .9 <= env.Kd_factors.min() and env.Kd_factors.max() <= 1.1
        snapshots.append(dict(mass_scale=env.cart_mass_scale.cpu().tolist(),
            total_mass=env.cart_mass_total.cpu().tolist(), payload=env.cart_payload_kg.cpu().tolist(),
            payload_com_xy=env.payload_com.cpu().tolist(), wheel_friction=env.wheel_friction.cpu().tolist(),
            wheel_damping=env.wheel_damping_per_wheel.cpu().tolist(), swivel_damping=env.swivel_damping_per_wheel.cpu().tolist(),
            kp_factors=env.Kp_factors.cpu().tolist(), kd_factors=env.Kd_factors.cpu().tolist(),
            heading_delay=env.heading_sensor.delay.cpu().tolist(), heading_bias=env.heading_sensor.bias.cpu().tolist(),
            hl_delay=env.hl_delay.cpu().tolist(), torque_delay=env.joint_torque_delay.cpu().tolist()))
        return actual_m, actual_i, actual_c, damping, materials

    for repeat in range(8):
        env.reset()
        readback()
    # Partial reset must never overwrite the other environment rows' plant parameters.
    before = readback()
    env.reset_idx(ids[:args.num_envs//2])
    after = readback()
    for a, b in zip(before, after):
        torch.testing.assert_close(a[args.num_envs//2:], b[args.num_envs//2:], rtol=0, atol=0)
    # A stalled caster returns to its own sampled damping, not to a shared mean.
    original_prob = task.randomization.wheel_stall_prob_per_s
    task.randomization.wheel_stall_prob_per_s = 0.
    abnormal = env.wheel_damping_per_wheel[:1].clone()*20
    env.cart.write_joint_damping_to_sim_index(damping=abnormal, joint_ids=env.wheel_joint_ids_sim, env_ids=ids[:1])
    env.stall_timer[0] = 1
    env._apply_cart_disturbances()
    restored = wp.to_torch(env.cart.root_view.get_dof_dampings()).to(args.device)[:, env.wheel_joint_ids]
    torch.testing.assert_close(restored[0], env.wheel_damping_per_wheel[0])
    task.randomization.wheel_stall_prob_per_s = original_prob
    env.reset()
    # Test the auto-pulse sampler's equivalent impulse scale; it is a force, never a qvel jump.
    env.cart_push_timer.fill_(1)
    env._apply_cart_disturbances()
    duration = round(task.randomization.cart_force_duration_s/cfg.sim.dt)*cfg.sim.dt
    equivalent_dv = env.cart_pulse_wrench[:, :2]*duration/env.cart_mass_total[:, None]
    if args.mode == 'full':
        assert equivalent_dv.abs().max() > .1 and equivalent_dv.abs().max() <= .500001
    else:
        assert not env.cart_pulse_wrench.any()
    pulse_check = equivalent_dv.cpu().tolist()
    env.reset()
    resets = valid_labels = numerical_faults = 0
    pulse_samples = stall_samples = 0
    max_force_axes = torch.zeros(2, 3, device=args.device)
    for step in range(args.steps):
        obs, critic, rew, done, info = env.step(torch.zeros(args.num_envs, 18, device=args.device))
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all() and torch.isfinite(rew).all()
        assert torch.isfinite(env.interaction_target).all() and torch.isfinite(env.cart_velocity_target).all()
        resets += int(done.sum())
        valid_labels += int((env.interaction_target[:, 12] > .5).sum())
        numerical_faults += int(env.term_reason_buf['blowup'].sum())
        pulse_samples += int((env.cart_applied_wrench.abs().sum(-1) > 0).sum())
        stall_samples += int((env.stall_timer > 0).sum())
        max_force_axes = torch.maximum(max_force_axes, env.ts.force_w_raw.abs().amax(0))
        if not enabled:
            assert not env.cart_applied_wrench.any() and not env.applied_robot_wrench.any()
            assert not env.stall_timer.any() and not env.hl_delay.any() and not env.joint_torque_delay.any()
        if step % 100 == 0:
            print(f'[DR audit] {args.mode} {step}/{args.steps}, resets={resets}', flush=True)
    import numpy as np
    keys = snapshots[0].keys()
    ranges = {k: [float(np.min([s[k] for s in snapshots])), float(np.max([s[k] for s in snapshots]))] for k in keys}
    assert valid_labels > 0
    source_paths = ['domain_randomization.py', 'task_config.py', 'task_env.py', 'observation.py', 'heading_sensor.py']
    report = dict(result='PASS_EXECUTION', run_id=os.environ['PUSHCART_V53_DR_CHILD'],
        version=53, implementation_revision=task.implementation_revision, mode=args.mode, level=level,
        num_envs=args.num_envs, steps=args.steps, seed=5303,
        property_snapshots=len(snapshots), ranges=ranges, max_readback_error=max_errors,
        partial_reset_preserved_other_rows=True, asymmetric_stall_restore_verified=True,
        equivalent_cart_pulse_dv=pulse_check, resets=resets, numerical_blowup_terminations=numerical_faults,
        valid_hand_labels=valid_labels, auto_cart_pulse_samples=pulse_samples, wheel_stall_samples=stall_samples,
        max_abs_hand_force_world_axes=max_force_axes.cpu().tolist(),
        randomization=env.randomization_metadata,
        robot_domain_config=class_to_dict(cfg.domain_rand), observations_config=class_to_dict(task.observations),
        source_sha256={f'HomieRL/homie_pushcart_v53/{f}': hashlib.sha256((ROOT/'HomieRL/homie_pushcart_v53'/f).read_bytes()).hexdigest() for f in source_paths},
        note='Readback/execution check with frozen HOMIE and zero residual; resets are not a trained-policy success rate. No deployment guarantee.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    args.output.with_suffix('.samples.json').write_text(json.dumps(snapshots), encoding='utf-8')
    print(json.dumps(dict(mode=args.mode, ranges=ranges, errors=max_errors, resets=resets,
                          numerical_blowup_terminations=numerical_faults)), flush=True)
except BaseException:
    import traceback
    error = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(result='FAIL', error=error), indent=2), encoding='utf-8')
    print(error, flush=True)
    raise
finally:
    launcher.app.close()
