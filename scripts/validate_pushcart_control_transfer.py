"""Replay identical joint targets and a known cart load in both engines.

This bypasses both learned policies during the probe. It checks the physical
response to a common input, not force-estimator accuracy. The frozen policies,
training drive and original reports are preserved. Use --engine compare after
running each engine in each drive mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
from eval_pushcart_v52_sim2sim import DEFAULT_CKPT


def inputs(preset, hold, step, dt):
    import numpy as np
    target = preset.copy()
    target[12:] = hold
    t = step * dt
    # A smooth, 0.02 rad pulse in the left shoulder roll joint.
    if .2 <= t < .6:
        target[14] += .02 * np.sin(np.pi * (t-.2) / .4)**2
    # Same world force at the cart BASE BODY COM, with zero applied couple.
    force = np.array([15., 0., 0.]) if .7 <= t < .9 else np.zeros(3)
    return target, force


def save(args, rows, metadata):
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    name = f'probe_{args.engine}_{args.upper_drive}'
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    np.savez_compressed(args.output / (name + '.npz'), **arrays)
    metadata.update(steps=len(rows['q']), requested_steps=args.steps, checkpoint=str(args.checkpoint),
                    note='No learned actions. Fixed joint targets plus known force at cart base COM. Free base, feet, wheels and two grasps remain active.')
    (args.output / (name + '.json')).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(name, 'steps', metadata['steps'], 'failure', metadata.get('failure'), flush=True)


def run_mujoco(args):
    import numpy as np
    from pushcart_bench.policies_v5x import InstrumentedEnv
    from pushcart_bench.sim2sim_scene import set_payload, model_report
    from homie_pushcart_v52.control_contract import nominal_configs
    cfg, _ = nominal_configs(args.checkpoint, args.upper_drive)
    e = InstrumentedEnv(True, scene_profile='isaac', cfg=cfg, decimation=cfg.control.decimation,
                        upper_drive=args.upper_drive)
    set_payload(e, 0.)
    e.reset()
    rows = {k: [] for k in ('target', 'force', 'q', 'dq', 'tau', 'hand_world', 'robot_position', 'cart_position')}
    hold = e.preset_dof_pos[12:] + e.upper_hold_offset
    failure = None
    for step in range(args.steps):
        target, force = inputs(e.preset_dof_pos, hold, step, e.dt)
        e.upper_delta[:] = (target[12:] - hold) / e.action_scale
        e.inject_cart_wrench(force, np.zeros(3), e.dt)
        _, fallen = e.step((target[:12] - e.default_dof_pos[:12]) / e.action_scale)
        values = dict(target=e.joint_pos_target.copy(), force=force, q=e.dof_pos.copy(), dq=e.dof_vel.copy(),
                      tau=e.upper_torque_proxy.copy(), hand_world=np.asarray(e.sub_hands).mean(0),
                      robot_position=e.data.qpos[e.root_qpos:e.root_qpos+3].copy(),
                      cart_position=e.data.xpos[e.cart_id].copy())
        for k, v in values.items():
            rows[k].append(v)
        if fallen:
            failure = 'fall'
            break
    save(args, rows, dict(engine='mujoco', control=model_report(e)['control'], failure=failure))


def run_isaac(args):
    import numpy as np
    import torch
    from homie_pushcart_v52.control_contract import nominal_configs, control_contract
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.sim_env import PushCartSimEnv
    torch.manual_seed(0)
    cfg, task = nominal_configs(args.checkpoint, args.upper_drive)
    checkpoint = getattr(cfg, 'pushcart_saved_checkpoint', None)
    if checkpoint:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
    e = PushCartEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=checkpoint)
    e.fixed_payload_kg, e.fixed_caster_angle, e.fixed_command = 0., 0., [0., 0., 0.]
    with torch.inference_mode():
        e.reset()
    rows = {k: [] for k in ('target', 'force', 'q', 'dq', 'tau', 'hand_world', 'robot_position', 'cart_position')}
    preset, hold = e.preset_dof_pos[0].cpu().numpy(), e.hold_target_q[0].cpu().numpy()
    failure = None
    for step in range(args.steps):
        target, force = inputs(preset, hold, step, e.dt)
        with torch.inference_mode():
            e.upper_delta[:] = torch.tensor((target[12:] - hold) / cfg.control.action_scale, device=args.device)
            e.cart.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor(force, dtype=torch.float32, device=args.device).reshape(1, 1, 3),
                torques=torch.zeros(1, 1, 3, device=args.device), body_ids=e._cart_pulse_body_ids, is_global=True)
            leg = (torch.tensor(target[:12], device=args.device)[None] - e.default_dof_pos[:, :12]) / cfg.control.action_scale
            _, _, _, done, _, _, _ = PushCartSimEnv.step(e, leg)
            if bool(done[0]):
                failure = 'Isaac termination/reset (terminal sample excluded)'
                break
            f, m, _, _ = e.grasp_wrench_interval_mean()
            values = dict(target=e.joint_pos_target[0], force=force, q=e.dof_pos[0], dq=e.dof_vel[0],
                          tau=e.torques[0, 12:], hand_world=torch.cat((f[0], m[0]), -1),
                          robot_position=e.root_states[0, :3], cart_position=e.cart_state()[0][0])
            for k, v in values.items():
                rows[k].append(v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else v.copy())
    contract = control_contract(cfg, e.dof_names, e.torque_limits.cpu().tolist(), e.upper_drive)
    save(args, rows, dict(engine='isaac_physx', control=contract, failure=failure))


def compare(args):
    import numpy as np
    report = {}
    for mode in ('training', 'explicit_pd'):
        paths = [args.output / f'probe_{engine}_{mode}' for engine in ('isaac', 'mujoco')]
        if not all(p.with_suffix('.npz').is_file() for p in paths):
            continue
        a, b = [np.load(p.with_suffix('.npz')) for p in paths]
        ma, mb = [json.loads(p.with_suffix('.json').read_text(encoding='utf-8')) for p in paths]
        contract_equal = ma['control'] == mb['control']
        n = min(len(a['q']), len(b['q']))
        if not n:
            raise ValueError('No common nonterminal samples')
        diff = lambda key: a[key][:n] - b[key][:n]
        rms = lambda value: np.sqrt(np.mean(value**2, axis=0)).tolist()
        report[mode] = dict(samples=n, nominal_contract_equal=contract_equal,
                            max_target_difference_rad=float(np.max(abs(diff('target')))),
                            max_input_force_difference_n=float(np.max(abs(diff('force')))),
                            joint_position_rmse_rad=rms(diff('q')), joint_velocity_rmse_rad_s=rms(diff('dq')),
                            upper_torque_observation_rmse_nm=rms(diff('tau')), hand_wrench_rmse=rms(diff('hand_world')),
                            robot_position_rmse_m=rms(diff('robot_position')), cart_position_rmse_m=rms(diff('cart_position')),
                            note='Trajectory differences under common targets; this is not paired-state estimator evaluation.')
        if not contract_equal or report[mode]['max_target_difference_rad'] > 1e-6 or report[mode]['max_input_force_difference_n'] > 1e-6:
            raise AssertionError(f'{mode}: input contract mismatch; do not interpret response metrics')
    (args.output / 'control_response_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('mujoco', 'isaac', 'compare'), required=True)
    p.add_argument('--upper_drive', choices=('training', 'explicit_pd'), default='training')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--output', type=Path, default=ROOT / 'logs/bench/v52_control_alignment')
    preliminary, _ = p.parse_known_args()
    if preliminary.engine == 'isaac':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(p)
        args = p.parse_args()
        app = AppLauncher(args)
        try:
            run_isaac(args)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            app.app.close()
    else:
        args = p.parse_args()
        (run_mujoco if args.engine == 'mujoco' else compare)(args)


if __name__ == '__main__':
    main()
