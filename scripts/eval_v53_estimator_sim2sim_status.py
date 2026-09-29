"""Evaluate a fixed V5.3 checkpoint on the existing nominal MuJoCo deployment.

Matches the 12 unique payload/command cases in the Isaac trained-grasp audit.
Does not alter training, weights, observations, plant parameters, or grasp tuning.
Estimator samples are taken before each action, paired with the preceding full
physics interval, matching the Isaac audit's estimator timing. Tracking samples
are taken after each action. This runner stops on falls; it does not reproduce
Isaac's full task termination/reset supervisor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]


def protocol(t, scenario):
    if t < 2:
        return 'settle', np.zeros(3)
    if t < 6:
        return 'start', np.array([-.2 if scenario == 'reverse' else .3, 0., 0.])
    if t < 12 or 16 <= t < 21:
        vx = .6 if scenario == 'straight' else (-.35 if scenario == 'reverse' else .35)
        wz = .25 if scenario == 'left' else (-.25 if scenario == 'right' else 0.)
        return ('main' if t < 12 else 'restart_turn_reverse'), np.array([vx, 0., wz if t < 12 else -wz])
    return ('brake' if t < 16 else 'final_stop'), np.zeros(3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(prediction, label):
    e = np.asarray(prediction, dtype=np.float64) - np.asarray(label, dtype=np.float64)
    if not len(e):
        return {'samples': 0}
    if not np.isfinite(e).all():
        raise FloatingPointError('Nonfinite estimation or tracking error')
    return dict(samples=len(e), mae=np.abs(e).mean(0).tolist(),
                rmse=np.sqrt(np.square(e).mean(0)).tolist(), bias=e.mean(0).tolist(),
                absolute_p95=np.quantile(np.abs(e), .95, axis=0).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--scene_version', choices=('original', 'four_spheres', 'new'), default='original')
    a = p.parse_args()
    if a.steps < 3:
        p.error('Need at least three control steps')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    from pushcart_bench.policies_v53 import V53Runner
    import mujoco
    before = sha(a.checkpoint)
    started = time.time()
    r = V53Runner(a.checkpoint, level=13, scene_version=a.scene_version)
    cases, aggregate = [], {k: [] for k in ('force_prediction', 'force_label', 'velocity_prediction', 'velocity_label',
                                           'tracking_velocity', 'tracking_reference')}
    for payload in (5., 15., 25.):
        for scenario in ('straight', 'left', 'right', 'reverse'):
            name = f'payload_{payload:g}_{scenario}'
            r.set_payload(payload)
            r.reset(caster_angle=0.)
            if not np.isclose(r.cart_mass_total, 14.4 + payload, atol=1e-6):
                raise ValueError('Unexpected cart mass')
            d = {k: [] for k in (*aggregate, 'estimator_time', 'tracking_time', 'command', 'stages',
                                 'last_hand_world', 'cart_tilt', 'robot_tilt', 'wheel_lift', 'substep_count')}
            replay_errors = []
            fallen = False
            for step in range(a.steps):
                stage, command = protocol(step * r.dt, scenario)
                # Exclude two initialization samples exactly as in the Isaac audit.
                if step >= 2:
                    if len(r.env.sub_hands) != r.env.decimation:
                        raise ValueError('Hand label is not a complete control interval')
                    world = np.mean(r.env.sub_hands, axis=0)
                    c, s = np.cos(r.ts['cart_yaw']), np.sin(r.ts['cart_yaw'])
                    rotation = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
                    label = np.concatenate((world[:, :3] @ rotation.T, world[:, 3:] @ rotation.T), axis=-1).reshape(12)
                    for k, value in (
                        ('force_prediction', r.pred_hands), ('force_label', label),
                        ('velocity_prediction', r.pred_cart_velocity),
                        ('velocity_label', np.r_[r.ts['cart_v'], r.ts['cart_wz']]),
                        ('estimator_time', step * r.dt), ('substep_count', len(r.env.sub_hands))):
                        d[k].append(np.asarray(value).copy())
                    if step == 2:
                        old_f, old_v = r.pred_hands.copy(), r.pred_cart_velocity.copy()
                        r._predict()
                        replay_errors.append(max(float(np.max(np.abs(old_f - r.pred_hands))),
                                                 float(np.max(np.abs(old_v - r.pred_cart_velocity)))))
                _, fallen = r.step(command)
                for k, value in (
                    ('tracking_velocity', np.r_[r.ts['cart_v'], r.ts['cart_wz']]),
                    ('tracking_reference', r.cmd_ref), ('tracking_time', (step + 1) * r.dt),
                    ('command', command), ('stages', stage), ('last_hand_world', r.env.sub_hands[-1]),
                    ('cart_tilt', r.ts['cart_tilt']), ('robot_tilt', r.ts['robot_tilt_rad']),
                    ('wheel_lift', r.ts['wheel_lift_m'])):
                    d[k].append(np.asarray(value).copy())
                if fallen:
                    break
            arrays = {k: np.stack(v) if len(v) else np.array([]) for k, v in d.items()}
            for k, v in arrays.items():
                if v.dtype.kind != 'U' and not np.isfinite(v).all():
                    raise FloatingPointError(f'Nonfinite {name}/{k}')
            np.savez_compressed(a.output / f'{name}.npz', **arrays)
            result = dict(name=name, payload_kg=payload, total_cart_mass_kg=r.cart_mass_total,
                          scenario=scenario, steps=step + 1, fall=bool(fallen),
                          force=metrics(arrays['force_prediction'], arrays['force_label']),
                          speed=metrics(arrays['velocity_prediction'], arrays['velocity_label']),
                          tracking=metrics(arrays['tracking_velocity'], arrays['tracking_reference']),
                          prediction_repeat_max_abs=max(replay_errors, default=None))
            for key in aggregate:
                aggregate[key].append(arrays[key])
            cases.append(result)
            (a.output / 'progress.json').write_text(json.dumps(cases, indent=2), encoding='utf8')
            print(json.dumps(dict(case=name, steps=step + 1, fall=bool(fallen), force_mae=result['force'].get('mae'),
                                  speed_mae=result['speed'].get('mae'))), flush=True)
    joined = {k: np.concatenate(v) for k, v in aggregate.items()}
    after = sha(a.checkpoint)
    assert before == after
    report = dict(result='PASS_EXECUTION', checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=before,
                  lowlevel_checkpoint=str(r.homie_checkpoint), lowlevel_sha256=sha(r.homie_checkpoint),
                  engine='MuJoCo', mujoco_version=mujoco.__version__, level=13, control_dt=r.dt,
                  physics_dt=float(r.env.model.opt.timestep), observation_noise=False, physical_randomization=False,
                  deterministic_actions=True, requested_steps=a.steps, cases=cases,
                  complete_cases=sum(c['steps'] == a.steps and not c['fall'] for c in cases),
                  force=metrics(joined['force_prediction'], joined['force_label']),
                  speed=metrics(joined['velocity_prediction'], joined['velocity_label']),
                  tracking=metrics(joined['tracking_velocity'], joined['tracking_reference']),
                  scene=dict(version=r.scene_version, xml=str(r.env.xml_path), profile='isaac parameter profile in MuJoCo',
                             grasp='nominal', inertia='urdf', upper_drive='explicit_pd'),
                  limitation='Runner checks falls, not the complete Isaac task termination supervisor. Same command/payload program; cross-engine physical and label equivalence is not assumed.',
                  source_sha256={str(f.relative_to(ROOT)): sha(f) for f in (Path(__file__).resolve(), ROOT/'pushcart_bench/policies_v53.py', ROOT/'pushcart_bench/policies_v5x.py', ROOT/'pushcart_bench/sim2sim_scene.py')},
                  elapsed_s=time.time() - started)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(dict(result=report['result'], complete_cases=report['complete_cases'], elapsed_s=report['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
