"""Bounded no-cart, moving-arm known-load check of V5.1's actual 200 Hz labels.

Run with the project Isaac Python and --headless. Importing this file or using
--help does not launch Kit. This writes measurements, not calibration settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('static_zero', 'static_positive', 'static_negative',
          'dynamic_zero', 'dynamic_loaded', 'dynamic_release')
POINT_OFFSET_PALM_AXES_M = np.array([[.025, .012, .015], [.025, -.012, .015]])
FORCE_BASE_N = np.array([[5., 2., 1.], [4., -2., 1.]])
COUPLE_BASE_NM = np.array([[.10, -.08, .15], [-.10, .08, .15]])


def prescribed_load(phase: str, time_s: float, scale: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """World-axis known force and free couple; motion is independently excited."""
    if phase in ('static_positive', 'static_negative'):
        sign = 1. if phase == 'static_positive' else -1.
        amplitude = sign * min(1., max(0., time_s / .25)) * scale
        return FORCE_BASE_N * amplitude, COUPLE_BASE_NM * amplitude
    if phase == 'dynamic_loaded':
        # The initial load step and later square pulses remain in the analysis.
        # Distinct frequencies excite all axes; sharp transitions expose delay.
        pulse = 1. if (time_s % 1.2) < .6 else -1.
        f = np.array([.35 + .45 * math.sin(2 * math.pi * .7 * time_s) + .25 * pulse,
                      .25 + .65 * math.sin(2 * math.pi * 1.1 * time_s + .3),
                      .30 + .60 * math.sin(2 * math.pi * .45 * time_s + .7)])
        m = np.array([.4 + .6 * math.sin(2 * math.pi * .9 * time_s),
                      .3 + .5 * math.sin(2 * math.pi * 1.3 * time_s + .6),
                      .3 + .5 * math.sin(2 * math.pi * .6 * time_s + .9) + .15 * pulse])
        return FORCE_BASE_N * f * scale, COUPLE_BASE_NM * m * scale
    if phase not in PHASES:
        raise ValueError(f'Unknown label phase: {phase}')
    return np.zeros((2, 3)), np.zeros((2, 3))


def external_reaction_at_palm(force, torque_about_com, sampled_com, sampled_palm):
    """Hand-on-environment target, after shifting applied COM wrench to palm.

    PhysX integrates a force and a COM-centered torque over each substep. The
    submitted COM torque already contains the requested application-point lever.
    Use the sampled COM-to-palm displacement for the output's sampled reference.
    This avoids treating a pre-step point as an end-step rigid-body point.
    """
    moment = torque_about_com + np.cross(sampled_com - sampled_palm, force)
    return -np.concatenate((force, moment), axis=-1)


def lag_diagnostic(measured: np.ndarray, expected: np.ndarray, dt: float, max_lag_s: float = .05) -> dict:
    """Common-window vector correlation; positive lag means measured is delayed.

    No realignment is used for primary error metrics. Undefined/constant signals
    return None; a zero estimate cannot acquire a spurious successful zero lag.
    """
    y, x = np.asarray(measured, dtype=float), np.asarray(expected, dtype=float)
    if y.shape != x.shape or y.ndim != 2 or y.shape[1] != 3:
        raise ValueError('Lag inputs must be matching [time, 3] arrays')
    limit = min(int(round(max_lag_s / dt)), (len(y) - 3) // 2)
    undefined = {'lag_s': None, 'correlation': None, 'aligned_vector_rms': None}
    if limit < 1 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return undefined
    indices = np.arange(limit, len(y) - limit)
    reference = x[indices]
    centered_x = reference - reference.mean(0)
    xx = np.sum(centered_x ** 2)
    if xx < 1.e-10:
        return undefined
    scores = []
    for lag in range(-limit, limit + 1):
        shifted = y[indices + lag]
        centered_y = shifted - shifted.mean(0)
        denominator = math.sqrt(xx * np.sum(centered_y ** 2))
        if denominator < 1.e-10:
            continue
        score = float(np.sum(centered_x * centered_y) / denominator)
        error = float(np.sqrt(np.mean(np.sum((shifted - reference) ** 2, -1))))
        scores.append((score, -abs(lag), lag, error))
    if not scores:
        return undefined
    score, _, lag, error = max(scores)
    return {'lag_s': lag * dt, 'correlation': score, 'aligned_vector_rms': error}


def _pair_metrics(measured, expected):
    error = measured - expected
    f, m = np.linalg.norm(error[:, :3], axis=-1), np.linalg.norm(error[:, 3:], axis=-1)
    return {
        'samples': len(error), 'bias_6d_N_Nm': error.mean(0).tolist(),
        'axis_rmse_6d_N_Nm': np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        'force_vector_rms_N': float(np.sqrt(np.mean(f ** 2))),
        'force_vector_peak_N': float(f.max()), 'force_vector_p95_N': float(np.percentile(f, 95)),
        'moment_vector_rms_Nm': float(np.sqrt(np.mean(m ** 2))),
        'moment_vector_peak_Nm': float(m.max()), 'moment_vector_p95_Nm': float(np.percentile(m, 95)),
        'measured_mean_6d_N_Nm': measured.mean(0).tolist(),
        'expected_mean_6d_N_Nm': expected.mean(0).tolist(),
        'expected_force_vector_rms_N': float(np.sqrt(np.mean(np.sum(expected[:, :3] ** 2, -1)))),
        'expected_moment_vector_rms_Nm': float(np.sqrt(np.mean(np.sum(expected[:, 3:] ** 2, -1)))),
    }


def summarize_trace(trace: dict[str, np.ndarray], *, dt: float, resets: int = 0,
                    expected_samples_per_phase: int | None = None) -> dict:
    """Keep each environment and hand separate; report all excluded samples."""
    measured = np.asarray(trace['measured_wrench_w'], dtype=float)
    expected = np.asarray(trace['expected_wrench_w'], dtype=float)
    if measured.ndim != 4 or measured.shape[-2:] != (2, 6) or measured.shape != expected.shape:
        raise ValueError('Wrench traces must be matching [time, environment, 2, 6] arrays')
    phase = np.asarray(trace['phase'], dtype=int)
    valid = np.asarray(trace['valid'], dtype=bool)
    fault = np.asarray(trace['fault'], dtype=bool)
    if phase.shape != (len(measured),) or valid.shape != measured.shape[:2] or fault.shape != valid.shape:
        raise ValueError('Phase/validity shapes differ from the wrench trace')
    finite = np.isfinite(measured).all(axis=(2, 3)) & np.isfinite(expected).all(axis=(2, 3))
    used = finite & valid & ~fault
    phases = {}
    complete = True
    for phase_id, name in enumerate(PHASES):
        select = phase == phase_id
        count = int(select.sum())
        complete &= count > 0 and (expected_samples_per_phase is None or count == expected_samples_per_phase)
        pairs = []
        for env_id in range(measured.shape[1]):
            mask = select & used[:, env_id]
            hands = []
            for hand in range(2):
                if not mask.any():
                    hands.append(None)
                    continue
                stats = _pair_metrics(measured[mask, env_id, hand], expected[mask, env_id, hand])
                stats['error_impulse_6d_Ns_Nms'] = ((measured[mask, env_id, hand]
                                                   - expected[mask, env_id, hand]).sum(0) * dt).tolist()
                if name == 'dynamic_loaded' and mask.sum() == count:
                    stats['force_lag'] = lag_diagnostic(measured[mask, env_id, hand, :3],
                                                       expected[mask, env_id, hand, :3], dt)
                    stats['moment_lag'] = lag_diagnostic(measured[mask, env_id, hand, 3:],
                                                        expected[mask, env_id, hand, 3:], dt)
                # For static probes report a settled tail in addition to the
                # complete phase; load onset remains in the primary metrics.
                if name.startswith('static_'):
                    phase_indices = np.flatnonzero(select)
                    tail = mask.copy()
                    tail[:phase_indices[int(.6 * count)]] = False
                    if tail.any():
                        stats['settled_tail'] = _pair_metrics(measured[tail, env_id, hand], expected[tail, env_id, hand])
                hands.append(stats)
            pairs.append(hands)
        phases[name] = {'physical_samples': count,
                        'excluded_samples_by_environment': (select[:, None] & ~used).sum(0).tolist(),
                        'environment_hand': pairs}
        if count and 'upper_joint_pos_rad' in trace and np.isfinite(trace['upper_joint_pos_rad'][select]).all():
            phases[name]['upper_joint_peak_to_peak_rad'] = np.ptp(trace['upper_joint_pos_rad'][select], axis=0).tolist()
        if count and 'palm_velocity_w' in trace and np.isfinite(trace['palm_velocity_w'][select]).all():
            v = trace['palm_velocity_w'][select]
            phases[name]['palm_linear_speed_rms_m_s'] = np.sqrt(np.mean(np.sum(v[..., :3] ** 2, -1), 0)).tolist()
            phases[name]['palm_angular_speed_rms_rad_s'] = np.sqrt(np.mean(np.sum(v[..., 3:] ** 2, -1), 0)).tolist()
    operational = bool(complete and finite.all() and valid.all() and not fault.any() and resets == 0)
    return {
        'report_type': 'bounded_validation_report', 'result': 'MEASURED' if operational else 'INVALID_EXECUTION',
        'dynamic_contact_wrench_calibrated': False, 'auxiliary_default_modified': False,
        'completed_all_phases': bool(complete), 'robot_resets': int(resets),
        'nonfinite_samples_by_environment': (~finite).sum(0).tolist(),
        'invalid_samples_by_environment': (~valid).sum(0).tolist(),
        'fault_samples_by_environment': fault.sum(0).tolist(),
        'sample_period_s': dt, 'phase_names': list(PHASES), 'phases': phases,
        'interpretation': [
            'Known external loads are applied to moving palms without a cart or grasp constraint.',
            'Targets use the hand-on-environment sign, world axes, and each sampled palm link origin.',
            'Primary RMS/peak errors include dynamic load onset, force pulses, release and arm transients; no lag correction.',
            'Errors and lags are kept separate for every environment/hand; opposite signs cannot cancel.',
            'MEASURED describes complete finite execution, not compliance with a published accuracy standard.',
            'No full-grasp calibration, contact label certification, estimator training or auxiliary-default change is implied.',
            'Hand damping, solver integration and end-step sampling can contribute to residual error.',
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, launcher):
    import torch
    import importlib
    from homie_isaaclab.math_utils import quat_mul
    package=f'homie_pushcart_v{args.version}'
    rotation_matrix_xyzw=importlib.import_module(package+'.dynamics').rotation_matrix_xyzw
    lowlevel=importlib.import_module(package+'.lowlevel')
    HimLowLevelPolicy,frozen_checkpoint=lowlevel.HimLowLevelPolicy,lowlevel.frozen_checkpoint
    PushCartSimEnv=importlib.import_module(package+'.sim_env').PushCartSimEnv
    play_config=importlib.import_module(package+'.teleop').play_config
    torch.set_num_threads(4)

    class LabelProbeEnv(PushCartSimEnv):
        """Diagnostic-only injection and capture around the unchanged fast path."""
        label_recording = False

        def __init__(self,*env_args,**env_kwargs):
            if args.version==52:
                self.task_cfg=importlib.import_module(package+'.task_config').PushCartTaskCfg()
                self.task_cfg.actions.upper_drive=args.upper_drive
            super().__init__(*env_args,**env_kwargs)

        def _apply_action(self):
            super()._apply_action()
            if not self.label_recording:
                return
            substep = self._physics_substep - 1
            phase_time = self.label_control_step * self.dt + substep * self.cfg.sim.dt
            self.label_interval_time = phase_time
            phase_name = PHASES[self.label_phase_id]
            f, m = prescribed_load(phase_name, phase_time, args.force_scale)
            force = torch.tensor(f, device=self.device, dtype=torch.float32).expand(self.num_envs, -1, -1).contiguous()
            couple = torch.tensor(m, device=self.device, dtype=torch.float32).expand(self.num_envs, -1, -1).contiguous()
            palm_pose = self.robot.data.body_link_pose_w.torch[:, self.palm_body_ids]
            com = self.robot.data.body_com_pose_w.torch[:, self.palm_body_ids, :3]
            rotation = rotation_matrix_xyzw(palm_pose[..., 3:])
            offset = (rotation @ self.label_point_offset[..., None]).squeeze(-1)
            point = com + offset
            composer = self.robot.permanent_wrench_composer
            composer.set_forces_and_torques_index(
                forces=force, torques=couple, positions=point.contiguous(),
                body_ids=self.palm_body_ids.to(torch.int32), is_global=True)
            # Read the actual composer output before scene.write_data_to_sim.
            # Its torque is COM-centered in body axes; compare the complete
            # prescribed lever arm as well as the subsequently measured label.
            submitted_f = (rotation @ composer.composed_force.torch[:, self.palm_body_ids, :, None]).squeeze(-1)
            submitted_m = (rotation @ composer.composed_torque.torch[:, self.palm_body_ids, :, None]).squeeze(-1)
            self.label_applied = {
                'applied_force_w': force, 'applied_couple_w': couple,
                'application_point_pre_w': point, 'palm_com_pre_w': com.clone(),
                'palm_origin_pre_w': palm_pose[..., :3].clone(),
                'prescribed_com_moment_w': couple + torch.cross(offset, force, dim=-1),
                'submitted_force_w': submitted_f.clone(), 'submitted_com_moment_w': submitted_m.clone(),
            }

        def _sample_dynamics(self, substep):
            super()._sample_dynamics(substep)
            if not self.label_recording:
                return
            palm = self.robot.data.body_link_pose_w.torch[:, self.palm_body_ids, :3]
            com = self.robot.data.body_com_pose_w.torch[:, self.palm_body_ids, :3]
            force = self.label_applied['submitted_force_w']
            moment = self.label_applied['submitted_com_moment_w'] + torch.cross(com-palm, force, dim=-1)
            row = {**self.label_applied,
                   'measured_wrench_w': self._hand_wrench,
                   'expected_wrench_w': -torch.cat((force, moment), -1),
                   'palm_origin_sample_w': palm, 'palm_com_sample_w': com,
                   'palm_velocity_w': self.robot.data.body_com_vel_w.torch[:, self.palm_body_ids],
                   'upper_joint_pos_rad': self.robot.data.joint_pos.torch[:, self.joint_ids[12:]],
                   'valid': self.dynamics_wrench_valid, 'fault': self.dynamics_fault}
            for key, value in row.items():
                self.label_records.setdefault(key, []).append(value.clone())
            self.label_records.setdefault('phase', []).append(self.label_phase_id)
            self.label_records.setdefault('phase_time_s', []).append(self.label_interval_time + self.cfg.sim.dt)
            self.label_records.setdefault('time_s', []).append(
                self.label_phase_id * args.steps * self.dt + self.label_interval_time + self.cfg.sim.dt)

    cfg = play_config(args.num_envs, episode_length_s=(args.steps * len(PHASES) + args.settle_steps) * .02 + 30.)
    if args.upper_drive=='explicit_pd' and args.version!=52:
        raise ValueError('Explicit PD label validation requires V5.2')
    cfg.pushcart_upper_drive=args.upper_drive
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint or frozen_checkpoint()
    env = LabelProbeEnv(cfg, ROOT, args.device, args.headless, with_cart=False, full_dynamics=False)
    policy = HimLowLevelPolicy(checkpoint, args.device)
    obs, _ = env.reset()
    env.set_command(0., 0., 0.)
    yaw = torch.arange(args.num_envs, device=args.device) * (math.pi / 2.)
    yaw_q = torch.zeros(args.num_envs, 4, device=args.device)
    yaw_q[:, 2], yaw_q[:, 3] = torch.sin(yaw / 2.), torch.cos(yaw / 2.)
    pose = env.robot.data.root_link_pose_w.torch.clone()
    pose[:, 3:] = quat_mul(yaw_q, pose[:, 3:])
    env.robot.write_root_pose_to_sim_index(root_pose=pose)
    env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(args.num_envs, 6, device=args.device))
    env.label_records = {}
    env.label_point_offset = torch.tensor(POINT_OFFSET_PALM_AXES_M, device=args.device, dtype=torch.float32)
    reset_events = []
    failure = None
    try:
        with torch.inference_mode():
            for step in range(args.settle_steps):
                obs, _, _, _, _, ids, _ = env.step(policy(obs))
                if len(ids):
                    reset_events.append({'phase': 'settle', 'step': step, 'env_ids': ids.cpu().tolist()})
            env.label_recording = True
            for phase_id, name in enumerate(PHASES):
                env.label_phase_id = phase_id
                print(f'[v{args.version}-labels] phase={name} control_steps={args.steps}', flush=True)
                for step in range(args.steps):
                    if not launcher.app.is_running():
                        raise RuntimeError('Simulation closed before label validation completed')
                    env.label_control_step = step
                    delta = torch.zeros_like(env.upper_delta)
                    if name.startswith('dynamic_'):
                        # Continue the moving trajectory through loaded->release
                        # so release does not coincide with stopping the arms.
                        motion_time = ((phase_id-3) * args.steps + step) * env.dt
                        ramp = min(1., max(0., motion_time / .5))
                        for side, shift in (('left', 0.), ('right', math.pi / 2.)):
                            for joint, amplitude, hz in (('shoulder_pitch', .12, .8),
                                                         ('shoulder_roll', .08, 1.), ('elbow', .12, 1.2)):
                                j = env.dof_names.index(f'{side}_{joint}_joint') - env.num_lower_dof
                                delta[:, j] = args.motion_scale * ramp * amplitude * math.sin(
                                    2 * math.pi * hz * motion_time + shift) / cfg.control.action_scale
                    env.set_upper_delta(delta)
                    obs, _, _, _, _, ids, _ = env.step(policy(obs))
                    if len(ids):
                        reset_events.append({'phase': name, 'step': step, 'env_ids': ids.cpu().tolist()})
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        env.label_recording = False
        env.robot.permanent_wrench_composer.reset()
        env.set_upper_delta(torch.zeros_like(env.upper_delta))

    trace = {key: torch.stack(value).cpu().numpy() if isinstance(value[0], torch.Tensor) else np.asarray(value)
             for key, value in env.label_records.items() if value}
    np.savez_compressed(args.output / 'traces.npz', **trace)
    if not trace:
        raise RuntimeError(f'No samples recorded: {failure}')
    every_substep=env.full_dynamics or getattr(env,'_label_substep_mean',False)
    label_dt=cfg.sim.dt if every_substep else env.dt
    report = summarize_trace(trace, dt=label_dt, resets=sum(len(x['env_ids']) for x in reset_events),
                             expected_samples_per_phase=args.steps * (cfg.control.decimation if every_substep else 1))
    injection_f = trace['submitted_force_w'] - trace['applied_force_w']
    injection_m = trace['submitted_com_moment_w'] - trace['prescribed_com_moment_w']
    injection_finite = bool(np.isfinite(injection_f).all() and np.isfinite(injection_m).all())
    report.update({
        'package':package,'upper_drive':args.upper_drive,
        'recorded_every_physics_substep':every_substep,
        'failure': failure, 'reset_events': reset_events, 'num_envs': args.num_envs,
        'control_steps_per_phase': args.steps, 'control_dt_s': env.dt, 'seed': args.seed,
        'initial_yaw_rad': yaw.cpu().tolist(), 'force_scale': args.force_scale, 'motion_scale': args.motion_scale,
        'application_point': 'offset from palm COM in palm link axes; recomputed every 200 Hz substep',
        'application_point_offsets_m': POINT_OFFSET_PALM_AXES_M.tolist(),
        'expected_moment_definition': '-(submitted world COM torque + (sampled palm COM - sampled palm origin) cross submitted world force)',
        'injection_finite': injection_finite,
        'injection_force_max_error_N': float(np.linalg.norm(injection_f, axis=-1).max()) if injection_finite else None,
        'injection_com_moment_max_error_Nm': float(np.linalg.norm(injection_m, axis=-1).max()) if injection_finite else None,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(Path(checkpoint)),
        'source_metadata': env.dynamics_metadata,
        'source_sha256': {str(path.relative_to(ROOT)): _sha256(path) for path in (
            Path(__file__).resolve(), ROOT / f'HomieRL/{package}/sim_env.py')},
    })
    # These tolerances only verify the injection API's float32 arithmetic,
    # not acceptable accuracy of the inferred force labels.
    if failure or not injection_finite or report['injection_force_max_error_N'] > 1.e-3 or report['injection_com_moment_max_error_Nm'] > 1.e-3:
        report['result'] = 'INVALID_EXECUTION'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    rows = []
    for name, result in report['phases'].items():
        entries = [v for hands in result['environment_hand'] for v in hands if v is not None]
        if entries:
            rows.append('| ' + name + ' | ' + ' | '.join(f'{max(v[key] for v in entries):.5f}' for key in (
                'force_vector_rms_N', 'force_vector_peak_N', 'moment_vector_rms_Nm', 'moment_vector_peak_Nm')) + ' |')
    (args.output / 'report.md').write_text(
        f'# V{args.version/10:.1f} bounded moving-hand known-load report\n\n'
        f"Execution: **{report['result']}**. Full-grasp dynamic calibration: **False**.\n\n"
        'Values are maxima over environment/hand pairs. All dynamic onsets and releases are included; no lag correction.\n\n'
        '| Phase | Force RMS (N) | Force peak (N) | Moment RMS (Nm) | Moment peak (Nm) |\n'
        '|---|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        'Per-hand biases, impulses and dynamic force/moment lags are in `report.json`; 200 Hz traces are in `traces.npz`.\n\n'
        + '\n'.join('- ' + item for item in report['interpretation']) + '\n', encoding='utf-8')
    print(f"[v{args.version}-labels] result={report['result']} dynamic_calibrated=False output={args.output}", flush=True)
    return report


def main(argv=None):
    sys.path[:0] = [str(ROOT / 'HomieRL'), str(ROOT / 'HomieRL/rsl_rl')]
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',type=int,choices=(51,52),default=51)
    parser.add_argument('--upper_drive',choices=('training','explicit_pd'),default='training')
    parser.add_argument('--num_envs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=150, help='control steps per phase, at least 100')
    parser.add_argument('--settle_steps', type=int, default=75)
    parser.add_argument('--force_scale', type=float, default=1.)
    parser.add_argument('--motion_scale', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=51)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/bench' / ('v51_labels_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.num_envs <= 8 or args.steps < 100 or args.settle_steps < 25:
        parser.error('Use 1..8 environments, at least 100 steps/phase and 25 settling steps')
    if not 0. < args.force_scale <= 2. or not 0. < args.motion_scale <= 2.:
        parser.error('force_scale and motion_scale must lie in (0, 2]')
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(args)
    try:
        report = run(args, launcher)
        return 0 if report['result'] == 'MEASURED' else 1
    finally:
        launcher.app.close()


if __name__ == '__main__':
    raise SystemExit(main())
