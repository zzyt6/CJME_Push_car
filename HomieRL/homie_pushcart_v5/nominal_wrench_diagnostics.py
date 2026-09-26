"""Offline identification diagnostics; simulator truth NEVER enters control.

No parameters are fitted or exported. Recorded actor residuals are kept separate
from an independent impulse-balance estimate of the hand load needed for the
ACTUAL motion. The latter is not the desired-task hand load: it also contains
acceleration caused by the current controller, gait and disturbance.
"""
from __future__ import annotations

import numpy as np

AXES = ('Fx_N', 'Fy_N', 'Mz_Nm')


def shift_spatial(value, source, target):
    """World-axis force/momentum + moment shifted to a common fixed point."""
    value = np.asarray(value)
    return np.concatenate((value[..., :3], value[..., 3:]+np.cross(source-target, value[..., :3])), -1)


def to_heading(planar_world, yaw):
    value = np.asarray(planar_world)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c*value[..., 0]+s*value[..., 1], -s*value[..., 0]+c*value[..., 1], value[..., 2]), -1)


def stats(error, mask=None, deadzone=None):
    value = np.asarray(error, dtype=float)
    good = np.isfinite(value).all(-1)
    if mask is not None:
        good &= mask
    result = {'samples': int(good.sum()), 'axes': list(AXES)}
    if not good.any():
        return {**result, 'bias': None, 'rmse': None, 'deadzone_exceedance_fraction': None}
    selected = value[good]
    result.update(bias=selected.mean(0).tolist(), rmse=np.sqrt((selected**2).mean(0)).tolist(),
                  absolute_p95=np.quantile(np.abs(selected), .95, axis=0).tolist())
    if deadzone is not None:
        threshold = np.broadcast_to(np.asarray(deadzone), value.shape)[good]
        result['deadzone_exceedance_fraction'] = (np.abs(selected)>threshold).mean(0).tolist()
    return result


def independent_control_interval_load(trace):
    """Recover planar hand demand from momentum/ground/external impulse only.

    At each control time t[k], hold the recorded nominal target for the NEXT
    control interval. Recover that interval's load about the cart COM at t[k]
    from both endpoints' momenta shifted to that fixed point. Contact and applied
    moments are independently shifted from their recorded reference points.
    Then rotate the planar load to cart-heading axes AT t[k]. This avoids mixing
    moving-frame derivatives, moving-COM moments, or current target with old load.

    Gravity is absent from Fx/Fy/Mz for a vertical gravity field, irrespective of
    COM translation. Fz/Mx/My are deliberately NOT returned without logged mass.
    Hand wrench is read only into a separate closure check, never reconstruction.
    """
    required = ('time_s', 'cart_yaw', 'required_wrench', 'physics_time_s', 'physics_cart_momentum',
                'physics_cart_com', 'physics_force_cart_com', 'physics_wheel_wrench',
                'physics_cart_applied_reference', 'physics_cart_applied_wrench',
                'physics_hand_wrench', 'physics_dt_s', 'physics_force_interval_start_s')
    missing = [key for key in required if key not in trace]
    if missing:
        raise ValueError('Missing independently timed dynamics channels: '+', '.join(missing))
    times = np.asarray(trace['physics_time_s'])
    control = np.asarray(trace['time_s'])
    dt = float(trace['physics_dt_s'])
    if len(times)<2 or not np.allclose(np.diff(times), dt, rtol=1e-6, atol=1e-9):
        raise ValueError('Physics timestamps must be continuous, uniform and reset-free')
    if not np.allclose(trace['physics_force_interval_start_s'], times-dt, atol=1e-8):
        raise ValueError('Force intervals do not agree with the physics timestamps')
    h, com = np.asarray(trace['physics_cart_momentum']), np.asarray(trace['physics_cart_com'])
    force_com = np.asarray(trace['physics_force_cart_com'])
    ground = np.asarray(trace['physics_wheel_wrench']).sum(1)
    applied = np.asarray(trace['physics_cart_applied_wrench'])
    applied_point = np.asarray(trace['physics_cart_applied_reference'])
    hand = np.asarray(trace['physics_hand_wrench'])
    valid = np.asarray(trace.get('physics_ground_valid', np.ones(len(times), dtype=bool)), dtype=bool)
    output = {key: [] for key in ('control_index', 'start_s', 'end_s', 'valid', 'demand', 'measured',
                                  'prediction', 'closure_error', 'prediction_error', 'ground_resistance')}
    for k in range(len(control)-1):
        start = int(np.searchsorted(times, control[k]-1.e-8))
        end = int(np.searchsorted(times, control[k+1]-1.e-8))
        if start>=len(times) or end>=len(times) or end<=start:
            continue
        if abs(times[start]-control[k])>1.e-7 or abs(times[end]-control[k+1])>1.e-7:
            raise ValueError('Control endpoints must coincide with recorded physics states')
        use = slice(start+1, end+1)
        reference = com[start]
        duration = times[end]-times[start]
        momentum_change = shift_spatial(h[end], com[end], reference)-shift_spatial(h[start], com[start], reference)
        ground_impulse = shift_spatial(ground[use], force_com[use], reference).sum(0)*dt
        applied_impulse = shift_spatial(applied[use], applied_point[use], reference).sum(0)*dt
        # THIS line is independent of all hand-force channels.
        demand_world = (momentum_change-ground_impulse-applied_impulse)/duration
        measured_world = shift_spatial(hand[use], force_com[use], reference).sum(0)*dt/duration
        yaw = np.asarray(trace['cart_yaw'])[k]
        demand = to_heading(demand_world[[0, 1, 5]], yaw)
        measured = to_heading(measured_world[[0, 1, 5]], yaw)
        predicted = np.asarray(trace['required_wrench'])[k]
        good = bool(valid[use].all() and np.isfinite(np.r_[demand, measured, predicted]).all())
        for name, value in dict(control_index=k, start_s=control[k], end_s=control[k+1], valid=good,
                                demand=demand, measured=measured, prediction=predicted,
                                closure_error=measured-demand, prediction_error=demand-predicted,
                                ground_resistance=to_heading((-ground_impulse/duration)[[0, 1, 5]], yaw)).items():
            output[name].append(value)
    return {key: np.asarray(value) for key, value in output.items()}


def excitation_diagnostic(twist, valid, velocity_scale):
    """Refuse unsupported resistance calibration; report a temporal split only.

    Potential regressors are [v,tanh(v/epsilon)]. No hand force is a regressor
    or target. Even adequate rank is necessary, not sufficient: caster heading,
    acceleration and load dependence need independent trials to validate.
    """
    twist = np.asarray(twist)[np.asarray(valid, dtype=bool)]
    split = int(len(twist)*.6)
    report = {'fit_performed': False, 'parameters_promoted': False,
              'split': 'first 60% training / last 40% held out, chronological', 'axes': {}}
    for axis, name in enumerate(AXES):
        reasons, details = [], {}
        for label, part in (('training', twist[:split, axis]), ('heldout', twist[split:, axis])):
            if len(part)<30:
                reasons.append(label+' has fewer than 30 samples')
                continue
            x = np.stack((part, np.tanh(part/velocity_scale[axis])), -1)
            singular = np.linalg.svd(x, compute_uv=False)
            condition = float(singular[0]/max(singular[-1], 1.e-15))
            positive, negative = int((part>.05).sum()), int((part<-.05).sum())
            details[label] = {'samples': len(part), 'minimum': float(part.min()), 'maximum': float(part.max()),
                              'std': float(part.std()), 'regressor_condition': condition,
                              'positive_samples': positive, 'negative_samples': negative}
            if min(positive, negative)<10:
                reasons.append(label+' lacks bidirectional excitation')
            if np.ptp(part)<.15 or part.std()<.025:
                reasons.append(label+' has insufficient velocity variation')
            if condition>100.:
                reasons.append(label+' viscous/rolling regressors are ill-conditioned')
        report['axes'][name] = {'passes_necessary_excitation_checks': not reasons, 'identifiable': False,
                               'refusal_reasons': reasons + ['single trajectory lacks independent caster/load/acceleration validation'],
                               **details}
    return report


def analyze_trace(trace, cfg, *, onset=None, release=None):
    """Return residual/model/closure statistics with explicit signal contracts."""
    ready = np.asarray(trace['sensor_valid'])>.5
    time = np.asarray(trace['time_s'])
    ready &= time>=2.
    residual = np.asarray(trace['sensor_wrench'])[:, [0, 1, 5]]-np.asarray(trace['aligned_required_wrench'])
    zone = np.asarray(cfg['uncertainty_abs'])+float(cfg['uncertainty_fraction'])*np.abs(trace['aligned_required_wrench'])
    recorded = np.asarray(trace['excess_wrench'])
    if not np.allclose(residual[ready], recorded[ready], atol=2.e-5, rtol=2.e-5):
        raise ValueError('Recorded excess wrench is inconsistent with sensor/target channels')
    windows = {'after_ramp': ready}
    if onset is not None:
        windows.update(before_event=ready & (time<onset), during_event=ready & (time>=onset) & (time<release),
                       after_release=ready & (time>=release))
    report = {'actor_residual': {name: stats(residual, mask, zone) for name, mask in windows.items()},
              'resistance_excitation': excitation_diagnostic(trace['cart_twist'], ready, cfg['resistance_velocity_scale'])}
    try:
        balance = independent_control_interval_load(trace)
    except ValueError as exc:
        report['independent_balance'] = {'available': False, 'reason': str(exc)}
        return report, None
    indices = balance['control_index'].astype(int)
    good = balance['valid'] & ready[indices]
    balance_zone = np.asarray(cfg['uncertainty_abs'])+float(cfg['uncertainty_fraction'])*np.abs(balance['prediction'])
    report['independent_balance'] = {'available': True,
        'closure_measured_minus_independent': stats(balance['closure_error'], good),
        'next_interval_demand_minus_held_nominal_prediction': stats(balance['prediction_error'], good, balance_zone),
        'mean_independent_load': balance['demand'][good].mean(0).tolist() if good.any() else None,
        'mean_held_nominal_prediction': balance['prediction'][good].mean(0).tolist() if good.any() else None,
        'mean_independent_ground_resistance': balance['ground_resistance'][good].mean(0).tolist() if good.any() else None,
        'windowed_prediction_error': {name: stats(balance['prediction_error'], good & mask[indices], balance_zone)
                                     for name, mask in windows.items()}}
    return report, balance


def paired_change(trace, reference, cfg, *, onset, release):
    """Compare each run in its own cart-heading coordinates, no subtraction of world vectors."""
    n = min(len(trace['time_s']), len(reference['time_s']))
    if not np.allclose(trace['time_s'][:n], reference['time_s'][:n], atol=1.e-8):
        raise ValueError('Paired control timestamps do not agree')
    a, b = independent_control_interval_load(trace), independent_control_interval_load(reference)
    size = min(len(a['start_s']), len(b['start_s']))
    time = a['start_s'][:size]
    good = (a['valid'][:size] & b['valid'][:size] & (time>=onset))
    return {'contract': 'time-aligned task-relative components; each run has its own heading/COM and gait phase',
            'after_onset_demand_change': stats(a['demand'][:size]-b['demand'][:size], good),
            'after_onset_prediction_change': stats(a['prediction'][:size]-b['prediction'][:size], good),
            'after_onset_prediction_error_change': stats(a['prediction_error'][:size]-b['prediction_error'][:size], good),
            'after_release_prediction_error_change': stats(a['prediction_error'][:size]-b['prediction_error'][:size], good & (time>=release))}
