"""Summarize independently measured label/constraint evidence and fixed heads."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
OUT = ROOT/'logs/bench/v52_sim2sim_root_cause'
BASE = ROOT/'logs/bench/v52_falcon_explicit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from homie_pushcart_v52.transfer_estimator import load_estimator
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE

    sources = {
        'isaac_known': BASE/'falcon_a3/known_load_v52_substeps_recheck/report.json',
        'mujoco_known': OUT/'known_load_mujoco/report.json',
        'isaac_grasp_balance': OUT/'cart_label_balance_matched/report.json',
        'nominal_grasp': OUT/'grasp_response_mujoco/report.json',
        'rigid_grasp': OUT/'grasp_response_high_impedance/report.json',
    }
    reports = {k: read(v) for k, v in sources.items()}
    for k in ('isaac_known', 'mujoco_known', 'isaac_grasp_balance'):
        assert reports[k]['result'] == 'MEASURED'
    traces = {k: np.load(v.parent/'traces.npz') for k, v in sources.items() if k in ('isaac_known', 'mujoco_known')}
    applied_difference = float(np.max(abs(traces['isaac_known']['applied_force_w']-traces['mujoco_known']['applied_force_w'])))
    time_difference = float(np.max(abs(traces['isaac_known']['phase_time_s']-traces['mujoco_known']['phase_time_s'])))
    calibration = {}
    for key in ('isaac_known', 'mujoco_known'):
        pairs = [v for phase in reports[key]['phases'].values() for env in phase['environment_hand'] for v in env if v]
        calibration[key] = {field: max(v[field] for v in pairs) for field in
                            ('force_vector_rms_N', 'force_vector_peak_N', 'moment_vector_rms_Nm')}

    heads = {}
    for name, folder, new_folder in (
        ('full', BASE/'closed_loop/explicit_pd_full', 'full_rigid_limit'),
        ('fullbody', BASE/'falcon_a3/closed_loop/explicit_pd_falcon_fullbody', 'fullbody_rigid_limit')):
        rows = {}
        for condition, path in (
            ('isaac', folder/'physx_explicit_pd.json'),
            ('mujoco_nominal', folder/'isaac_explicit_pd.json'),
            ('mujoco_rigid_limit', OUT/new_folder/'isaac_explicit_pd_rigid_limit.json')):
            r = read(path)
            assert r['completed_conditions'] == r['total_conditions'] == 12
            assert r['handle']['n'] == 4800 and all(c['failure'] is None for c in r['conditions'])
            rows[condition] = dict(handle_rmse=r['handle']['rmse'], hands_rmse=r['hands']['rmse'],
                                   conditions=12, samples=4800,
                                   tracking_rmse=np.sqrt(np.mean(np.array([c['tracking_rmse'] for c in r['conditions']])**2, 0)).tolist(),
                                   source=str(path.relative_to(ROOT)))
        heads[name] = rows

    isaac = np.load(OUT/'cart_label_balance_matched/traces.npz')
    nominal = np.load(OUT/'grasp_response_mujoco/traces.npz')
    rigid = np.load(OUT/'grasp_response_high_impedance/traces.npz')
    t = (np.arange(len(isaac['q']))+20)*.005
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for name, d, color in [('Isaac fixed joints', isaac, '#1b4f72'), ('MuJoCo nominal weld', nominal, '#b03a2e'), ('MuJoCo rigid_limit', rigid, '#148f77')]:
        force = d['measured_hand_wrench'][:, 2, :, :3]
        for ax, value in zip(axes, (force[:, 0, 1], force[:, 1, 1], force[:, :, 1].sum(1))):
            ax.plot(t, value, label=name, color=color, lw=1.3)
    for ax, label in zip(axes, ('Left hand Fy (N)', 'Right hand Fy (N)', 'Combined Fy (N)')):
        ax.set_ylabel(label)
        ax.axvspan(.4, 1.2, color='gray', alpha=.09)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].set_title('Same targets and known cart load; no learned policy, no ground contact')
    axes[-1].set_xlabel('Time (s); shaded: prescribed cart load active')
    fig.savefig(OUT/'dual_grasp_response.png', dpi=170)
    plt.close(fig)

    torch.set_num_threads(4)
    capture = torch.load(OUT/'full_rigid_limit/isaac_explicit_pd_rigid_limit_samples.pt', map_location='cpu', weights_only=False)
    model, _ = load_estimator(BASE/'model_explicit_pd_full_22/estimator.pt')
    with torch.inference_mode():
        pred = model(capture['x']).numpy()*np.tile(HAND_WRENCH_SCALE, 2)
    true = capture['y'].numpy()*np.tile(HAND_WRENCH_SCALE, 2)
    replay_rmse = np.sqrt(np.mean((pred-true)**2, 0))
    np.testing.assert_allclose(replay_rmse, heads['full']['mujoco_rigid_limit']['hands_rmse'], atol=2e-5)
    chosen = capture['condition_id'].numpy() == 0
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, layout='constrained')
    time = 2.+(np.arange(chosen.sum())+1)*.02
    for ax, col, label in zip(axes.flat, (0, 1, 6, 7), ('Left Fx', 'Left Fy', 'Right Fx', 'Right Fy')):
        ax.plot(time, true[chosen, col], color='#1b4f72', label='MuJoCo reference', lw=1.3)
        ax.plot(time, pred[chosen, col], color='#d35400', label='Frozen force estimator', lw=1.)
        ax.set_ylabel(label+' (N)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle('Actual closed-loop prediction: rigid_limit, 0 kg payload, 0.3 m/s command')
    fig.savefig(OUT/'prediction_example.png', dpi=170)
    plt.close(fig)

    summary = dict(known_load_calibration=calibration, applied_force_max_difference_n=applied_difference,
                   applied_time_max_difference_s=time_difference,
                   cart_label_balance=reports['isaac_grasp_balance'],
                   no_policy_grasp_response={k: reports[k] for k in ('nominal_grasp', 'rigid_grasp')},
                   fixed_head_evaluation=heads,
                   prediction_replay_rmse_max_difference=float(abs(replay_rmse-np.array(heads['full']['mujoco_rigid_limit']['hands_rmse'])).max()),
                   source_reports={k: str(v.relative_to(ROOT)) for k, v in sources.items()},
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (ROOT/'scripts/audit_v52_cart_label_balance.py', ROOT/'scripts/audit_v52_mujoco_grasp_response.py',
                                   ROOT/'scripts/validate_v52_mujoco_known_load.py', Path(__file__))})
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(calibration=calibration, fixed_head_evaluation=heads), indent=2))


if __name__ == '__main__':
    main()
