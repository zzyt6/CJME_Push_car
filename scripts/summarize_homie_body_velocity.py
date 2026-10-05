"""Summarize available HOMIE/HIM audit runs without retraining or old-version imports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval_homie_body_velocity import metrics

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'logs/bench/homie_body_velocity_20260928'


def vec(values):
    return ' / '.join(f'{x:.4f}' for x in values)


def summarize(folder):
    reports, data, statistics = {}, {}, {}
    for run in sorted(folder.iterdir()):
        report_path, sample_path = run / 'report.json', run / 'samples.npz'
        if not report_path.is_file() or not sample_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding='utf8'))
        if report.get('result') != 'PASS_EXECUTION' or 'current' not in report:
            continue
        name = run.name
        if report['engine'] == 'isaac':
            assert report['parent_verified'] and report['child_exit_code'] == 0, name
        reports[name] = report
        with np.load(sample_path) as samples:
            data[name] = x = {key: samples[key] for key in samples.files}
        valid = x['valid'].astype(bool) & (x['time'] >= report.get('warmup_s', 2.))
        current = metrics(x['prediction'][valid], x['truth_com'][valid])
        np.testing.assert_allclose(current['mae'], report['current']['mae'], atol=1e-12)
        case_stats = []
        for i, case in enumerate(report['cases']):
            selection = valid[:, i]
            stable = selection & (x['time'][:, i] >= 8.) & (x['time'][:, i] < 12.)
            mean = lambda key: x[key][stable, i].mean(0).tolist() if stable.any() else None
            case_stats.append({**case, 'current': metrics(x['prediction'][selection, i], x['truth_com'][selection, i]),
                               'steady_true': mean('truth_com'), 'steady_prediction': mean('prediction'),
                               'steady_command': mean('command')})
        statistics[name] = dict(current=current, action_time_to_next=report['action_time_to_next'],
                                origin=report['origin'], reference_point_difference=report['reference_point_difference'],
                                cases=case_stats,
                                gyro_rms=np.sqrt(np.square(x['gyro'][valid]).mean(0)).tolist(),
                                joint_velocity_rms=float(np.sqrt(np.square(x['dq'][valid]).mean())))
    if not reports:
        raise ValueError(f'No completed HOMIE body-velocity runs in {folder}')
    hashes = sorted({r['low_sha256'] for r in reports.values()})
    checks_path = folder / 'replay_checks.json'
    checks = json.loads(checks_path.read_text(encoding='utf8')) if checks_path.is_file() else None
    if checks is not None:
        checks = {name: check for name, check in checks.items() if name in reports}
        for name, check in checks.items():
            assert check['cpu_replay_max_abs'] < 2e-6, name
            assert check['latest_frame_reconstruction_max_abs'] < 1e-6, name
            assert check['noncommand_history_shift_max_abs'] == 0., name
            assert check['com_velocity_identity_max_abs'] < 1e-6, name
    summary = dict(low_sha256=hashes[0] if len(hashes) == 1 else hashes, conditions=statistics,
                   replay_checks=checks, included_runs=list(reports),
                   all_runs_without_termination=all(r['falls_or_terminations'] == 0 for r in reports.values()))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    free = [name for name, report in reports.items() if report['mode'] == 'free']
    plot_path = None
    if free:
        plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .22})
        fig, axes = plt.subplots(2, len(free), figsize=(6 * len(free), 7), squeeze=False, sharex=True, sharey='row')
        for col, name in enumerate(free):
            report, x = reports[name], data[name]
            for row, scenario in enumerate(('straight', 'reverse')):
                selected = next((i for i, case in enumerate(report['cases']) if case['scenario'] == scenario), None)
                ax = axes[row, col]
                if selected is None:
                    ax.set_visible(False)
                    continue
                t = x['time'][:, selected]
                ax.axvspan(0, report.get('warmup_s', 2.), color='gray', alpha=.12)
                ax.plot(t, x['truth_com'][:, selected, 0], label='Actual body vx', lw=1.)
                ax.plot(t, x['prediction'][:, selected, 0], label='HIM estimate', lw=1.)
                ax.plot(t, x['command'][:, selected, 0], label='Applied command', color='black', ls='--', lw=1.)
                ax.set_title(f'{name}: {scenario}')
                ax.set_ylabel('Pelvis COM vx (m/s)')
                ax.set_xlabel('Simulation time (s)')
                ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = folder / 'free_forward_reverse.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    lines = ['# HOMIE/HIM body-velocity audit', '',
             'This summary includes only the completed runs still present in this folder. Archived cart runs are optional.', '',
             'The velocity axes are pelvis-body COM [vx, vy, vz] in m/s. The third axis is not cart yaw rate.',
             'Metrics exclude warm-up and invalid/reset samples. Simulation conditions and checkpoints are read from each report; historical runs may use an earlier plant.', '',
             '| Run | Engine / mode | MAE vx / vy / vz | RMSE vx / vy / vz | Falls / terminations |',
             '|---|---|---|---|---:|']
    for name, report in reports.items():
        current = statistics[name]['current']
        lines.append(f'| {name} | {report["engine"]} / {report["mode"]} | {vec(current["mae"])} | {vec(current["rmse"])} | {report["falls_or_terminations"]} |')
    lines += ['', 'HIM is trained against the next control-step velocity. Both current-time and action-to-next-step comparisons are retained:', '',
              '| Run | Action-to-next-step MAE vx / vy / vz | Origin-vs-COM MAE |', '|---|---|---|']
    for name, report in reports.items():
        lines.append(f'| {name} | {vec(report["action_time_to_next"]["mae"])} | {vec(report["reference_point_difference"]["mae"])} |')
    if plot_path:
        lines += ['', f'![Free forward/reverse trajectories]({plot_path.name})']
    lines += ['', 'Checkpoint hashes and per-case statistics are in [summary.json](summary.json). Each included run retains its own report and raw samples.',
              'Reproduce free motion with `scripts/eval_homie_body_velocity.py --engine isaac|mujoco --mode free --output <new-directory>`.',
              'Cart operation additionally requires `--cart_version v3.2|v6 --checkpoint <explicit-checkpoint>`.',
              'MuJoCo now uses the single legacy four-sphere plant. Historical reports are not rewritten by the evaluation command.']
    (folder / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=FOLDER)
    args = parser.parse_args()
    summary = summarize(args.folder)
    print(json.dumps({key: value['current']['mae'] for key, value in summary['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
