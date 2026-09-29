"""Freeze final sim2sim evidence without overwriting the grasp-only comparison."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
OUT = ROOT/'logs/bench/v52_sim2sim_root_cause'
BASE = ROOT/'logs/bench/v52_falcon_explicit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    from homie_pushcart_v52.observation import frame_layout, HAND_WRENCH_SCALE
    from homie_pushcart_v52.transfer_estimator import load_estimator
    torch.set_num_threads(4)
    sources = {
        'original': [ROOT/'logs/bench/v52_sim2sim_alignment/physx.json',
                     ROOT/'logs/bench/v52_sim2sim_alignment/isaac.json',
                     OUT/'original_checkpoint_default_grasp/isaac_rigid_limit.json',
                     OUT/'original_checkpoint_final_alignment/isaac_rigid_limit_physx_inertia.json'],
        'full': [BASE/'closed_loop/explicit_pd_full/physx_explicit_pd.json',
                 OUT/'full_nominal_capture/isaac_explicit_pd.json',
                 OUT/'full_rigid_limit/isaac_explicit_pd_rigid_limit.json',
                 OUT/'full_physx_inertia/isaac_explicit_pd_rigid_limit_physx_inertia.json'],
        'fullbody': [BASE/'falcon_a3/closed_loop/explicit_pd_falcon_fullbody/physx_explicit_pd.json',
                     BASE/'falcon_a3/closed_loop/explicit_pd_falcon_fullbody/isaac_explicit_pd.json',
                     OUT/'fullbody_rigid_limit/isaac_explicit_pd_rigid_limit.json',
                     OUT/'fullbody_final_alignment/isaac_explicit_pd_rigid_limit_physx_inertia.json'],
    }
    results = {}
    for head, paths in sources.items():
        results[head] = {}
        for name, path in zip(('isaac', 'mujoco_nominal', 'grasp_only', 'final'), paths):
            r = read(path)
            assert r['completed_conditions'] == r['total_conditions'] == 12
            assert r['handle']['n'] == 4800 and all(c['failure'] is None for c in r['conditions'])
            results[head][name] = dict(handle=r['handle'], hands=r['hands'],
                tracking_rmse=(np.sqrt(np.mean(np.array([c['tracking_rmse'] for c in r['conditions']])**2, 0)).tolist()
                               if all('tracking_rmse' in c for c in r['conditions']) else None),
                upper_drive=r.get('upper_drive', 'training'), source=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    # Reference bands use the ACTUAL full-head training collection, training
    # environments only. Evaluate matching captures with the same frozen head.
    training_path = BASE/'data_explicit_pd/dataset.pt'
    data = torch.load(training_path, map_location='cpu', weights_only=True)
    train = data['x'][:, -136:].float()[data['env_id'] % 8 >= 2]
    quantiles = torch.quantile(train, torch.tensor([.01, .99]), dim=0)
    capture_paths = {name: path.with_name(path.stem+'_samples.pt') for name, path in
                     zip(('nominal', 'grasp_only', 'final'), sources['full'][1:])}
    fields = frame_layout()
    coverage = {}
    for name, path in capture_paths.items():
        capture = torch.load(path, map_location='cpu', weights_only=False)
        x = capture['x'][:, -136:].float()
        coverage[name] = {}
        for field in fields:
            s = slice(field['start'], field['start']+field['dim'])
            coverage[name][field['name']] = 100*float(((x[:, s]<quantiles[0, s])|(x[:, s]>quantiles[1, s])).float().mean())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model, _ = load_estimator(BASE/'model_explicit_pd_full_22/estimator.pt')
    capture = torch.load(capture_paths['final'], map_location='cpu', weights_only=False)
    with torch.inference_mode():
        prediction = model(capture['x']).numpy()*np.tile(HAND_WRENCH_SCALE, 2)
    target = capture['y'].numpy()*np.tile(HAND_WRENCH_SCALE, 2)
    rmse = np.sqrt(np.mean((prediction-target)**2, 0))
    np.testing.assert_allclose(rmse, results['full']['final']['hands']['rmse'], atol=2e-5)
    chosen = capture['condition_id'].numpy() == 0
    time = 2.+(np.arange(chosen.sum())+1)*.02
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), layout='constrained', sharex=True)
    for ax, col, label in zip(axes.flat, (0, 1, 6, 7), ('Left Fx', 'Left Fy', 'Right Fx', 'Right Fy')):
        ax.plot(time, target[chosen, col], label='MuJoCo reference', color='#174f75', lw=1.2)
        ax.plot(time, prediction[chosen, col], label='Frozen estimator', color='#c96824', lw=1.)
        ax.set_ylabel(label+' (N)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]: ax.set_xlabel('Time (s)')
    fig.savefig(OUT/'final_prediction_example.png', dpi=180)
    plt.close(fig)
    report = dict(fixed_heads=results, coverage=coverage,
        coverage_definition='Percent of scalar entries outside training-only coordinate 1%-99% bands, latest frame only; not independent trajectory fractions.',
        coverage_training_source=training_path.relative_to(ROOT).as_posix(), training_samples=len(train),
        replay_rmse_max_difference=float(abs(rmse-np.array(results['full']['final']['hands']['rmse'])).max()),
        runtime_parameters=read(OUT/'runtime_parameters_physx_inertia_final.json')['payloads'],
        cart_inertia_reset_max_error=read(OUT/'cart_inertia_reset.json')['max_inertia_tensor_error_kg_m2'])
    (OUT/'final_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(results={h:{k:v['handle']['rmse'] for k,v in rows.items()} for h,rows in results.items()},
                          coverage=coverage), indent=2))


if __name__ == '__main__':
    main()
