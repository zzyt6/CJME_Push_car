"""Analyze existing V5 NPZ data offline; no simulator, training or parameter fit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))
from homie_pushcart_v5.nominal_wrench_diagnostics import analyze_trace, paired_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.benchmark/'summary.json').read_text(encoding='utf-8'))
    report = {'source': str(args.benchmark.resolve()), 'axes': ['Fx_N', 'Fy_N', 'Mz_Nm'], 'modes': {},
              'contracts': [
                  'Actor residual compares saved delayed/filtered sensor to identically delayed/filtered target in recorded cart-heading axes.',
                  'Independent actual-motion load uses only complete-cart momentum, ground contact and external load; hand force is used solely for closure validation.',
                  'Physical interval moments and endpoint momenta are shifted to a fixed control-start COM before differencing/integration; results rotate to control-start heading.',
                  'Nominal target at t is held as a prediction for the next control interval; this is a predictive diagnostic, not an exactly instantaneous target/measurement subtraction.',
                  'Gravity cancels only in the reported planar Fx/Fy/Mz components for vertical gravity; no vertical or roll/pitch inference is made.',
                  'Actual-motion force demand includes achieved acceleration and disturbance response. It is not a unique desired-task force target.',
                  'No resistance parameters are fitted/promoted; a single narrow trajectory is insufficient evidence for calibration.'],
              'additional_channels_for_identification': ['physics_cart_yaw', 'physics_cart_mass', 'physics_cart_inertia_world',
                  'physics_cart_com_velocity', 'estimated_cart_twist', 'motion_estimator_ready',
                  'observer_heading_delta', 'cart_pose_delay_steps', 'physical_wheel_heading_and_speed']}
    for mode, values in metadata['modes'].items():
        cfg = values['config']['reaction']
        reference = dict(np.load(args.benchmark/f'{mode}_reference.npz', allow_pickle=False))
        analysis, _ = analyze_trace(reference, cfg)
        output = {'reference': analysis, 'scenarios': {}}
        for name, scenario_report in values['scenarios'].items():
            source = args.benchmark/f'{mode}_{name}.npz'
            if not source.is_file():
                continue
            trace = dict(np.load(source, allow_pickle=False))
            scenario = scenario_report['scenario']
            timing = {'onset': scenario['onset_s'], 'release': scenario['release_s']}
            analysis, _ = analyze_trace(trace, cfg, **timing)
            try:
                analysis['paired_change'] = paired_change(trace, reference, cfg, **timing)
            except ValueError as exc:
                analysis['paired_change'] = {'available': False, 'reason': str(exc)}
            output['scenarios'][name] = analysis
        report['modes'][mode] = output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for mode, values in report['modes'].items():
        reference = values['reference']
        print(mode, json.dumps({'actor_residual': reference['actor_residual']['after_ramp'],
                               'independent_balance': reference['independent_balance']}))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
