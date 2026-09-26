"""Run the predeclared matched-drive estimator experiment, with durable logs."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/bench/v52_falcon_explicit'


def run(name, args):
    status = dict(stage=name, status='running', started_unix=time.time(), command=args)
    (OUT/'job.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('START', name, flush=True)
    with (OUT/f'{name}.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', *args], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
        status['pid'] = proc.pid
        (OUT/'job.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        code = proc.wait()
    status.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
    (OUT/'job.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('END', name, code, flush=True)
    if code:
        raise SystemExit(code)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = [('training', False), ('explicit_pd', False), ('explicit_pd', True)]
    for drive, randomized in datasets:
        name = drive + ('_randomized' if randomized else '')
        dataset = OUT/f'data_{name}'
        run(f'collect_{name}', ['scripts/train_pushcart_v52_transfer.py', 'collect', '--upper_drive', drive,
            '--num_envs', '64', '--steps', '4096', '--seed', '101', '--output', str(dataset), '--viz', 'none',
            *(['--actuator_randomization'] if randomized else [])])
        for features in ('full', 'proprio'):
            for seed in (11, 22, 33):
                run(f'train_{name}_{features}_{seed}', ['scripts/train_pushcart_v52_transfer.py', 'train',
                    '--dataset', str(dataset), '--features', features, '--seed', str(seed), '--epochs', '100',
                    '--patience', '15', '--output', str(OUT/f'model_{name}_{features}_{seed}')])
    (OUT/'training_complete.json').write_text(json.dumps(dict(completed_unix=time.time(),
        datasets=3, trained_models=18, note='Selection uses Isaac validation only; cross-engine evaluation is separate.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
