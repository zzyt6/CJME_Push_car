"""Collect one paired dataset and fit A.3 and full-body estimator variants."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/bench/v52_falcon_explicit/falcon_a3'


def run(output, name, command):
    print('START', name, flush=True)
    status = dict(stage=name, command=command, status='running', started_unix=time.time())
    with (output / f'{name}.log').open('w', encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', *command], cwd=ROOT,
                             stdout=log, stderr=subprocess.STDOUT)
        status['pid'] = p.pid
        (output/'job.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        code = p.wait()
    status.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_unix=time.time())
    (output/'job.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('END', name, code, flush=True)
    if code:
        raise SystemExit(code)


def main(args):
    # Refuse before opening any log, so an accidental rerun cannot erase evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve() if args.dataset else args.output/'data_explicit_pd'
    if args.dataset is None:
        run(args.output, 'collect', ['scripts/train_pushcart_v52_transfer.py', 'collect', '--upper_drive', 'explicit_pd',
            '--num_envs', '64', '--steps', '4096', '--seed', '101', '--output', str(dataset), '--viz', 'none'])
    for features in args.features:
        for seed in (11, 22, 33):
            run(args.output, f'train_{features}_{seed}', ['scripts/train_pushcart_v52_transfer.py', 'train',
                '--dataset', str(dataset), '--features', features, '--seed', str(seed),
                '--output', str(args.output/f'model_explicit_pd_{features}_{seed}')])
    (args.output/'training_complete.json').write_text(json.dumps(dict(
        models=3*len(args.features), features=args.features, dataset=str(dataset),
        completed_unix=time.time()), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='New experiment directory; must not exist')
    parser.add_argument('--dataset',type=Path,help='Reuse a paired dataset with falcon_x; otherwise collect in Isaac')
    parser.add_argument('--features',nargs='+',choices=('falcon','falcon_fullbody'),
                        default=['falcon','falcon_fullbody'])
    main(parser.parse_args())
