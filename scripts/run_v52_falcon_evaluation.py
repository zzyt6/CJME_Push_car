"""Evaluate A.3 inputs with explicit PD; reuse only complete original captures."""
import argparse
import json
from pathlib import Path

from run_v52_transfer_validation import run, main as closed_loop

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    prefix='physx' if args.engine=='isaac' else 'isaac'
    capture=args.output/'captures'/f'{prefix}_explicit_pd.json'
    samples=capture.with_name(f'{prefix}_explicit_pd_samples.pt')
    if capture.exists() and samples.exists():
        import torch
        r=json.loads(capture.read_text(encoding='utf-8'))
        d=torch.load(samples,map_location='cpu',weights_only=False)
        assert r['total_conditions']==12 and len(r['conditions'])==12
        assert not r.get('estimator_override') and d['falcon_x'].shape[1]==360
        print('REUSE original capture',capture,flush=True)
    else:
        command=['scripts/eval_pushcart_v52_sim2sim.py','--engine',args.engine,'--profiles','isaac',
                 '--upper_drive','explicit_pd','--capture','--output',str(args.output/'captures')]
        if args.engine=='isaac':
            command+=['--viz','none']
        run(args,f'capture_{args.engine}_explicit_pd',command)
    args.capture=False
    args.groups=['explicit_pd_falcon']
    closed_loop(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('isaac','mujoco'),required=True)
    p.add_argument('--job_label',default='falcon')
    p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit/falcon_a3')
    main(p.parse_args())
