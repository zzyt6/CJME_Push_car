"""Twelve-condition closed-loop tests; choose seeds using Isaac validation only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'logs/bench/v52_falcon_explicit'
GROUPS=('training_full','training_proprio','explicit_pd_full','explicit_pd_proprio',
        'explicit_pd_randomized_full','explicit_pd_randomized_proprio','explicit_pd_falcon','explicit_pd_falcon_fullbody')


def run(args,name,command):
    state=dict(stage=name,command=command,started_unix=time.time(),status='running')
    suffix=f'_{args.job_label}' if args.job_label else ''
    path=args.output/f'validation_{args.engine}{suffix}_job.json'
    print('START',name,flush=True)
    with (args.output/f'{name}.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-u','-X','utf8',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state['pid']=p.pid
        path.write_text(json.dumps(state,indent=2),encoding='utf-8')
        code=p.wait()
    state.update(status='complete' if code==0 else 'failed',exit_code=code,ended_unix=time.time())
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('END',name,code,flush=True)
    if code:
        raise SystemExit(code)


def main(args):
    base=['scripts/eval_pushcart_v52_sim2sim.py','--engine',args.engine,'--profiles','isaac','--steps','500']
    if args.engine=='isaac':
        base+=['--viz','none']
    if args.capture:
        for drive in ('training','explicit_pd'):
            run(args,f'capture_{args.engine}_{drive}',base+['--upper_drive',drive,'--capture','--output',str(args.output/'captures')])
    for group in args.groups:
        candidates=[]
        for seed in (11,22,33):
            directory=args.output/f'model_{group}_{seed}'
            report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
            candidates.append((report['scores']['validation']['selection_mse'],seed,directory/'estimator.pt',report))
        metric,seed,model,report=min(candidates,key=lambda x:x[0])
        out=args.output/'closed_loop'/group
        out.mkdir(parents=True,exist_ok=True)
        (out/'selection.json').write_text(json.dumps(dict(group=group,seed=seed,model=str(model),
            isaac_validation_mse=metric,rule='minimum Isaac validation normalized twelve-dimensional MSE among seeds 11/22/33'),indent=2),encoding='utf-8')
        run(args,f'closed_{args.engine}_{group}',base+['--upper_drive',report['upper_drive'],
            '--estimator',str(model),'--output',str(out)])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=('mujoco','isaac'),required=True)
    p.add_argument('--capture',action='store_true')
    p.add_argument('--job_label',default='',help='separate status files for independent validation subsets')
    p.add_argument('--groups',nargs='*',choices=GROUPS,default=list(GROUPS[:6]))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    main(p.parse_args())
