"""Run V5.3 on MuJoCo; per-hand axes and learned cart velocity are logged separately."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--warmup_steps',type=int,default=100)
    p.add_argument('--level',type=int,default=0)
    p.add_argument('--command',type=float,nargs=3,default=[.25,0.,0.])
    p.add_argument('--payload',type=float,default=5.)
    p.add_argument('--scene_version',choices=('original','four_spheres','new'),default='original',
                   help='original: existing foot mesh; new/four_spheres: four contact spheres per foot')
    p.add_argument('--viewer',action='store_true',help='Show the native MuJoCo viewer and run at real time')
    p.add_argument('--output',type=Path,help='New result directory; default uses scene version and current time')
    args=p.parse_args(argv)
    if not 0 <= args.warmup_steps < args.steps:
        p.error('Require 0 <= warmup_steps < steps')
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.policies_v53 import V53Runner
    r=V53Runner(args.checkpoint,level=args.level,scene_version=args.scene_version)
    if args.output is None:
        args.output=ROOT/'logs/bench'/f'v53_mujoco_{r.scene_version}_{time.strftime("%Y%m%d_%H%M%S")}'
    args.output.mkdir(parents=True,exist_ok=False)
    r.set_payload(args.payload); r.reset()
    values={k:[] for k in ('force_prediction','force_label','velocity_prediction','velocity_label','observation')}
    fallen_count=0
    print(f'[V5.3 MuJoCo] scene={r.scene_version} xml={r.env.xml_path}',flush=True)
    with ExitStack() as stack:
        viewer=None
        if args.viewer:
            import mujoco.viewer
            viewer=stack.enter_context(mujoco.viewer.launch_passive(r.env.model,r.env.data))
        for step in range(args.steps):
            started=time.perf_counter()
            if viewer is not None and not viewer.is_running(): break
            obs,fallen=r.step(args.command)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.,r.dt-(time.perf_counter()-started)))
            if fallen:
                fallen_count+=1
                break
            if step < args.warmup_steps: continue
            # Physics labels are recorded only AFTER the measurement-only inference.
            world=np.mean(r.env.sub_hands,axis=0)
            c,s=np.cos(r.ts['cart_yaw']),np.sin(r.ts['cart_yaw'])
            rotation=np.array([[c,s,0],[-s,c,0],[0,0,1]])
            label=np.concatenate([(rotation@world[:, :3].T).T,(rotation@world[:, 3:].T).T],axis=-1).reshape(12)
            row=(r.pred_hands,label,r.pred_cart_velocity,np.r_[r.ts['cart_v'],r.ts['cart_wz']],obs)
            for key,value in zip(values,row): values[key].append(np.asarray(value).copy())
    if not values['observation']:
        if args.viewer and not fallen_count:
            print('Viewer closed before the scoring interval; no accuracy report was produced.')
            return
        raise RuntimeError('No scored samples; check the checkpoint and rollout')
    arrays={key:np.stack(value) for key,value in values.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite V5.3 runtime data')
    force_error=np.abs(arrays['force_prediction']-arrays['force_label'])
    names=[f'{side}_{axis}' for side in ('L','R') for axis in ('fx','fy','fz','mx','my','mz')]
    report=dict(version=53,checkpoint=str(args.checkpoint),samples=len(force_error),falls=fallen_count,
        command=args.command,payload_kg=args.payload,frame_dim=r.frame_dim,
        force_mae={name:float(value) for name,value in zip(names,force_error.mean(0))},
        velocity_mae=dict(zip(('vx','vy','omega_z'),np.abs(arrays['velocity_prediction']-arrays['velocity_label']).mean(0).tolist())),
        heading_source='simulated vision relative yaw; noise/delay disabled for this nominal evaluation',
        scene=dict(version=r.scene_version,xml=str(r.env.xml_path),grasp='nominal',inertia='urdf',upper_drive='explicit_pd'),
        note='Single-condition evaluation; smoke-trained checkpoints establish execution only, not estimator accuracy.')
    np.savez_compressed(args.output/'samples.npz',**arrays)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(scene_version=r.scene_version,samples=report['samples'],falls=fallen_count,
                         frame_dim=r.frame_dim,output=str(args.output))))


if __name__=='__main__': main()
