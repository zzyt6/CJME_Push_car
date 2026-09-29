"""Actual Isaac smoke: sensor history, independent labels, visual dropout and partial reset."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'HomieRL'))
# Kit shutdown can turn a pending Python exception into exit code 0. The parent
# requires a success artifact from THIS child invocation, never a stale report.
if not os.environ.get('PUSHCART_V53_RUNTIME_CHILD'):
    run_id=uuid.uuid4().hex
    env=dict(os.environ, PUSHCART_V53_RUNTIME_CHILD=run_id)
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env=env)
    if '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(result.returncode)
    probe=argparse.ArgumentParser(add_help=False)
    probe.add_argument('--output',type=Path,default=ROOT/'logs/bench/v53_implementation/isaac_runtime.json')
    known,_=probe.parse_known_args()
    report=json.loads(known.output.read_text(encoding='utf-8')) if known.output.is_file() else {}
    success=result.returncode==0 and report.get('run_id')==run_id and report.get('result')=='PASS_EXECUTION'
    raise SystemExit(0 if success else 1)
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps',type=int,default=64)
p.add_argument('--num_envs',type=int,default=8)
p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v53_implementation/isaac_runtime.json')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
launcher=AppLauncher(args)
try:
    import torch
    from homie_pushcart_v53.env_smoke import make_configs
    from homie_pushcart_v53.task_env import PushCartEnv
    from homie_pushcart_v53.observation import FRAME_DIM, frame_layout
    torch.manual_seed(53)
    cfg,task=make_configs(args.num_envs,True,level=8,episode_length_s=20.)
    task.curriculum.enabled=False
    task.diagnostics.finite_check_interval_steps=1
    env=PushCartEnv(cfg,task,ROOT,args.device,True,level=8)
    env.fixed_command=[.25,0.,.1]
    env.reset()
    valid_speed=valid_force=dropped=partial_resets=0
    max_velocity_label_error=0.
    fields={f['name']:slice(f['start'],f['start']+f['dim']) for f in frame_layout()}
    for step in range(args.steps):
        old_obs,old_label=env.hl_obs,env.cart_velocity_target
        saved_obs,saved_label=old_obs.clone(),old_label.clone()
        obs,critic,reward,done,extra=env.step(torch.zeros(args.num_envs,18,device=args.device))
        torch.testing.assert_close(old_obs,saved_obs,rtol=0,atol=0)
        torch.testing.assert_close(old_label,saved_label,rtol=0,atol=0)
        assert obs.shape==(args.num_envs,FRAME_DIM*20) and critic.shape==(args.num_envs,202)
        assert not hasattr(env,'fixed_grasp')
        assert torch.isfinite(obs).all() and torch.isfinite(env.interaction_target).all()
        valid=env.cart_velocity_target[:,3]>.5
        expected=torch.cat((env.ts.cart_v,env.ts.cart_wz[:,None]),-1)*env.cart_velocity_scale
        if valid.any():
            error=(expected[valid]-env.cart_velocity_target[valid,:3]).abs().max().item()
            max_velocity_label_error=max(max_velocity_label_error,error)
            torch.testing.assert_close(expected[valid],env.cart_velocity_target[valid,:3],rtol=0,atol=0)
        valid_speed+=int(valid.sum())
        force_valid=env.interaction_target[:,12]>.5
        valid_force+=int(force_valid.sum())
        assert (env.hand_wrench_count[force_valid]==4).all()
        dropped+=int((~env.heading_valid).sum())
        if step==args.steps//2:
            ids=torch.tensor([0],device=args.device)
            env.reset_idx(ids)
            env._update_hl_observations()
            assert env.actor_hist[0,:-1].abs().sum()==0
            assert env.cart_velocity_target[0,3]==0 and env.interaction_target[0,12]==0
            assert env.actor_hist[0,-1,fields['joint_torque']].abs().sum()==0
            partial_resets+=1
    assert valid_speed>0 and valid_force>0 and dropped>0 and partial_resets==1
    report=dict(version=53,result='PASS_EXECUTION',run_id=os.environ['PUSHCART_V53_RUNTIME_CHILD'],num_envs=args.num_envs,steps=args.steps,
        valid_velocity_labels=valid_speed,valid_hand_labels=valid_force,invalid_vision_samples=dropped,
        partial_resets=partial_resets,velocity_label_alignment_max_error=max_velocity_label_error,
        force_substeps_per_valid_label=4,observable_frame_dim=FRAME_DIM,
        note='Runtime/supervision integrity check only; no estimator accuracy claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
except BaseException:
    import traceback
    error=traceback.format_exc()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.failure.json').write_text(json.dumps(dict(result='FAIL',error=error),indent=2),encoding='utf-8')
    print(error,flush=True)
    raise
finally:
    launcher.app.close()
