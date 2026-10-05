"""Paper motion-ability data: independent Isaac experiments for baseline/CWC.

Each axis/method gets a fresh simulator process. No training files or production
task implementation are changed. Plotting consumes the measured raw velocities.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback
import types
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]
NONCE='CWC_MOTION_ABILITY_RUN'
AXES=('vx','vy','wz')
METHODS=('baseline','CWC')


def protocol(axis, idle_s=0., plateau_s=8.):
    # User-selected independent-axis protocols; wz is not scaled from vx.
    values={'vx':[.3,.5,.8],'vy':[.15,.3],'wz':[.15,.3]}[axis]
    stages=[];start=0.
    if idle_s>0:
        stages.append({'start_s':0.,'end_s':idle_s,'value':0.})
        start=idle_s
    for value in values:
        end=start+plateau_s
        stages.append({'start_s':start,'end_s':end,'value':value})
        start=end
    return stages


def schedule(axis,t,stages):
    command=np.zeros(3,dtype=np.float32)
    stage=next((s for s in stages if t<s['end_s']-1.e-9),stages[-1])
    command[AXES.index(axis)]=stage['value']
    return command


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(checkpoint,duration):
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart_v6.baseline import apply_config,homie_baseline
    from homie_pushcart_v6.task_config import task_cfg_from_run
    cfg=G1RoughCfg();apply_config(cfg,homie_baseline());cfg.env.num_envs=1
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_') or name in ('use_random','push_robots','delay','upper_body_randomization'):
            setattr(cfg.domain_rand,name,False)
    cfg.noise.add_noise=False
    task=task_cfg_from_run(checkpoint)
    task.curriculum.enabled=False;task.curriculum.init_level=17
    task.env.episode_length_s=duration+10.
    rz=task.randomization
    rz.init_xy_m=0.;rz.init_yaw=False;rz.payload_com_offset_m=0.;rz.grasp_threshold_jitter=0.
    rz.robot_push=False;rz.cart_push_vel_max=0.;rz.wheel_stall_prob_per_s=0.
    rz.wheel_friction_range=[.8,.8];rz.wheel_damping_range=[.005,.005];rz.swivel_damping_range=[.05,.05]
    for name in ('cart_pose_noise_pos_m','cart_pose_noise_yaw_rad','upper_torque_noise_frac','est_vel_noise_m_s'):
        setattr(task.observations,name,0.)
    task.observations.cart_pose_delay_steps_max=0;task.actions.delay_steps_max=0
    return cfg,task


def capture_physics(env):
    import warp as wp
    def a(x):return x.detach().cpu().numpy().copy()
    data={}
    for label,asset in (('robot',env.robot),('cart',env.cart)):
        for name in ('body_mass','body_inertia','body_com_pose_b','body_link_pose_w','root_com_vel_w','joint_pos','joint_vel',
                     'joint_stiffness','joint_damping','joint_armature'):
            data[f'{label}.{name}']=a(getattr(asset.data,name).torch)
        data[f'{label}.materials']=a(wp.to_torch(asset.root_view.get_material_properties()))
    for name in ('Kp_factors','Kd_factors','actuation_offset','joint_injection','cart_pose_delay','hl_delay',
                 'payload_com','wheel_friction','wheel_damping','swivel_damping','grasp_jitter'):
        data['effective.'+name]=a(getattr(env,name))
    return data


def run_child(args):
    from isaaclab.app import AppLauncher
    launcher=None
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'raw';raw.mkdir(exist_ok=True)
    key=f'{args.axis}_{args.method}'
    run_id=os.environ.get(NONCE)
    if not run_id:raise RuntimeError('Child requires parent nonce')
    try:
        launcher=AppLauncher(headless=True,device=args.device)
        import torch
        import warp as wp
        from homie_pushcart_v6.task_env import PushCartEnv,TERMINATION_REASONS
        from homie_pushcart_v6.hl_policy import HighLevelPolicy
        from homie_pushcart_v6.baseline import config_dict
        torch.set_num_threads(1)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        stages=protocol(args.axis,args.idle_s,args.plateau_s);duration=stages[-1]['end_s']
        cfg,task=configure(args.checkpoint,duration)
        assert task.commands.wz_max==.4,'Revisit the wz protocol if the trained range changes'
        env=PushCartEnv(cfg,task,ROOT,args.device,True,checkpoint=None,level=17)
        env.fixed_payload_kg=args.payload;env.fixed_caster_angle=0.;env.fixed_command=[0.,0.,0.]
        env.reset()
        controller=HighLevelPolicy(args.checkpoint,device=args.device)
        assert controller.iteration==args.iteration
        assert not env.add_noise and not cfg.domain_rand.delay
        assert not cfg.domain_rand.push_robots and not task.randomization.robot_push
        assert torch.count_nonzero(env.hl_delay)==0 and torch.count_nonzero(env.cart_pose_delay)==0
        assert torch.count_nonzero(env.actuation_offset)==0 and torch.count_nonzero(env.joint_injection)==0
        torch.testing.assert_close(env.Kp_factors,torch.ones_like(env.Kp_factors),rtol=0,atol=0)
        torch.testing.assert_close(env.Kd_factors,torch.ones_like(env.Kd_factors),rtol=0,atol=0)
        initial=capture_physics(env)
        np.savez_compressed(raw/f'{key}_initial_physics.npz',**initial)
        # A real reset-state sample at t=0, separate from post-physics intervals.
        # The default protocol immediately applies its first nonzero command;
        # it does not synthesize an idle rollout or freeze a simulated trajectory.
        cart_pos,cart_quat,cart_lin,cart_ang=env.cart_state()
        from homie_isaaclab.math_utils import euler_from_quaternion
        _,_,initial_yaw=euler_from_quaternion(cart_quat)
        c,s=torch.cos(initial_yaw[:,0]),torch.sin(initial_yaw[:,0])
        initial_v=torch.stack((c*cart_lin[:,0]+s*cart_lin[:,1],-s*cart_lin[:,0]+c*cart_lin[:,1],cart_ang[:,2]),-1)
        assert torch.count_nonzero(initial_v)==0,'Reset did not produce stationary cart state'
        assert not np.count_nonzero(initial['robot.root_com_vel_w'])
        reset_state={'t':np.asarray(0.),'cart_velocity':initial_v[0].cpu().numpy().copy(),
                     'cart_pose':torch.cat((cart_pos,cart_quat),-1)[0].cpu().numpy().copy(),
                     'cart_com_velocity_world':torch.cat((cart_lin,cart_ang),-1)[0].cpu().numpy().copy(),
                     'robot_pose':env.root_states[0,:7].cpu().numpy().copy(),
                     'robot_com_velocity_world':initial['robot.root_com_vel_w'][0].copy(),
                     'command':env.cmd_task[0].cpu().numpy().copy(),
                     'reference':env.cmd_ref[0].cpu().numpy().copy(),
                     'homie_command':env.cmd_homie[0].cpu().numpy().copy(),
                     'first_applied_command':schedule(args.axis,0.,stages),
                     'episode_length':env.episode_length_buf[0].cpu().numpy().copy(),
                     'actor_observation':env.hl_obs[0].cpu().numpy().copy(),
                     'estimator_observation':env.estimator_obs[0].cpu().numpy().copy(),
                     'homie_observation':env.obs_buf[0].cpu().numpy().copy()}
        assert int(reset_state['episode_length'])==0
        initial_state_path=raw/f'{key}_initial_state.npz'
        np.savez_compressed(initial_state_path,**reset_state)
        # Read state BEFORE automatic reset. Terminated trajectories end here;
        # a new episode is never appended as if it were continuous motion.
        pending={}
        original_check=env.check_termination
        def check_and_capture(this):
            original_check()
            ts=this.ts
            fields={
                'command':this.cmd_task,'reference':this.cmd_ref,'homie_command':this.cmd_homie,
                'cart_velocity':torch.cat((ts.cart_v,ts.cart_wz[:,None]),-1),
                'filtered_cart_velocity':this.cart_vel_f,
                'cart_pose':this.cart.data.root_link_pose_w.torch,
                'robot_pose':this.root_states[:,:7],
                'high_action':this.hl_actions,'high_action_raw':this.hl_raw,
                'upper_delta_q':this.upper_delta_q,'lower_action':this.actions[:,:12],
                'joint_position':this.dof_pos,'joint_velocity':this.dof_vel,
                'joint_torque_proxy':this.torques,
                'hand_force_proxy_world':ts.force_w.flatten(1),'hand_moment_proxy_world':ts.torque_w.flatten(1),
                'dynamic_hand_target':this._pre_reset_hands,
                'terminated':this.terminated_buf[:,None],'timeout':this.time_out_buf[:,None],
                'termination_flags':torch.stack([this.term_reason_buf[n] for n in TERMINATION_REASONS],-1),
            }
            # One packed transfer per recorded control sample; preserve values
            # before old reset functions can mutate their buffers.
            names=list(fields);widths=[fields[n].shape[1] for n in names]
            packed=torch.cat([fields[n].to(torch.float32) for n in names],-1).detach().cpu().numpy()[0]
            offset=0
            for name,width in zip(names,widths):
                pending[name]=packed[offset:offset+width].copy();offset+=width
        env.check_termination=types.MethodType(check_and_capture,env)
        records={};dt=float(env.dt);n=int(round(duration/dt));terminated_reason=None
        with torch.inference_mode():
            for tick in range(n):
                start=tick*dt;command=schedule(args.axis,start,stages)
                env.cmd_task.copy_(torch.as_tensor(command,device=args.device)[None]);env.cmd_timer.fill_(10**9)
                if args.method=='CWC':
                    action=controller(env.hl_obs,env.estimator_obs)
                else:
                    action=torch.zeros(1,18,device=args.device)
                env.step(action)
                sample={'t':(tick+1)*dt,'interval_start':start,**pending}
                if args.record_estimators:
                    # The post-step observable histories and hand label describe
                    # the same state. Cart HIM predicts the NEXT state. Capture
                    # the exact network inputs for independent offline replay.
                    speed,hands=controller.physical_estimates(env.hl_obs,env.estimator_obs)
                    valid=not (pending['terminated'][0] or pending['timeout'][0])
                    if valid:
                        np.testing.assert_array_equal(env.hand_target[0].cpu().numpy(),pending['dynamic_hand_target'])
                    sample.update(pred_cart_velocity=speed[0].cpu().numpy().copy(),
                                  pred_hand_wrench=hands[0].cpu().numpy().copy(),
                                  true_hand_wrench=pending['dynamic_hand_target'][:12]*env.hand_scale.cpu().numpy(),
                                  hand_label_valid=bool(pending['dynamic_hand_target'][12]>.5),
                                  estimator_sample_valid=valid,
                                  actor_observation=env.hl_obs[0].cpu().numpy().copy(),
                                  estimator_observation=env.estimator_obs[0].cpu().numpy().copy())
                for name,value in sample.items():records.setdefault(name,[]).append(value)
                if pending['terminated'][0] or pending['timeout'][0]:
                    terminated_reason=[name for name,flag in zip(TERMINATION_REASONS,pending['termination_flags']) if flag]
                    if pending['timeout'][0]:terminated_reason.append('time_out')
                    break
        arrays={k:np.asarray(v) for k,v in records.items()}
        for k in ('terminated','timeout'):arrays[k]=arrays[k].reshape(-1).astype(bool)
        assert all(np.isfinite(x).all() for x in arrays.values())
        np.savez_compressed(raw/f'{key}.npz',**arrays)
        with (raw/f'{key}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            names=['t_s','interval_start_s']+[f'{prefix}_{axis}' for prefix in ('command','reference','actual','homie') for axis in AXES]+['terminated','timeout']
            writer=csv.writer(f);writer.writerow(names)
            for i,t in enumerate(arrays['t']):
                writer.writerow([float(t),float(arrays['interval_start'][i]),*arrays['command'][i],*arrays['reference'][i],
                    *arrays['cart_velocity'][i],*arrays['homie_command'][i],int(arrays['terminated'][i]),int(arrays['timeout'][i])])
        if args.method=='baseline':assert np.count_nonzero(arrays['high_action'])==0
        final_physics=capture_physics(env)
        np.savez_compressed(raw/f'{key}_final_physics.npz',**final_physics)
        for label in ('robot','cart'):
            for name in ('body_mass','body_inertia','body_com_pose_b','joint_stiffness','joint_damping','joint_armature','materials'):
                np.testing.assert_array_equal(initial[f'{label}.{name}'],final_physics[f'{label}.{name}'])
        metadata={
            'result':'PASS_EXECUTION','run_id':run_id,'engine':'Isaac Sim / PhysX','axis':args.axis,'component':AXES.index(args.axis),
            'method':args.method,'payload_kg':args.payload,'cart_assembly_mass_kg':float(initial['cart.body_mass'].sum()),
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),'iteration':controller.iteration,
            'homie_checkpoint':str(env.policy.checkpoint),'homie_sha256':sha(env.policy.checkpoint),
            'dt':dt,'physics_dt':cfg.sim.dt,'stages':stages,'duration_s':duration,'samples':len(arrays['t']),
            'initial_state_file':f'raw/{initial_state_path.name}','initial_state_sha256':sha(initial_state_path),
            'initial_zero_is_instant_not_dwell':args.idle_s==0.,'initial_idle_rollout_s':args.idle_s,
            'reset_velocity_exactly_zero':True,'initialization':'Fresh simulator process; full reset; record actual t=0 state before any controlled physics step',
            'completed':len(arrays['t'])==n and terminated_reason is None,'failure':terminated_reason,
            'seed':args.seed,'num_envs':1,'independent_simulator_process':True,
            'baseline_definition':'Zero all18 high-level residuals; same frozen HOMIE, fixed grasp, command/ref filters and plant',
            'CWC_definition':'V6 deterministic Actor mean with both learned estimators; no training updates',
            'velocity_source':'raw physical cart root COM velocity; translation in cart-heading axes, angular velocity world-z',
            'velocity_is_estimate':False,'velocity_lowpass_applied_to_saved_actual':False,
            'sampling':'post-physics, pre-reset; t is interval end, interval_start is command application time',
            'reference_tau_s':task.commands.ref_tau_s,'homie_command_tau_s':task.actions.cmd_filter_tau_s,
            'wz_schedule_basis':'User-selected angular-axis steps .15, .3 rad/s, from a recorded initial zero state; within V6 trained .4 rad/s bound',
            'vy_requested_extrapolation':args.axis=='vy' and .3>task.commands.vy_max,
            'terminations':list(TERMINATION_REASONS),'npz_sha256':sha(raw/f'{key}.npz'),
            'physics_contract':{'static_ground':True,'upper_drive':'V3.2 implicit position','nominal_parameters_verified_unchanged':True},
            'randomization_effective':False,'observation_noise_effective':False,'action_sampling':False,
            'arrays':{k:list(v.shape) for k,v in arrays.items()},
            'cfg':config_dict(cfg),'task_cfg':config_dict(task)}
        if args.record_estimators:
            from homie_pushcart_v6.hand_labels import ReadOnlyHandLabels
            metadata.update(estimator_recording=True,cart_prediction_horizon_s=dt,
                            hand_reference=ReadOnlyHandLabels.metadata(),
                            estimator_sampling='Post-step observable histories; synchronous hand label; cart prediction targets next step',
                            estimator_inputs_saved=True,collector_sha256=sha(__file__))
        (raw/f'{key}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in metadata.items() if k in ('axis','method','completed','samples','failure','cart_assembly_mass_kg')},ensure_ascii=False),flush=True)
    except BaseException:
        error=traceback.format_exc()
        (args.output/f'{key}.failure.json').write_text(json.dumps({'run_id':run_id,'error':error},indent=2),encoding='utf-8')
        print(error,flush=True)
        raise
    finally:
        if launcher is not None:launcher.app.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--payload',type=float,default=15.)
    parser.add_argument('--idle_s',type=float,default=0.,help='Optional simulated zero-command dwell; default 0 records only the reset instant before immediate nonzero steps')
    parser.add_argument('--plateau_s',type=float,default=8.)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--record_estimators',action='store_true',help='Save exact estimator inputs, physical predictions and synchronous hand labels')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/Picture/motion_ability')
    parser.add_argument('--axis',choices=AXES,help=argparse.SUPPRESS)
    parser.add_argument('--method',choices=METHODS,help=argparse.SUPPRESS)
    parser.add_argument('--iteration',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if os.environ.get(NONCE):return run_child(args)
    if args.idle_s<0 or args.plateau_s<=0:parser.error('idle_s must be nonnegative and plateau_s positive')
    if args.checkpoint is None:
        runs=sorted(r for r in (ROOT/'logs/rsl_rl/homie_pushcart_v6').iterdir() if r.is_dir() and any(r.glob('model_*.pt')))
        if not runs:raise FileNotFoundError('No V6 model exists')
        args.checkpoint=max(runs[-1].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    args.checkpoint=args.checkpoint.resolve()
    # Do not import Isaac or a GPU policy in this parent.
    import torch
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    args.iteration=int(ckpt['iter']);del ckpt
    checkpoint_hash=sha(args.checkpoint)
    run_id=uuid.uuid4().hex
    attempt=args.output/'runs'/f"{datetime.now():%Y%m%d_%H%M%S}_{run_id[:8]}"
    attempt.mkdir(parents=True,exist_ok=False)
    manifest={'run_id':run_id,'engine':'Isaac Sim / PhysX','checkpoint':str(args.checkpoint),
              'checkpoint_sha256':checkpoint_hash,'iteration':args.iteration,'payload_kg':args.payload,
              'seed':args.seed,'device':args.device,'protocol':{a:protocol(a,args.idle_s,args.plateau_s) for a in AXES},
              'initial_zero_is_instant_not_dwell':args.idle_s==0.,'initial_idle_rollout_s':args.idle_s,
              'independence':'6 separate simulator processes, one fresh environment per axis/method',
              'attempt_dir':str(attempt.resolve()),'runs':[]}
    (attempt/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for axis in AXES:
        for method in METHODS:
            key=f'{axis}_{method}'
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--axis',axis,'--method',method,
                     '--checkpoint',str(args.checkpoint),'--iteration',str(args.iteration),'--payload',str(args.payload),
                     '--idle_s',str(args.idle_s),'--plateau_s',str(args.plateau_s),'--seed',str(args.seed),
                     '--device',args.device,'--output',str(attempt)]
            if args.record_estimators:command.append('--record_estimators')
            print(f'[motion] starting {key}',flush=True)
            with (attempt/f'{key}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,**{NONCE:run_id}),stdout=log,stderr=subprocess.STDOUT)
            report_file=attempt/'raw'/f'{key}.json'
            if result.returncode!=0 or not report_file.is_file():
                raise RuntimeError(f'{key} did not finish; exit={result.returncode}; see {attempt/key}.log')
            report=json.loads(report_file.read_text(encoding='utf-8'))
            assert report['run_id']==run_id and report['result']=='PASS_EXECUTION'
            assert report['checkpoint_sha256']==checkpoint_hash
            assert report['npz_sha256']==sha(attempt/'raw'/f'{key}.npz')
            assert report['initial_state_sha256']==sha(attempt/report['initial_state_file'])
            report['parent_verified']=True;report['child_exit_code']=result.returncode
            report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            manifest['runs'].append({k:report[k] for k in ('axis','method','completed','samples','failure','npz_sha256')})
            print('[motion]',key,'completed=',report['completed'],'samples=',report['samples'],flush=True)
    equivalence=[]
    reference=None
    for axis in AXES:
        for method in METHODS:
            with np.load(attempt/'raw'/f'{axis}_{method}_initial_physics.npz') as data:
                current={k:data[k] for k in data.files}
            if reference is None:reference=current
            for k in reference:np.testing.assert_array_equal(reference[k],current[k],err_msg=f'{axis}/{method}/{k}')
            equivalence.append({'axis':axis,'method':method,'all_initial_physics_and_state_equal':True,'arrays':len(current)})
    assert checkpoint_hash==sha(args.checkpoint)
    manifest['validation']={'initial_physics':equivalence,'checkpoint_unchanged':True,
                            'all_programs_completed':all(r['completed'] for r in manifest['runs'])}
    (attempt/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Publish the selected complete acquisition without deleting past attempts.
    dest=args.output/'raw';dest.mkdir(exist_ok=True,parents=True)
    for f in (attempt/'raw').iterdir():shutil.copy2(f,dest/f.name)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[motion] verified data:',args.output.resolve(),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
