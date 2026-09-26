"""Export trained estimator + exact PD parameters and verified G1 SDK mapping."""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def export(args):
    import numpy as np
    import torch
    from homie_pushcart_v52.control_contract import nominal_configs,control_contract,pd_effort
    from homie_pushcart_v52.paths import HOMIE_JOINTS,G1_GRASP_URDF
    from homie_pushcart_v52.pd_command import HOMIE_TO_UNITREE_G1
    from homie_pushcart_v52.observation import frame_layout
    from homie_pushcart_v52.transfer_estimator import load_estimator,PhysicalWrenchExport,SelectedWrenchExport,FEATURE_SETS
    torch.set_num_threads(1)
    model,bundle=load_estimator(args.estimator)
    metadata=bundle['training']['data_metadata']
    if metadata['upper_drive']!='explicit_pd':
        raise ValueError('Deployment experiment requires a head trained under explicit PD')
    cfg,task=nominal_configs(Path(metadata['behavior_checkpoint']),'explicit_pd')
    joints={j.get('name'):j.find('limit') for j in ET.parse(G1_GRASP_URDF).getroot().findall('joint')}
    contract=control_contract(cfg,HOMIE_JOINTS,[float(joints[n].get('effort')) for n in HOMIE_JOINTS],'explicit_pd')
    contract.update(position_lower=[float(joints[n].get('lower')) for n in HOMIE_JOINTS],
                    position_upper=[float(joints[n].get('upper')) for n in HOMIE_JOINTS],
                    velocity_limits=[float(joints[n].get('velocity')) for n in HOMIE_JOINTS],
                    motor_indices=list(HOMIE_TO_UNITREE_G1),unmodeled_motor_indices=[13,14],
                    sdk_fields=['q','dq','kp','kd','tau'],mode_pr=0,
                    command_equation='clip(tau_ff + kp*(q_des-q) + kd*(dq_des-dq), -limit, limit)',
                    unmodeled_waist='27-DOF model assumes waist roll/pitch fixed; holding is integration-owned',
                    clipping='Total torque is software-clipped in simulators/host_effort; verify firmware limits in motor_pd mode')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'control.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    exports={'estimator_physical.jit':PhysicalWrenchExport(model), 'estimator_selected.jit':SelectedWrenchExport(model)}
    x=torch.randn(32,model.input_width)
    selected=x[:,model.indices]
    checks={}
    with torch.inference_mode():
        ref=exports['estimator_physical.jit'](x)
        for name,module in exports.items():
            compiled=torch.jit.script(module)
            compiled.save(str(args.output/name))
            loaded=torch.jit.load(str(args.output/name))
            z=x if name=='estimator_physical.jit' else selected
            error=float((loaded(z)-ref).abs().max())
            if error>1e-4:
                raise AssertionError('Export predictions differ')
            sample=z[:1]
            for _ in range(20):
                loaded(sample)
            durations=[]
            for _ in range(200):
                t=time.perf_counter()
                loaded(sample)
                durations.append((time.perf_counter()-t)*1000)
            checks[name]=dict(max_abs_error=error,cpu_latency_ms_median=float(np.median(durations)),
                              cpu_latency_ms_p95=float(np.percentile(durations,95)))
    names=FEATURE_SETS[model.features]
    fields=[]
    selected_start=0
    for f in frame_layout():
        if names is None or f['name'] in names:
            fields.append(dict(name=f['name'],source_start=f['start'],selected_start=selected_start,dim=f['dim']))
            selected_start+=f['dim']
    if model.features.startswith('falcon'):
        fields=[]
        selected_start=0
        for name,width in (('joint_position',27),('joint_velocity',27),('imu_gyro',3),
                           ('projected_gravity',3),('previous_lower_action',12)):
            fields.append(dict(name=name,selected_start=selected_start,dim=width))
            selected_start+=width
        if model.features == 'falcon_fullbody':
            fields.append(dict(name='upper_target_applied',selected_start=selected_start,dim=15))
    # Small portable package: NumPy for PD/history and Torch only for JIT
    # inference. It does not import Isaac, MuJoCo, RSL-RL, or robot transport.
    portable=args.output/'v52_deploy'
    portable.mkdir()
    (portable/'__init__.py').write_text('from .pd_command import PDCommand, command_from_contract\nfrom .deploy_proprio import ProprioHistory\n',encoding='utf-8')
    (portable/'control_contract.py').write_text(inspect.getsource(pd_effort),encoding='utf-8')
    for filename in ('pd_command.py','deploy_proprio.py'):
        shutil.copyfile(ROOT/'HomieRL/homie_pushcart_v52'/filename,portable/filename)
    manifest=dict(schema='v52_deploy_bundle_v1',estimator=str(args.estimator),
                  estimator_sha256=hashlib.sha256(args.estimator.read_bytes()).hexdigest(),
                  features=model.features,history=5,frame_dim=model.input_width//5,selected_frame_dim=len(model.indices)//5,
                  selected_fields=fields,
                  output='left Fx,Fy,Fz,Mx,My,Mz then right; N and Nm; palm origins; cart-heading axes',
                  history_order='oldest first; reset prior frames to zero; sample at 50 Hz',checks=checks,
                  proprio_runtime='v52_deploy.ProprioHistory' if model.features=='proprio' else None,
                  pd_runtime='v52_deploy.command_from_contract',
                  falcon_reference_commit='a967a6d8494f57777cf8d266a644ac8e45833301',
                  hardware_tested=False,
                  integration='Populate existing Unitree SDK2 LowCmd; transport, CRC, mode_machine, watchdog and unmodeled waist belong to robot integration')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--estimator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    export(p.parse_args())
