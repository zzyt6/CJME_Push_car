"""Audit compiled MuJoCo geometry/kinematics against independent training URDF FK."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def transform(origin):
    t=np.eye(4)
    if origin is not None:
        t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


def urdf_fk(path,base_pose,values):
    root=ET.parse(path).getroot()
    joints=list(root.findall('joint'))
    child_names={j.find('child').get('link') for j in joints}
    base=next(l.get('name') for l in root.findall('link') if l.get('name') not in child_names)
    poses={base:base_pose}
    axes={}
    while joints:
        progress=False
        for j in joints[:]:
            parent=j.find('parent').get('link')
            if parent not in poses:
                continue
            joint=poses[parent]@transform(j.find('origin'))
            axis=np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ') if j.find('axis') is not None else np.array([1.,0.,0.])
            axes[j.get('name')]=(joint[:3,3].copy(),joint[:3,:3]@axis)
            motion=np.eye(4)
            value=values.get(j.get('name'),0.)
            if j.get('type') in ('revolute','continuous'):
                motion[:3,:3]=Rotation.from_rotvec(axis*value).as_matrix()
            elif j.get('type')=='prismatic':
                motion[:3,3]=axis*value
            poses[j.find('child').get('link')]=joint@motion
            joints.remove(j)
            progress=True
        if not progress:
            raise ValueError('Disconnected URDF tree')
    return poses,axes


def main(args):
    from homie_pushcart_v52.control_contract import nominal_configs
    from homie_pushcart_v52.paths import G1_GRASP_URDF,CART_URDF,HOMIE_JOINTS
    from pushcart_bench.policies_v5x import InstrumentedEnv
    from pushcart_bench.sim2sim_scene import model_report
    checkpoint=ROOT/'logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt'
    cfg,_=nominal_configs(checkpoint,'explicit_pd')
    env=InstrumentedEnv(True,scene_profile='isaac',cfg=cfg,upper_drive='explicit_pd')
    env.reset()
    m,d=env.model,env.data
    baseline=d.qpos.copy()
    rng=np.random.default_rng(20260927)
    errors={}
    axes_errors={}
    for sample in range(args.samples):
        d.qpos[:]=baseline
        if sample:
            d.qpos[env.qpos_adr]+=rng.uniform(-.2,.2,27)
        mujoco.mj_forward(m,d)
        for path,base_name,rename in ((G1_GRASP_URDF,'pelvis',{}),(CART_URDF,'cart_base',{'base_link':'cart_base'})):
            base_id=m.body(base_name).id
            pose=np.eye(4)
            pose[:3,:3]=d.xmat[base_id].reshape(3,3)
            pose[:3,3]=d.xpos[base_id]
            joint_names=[j.get('name') for j in ET.parse(path).getroot().findall('joint') if j.get('type')!='fixed']
            values={name:float(d.qpos[m.joint(name).qposadr[0]]) for name in joint_names}
            poses,axes=urdf_fk(path,pose,values)
            for name,expected in poses.items():
                actual_name=rename.get(name,name)
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,actual_name)
                if bid<0:
                    continue  # Fixed links may be folded into their parent.
                pos=float(np.linalg.norm(d.xpos[bid]-expected[:3,3]))
                angle=float(Rotation.from_matrix(d.xmat[bid].reshape(3,3)@expected[:3,:3].T).magnitude())
                row=errors.setdefault(actual_name,dict(position_max_m=0.,rotation_max_rad=0.))
                row['position_max_m']=max(row['position_max_m'],pos)
                row['rotation_max_rad']=max(row['rotation_max_rad'],angle)
            for name in joint_names:
                jid=m.joint(name).id
                point,axis=axes[name]
                row=axes_errors.setdefault(name,dict(anchor_max_m=0.,axis_max_error=0.))
                row['anchor_max_m']=max(row['anchor_max_m'],float(np.linalg.norm(d.xanchor[jid]-point)))
                row['axis_max_error']=max(row['axis_max_error'],float(np.linalg.norm(d.xaxis[jid]-axis)))
    report=dict(baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                samples=args.samples,model=model_report(env),body_kinematics=errors,joint_kinematics=axes_errors,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (G1_GRASP_URDF,CART_URDF,env.xml_path)},
                equality_constraints=[dict(name=m.eq(i).name,type=int(m.eq_type[i]),body1=m.body(int(m.eq_obj1id[i])).name,
                                           body2=m.body(int(m.eq_obj2id[i])).name,data=m.eq_data[i].tolist(),
                                           solref=m.eq_solref[i].tolist(),solimp=m.eq_solimp[i].tolist()) for i in range(m.neq)],
                contact_geometries=[dict(name=m.geom(i).name,body=m.body(int(m.geom_bodyid[i])).name,
                                         type=int(m.geom_type[i]),contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),
                                         margin=float(m.geom_margin[i]),gap=float(m.geom_gap[i]),size=m.geom_size[i].tolist(),
                                         friction=m.geom_friction[i].tolist()) for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]],
                options=dict(integrator=int(m.opt.integrator),solver=int(m.opt.solver),iterations=int(m.opt.iterations),
                             tolerance=float(m.opt.tolerance),cone=int(m.opt.cone),timestep=float(m.opt.timestep)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'asset_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(max_position_m=max(r['position_max_m'] for r in errors.values()),
                          max_rotation_rad=max(r['rotation_max_rad'] for r in errors.values()),
                          worst_bodies=sorted(errors.items(),key=lambda kv:kv[1]['position_max_m'],reverse=True)[:6],
                          worst_joints=sorted(axes_errors.items(),key=lambda kv:kv[1]['axis_max_error'],reverse=True)[:6],
                          equality=report['equality_constraints'],options=report['options']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_sim2sim_root_cause')
    main(p.parse_args())
