"""Identify inertia-frame semantics using an isolated body and known torques."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab_physx.physics import PhysxCfg
    from pxr import UsdPhysics, Gf
    from scipy.spatial.transform import Rotation
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device,dt=.001,gravity=(0.,0.,0.),physics=PhysxCfg()))
    cfg=RigidObjectCfg(prim_path='/World/InertiaProbe',spawn=sim_utils.CuboidCfg(size=(.4,.3,.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,linear_damping=0.,angular_damping=0.),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.),collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,3.)))
    obj=RigidObject(cfg)
    diag=np.array([.05,.08,.1])
    q=Rotation.from_euler('xyz',[.2,.5,.3]).as_quat()
    rotation=Rotation.from_quat(q).as_matrix()
    api=UsdPhysics.MassAPI.Apply(sim.stage.GetPrimAtPath('/World/InertiaProbe'))
    api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diag))
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.,0.,0.))
    sim.reset();obj.update(.001)
    reported=np.asarray(obj.data.body_inertia.torch[0].cpu()).reshape(3,3).T
    responses=[]
    for axis in range(3):
        obj.write_root_pose_to_sim_index(root_pose=torch.tensor([[0.,0.,3.,0.,0.,0.,1.]],device=args.device))
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1,6,device=args.device))
        torque=torch.zeros(1,1,3,device=args.device);torque[0,0,axis]=1.
        obj.permanent_wrench_composer.set_forces_and_torques_index(forces=torch.zeros_like(torque),torques=torque,is_global=True)
        obj.write_data_to_sim();sim.step(render=False);obj.update(.001)
        responses.append(obj.data.body_com_vel_w.torch[0,0,3:].cpu().numpy()/.001)
    measured=np.linalg.inv(np.stack(responses,axis=1))
    expected=rotation@np.diag(diag)@rotation.T
    reversed_frame=rotation.T@np.diag(diag)@rotation
    report=dict(usd_principal_xyzw=q.tolist(),diagonal=diag.tolist(),reported_tensor=reported.tolist(),
                physically_identified_tensor=measured.tolist(),usd_expected_tensor=expected.tolist(),
                reversed_rotation_tensor=reversed_frame.tolist(),
                measured_minus_reported_norm=float(np.linalg.norm(measured-reported)),
                measured_minus_usd_expected_norm=float(np.linalg.norm(measured-expected)),
                measured_minus_reversed_norm=float(np.linalg.norm(measured-reversed_frame)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args)
    try:run(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:app.app.close()
