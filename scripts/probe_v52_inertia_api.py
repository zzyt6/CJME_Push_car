"""Bounded PhysX inertia read/write and frame probe; never edits training files."""
import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def run(args):
    import torch
    from pxr import Usd, UsdPhysics
    from homie_pushcart_v52.sim_env import PushCartSimEnv
    from homie_pushcart_v52.teleop import play_config
    from homie_pushcart_v52.task_config import PushCartTaskCfg

    snapshots={}
    def save(key,asset):
        snapshots[key]=dict(names=asset.body_names,inertia=asset.data.body_inertia.torch[0].cpu().tolist(),
                            com=asset.data.body_com_pose_b.torch[0].cpu().tolist(),
                            pose=asset.data.body_link_pose_w.torch[0].cpu().tolist(),
                            joint_names=asset.joint_names,joint_pos=asset.data.joint_pos.torch[0].cpu().tolist())
    class Probe(PushCartSimEnv):
        def _randomize_body_properties(self):
            save('robot_before_randomizer',self.robot)
            super()._randomize_body_properties()
            save('robot_after_randomizer',self.robot)
    cfg=play_config(1,10.)
    cfg.pushcart_upper_drive='explicit_pd'
    e=Probe(cfg,ROOT,args.device,True,with_cart=True,full_dynamics=False)
    print('physics API file',inspect.getfile(type(e.robot.root_view)),flush=True)
    source=inspect.getfile(type(e.robot.root_view))
    usd_bodies={}
    for prim in Usd.PrimRange(e.sim.stage.GetPrimAtPath('/World/envs/env_0/Scene'),Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api=UsdPhysics.MassAPI(prim);q=api.GetPrincipalAxesAttr().Get()
            usd_bodies[str(prim.GetPath())]=dict(diagonal=list(api.GetDiagonalInertiaAttr().Get()),
                                               principal_xyzw=[*q.GetImaginary(),q.GetReal()])
    for name,asset in (('robot',e.robot),('cart',e.cart)):
        save(name+'_before_roundtrip',asset)
        inertia=asset.data.body_inertia.torch.clone()
        com=asset.data.body_com_pose_b.torch.clone()
        asset.set_inertias_index(inertias=inertia)
        save(name+'_after_set_inertia',asset)
        asset.set_coms_index(coms=com)
        save(name+'_after_restore_com',asset)
    # Query after stepping and a changed world orientation. A local-body
    # tensor must not follow the robot's world orientation.
    pose=e.robot.data.root_link_pose_w.torch.clone()
    from homie_isaaclab.math_utils import quat_mul
    import math
    q=torch.tensor([[0.,0.,math.sin(.3/2),math.cos(.3/2)]],device=args.device)
    pose[:,3:]=quat_mul(q,pose[:,3:])
    e.robot.write_root_pose_to_sim_index(root_pose=pose)
    e.sim.step(render=False)
    e.scene.update(cfg.sim.dt)
    save('robot_after_world_motion',e.robot)
    save('cart_after_step',e.cart)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(dict(api_source=source,snapshots=snapshots,usd_bodies=usd_bodies),indent=2),encoding='utf-8')
    print('inertia_probe complete',flush=True)


if __name__=='__main__':
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args();app=AppLauncher(args)
    try:run(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:app.app.close()
