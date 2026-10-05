"""A separate 20 s zero-command diagnostic, never used as paper step data."""
import argparse
import json
import os
from pathlib import Path
import uuid

import collect_motion_ability_isaac as collector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--release-grasp', action='store_true', help='Diagnostic only: disable both hand-cart fixed joints before physics starts')
    parser.add_argument('--explicit-upper', action='store_true', help='Diagnostic only: apply the same PD gains as explicit efforts to upper joints')
    parser.add_argument('--output', type=Path, default=collector.ROOT / 'logs/Picture/motion_ability/diagnostics/zero20')
    args = parser.parse_args()
    if args.release_grasp and args.explicit_upper:
        parser.error('Use one intervention per diagnostic run')
    paper = json.loads((collector.ROOT / 'logs/Picture/motion_ability/manifest.json').read_text(encoding='utf-8'))
    os.environ[collector.NONCE] = uuid.uuid4().hex
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostic = dict(run_id=os.environ[collector.NONCE], duration_s=args.duration,
                      all_commands_zero=True, release_grasp=args.release_grasp, explicit_upper=args.explicit_upper,
                      paper_data_replacement=False, asset_files_modified=False,
                      intervention='Both hand-cart fixed joints disabled only in the live diagnostic stage'
                                   if args.release_grasp else ('Upper-body drive changed from implicit position to explicit PD effort at every physics substep; gains and torque limits unchanged'
                                                               if args.explicit_upper else 'None; original plant and controller'))
    (args.output / 'diagnostic.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')
    if args.release_grasp:
        original_configure = collector.configure
        def configure(checkpoint, duration):
            cfg, task = original_configure(checkpoint, duration)
            from isaaclab.sim import SimulationContext
            from pxr import UsdPhysics
            original_reset = SimulationContext.reset
            def reset_with_released_grasp(context, *positional, **keywords):
                joints = []
                for side in ['left', 'right']:
                    path = f'/World/envs/env_0/Scene/GraspJoints/{side}'
                    prim = context.stage.GetPrimAtPath(path)
                    assert prim.IsValid(), path
                    while prim.IsInstanceProxy():
                        parent = prim.GetParent()
                        while parent.IsInstanceProxy():
                            parent = parent.GetParent()
                        parent.SetInstanceable(False)
                        prim = context.stage.GetPrimAtPath(path)
                    joint = UsdPhysics.FixedJoint(prim)
                    assert joint, path
                    joint.CreateJointEnabledAttr().Set(False)
                    assert joint.GetJointEnabledAttr().Get() is False
                    joints.append(dict(path=path, enabled=False,
                                       body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                                       body1=[str(p) for p in joint.GetBody1Rel().GetTargets()]))
                (args.output / 'released_joints.json').write_text(json.dumps(joints, indent=2), encoding='utf-8')
                return original_reset(context, *positional, **keywords)
            SimulationContext.reset = reset_with_released_grasp
            return cfg, task
        collector.configure = configure
    if args.explicit_upper:
        original_configure = collector.configure
        def configure(checkpoint, duration):
            cfg, task = original_configure(checkpoint, duration)
            import torch
            import types
            from homie_pushcart_v6.task_env import PushCartEnv
            original_init = PushCartEnv.__init__
            def explicit_action(this):
                assert not this.cfg.domain_rand.delay
                this.joint_pos_target = this.default_dof_pos + this.cfg.control.action_scale * this.actions
                tau = this.p_gains * this.Kp_factors * (this.joint_pos_target - this.dof_pos) - this.d_gains * this.Kd_factors * this.dof_vel
                tau = (tau + this.actuation_offset + this.joint_injection).clamp(-this.torque_limits, this.torque_limits)
                this.robot.set_joint_effort_target_index(target=tau, joint_ids=this.joint_ids)
                this._physics_substep += 1
            def init_with_explicit_upper(this, *positional, **keywords):
                original_init(this, *positional, **keywords)
                zeros = torch.zeros(this.num_envs, 15, device=this.device)
                this.robot.write_joint_stiffness_to_sim_index(stiffness=zeros, joint_ids=this.joint_ids[12:])
                this.robot.write_joint_damping_to_sim_index(damping=zeros, joint_ids=this.joint_ids[12:])
                this._apply_action = types.MethodType(explicit_action, this)
                data = dict(implicit_upper_kp=this.robot.data.joint_stiffness.torch[:, this.joint_ids[12:]].cpu().tolist(),
                            implicit_upper_kd=this.robot.data.joint_damping.torch[:, this.joint_ids[12:]].cpu().tolist(),
                            explicit_kp=this.p_gains.cpu().tolist(), explicit_kd=this.d_gains.cpu().tolist(),
                            torque_limits=this.torque_limits.cpu().tolist())
                (args.output / 'explicit_upper.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
            PushCartEnv.__init__ = init_with_explicit_upper
            return cfg, task
        collector.configure = configure
    collector.protocol = lambda axis, idle_s, plateau_s: [dict(start_s=0., end_s=args.duration, value=0.)]
    child = argparse.Namespace(output=args.output, axis='wz', method='baseline',
                               checkpoint=Path(paper['checkpoint']), iteration=paper['iteration'],
                               seed=paper['seed'], payload=paper['payload_kg'], device=paper['device'],
                               idle_s=args.duration, plateau_s=8.)
    collector.run_child(child)


if __name__ == '__main__':
    main()
