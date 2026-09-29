"""Audit force labels during real closed-loop cart pushing, with floor contacts.

Read-only instrumentation: independent shoulder-subtree force balance per hand,
cart/robot assembly balance using measured ground loads, and actual runtime
properties. It does not replace cart labels by prescribed hand forces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
from eval_pushcart_v52_sim2sim import DEFAULT_CKPT, COMMANDS


def run(args):
    import numpy as np
    import torch
    import warp as wp
    from pxr import Usd, UsdPhysics
    from homie_pushcart_v52.control_contract import nominal_configs
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.hl_policy import HighLevelPolicy
    from homie_pushcart_v52.paths import G1_GRASP_URDF
    from homie_pushcart_v52.dynamics import rotation_matrix_xyzw
    from homie_pushcart_v52.transfer_estimator import install_estimator

    torch.set_num_threads(4)
    torch.manual_seed(0)

    class AuditEnv(PushCartEnv):
        audit_recording = False

        def _sample_dynamics(self, substep):
            super()._sample_dynamics(substep)
            if not self.audit_recording:
                return
            required, damping, states = {}, {}, {}
            gravity = torch.tensor(self.cfg.sim.gravity, device=self.device)
            for name, asset in (('robot', self.robot), ('cart', self.cart)):
                mass, pos, vel, inertia = self._body_dynamics(asset)
                acc = asset.data.body_com_acc_w.torch
                center = (mass[..., None]*pos).sum(1)/mass.sum(1)[:, None]
                effective = mass[..., None]*(acc[..., :3]-gravity)
                spin = (inertia @ vel[..., 3:, None]).squeeze(-1)
                angular = ((inertia @ acc[..., 3:, None]).squeeze(-1) +
                           torch.cross(vel[..., 3:], spin, dim=-1) +
                           torch.cross(pos-center[:, None], effective, dim=-1)).sum(1)
                required[name] = torch.cat((effective.sum(1), angular), -1)
                df = -self.cfg.asset.linear_damping*mass[..., None]*vel[..., :3]
                dm = -self.cfg.asset.angular_damping*spin
                damping[name] = torch.cat((df.sum(1), (dm+torch.cross(pos-center[:, None], df, dim=-1)).sum(1)), -1)
                states[name] = mass, pos, vel, inertia, acc, center
            # Separate incoming-force channels, proximal to the whole arms.
            # Joint actuation contributes torque, not net external linear force.
            all_incoming = wp.to_torch(self.robot.root_view.get_link_incoming_joint_force()).reshape(1, -1, 6)
            rotation = rotation_matrix_xyzw(self.robot.data.body_link_pose_w.torch[:, self.audit_shoulder_ids, 3:])
            incoming = (rotation @ all_incoming[:, self.audit_shoulder_ids, :3, None]).squeeze(-1)
            mass, pos, vel, inertia, acc, center = states['robot']
            arm_inertial = torch.stack([(mass[:, ids, None]*(acc[:, ids, :3]-gravity)).sum(1)
                                        for ids in self.audit_arm_ids], 1)
            arm_damping = torch.stack([(-self.cfg.asset.linear_damping*mass[:, ids, None]*vel[:, ids, :3]).sum(1)
                                       for ids in self.audit_arm_ids], 1)
            # Remove the extra arm damping that the palm-only label excludes.
            palm_damping = torch.stack([(-self.cfg.asset.linear_damping*mass[:, ids, None]*vel[:, ids, :3]).sum(1)
                                        for ids in self._hand_subtree_ids], 1)
            shoulder_force = incoming-arm_inertial+arm_damping-palm_damping
            fast_mass, fast_pos, fast_vel, fast_inertia = self._body_dynamics(self.robot, self._hand_body_ids)
            fast, _, _ = self._measure_hand_wrench(fast_mass, fast_pos, fast_vel, fast_inertia,
                                                   acc[:, self._hand_body_ids], self._hand_local_ids)
            values = dict(hand_wrench=self._hand_wrench, shoulder_force=shoulder_force,
                          shoulder_incoming=incoming, arm_inertial=arm_inertial,
                          fast_minus_full=fast-self._hand_wrench,
                          cart_required=required['cart'], robot_required=required['robot'],
                          cart_damping=damping['cart'], robot_damping=damping['robot'],
                          cart_hand=self.physics_samples['cart_hand_wrench'][substep],
                          robot_hand=self.physics_samples['robot_reaction_wrench'][substep],
                          cart_ground=self._wheel_wrench.sum(1), robot_ground=self._feet_wrench.sum(1),
                          cart_applied=self.physics_samples['cart_applied_wrench_com'][substep],
                          dq=self.robot.data.joint_vel.torch[:, self.joint_ids],
                          q=self.robot.data.joint_pos.torch[:, self.joint_ids],
                          target=self.joint_pos_target, ground_valid=self.ground_wrench_valid,
                          label_valid=self.dynamics_wrench_valid)
            for k, value in values.items():
                self.audit_rows.setdefault(k, []).append(value[0].detach().cpu().numpy().copy())

    cfg, task = nominal_configs(args.checkpoint, 'explicit_pd', args.steps*.02+5.)
    task.diagnostics.full_dynamics = True
    low = Path(cfg.pushcart_saved_checkpoint)
    if not low.is_absolute():
        low = ROOT/low
    env = AuditEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=low)
    env.fixed_caster_angle = 0.
    hl = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    install_estimator(hl.actor, args.estimator)
    parents = {j.find('child').get('link'): j.find('parent').get('link') for j in ET.parse(G1_GRASP_URDF).getroot().findall('joint')}
    env.audit_arm_ids, env.audit_shoulder_ids = [], []
    for side in ('left', 'right'):
        name = side+'_shoulder_pitch_link'
        names = {name}
        while True:
            more = names | {c for c, p in parents.items() if p in names}
            if more == names:
                break
            names = more
        env.audit_shoulder_ids.append(env.body_names.index(name))
        env.audit_arm_ids.append(torch.tensor([i for i, n in enumerate(env.body_names) if n in names], device=args.device))
    env.audit_shoulder_ids = torch.tensor(env.audit_shoulder_ids, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)

    def runtime_properties():
        result = dict(physics=env.physics_report, bodies={}, joints={}, stage_rigid_properties=[])
        for name, asset in (('robot', env.robot), ('cart', env.cart)):
            result['bodies'][name] = dict(names=asset.body_names,
                mass=asset.data.body_mass.torch[0].cpu().tolist(),
                inertia=asset.data.body_inertia.torch[0].cpu().tolist(),
                com_pose=asset.data.body_com_pose_b.torch[0].cpu().tolist(),
                materials=wp.to_torch(asset.root_view.get_material_properties())[0].cpu().tolist())
            result['joints'][name] = dict(names=asset.joint_names,
                stiffness=asset.data.joint_stiffness.torch[0].cpu().tolist(),
                damping=asset.data.joint_damping.torch[0].cpu().tolist(),
                armature=asset.data.joint_armature.torch[0].cpu().tolist(),
                friction=asset.data.joint_friction_coeff.torch[0].cpu().tolist(),
                velocity_limits=asset.data.joint_vel_limits.torch[0].cpu().tolist(),
                effort_limits=asset.data.joint_effort_limits.torch[0].cpu().tolist())
        for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Scene'), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                row = dict(path=str(prim.GetPath()))
                for field in ('linearDamping','angularDamping','disableGravity','enableGyroscopicForces','maxLinearVelocity','maxAngularVelocity','maxDepenetrationVelocity'):
                    row[field] = prim.GetAttribute('physxRigidBody:'+field).Get()
                result['stage_rigid_properties'].append(row)
        return result

    conditions, all_traces = [], []
    for payload in args.payloads:
        for command_name in args.commands:
            env.audit_recording = False
            env.fixed_payload_kg, env.fixed_command = payload, COMMANDS[command_name]
            with torch.inference_mode():
                obs, _ = env.reset()
            properties = runtime_properties()
            (args.output/f'runtime_{payload:g}_{command_name}.json').write_text(json.dumps(properties, indent=2), encoding='utf-8')
            env.audit_rows = {}
            failure = None
            for step in range(args.steps):
                env.audit_recording = step >= round(args.warmup_s/env.dt)
                with torch.inference_mode():
                    obs, _, _, done, _ = env.step(hl(obs, estimator_obs=env.falcon_obs))
                if bool(done[0]):
                    failure = 'termination/reset; terminal samples retained and marked'
                    break
            arrays = {k: np.asarray(v) for k, v in env.audit_rows.items()}
            finite = all(np.isfinite(v).all() for v in arrays.values())
            cart_error = arrays['cart_hand']+arrays['cart_ground']+arrays['cart_applied']+arrays['cart_damping']-arrays['cart_required']
            robot_error = arrays['robot_hand']+arrays['robot_ground']+arrays['robot_damping']-arrays['robot_required']
            arm_error = arrays['hand_wrench'][..., :3]-arrays['shoulder_force']
            rms = lambda x: np.sqrt(np.mean(x*x, axis=0)).tolist()
            row = dict(payload_kg=payload, command=command_name, steps=step+1, samples=len(cart_error),
                       failure=failure, finite=finite, ground_valid_fraction=float(arrays['ground_valid'].mean()),
                       label_valid_fraction=float(arrays['label_valid'].mean()),
                       cart_balance_component_rms=rms(cart_error), robot_balance_component_rms=rms(robot_error),
                       per_hand_shoulder_minus_palm_force_rms=rms(arm_error),
                       fast_minus_full_component_rms=rms(arrays['fast_minus_full']),
                       max_joint_velocity_rad_s=np.max(abs(arrays['dq']), 0).tolist())
            arrays['condition_id'] = np.full(len(cart_error), len(conditions))
            all_traces.append(arrays)
            conditions.append(row)
            print('walking_dynamics', payload, command_name, 'cart rms', row['cart_balance_component_rms'], 'arm rms', row['per_hand_shoulder_minus_palm_force_rms'], flush=True)
            (args.output/'partial.json').write_text(json.dumps(conditions, indent=2), encoding='utf-8')
    joined = {k: np.concatenate([a[k] for a in all_traces]) for k in all_traces[0]}
    np.savez_compressed(args.output/'traces.npz', **joined)
    report = dict(conditions=conditions, requested_conditions=len(args.payloads)*len(args.commands),
                  completed_conditions=sum(c['failure'] is None for c in conditions),
                  total_samples=len(joined['hand_wrench']), checkpoint=str(args.checkpoint), estimator=str(args.estimator),
                  full_dynamics=True, floor_and_cart_contacts=True,
                  note='Independent proximal-arm and assembly balance checks during frozen-policy cart pushing. Ground-force extraction is also under audit; a residual alone does not identify the hand label as wrong.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('walking_dynamics complete', report['completed_conditions'], 'samples', report['total_samples'], flush=True)


if __name__ == '__main__':
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--estimator', type=Path, default=ROOT/'logs/bench/v52_falcon_explicit/model_explicit_pd_full_22/estimator.pt')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--warmup_s', type=float, default=2.)
    parser.add_argument('--payloads', type=float, nargs='+', default=[0.,12.5,25.])
    parser.add_argument('--commands', nargs='+', choices=tuple(COMMANDS), default=list(COMMANDS))
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        run(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
