"""Compare measured PhysX body/joint properties to compiled MuJoCo properties.

PhysX retains fixed palm/finger bodies that MuJoCo folds into wrist bodies.
Aggregate them using fixed URDF transforms before comparing inertias and COMs.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
from audit_v52_sim2sim_assets import urdf_fk
from homie_pushcart_v52.paths import G1_GRASP_URDF, CART_URDF
from homie_pushcart_v52.control_contract import nominal_configs
from pushcart_bench.policies_v5x import InstrumentedEnv
from pushcart_bench.sim2sim_scene import set_payload
from eval_pushcart_v52_sim2sim import DEFAULT_CKPT


def aggregate_runtime(properties, urdf, model, rename):
    parents = {j.find('child').get('link'): j.find('parent').get('link') for j in ET.parse(urdf).getroot().findall('joint')}
    poses, _ = urdf_fk(urdf, np.eye(4), {})
    grouped = {}
    for i, name in enumerate(properties['names']):
        parent = name
        while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rename.get(parent, parent)) < 0:
            parent = parents[parent]
        rel = np.linalg.inv(poses[parent]) @ poses[name]
        mass = properties['mass'][i]
        position = rel[:3, 3]+rel[:3, :3] @ np.array(properties['com_pose'][i][:3])
        inertia = rel[:3, :3] @ np.array(properties['inertia'][i]).reshape(3, 3).T @ rel[:3, :3].T
        grouped.setdefault(rename.get(parent, parent), []).append((mass, position, inertia))
    result = {}
    for name, parts in grouped.items():
        mass = sum(p[0] for p in parts)
        com = sum(p[0]*p[1] for p in parts)/mass
        inertia = sum(p[2]+p[0]*(np.dot(p[1]-com, p[1]-com)*np.eye(3)-np.outer(p[1]-com, p[1]-com)) for p in parts)
        body = model.body(name)
        rotation = Rotation.from_quat(np.roll(body.iquat, -1)).as_matrix()
        mj_inertia = rotation @ np.diag(body.inertia) @ rotation.T
        result[name] = dict(folded_bodies=len(parts), mass_abs_error_kg=abs(float(body.mass[0])-mass),
                            com_error_m=float(np.linalg.norm(body.ipos-com)),
                            inertia_error_kg_m2=float(np.linalg.norm(mj_inertia-inertia)),
                            physx_mass_kg=mass, mujoco_mass_kg=float(body.mass[0]))
    return result


def main(args):
    cfg, _ = nominal_configs(DEFAULT_CKPT, 'explicit_pd')
    env = InstrumentedEnv(True, scene_profile='isaac', cfg=cfg, upper_drive='explicit_pd', inertia_profile=args.inertia_profile)
    results = {}
    for payload in (0.,12.5,25.):
        runtime = json.loads((args.runtime/f'runtime_{payload:g}_forward03.json').read_text(encoding='utf-8'))
        set_payload(env, payload)
        bodies = {}
        bodies.update(aggregate_runtime(runtime['bodies']['robot'], G1_GRASP_URDF, env.model, {}))
        bodies.update(aggregate_runtime(runtime['bodies']['cart'], CART_URDF, env.model, {'base_link':'cart_base'}))
        joints = {}
        for group, values in runtime['joints'].items():
            for i, name in enumerate(values['names']):
                j = env.model.joint(name)
                dof = j.dofadr[0]
                row = dict(armature_abs_error=abs(values['armature'][i]-float(env.model.dof_armature[dof])),
                           passive_friction_abs_error=abs(values['friction'][i]-float(env.model.dof_frictionloss[dof])),
                           physx_native_stiffness=values['stiffness'][i],
                           physx_native_damping=values['damping'][i],
                           physx_velocity_limit=values['velocity_limits'][i])
                if group == 'robot':
                    row['effort_limit_abs_error_nm'] = abs(values['effort_limits'][i]-float(env.model.jnt_actfrcrange[j.id, 1]))
                    row['native_drive_disabled'] = values['stiffness'][i] == values['damping'][i] == 0.
                else:
                    row['passive_damping_abs_error'] = abs(values['damping'][i]-float(env.model.dof_damping[dof]))
                joints[name] = row
        results[str(payload)] = dict(bodies=bodies, joints=joints,
            max_body_errors={k: max(v[k] for v in bodies.values()) for k in ('mass_abs_error_kg','com_error_m','inertia_error_kg_m2')})
    trace = np.load(args.runtime/'traces.npz')
    joint_props = runtime['joints']['robot']
    # Explicitly use HOMIE ordering, independent of dictionary insertion order.
    from homie_pushcart_v52.paths import HOMIE_JOINTS
    limits = np.array([joint_props['velocity_limits'][joint_props['names'].index(n)] for n in HOMIE_JOINTS])
    speed_ratio = abs(trace['dq'])/limits
    report = dict(payloads=results, inertia_profile=args.inertia_profile,
                  compared_to='actual PhysX runtime property arrays; fixed-body inertia aggregation',
                  body_transform_source='training URDF fixed-joint transforms (validated separately against compiled MuJoCo FK)',
                  maximum_observed_joint_velocity_limit_ratio=float(speed_ratio.max()),
                  physics_samples_at_99pct_joint_velocity_limit=int((speed_ratio >= .99).any(-1).sum()),
                  total_physics_samples=len(speed_ratio),
                  limits_note='MuJoCo has no equivalent velocity cap in this scene; this records whether the PhysX test reaches its cap.',
                  runtime_rigid_properties=runtime['stage_rigid_properties'])
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(maxima={k:v['max_body_errors'] for k,v in results.items()},
                         maximum_observed_joint_velocity_limit_ratio=report['maximum_observed_joint_velocity_limit_ratio'],
                         samples_at_limit=report['physics_samples_at_99pct_joint_velocity_limit']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, default=ROOT/'logs/bench/v52_sim2sim_root_cause/walking_dynamics')
    p.add_argument('--output', type=Path, default=ROOT/'logs/bench/v52_sim2sim_root_cause/runtime_parameters.json')
    p.add_argument('--inertia_profile', choices=('urdf', 'physx'), default='urdf')
    main(p.parse_args())
