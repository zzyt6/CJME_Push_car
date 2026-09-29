"""Read compiled scene differences and test isolated changes with fixed policies.

All XMLs and rollouts are written to a new audit directory. Production scene
builders, model assets, checkpoints, controllers and default profiles are read-only.
"""
import argparse
import collections
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL'), str(ROOT / 'scripts')]
from eval_homie_body_velocity import LOW, cases, command, config_free
from eval_homie_v2_native_mujoco import official_observation_builder, session
from eval_v53_estimator_sim2sim_status import metrics

OURS = ROOT / 'logs/bench/homie_body_velocity_20260928/mujoco_free_final/free_robot.xml'
OFFICIAL = ROOT / '.cache/homie_official_weights/v2_native/g1_gear_wbc.xml'
WEIGHTS = ROOT / '.cache/homie_official_weights'
FEET = ['left_ankle_roll_link', 'right_ankle_roll_link']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def serialize(v):
    return v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, np.generic) else v)


def inventory(path, name):
    m = mujoco.MjModel.from_xml_path(str(path))
    xml_dt = float(m.opt.timestep)
    m.opt.timestep = .005  # Both actual entrypoints override/set this physics period.
    bodies = {}
    for b in range(m.nbody):
        rot = np.empty(9); mujoco.mju_quat2Mat(rot, m.body_iquat[b]); rot = rot.reshape(3, 3)
        bodies[m.body(b).name] = dict(mass=m.body_mass[b], com=m.body_ipos[b],
                                      inertia_body=rot @ np.diag(m.body_inertia[b]) @ rot.T,
                                      parent=m.body(m.body_parentid[b]).name, joints=int(m.body_jntnum[b]))
    joints = {}
    for j in range(m.njnt):
        d = m.jnt_dofadr[j]; width = 6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        joints[m.joint(j).name] = dict(type=int(m.jnt_type[j]), range=m.jnt_range[j],
                                      torque_range=m.jnt_actfrcrange[j], torque_limited=int(m.jnt_actfrclimited[j]),
                                      damping=m.dof_damping[d:d+width], frictionloss=m.dof_frictionloss[d:d+width],
                                      armature=m.dof_armature[d:d+width])
    geoms = []
    for g in range(m.ngeom):
        geoms.append(dict(id=g, name=m.geom(g).name, body=m.body(m.geom_bodyid[g]).name,
                          type=int(m.geom_type[g]), position=m.geom_pos[g], size=m.geom_size[g],
                          active=bool(m.geom_contype[g] or m.geom_conaffinity[g]),
                          contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
                          condim=int(m.geom_condim[g]), friction=m.geom_friction[g],
                          solref=m.geom_solref[g], solimp=m.geom_solimp[g]))
    pairs = [dict(name=m.pair(i).name, geom1=m.geom(m.pair_geom1[i]).name, geom2=m.geom(m.pair_geom2[i]).name,
                  dimension=int(m.pair_dim[i]), friction=m.pair_friction[i], solref=m.pair_solref[i], solimp=m.pair_solimp[i])
             for i in range(m.npair)]
    opts = {k: getattr(m.opt, k) for k in ('integrator', 'solver', 'cone', 'iterations', 'ls_iterations',
                                         'tolerance', 'impratio', 'gravity', 'disableflags', 'enableflags')}
    return dict(name=name, source=str(path), sha256=sha(path), nq=m.nq, nv=m.nv, nu=m.nu, neq=m.neq,
                xml_timestep=xml_dt, runtime_timestep=.005, mass=float(m.body_mass.sum()), bodies=bodies,
                joints=joints, geoms=geoms, pairs=pairs, options=opts, exclusions=int(m.nexclude))


def load_xml(path):
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    meshdir = Path(compiler.get('meshdir', '.'))
    if not meshdir.is_absolute(): compiler.set('meshdir', str((path.parent / meshdir).resolve()))
    return root


def foot_elements(root):
    return [g for n in FEET for g in root.find(f".//body[@name='{n}']").findall('geom')
            if int(g.get('contype', '1')) or int(g.get('conaffinity', '1'))]


def disable_foot_contacts(root):
    for g in foot_elements(root):
        g.set('contype', '0'); g.set('conaffinity', '0')
    for contact in root.findall('contact'):
        for child in list(contact):
            if child.tag == 'pair' and any('ankle_roll' in child.get(k, '') for k in ('geom1', 'geom2')):
                contact.remove(child)


def add_floor_pairs(root, geoms, mu):
    contact = root.find('contact')
    if contact is None: contact = ET.SubElement(root, 'contact')
    names = {g.get('name') for g in geoms}
    for c in list(contact):
        if c.tag == 'pair' and (c.get('geom1') in names or c.get('geom2') in names): contact.remove(c)
    for i, g in enumerate(geoms):
        if not g.get('name'): g.set('name', f'audit_foot_contact_{i}')
        ET.SubElement(contact, 'pair', name=f'audit_floor_{i}', geom1=g.get('name'), geom2='floor',
                      condim='3', friction=f'{mu} {mu} 0 0 0')


def variant_xml(base, changes, output):
    source = OURS if base == 'ours' else OFFICIAL
    root = load_xml(source)
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.set('timestep', '.005')
    if 'integrator' in changes:
        option.set('integrator', 'Euler' if base == 'ours' else 'implicitfast')
    if 'geometry' in changes:
        disable_foot_contacts(root)
        if base == 'ours':
            reference = load_xml(OFFICIAL)
            for n in FEET:
                body = root.find(f".//body[@name='{n}']")
                spheres = [g for g in reference.find(f".//body[@name='{n}']").findall('geom')
                           if g.get('type', 'sphere') == 'sphere' and not g.get('mesh')]
                assert len(spheres) == 4
                for i, g in enumerate(spheres):
                    g = copy.deepcopy(g); g.set('name', f'audit_{n}_{i}'); g.set('contype', '1'); g.set('conaffinity', '1')
                    body.append(g)
            add_floor_pairs(root, foot_elements(root), 1. if 'friction' in changes else .75)
        else:
            reference = load_xml(OURS)
            for n in FEET:
                g = copy.deepcopy(reference.find(f".//geom[@name='isaac_{n}_0']"))
                mesh = copy.deepcopy(reference.find(f".//mesh[@name='{g.get('mesh')}']"))
                meshfile = Path(mesh.get('file'))
                if not meshfile.is_absolute(): mesh.set('file', str((Path(reference.find('compiler').get('meshdir')) / meshfile).resolve()))
                root.find('asset').append(mesh)
                g.set('friction', '1 .005 .0001'); root.find(f".//body[@name='{n}']").append(g)
    if 'friction' in changes:
        add_floor_pairs(root, foot_elements(root), 1. if base == 'ours' else .75)
    if 'passive' in changes:
        for j in root.iter('joint'):
            if j.get('name'):
                j.set('damping', '.001' if base == 'ours' else '0')
                j.set('frictionloss', '.1' if base == 'ours' else '0')
                j.set('armature', '.01')
            elif j in root.find('default').findall('joint'):
                j.set('damping', '.001' if base == 'ours' else '0')
                j.set('frictionloss', '.1' if base == 'ours' else '0')
                j.set('armature', '.01' if base == 'ours' else '0')
        # freejoint shorthand overrides inherited defaults: a named joint type
        # explicitly assigns all six free-DOF properties in this intervention.
        for body in root.iter('body'):
            for j in list(body):
                if j.tag == 'freejoint':
                    body.remove(j)
                    ET.SubElement(body, 'joint', name=j.get('name'), type='free',
                                  damping='.001' if base == 'ours' else '0',
                                  frictionloss='.1' if base == 'ours' else '0', armature='.01' if base == 'ours' else '0')
                elif j.tag == 'joint' and j.get('type') == 'free':
                    j.set('armature', '.01' if base == 'ours' else '0')
    if 'collision_scope' in changes and base == 'v2':
        for body in root.iter('body'):
            if body.get('name') not in FEET:
                for g in body.findall('geom'): g.set('contype', '0'); g.set('conaffinity', '0')
        contact = root.find('contact')
        if contact is None: contact = ET.SubElement(root, 'contact')
        ET.SubElement(contact, 'exclude', body1=FEET[0], body2=FEET[1])
    ET.ElementTree(root).write(output, encoding='unicode')
    return output


def contact_diagnostics(m, d):
    floor = m.geom('floor').id; feet = {m.body(n).id for n in FEET}
    counts = [0, 0]; loaded_speed = []; normal = []
    foot_index = {m.body(n).id: i for i, n in enumerate(FEET)}
    other = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (c.geom1, c.geom2): continue
        g = c.geom2 if c.geom1 == floor else c.geom1
        b = int(m.geom_bodyid[g])
        if b not in feet:
            other += 1; continue
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        if f[0] <= 5.: continue
        counts[foot_index[b]] += 1
        v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        point_v = v[3:] + np.cross(v[:3], c.pos - d.xipos[b])
        loaded_speed.append(float(np.linalg.norm(point_v[:2]))); normal.append(float(f[0]))
    return np.array(counts), float(np.average(loaded_speed, weights=normal)) if normal else 0., other


class OursRunner:
    def __init__(self, xml, low, changes):
        from pushcart_bench.policies_v5x import InstrumentedEnv
        self.low = low
        cfg = config_free(1)
        self.env = e = InstrumentedEnv(False, xml_path=xml, scene_profile='legacy', cfg=cfg, upper_drive='training')
        e.scene_profile = 'isaac'
        e.upper_hold_actions[:] = 0.; e.upper_hold_offset[:] = 0.
        e.preset_dof_pos[:] = e.default_dof_pos; e.preset[e.qpos_adr] = e.default_dof_pos
        e.preset[e.root_qpos:e.root_qpos+3] = cfg.init_state.pos
        e.preset[e.root_qpos+3:e.root_qpos+7] = [1., 0., 0., 0.]
        if 'rigid_damping' in changes: e._body_damping = lambda: e.data.xfrc_applied.fill(0.)
        self.m, self.d = e.model, e.data
        self.dt = e.dt
    def reset(self):
        self.env.reset(); self.env.set_command(0., 0., 0.)
    def step(self, cmd):
        e = self.env
        e.set_command(*cmd); e.obs_hist[-1, :3] = cmd * e.commands_scale
        act, _ = self.low.act_and_estimate(torch.from_numpy(e.obs_hist.reshape(1, -1)))
        obs, fallen = e.step(act[0].numpy())
        pred = self.low.estimate_velocity(torch.from_numpy(obs[None]))[0].numpy() / e.obs_scales['lin_vel']
        return pred, fallen


class V2Runner:
    def __init__(self, xml, models):
        self.m = mujoco.MjModel.from_xml_path(str(xml)); self.m.opt.timestep = .005
        self.d = mujoco.MjData(self.m)
        self.builder = official_observation_builder(self.d, self.m)
        self.cfg = yaml.safe_load((WEIGHTS / 'v2_sim.yaml').read_text())
        for k in ('kps', 'kds', 'default_angles', 'cmd_scale'): self.cfg[k] = np.array(self.cfg[k], dtype=np.float32)
        self.models, self.dt = models, .02
    def reset(self):
        mujoco.mj_resetData(self.m, self.d); mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(15, dtype=np.float32); self.target = self.cfg['default_angles'].copy()
        self.history = collections.deque([np.zeros(86, dtype=np.float32) for _ in range(6)], maxlen=6)
    def step(self, cmd):
        m, d, cfg, b = self.m, self.d, self.cfg, self.builder
        for _ in range(4):
            d.ctrl[:15] = b.pd_control(self.target, d.qpos[7:22], cfg['kps'], np.zeros(15), d.qvel[6:21], cfg['kds'])
            d.ctrl[15:] = b.pd_control(np.zeros(14), d.qpos[22:36], np.full(14, 100.), np.zeros(14), d.qvel[21:35], np.full(14, .5))
            mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
        ctrl = dict(loco_cmd=cmd, height_cmd=cfg['height_cmd'], rpy_cmd=cfg['rpy_cmd'])
        obs, _ = b.compute_observation(d, cfg, self.action, ctrl, 29); self.history.append(obs)
        kind = 'balance' if np.linalg.norm(cmd) <= .05 else 'walk'
        action, state = self.models[kind].run(None, {'input': np.concatenate(self.history)[None]})
        self.action = action[0]; self.target = self.action * cfg['action_scale'] + cfg['default_angles']
        gravity = b.get_gravity_orientation(d.qpos[3:7])
        return state[0, :3]/2., bool(np.linalg.norm(gravity[:2]) > .8 or d.qpos[2] < .35)


def evaluate(runner, directory, selected_cases, steps):
    from homie_pushcart.mujoco_env import quat_rotate_inverse_wxyz
    reports, pooled = [], []
    directory.mkdir()
    for case in selected_cases:
        runner.reset(); cmd = np.zeros(3); rows = {}
        for step in range(steps):
            cmd += min(1., runner.dt/.4) * (command(step*runner.dt, case['scenario']) - cmd)
            pred, fallen = runner.step(cmd)
            m, d = runner.m, runner.d
            b = m.body('pelvis').id; v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            quat = d.qpos[3:7]
            truth = quat_rotate_inverse_wxyz(quat, v[3:])
            count, slip, other = contact_diagnostics(m, d)
            values = dict(time=(step+1)*runner.dt, prediction=pred, truth=truth, command=cmd,
                          gyro=d.qvel[3:6], contact_count=count, contact_point_speed=slip,
                          nonfoot_floor_contacts=other, done=fallen)
            for k, value in values.items(): rows.setdefault(k, []).append(np.asarray(value).copy())
            if fallen: break
        arrays = {k: np.stack(v) for k, v in rows.items()}
        np.savez_compressed(directory / f'{case["scenario"]}.npz', **arrays)
        valid = (arrays['time'] >= 2.) & ~arrays['done']
        stable = valid & (arrays['time'] >= 8.) & (arrays['time'] < 12.)
        result = dict(scenario=case['scenario'], steps=step+1, fall=bool(fallen),
                      estimation=metrics(arrays['prediction'][valid], arrays['truth'][valid]),
                      tracking_xy=metrics(arrays['truth'][valid, :2], arrays['command'][valid, :2]),
                      steady_vx=float(arrays['truth'][stable, 0].mean()) if stable.any() else None,
                      steady_prediction_vx=float(arrays['prediction'][stable, 0].mean()) if stable.any() else None,
                      contact_count_mean=arrays['contact_count'][valid].mean(0).tolist() if valid.any() else None,
                      loaded_contact_speed_mean=float(arrays['contact_point_speed'][valid].mean()) if valid.any() else None,
                      nonfoot_contact_samples=int((arrays['nonfoot_floor_contacts'] > 0).sum()))
        reports.append(result); pooled.append({k: v[valid] for k, v in arrays.items()})
    joined = {k: np.concatenate([x[k] for x in pooled]) for k in pooled[0]}
    summary = dict(cases=reports, complete_cases=sum(not r['fall'] and r['steps']==steps for r in reports),
                   estimation=metrics(joined['prediction'], joined['truth']),
                   tracking_xy=metrics(joined['truth'][:, :2], joined['command'][:, :2]))
    (directory / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', choices=('ours', 'v2', 'both'), default='both')
    p.add_argument('--variants', nargs='+', default=['baseline', 'geometry', 'friction', 'integrator', 'passive', 'geometry+friction+integrator'])
    p.add_argument('--cases', nargs='+', default=['straight', 'left', 'right', 'reverse', 'lateral_left', 'lateral_right'])
    p.add_argument('--steps', type=int, default=1200)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(20260928); np.random.seed(20260928)
    initial = {str(p): sha(p) for p in (OURS, OFFICIAL, LOW)}
    inv = {k: inventory(p, k) for k, p in (('ours', OURS), ('official_v2', OFFICIAL))}
    (a.output / 'inventory.json').write_text(json.dumps(inv, indent=2, default=serialize), encoding='utf8')
    from homie_pushcart_v52.lowlevel import HimLowLevelPolicy
    with contextlib.redirect_stdout(io.StringIO()): low = HimLowLevelPolicy(LOW, 'cpu')
    models = {k: session(WEIGHTS / f'GR00T-WholeBodyControl-{k.title()}.onnx')[0] for k in ('balance', 'walk')}
    results = {}
    for base in (('ours', 'v2') if a.base == 'both' else (a.base,)):
        for variant in a.variants:
            changes = set() if variant == 'baseline' else set(variant.split('+'))
            name = base + '_' + variant.replace('+', '_')
            xml = variant_xml(base, changes, a.output / f'{name}.xml')
            runner = OursRunner(xml, low, changes) if base == 'ours' else V2Runner(xml, models)
            run = evaluate(runner, a.output / name, [c for c in cases('free') if c['scenario'] in a.cases], a.steps)
            run.update(base=base, changes=sorted(changes), xml_sha256=sha(xml),
                       physics=inventory(xml, name)['options'])
            results[name] = run
            (a.output / 'results.json').write_text(json.dumps(results, indent=2, default=serialize), encoding='utf8')
            reverse = next((c for c in run['cases'] if c['scenario']=='reverse'), {})
            print(json.dumps(dict(variant=name, complete=run['complete_cases'], mae=run['estimation'].get('mae'),
                                  reverse_vx=reverse.get('steady_vx'), reverse_slip=reverse.get('loaded_contact_speed_mean'))), flush=True)
    assert initial == {p: sha(p) for p in initial}
    (a.output / 'validation.json').write_text(json.dumps(dict(result='PASS_EXECUTION', original_sources_unchanged=initial,
                                                              experiments=len(results), mujoco_version=mujoco.__version__,
                                                              source_sha256=sha(Path(__file__))), indent=2), encoding='utf8')


if __name__ == '__main__': main()
