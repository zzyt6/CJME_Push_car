"""Headless diagnostic of official HOMIE v2 / Decoupled WBC Walk+Balance.

Uses the downloaded official 29-DoF XML, LFS-verified meshes, published standalone
simulator gains, and its observation-building methods. This is a separate
embodiment/plant check, not a replacement inside the current 27-DoF cart task.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import sys
import types

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_homie_body_velocity import command, cases
from eval_homie_body_velocity import metrics

SOURCE = ROOT / '.cache/homie_official_weights'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def session(path):
    m = onnx.load(str(path))
    m.graph.output.append(onnx.helper.make_tensor_value_info('/estimator/4/Gemm_output_0', onnx.TensorProto.FLOAT, ['batch_size', 35]))
    onnx.checker.check_model(m)
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 1; opt.inter_op_num_threads = 1
    instrumented = ort.InferenceSession(m.SerializeToString(), sess_options=opt, providers=['CPUExecutionProvider'])
    original = ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])
    return instrumented, original


def official_observation_builder(data, model):
    # Reuse only the reviewed pure observation/math methods; no viewer,
    # keyboard hooks, networking, or full deployment stack is started.
    tree = ast.parse((SOURCE / 'v2_run_mujoco.py').read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GearWbcController')
    wanted = {'compute_observation', 'quat_rotate_inverse', 'get_gravity_orientation', 'pd_control'}
    defs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in defs} == wanted
    space = {'np': np}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(SOURCE / 'v2_run_mujoco.py'), 'exec'), space)
    obj = types.SimpleNamespace(data=data, torso_index=model.body('torso_link').id)
    for name in wanted:
        setattr(obj, name, types.MethodType(space[name], obj))
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((SOURCE / 'v2_sim.yaml').read_text())
    for key in ('kps', 'kds', 'default_angles', 'cmd_scale'):
        cfg[key] = np.asarray(cfg[key], dtype=np.float32)
    mesh_manifest = json.loads((SOURCE / 'v2_mesh_manifest.json').read_text())
    assert len(mesh_manifest) == 49 and all(x['git_lfs_pointer_blob_matches'] for x in mesh_manifest)
    xml = SOURCE / 'v2_native/g1_gear_wbc.xml'
    m = mujoco.MjModel.from_xml_path(str(xml)); m.opt.timestep = cfg['simulation_dt']
    assert m.nu == 29 and m.nq == 36 and m.nv == 35
    d = mujoco.MjData(m); builder = official_observation_builder(d, m)
    files = {k: SOURCE / f'GR00T-WholeBodyControl-{k.title()}.onnx' for k in ('balance', 'walk')}
    models = {k: session(v) for k, v in files.items()}
    dt = cfg['simulation_dt'] * cfg['control_decimation']
    all_data, reports, checks = [], [], []
    for case in cases('free'):
        mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        previous_action = np.zeros(15, dtype=np.float32)
        target = cfg['default_angles'].copy()
        history = collections.deque([np.zeros(86, dtype=np.float32) for _ in range(6)], maxlen=6)
        filtered = np.zeros(3)
        rows = {}; failed = False
        for step in range(a.steps):
            filtered += min(1., dt / .4) * (command(step * dt, case['scenario']) - filtered)
            for _ in range(cfg['control_decimation']):
                d.ctrl[:15] = builder.pd_control(target, d.qpos[7:22], cfg['kps'], np.zeros(15), d.qvel[6:21], cfg['kds'])
                d.ctrl[15:] = builder.pd_control(np.zeros(14), d.qpos[22:36], np.full(14, 100.), np.zeros(14), d.qvel[21:35], np.full(14, .5))
                mujoco.mj_step(m, d)
            # Refresh current rigid transforms for the scoring reference.
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_comVel(m, d)
            ctrl = dict(loco_cmd=filtered, height_cmd=cfg['height_cmd'], rpy_cmd=cfg['rpy_cmd'])
            obs, width = builder.compute_observation(d, cfg, previous_action, ctrl, 29)
            assert width == 86
            history.append(obs)
            x = np.concatenate(history)[None]
            kind = 'balance' if np.linalg.norm(filtered) <= .05 else 'walk'
            policy, original = models[kind]
            action, encoder = policy.run(None, {'input': x})
            if step % 100 == 0:
                check = float(np.abs(action - original.run(None, {'input': x})[0]).max())
                checks.append(check); assert check < 2e-6
            previous_action = action[0]
            target = previous_action * cfg['action_scale'] + cfg['default_angles']
            v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body('pelvis').id, v, 0)
            q = d.qpos[3:7]
            truth = builder.quat_rotate_inverse(q, v[3:])
            origin = builder.quat_rotate_inverse(q, d.qvel[:3])
            gravity = builder.get_gravity_orientation(q)
            failed = bool(np.linalg.norm(gravity[:2]) > .8 or d.qpos[2] < .35)
            values = dict(time=(step + 1) * dt, prediction=encoder[0, :3] / 2., truth_com=truth, truth_link=origin,
                          actual_twist=np.r_[origin[:2], d.qvel[5]], command=filtered,
                          low_observation=x[0], action=previous_action, mode=kind, done=failed)
            for key, value in values.items(): rows.setdefault(key, []).append(np.asarray(value).copy())
            if failed: break
        arrays = {k: np.stack(v) for k, v in rows.items()}
        np.savez_compressed(a.output / f'{case["scenario"]}.npz', **arrays)
        valid = (arrays['time'] >= 2.) & ~arrays['done']
        result = dict(**case, steps=step + 1, fall=failed,
                      estimation=metrics(arrays['prediction'][valid], arrays['truth_com'][valid]),
                      tracking=metrics(arrays['actual_twist'][valid], arrays['command'][valid]))
        reports.append(result); all_data.append({k: v[valid] for k, v in arrays.items()})
        print(json.dumps(result), flush=True)
    joined = {k: np.concatenate([d[k] for d in all_data]) for k in all_data[0]}
    report = dict(result='PASS_EXECUTION', policy='Official HOMIE v2 / Decoupled WBC Balance+Walk',
                  engine='MuJoCo', embodiment='official 29-DoF G1 XML', input_dim=516, action_dim=15,
                  cases=reports, complete_cases=sum(c['steps'] == a.steps and not c['fall'] for c in reports),
                  estimation=metrics(joined['prediction'], joined['truth_com']),
                  tracking=metrics(joined['actual_twist'], joined['command']),
                  action_output_unchanged_max_abs=max(checks), weights_sha256={k: sha(v) for k, v in files.items()},
                  source_commit='b042411fae38ee4d1af9aac82a37a1f8d14d6dd0', source_script_sha256=sha(SOURCE/'v2_run_mujoco.py'),
                  xml_sha256=sha(xml), mesh_count=49, control_dt=dt,
                  source_sha256=sha(Path(__file__)),
                  note='Standalone native model diagnostic using published PD/observation code, smoothed benchmark commands. Estimator first three channels /2 follows released HIM architecture and base velocity normalization; not an official published accuracy result. Different plant and DoFs from current cart model, so not a weights-only comparison.')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: report[k] for k in ('result', 'complete_cases', 'estimation', 'tracking')}), flush=True)


if __name__ == '__main__':
    main()
