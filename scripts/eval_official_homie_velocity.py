"""Test official HOMIE v1 ONNX on the isolated no-cart body-velocity benchmark.

Exposes the existing 35-D encoder output for diagnostics without changing any
weights or action computation. Does not replace the project's frozen policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL'), str(ROOT / 'scripts')]
MODEL = ROOT / '.cache/homie_official_weights/deploy.onnx'


class OfficialOnnxPolicy:
    def __init__(self, checkpoint=None, device='cpu', obs_profile='official_deploy'):
        import onnx
        import onnxruntime as ort
        self.checkpoint, self.device, self.obs_profile = MODEL, device, obs_profile
        self.num_obs, self.num_actions = 456, 12
        graph = onnx.load(str(MODEL))
        assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 456
        self.encoder_output = '/estimator/4/Gemm_output_0'
        assert any(self.encoder_output in n.output for n in graph.graph.node)
        graph.graph.output.append(onnx.helper.make_tensor_value_info(self.encoder_output, onnx.TensorProto.FLOAT, ['batch_size', 35]))
        onnx.checker.check_model(graph)
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 1; opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(graph.SerializeToString(), sess_options=opt, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.original = ort.InferenceSession(str(MODEL), sess_options=opt, providers=['CPUExecutionProvider'])

    def prepare(self, obs):
        import numpy as np
        x = obs.detach().cpu().numpy().astype(np.float32, copy=True)
        if self.obs_profile == 'official_deploy':
            # Project frame uses command yaw scale 0.5; official LCMAgent uses 0.25.
            x.reshape(-1, 6, 76)[:, :, 2] *= .5
        return x

    def outputs(self, obs):
        import torch
        action, state = self.session.run(None, {self.input_name: self.prepare(obs)})
        return torch.as_tensor(action, device=obs.device), torch.as_tensor(state[:, :3], device=obs.device)

    def act_and_estimate(self, obs):
        return self.outputs(obs)

    def estimate_velocity(self, obs):
        return self.outputs(obs)[1]

    def __call__(self, obs):
        return self.outputs(obs)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('isaac', 'mujoco'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gain_profile', choices=('local', 'official_cpp'), default='official_cpp')
    p.add_argument('--obs_profile', choices=('official_deploy', 'rl_config'), default='official_deploy')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); a.mode = 'free'
    if a.engine == 'isaac' and not os.environ.get('OFFICIAL_HOMIE_CHILD'):
        token = uuid.uuid4().hex
        child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                               env=dict(os.environ, OFFICIAL_HOMIE_CHILD=token, HOMIE_BODY_AUDIT_CHILD=token))
        path = a.output / 'report.json'
        r = json.loads(path.read_text()) if path.exists() else {}
        ok = child.returncode == 0 and r.get('run_id') == token and r.get('result') == 'PASS_EXECUTION'
        if r.get('run_id') == token:
            r.update(parent_verified=ok, child_exit_code=child.returncode)
            path.write_text(json.dumps(r, indent=2), encoding='utf8')
        print(f'[official HOMIE parent] exit={child.returncode} verified={ok}', flush=True)
        raise SystemExit(0 if ok else 1)
    a.output.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if a.engine == 'isaac':
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device=a.device)
    import eval_homie_body_velocity as bench
    import homie_pushcart.lowlevel as low
    import numpy as np
    import torch
    low.HimLowLevelPolicy = lambda checkpoint, device='cpu': OfficialOnnxPolicy(checkpoint, device, a.obs_profile)
    bench.LOW = MODEL
    original_config = bench.config_free
    def config(count):
        cfg = original_config(count)
        if a.gain_profile == 'official_cpp':
            cfg.control.stiffness = {'hip_yaw': 150., 'hip_roll': 150., 'hip_pitch': 150., 'knee': 300.,
                                     'ankle': 40., 'waist': 300., 'shoulder': 150., 'elbow': 100.,
                                     'wrist_roll': 10., 'wrist_pitch': 10., 'wrist_yaw': 5., 'hand': 10.}
            cfg.control.damping = {'hip_yaw': 2., 'hip_roll': 2., 'hip_pitch': 2., 'knee': 4.,
                                   'ankle': 2., 'waist': 5., 'shoulder': 4., 'elbow': 1.,
                                   'wrist_roll': .5, 'wrist_pitch': .5, 'wrist_yaw': .5, 'hand': 2.}
        return cfg
    bench.config_free = config
    try:
        probe = OfficialOnnxPolicy(obs_profile=a.obs_profile)
        # Check the diagnostic graph without depending on an archived benchmark.
        # Real rollout observations are still used for the velocity assessment below.
        observations = np.random.default_rng(20260928).normal(0., .2, (64, 456)).astype(np.float32)
        x = probe.prepare(torch.tensor(observations))
        original = probe.original.run(None, {probe.input_name: x})[0]
        diagnostic = probe.session.run(None, {probe.input_name: x})[0]
        action_diff = float(np.abs(original - diagnostic).max())
        assert action_diff < 2e-6
        if a.engine == 'isaac': bench.run_isaac(a, launcher.app)
        else: bench.run_mujoco(a)
        report_path = a.output/'report.json'
        r = json.loads(report_path.read_text())
        r.update(policy='OpenHomie official v1 deploy.onnx', gain_profile=a.gain_profile, obs_profile=a.obs_profile,
                 onnx_action_diagnostic_max_abs=action_diff, onnx_check_samples=len(x),
                 official_commit='33071f79f1884bb122bd45631c8c60d61dac1eb8', wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 note='Diagnostic encoder output is the existing first three channels before latent normalization; no weights changed. Official C++ gains differ from local training gains.')
        report_path.write_text(json.dumps(r, indent=2), encoding='utf8')
        assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == before
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        if a.engine == 'isaac': launcher.app.close(exit_code=1)
        raise
    else:
        if a.engine == 'isaac': launcher.app.close()


if __name__ == '__main__':
    main()
