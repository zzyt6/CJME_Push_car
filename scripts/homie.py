"""Project entry point. Start Kit before importing any simulation modules."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))
# HOMIE's HIM algorithm is a fork. Scope it to this process, never pip-install
# over the RSL-RL distribution used by other Isaac Lab projects.
sys.path.insert(0, str(ROOT / 'HomieRL' / 'rsl_rl'))


def check_runtime():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 from RLisaac601: .\\run_homie.ps1 doctor')
    for name, expected in [('isaacsim', '6.0.1.0'), ('isaaclab', '6.1.17'), ('isaaclab_physx', '1.1.3')]:
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'This local adapter requires {name}=={expected}, found {actual}. Use RLisaac601.')


def doctor():
    import torch
    import rsl_rl

    report = {'python': sys.version, 'executable': sys.executable, 'packages': {}}
    for name in ('isaacsim', 'isaaclab', 'isaaclab_physx', 'isaaclab_assets',
                 'isaaclab_tasks', 'isaaclab_rl', 'torch', 'numpy',
                 'rsl-rl-lib', 'tensorboard', 'gitpython'):
        report['packages'][name] = metadata.version(name)
    report['cuda'] = torch.version.cuda
    report['cuda_available'] = torch.cuda.is_available()
    report['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    report['homie_rsl_rl'] = rsl_rl.__file__
    report['isaaclab_checkout'] = 'E:/IsaacLab_601'
    report['isaaclab_repository_version'] = Path('E:/IsaacLab_601/VERSION').read_text().strip()
    report['isaaclab_git_commit'] = subprocess.check_output(
        ['git', '-C', report['isaaclab_checkout'], 'rev-parse', 'HEAD'], text=True).strip()
    report['isaacsim_source_build'] = Path('E:/isaacsim/_build/windows-x86_64/release/VERSION').read_text().strip()
    report['urdf'] = str(ROOT / 'HomieRL/legged_gym/resources/robots/g1_description/g1.urdf')
    from homie_isaaclab.config import G1RoughCfg, G1RoughCfgPPO
    from rsl_rl.runners import HIMOnPolicyRunner
    assert G1RoughCfg().env.num_observations == 456
    assert G1RoughCfgPPO().algorithm.use_flip
    assert sys.version_info[:2] == (3, 12), 'This local profile requires Python 3.12.'
    assert report['packages']['isaacsim'] == '6.0.1.0', 'This profile targets Isaac Sim 6.0.1.'
    assert report['cuda_available'], 'CUDA is unavailable.'
    print(json.dumps(report, indent=2, ensure_ascii=False))
    report_path = ROOT / 'logs/environment.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('DOCTOR PASS')


def main():
    check_runtime()
    if len(sys.argv) == 1 or sys.argv[1] == 'doctor':
        doctor()
        return
    if sys.argv[1] == 'train':
        # Training now runs NVIDIA's official entrypoint and modern runner in
        # a clean child interpreter. Its rsl_rl must not be this legacy fork.
        from homie_isaaclab.training import train
        raise SystemExit(train(sys.argv[2:]))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description='HOMIE G1 on local Isaac Lab 3 / Isaac Sim 6.0.1')
    parser.add_argument('mode', choices=['smoke', 'train', 'play'])
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--run_name', default='g1_isaaclab')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_randomization', action='store_true')
    parser.add_argument(
        '--play_upper_randomization',
        action='store_true',
        help='In play mode, enable only the trained upper-body pose curriculum at ratio 1.0',
    )
    parser.add_argument('--record_ovd', type=Path, help='Record an OmniPVD physics trace to this directory')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.max_iterations < 1:
        parser.error('num_envs, steps, and max_iterations must be positive')
    if args.mode == 'play' and args.checkpoint is None:
        parser.error('play requires --checkpoint /path/to/model.pt')
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.record_ovd is not None:
        trace_dir = args.record_ovd.resolve()
        if any(c.isspace() for c in str(trace_dir)):
            parser.error('This AppLauncher splits kit_args on spaces; use an OVD directory without spaces')
        trace_dir.mkdir(parents=True, exist_ok=True)
        # Official ordering: recording directory must precede enabling output.
        args.kit_args = (args.kit_args or '') + f' --/persistent/physics/omniPvdOvdRecordingDirectory={trace_dir.as_posix()} --/physics/omniPvdOutputEnabled=true'
    launcher = AppLauncher(args)
    try:
        from homie_isaaclab.run import run
        run(args, launcher.app, ROOT)
    except BaseException:
        # Kit's close() can terminate the interpreter; print failures beforehand.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(exit_code=1)
        raise
    else:
        launcher.app.close()


if __name__ == '__main__':
    main()
