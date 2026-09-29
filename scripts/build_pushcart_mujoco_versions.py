"""Materialize the original and four-sphere V5.3 scenes as two named XML files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'assets/g1_pushcart_v53/mujoco_scenes')
    a = p.parse_args()
    from homie_pushcart_v53.env_smoke import make_configs
    from pushcart_bench.sim2sim_scene import build_isaac_scene
    import mujoco
    cfg, task = make_configs(1, False, 13, checkpoint=a.checkpoint)
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint_config=str(a.checkpoint.resolve()), default_scene='original',
                  runtime='27 robot DoFs; existing HOMIE + V5.3; nominal grasp; URDF inertia; explicit PD', scenes={})
    for version, filename in [('original', 'original.xml'), ('four_spheres', 'new_four_spheres.xml')]:
        source = build_isaac_scene(cfg, 'explicit_pd', 'urdf', scene_version=version)
        target = a.output / filename
        contents = source.read_bytes()
        if target.exists() and target.read_bytes() != contents:
            raise FileExistsError(f'Refusing to replace an existing different scene: {target}; choose a new --output directory')
        target.write_bytes(contents)
        model = mujoco.MjModel.from_xml_path(str(target))
        result['scenes'][version] = dict(file=str(target.resolve()), sha256=hashlib.sha256(contents).hexdigest(),
                                         source_cache=str(source), nq=model.nq, nv=model.nv, actuators=model.nu,
                                         grasp_equalities=model.neq, physics_dt=float(model.opt.timestep),
                                         foot_floor_friction=.75)
    (a.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
