"""Export composed training USD mass properties; no simulator or asset mutation.

The checked-in reference lets the MuJoCo-only entry run without importing USD.
Every source layer is hashed so stale reference values cannot be used silently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
from homie_pushcart_v52.paths import PACKED_USD, G1_GRASP_URDF, CART_URDF


def export(path):
    stage = Usd.Stage.Open(str(PACKED_USD))
    groups = {'robot': {}, 'cart': {}}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not (prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            continue
        group = 'robot' if str(prim.GetPath()).startswith('/Scene/Robot/') else 'cart'
        api = UsdPhysics.MassAPI(prim)
        q = api.GetPrincipalAxesAttr().Get()
        quat = [*q.GetImaginary(), q.GetReal()]
        axes = Rotation.from_quat(quat).as_matrix()
        diagonal = list(api.GetDiagonalInertiaAttr().Get())
        name = prim.GetName()
        if name in groups[group]:
            raise ValueError(f'Duplicate rigid-body name: {name}')
        groups[group][name] = dict(mass=float(api.GetMassAttr().Get()),
            com=list(api.GetCenterOfMassAttr().Get()),
            inertia=(axes @ np.diag(diagonal) @ axes.T).tolist(),
            principal_xyzw=quat, diagonal=diagonal, prim=str(prim.GetPath()))
    sources = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            source = Path(layer.realPath).resolve()
            sources[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    # These verified fixed-link transforms are used when folding USD bodies
    # into MuJoCo bodies. Their changes also require re-auditing the reference.
    for source in (G1_GRASP_URDF, CART_URDF):
        sources[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    report = dict(schema='v52_composed_usd_mass_properties_v1',
        frame='COM-centered inertia in rigid-body prim axes; quaternion xyzw',
        source=PACKED_USD.relative_to(ROOT).as_posix(),
        source_sha256=dict(sorted(sources.items())), bodies=groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(path, {k:len(v) for k,v in groups.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'pushcart_bench/data/v52_usd_mass_properties.json')
    export(p.parse_args().output)
