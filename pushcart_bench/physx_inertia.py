"""Measured V5.2 training-domain inertia compatibility, separate from URDF.

This preserves the old Isaac reset behavior for existing checkpoints. It does
not repair the Isaac setter order or modify the training domain.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).parent/'data/v52_usd_mass_properties.json'


@lru_cache(maxsize=1)
def usd_mass_properties():
    data = json.loads(REFERENCE.read_text(encoding='utf-8'))
    if data['schema'] != 'v52_composed_usd_mass_properties_v1':
        raise ValueError('Unsupported USD mass-property reference')
    for name, digest in data['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Stale USD reference: {name}; rerun export_v52_usd_mass_properties.py')
    return data


def cart_reset_inertia(intended, body_name='base_link'):
    """Actual V5.2 set_inertias -> restore old COM rotation, for XY payloads.

    The current box/XY-offset model has only an XY off-diagonal term. PhysX
    diagonalizes it with a Jacobi rotation of at most pi/4 (axes are NOT sorted
    by eigenvalue), then set_coms replaces those axes with the original USD
    principal axes. Reconstruct that physical tensor, including off-center
    payloads. Ref: PhysX PxDiagonalize, foundation/FdMathUtils.cpp. Runtime
    checks must accompany any future change to the training reset or asset.
    """
    tensor = np.asarray(intended, dtype=float)
    if tensor.shape != (3, 3) or not np.isfinite(tensor).all():
        raise ValueError('Expected finite 3x3 inertia')
    if np.max(abs(tensor[[0, 1], 2])) > 1e-6 or not np.allclose(tensor, tensor.T, atol=1e-8):
        raise ValueError('V5.2 compatibility supports the verified XY payload model only')
    a, b, c = tensor[0, 0], tensor[1, 1], tensor[0, 1]
    diagonal = np.diag(tensor).copy()
    if c != 0 and abs(a-b) <= 2e6*abs(2*c):
        w = (a-b)/(2*c)
        t = np.copysign(1., w)/(abs(w)+np.sqrt(w*w+1))
        diagonal[:2] = a+c*t, b-c*t
    q = usd_mass_properties()['bodies']['cart'][body_name]['principal_xyzw']
    axes = Rotation.from_quat(q).as_matrix()
    return axes @ np.diag(diagonal) @ axes.T
