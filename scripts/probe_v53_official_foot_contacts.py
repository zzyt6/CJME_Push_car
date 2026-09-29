"""Isolated V5.3 cart rollout with only foot contact geometry changed.

Reuses the fixed-checkpoint benchmark. Creates a private scene copy and leaves
the production builder, defaults, checkpoints and grasp parameters untouched.
"""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL'), str(ROOT/'scripts')]
import audit_homie_mujoco_scene_versions as audit
import eval_v53_estimator_sim2sim_status as bench
import pushcart_bench.policies_v53 as policies

OriginalRunner = policies.V53Runner
SCENE = ROOT / '.cache/homie_scene_version_cart_probe'
SCENE.mkdir(exist_ok=True)
details = {}


class ProbeRunner(OriginalRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env = self.env
        source = env.xml_path
        original_source = audit.OURS
        audit.OURS = source
        try:
            name = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            new = audit.variant_xml('ours', {'geometry'}, SCENE/f'four_spheres_{name}.xml')
        finally:
            audit.OURS = original_source
        m = mujoco.MjModel.from_xml_path(str(new)); old = env.model
        assert (m.nq, m.nv, m.nu, m.nbody, m.neq) == (old.nq, old.nv, old.nu, old.nbody, old.neq)
        for field in ('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'jnt_range', 'jnt_actfrcrange',
                      'dof_damping', 'dof_armature', 'dof_frictionloss', 'actuator_gainprm', 'actuator_biasprm',
                      'eq_data', 'eq_solref', 'eq_solimp'):
            np.testing.assert_allclose(getattr(m, field), getattr(old, field), atol=1e-12, rtol=0.)
        assert m.opt.integrator == old.opt.integrator and m.opt.timestep == old.opt.timestep
        env.model, env.data, env.xml_path = m, mujoco.MjData(m), new
        details.update(source=str(source), variant=str(new), source_sha256=audit.sha(source),
                       variant_sha256=audit.sha(new), change='Foot convex mesh -> four 5 mm spheres per foot; effective foot-floor friction remains 0.75',
                       preserved='Mass, inertia, DoFs, limits, passive dynamics, actuator gains/bias, grasp equalities, timestep and integrator verified unchanged')


if __name__ == '__main__':
    policies.V53Runner = ProbeRunner
    bench.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    p = output/'report.json'; report = json.loads(p.read_text())
    report['foot_contact_probe'] = details
    report['probe_source_sha256'] = audit.sha(Path(__file__))
    p.write_text(json.dumps(report, indent=2), encoding='utf8')
