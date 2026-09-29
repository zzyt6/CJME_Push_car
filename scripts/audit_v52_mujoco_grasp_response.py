"""MuJoCo counterpart of the suspended Isaac cart-label fixture.

Same target and external-load schedule; compare responses, not paired states.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]
from homie_pushcart_v52.teleop import play_config
from pushcart_bench.policies_v5x import InstrumentedEnv


class Probe(InstrumentedEnv):
    def _clear(self):
        super()._clear()
        self.probe_substep = 0

    def _body_damping(self):
        super()._body_damping()
        t = self.control_step*self.dt+self.probe_substep*self.model.opt.timestep
        if .4 <= t < 1.2:
            self.data.xfrc_applied[self.cart_id] = [25*math.sin(2*math.pi*.8*(t-.4)), -12., 8., 0., 0., 1.2]
        self.probe_substep += 1

    def _sample(self):
        super()._sample()
        if self.control_step < 5:
            return
        hand, palm = self.sub_hands[-1], self.sub_palm[-1]
        com = self.data.subtree_com[self.cart_id]
        total = np.r_[hand[:, :3].sum(0), (hand[:, 3:]+np.cross(palm-com, hand[:, :3])).sum(0)]
        closure = []
        for side in ('left', 'right'):
            eid = self.model.eq(f'grasp_{side}').id
            rows = (self.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (self.data.efc_id == eid)
            residual = self.data.efc_pos[rows]
            closure.append(float(np.linalg.norm(residual[:3])) if len(residual) else 0.)
        for k, v in dict(measured_hand_wrench=hand, measured_resultant=total,
                         q=self.dof_pos, dq=self.dof_vel, target=self.joint_pos_target,
                         weld_translation_residual_m=closure).items():
            self.rows.setdefault(k, []).append(np.array(v, copy=True))


def main(args):
    cfg = play_config(1, 30.)
    cfg.asset.linear_damping = cfg.asset.angular_damping = 0.
    cfg.sim.gravity = [0., 0., 0.]
    traces = []
    for mode in ('left', 'right', 'both'):
        env = Probe(True, scene_profile='isaac', cfg=cfg, upper_drive='explicit_pd', inertia_profile=args.inertia_profile)
        if args.weld_impedance is not None:
            for side in ('left', 'right'):
                env.model.eq_solimp[env.model.eq(f'grasp_{side}').id, :2] = args.weld_impedance
        env.reset()
        for side in ('left', 'right'):
            env.data.eq_active[env.model.eq(f'grasp_{side}').id] = mode in (side, 'both')
        env.data.qpos[env.root_qpos+2] += 3.
        cart_joint = env.model.body(env.cart_id).jntadr[0]
        env.data.qpos[env.model.jnt_qposadr[cart_joint]+2] += 3.
        mujoco.mj_forward(env.model, env.data)
        env.rows = {}
        for step in range(args.steps):
            env.control_step = step
            env.upper_delta[:] = 0.
            env.upper_delta[2] = .02*math.sin(2*math.pi*.7*step*env.dt)/env.action_scale
            lower = (env.preset_dof_pos[:12]-env.default_dof_pos[:12])/env.action_scale
            env.step(lower)
        traces.append({k: np.asarray(v) for k, v in env.rows.items()})
    trace = {k: np.stack([v[k] for v in traces], axis=1) for k in traces[0]}
    isaac = np.load(args.isaac/'traces.npz')
    if len(isaac['measured_hand_wrench']) != len(trace['measured_hand_wrench']):
        raise ValueError('Different sample counts; cannot compare schedules')
    rows = {}
    for i, mode in enumerate(('left', 'right', 'both')):
        a, b = isaac['measured_hand_wrench'][:, i, :, :3], trace['measured_hand_wrench'][:, i, :, :3]
        stats = dict(hand_force_vector_difference_rms_n=np.sqrt(np.mean(np.sum((a-b)**2, -1), 0)).tolist(),
                     resultant_force_difference_rms_n=float(np.sqrt(np.mean(np.sum((a.sum(1)-b.sum(1))**2, -1)))),
                     isaac_hand_force_rms_n=np.sqrt(np.mean(np.sum(a*a, -1), 0)).tolist(),
                     mujoco_hand_force_rms_n=np.sqrt(np.mean(np.sum(b*b, -1), 0)).tolist(),
                     weld_translation_residual_rms_m=np.sqrt(np.mean(trace['weld_translation_residual_m'][:, i]**2, 0)).tolist())
        for k in ('q', 'dq', 'target'):
            if k in isaac:
                diff = isaac[k][:, i]-trace[k][:, i]
                stats[k+'_max_abs_difference'] = float(abs(diff).max())
                stats[k+'_rms_difference'] = float(np.sqrt(np.mean(diff**2)))
        rows[mode] = stats
    report = dict(scenes=rows, samples=len(trace['q']), isaac_reference=str(args.isaac),
                  diagnostic_weld_impedance=args.weld_impedance,
                  inertia_profile=args.inertia_profile,
                  note='Same input schedule in isolated single/dual grasps; differences are physical responses, not neural prediction errors. Isaac poses are post-integration, MuJoCo weld labels use its solver sample.',
                  production_scene_modified=False)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'traces.npz', **trace)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--isaac', type=Path, required=True)
    p.add_argument('--weld-impedance', type=float, help='Diagnostic-only constant weld impedance; never changes production assets or policies')
    p.add_argument('--inertia_profile', choices=('urdf', 'physx'), default='urdf')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.weld_impedance is not None and not 0 < args.weld_impedance < 1:
        p.error('Impedance must be in (0,1).')
    main(args)
