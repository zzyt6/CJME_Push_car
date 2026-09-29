"""Known applied hand loads in MuJoCo, using the Isaac label probe's protocol.

This is a label calibration, not a neural-estimator score or cart-force test.
The cart is disconnected and moved away. Whole-body HOMIE remains active.
Arm inverse dynamics reconstructs the external load without reading the
prescribed hand forces, equality forces, or qfrc_constraint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
from validate_pushcart_v51_labels import (PHASES, POINT_OFFSET_PALM_AXES_M,
                                         prescribed_load, summarize_trace)
from homie_pushcart_v52.lowlevel import HimLowLevelPolicy, frozen_checkpoint
from homie_pushcart_v52.paths import G1_GRASP_URDF, HOMIE_JOINTS
from homie_pushcart_v52.teleop import play_config
from pushcart_bench.policies_v5x import InstrumentedEnv, PALM_OFFSET


def inverse_dynamics_load(env, pre_velocity, palm_points, nonhand_loads):
    """Hand-on-environment wrench from seven arm-joint equations per hand."""
    m, d = env.model, env.data
    sample = copy.copy(d)
    sample.qvel[:] = pre_velocity
    required = np.empty(m.nv)
    mujoco.mj_rne(m, sample, 1, required)
    required += m.dof_armature * d.qacc
    residual = d.qfrc_actuator + d.qfrc_passive + d.qfrc_applied - required
    for body in np.flatnonzero(np.any(nonhand_loads, axis=1)):
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jp, jr, int(body))
        residual += jp.T @ nonhand_loads[body, :3] + jr.T @ nonhand_loads[body, 3:]
    result, conditions, equation_errors = [], [], []
    for i, side in enumerate(('left', 'right')):
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, palm_points[i], env.wrist_ids[side])
        ids = env.qvel_adr[13 + 7*i:20 + 7*i]
        jac = np.vstack((jp[:, ids], jr[:, ids])).T
        value, _, rank, singular = np.linalg.lstsq(jac, residual[ids], rcond=None)
        if rank != 6:
            raise RuntimeError('Arm Jacobian cannot identify all six wrench axes')
        result.append(value)
        conditions.append(singular[0] / singular[-1])
        equation_errors.append(np.linalg.norm(jac @ value - residual[ids]))
    return np.asarray(result), conditions, equation_errors


class KnownLoadEnv(InstrumentedEnv):
    recording = False

    def __init__(self, cfg, args):
        super().__init__(True, scene_profile='isaac', cfg=cfg, upper_drive='explicit_pd')
        self.args = args
        urdf = ET.parse(G1_GRASP_URDF).getroot()
        self.palm_com_local = np.array([
            np.fromstring(urdf.find(f"link[@name='{s}_hand_palm_link']/inertial/origin").get('xyz'), sep=' ')
            for s in ('left', 'right')])
        self.rows = {}
        self.audit_newton_euler = False

    def reset(self):
        super().reset()
        self.data.eq_active[:] = 0
        cart_joint = self.model.body(self.cart_id).jntadr[0]
        cart_qpos = self.model.jnt_qposadr[cart_joint]
        self.data.qpos[cart_qpos] += 50.
        mujoco.mj_forward(self.model, self.data)
        return self.compute_observations()

    def _clear(self):
        super()._clear()
        self.probe_substep = 0

    def _body_damping(self):
        super()._body_damping()
        self.pre_velocity = self.data.qvel.copy()
        self.nonhand_loads = self.data.xfrc_applied.copy()
        if not self.recording:
            return
        phase_time = self.control_step*self.dt + self.probe_substep*self.model.opt.timestep
        self.phase_time = phase_time
        f, couple = prescribed_load(PHASES[self.phase_id], phase_time, self.args.force_scale)
        palms, points = [], []
        for i, side in enumerate(('left', 'right')):
            body = self.wrist_ids[side]
            rotation = self.data.xmat[body].reshape(3, 3)
            palm = self.data.xpos[body] + rotation @ PALM_OFFSET[side]
            point = palm + rotation @ (self.palm_com_local[i] + POINT_OFFSET_PALM_AXES_M[i])
            self.data.xfrc_applied[body, :3] += f[i]
            self.data.xfrc_applied[body, 3:] += couple[i] + np.cross(point-self.data.xipos[body], f[i])
            palms.append(palm)
            points.append(point)
        self.palms = np.asarray(palms)
        self.expected = -np.concatenate((f, couple + np.cross(np.asarray(points)-self.palms, f)), -1)
        self.applied = f
        self.probe_substep += 1

    def _sample(self):
        if not self.recording:
            return
        measured, condition, equation_error = inverse_dynamics_load(
            self, self.pre_velocity, self.palms, self.nonhand_loads)
        row = dict(measured_wrench_w=measured, expected_wrench_w=self.expected,
                   applied_force_w=self.applied, upper_joint_pos_rad=self.dof_pos[12:],
                   valid=True, fault=False, jacobian_condition=condition,
                   arm_equation_residual_nm=equation_error)
        for key, value in row.items():
            self.rows.setdefault(key, []).append(np.array(value, copy=True))
        for key, value in dict(phase=self.phase_id, phase_time_s=self.phase_time+self.model.opt.timestep,
                               time_s=self.phase_id*self.args.steps*self.dt+self.phase_time+self.model.opt.timestep).items():
            self.rows.setdefault(key, []).append(value)


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(51)
    cfg = play_config(1, episode_length_s=120.)
    policy = HimLowLevelPolicy(frozen_checkpoint(), 'cpu')
    traces, failures = [], []
    for env_id in range(args.num_envs):
        env = KnownLoadEnv(cfg, args)
        env.reset()
        yaw = env_id*math.pi/2
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]),
                          env.data.qpos[env.root_qpos+3:env.root_qpos+7])
        env.data.qpos[env.root_qpos+3:env.root_qpos+7] = quat
        mujoco.mj_forward(env.model, env.data)
        obs = env.compute_observations()
        with torch.inference_mode():
            for _ in range(args.settle_steps):
                obs, fallen = env.step(policy(torch.tensor(obs[None], dtype=torch.float32))[0].numpy())
                if fallen:
                    failures.append(dict(env_id=env_id, phase='settle'))
            env.recording = True
            for phase_id, name in enumerate(PHASES):
                env.phase_id = phase_id
                for step in range(args.steps):
                    env.control_step = step
                    env.upper_delta[:] = 0.
                    if name.startswith('dynamic_'):
                        t = ((phase_id-3)*args.steps+step)*env.dt
                        ramp = min(1., max(0., t/.5))
                        for side, shift in (('left', 0.), ('right', math.pi/2)):
                            for joint, amplitude, hz in (('shoulder_pitch', .12, .8), ('shoulder_roll', .08, 1.), ('elbow', .12, 1.2)):
                                j = HOMIE_JOINTS.index(f'{side}_{joint}_joint')-12
                                env.upper_delta[j] = args.motion_scale*ramp*amplitude*math.sin(2*math.pi*hz*t+shift)/env.action_scale
                    obs, fallen = env.step(policy(torch.tensor(obs[None], dtype=torch.float32))[0].numpy())
                    if fallen:
                        failures.append(dict(env_id=env_id, phase=name, step=step))
                print(f'mujoco_known_load env={env_id} phase={name}', flush=True)
        traces.append({k: np.asarray(v) for k, v in env.rows.items()})
    trace = {k: (traces[0][k] if k in ('phase', 'phase_time_s', 'time_s')
                 else np.stack([r[k] for r in traces], axis=1)) for k in traces[0]}
    report = summarize_trace(trace, dt=cfg.sim.dt, resets=len(failures), expected_samples_per_phase=args.steps*cfg.control.decimation)
    report.update(engine='mujoco', upper_drive='explicit_pd', force_scale=args.force_scale,
                  initial_yaw_rad=[i*math.pi/2 for i in range(args.num_envs)], failures=failures,
                  source='arm inverse dynamics excluding prescribed hand force; no equality force or constraint generalized force',
                  external_force_convention='environment-on-hand in world axes; negative of measured hand-on-environment force',
                  label_reference='each palm origin at the MuJoCo solver sample (pre-integration)',
                  force_protocol='scripts/validate_pushcart_v51_labels.py:prescribed_load',
                  maximum_jacobian_condition=float(trace['jacobian_condition'].max()),
                  maximum_arm_equation_residual_nm=float(trace['arm_equation_residual_nm'].max()),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), ROOT/'scripts/validate_pushcart_v51_labels.py')})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'traces.npz', **trace)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(report['result'], 'max arm residual', report['maximum_arm_equation_residual_nm'], flush=True)
    return 0 if report['result'] == 'MEASURED' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--settle-steps', type=int, default=25)
    p.add_argument('--num-envs', type=int, default=2)
    p.add_argument('--force-scale', type=float, default=1.)
    p.add_argument('--motion-scale', type=float, default=1.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.steps < 100 or args.settle_steps < 25 or not 1 <= args.num_envs <= 8 or not 0 < args.force_scale <= 2 or not 0 < args.motion_scale <= 2:
        p.error('Use >=100 steps, >=25 settling steps, 1..8 environments and scales in (0,2].')
    raise SystemExit(main(args))
