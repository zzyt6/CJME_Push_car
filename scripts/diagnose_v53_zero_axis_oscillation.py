"""Instrument zero-vy/wz cart commands and isolate high-level action channels.

Diagnostic action masks are evaluated only in private runners. No training,
checkpoint or production configuration is modified. These masks are not
retrained baseline policies and do not establish an optimal controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL'), str(ROOT/'scripts')]
from eval_v53_estimator_sim2sim_status import protocol, metrics
from pushcart_bench.policies_v53 import V53Runner

CHECKPOINT = ROOT/'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-37-10_v53_final_20k/model_15000.pt'


def state_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def foot_loads(m, d):
    loads = np.zeros(2); floor = m.geom('floor').id
    feet = {m.body(f'{s}_ankle_roll_link').id: i for i, s in enumerate(('left', 'right'))}
    for k in range(d.ncon):
        c = d.contact[k]
        if floor not in (c.geom1, c.geom2): continue
        other = c.geom2 if c.geom1 == floor else c.geom1
        b = int(m.geom_bodyid[other])
        if b in feet:
            f = np.zeros(6); mujoco.mj_contactForce(m, d, k, f)
            loads[feet[b]] += abs(f[0])
    return loads


def peak(x, dt=.02):
    f, p = periodogram(x, fs=1/dt, window='hann', detrend='constant')
    keep = (f >= .5) & (f <= 12.)
    return float(f[keep][np.argmax(p[keep])]) if np.std(x)>1e-10 else None


def summarize(x):
    valid = (x['time'] >= 8.) & (x['time'] < 12.)
    report = {}
    for name, data in [('velocity', x['velocity']), ('prediction', x['prediction']),
                       ('homie_command', x['homie_command']), ('residual_scaled', x['residual_scaled']),
                       ('robot_velocity_cart', x['robot_velocity_cart']), ('cart_com_velocity', x['cart_com_velocity'])]:
        v = data[valid]
        report[name] = dict(mean=v.mean(0).tolist(), std=v.std(0).tolist(), minimum=v.min(0).tolist(), maximum=v.max(0).tolist())
    report['estimation'] = metrics(x['prediction'][valid], x['velocity'][valid])
    report['reference_point_difference'] = metrics(x['velocity'][valid, :2], x['cart_com_velocity'][valid, :2])
    signals = dict(cart_vy=x['velocity'][:, 1], cart_wz=x['velocity'][:, 2],
                   robot_vy=x['robot_velocity_cart'][:, 1], robot_wz=x['robot_omega_world'][:, 2],
                   alternating_foot_load=x['foot_load'][:, 0]-x['foot_load'][:, 1],
                   hand_force_y=x['cart_hand_force'][:, 1], hand_moment_z=x['cart_hand_moment'][:, 2],
                   residual_y=x['residual_scaled'][:, 1], residual_wz=x['residual_scaled'][:, 2])
    report['peak_hz'] = {k: peak(v[valid]) for k, v in signals.items()}
    # Same-window simultaneous correlations are descriptive, not causal proof.
    for a, b in [('cart_vy', 'robot_vy'), ('cart_wz', 'robot_wz')]:
        report[a+'_'+b+'_correlation'] = float(np.corrcoef(signals[a][valid], signals[b][valid])[0, 1])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(20260928); np.random.seed(20260928)
    results = {}
    for scenario in ('straight', 'reverse'):
        initial = None
        for mode in ('full', 'no_yz_residual', 'fixed_upper', 'passthrough'):
            r = V53Runner(CHECKPOINT, level=13, scene_version='four_spheres')
            r.set_payload(15.)
            # All counterfactuals start with the same complete MuJoCo state.
            mujoco.mj_resetData(r.env.model, r.env.data)
            r.reset(caster_angle=0.)
            snapshot = np.r_[r.env.data.qpos, r.env.data.qvel]
            if initial is None: initial=snapshot.copy()
            else: np.testing.assert_array_equal(initial, snapshot)
            actor_hash = state_hash(r.actor)
            original_forward = r.actor.forward
            candidate = None
            def masked_forward(obs, *args, **kwargs):
                nonlocal candidate
                out = original_forward(obs, *args, **kwargs)
                candidate = out[0].detach().cpu().numpy().copy()
                out = out.clone()
                if mode == 'no_yz_residual': out[:, 1:3] = 0.
                elif mode == 'fixed_upper': out[:, 3:] = 0.
                elif mode == 'passthrough': out[:] = 0.
                return out
            r.actor.forward = masked_forward
            rows = {}; fallen = False
            for step in range(1200):
                stage, cmd = protocol(step*r.dt, scenario)
                _, fallen = r.step(cmd)
                e, m, d = r.env, r.env.model, r.env.data
                c, s = np.cos(r.ts['cart_yaw']), np.sin(r.ts['cart_yaw'])
                rotation = np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
                robot = np.zeros(6); cart = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, e.pelvis_id, robot, 0)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, e.cart_id, cart, 0)
                values = dict(time=(step+1)*r.dt, stage=stage, velocity=np.r_[r.ts['cart_v'],r.ts['cart_wz']],
                              prediction=r.pred_cart_velocity, reference=r.cmd_ref, homie_command=r.cmd_homie,
                              high_action=r.hl_actions, unmodified_action=candidate,
                              residual_scaled=r.hl_actions[:3]*r.residual_scale, upper_delta=r.upper_delta_q,
                              robot_velocity_cart=rotation@robot[3:], robot_omega_world=robot[:3],
                              cart_com_velocity=rotation@cart[3:], cart_omega_world=cart[:3],
                              gravity=e.projected_gravity, q=e.dof_pos, dq=e.dof_vel,
                              foot_load=foot_loads(m,d), cart_hand_force=rotation@r.ts['cart_force_w'],
                              cart_hand_moment=rotation@r.ts['cart_moment_w'],
                              hand_wrench=np.mean(e.sub_hands,axis=0),
                              cart_position=d.xpos[e.cart_id], cart_yaw=r.ts['cart_yaw'], rel_yaw=r.ts['rel_yaw'])
                for k,v in values.items(): rows.setdefault(k,[]).append(np.asarray(v).copy())
                if fallen: break
            arrays = {k:np.stack(v) for k,v in rows.items()}
            name=scenario+'_'+mode
            np.savez_compressed(a.output/f'{name}.npz',**arrays)
            assert actor_hash==state_hash(r.actor)
            if len(arrays['time'])<600: raise RuntimeError(f'Trial {name} ended before the steady window')
            report=summarize(arrays)
            report.update(fall=bool(fallen),steps=step+1,checkpoint=str(CHECKPOINT),
                          checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),mode=mode,
                          scene_version=r.scene_version,initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                          rewards=dict(tracking_sigma=r.task_cfg.rewards.tracking_sigma,
                                       measure_filter_tau_s=r.task_cfg.rewards.measure_filter_tau_s,
                                       track_cart_vy=r.task_cfg.rewards.scales.track_cart_vy,
                                       track_cart_wz=r.task_cfg.rewards.scales.track_cart_wz,
                                       cart_vy_ripple=r.task_cfg.rewards.scales.cart_vy_ripple,
                                       cart_yaw_ripple=r.task_cfg.rewards.scales.cart_yaw_ripple))
            results[name]=report
            (a.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf8')
            print(json.dumps(dict(trial=name,mean=report['velocity']['mean'],std=report['velocity']['std'],
                                  homie=report['homie_command'],peaks=report['peak_hz'])),flush=True)
    (a.output/'validation.json').write_text(json.dumps(dict(result='PASS_EXECUTION',trials=len(results),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Fixed-weight diagnostic action masks, same reset state and new scene. Not retrained baselines; changes in forward speed must be considered.'),indent=2),encoding='utf8')


if __name__=='__main__':main()
