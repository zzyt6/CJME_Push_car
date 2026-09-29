"""Independent cart-side balance against actual Isaac palm labels.

Diagnostic scenes contain left-only, right-only, and dual grasps. They are
translated clear of the floor with gravity and body damping disabled, so the
cart's external wrench is its known applied load plus the grasp reactions.
This isolates closed-chain label reconstruction, not ground-contact fidelity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]


def run(args):
    import math
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics, PhysxSchema
    import homie_pushcart_v52.sim_env as module
    from homie_pushcart_v52.teleop import play_config
    from homie_pushcart_v52.task_config import PushCartTaskCfg

    torch.set_num_threads(4)
    torch.manual_seed(52)
    modes = ('left', 'right', 'both')
    original_scene = module.InteractiveScene
    joint_report = []

    def make_scene(cfg):
        cfg.replicate_physics = False
        scene = original_scene(cfg)
        from isaaclab.sim import SimulationContext
        stage = SimulationContext.instance().stage
        # Replication is disabled to permit different grasp topologies. The
        # production environment authors body overrides on env_0 and relies
        # on replication; here every diagnostic copy needs its own overrides.
        for i in range(len(modes)):
            source = stage.GetPrimAtPath(f'/World/envs/env_{i}/Scene')
            paths = [str(p.GetPath()) for p in Usd.PrimRange(source, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                while prim.IsInstanceProxy():
                    parent = prim.GetParent()
                    while parent.IsInstanceProxy():
                        parent = parent.GetParent()
                    parent.SetInstanceable(False)
                    prim = stage.GetPrimAtPath(path)
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateDisableGravityAttr().Set(True)
                api.CreateLinearDampingAttr().Set(0.)
                api.CreateAngularDampingAttr().Set(0.)
        for i, mode in enumerate(modes):
            for side in ('left', 'right'):
                path = f'/World/envs/env_{i}/Scene/GraspJoints/{side}'
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f'Missing diagnostic grasp: {path}')
                while prim.IsInstanceProxy():
                    parent = prim.GetParent()
                    while parent.IsInstanceProxy():
                        parent = parent.GetParent()
                    parent.SetInstanceable(False)
                    prim = stage.GetPrimAtPath(path)
                joint = UsdPhysics.Joint(prim)
                enabled = mode in ('both', side)
                joint.CreateJointEnabledAttr().Set(enabled)
                joint_report.append(dict(path=path, enabled=joint.GetJointEnabledAttr().Get()))
        return scene

    class Probe(module.PushCartSimEnv):
        recording = False

        def __init__(self, *a, **kw):
            self.task_cfg = PushCartTaskCfg()
            self.task_cfg.actions.upper_drive = 'explicit_pd'
            self.rows = {}
            super().__init__(*a, **kw)

        def _apply_action(self):
            super()._apply_action()
            t = self.control_step*self.dt + (self._physics_substep-1)*self.cfg.sim.dt
            phase = int(t >= .4) + int(t >= 1.2)
            force = ([0., 0., 0.] if phase != 1 else
                     [25.*math.sin(2*math.pi*.8*(t-.4)), -12., 8.])
            couple = [0., 0., 1.2] if phase == 1 else [0., 0., 0.]
            self.known_f = torch.tensor(force, device=self.device).expand(3, -1).clone()
            self.known_m = torch.tensor(couple, device=self.device).expand(3, -1).clone()
            self.applied_reference = self.cart.data.body_com_pose_w.torch[:, self.cart_base_id, :3].clone()
            self.cart.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self.known_f[:, None], torques=self.known_m[:, None],
                body_ids=torch.tensor([self.cart_base_id], device=self.device, dtype=torch.int32), is_global=True)

        def _sample_dynamics(self, substep):
            super()._sample_dynamics(substep)
            if not self.recording:
                return
            # Independent articulation: no palm incoming force or hand inertia
            # enters this cart-side calculation.
            mass, pos, vel, inertia = self._body_dynamics(self.cart)
            acc = self.cart.data.body_com_acc_w.torch
            com = (mass[..., None]*pos).sum(1)/mass.sum(1)[:, None]
            required_f = (mass[..., None]*acc[..., :3]).sum(1)
            spin = (inertia @ vel[..., 3:, None]).squeeze(-1)
            required_m = ((inertia @ acc[..., 3:, None]).squeeze(-1) +
                          torch.cross(vel[..., 3:], spin, dim=-1) +
                          torch.cross(pos-com[:, None], mass[..., None]*acc[..., :3], dim=-1)).sum(1)
            palm = self.robot.data.body_link_pose_w.torch[:, self.palm_body_ids, :3]
            hand = self._hand_wrench
            resultant = torch.cat((hand[..., :3].sum(1),
                                  (hand[..., 3:]+torch.cross(palm-com[:, None], hand[..., :3], dim=-1)).sum(1)), -1)
            cart_load = torch.cat((self.known_f, self.known_m + torch.cross(self.applied_reference-com, self.known_f, dim=-1)), -1)
            expected = torch.cat((required_f, required_m), -1)-cart_load
            values = dict(measured_hand_wrench=hand, measured_resultant=resultant,
                          cart_balance_reference=expected, applied_wrench=cart_load,
                          cart_com=com, cart_body_acc=acc, cart_body_mass=mass,
                          cart_body_pos=pos, valid=self.dynamics_wrench_valid,
                          q=self.robot.data.joint_pos.torch[:, self.joint_ids],
                          dq=self.robot.data.joint_vel.torch[:, self.joint_ids],
                          target=self.joint_pos_target,
                          foot_contact=self.contact_forces[:, self.feet_indices])
            for k, v in values.items():
                self.rows.setdefault(k, []).append(v.detach().cpu().numpy().copy())

    cfg = play_config(3, 30.)
    cfg.asset.disable_gravity = True
    cfg.sim.gravity = [0., 0., 0.]
    cfg.asset.linear_damping = cfg.asset.angular_damping = 0.
    cfg.pushcart_upper_drive = 'explicit_pd'
    module.InteractiveScene = make_scene
    try:
        env = Probe(cfg, ROOT, args.device, True, with_cart=True, full_dynamics=False)
    finally:
        module.InteractiveScene = original_scene
    env.control_step = 0
    env.cart_base_id = env.cart.body_names.index('base_link')
    env.reset()
    for asset in (env.robot, env.cart):
        pose = asset.data.root_link_pose_w.torch.clone()
        pose[:, 2] += 3.
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(3, 6, device=args.device))
    resets = []
    with torch.inference_mode():
        for step in range(args.steps):
            env.control_step = step
            env.recording = step >= 5
            # Small prescribed arm motion, identical across the three scenes.
            env.upper_delta[:] = 0.
            t = step*env.dt
            env.upper_delta[:, 2] = .02*math.sin(2*math.pi*.7*t)/cfg.control.action_scale
            lower = (env.preset_dof_pos[:, :12]-env.default_dof_pos[:, :12])/cfg.control.action_scale
            _, _, _, _, _, ids, _ = env.step(lower.expand(3, -1))
            if len(ids):
                resets.append(dict(step=step, ids=ids.cpu().tolist()))
            if step % 25 == 0:
                print('cart_label_balance step', step, flush=True)
    trace = {k: np.asarray(v) for k, v in env.rows.items()}
    error = trace['measured_resultant']-trace['cart_balance_reference']
    finite = all(np.isfinite(v).all() for v in trace.values())
    rows = {}
    for i, mode in enumerate(modes):
        rows[mode] = dict(force_vector_rms_n=float(np.sqrt(np.mean(np.sum(error[:, i, :3]**2, -1)))),
                          force_vector_peak_n=float(np.linalg.norm(error[:, i, :3], axis=-1).max()),
                          moment_vector_rms_nm=float(np.sqrt(np.mean(np.sum(error[:, i, 3:]**2, -1)))),
                          bias=error[:, i].mean(0).tolist(),
                          hand_force_rms_n=np.sqrt(np.mean(np.sum(trace['measured_hand_wrench'][:, i, :, :3]**2, -1), 0)).tolist(),
                          resultant_rms=np.sqrt(np.mean(trace['cart_balance_reference'][:, i]**2, 0)).tolist())
    contact_max = float(np.linalg.norm(trace['foot_contact'], axis=-1).max())
    height_min = float(trace['cart_body_pos'][..., 2].min())
    report = dict(result='MEASURED' if finite and not resets and trace['valid'].all() and contact_max == 0. and height_min > 1. else 'INVALID_EXECUTION',
                  samples=len(error), scenes=rows, resets=resets, finite=finite,
                  joint_configuration=joint_report, simulation_dt_s=cfg.sim.dt,
                  gravity_enabled=False, body_damping=0., upward_translation_m=3.,
                  minimum_cart_body_height_m=height_min,
                  maximum_foot_contact_n=contact_max,
                  limitations=['No floor contacts, gravity or body damping in this diagnostic.',
                               'Dual-grasp cart balance tests only resultant; equal/opposite individual errors can cancel.',
                               'Per-hand labels additionally tested using single-grasp scenes; this does not certify full walking cart contact dynamics.'])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'traces.npz', **trace)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return 0 if report['result'] == 'MEASURED' else 1


if __name__ == '__main__':
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    launcher = AppLauncher(args)
    try:
        code = run(args)
    except BaseException:
        # Kit shutdown may exit before the pending traceback is printed.
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
    raise SystemExit(code)
