"""Isaac Lab diagnostics for the cart alone: does it roll on the ground without hopping?

Spawns ground + cart (no robot), gives the cart an initial forward velocity, and logs
wheel lift-off and vertical velocity for a cylinder-collider and a sphere-collider wheel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .assets.cart_urdf import load_cart_params, write_cart_urdf
from .paths import ASSET_DIR, ROOT


def _metrics(wheel_bottom: np.ndarray, root_z: np.ndarray, dt: float) -> dict:
    """wheel_bottom: [T, 4] height of the lowest wheel point above ground; root_z: [T]."""
    lift = np.maximum(wheel_bottom, 0.0)
    vz = np.diff(root_z) / dt
    return {
        "max_wheel_lift_mm": float(lift.max() * 1000),
        "mean_wheel_lift_mm": float(lift.mean() * 1000),
        "steps_any_wheel_lifted_gt_5mm_pct": float((lift.max(axis=1) > 0.005).mean() * 100),
        "steps_any_wheel_lifted_gt_2mm_pct": float((lift.max(axis=1) > 0.002).mean() * 100),
        "root_z_peak_to_peak_mm": float((root_z.max() - root_z.min()) * 1000),
        "root_vz_rms_mps": float(np.sqrt(np.mean(vz ** 2))) if len(vz) else 0.0,
    }


def run_cart_diag(args, launcher) -> dict:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg
    import shutil

    params = load_cart_params()
    results = {}
    diag_dir = ASSET_DIR / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    dt = 0.005

    for collider in args.colliders:
        p = json.loads(json.dumps(params))
        p["wheels"]["collider"] = collider
        urdf = diag_dir / f"cart_{collider}.urdf"
        write_cart_urdf(p, params["derived"]["handle_height"], urdf)
        usd_dir = diag_dir / f"usd_{collider}"
        if usd_dir.exists():
            shutil.rmtree(usd_dir)
        converter = UrdfConverter(UrdfConverterCfg(
            asset_path=str(urdf), usd_dir=str(usd_dir), force_usd_conversion=True, fix_base=False,
            merge_fixed_joints=False, self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))))

        sim_cfg = sim_utils.SimulationCfg(device=device, dt=dt, render_interval=4,
                                          physics=PhysxCfg(solver_type=1, max_position_iteration_count=args.pos_iters,
                                                           max_velocity_iteration_count=args.vel_iters,
                                                           bounce_threshold_velocity=0.5))
        sim = sim_utils.SimulationContext(sim_cfg)

        @configclass
        class DiagSceneCfg(InteractiveSceneCfg):
            pass

        scene_cfg = DiagSceneCfg(num_envs=1, env_spacing=5., replicate_physics=True, filter_collisions=True)
        scene_cfg.ground = AssetBaseCfg(
            prim_path='/World/Ground',
            spawn=sim_utils.CuboidCfg(size=(200., 200., .1), collision_props=CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)))
        scene_cfg.light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=2000.))
        w = p["wheels"]
        scene_cfg.cart = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Cart', articulation_root_prim_path='/Geometry/base_link',
            spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path, activate_contact_sensors=False,
                                       collision_props=CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.),
                                       rigid_props=RigidBodyPropertiesCfg(linear_damping=0.01, angular_damping=0.01, max_depenetration_velocity=1.0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.), joint_pos={'.*': 0.}, joint_vel={'.*': 0.}),
            actuators={'casters': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0.,
                                                      damping={'caster_.*_swivel_joint': float(w["swivel_damping"]),
                                                               'wheel_.*_joint': float(w["wheel_damping"])})})
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.update(dt)
        cart = scene['cart']
        wheel_ids = [i for i, n in enumerate(cart.body_names) if n.startswith("wheel_")]
        pose = cart.data.default_root_pose.torch.clone()
        pose[:, :3] += scene.env_origins
        cart.write_root_pose_to_sim_index(root_pose=pose)
        vel = torch.zeros(1, 6, device=device)
        vel[0, 0] = args.v0
        cart.write_root_velocity_to_sim_index(root_velocity=vel)
        sim.forward()
        scene.update(0.)
        wheel_bottom, root_z, speed = [], [], []
        for _ in range(args.steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            wz = cart.data.body_link_pose_w.torch[0, wheel_ids, 2].cpu().numpy() - w["radius"]
            wheel_bottom.append(wz)
            root_z.append(float(cart.data.root_link_pose_w.torch[0, 2]))
            speed.append(float(cart.data.root_com_vel_w.torch[0, :2].norm()))
        res = _metrics(np.array(wheel_bottom), np.array(root_z), dt)
        res.update({"collider": collider, "v0": args.v0, "steps": args.steps, "pos_iters": args.pos_iters,
                    "speed_start": speed[0], "speed_end": speed[-1],
                    "wheel_bottom_first_step_mm": (np.array(wheel_bottom[0]) * 1000).round(2).tolist()})
        print("[cartdiag] " + json.dumps(res), flush=True)
        results[collider] = res
        break  # one simulation context per process; run other colliders in separate invocations
    path = ROOT / "logs/pushcart_cart_diag.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    previous.update({f"{k}_iters{args.pos_iters}_{args.vel_iters}": v for k, v in results.items()})
    path.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    return results
