"""G1 (preset grasp pose) with an optional cart, driven by the frozen HOMIE low-level policy.

``PushCartSimEnv`` subclasses ``HomieIsaacLabEnv`` and overrides only what the
push-cart scene needs: the scene (packed USD instead of the robot URDF), the
upper-body target (held at the preset grasp pose plus an optional correction),
externally supplied velocity commands, and resets to the preset pose. Every
other piece of HOMIE's observation / PD / stepping code is inherited unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, CollisionPropertiesCfg, RigidBodyPropertiesCfg

from homie_isaaclab.env import HomieIsaacLabEnv, HomieSceneCfg
from homie_isaaclab.math_utils import torch_rand_float

from .assets.cart_urdf import load_cart_params
from .paths import ASSET_REPORT, G1_GRASP_URDF, PACKED_USD, PALM_LINKS
from .pose import load_initial_pose
from .dynamics import (centroidal_quantities, compensate_subtree_wrench,
                       contact_buffer_wrench, rotate_inertia, rotation_matrix_xyzw, shift_wrench)


class PushCartSimEnv(HomieIsaacLabEnv):
    def __init__(self, cfg, project_root: Path, device="cuda:0", headless=True, *, with_cart=True,
                 pose=None, cart_params=None, sim_context=None, allow_robot_cart_contact=False,
                 full_dynamics=None):
        self.with_cart = with_cart
        self._full_dynamics_override = full_dynamics
        self.allow_robot_cart_contact = allow_robot_cart_contact      # diagnostics only: undo the asset-level filter
        self.pose = pose or load_initial_pose()
        self.cart_params = cart_params or load_cart_params()
        if with_cart and self.cart_params["model"] != "caster":
            raise NotImplementedError("PushCartSimEnv currently loads the caster articulation model only.")
        if not ASSET_REPORT.is_file() or not PACKED_USD.is_file():
            raise FileNotFoundError("packed asset missing; run `pushcart.py pack` first")
        report = json.loads(ASSET_REPORT.read_text(encoding="utf-8"))
        robot_usd = Path(report["robot_usd"])

        # ---- identical preamble to HomieIsaacLabEnv.__init__ (g1_grasp.urdf has the same 27 joints) ----
        self.cfg, self.device, self.headless = cfg, device, headless
        self.num_envs = cfg.env.num_envs
        self.num_actions = self.num_dof = cfg.env.num_dofs
        self.num_lower_dof = cfg.env.num_actions
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_one_step_obs = cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = cfg.env.num_one_step_privileged_obs
        self.actor_history_length = cfg.env.num_actor_history
        self.critic_history_length = cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_obs
        self.critic_proprioceptive_obs_length = self.num_privileged_obs
        self.sim_params = SimpleNamespace(dt=cfg.sim.dt)
        self._parse_cfg(cfg)
        self.max_episode_length = int(self.max_episode_length)
        self.custom_origins = False

        urdf = G1_GRASP_URDF
        xml = ET.parse(urdf).getroot()
        joints = [j for j in xml.findall('joint') if j.get('type') in ('revolute', 'continuous', 'prismatic')]
        self.dof_names = [j.get('name') for j in joints]
        if len(self.dof_names) != 27:
            raise ValueError(f'HIM symmetry and observation layout require 27 joints, found {len(joints)}')
        stiffness, damping = {}, {}
        for name in self.dof_names:
            key = next(k for k in cfg.control.stiffness if k in name)
            stiffness[name], damping[name] = cfg.control.stiffness[key], cfg.control.damping[key]
        limits = {j.get('name'): float(j.find('limit').get('effort')) for j in joints}
        velocities = {j.get('name'): float(j.find('limit').get('velocity')) for j in joints}
        if cfg.sim.substeps != 1 or cfg.terrain.mesh_type != 'plane' or cfg.control.control_type != 'M':
            raise ValueError('This HOMIE profile supports plane terrain, substeps=1, and mixed control M.')

        sim_cfg = sim_utils.SimulationCfg(
            device=device, dt=cfg.sim.dt, gravity=tuple(cfg.sim.gravity),
            render_interval=cfg.control.decimation,
            physics=PhysxCfg(
                solver_type=cfg.sim.physx.solver_type,
                max_position_iteration_count=cfg.sim.physx.num_position_iterations,
                max_velocity_iteration_count=cfg.sim.physx.num_velocity_iterations,
                bounce_threshold_velocity=cfg.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=cfg.sim.physx.max_gpu_contact_pairs),
        )
        self.sim = sim_context if sim_context is not None else sim_utils.SimulationContext(sim_cfg)
        scene_cfg = HomieSceneCfg(num_envs=self.num_envs, env_spacing=cfg.env.env_spacing,
                                  replicate_physics=True, filter_collisions=True)
        scene_cfg.ground = AssetBaseCfg(
            prim_path='/World/Ground',
            spawn=sim_utils.CuboidCfg(
                size=(2000., 2000., .1), collision_props=CollisionPropertiesCfg(
                    contact_offset=cfg.sim.physx.contact_offset, rest_offset=cfg.sim.physx.rest_offset),
                # PhysX GPU filtered point/friction reports require a rigid
                # partner. A fixed kinematic ground has the same contact
                # geometry, while a static collider silently yields no report.
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.16, .18, .21)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)),
        )
        scene_cfg.light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=2000.))

        # ---- robot (+ cart) from the packed asset, same spawn properties as HOMIE's UrdfFileCfg ----
        spawn_props = dict(
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(
                contact_offset=cfg.sim.physx.contact_offset, rest_offset=cfg.sim.physx.rest_offset),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=cfg.asset.disable_gravity,
                linear_damping=cfg.asset.linear_damping, angular_damping=cfg.asset.angular_damping,
                max_linear_velocity=cfg.asset.max_linear_velocity,
                max_angular_velocity=cfg.asset.max_angular_velocity,
                max_depenetration_velocity=cfg.sim.physx.max_depenetration_velocity),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=not bool(cfg.asset.self_collisions),
                solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        )
        preset = self.pose["robot"]["joint_pos"]
        robot_init = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.pose["robot"]["base_pos"]), rot=tuple(self.pose["robot"]["base_quat_xyzw"]),
            joint_pos={n: float(preset[n]) for n in self.dof_names}, joint_vel={'.*': 0.})
        actuators = {
            'legs': ImplicitActuatorCfg(
                joint_names_expr=self.dof_names[:12], stiffness=0., damping=0., armature=cfg.asset.armature,
                effort_limit_sim={n: limits[n] for n in self.dof_names[:12]},
                velocity_limit_sim={n: velocities[n] for n in self.dof_names[:12]}),
            'upper': ImplicitActuatorCfg(
                joint_names_expr=self.dof_names[12:],
                armature=cfg.asset.armature,
                stiffness={n: stiffness[n] for n in self.dof_names[12:]},
                damping={n: damping[n] for n in self.dof_names[12:]},
                effort_limit_sim={n: limits[n] for n in self.dof_names[12:]},
                velocity_limit_sim={n: velocities[n] for n in self.dof_names[12:]}),
        }
        if with_cart:
            robot_prim = '{ENV_REGEX_NS}/Scene/Robot'
            scene_cfg.scene_usd = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Scene',
                                               spawn=sim_utils.UsdFileCfg(usd_path=str(PACKED_USD), **spawn_props))
            scene_cfg.robot = ArticulationCfg(prim_path=robot_prim, articulation_root_prim_path='/Geometry/pelvis',
                                              spawn=None, init_state=robot_init, actuators=actuators)
            derived = self.cart_params["derived"]
            yaw = float(derived["base_yaw"])
            wheels = self.cart_params["wheels"]
            scene_cfg.cart = ArticulationCfg(
                prim_path='{ENV_REGEX_NS}/Scene/Cart', articulation_root_prim_path='/Geometry/base_link', spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(float(v) for v in derived["base_pos_world"]),
                    rot=(0., 0., math.sin(yaw / 2), math.cos(yaw / 2)), joint_pos={'.*': 0.}, joint_vel={'.*': 0.}),
                actuators={'casters': ImplicitActuatorCfg(
                    joint_names_expr=['.*'], stiffness=0.,
                    damping={'caster_.*_swivel_joint': float(wheels["swivel_damping"]),
                             'wheel_.*_joint': float(wheels["wheel_damping"])})},
            )
        else:
            robot_prim = '{ENV_REGEX_NS}/Robot'
            scene_cfg.robot = ArticulationCfg(
                prim_path=robot_prim, articulation_root_prim_path='/Geometry/pelvis',
                spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd), **spawn_props),
                init_state=robot_init, actuators=actuators)
        parents = {j.find('child').get('link'): j.find('parent').get('link') for j in xml.findall('joint')}
        self._contact_names = ['left_ankle_roll_link', 'right_ankle_roll_link', 'torso_link']
        for i, name in enumerate(self._contact_names):
            chain = [name]
            while chain[0] in parents:
                chain.insert(0, parents[chain[0]])
            setattr(scene_cfg, f'contact_{i}', ContactSensorCfg(
                prim_path=robot_prim + '/Geometry/' + '/'.join(chain), update_period=0.))
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene['robot']
        self.cart = self.scene['cart'] if with_cart else None
        self.contacts = [self.scene[f'contact_{i}'] for i in range(len(self._contact_names))]

        # ---- per-link physics properties, exactly as HomieIsaacLabEnv does for the nested Sim 6 hierarchy ----
        from pxr import PhysxSchema, Usd, UsdPhysics
        roots = [robot_prim.replace('{ENV_REGEX_NS}', '/World/envs/env_0')]
        if with_cart:
            roots.append('/World/envs/env_0/Scene/Cart')
        for root_path in roots:
            source_root = self.sim.stage.GetPrimAtPath(root_path)
            body_paths = [str(p.GetPath()) for p in Usd.PrimRange(source_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            for path in body_paths:
                prim = self.sim.stage.GetPrimAtPath(path)
                if prim.IsInstance():
                    prim.SetInstanceable(False)
                sim_utils.modify_rigid_body_properties(path, spawn_props['rigid_props'], stage=self.sim.stage)
            collision_paths = [str(p.GetPath()) for p in Usd.PrimRange(source_root, Usd.TraverseInstanceProxies())
                               if p.HasAPI(UsdPhysics.CollisionAPI)]
            for path in collision_paths:
                prim = self.sim.stage.GetPrimAtPath(path)
                while prim.IsInstanceProxy():
                    parent = prim.GetParent()
                    while parent.IsInstanceProxy():
                        parent = parent.GetParent()
                    parent.SetInstanceable(False)
                    prim = self.sim.stage.GetPrimAtPath(path)
                sim_utils.modify_collision_properties(path, spawn_props['collision_props'], stage=self.sim.stage)
        for sensor in self.contacts:
            for env_id in range(self.num_envs):
                path = sensor.cfg.prim_path.replace('env_.*', f'env_{env_id}')
                prim = self.sim.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise ValueError(f'Contact link missing from imported USD: {path}')
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if with_cart:
            for env_id in range(self.num_envs):
                for i in range(4):
                    path = f'/World/envs/env_{env_id}/Scene/Cart/Geometry/base_link/caster_{i}_swivel/wheel_{i}'
                    prim = self.sim.stage.GetPrimAtPath(path)
                    if not prim.IsValid():
                        raise ValueError(f'V5 wheel rigid body missing: {path}')
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        if with_cart and allow_robot_cart_contact:
            for env_id in range(self.num_envs):
                prim = self.sim.stage.GetPrimAtPath(f'/World/envs/env_{env_id}/Scene/Cart')
                if prim.IsInstance():
                    prim.SetInstanceable(False)
                UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().ClearTargets(True)
            print('[pushcart] DIAGNOSTIC: robot <-> cart collisions enabled', flush=True)
        if with_cart and "solver" in self.cart_params:
            # The spawn applied HOMIE's 4 / 0 to both articulations; give the cart its own (higher) counts so the
            # robot-grasp-cart constraint loop is solved more tightly (see cart_params.yaml: solver).
            solver = self.cart_params["solver"]
            cart_root_0 = next(p for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/envs/env_0/Scene/Cart'),
                                                        Usd.TraverseInstanceProxies())
                               if p.HasAPI(UsdPhysics.ArticulationRootAPI))
            for env_id in range(self.num_envs):
                path = str(cart_root_0.GetPath()).replace('env_0', f'env_{env_id}')
                prim = self.sim.stage.GetPrimAtPath(path)
                while prim.IsInstanceProxy():          # authoring needs a real prim, not an instance proxy
                    parent = prim.GetParent()
                    while parent.IsInstanceProxy():
                        parent = parent.GetParent()
                    parent.SetInstanceable(False)
                    prim = self.sim.stage.GetPrimAtPath(path)
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr().Set(int(solver["position_iterations"]))
                api.CreateSolverVelocityIterationCountAttr().Set(int(solver["velocity_iterations"]))
        self.sim.reset()
        self.scene.update(cfg.sim.dt)
        self.env_origins = self.scene.env_origins
        self.sim.set_camera_view([2.8, -2.6, 1.9], [0.5, 0., 0.7])
        joint_ids, ordered_names = self.robot.find_joints(self.dof_names, preserve_order=True)
        if ordered_names != self.dof_names:
            raise ValueError('Isaac Lab joint-name resolution changed the HOMIE policy order')
        self.joint_ids = torch.tensor(joint_ids, device=device, dtype=torch.int32)
        self.body_names = list(self.robot.body_names)
        self._init_virtual_links(xml)
        self.num_bodies = len(self.body_names)
        self.contact_body_ids = torch.tensor([self.body_names.index(n) for n in self._contact_names], device=device)
        self._init_indices()
        self.hard_dof_pos_limits = torch.tensor(
            [[float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))] for j in joints], device=device)
        middle = self.hard_dof_pos_limits.mean(-1)
        half = (self.hard_dof_pos_limits[:, 1] - self.hard_dof_pos_limits[:, 0]) * .5 * cfg.rewards.soft_dof_pos_limit
        self.dof_pos_limits = torch.stack((middle - half, middle + half), -1)
        self.dof_vel_limits = torch.tensor(list(velocities.values()), device=device)
        self.torque_limits = torch.tensor(list(limits.values()), device=device)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=device)
        self.rew_buf = torch.zeros(self.num_envs, device=device)
        self.reset_buf = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        # external command / upper-body buffers must exist before _init_task_buffers -> _resample_commands paths
        self.command_buffer = torch.zeros(self.num_envs, 3, device=device)
        self._sync_states()
        self._init_task_buffers()
        self._prepare_reward_function()
        # preset pose in HOMIE's joint order; upper body held there (in HOMIE action units), plus a correction slot.
        # The hold *target* is the preset plus the measured gravity sag (stance --apply), so the arms rest at the preset.
        self.preset_dof_pos = torch.tensor([float(preset[n]) for n in self.dof_names], device=device).unsqueeze(0)
        offset = self.pose["robot"].get("upper_hold_offset") or {}
        self.upper_hold_offset = torch.tensor([float(offset.get(n, 0.)) for n in self.dof_names[self.num_lower_dof:]],
                                              device=device).unsqueeze(0)
        hold = (self.preset_dof_pos[:, self.num_lower_dof:] + self.upper_hold_offset
                - self.default_dof_pos[:, self.num_lower_dof:]) / cfg.control.action_scale
        self.upper_hold_actions = hold.clamp(self.action_min[:, self.num_lower_dof:], self.action_max[:, self.num_lower_dof:])
        self.upper_delta = torch.zeros(self.num_envs, self.num_actions - self.num_lower_dof, device=device)
        self.joint_pos_target = self.preset_dof_pos.repeat(self.num_envs, 1)
        # Grasp wrench bookkeeping: the palm link's incoming (articulation) joint force balances the grasp
        # constraint force plus the weight of the palm + finger subtree, so the subtree mass is needed.
        self.palm_body_ids = torch.tensor([self.body_names.index(PALM_LINKS[s]) for s in ("left", "right")],
                                          device=device, dtype=torch.long)
        masses = self.robot.data.body_mass.torch[0]
        self.hand_subtree_mass = torch.tensor(
            [float(sum(masses[i] for i, n in enumerate(self.robot.body_names) if n.startswith(f"{s}_hand")))
             for s in ("left", "right")], device=device)
        self._randomize_body_properties()
        # Cache only after the actual simulator mass/inertia/COM randomization.
        self.refresh_dynamics_properties(self.robot)
        if self.cart is not None:
            self.refresh_dynamics_properties(self.cart)
        self._init_dynamics(xml)
        self._material_buckets = torch.empty(64, 3)
        self._material_buckets[:, :2] = torch_rand_float(*cfg.domain_rand.friction_range, (64, 1), 'cpu')
        self._material_buckets[:, 2] = torch_rand_float(*cfg.domain_rand.restitution_range, (64,), 'cpu')
        print(f'[pushcart] {self.num_envs} envs, {len(self.dof_names)} joints, {self.num_bodies} bodies; '
              f'cart={"caster" if with_cart else "none"}; actor=456 critic=79 action=12')
        print(f'[pushcart] Lab joint IDs in HOMIE order: {self.joint_ids.tolist()}')
        self.physics_report = self._physics_report()

    def _physics_report(self):
        """Same content as HOMIE's report, but the robot prim lives under /Scene when the cart is present."""
        from pxr import Usd, UsdPhysics
        root_path = '/World/envs/env_0/Scene/Robot' if self.with_cart else '/World/envs/env_0/Robot'
        root = self.sim.stage.GetPrimAtPath(root_path)
        shapes, bodies = [], []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append({'path': str(prim.GetPath()),
                               'linear_damping': prim.GetAttribute('physxRigidBody:linearDamping').Get(),
                               'angular_damping': prim.GetAttribute('physxRigidBody:angularDamping').Get()})
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                shapes.append({'path': str(prim.GetPath()),
                               'contact_offset': prim.GetAttribute('physxCollision:contactOffset').Get(),
                               'rest_offset': prim.GetAttribute('physxCollision:restOffset').Get()})
        report = {
            'physics_dt_s': self.sim.get_physics_dt(), 'control_dt_s': self.dt,
            'decimation': self.cfg.control.decimation, 'max_episode_steps': self.max_episode_length,
            'solver_type': self.sim.cfg.physics.solver_type,
            'joint_order': self.dof_names,
            'armature': self.robot.data.joint_armature.torch[0, self.joint_ids].cpu().tolist(),
            'stiffness': self.robot.data.joint_stiffness.torch[0, self.joint_ids].cpu().tolist(),
            'damping': self.robot.data.joint_damping.torch[0, self.joint_ids].cpu().tolist(),
            'body_names': self.robot.body_names,
            'body_masses': self.robot.data.body_mass.torch[0].cpu().tolist(),
            'body_inertias': self.robot.data.body_inertia.torch[0].cpu().tolist(),
            'body_com_poses': self.robot.data.body_com_pose_b.torch[0].cpu().tolist(),
            'rigid_body_properties': bodies, 'collision_shapes': shapes,
            'replicate_physics': self.scene.cfg.replicate_physics,
            'filter_collisions': self.scene.cfg.filter_collisions,
            'randomization': self.cfg.domain_rand.use_random,
            'with_cart': self.with_cart,
        }
        if self.cart is not None:
            cart_root = next(p for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/envs/env_0/Scene/Cart'))
                             if p.HasAPI(UsdPhysics.ArticulationRootAPI))
            report['cart'] = {
                'joint_names': list(self.cart.joint_names),
                'joint_damping': self.cart.data.joint_damping.torch[0].cpu().tolist(),
                'body_names': list(self.cart.body_names),
                'body_masses': self.cart.data.body_mass.torch[0].cpu().tolist(),
                'solver_position_iterations': cart_root.GetAttribute('physxArticulation:solverPositionIterationCount').Get(),
                'solver_velocity_iterations': cart_root.GetAttribute('physxArticulation:solverVelocityIterationCount').Get(),
                'robot_cart_collisions_filtered': [str(p) for p in UsdPhysics.FilteredPairsAPI(
                    self.sim.stage.GetPrimAtPath('/World/envs/env_0/Scene/Cart')).GetFilteredPairsRel().GetTargets()],
            }
        return report

    # ------------------------------------------------------------------ external interfaces
    def set_command(self, vx: float, vy: float, wz: float, env_ids=None):
        """Velocity command in HOMIE's convention; height stays at the trained default."""
        ids = slice(None) if env_ids is None else env_ids
        self.command_buffer[ids, 0] = float(vx)
        self.command_buffer[ids, 1] = float(vy)
        self.command_buffer[ids, 2] = float(wz)
        self.commands[ids, :3] = self.command_buffer[ids]
        self.commands[ids, 3] = 0.
        self.commands[ids, 4] = self.cfg.rewards.base_height_target

    def set_upper_delta(self, delta: torch.Tensor):
        """Correction on top of the held grasp pose, in HOMIE action units (phase B high-level policy)."""
        self.upper_delta[:] = delta.to(self.device)

    def cart_state(self):
        """Pose and velocity of the fixed base-link geometric origin, world axes.

        Task commands and evaluation use this same point, independent of payload
        COM. The fallback shifts the root body's COM velocity, never the entire
        multibody cart COM velocity, to the root link origin.
        """
        if self.cart is None:
            return None
        data = self.cart.data
        pose = data.root_link_pose_w.torch
        if hasattr(data, 'root_link_vel_w'):
            vel = data.root_link_vel_w.torch
            linear = vel[:, :3]
        else:
            vel = data.root_com_vel_w.torch
            linear = vel[:, :3] + torch.cross(
                vel[:, 3:6], pose[:, :3]-data.root_com_pose_w.torch[:, :3], dim=-1)
        return pose[:, :3], pose[:, 3:7], linear, vel[:, 3:6]

    def grasp_wrench(self):
        """Latest physics-step subtree-compensated hand->cart load, at palm origins.

        Returns world force, world moment, palm-axis force, all [N,2,3]. This is
        a model-based estimate, not an independent contact sensor. Samples after
        reset are zero and invalid until two physical steps have elapsed; callers
        must use ``dynamics_wrench_valid`` for reward/measurement gating.
        """
        return (self._hand_wrench[..., :3], self._hand_wrench[..., 3:], self._hand_force_palm)

    def cart_com_world(self):
        """Mass-weighted COM of the entire cart articulation, including payload and casters."""
        cached = getattr(self, '_dynamics_property_cache', {}).get(id(self.cart))
        if cached is None:
            # Lightweight objects without initialized caches retain the audit path.
            mass = self.cart.data.body_mass.torch
            return (self.cart.data.body_com_pose_w.torch[..., :3]*mass[..., None]).sum(1)/mass.sum(1, keepdim=True)
        pose = self.cart.data.body_link_pose_w.torch
        rotation = rotation_matrix_xyzw(pose[..., 3:7])
        position = pose[..., :3] + (rotation @ cached['com_offset'][..., None]).squeeze(-1)
        return (position * cached['mass'][..., None]).sum(1) / cached['total_mass']

    def handle_center_world(self):
        """Known handle-center geometry transformed by the cart base-link pose.

        This simulation reference is for labels/rewards/audits; the Actor obtains
        the same geometric point through its separate fixed-grasp FK chain.
        """
        pose = self.cart.data.root_link_pose_w.torch
        if hasattr(self, 'bar_local'):
            local = self.bar_local
        else:
            derived = self.cart_params['derived']
            offset = pose.new_tensor(derived['bar_center_world']) - pose.new_tensor(derived['base_pos_world'])
            c, s = math.cos(float(derived['base_yaw'])), math.sin(float(derived['base_yaw']))
            local = torch.stack((c*offset[0]+s*offset[1], -s*offset[0]+c*offset[1], offset[2]))
        return pose[:, :3] + (rotation_matrix_xyzw(pose[:, 3:7]) @ local.reshape(3, 1)).squeeze(-1)

    def cart_hand_wrench(self, force_w, torque_w, *, reference='handle_center'):
        """Aggregate compensated hand loads about a named world-frame point.

        Both input moments and points refer to palm origins. Re-expressing both
        at the grasp anchors produces the identical resultant, including r x F.
        The V5.1 label reference is handle_center; cart_com is an explicit audit
        option. The returned third value is the selected reference, not a COM.
        """
        from .cart_quality import resultant_wrench
        if reference == 'handle_center':
            point = self.handle_center_world()
        elif reference == 'cart_com':
            point = self.cart_com_world()
        else:
            raise ValueError(f'Unknown cart wrench reference: {reference}')
        points = self.rigid_body_states[:, self.palm_body_ids, :3]
        force, moment = resultant_wrench(force_w, torque_w, points, point)
        return force, moment, point

    def _randomize_body_properties(self):
        super()._randomize_body_properties()
        if hasattr(self, '_dynamics_property_cache'):
            self.refresh_dynamics_properties(self.robot)

    def refresh_dynamics_properties(self, articulation, env_ids=None, *,
                                    masses=None, inertias=None, coms=None):
        """Refresh immutable-between-randomizations properties after simulator writes.

        Values use the setter convention: selected environment rows, all bodies,
        mass [N,B], column-major inertia [N,B,9] (or [N,B,3,3]), and local COM
        pose [N,B,7], xyzw. Omitted values are read from the simulator. Callers
        that change any property must invoke this hook after their setters; the
        robot/cart randomization paths do so. Copies prevent setter-buffer aliasing.
        Full audit ``_body_dynamics(asset)`` deliberately bypasses this cache.
        """
        if not hasattr(self, '_dynamics_property_cache'):
            self._dynamics_property_cache = {}
        key = id(articulation)
        cached = self._dynamics_property_cache.get(key)
        if env_ids is not None and cached is None:
            self.refresh_dynamics_properties(articulation)
            cached = self._dynamics_property_cache[key]
        data = articulation.data
        ids = slice(None) if env_ids is None else env_ids
        mass = data.body_mass.torch[ids] if masses is None else masses
        inertia = data.body_inertia.torch[ids] if inertias is None else inertias
        com = data.body_com_pose_b.torch[ids] if coms is None else coms
        if mass.ndim != 2 or com.shape != (*mass.shape, 7):
            raise ValueError('dynamics refresh needs mass [N,B] and local COM pose [N,B,7]')
        if inertia.shape not in ((*mass.shape, 9), (*mass.shape, 3, 3)):
            raise ValueError('dynamics refresh needs column-major inertia [N,B,9] or [N,B,3,3]')
        values = {'mass': mass, 'com_offset': com[..., :3],
                  'inertia': inertia.reshape(*mass.shape, 3, 3).transpose(-1, -2),
                  'total_mass': mass.sum(1, keepdim=True)}
        if env_ids is None:
            self._dynamics_property_cache[key] = {name: value.detach().clone() for name, value in values.items()}
        else:
            for name, value in values.items():
                if cached[name][env_ids].shape != value.shape:
                    raise ValueError('dynamics refresh values must match the selected environment rows')
                cached[name][env_ids] = value

    def _body_dynamics(self, articulation, body_ids=None):
        data = articulation.data
        cached = getattr(self, '_dynamics_property_cache', {}).get(id(articulation))
        if body_ids is not None and cached is not None:
            pose = data.body_link_pose_w.torch[:, body_ids]
            rotation = rotation_matrix_xyzw(pose[..., 3:7])
            position = pose[..., :3] + (rotation @ cached['com_offset'][:, body_ids, :, None]).squeeze(-1)
            inertia = rotation @ cached['inertia'][:, body_ids] @ rotation.transpose(-1, -2)
            return (cached['mass'][:, body_ids], position,
                    data.body_com_vel_w.torch[:, body_ids], inertia)
        ids = slice(None) if body_ids is None else body_ids
        pose = data.body_com_pose_w.torch[:, ids]
        velocity = data.body_com_vel_w.torch[:, ids]
        # Isaac Lab 6.1.17's body_inertia docstring says "world", but its
        # implementation calls PhysX get_inertias(): COM-centered, prim-frame
        # axes, column-major. See omni.physics.tensors.api.ArticulationView.
        inertia_link = data.body_inertia.torch[:, ids].reshape(self.num_envs, -1, 3, 3).transpose(-1, -2)
        return (data.body_mass.torch[:, ids], pose[..., :3], velocity,
                rotate_inertia(inertia_link, data.body_link_pose_w.torch[:, ids, 3:7]))

    def centroidal_state(self, articulation):
        """Actual assembly mass, COM, velocity, momenta and inertia in world axes.

        Includes every physical body (including wheels), excludes virtual links.
        COM velocities and positions consistently refer to the same points.
        """
        mass, position, velocity, inertia = self._body_dynamics(articulation)
        return centroidal_quantities(mass, position, velocity[..., :3], velocity[..., 3:], inertia)

    def _init_dynamics(self, xml):
        diagnostics = getattr(getattr(self, 'task_cfg', None), 'diagnostics', None)
        override = getattr(self, '_full_dynamics_override', None)
        self.full_dynamics = bool(getattr(diagnostics, 'full_dynamics', False) if override is None else override)
        # Use exact rigid-body paths. The stock ContactSensor resolver can
        # also select same-named visual/collider descendants in imported USD.
        # These dedicated views contain exactly one sensing body per env.
        ground_paths = []
        if self.full_dynamics:
            from isaaclab_physx.physics import PhysxManager
            physics_view = PhysxManager.get_physics_sim_view()
            ground_paths = [sensor.cfg.prim_path.replace('env_.*', 'env_*') for sensor in self.contacts[:2]]
            if self.with_cart:
                ground_paths += [f'/World/envs/env_*/Scene/Cart/Geometry/base_link/caster_{i}_swivel/wheel_{i}'
                                 for i in range(4)]
        self._ground_contact_views = [physics_view.create_rigid_contact_view(
            path, filter_patterns=['/World/Ground'], max_contact_data_count=32*self.num_envs)
            for path in ground_paths]
        for path, view in zip(ground_paths, self._ground_contact_views):
            if view.sensor_count != self.num_envs or view.filter_count != 1:
                raise RuntimeError(f'V5 ground contact view did not resolve exactly: {path}, '
                                   f'sensors={view.sensor_count}, filters={view.filter_count}')
        parents = {joint.find('child').get('link'): joint.find('parent').get('link')
                   for joint in xml.findall('joint')}
        self._hand_subtree_ids = []
        for side in ('left', 'right'):
            palm = PALM_LINKS[side]
            names = {palm}
            while True:
                enlarged = names | {child for child, parent in parents.items() if parent in names}
                if enlarged == names:
                    break
                names = enlarged
            ids = [i for i, name in enumerate(self.robot.body_names) if name in names]
            if not ids or palm not in self.robot.body_names:
                raise ValueError(f'V5 requires a physical palm and its distal subtree: {palm}')
            self._hand_subtree_ids.append(torch.tensor(ids, device=self.device, dtype=torch.long))
        selected = sorted({int(i) for ids in self._hand_subtree_ids for i in ids.tolist()})
        self._hand_body_ids = torch.tensor(selected, device=self.device, dtype=torch.long)
        self._hand_local_ids = [torch.tensor([selected.index(i) for i in ids.tolist()],
                                            device=self.device, dtype=torch.long)
                                for ids in self._hand_subtree_ids]
        self._dynamics_gravity = torch.tensor(self.cfg.sim.gravity, device=self.device)
        if self.cfg.asset.disable_gravity:
            self._dynamics_gravity.zero_()
        self._dynamics_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._all_dynamics_valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.dynamics_wrench_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.dynamics_fault = torch.zeros_like(self.dynamics_wrench_valid)
        self.ground_wrench_valid = torch.zeros_like(self.dynamics_wrench_valid)
        self._hand_wrench = torch.zeros(self.num_envs, 2, 6, device=self.device)
        self._hand_force_palm = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.hand_force_peak = torch.zeros(self.num_envs, 2, device=self.device)
        self.hand_moment_peak = torch.zeros_like(self.hand_force_peak)
        self._feet_wrench = torch.zeros(self.num_envs, 2, 6, device=self.device)
        self._wheel_wrench = torch.zeros(self.num_envs, 4, 6, device=self.device)
        steps = self.cfg.control.decimation
        shapes = {'hand_wrench': (2, 6), 'feet_wrench': (2, 6), 'wheel_wrench': (4, 6),
                  'cart_hand_wrench': (6,), 'robot_reaction_wrench': (6,),
                  'cart_applied_wrench': (6,), 'cart_applied_reference': (3,), 'cart_applied_wrench_com': (6,),
                  'hand_reference': (2, 3),
                  'robot_momentum': (6,), 'cart_momentum': (6,), 'robot_com': (3,), 'cart_com': (3,)}
        self.physics_samples = ({name: torch.zeros(steps, self.num_envs, *shape, device=self.device)
                                 for name, shape in shapes.items()} if self.full_dynamics else {})
        if self.full_dynamics:
            self.physics_samples['valid'] = torch.zeros(steps, self.num_envs, device=self.device, dtype=torch.bool)
            self.physics_samples['ground_valid'] = torch.zeros_like(self.physics_samples['valid'])
            self.physics_samples['fault'] = torch.zeros_like(self.physics_samples['valid'])
        self.physics_impulses = {name: torch.zeros(self.num_envs, *shapes[name], device=self.device)
                                for name in (('hand_wrench', 'feet_wrench', 'wheel_wrench',
                                              'cart_hand_wrench', 'robot_reaction_wrench', 'cart_applied_wrench')
                                             if self.full_dynamics else ())}
        self.physics_valid_duration = torch.zeros(self.num_envs, device=self.device)
        self.physics_impulse_reference = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.physics_fixed_origin_impulses = {name: torch.zeros(self.num_envs, 6, device=self.device)
            for name in (('feet_wrench', 'wheel_wrench', 'cart_hand_wrench', 'robot_reaction_wrench', 'cart_applied_wrench')
                         if self.full_dynamics else ())}
        self._cart_applied_reference = torch.zeros(self.num_envs, 3, device=self.device)
        self.terminal_physics = None
        self.dynamics_metadata = {
            'hand_source': 'incoming joint wrench minus hand-subtree Newton-Euler inertial/gravity load',
            'acceleration_source': 'PhysX get_link_accelerations, world COM linear and world angular',
            'inertia_source_frame': 'PhysX COM-centered tensor in rigid-body-prim axes',
            'ground_source': 'individual ground normal and friction forces at their own contact points',
            'ground_body': 'fixed kinematic ground, enables PhysX GPU rigid partner filtering',
            'sample_period_s': float(self.cfg.sim.dt), 'reset_warmup_physics_steps': 2,
            'hand_moment_reference': 'palm link origin',
            'feet_moment_reference': 'instantaneous robot assembly COM',
            'wheel_moment_reference': 'instantaneous cart assembly COM',
            'physics_impulses_definition': 'integrals at each sample reference; moving-reference moments are not fixed-origin angular impulses',
            'physics_fixed_origin_impulses_definition': 'world-axis impulses about control-interval initial robot/cart COM; references[:,0/1]',
            'cart_applied_wrench_reference': 'pre-step cart COM, saved as cart_applied_reference; cart_applied_wrench_com is shifted to sampled COM',
            'hand_source_limits': 'other hand external loads and body damping enter the estimate; validity is numerical/warmup validity, not calibration proof',
            'dynamic_contact_wrench_calibrated': False,
            'mode': 'audit' if self.full_dynamics else 'train',
            'ground_measurement_enabled': self.full_dynamics,
            'full_physics_trajectory_enabled': self.full_dynamics,
            'cart_task_reference': 'cart base-link geometric origin',
            'cart_hand_label_reference': 'known handle center',
            'hand_peak_definition': 'uncapped per-hand norm maxima over current control interval; warm finite samples only',
        }

    def _reset_dynamics(self, env_ids):
        """Invalidate pre-reset solver data and clear only the selected environments."""
        if not hasattr(self, '_dynamics_steps') or len(env_ids) == 0:
            return
        self._dynamics_steps[env_ids] = 0
        self._all_dynamics_valid[env_ids] = True
        self.dynamics_wrench_valid[env_ids] = False
        self.dynamics_fault[env_ids] = False
        self.ground_wrench_valid[env_ids] = False
        for buffer in (self._hand_wrench, self._hand_force_palm, self._feet_wrench, self._wheel_wrench,
                       self.hand_force_peak, self.hand_moment_peak,
                       self.physics_valid_duration, self.physics_impulse_reference, self._cart_applied_reference,
                       *self.physics_impulses.values(), *self.physics_fixed_origin_impulses.values()):
            buffer[env_ids] = 0
        for buffer in self.physics_samples.values():
            buffer[:, env_ids] = 0

    def ground_contact_wrenches(self):
        """Audit ground wrenches at assembly COMs; validity is False in train mode."""
        return {'feet': self._feet_wrench, 'wheels': self._wheel_wrench,
                'valid': self.ground_wrench_valid}

    def _contact_wrench(self, view, reference):
        import warp as wp
        forces, points, normals, _, counts, starts = view.get_contact_data(dt=float(self.cfg.sim.dt))
        # get_friction_data reuses count/start buffers; consume normal data first.
        normal, valid = contact_buffer_wrench(
            wp.to_torch(forces)*wp.to_torch(normals), wp.to_torch(points),
            wp.to_torch(counts).reshape(self.num_envs, -1),
            wp.to_torch(starts).reshape(self.num_envs, -1), reference)
        forces, points, counts, starts = view.get_friction_data(dt=float(self.cfg.sim.dt))
        friction, friction_valid = contact_buffer_wrench(
            wp.to_torch(forces), wp.to_torch(points),
            wp.to_torch(counts).reshape(self.num_envs, -1),
            wp.to_torch(starts).reshape(self.num_envs, -1), reference)
        return normal+friction, valid & friction_valid

    def _incoming_hand_wrench(self):
        """PhysX incoming joint wrenches, in each palm's link axes."""
        import warp as wp
        return wp.to_torch(self.robot.root_view.get_link_incoming_joint_force()).reshape(
            self.num_envs, self.robot.num_bodies, 6)[:, self.palm_body_ids]

    def _measure_hand_wrench(self, mass, position, velocity, inertia, acceleration, subtree_ids):
        palm_pose = self.robot.data.body_link_pose_w.torch[:, self.palm_body_ids]
        rotation = rotation_matrix_xyzw(palm_pose[..., 3:7])
        incoming = self._incoming_hand_wrench()
        incoming_world = torch.cat(((rotation @ incoming[..., :3, None]).squeeze(-1),
                                    (rotation @ incoming[..., 3:, None]).squeeze(-1)), -1)
        hands = [compensate_subtree_wrench(
            incoming_world[:, side], palm_pose[:, side, :3], mass[:, ids], position[:, ids],
            velocity[:, ids, 3:], acceleration[:, ids, :3], acceleration[:, ids, 3:],
            inertia[:, ids], self._dynamics_gravity)
            for side, ids in enumerate(subtree_ids)]
        return torch.stack(hands, 1), palm_pose, rotation

    def _publish_hand_wrench(self, measured, rotation, finite):
        """Numerical validity, uncapped interval peaks, and reset warmup only.

        These flags do not assert dynamically calibrated measurement accuracy.
        Ground-report validity deliberately cannot invalidate a finite hand load.
        """
        self._dynamics_steps += 1
        warm = self._dynamics_steps >= 2
        finite = finite & torch.isfinite(measured).all(dim=(1, 2))
        valid = warm & finite
        self.dynamics_fault |= warm & ~finite
        self.dynamics_wrench_valid.copy_(valid)
        self._hand_wrench.copy_(torch.where(valid[:, None, None], measured, 0.))
        force_palm = (rotation.transpose(-1, -2) @ self._hand_wrench[..., :3, None]).squeeze(-1)
        self._hand_force_palm.copy_(torch.where(valid[:, None, None], force_palm, 0.))
        self.hand_force_peak.copy_(torch.maximum(self.hand_force_peak, self._hand_wrench[..., :3].norm(dim=-1)))
        self.hand_moment_peak.copy_(torch.maximum(self.hand_moment_peak, self._hand_wrench[..., 3:].norm(dim=-1)))
        termination = getattr(getattr(self, 'task_cfg', None), 'termination', None)
        limit = float(getattr(termination, 'blowup_force_n', float('inf')))
        self.dynamics_fault |= (self.hand_force_peak > limit).any(-1)
        return valid, warm

    def _sample_training_dynamics(self):
        """200 Hz local hand correction, without centroidal/contact audit work.

        Isaac may fetch an articulation-wide state tensor; only distal hand
        entries are rotated/reduced. Physical integration, solver settings and
        constraints are unchanged. State checks still detect unrelated NaNs.
        """
        mass, position, velocity, inertia = self._body_dynamics(self.robot, self._hand_body_ids)
        acceleration = self.robot.data.body_com_acc_w.torch[:, self._hand_body_ids]
        measured, _, rotation = self._measure_hand_wrench(
            mass, position, velocity, inertia, acceleration, self._hand_local_ids)
        # The estimator's output is the validity gate.  Re-reading every
        # articulation root/joint tensor and reducing it with ``isfinite().all``
        # at every 200 Hz substep was a major host/device synchronization point.
        # Keep that broad numerical audit sparse in the real training env.  The
        # CPU test fixture has no interval attribute and intentionally retains
        # the old fail-fast check.
        interval = self.__dict__.get('_finite_check_interval_steps', None)
        counter = self.__dict__.get('common_step_counter', 0)
        if interval is None:
            check_state = True
        else:
            interval, counter = int(interval), int(counter)
            check_state = interval > 0 and counter % interval == 0
        finite = self._all_dynamics_valid
        if check_state:
            finite = ((mass > 0).all(-1) & torch.isfinite(mass).all(-1)
                      & torch.isfinite(position).all(dim=(1, 2))
                      & torch.isfinite(velocity).all(dim=(1, 2))
                      & torch.isfinite(inertia).all(dim=(1, 2, 3))
                      & torch.isfinite(acceleration).all(dim=(1, 2)))
            for asset in (self.robot, self.cart):
                if asset is not None:
                    for value in (asset.data.root_link_pose_w.torch, asset.data.root_com_vel_w.torch,
                                  asset.data.joint_pos.torch, asset.data.joint_vel.torch):
                        finite &= torch.isfinite(value).reshape(self.num_envs, -1).all(-1)
            self._all_dynamics_valid.copy_(finite)
        valid, _ = self._publish_hand_wrench(measured, rotation, finite)
        self.physics_valid_duration.add_(valid.to(mass.dtype), alpha=float(self.cfg.sim.dt))
        # No ground sample has been taken. Zero buffers are not measurements.
        self.ground_wrench_valid.zero_()

    def _sample_dynamics(self, substep):
        """Read aligned measurements once immediately after each physical step."""
        if not self.full_dynamics:
            self._sample_training_dynamics()
            return
        mass, position, velocity, inertia = self._body_dynamics(self.robot)
        robot = centroidal_quantities(mass, position, velocity[..., :3], velocity[..., 3:], inertia)
        self.robot_centroidal = robot
        acceleration = self.robot.data.body_com_acc_w.torch
        measured, palm_pose, rotation = self._measure_hand_wrench(
            mass, position, velocity, inertia, acceleration, self._hand_subtree_ids)
        finite = (torch.isfinite(measured).all(dim=(1, 2)) & torch.isfinite(robot['com']).all(-1)
                  & torch.isfinite(robot['linear_momentum']).all(-1)
                  & torch.isfinite(robot['angular_momentum']).all(-1) & (robot['mass'] > 0))
        valid, warm = self._publish_hand_wrench(measured, rotation, finite)
        ground_valid = self._dynamics_steps >= 2
        for i, view in enumerate(self._ground_contact_views[:2]):
            measured, good = self._contact_wrench(view, robot['com'])
            self.dynamics_fault |= warm & ~torch.isfinite(measured).all(-1)
            self._feet_wrench[:, i] = measured
            ground_valid &= good
        cart = self.centroidal_state(self.cart) if self.cart is not None else None
        self.cart_centroidal = cart
        if cart is not None:
            cart_finite = (torch.isfinite(cart['com']).all(-1) & torch.isfinite(cart['linear_momentum']).all(-1)
                           & torch.isfinite(cart['angular_momentum']).all(-1) & (cart['mass'] > 0))
            self.dynamics_fault |= warm & ~cart_finite
            ground_valid &= cart_finite
            for i, view in enumerate(self._ground_contact_views[2:]):
                measured, good = self._contact_wrench(view, cart['com'])
                self.dynamics_fault |= warm & ~torch.isfinite(measured).all(-1)
                self._wheel_wrench[:, i] = measured
                ground_valid &= good
        self.ground_wrench_valid.copy_(ground_valid)
        self._feet_wrench.masked_fill_(~ground_valid[:, None, None], 0.)
        self._wheel_wrench.masked_fill_(~ground_valid[:, None, None], 0.)
        for key, value in (('hand_wrench', self._hand_wrench), ('feet_wrench', self._feet_wrench),
                           ('wheel_wrench', self._wheel_wrench), ('valid', valid), ('ground_valid', ground_valid),
                           ('fault', self.dynamics_fault),
                           ('hand_reference', palm_pose[..., :3]),
                           ('robot_com', robot['com']),
                           ('robot_momentum', torch.cat((robot['linear_momentum'], robot['angular_momentum']), -1))):
            self.physics_samples[key][substep].copy_(value)
        if cart is not None:
            self.physics_samples['cart_com'][substep].copy_(cart['com'])
            self.physics_samples['cart_momentum'][substep].copy_(torch.cat((cart['linear_momentum'], cart['angular_momentum']), -1))
            self.physics_samples['cart_hand_wrench'][substep].copy_(shift_wrench(
                self._hand_wrench, palm_pose[..., :3], cart['com'][:, None]).sum(1))
        self.physics_samples['robot_reaction_wrench'][substep].copy_(-shift_wrench(
            self._hand_wrench, palm_pose[..., :3], robot['com'][:, None]).sum(1))
        if hasattr(self, 'cart_applied_wrench'):
            self.physics_samples['cart_applied_wrench'][substep].copy_(self.cart_applied_wrench)
        else:
            self.physics_samples['cart_applied_wrench'][substep].zero_()
        self.physics_samples['cart_applied_reference'][substep].copy_(self._cart_applied_reference)
        if cart is not None:
            self.physics_samples['cart_applied_wrench_com'][substep].copy_(shift_wrench(
                self.physics_samples['cart_applied_wrench'][substep], self._cart_applied_reference, cart['com']))
        robot_ref, cart_ref = self.physics_impulse_reference[:, 0], self.physics_impulse_reference[:, 1]
        for key in self.physics_fixed_origin_impulses:
            wrench = self.physics_samples[key][substep]
            if key in ('feet_wrench', 'wheel_wrench'):
                wrench = wrench.sum(1)
            is_robot = key in ('feet_wrench', 'robot_reaction_wrench')
            source = robot['com'] if is_robot else (cart['com'] if cart is not None else cart_ref)
            if key == 'cart_applied_wrench':
                source = self._cart_applied_reference
            self.physics_fixed_origin_impulses[key].add_(shift_wrench(
                wrench, source, robot_ref if is_robot else cart_ref), alpha=float(self.cfg.sim.dt))
        for key in self.physics_impulses:
            self.physics_impulses[key].add_(self.physics_samples[key][substep], alpha=float(self.cfg.sim.dt))
        self.physics_valid_duration.add_((valid & ground_valid).to(mass.dtype), alpha=float(self.cfg.sim.dt))

    @staticmethod
    def _quat_conj(q):
        return torch.cat((-q[:, :3], q[:, 3:4]), -1)

    # ------------------------------------------------------------------ HOMIE overrides
    def _snapshot_terminal_physics(self, env_ids):
        """Preserve terminal measurements before automatic reset clears buffers.

        Only selected environments are copied, avoiding a full-batch clone.
        The snapshot stays available until the next control step.
        """
        if not self.full_dynamics or len(env_ids) == 0:
            return None
        return {'env_ids': env_ids.clone(),
                'samples': {name: value[:, env_ids].clone() for name, value in self.physics_samples.items()},
                'impulses': {name: value[env_ids].clone() for name, value in self.physics_impulses.items()},
                'fixed_origin_impulses': {name: value[env_ids].clone() for name, value in self.physics_fixed_origin_impulses.items()},
                'impulse_reference': self.physics_impulse_reference[env_ids].clone(),
                'valid_duration': self.physics_valid_duration[env_ids].clone()}

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_dynamics(env_ids)

    def step(self, actions):
        """HOMIE stepping, with V5 measurements at the physics rate.

        Kept local to V5 so importing/running legacy V4 keeps its exact stepping
        implementation. Public return values and reset ordering remain HOMIE's.
        """
        from homie_isaaclab.math_utils import quat_rotate_inverse
        cfg = self.cfg
        self.terminal_physics = None
        self.hand_force_peak.zero_()
        self.hand_moment_peak.zero_()
        for value in self.physics_impulses.values():
            value.zero_()
        for value in self.physics_fixed_origin_impulses.values():
            value.zero_()
        if self.full_dynamics:
            self.physics_impulse_reference[:, 0].copy_(self.centroidal_state(self.robot)['com'])
            if self.cart is not None:
                self.physics_impulse_reference[:, 1].copy_(self.cart_com_world())
        self.physics_valid_duration.zero_()
        self._pre_physics_step(actions)
        for i in range(cfg.control.decimation):
            self._apply_action()
            if self.full_dynamics and self.cart is not None:
                self._cart_applied_reference.copy_(self.cart_com_world())
            self.scene.write_data_to_sim()
            self.sim.step(render=not self.headless and i == cfg.control.decimation - 1)
            self.scene.update(cfg.sim.dt)
            # The high-level policy, rewards, and interaction target are
            # evaluated once per control step.  In training mode the dynamics
            # label is therefore sampled from the final, fully integrated
            # physics substep.  Sampling the same label on all four substeps
            # performs four PhysX tensor fetches and four Newton--Euler
            # reductions per control step, while the intermediate values are
            # discarded before the actor/critic consumes them.  Keep the
            # original per-substep path for ``full_dynamics`` because it
            # integrates impulses over every physical interval.
            if self.full_dynamics or i == cfg.control.decimation - 1:
                self._sample_dynamics(i)
            self.dof_pos = self.robot.data.joint_pos.torch[:, self.joint_ids]
            self.dof_vel = self.robot.data.joint_vel.torch[:, self.joint_ids]
        self.torques = self.robot.data.applied_torque.torch[:, self.joint_ids].clone()
        self._sync_states()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        contact = self.contact_forces[:, self.feet_indices].norm(dim=-1) > 1.
        self.contact_filt = contact | self.last_contacts
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) & self.contact_filt
        self.feet_air_time += self.dt
        self.feet_max_height = torch.maximum(self.feet_max_height, self._get_feet_heights()[0])
        self._post_physics_step_callback()
        terminated, truncated = self._get_dones()
        self._get_rewards()
        ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_obs = self.compute_termination_observations(ids)
        self.terminal_physics = self._snapshot_terminal_physics(ids)
        self.extras = {'time_outs': truncated.clone(), 'terminated': terminated.clone()}
        if self.terminal_physics is not None:
            self.extras['terminal_physics'] = self.terminal_physics
        self.reset_idx(ids)
        self._get_observations()
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.last_root_vel.copy_(self.root_states[:, 7:13])
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt
        clip = cfg.normalization.clip_observations
        self.obs_buf.clamp_(-clip, clip)
        self.privileged_obs_buf.clamp_(-clip, clip)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, ids, terminal_obs

    def _resample_commands(self, env_ids):
        # Commands come from outside (gamepad or the high-level policy); nothing is sampled.
        self.commands[env_ids, :3] = self.command_buffer[env_ids]
        self.commands[env_ids, 3] = 0.
        self.commands[env_ids, 4] = self.cfg.rewards.base_height_target

    def _pre_physics_step(self, actions):
        cfg = self.cfg
        lower = self.num_lower_dof
        self.current_upper_actions = (self.upper_hold_actions + self.upper_delta).clamp(
            self.action_min[:, lower:], self.action_max[:, lower:])
        self.delta_upper_actions.zero_()
        self.actions = torch.cat((actions.to(self.device), self.current_upper_actions), -1).clamp(
            -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        self.origin_actions = self.actions.clone()
        if cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(*cfg.domain_rand.joint_injection_range, self.actions.shape,
                                                    self.device) * self.torque_limits
        self._delay_steps = torch.randint(cfg.control.decimation, (self.num_envs, 1), device=self.device)
        self._physics_substep = 0

    def _reset_dofs(self, env_ids):
        pos = self.preset_dof_pos.repeat(len(env_ids), 1)
        cfg = self.cfg.domain_rand
        if cfg.randomize_initial_joint_pos:
            lower = self.num_lower_dof
            pos[:, :lower] *= torch_rand_float(*cfg.initial_joint_pos_scale, pos[:, :lower].shape, self.device)
            pos[:, :lower] += torch_rand_float(*cfg.initial_joint_pos_offset, pos[:, :lower].shape, self.device)
        pos = pos.clamp(self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        self.robot.write_joint_position_to_sim_index(position=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_effort_target_index(target=torch.zeros_like(pos), joint_ids=self.joint_ids, env_ids=env_ids)

    def _reset_root_states(self, env_ids):
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        if self.cart is not None:
            self._reset_cart(env_ids)

    def _reset_cart(self, env_ids):
        pose = self.cart.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.env_origins[env_ids]
        self.cart.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.cart.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        n = self.cart.num_joints
        if n:
            ids = torch.arange(n, device=self.device, dtype=torch.int32)
            zeros = torch.zeros(len(env_ids), n, device=self.device)
            self.cart.write_joint_position_to_sim_index(position=zeros, joint_ids=ids, env_ids=env_ids)
            self.cart.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=ids, env_ids=env_ids)
