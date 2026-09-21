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


class PushCartSimEnv(HomieIsaacLabEnv):
    def __init__(self, cfg, project_root: Path, device="cuda:0", headless=True, *, with_cart=True,
                 pose=None, cart_params=None, sim_context=None, allow_robot_cart_contact=False):
        self.with_cart = with_cart
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
            setattr(scene_cfg, f'contact_{i}', ContactSensorCfg(prim_path=robot_prim + '/Geometry/' + '/'.join(chain),
                                                                update_period=0.))
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
        """(pos_w [N,3], quat_xyzw [N,4], lin_vel_w [N,3], ang_vel_w [N,3]) or None without a cart."""
        if self.cart is None:
            return None
        data = self.cart.data
        pose = data.root_link_pose_w.torch
        vel = data.root_com_vel_w.torch
        return pose[:, :3], pose[:, 3:7], vel[:, :3], vel[:, 3:6]

    def grasp_wrench(self):
        """Legacy gravity-compensated proxy for each hand's cart force / torque.

        PhysX reports, per link, the spatial force the parent transmits through the incoming articulation
        joint, in its joint frame. This conversion assumes that frame/origin coincide with the palm's.
        Neglecting hand-subtree acceleration and other external loads, the quasi-static estimate is
            F_hand->cart = F_incoming + m_hand * g          (world frame, after rotating F_incoming)
        Returns (force_w [N,2,3], torque_w [N,2,3], force_palm [N,2,3]); torque is about the palm origin
        and omits the gravity moment of the finger subtree and dynamic inertia compensation. Without
        a cart it should be near zero at rest, not necessarily during hand motion. V3.1 labels this
        as a proxy; MuJoCo's independently recovered constraint wrench has a different measurement source.
        """
        import warp as wp
        from homie_isaaclab.math_utils import quat_apply
        wrench = wp.to_torch(self.robot.root_view.get_link_incoming_joint_force())
        wrench = wrench.view(self.num_envs, self.robot.num_bodies, 6)[:, self.palm_body_ids]
        palm_quat = self.rigid_body_states[:, self.palm_body_ids, 3:7]
        force_w = torch.stack([quat_apply(palm_quat[:, i], wrench[:, i, :3]) for i in range(2)], 1)
        torque_w = torch.stack([quat_apply(palm_quat[:, i], wrench[:, i, 3:]) for i in range(2)], 1)
        gravity = self.hand_subtree_mass.view(1, 2, 1) * torch.tensor([0., 0., -9.81], device=self.device).view(1, 1, 3)
        force_w = force_w + gravity
        force_palm = torch.stack([quat_apply(self._quat_conj(palm_quat[:, i]), force_w[:, i]) for i in range(2)], 1)
        return force_w, torque_w, force_palm

    def cart_com_world(self):
        """Mass-weighted COM of the entire cart articulation, including payload and casters."""
        masses = self.cart.data.body_mass.torch
        positions = self.cart.data.body_com_pose_w.torch[..., :3]
        return (positions * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(dim=1, keepdim=True)

    def cart_hand_wrench(self, force_w, torque_w):
        """Aggregate the existing palm-referenced load proxy about the cart assembly COM.

        This retains grasp_wrench's gravity-only compensation and frame assumptions; it is NOT
        an independently measured constraint wrench. Raw inputs are used before reward clipping.
        """
        from .cart_quality import resultant_wrench
        com = self.cart_com_world()
        points = self.rigid_body_states[:, self.palm_body_ids, :3]
        force, moment = resultant_wrench(force_w, torque_w, points, com)
        return force, moment, com

    @staticmethod
    def _quat_conj(q):
        return torch.cat((-q[:, :3], q[:, 3:4]), -1)

    # ------------------------------------------------------------------ HOMIE overrides
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
