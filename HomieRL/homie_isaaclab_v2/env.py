"""Native Isaac Lab 3 physics adapter for the original HOMIE G1 task.

Lab 3 uses xyzw quaternions and ProxyArray data (.torch). Joint tensors are
explicitly reordered to the URDF / original HIM policy order on every boundary.
"""
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, CollisionPropertiesCfg, RigidBodyPropertiesCfg

from .math_utils import euler_from_quaternion, quat_apply, quat_mul, quat_rotate_inverse, torch_rand_float
from .task_core import HomieTaskCore


@configclass
class HomieSceneCfg(InteractiveSceneCfg):
    pass


class HomieIsaacLabEnv(HomieTaskCore):
    def __init__(self, cfg, project_root: Path, device='cuda:0', headless=True, *, sim_context=None):
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
        if cfg.balance.enabled:
            for name, scale in cfg.balance.reward_scales.items():
                setattr(cfg.rewards.scales, name, scale)
            cfg.domain_rand.max_push_vel_xy = cfg.balance.max_push_vel_xy
            cfg.domain_rand.init_upper_ratio = cfg.balance.upper_ratio
        self._parse_cfg(cfg)
        self.max_episode_length = int(self.max_episode_length)
        self.custom_origins = False

        urdf = project_root / 'HomieRL/legged_gym/resources/robots/g1_description/g1.urdf'
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
        # Official migration guide: set actor and solver parameters explicitly;
        # PhysX/Isaac Sim defaults are not the Isaac Gym asset defaults.
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
        # A registered DirectRLEnv supplies its own context, created by the
        # official base class from the resolved CLI/Hydra simulation config.
        self.sim = sim_context if sim_context is not None else sim_utils.SimulationContext(sim_cfg)
        scene_cfg = HomieSceneCfg(num_envs=self.num_envs, env_spacing=cfg.env.env_spacing,
                                  replicate_physics=True, filter_collisions=True)
        # Local static ground avoids an unnecessary online asset download.
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
        scene_cfg.robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot',
            articulation_root_prim_path='/Geometry/pelvis',
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(urdf), usd_dir=str(project_root / '.cache/g1_isaaclab_v2'),
                usd_file_name='g1.usd', fix_base=False, merge_fixed_joints=False,
                activate_contact_sensors=True,
                collision_props=CollisionPropertiesCfg(
                    contact_offset=cfg.sim.physx.contact_offset, rest_offset=cfg.sim.physx.rest_offset),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=cfg.asset.disable_gravity,
                    linear_damping=cfg.asset.linear_damping, angular_damping=cfg.asset.angular_damping,
                    max_linear_velocity=cfg.asset.max_linear_velocity,
                    # Isaac Gym reads rad/s; this schema writes deg/s (1000 raw = a 17.45 rad/s link clamp).
                    max_angular_velocity=math.degrees(cfg.asset.max_angular_velocity),
                    max_depenetration_velocity=cfg.sim.physx.max_depenetration_velocity),
                articulation_props=ArticulationRootPropertiesCfg(
                    enabled_self_collisions=not bool(cfg.asset.self_collisions),
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(cfg.init_state.pos), rot=tuple(cfg.init_state.rot),
                joint_pos={n: cfg.init_state.default_joint_angles[n] for n in self.dof_names}, joint_vel={'.*': 0.}),
            actuators={
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
            },
        )
        # Sim 6 imports a nested link hierarchy. One sensor per explicit body
        # avoids matching only the top-level pelvis (or an instanced mesh).
        parents = {j.find('child').get('link'): j.find('parent').get('link') for j in xml.findall('joint')}
        self._contact_names = ['left_ankle_roll_link', 'right_ankle_roll_link', 'torso_link']
        for i, name in enumerate(self._contact_names):
            chain = [name]
            while chain[0] in parents:
                chain.insert(0, parents[chain[0]])
            path = '{ENV_REGEX_NS}/Robot/Geometry/' + '/'.join(chain)
            setattr(scene_cfg, f'contact_{i}', ContactSensorCfg(prim_path=path, update_period=0.))
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene['robot']
        self.contacts = [self.scene[f'contact_{i}'] for i in range(len(self._contact_names))]
        # The generic recursive helper stops at the first rigid parent in this
        # nested Sim 6 hierarchy. Author the reporting API on the required links.
        from pxr import PhysxSchema, Usd, UsdPhysics
        # Apply link properties individually because nested rigid links stop the
        # generic recursive writer after its first successful (pelvis) match.
        source_root = self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        body_paths = [str(p.GetPath()) for p in Usd.PrimRange(source_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for path in body_paths:
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim.IsInstance():
                prim.SetInstanceable(False)
            sim_utils.modify_rigid_body_properties(path, scene_cfg.robot.spawn.rigid_props, stage=self.sim.stage)
        # Collision meshes are instance proxies in the new URDF importer. Make
        # only their collision instances editable before authoring offsets.
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
            sim_utils.modify_collision_properties(path, scene_cfg.robot.spawn.collision_props, stage=self.sim.stage)
        for sensor in self.contacts:
            for env_id in range(self.num_envs):
                path = sensor.cfg.prim_path.replace('env_.*', f'env_{env_id}')
                prim = self.sim.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise ValueError(f'Contact link missing from imported USD: {path}')
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self.sim.reset()
        self.scene.update(cfg.sim.dt)
        self.env_origins = self.scene.env_origins
        self.sim.set_camera_view([3., 3., 2.], [0., 0., .7])
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
        self._sync_states()
        self._init_task_buffers()
        self._prepare_reward_function()
        self._init_balance(project_root)
        self.joint_pos_target = self.default_dof_pos.repeat(self.num_envs, 1)
        self._randomize_body_properties()
        self._material_buckets = torch.empty(64, 3)
        self._material_buckets[:, :2] = torch_rand_float(*cfg.domain_rand.friction_range, (64, 1), 'cpu')
        self._material_buckets[:, 2] = torch_rand_float(*cfg.domain_rand.restitution_range, (64,), 'cpu')
        print(f'[HOMIE] {self.num_envs} envs, {len(self.dof_names)} joints, {self.num_bodies} bodies; actor=456 critic=79 action=12')
        print(f'[HOMIE] Lab joint IDs in original policy order: {self.joint_ids.tolist()}')
        self.physics_report = self._physics_report()

    def _init_balance(self, project_root):
        """Stage-2 grasp stance and walk-to-stop switch states for the balance model."""
        self._bank = None
        b = self.cfg.balance
        if not b.enabled:
            return
        import yaml
        robot = yaml.safe_load((project_root / b.stance_path).read_text(encoding='utf-8'))['robot']
        self._stance_joint_pos = torch.tensor([robot['joint_pos'][n] for n in self.dof_names], device=self.device)
        self._stance_root_pose = torch.tensor([*robot['base_pos'], *robot['base_quat_xyzw']], device=self.device)
        # Arm hold target in action units (joint target = default + action_scale * action).
        self._grasp_upper_actions = ((self._stance_joint_pos - self.default_dof_pos[0])
                                     / self.cfg.control.action_scale)[self.num_lower_dof:]
        if b.bank_ratio > 0:
            path = project_root / b.bank_path
            if not path.is_file():
                raise FileNotFoundError(f'Balance switch bank missing: {path}. Record it with: run_homie_v2.ps1 bank')
            self._bank = {k: v.to(self.device) for k, v in torch.load(path, map_location='cpu')['states'].items()}
            print(f'[HOMIE] balance: {len(self._bank["obs"])} switch states from {path}', flush=True)

    def _balance_reset_states(self, env_ids):
        """Write the start state: a recorded walk-to-stop switch (bank_ratio) or the stage-2 grasp stance."""
        b, n, ids = self.cfg.balance, len(env_ids), env_ids.long()
        noise = torch_rand_float(-b.stance_joint_noise, b.stance_joint_noise, (n, self.num_dof), self.device)
        pos = (self._stance_joint_pos + noise).clamp(self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        vel = torch.zeros_like(pos)
        pose = self._stance_root_pose.repeat(n, 1)
        root_vel = torch_rand_float(-b.stance_root_vel, b.stance_root_vel, (n, 6), self.device)
        use_bank = torch.zeros(n, dtype=torch.bool, device=self.device)
        index = torch.zeros(n, dtype=torch.long, device=self.device)
        if self._bank is not None:
            use_bank = torch.rand(n, device=self.device) < b.bank_ratio
            index = torch.randint(len(self._bank['obs']), (n,), device=self.device)
            k = index[use_bank]
            pos[use_bank], vel[use_bank] = self._bank['joint_pos'][k], self._bank['joint_vel'][k]
            pose[use_bank], root_vel[use_bank] = self._bank['root_pose'][k], self._bank['root_vel'][k]
        pose[:, :3] += self.env_origins[ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=vel, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        return use_bank, index

    def _balance_reset_buffers(self, env_ids, use_bank, index):
        """Policy-side state at the start: arms holding the grasp, or the walking policy's buffers at the switch."""
        ids = env_ids.long()
        stance = ids[~use_bank]
        for buf in (self.actions, self.last_actions, self.last_last_actions):
            buf[stance, self.num_lower_dof:] = self._grasp_upper_actions
        self.current_upper_actions[stance] = self._grasp_upper_actions
        self.delta_upper_actions[stance] = 0.
        switched, k = ids[use_bank], index[use_bank]
        if not len(switched):
            return
        # The step after reset rolls actions into last_actions and last_actions into last_last_actions.
        self.obs_buf[switched] = self._bank['obs'][k]
        self.privileged_obs_buf[switched] = self._bank['critic_obs'][k]
        self.actions[switched] = self._bank['actions'][k]
        self.last_actions[switched] = self._bank['last_actions'][k]
        self.current_upper_actions[switched] = self._bank['current_upper_actions'][k]
        self.delta_upper_actions[switched] = self._bank['delta_upper_actions'][k]

    def _reward_stance_height(self):
        # Absolute pelvis height of the stage-2 grasp stance (flat ground), on a 5 mm scale. The
        # feet-relative tracking_base_height puts a flat-footed stance 4 mm lower: the walking model's
        # 0.759 m came with its stepping ankles at 0.0385 m, flat feet put them at 0.0353 m.
        return torch.exp(-torch.square((self.root_states[:, 2] - self._stance_root_pose[2]) / .005))

    def _reward_com_centering(self):
        # Squared horizontal distance of the whole-body CoM from the middle of the two foot centres
        # (mean of each foot's front/mid/hind sole markers): the stability margin of standing.
        com = (self.robot.data.body_com_pos_w.torch * self._body_mass[..., None]).sum(1) / self._body_mass.sum(1, keepdim=True)
        support = torch.stack((self.rigid_body_states[:, self.left_foot_indices, :2].mean(1),
                               self.rigid_body_states[:, self.right_foot_indices, :2].mean(1)), 1).mean(1)
        return torch.sum(torch.square(com[:, :2] - support), dim=1)

    def _reward_pitch_band(self):
        # Pelvis pitch outside the walking model's standing pitch +- band (linear in the excess): the
        # switch from walking must not rotate the pelvis; inside the band the posture is free.
        b = self.cfg.balance
        return (torch.abs(self.pitch[:, 0] - b.pitch_ref) - b.pitch_band).clamp(min=0.)

    def _reward_load_balance(self):
        # Normal-force imbalance |F_l - F_r| / (F_l + F_r) while both feet carry load; unloading one
        # foot is how a step starts.
        force = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.)
        both = (force > 1.).all(dim=1)
        return (force[:, 0] - force[:, 1]).abs() / (force.sum(1) + 1e-6) * both

    def _physics_report(self):
        """Capture effective physics properties as recommended by the migration guide."""
        from pxr import Usd, UsdPhysics
        root = self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
        shapes, bodies = [], []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append({'path': str(prim.GetPath()),
                    'linear_damping': prim.GetAttribute('physxRigidBody:linearDamping').Get(),
                    'angular_damping': prim.GetAttribute('physxRigidBody:angularDamping').Get(),
                    'max_angular_velocity_deg_s': prim.GetAttribute('physxRigidBody:maxAngularVelocity').Get()})
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                shapes.append({'path': str(prim.GetPath()),
                    'contact_offset': prim.GetAttribute('physxCollision:contactOffset').Get(),
                    'rest_offset': prim.GetAttribute('physxCollision:restOffset').Get()})
        return {
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
            'balance': {'enabled': self.cfg.balance.enabled,
                        'switch_states': 0 if self._bank is None else len(self._bank['obs']),
                        'max_push_vel_xy': self.cfg.domain_rand.max_push_vel_xy,
                        'init_upper_ratio': self.action_curriculum_ratio,
                        'reward_scales_per_s': {k: v / self.dt for k, v in self.reward_scales.items()}},
        }

    def _init_indices(self):
        def bodies(names):
            return torch.tensor([self.body_names.index(n) for n in names], device=self.device, dtype=torch.long)
        self.feet_indices = bodies(['left_ankle_roll_link', 'right_ankle_roll_link'])
        self.left_foot_indices = bodies([n for n in self.body_names if 'left_foot' in n])
        self.right_foot_indices = bodies([n for n in self.body_names if 'right_foot' in n])
        if len(self.left_foot_indices) < 3 or len(self.right_foot_indices) < 3:
            raise ValueError('Original foot reference links are required; do not merge fixed joints.')
        self.knee_indices = bodies(self.cfg.asset.knee_names)
        self.termination_contact_indices = bodies([n for n in self.body_names if any(k in n for k in self.cfg.asset.terminate_after_contacts_on)])
        self.penalised_contact_indices = bodies([n for n in self.body_names if any(k in n for k in self.cfg.asset.penalize_contacts_on)])
        for key in ['left_leg', 'right_leg', 'left_hip', 'right_hip', 'hip_pitch', 'ankle', 'knee']:
            names = getattr(self.cfg.asset, key + '_joints')
            setattr(self, key + '_joint_indices', torch.tensor([self.dof_names.index(n) for n in names], device=self.device))
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
        self.upper_body_index = self.torso_body_index = self.body_names.index('torso_link')
        self.base_body_index = self.body_names.index('pelvis')
        self.imu_index = self.body_names.index(self.cfg.asset.imu_link)

    def _init_virtual_links(self, xml):
        # Isaac Sim 6 imports massless URDF markers as Xforms. Reconstruct their
        # kinematics instead of adding artificial mass or dropping foot rewards.
        self._virtual_links = []
        fixed = {j.find('child').get('link'): j for j in xml.findall('joint') if j.get('type') == 'fixed'}

        def add(name):
            if name in self.body_names:
                return
            joint = fixed[name]
            parent = joint.find('parent').get('link')
            add(parent)
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            r, p, y = [float(x) / 2 for x in origin.get('rpy', '0 0 0').split()]
            cr, cp, cy, sr, sp, sy = np.cos(r), np.cos(p), np.cos(y), np.sin(r), np.sin(p), np.sin(y)
            quat = [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
            self._virtual_links.append((self.body_names.index(parent),
                torch.tensor(xyz, device=self.device, dtype=torch.float32).expand(self.num_envs, -1),
                torch.tensor(quat, device=self.device, dtype=torch.float32).expand(self.num_envs, -1)))
            self.body_names.append(name)

        for name in [self.cfg.asset.imu_link] + [n for n in fixed if 'left_foot' in n or 'right_foot' in n]:
            add(name)

    def _sync_states(self):
        data = self.robot.data
        self.dof_pos = data.joint_pos.torch[:, self.joint_ids]
        self.dof_vel = data.joint_vel.torch[:, self.joint_ids]
        self.root_states = torch.cat((data.root_link_pose_w.torch, data.root_com_vel_w.torch), -1)
        physical = torch.cat((data.body_link_pose_w.torch, data.body_com_vel_w.torch), -1)
        self.rigid_body_states = torch.zeros(self.num_envs, self.num_bodies, 13, device=self.device)
        self.rigid_body_states[:, :self.robot.num_bodies] = physical
        link_vel = data.body_link_vel_w.torch
        for i, (parent, offset, rotation) in enumerate(self._virtual_links):
            state = self.rigid_body_states[:, parent]
            world_offset = quat_apply(state[:, 3:7], offset)
            velocity = link_vel[:, parent] if parent < self.robot.num_bodies else state[:, 7:13]
            virtual = self.rigid_body_states[:, self.robot.num_bodies + i]
            virtual[:, :3] = state[:, :3] + world_offset
            virtual[:, 3:7] = quat_mul(state[:, 3:7], rotation)
            virtual[:, 7:10] = velocity[:, :3] + torch.cross(velocity[:, 3:], world_offset, dim=-1)
            virtual[:, 10:13] = velocity[:, 3:]
        self.base_quat = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.contact_forces[:, self.contact_body_ids] = torch.cat([sensor.data.net_forces_w.torch for sensor in self.contacts], dim=1)

    def _randomize_body_properties(self):
        cfg = self.cfg.domain_rand
        masses = self.robot.data.body_mass.torch.clone()
        original = masses.clone()
        if cfg.randomize_link_mass:
            # Official HOMIE rescales every link except the floating base (pelvis, body 0) from its default mass.
            scale = torch_rand_float(*cfg.link_mass_range, masses.shape, self.device)
            scale[:, self.base_body_index] = 1.
            masses *= scale
        if cfg.randomize_payload_mass:
            # Off in v2 (see config): added on top of the rescaled links when enabled.
            masses[:, self.torso_body_index] += self.payload[:, 0]
            for i, side in enumerate(('left', 'right')):
                masses[:, self.body_names.index(side + '_hand_palm_link')] += self.hand_payload[:, i]
        # Negative masses from additive randomization are not valid PhysX inputs.
        masses = masses.clamp_min(.001)
        self.robot.set_masses_index(masses=masses)
        # GPU copy for the whole-body CoM (reading body_mass copies PhysX's CPU masses on every access).
        self._body_mass = masses.to(self.device)
        inertias = self.robot.data.body_inertia.torch.clone()
        scale = masses / original.clamp_min(.001)
        self.robot.set_inertias_index(inertias=inertias * scale.reshape(*scale.shape, *([1] * (inertias.ndim - scale.ndim))))
        if cfg.randomize_body_displacement or cfg.randomize_com_displacement:
            coms = self.robot.data.body_com_pose_b.torch.clone()
            if cfg.randomize_body_displacement:
                coms[:, self.torso_body_index, :3] += self.body_displacement
            if cfg.randomize_com_displacement:
                coms[:, 0, :3] += self.com_displacement
            self.robot.set_coms_index(coms=coms)

    def refresh_actor_rigid_shape_props(self, env_ids):
        cfg = self.cfg.domain_rand
        if not (cfg.randomize_friction or cfg.randomize_restitution):
            return
        ids = env_ids.to(device='cpu', dtype=torch.int32)
        materials = wp.to_torch(self.robot.root_view.get_material_properties())
        samples = self._material_buckets[torch.randint(64, (len(ids), 1))]
        if cfg.randomize_friction:
            materials[ids, :, :2] = samples[..., :2]
        if cfg.randomize_restitution:
            materials[ids, :, 2] = samples[..., 2]
        self.robot.root_view.set_material_properties(wp.from_torch(materials), wp.from_torch(ids))

    def _reset_dofs(self, env_ids):
        pos = self.default_dof_pos.repeat(len(env_ids), 1)
        cfg = self.cfg.domain_rand
        if cfg.randomize_initial_joint_pos:
            pos *= torch_rand_float(*cfg.initial_joint_pos_scale, pos.shape, self.device)
            pos += torch_rand_float(*cfg.initial_joint_pos_offset, pos.shape, self.device)
        pos = pos.clamp(self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        self.robot.write_joint_position_to_sim_index(position=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=pos, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_effort_target_index(target=torch.zeros_like(pos), joint_ids=self.joint_ids, env_ids=env_ids)

    def _reset_root_states(self, env_ids):
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        vel = torch_rand_float(-.5, .5, (len(env_ids), 6), self.device) if self.cfg.domain_rand.use_random else torch.zeros(len(env_ids), 6, device=self.device)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    def _push_robots(self):
        vel = self.robot.data.root_com_vel_w.torch.clone()
        limit = self.cfg.domain_rand.max_push_vel_xy
        vel[:, :2] = torch_rand_float(-limit, limit, (self.num_envs, 2), self.device)
        self.robot.write_root_velocity_to_sim_index(root_velocity=vel)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(dtype=torch.int32)
        super().reset_idx(env_ids)
        balance_start = self._balance_reset_states(env_ids) if self.cfg.balance.enabled else None
        self.scene.reset(env_ids)
        self.sim.forward()
        self.scene.update(0.)
        self._sync_states()
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.obs_buf[env_ids] = 0.
        self.privileged_obs_buf[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.feet_max_height[env_ids] = 0.
        if balance_start is not None:
            self._balance_reset_buffers(env_ids, *balance_start)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def _pre_physics_step(self, actions):
        """Process policy actions once, following the official direct workflow."""
        cfg = self.cfg
        upper_randomization = cfg.domain_rand.use_random or getattr(
            cfg.domain_rand, 'upper_body_randomization', False,
        )
        if not upper_randomization:
            if cfg.balance.enabled:
                self.current_upper_actions[:] = self._grasp_upper_actions
            else:
                self.current_upper_actions.zero_()
            self.delta_upper_actions.zero_()
        elif self.common_step_counter % cfg.domain_rand.upper_interval == 0:
            ratio = min(self.action_curriculum_ratio, 1.)
            rate = 20 * (1 - ratio * .99)
            u = torch.rand_like(self.current_upper_actions)
            amplitude = -torch.log(1 - u + u * np.exp(-rate)) / rate
            amplitude *= torch.rand_like(amplitude)
            choose_lower = torch.rand_like(amplitude) >= .5
            target = torch.where(choose_lower, self.action_min[:, 12:], self.action_max[:, 12:]) * amplitude
            if cfg.balance.enabled:
                hold = torch.rand(self.num_envs, 1, device=self.device) < cfg.balance.upper_hold_prob
                target = torch.where(hold, self._grasp_upper_actions, target)
            self.delta_upper_actions = (target - self.current_upper_actions) / cfg.domain_rand.upper_interval
        self.current_upper_actions += self.delta_upper_actions
        self.actions = torch.cat((actions.to(self.device), self.current_upper_actions), -1).clamp(-cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        self.origin_actions = self.actions.clone()
        if cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(*cfg.domain_rand.joint_injection_range, self.actions.shape, self.device) * self.torque_limits
        self._delay_steps = torch.randint(cfg.control.decimation, (self.num_envs, 1), device=self.device)
        self._physics_substep = 0

    def _apply_action(self):
        """Recompute the PD effort before every 200 Hz physics substep."""
        cfg = self.cfg
        applied = torch.where(self._physics_substep >= self._delay_steps, self.actions, self.last_actions) if cfg.domain_rand.delay else self.actions
        self.joint_pos_target = self.default_dof_pos + cfg.control.action_scale * applied
        torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        torques = (torques + self.actuation_offset + self.joint_injection).clamp(-self.torque_limits, self.torque_limits)
        self.robot.set_joint_effort_target_index(target=torques[:, :12], joint_ids=self.joint_ids[:12])
        self.robot.set_joint_position_target_index(target=self.joint_pos_target[:, 12:], joint_ids=self.joint_ids[12:])
        self._physics_substep += 1

    def _get_dones(self):
        self.check_termination()
        return self.terminated_buf, self.time_out_buf

    def _get_rewards(self):
        self.compute_reward()
        return self.rew_buf

    def _get_observations(self):
        self.compute_observations()
        return {'policy': self.obs_buf, 'critic': self.privileged_obs_buf}

    def step(self, actions):
        """Use Lab's step ordering and adapt its results to HIM's seven outputs.

        HIM additionally needs pre-reset terminal critic observations to train
        its estimator. The registered DirectRLEnv transports these in extras.
        """
        cfg = self.cfg
        self._pre_physics_step(actions)
        for i in range(cfg.control.decimation):
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=not self.headless and i == cfg.control.decimation - 1)
            self.scene.update(cfg.sim.dt)
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
        self.extras = {'time_outs': truncated.clone(), 'terminated': terminated.clone()}
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
