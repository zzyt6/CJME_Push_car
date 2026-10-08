"""V7 task: the V6 environment (V3.2 dynamics/control/rewards, projected observations, read-only labels) on
the V7 plant -- a Dex3 grasp on a square handle tube that can slide (load-dependent elasto-plastic friction) and pivot
(elastic up to the finger capacity, yielding beyond it), rigid finger-limited pull / lift, Coulomb cart
resistance, force bumps / slope bias on the cart and caster headings that are random from the first level.
"""
import math

import torch
import warp as wp

from homie_pushcart.task_env import PushCartEnv as V32Env
from isaaclab.actuators import ImplicitActuatorCfg
from homie_isaaclab.math_utils import quat_apply, quat_mul, quat_rotate_inverse, torch_rand_float
from .task_config import num_actor_obs, num_critic_obs, validate_task_schema
from .curriculum import LEVELS, ContinuousCurriculum
from .observation import (
    HIMHistory, him_frame, mask_homie_velocity, without_handle_position, HAND_WRENCH_SCALE,
)
from .him_estimator import make_next_critic_observation
from .hand_labels import ReadOnlyHandLabels
from .assets.build import load_v7_params, load_v7_pose
from .paths import ASSET_REPORT, PACKED_USD, ROTATION_JOINTS, SIDES, SLIDE_JOINTS
from .physics import swivel_torque, validate_assets

GRAVITY = 9.81
ROTATIONS = ("roll", "tilt", "yaw")          # about the tube axis (cart y), cart x, cart z
OVERLOAD_KINDS = ("pull", "lift", "push", "pivot", "slide")


def _uniform(lo_hi, shape, device):
    return torch_rand_float(float(lo_hi[0]), float(lo_hi[1]), shape, device)


class PushCartEnv(V32Env):
    packed_usd_path = PACKED_USD
    asset_report_path = ASSET_REPORT

    def __init__(self, cfg, task_cfg, *args, **kwargs):
        validate_task_schema(task_cfg)
        validate_assets()
        self._transition_active = False
        kwargs.setdefault("pose", load_v7_pose())
        kwargs.setdefault("cart_params", load_v7_params(pose=kwargs["pose"]))
        super().__init__(cfg, task_cfg, *args, **kwargs)
        self._labels = ReadOnlyHandLabels(self)
        self._labels_step = -1
        # HOMIE's per-reward-term finiteness check synchronises the GPU 28 times per step; every 50 steps (1 s)
        # still fails fast on a NaN without stalling every step. Rewards themselves are unchanged.
        self._finite_check_interval_steps = 50
        self.num_hl_obs = num_actor_obs(task_cfg)
        self.num_hl_critic_obs = num_critic_obs(task_cfg)

    # ------------------------------------------------------------------ scene
    def _cart_actuators(self, wheels: dict) -> dict:
        # Casters: viscous drive re-sampled at reset. Slide: implicit PD to a moving anchor whose force limit is the
        # load-dependent friction limit (elasto-plastic friction; target and limit written every control step).
        # Roll / tilt / yaw: implicit PD to zero whose force limit is the finger capacity (elastic, then yield);
        # stiffness, damping and limit are written per environment at reset.
        g = self.cart_params["grasp"]
        k_slide = 0.5 * sum(g["slide_stiffness_range"])
        return {
            "casters": ImplicitActuatorCfg(
                joint_names_expr=["caster_.*_swivel_joint", "wheel_.*_joint"], stiffness=0.,
                damping={"caster_.*_swivel_joint": float(wheels["swivel_damping"]),
                         "wheel_.*_joint": float(wheels["wheel_damping"])}),
            "grasp_slide": ImplicitActuatorCfg(
                joint_names_expr=["grasp_.*_slide_joint"], stiffness=k_slide,
                damping=2. * g["slide_damping_ratio"] * (k_slide * g["slide_damping_mass_kg"]) ** 0.5,
                effort_limit_sim=50., armature=float(g["slide_armature"])),
            "grasp_pivot": ImplicitActuatorCfg(joint_names_expr=["grasp_.*_(roll|tilt|yaw)_joint"], stiffness=190.,
                                               damping=2., effort_limit_sim=4.5,
                                               armature=float(self.cart_params["grasp"]["rotation_armature"])),
        }

    # ------------------------------------------------------------------ set-up
    def _init_task_constants(self):
        super()._init_task_constants()
        dev = self.device

        def ids(names):
            found, resolved = self.cart.find_joints(names, preserve_order=True)
            if resolved != names:
                raise ValueError(f"V7 cart asset lacks {names}; rebuild the V7 assets")
            return torch.tensor(found, device=dev, dtype=torch.long)

        self.slide_ids = ids([SLIDE_JOINTS[s] for s in SIDES])
        self.rot_ids = {r: ids([ROTATION_JOINTS[r][s] for s in SIDES]) for r in ROTATIONS}
        self.rot_all_ids = torch.cat([self.rot_ids[r] for r in ROTATIONS])                 # roll L R, tilt L R, yaw L R
        self.slide_ids_sim = self.slide_ids.to(torch.int32)
        self.rot_all_ids_sim = self.rot_all_ids.to(torch.int32)
        gp = self.pose["grasp"]["point_in_palm"]
        self.grasp_point_palm = torch.tensor([gp[s] for s in SIDES], device=dev, dtype=torch.float32)
        self.tracking_sigma = torch.tensor(self.task_cfg.rewards.tracking_sigma_axes, device=dev)
        self.success_rmse = torch.tensor(self.task_cfg.curriculum_v7.success_rmse, device=dev)
        self.wheel_radius_t = float(self.cart_params["wheels"]["radius"])
        self._cart_mass = self.cart_default_mass.clone()          # GPU mirror of the cart masses (set at reset)
        self._cart_com_b = self.cart_default_com[..., :3].clone()  # GPU mirror of the cart COM offsets (set at reset)
        self._maxf_cpu = None                                      # CPU mirror of PhysX dof max forces
        self._fric_cpu = None                                      # CPU mirror of PhysX dof friction (static, dynamic, viscous)
        self.caster_swivel_ids = ids([f"caster_{i}_swivel_joint" for i in range(4)])   # paired with ...
        self.caster_wheel_ids = ids([f"wheel_{i}_joint" for i in range(4)])            # ... these wheels
        self._all_env_idx_cpu = torch.arange(self.num_envs, dtype=torch.int32)

    def _init_hl_buffers(self):
        # Keep the original 124x5 history and 652 critic construction internally.
        super()._init_hl_buffers()
        dev, N = self.device, self.num_envs
        z = lambda *shape: torch.zeros(*shape, device=dev)                                # noqa: E731
        self.estimator_history = HIMHistory(N, dev)
        self.estimator_obs = self.estimator_history.buffer.flatten(1)
        self.legacy_actor_obs = self.hl_obs
        self.legacy_critic_obs = self.hl_critic_obs
        self.hand_scale = torch.tensor(HAND_WRENCH_SCALE, device=dev)
        self.hand_target = z(N, 13)
        self.him_next_target = z(N, 81)
        self._pre_reset_hands = self.hand_target.clone()
        self._hand_cart = z(N, 2, 6)                    # Newton-Euler hand loads, cart axes, about the palm origin
        self._hand_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        # V7 per-episode grasp physics, per hand
        self.grip_mu, self.grip_normal, self.slide_dyn_ratio = z(N, 2), z(N, 2), torch.ones(N, 2, device=dev)
        self.slide_static = z(N, 2)                     # current (load dependent) static slide limit
        self.slide_limit_now = z(N, 2)                  # limit in force (x dynamic ratio while sliding)
        self.slide_stiffness = z(N, 2)
        self.slide_anchor, self._anchor_prev = z(N, 2), z(N, 2)       # elasto-plastic friction anchor (plastic slip)
        self.slide_sliding = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.rot_stiffness, self.roll_stiffness = z(N, 2), z(N, 2)        # tilt / yaw, roll
        self.roll_capacity, self.moment_capacity = z(N, 2), z(N, 2)      # physical values (x capacity_mult in sim)
        self.pull_capacity, self.lift_capacity = z(N, 2), z(N, 2)
        self.slide_init = z(N, 2)
        # cart
        self.rolling_coeff, self.swivel_static = z(N), z(N)      # swivel_static: tyre scrub at standstill
        self.swivel_bearing, self.swivel_dyn_ratio = z(N), torch.ones(N, device=dev)
        self.yaw_damping = z(N)                         # V7.1 steering-authority randomization, N.m.s/rad
        self.robot_velocity_target = z(N, 4)            # V7.1 label: pelvis velocity x 2 (body frame), valid
        self.cart_true_mass = torch.full((N,), self.cart_base_kg, device=dev)
        self.bias_force_w = z(N, 3)
        self.bump_force_w, self.bump_point_w = z(N, 3), z(N, 3)
        self.bump_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.bump_wait = torch.zeros(N, dtype=torch.long, device=dev)
        self.grasp_overload_kind = torch.zeros(N, len(OVERLOAD_KINDS), dtype=torch.bool, device=dev)
        self.grasp_kind_counts = torch.zeros(len(OVERLOAD_KINDS), device=dev)       # cumulative, diagnostics
        self.slip_sum, self.drift_max, self.pivot_max = z(N), z(N), z(N)
        self.short_overloads = z(N)                     # over-capacity runs of >= 0.1 s that ended before termination
        self._slide_q_prev, self._rot_q_prev, self._chain_step = z(N, 2), z(N, 3, 2), -1
        self.ts.slide_v, self.ts.rot_v = z(N, 2), z(N, 3, 2)
        self.ts.slip_v = z(N, 2)                        # plastic slip rate (anchor velocity)
        self.ts.internal_cost, self.ts.internal_frac = z(N), z(N)      # read by the logger before the first step
        self.ts.internal_frac_raw = z(N)
        self.ts.internal_share = z(N, 6)                # cost share per component (Fx Fy Fz Mx My Mz), diagnostics
        self._load_f, self._load_f_step = z(N, 2, 6), -1                # low-passed hand loads (grasp point)
        self._load_alpha = min(1., self.dt / max(self.task_cfg.grasp.internal_filter_tau_s, 1e-6))
        # per-environment continuous curriculum; the V3.2 level (--level) only sets the starting difficulty
        self.cur = ContinuousCurriculum(self.task_cfg, N, dev, init_difficulty=self.curriculum.level / LEVELS)
        self.capacity_mult = torch.full((N,), float(self.task_cfg.grasp.capacity_mult_start), device=dev)
        self.disturbance_frac = z(N)
        self.push_scale = z(N)
        self.level_tensor = self.cur.d * LEVELS        # the critic's level slot carries each environment's difficulty

    def _apply_level(self, params):
        """Global (population-mean) parameters for the remaining shared uses; per-env values come from self.cur.
        V3.2 calls this with a discrete level at construction; once the curriculum exists it is the only source."""
        if hasattr(self, "cur"):
            params = self.cur.global_params()
        self.level_params = params
        self.upper_scale = torch.tensor([params.waist_scale] + [params.arm_scale] * 14, device=self.device).unsqueeze(0)
        self.cfg.domain_rand.push_robots = bool(self.task_cfg.randomization.robot_push)     # magnitude per env
        if hasattr(self, "cur"):
            self.level_tensor = self.cur.d * LEVELS

    def _refresh_global_level(self):
        self._apply_level(self.cur.global_params())

    # ------------------------------------------------------------------ reset-time physics
    def _write_slide_limit(self):
        """Per-step slide force limit (drive max force) for every environment. PhysX keeps max forces in its CPU
        model and Isaac Lab's writer copies the whole (N, J) buffer GPU -> CPU on each call; a CPU mirror (initialised
        from PhysX, updated at every reset write) means only the (N, 2) slide values cross the bus."""
        rv = self.cart.root_view
        if self._maxf_cpu is None:
            self._maxf_cpu = wp.to_torch(rv.get_dof_max_forces()).clone()
        self._maxf_cpu[:, self.slide_ids.cpu()] = self.slide_limit_now.cpu()
        rv.set_dof_max_forces(wp.from_torch(self._maxf_cpu), indices=wp.from_torch(self._all_env_idx_cpu))

    def _write_swivel_friction(self):
        """Per-step caster swivel friction from each wheel's rolling speed (physics.swivel_torque). PhysX keeps joint
        friction in its CPU model; a CPU mirror (initialised from PhysX, updated at every reset write) means only
        the (N, 4) swivel values are rewritten, as for the slide limit."""
        cp = self.task_cfg.cart_physics
        rim = self.cart.data.joint_vel.torch[:, self.caster_wheel_ids] * self.wheel_radius_t
        tau = swivel_torque(self.swivel_static[:, None], self.swivel_bearing[:, None], rim, float(cp.swivel_scrub_speed_m_s))
        rv = self.cart.root_view
        if self._fric_cpu is None:
            self._fric_cpu = wp.to_torch(rv.get_dof_friction_properties()).clone()
        cols = self.caster_swivel_ids.cpu()
        tau_c = tau.cpu()
        self._fric_cpu[:, cols, 0] = tau_c
        self._fric_cpu[:, cols, 1] = tau_c * self.swivel_dyn_ratio.cpu()[:, None]
        rv.set_dof_friction_properties(wp.from_torch(self._fric_cpu), indices=wp.from_torch(self._all_env_idx_cpu))
        self.swivel_now = tau

    def _update_slide_anchor(self):
        """Elasto-plastic (reset-integrator) friction, once per control step: the anchor is dragged so that the
        spring force k (anchor - q) never exceeds the limit; while it moves the dynamic limit applies. Between
        updates the drive's max force caps the spring, so the hand slides at the limit within a step."""
        q = self.ts.slide_q
        k = self.slide_stiffness.clamp_min(1.)
        e = self.slide_anchor - q
        reach = self.slide_static / k                               # elastic travel at the static limit
        over = e.abs() > reach
        self._anchor_prev = self.slide_anchor.clone()
        self.slide_anchor = torch.where(over, q + torch.sign(e) * reach, self.slide_anchor)
        self.ts.slip_v = (self.slide_anchor - self._anchor_prev) / self.dt
        self.slide_sliding = over
        self.slide_limit_now = torch.where(over, self.slide_static * self.slide_dyn_ratio, self.slide_static)
        self.cart.set_joint_position_target_index(target=self.slide_anchor, joint_ids=self.slide_ids_sim)

    def _slide_limit(self, load_across):
        g = self.task_cfg.grasp
        load = load_across if g.load_dependent_friction else torch.zeros_like(load_across)
        return (self.grip_mu * (self.grip_normal + load) + g.slide_offset_n).clamp_min(g.slide_min_n)

    def _randomize_cart(self, env_ids):
        """Per-episode physics at each environment's own difficulty (no V3.2 global level is read here)."""
        n, dev, tc = len(env_ids), self.device, self.task_cfg
        g, cp, rz, cur = tc.grasp, tc.cart_physics, tc.randomization, self.cur
        ids = env_ids.to(torch.long)
        # curriculum quantities fixed for the episode
        self.capacity_mult[ids] = cur.capacity_mult(ids)
        self.disturbance_frac[ids] = cur.disturbance(ids)
        self.push_scale[ids] = cur.robot_push(ids)
        # payload (nominal total mass is what the policy is told), its COM offset, and the true base mass
        payload = torch.rand(n, device=dev) * cur.payload_max(ids)
        if self.fixed_payload_kg is not None:
            payload = torch.full((n,), float(self.fixed_payload_kg), device=dev)
        self.cart_mass_total[ids] = self.cart_base_kg + payload
        com_xy = cur.sample((-rz.payload_com_offset_m, rz.payload_com_offset_m), ids, (2,))
        self.payload_com[ids] = com_xy
        true_mass = self.cart_base_kg * cur.sample(cp.base_mass_scale_range, ids) + payload
        self.cart_true_mass[ids] = true_mass
        masses = self.cart_default_mass[ids].clone()
        masses[:, self.cart_base_body] = true_mass
        self.cart.set_masses_index(masses=masses, env_ids=env_ids)
        self._cart_mass[ids] = masses
        inertias = self.cart_default_inertia[ids].clone()
        inertias[:, self.cart_base_body] *= (true_mass / self.cart_base_kg).view(n, *([1] * (inertias.ndim - 2)))
        self.cart.set_inertias_index(inertias=inertias, env_ids=env_ids)
        coms = self.cart_default_com[ids].clone()
        coms[:, self.cart_base_body, :2] += (payload / true_mass).unsqueeze(1) * com_xy
        self.cart.set_coms_index(coms=coms, env_ids=env_ids)
        self._cart_com_b[ids] = coms[..., :3]
        # tyre friction (material)
        mu = cur.sample(cp.wheel_friction_range, ids)
        self.wheel_friction[ids] = mu
        ids_cpu = env_ids.to(device="cpu", dtype=torch.int32)
        materials = wp.to_torch(self.cart.root_view.get_material_properties())
        materials[ids_cpu, :, :2] = mu.cpu().to(materials.dtype).view(n, 1, 1)
        self.cart.root_view.set_material_properties(wp.from_torch(materials), wp.from_torch(ids_cpu))
        # grasp: grip, elastic pivots (stiffness, damping, capacity x curriculum multiplier), finger capacities
        self.grip_mu[ids] = cur.sample(g.friction_coeff_range, ids, (2,))
        self.grip_normal[ids] = cur.sample(g.normal_force_range, ids, (2,))
        self.slide_dyn_ratio[ids] = _uniform(g.dynamic_over_static, (n, 2), dev)
        self.rot_stiffness[ids] = cur.sample(g.rotation_stiffness_range, ids, (2,))
        self.roll_stiffness[ids] = cur.sample(g.roll_stiffness_range, ids, (2,))
        self.roll_capacity[ids] = cur.sample(g.roll_capacity_range, ids, (2,))
        self.moment_capacity[ids] = cur.sample(g.moment_capacity_range, ids, (2,))
        self.pull_capacity[ids] = cur.sample(g.pull_capacity_range, ids, (2,))
        self.lift_capacity[ids] = cur.sample(g.lift_capacity_range, ids, (2,))
        k = torch.cat((self.roll_stiffness[ids], self.rot_stiffness[ids], self.rot_stiffness[ids]), 1)   # roll, tilt, yaw
        d = 2.0 * g.rotation_damping_ratio * torch.sqrt(k * g.rotation_armature)
        mult = self.capacity_mult[ids].unsqueeze(1)
        cap = torch.cat((self.roll_capacity[ids], self.moment_capacity[ids], self.moment_capacity[ids]), 1) * mult
        # slide: anchor spring (pad shear stiffness), limit = unloaded static friction until the first update
        ks = cur.sample(g.slide_stiffness_range, ids, (2,))
        self.slide_stiffness[ids] = ks
        ds = 2.0 * g.slide_damping_ratio * torch.sqrt(ks * g.slide_damping_mass_kg)
        self.slide_static[ids] = (self.grip_mu[ids] * self.grip_normal[ids] + g.slide_offset_n).clamp_min(g.slide_min_n)
        self.slide_limit_now[ids] = self.slide_static[ids]
        joint_ids = torch.cat((self.rot_all_ids_sim, self.slide_ids_sim))
        self.cart.write_joint_stiffness_to_sim_index(stiffness=torch.cat((k, ks), 1), joint_ids=joint_ids,
                                                     env_ids=env_ids)
        limits = torch.cat((cap, self.slide_static[ids]), 1)
        self.cart.write_joint_effort_limit_to_sim_index(limits=limits, joint_ids=joint_ids, env_ids=env_ids)
        if self._maxf_cpu is not None:                     # keep the per-step writer's CPU mirror equal to PhysX
            rows = ids.cpu().view(-1, 1)
            self._maxf_cpu[rows, joint_ids.long().cpu().view(1, -1)] = limits.cpu()
        # damping of every cart joint: small viscous wheel / swivel drag (V7 ranges replace V3.2's), pivot damping
        wd = _uniform(cp.wheel_viscous_range, (n, 1), dev)
        sd = _uniform(cp.swivel_viscous_range, (n, 1), dev)
        self.wheel_damping[ids], self.swivel_damping[ids] = wd[:, 0], sd[:, 0]
        damping = torch.zeros(n, self.cart.num_joints, device=dev)
        damping[:, self.wheel_joint_ids] = wd
        damping[:, self.swivel_joint_ids] = sd
        damping[:, self.rot_all_ids] = d
        damping[:, self.slide_ids] = ds
        self.cart.write_joint_damping_to_sim_index(damping=damping, env_ids=env_ids)
        # Coulomb joint friction (static >= dynamic, PhysX requirement): wheels and swivels; none on the slide
        static = torch.zeros(n, self.cart.num_joints, device=dev)
        dynamic = torch.zeros_like(static)
        c_rr = cur.sample(cp.rolling_coeff_range, ids)
        rolling = c_rr * true_mass * GRAVITY / 4 * self.wheel_radius_t
        dynamic[:, self.wheel_joint_ids] = rolling[:, None]
        static[:, self.wheel_joint_ids] = (rolling * cur.sample(cp.breakaway_over_rolling, ids))[:, None]
        # swivel: tyre scrub at standstill, fading with the wheel's rolling speed to the bearing friction
        # (physics.swivel_torque, rewritten every control step); at reset the casters stand still -> full scrub
        swivel = cur.sample(cp.swivel_friction_range, ids)
        bearing = torch.minimum(_uniform(cp.swivel_bearing_range, (n,), dev), swivel)
        dyn_ratio = _uniform(g.dynamic_over_static, (n,), dev)
        static[:, self.swivel_joint_ids] = swivel[:, None]
        dynamic[:, self.swivel_joint_ids] = (swivel * dyn_ratio)[:, None]
        self.cart.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=static, joint_dynamic_friction_coeff=dynamic,
            joint_viscous_friction_coeff=torch.zeros_like(static), env_ids=env_ids)
        if self._fric_cpu is not None:                     # keep the per-step writer's CPU mirror equal to PhysX
            self._fric_cpu[ids.cpu()] = torch.stack((static, dynamic, torch.zeros_like(static)), -1).cpu()
        self.rolling_coeff[ids], self.swivel_static[ids] = c_rr, swivel
        self.yaw_damping[ids] = cur.sample(cp.yaw_damping_range, ids)
        self.swivel_bearing[ids], self.swivel_dyn_ratio[ids] = bearing, dyn_ratio
        # slope-equivalent bias force for the whole episode, scaled by the environment's disturbance level
        frac = self.disturbance_frac[ids]
        angle = _uniform((0., 2 * math.pi), (n,), dev)
        slope = torch.deg2rad(_uniform((0., tc.disturbance.slope_deg_max), (n,), dev)) * frac
        mag = true_mass * GRAVITY * torch.sin(slope)
        self.bias_force_w[ids] = torch.stack((mag * torch.cos(angle), mag * torch.sin(angle), torch.zeros_like(mag)), -1)
        self.bump_timer[ids] = 0
        self.bump_wait[ids] = (_uniform(tc.disturbance.bump_interval_s, (n,), dev) / self.dt).long()
        self.bump_force_w[ids] = 0.

    def _reset_root_states(self, env_ids):
        """Robot and cart move as one rigid system; the grip sits at a random lateral offset on the bar and
        every caster starts at an independent random heading (V7: from the first level)."""
        n, rz, dev = len(env_ids), self.task_cfg.randomization, self.device
        yaw = torch_rand_float(-math.pi, math.pi, (n, 1), dev) if rz.init_yaw else torch.zeros(n, 1, device=dev)
        xy = torch_rand_float(-rz.init_xy_m, rz.init_xy_m, (n, 2), dev)
        q_yaw = torch.cat((torch.zeros(n, 2, device=dev), torch.sin(yaw / 2), torch.cos(yaw / 2)), -1)
        offset = torch_rand_float(-self.task_cfg.grasp.initial_offset_m, self.task_cfg.grasp.initial_offset_m, (n, 1), dev)
        for asset in (self.robot, self.cart):
            pose = asset.data.default_root_pose.torch[env_ids].clone()
            pos = quat_apply(q_yaw, pose[:, :3])
            pos[:, :2] += xy
            pos += self.env_origins[env_ids]
            quat = quat_mul(q_yaw, pose[:, 3:7])
            if asset is self.cart:
                # hands stay put, the bar shifts under them: slide joints at +offset, cart base at -offset (cart y)
                lateral = quat_apply(quat, torch.tensor([[0., 1., 0.]], device=dev).expand(n, 3))
                pos -= lateral * offset
            asset.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), -1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        joints = torch.zeros(n, self.cart.num_joints, device=dev)        # pivots start seated (0)
        r = self.task_cfg.cart_physics.caster_heading_range_rad
        if self.fixed_caster_angle is not None:
            joints[:, self.swivel_joint_ids] = float(self.fixed_caster_angle)
        elif r > 0:
            joints[:, self.swivel_joint_ids] = torch_rand_float(-r, r, (n, len(self.swivel_joint_ids)), dev)
        joints[:, self.slide_ids] = offset
        self.slide_init[env_ids.to(torch.long)] = offset.expand(n, 2)
        self.slide_anchor[env_ids.to(torch.long)] = offset.expand(n, 2)       # unloaded spring at the reset grip
        self._anchor_prev[env_ids.to(torch.long)] = offset.expand(n, 2)
        self.ts.slip_v[env_ids.to(torch.long)] = 0.
        self.slide_sliding[env_ids.to(torch.long)] = False
        self.cart.set_joint_position_target_index(target=offset.expand(n, 2).contiguous(), joint_ids=self.slide_ids_sim,
                                                  env_ids=env_ids)
        self._slide_q_prev[env_ids.to(torch.long)] = offset.expand(n, 2)      # no spurious velocity after a reset
        self._rot_q_prev[env_ids.to(torch.long)] = 0.
        self.cart.write_joint_position_to_sim_index(position=joints, joint_ids=self.cart_all_joint_ids, env_ids=env_ids)
        self.cart.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), joint_ids=self.cart_all_joint_ids, env_ids=env_ids)

    def cart_com_world(self):
        """V3.2's mass-weighted cart COM, from the GPU mass mirror (Isaac Lab re-reads PhysX's CPU masses on every
        body_mass access: 2 ms at 4096 envs, twice per step)."""
        positions = self._cart_com_positions()
        return (positions * self._cart_mass.unsqueeze(-1)).sum(dim=1) / self._cart_mass.sum(dim=1, keepdim=True)

    def _cart_com_positions(self):
        """World COM of every cart body = link pose (+) mirrored COM offset (body_com_pose_w would re-fetch the
        offsets from the CPU model after every physics step: 3.3 ms at 4096 envs)."""
        from .hand_labels import rotation_matrix_xyzw
        link = self.cart.data.body_link_pose_w.torch
        return link[..., :3] + (rotation_matrix_xyzw(link[..., 3:7]) @ self._cart_com_b.unsqueeze(-1)).squeeze(-1)

    # ------------------------------------------------------------------ disturbances
    def _apply_cart_disturbances(self):
        super()._apply_cart_disturbances()      # wheel stalls (the V3.2 velocity kick is disabled in V7)
        tc, dev = self.task_cfg.disturbance, self.device
        self.bump_wait -= 1
        start = (self.bump_wait <= 0) & (self.bump_timer <= 0) & (self.disturbance_frac > 0)
        if start.any():
            ids = start.nonzero(as_tuple=False).flatten()
            n = len(ids)
            angle = _uniform((0., 2 * math.pi), (n,), dev)
            mag = torch.rand(n, device=dev) * tc.bump_force_max_n * self.disturbance_frac[ids]
            self.bump_force_w[ids] = torch.stack((mag * torch.cos(angle), mag * torch.sin(angle), torch.zeros_like(mag)), -1)
            local = torch.stack((_uniform(tc.bump_point_x_m, (n,), dev), _uniform(tc.bump_point_y_m, (n,), dev),
                                 torch.full((n,), tc.bump_point_z_m, device=dev)), -1)
            pose = self.cart.data.root_link_pose_w.torch[ids]
            self.bump_point_w[ids] = pose[:, :3] + quat_apply(pose[:, 3:7], local)
            self.bump_timer[ids] = (_uniform(tc.bump_duration_s, (n,), dev) / self.dt).long().clamp_min(1)
            self.bump_wait[ids] = (_uniform(tc.bump_interval_s, (n,), dev) / self.dt).long()
        active = (self.bump_timer > 0).float().unsqueeze(1)
        bump = self.bump_force_w * active
        base = torch.tensor([self.cart_base_body], device=dev, dtype=torch.long)
        com = self._cart_com_positions()[:, self.cart_base_body]
        torque = torch.cross(self.bump_point_w - com, bump, dim=-1)          # bump applied off-centre
        # V7.1: cart yaw damping (floor / caster turning resistance), tau_z = -c wz, held over the control step
        # (c dt / I_yaw <= 10 x 0.02 / 1.3 = 0.15: stable as an explicit term)
        torque[:, 2] -= self.yaw_damping * self.cart.data.root_ang_vel_w.torch[:, 2]
        self.cart.permanent_wrench_composer.set_forces_and_torques_index(
            forces=(self.bias_force_w + bump).unsqueeze(1), torques=torque.unsqueeze(1), body_ids=base, is_global=True)
        self.bump_timer = (self.bump_timer - 1).clamp_min(0)

    # ------------------------------------------------------------------ per-step state
    def _compute_task_state(self):
        super()._compute_task_state()
        ts, N = self.ts, self.num_envs
        jp = self.cart.data.joint_pos.torch
        ts.slide_q = jp[:, self.slide_ids]
        ts.rot_q = torch.stack([jp[:, self.rot_ids[r]] for r in ROTATIONS], 1)          # (N, 3, 2)
        # Chain velocities by finite difference over the control step: PhysX reports biased joint velocities
        # for joints with an implicit drive (validated: a pivot at rest reads -0.055 rad/s), positions are exact.
        if self.common_step_counter != self._chain_step:
            self._chain_step = self.common_step_counter
            ts.slide_v = (ts.slide_q - self._slide_q_prev) / self.dt
            ts.rot_v = (ts.rot_q - self._rot_q_prev) / self.dt
            self._slide_q_prev, self._rot_q_prev = ts.slide_q.clone(), ts.rot_q.clone()
        if hasattr(self, "_labels"):
            # once per control step: this method runs twice per step (post-physics and observation update) with no
            # physics step in between; only environments reset in between change, and their labels are invalid
            if self.common_step_counter != self._labels_step:
                self._labels_step = self.common_step_counter
                hands, valid = self._labels.sample()                # (N, 12) cart axes, about each palm origin
                self._hand_cart = hands.view(N, 2, 6)
                self._hand_valid = valid
            else:
                fresh = self.episode_length_buf == 0
                self._hand_valid = self._hand_valid & ~fresh
                self._hand_cart = torch.where(fresh.view(N, 1, 1), 0., self._hand_cart)
        load = self._hand_cart
        force, moment_palm = load[..., :3], load[..., 3:]
        # move the moment to the grasp point (tube axis): M_g = M_palm - r x F, r = palm origin -> grasp point
        palm_pose = self.rigid_body_states[:, self.palm_body_ids, :7]
        r_w = torch.stack([quat_apply(palm_pose[:, i, 3:7], self.grasp_point_palm[i].expand(N, 3)) for i in range(2)], 1)
        c, s = torch.cos(ts.cart_yaw).view(N, 1), torch.sin(ts.cart_yaw).view(N, 1)
        r_c = torch.stack((c * r_w[..., 0] + s * r_w[..., 1], -s * r_w[..., 0] + c * r_w[..., 1], r_w[..., 2]), -1)
        moment = moment_palm - torch.cross(r_c, force, dim=-1)
        ts.grasp_force_c, ts.grasp_moment_c = force, moment
        ts.load_across = torch.hypot(force[..., 0], force[..., 2]) * self._hand_valid.float().unsqueeze(1)
        mult = self.capacity_mult.view(N, 1)
        ts.grasp_ratio = torch.stack((
            (-force[..., 0]).clamp_min(0.) / (self.pull_capacity * mult),
            force[..., 2].clamp_min(0.) / (self.lift_capacity * mult),
            moment[..., 1].abs() / (self.roll_capacity * mult),
            moment[..., 0].abs() / (self.moment_capacity * mult),
            moment[..., 2].abs() / (self.moment_capacity * mult)), -1)                  # (N, 2, 5)
        ts.grasp_push_ratio = force[..., 0].clamp_min(0.) / (self.task_cfg.grasp.push_capacity_n * mult)
        # internal load on the low-passed loads (once per control step; zero for an invalid / fresh hand)
        load = torch.cat((force, moment), -1) * self._hand_valid.float().view(N, 1, 1)
        if self.common_step_counter != self._load_f_step:
            self._load_f_step = self.common_step_counter
            self._load_f = self._load_f + self._load_alpha * (load - self._load_f)
        ts.internal_cost, ts.internal_frac, ts.internal_share = self._internal_load(
            self._load_f[..., :3], self._load_f[..., 3:], mult)
        ts.internal_frac_raw = self._internal_load(force, moment, mult)[1]
        g = self.task_cfg.grasp
        ts.pivot = torch.maximum(ts.rot_q[:, 0].abs() / g.roll_break_rad,
                                 torch.maximum(ts.rot_q[:, 1].abs(), ts.rot_q[:, 2].abs()) / g.tilt_break_rad)   # (N, 2)

    def _internal_load(self, force, moment, mult):
        """Split the bimanual hand load into its cart-driving part and the internal rest.

        f (12) = per hand [Fx Fy Fz Mx My Mz] at the grasp point, cart axes. The components that act on the cart's
        planar motion are d = U f = [Fx_L, Fx_R, Fy_L + Fy_R, Mz_L + Mz_R]: each hand's push / pull (their sum
        drives, their difference steers), the common lateral force and the common yaw moment. The driving part is
        the capacity-weighted projection f* = S U^T (U S U^T)^-1 U f, S = diag(s^2) with s each component's
        capacity, and f_int = f - f* lies in the null space of U: opposing lateral forces (squeeze), opposing yaw
        moments, vertical forces and the tilt / roll moments -- loads that move nothing. Returns |S^-1/2 f_int|^2
        (dimensionless) and the internal share |f_int|_S / |f|_S, per environment.
        """
        N, dev = force.shape[0], force.device
        f = torch.cat((force, moment), -1).reshape(N, 12)
        m = mult.view(N, 1)
        s = torch.stack((self.pull_capacity * m, self.slide_static, self.lift_capacity * m,
                         self.moment_capacity * m, self.roll_capacity * m, self.moment_capacity * m), -1).reshape(N, 12)
        # U S U^T is diagonal (U selects Fx_L, Fx_R and sums the Fy and Mz pairs), so the projection has a closed
        # form: Fx_L / Fx_R are all driving; the Fy and Mz pairs keep their sum, split in proportion to s^2; every
        # other component is internal. Identical to S U^T (U S U^T)^-1 U f without a batched solve (whose error
        # check synchronises the GPU every step).
        # Friction-needed internal load (Kerr & Roth 1986: internal grasp forces keep the contacts inside their
        # friction cones): when a hand's force along the tube exceeds its unloaded friction limit, the load across
        # the tube that lifts the limit to |Fy|, L_req = (|Fy| - offset) / mu - N, is required. Fx is a driving
        # component already; the vertical part sqrt(L_req^2 - Fx^2) of |Fz| is therefore not counted as internal.
        f_star = torch.zeros_like(f)
        f_star[:, 0], f_star[:, 6] = f[:, 0], f[:, 6]
        for a, b in ((1, 7), (5, 11)):
            wa, wb = s[:, a].square(), s[:, b].square()
            total = (f[:, a] + f[:, b]) / (wa + wb)
            f_star[:, a], f_star[:, b] = wa * total, wb * total
        g = self.task_cfg.grasp
        unloaded = (self.grip_mu * self.grip_normal + g.slide_offset_n).clamp_min(g.slide_min_n)
        l_req = torch.where(force[..., 1].abs() > unloaded,
                            (force[..., 1].abs() - g.slide_offset_n) / self.grip_mu.clamp_min(1e-3) - self.grip_normal, 0.)
        fz_ok = (l_req.clamp_min(0.).square() - force[..., 0].square()).clamp_min(0.).sqrt()          # (N, 2)
        fz = force[..., 2]
        f_star[:, 2] = torch.sign(fz[:, 0]) * torch.minimum(fz[:, 0].abs(), fz_ok[:, 0])
        f_star[:, 8] = torch.sign(fz[:, 1]) * torch.minimum(fz[:, 1].abs(), fz_ok[:, 1])
        f_int = (f - f_star) / s
        sq = f_int.square()
        cost = sq.sum(-1)
        frac = cost.sqrt() / (f / s).square().sum(-1).sqrt().clamp_min(1e-6)
        share = (sq[:, :6] + sq[:, 6:]) / cost.clamp_min(1e-9).unsqueeze(1)
        valid = self._hand_valid.float()
        return cost * valid, frac * valid, share * valid.unsqueeze(1)

    def _update_persistence_counters(self):
        ts, tc = self.ts, self.task_cfg.termination
        valid = self._hand_valid.unsqueeze(1)
        kinds = torch.stack((((ts.grasp_ratio[..., 0] > 1.) & valid).any(1), ((ts.grasp_ratio[..., 1] > 1.) & valid).any(1),
                             ((ts.grasp_push_ratio > 1.) & valid).any(1), torch.zeros_like(valid[:, 0])), -1)
        self.overload_dir = kinds                                             # V3.2 slots: pull/lift/push/(pivot)
        over = kinds.any(dim=1)
        self.short_overloads += (~over & (self.overload_counter >= 5)).float()
        self.overload_counter = torch.where(over, self.overload_counter + 1, torch.zeros_like(self.overload_counter))
        lifted = ts.wheel_lift.max(dim=1).values > tc.wheel_lift_m
        self.wheel_lift_counter = torch.where(lifted, self.wheel_lift_counter + 1, torch.zeros_like(self.wheel_lift_counter))
        self.slip_sum += ts.slip_v.abs().sum(dim=1) * self.dt
        self.drift_max = torch.maximum(self.drift_max, (self.slide_anchor - self.slide_init).abs().max(dim=1).values)
        self.pivot_max = torch.maximum(self.pivot_max, ts.pivot.max(dim=1).values)

    def _post_physics_step_callback(self):
        # This call alone computes the original state, filters and diagnostics (and the V7 grasp state).
        super()._post_physics_step_callback()
        if not hasattr(self, "_labels"):
            return
        # load-dependent slide limit mu * (grip + load across the tube) + offset, then the elasto-plastic anchor;
        # the new target and limit act during the next control step
        g = self.task_cfg.grasp
        if g.load_dependent_friction:
            self.slide_static = self._slide_limit(self.ts.load_across)
        self._update_slide_anchor()
        if self.common_step_counter % max(1, int(g.friction_update_interval)) == 0:
            self._write_slide_limit()
        if float(self.task_cfg.cart_physics.swivel_scrub_speed_m_s) < 1.0e6:      # revision 8: constant scrub
            self._write_swivel_friction()
        hands, valid_hand = self._hand_cart.reshape(self.num_envs, 12), self._hand_valid
        self._pre_reset_hands = torch.cat((hands/self.hand_scale, valid_hand[:, None]), -1).detach()
        # HOMIE's target branch uses the clean next privileged frame, not noisy actor history.
        imu_q = self.rigid_body_states[:, self.imu_index, 3:7]
        imu_w = quat_rotate_inverse(imu_q, self.rigid_body_states[:, self.imu_index, 10:13])
        imu_g = quat_rotate_inverse(imu_q, self.gravity_vec)
        clean_low = torch.cat((self.commands[:, :3]*self.commands_scale,
            self.commands[:, 4:5], imu_w*self.obs_scales.ang_vel, imu_g,
            (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
            self.dof_vel*self.obs_scales.dof_vel, self.actions[:, :12]), -1)
        next_frame = torch.cat((clean_low, self.ts.rel_yaw[:, None]), -1)
        speed = torch.cat((self.ts.cart_v, self.ts.cart_wz[:, None]), -1)
        target = make_next_critic_observation(next_frame, speed)
        valid_speed = torch.isfinite(target).all(-1) & (self.episode_length_buf > 0)
        self.him_next_target = torch.cat((torch.where(valid_speed[:, None], target, 0.),
                                          valid_speed[:, None]), -1).detach()

    def check_termination(self):
        super().check_termination()
        pivoted = self.ts.pivot.max(dim=1).values > 1.0          # hand levered past its break angle: grip lost
        slid = (self.slide_anchor - self.slide_init).abs().max(dim=1).values > self.task_cfg.grasp.slide_drift_break_m
        overload = self.term_reason_buf["grasp_overload"]
        kinds = torch.cat((self.overload_dir & overload.unsqueeze(1), slid.unsqueeze(1)), 1)
        kinds[:, 3] = pivoted
        self.grasp_overload_kind = kinds
        self.grasp_kind_counts += kinds.float().sum(0)
        lost = pivoted | slid
        self.term_reason_buf["grasp_overload"] = overload | lost
        self.terminated_buf = self.terminated_buf | lost
        self.reset_buf = self.terminated_buf | self.time_out_buf

    # ------------------------------------------------------------------ V7 rewards
    def _track(self, axis):
        e = self.cmd_ref_f[:, axis] - self.cart_vel_f[:, axis]          # V3.2's equally filtered reference / velocity
        return torch.exp(-(e / self.tracking_sigma[axis]).square())

    def _reward_track_cart_vx(self):
        return self._track(0)

    def _reward_track_cart_vy(self):
        return self._track(1)

    def _reward_track_cart_wz(self):
        return self._track(2)

    def _reward_hl_action_clip(self):
        """Squared excess of the unclipped high-level output over the action clip (zero inside the clip)."""
        return (self.hl_raw.abs() - self.task_cfg.actions.clip).clamp_min(0.).square().sum(dim=1)

    def _reward_grasp_internal(self):
        return self.ts.internal_cost

    def _reward_grasp_margin(self):
        excess = (self.ts.grasp_ratio - self.task_cfg.grasp.margin_free_frac).clamp_min(0.)
        return excess.square().sum(dim=(1, 2)) * self._hand_valid.float()

    def _reward_grasp_slip(self):
        g = self.task_cfg.grasp
        slide = (self.ts.slip_v / g.slip_speed_scale_m_s).square()             # plastic slip only
        rot = (self.ts.rot_v / g.rotation_speed_scale_rad_s).square().sum(dim=1)
        return (slide + g.rotation_slip_weight * rot).sum(dim=1)

    def _reward_grasp_drift(self):
        g = self.task_cfg.grasp
        slide = (((self.slide_anchor - self.slide_init).abs() - g.drift_free_m).clamp_min(0.) / g.drift_scale_m).square()
        rot = ((self.ts.rot_q.abs() - g.rotation_free_rad).clamp_min(0.) / g.rotation_scale_rad).square().sum(dim=1)
        return (slide + rot).sum(dim=1)

    # ------------------------------------------------------------------ V6 observation pipeline
    def _update_hl_observations(self):
        # Exactly one original observation update: same noisy yaw, same random
        # draw count, same original history and reward-only true state sources.
        super()._update_hl_observations()
        self.legacy_actor_obs = self.hl_obs
        self.legacy_critic_obs = self.hl_critic_obs
        rv = self.task_cfg.robot_velocity
        if rv.replace_homie_estimate:                  # V7.1: the push-cart-tuned estimator replaces these slots
            self.legacy_actor_obs = mask_homie_velocity(self.legacy_actor_obs)
            self.legacy_critic_obs = mask_homie_velocity(self.legacy_critic_obs)
        if rv.enabled:
            valid = (self.episode_length_buf >= int(rv.valid_after_steps)).float().unsqueeze(1)
            self.robot_velocity_target = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, valid), -1).detach()
        fresh = self.episode_length_buf == 0
        source = self.legacy_actor_obs.reshape(self.num_envs, 5, 124)
        self.hl_obs = without_handle_position(self.legacy_actor_obs)
        self.estimator_obs = self.estimator_history.push(him_frame(source[:, -1]), fresh)
        # A reset observation has no hand label. Never change the separate
        # pre-reset t+1 target: PPO still needs it for the terminal transition.
        self.hand_target = torch.where(fresh[:, None], 0., self._pre_reset_hands).clone()
        self.hl_critic_obs = torch.cat((self.legacy_critic_obs, self.hand_target, self._v7_privileged()), -1)

    def _v7_privileged(self):
        ts, N = self.ts, self.num_envs
        c, s = torch.cos(ts.cart_yaw), torch.sin(ts.cart_yaw)

        def planar(v):          # world xy -> cart axes
            return torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), -1)

        hand = torch.stack((ts.slide_q / 0.1, ts.slip_v / 0.2, ts.rot_q[:, 0] / 0.3, ts.rot_q[:, 1] / 0.3,
                            ts.rot_q[:, 2] / 0.3, self.slide_static / 60., self.grip_mu * self.grip_normal / 60.,
                            self.roll_stiffness / 60., self.roll_capacity / 2.5, self.moment_capacity / 6.,
                            self.pull_capacity / 80., self.lift_capacity / 80.), -1)               # (N, 2, 12)
        bump = self.bump_force_w * (self.bump_timer > 0).float().unsqueeze(1)
        cart = torch.cat(((self.rolling_coeff / 0.04).unsqueeze(1), (self.swivel_static / 0.4).unsqueeze(1),
                          planar(self.bias_force_w) / 20., planar(bump) / 60.), -1)
        out = torch.cat((hand.reshape(N, 24), cart), -1)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite V7 privileged observation")
        return out.clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

    # ------------------------------------------------------------------ episodes
    def _log_episode_diagnostics(self, env_ids):
        super()._log_episode_diagnostics(env_ids)
        ep = self.extras.setdefault("episode", {})
        for name in ("term_overload_tangential", "term_overload_torque"):
            ep.pop(name, None)
        ids = env_ids.to(torch.long)
        for i, name in enumerate(OVERLOAD_KINDS):
            ep[f"term_grasp_{name}"] = self.grasp_overload_kind[ids, i].float().mean()
        ep["grasp_capacity_mult"] = self.capacity_mult.mean()
        ep["grasp_internal_frac"] = self.ts.internal_frac.mean()              # low-passed loads (the rewarded one)
        ep["grasp_internal_frac_raw"] = self.ts.internal_frac_raw.mean()      # instantaneous loads
        share = self.ts.internal_share.mean(0)
        for i, name in enumerate(("fy_squeeze", "fz", "mx_tilt", "my_roll", "mz_twist")):
            ep[f"grasp_internal_share_{name}"] = share[i + 1]
        st = self.cur.stats()
        ep["curriculum_level"] = torch.tensor(st["mean"] * LEVELS, device=self.device)
        for key in ("mean", "p10", "p50", "p90", "at_max_frac"):
            ep[f"curriculum_difficulty_{key}"] = torch.tensor(st[key], device=self.device)
        ep["grasp_slide_travel_m"] = self.slip_sum[ids].mean()
        ep["grasp_drift_max_m"] = self.drift_max[ids].mean()
        ep["grasp_short_overload_per_ep"] = self.short_overloads[ids].mean()
        ep["grasp_pivot_max_frac"] = self.pivot_max[ids].mean()
        ep["grasp_slide_static_N"] = self.slide_static[ids].mean()
        ep["grasp_sliding_frac"] = self.slide_sliding.float().mean()
        ep["cart_rolling_coeff"] = self.rolling_coeff[ids].mean()
        ep["cart_yaw_damping"] = self.yaw_damping[ids].mean()

    def _record_episode_stats(self, env_ids):
        super()._record_episode_stats(env_ids)          # V3.2 window statistics (its level tracker is passive)
        done = env_ids[self.episode_length_buf[env_ids] > 0].to(torch.long)   # the very first reset is not an episode
        if len(done) == 0 or not hasattr(self, "cur"):
            return
        rmse = torch.sqrt(self.track_err2[done] / self.track_steps[done].clamp_min(1.).unsqueeze(1))    # (n, 3)
        failed = self.terminated_buf[done]
        success = ~failed & (rmse <= self.success_rmse).all(dim=1)
        self.cur.update(done, success, failed)
        self._refresh_global_level()

    def _sample_task_command(self, env_ids):
        """V3.2 task command sampling with each environment's own command ranges."""
        cc, dev, n = self.task_cfg.commands, self.device, len(env_ids)
        if self.fixed_command is not None:
            self.cmd_task[env_ids] = torch.tensor(self.fixed_command, device=dev, dtype=torch.float32)
            self.cmd_timer[env_ids] = 10 ** 9
            return
        vx_lo, vx_hi, vy_max, wz_max = self.cur.commands(env_ids.to(torch.long))
        u = torch.rand(n, 3, device=dev)
        vx = vx_lo + (vx_hi - vx_lo) * u[:, 0]
        vy = vy_max * (2 * u[:, 1] - 1)
        wz = wz_max * (2 * u[:, 2] - 1)
        keep = (torch.rand(n, device=dev) >= cc.zero_prob).float()
        self.cmd_task[env_ids] = torch.stack((vx, vy, wz), -1) * keep.unsqueeze(1)
        lo, hi = int(cc.resample_time_s[0] / self.dt), int(cc.resample_time_s[1] / self.dt)
        self.cmd_timer[env_ids] = torch.randint(lo, hi + 1, (n,), device=dev)

    def _push_robots(self):
        """V3.2's pelvis force pulse, with its magnitude scaled by each environment's push level."""
        super()._push_robots()
        if self.task_cfg.randomization.robot_push_duration_s > 0:
            self.push_force *= self.push_scale.view(-1, 1, 1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not len(env_ids) or not hasattr(self, "estimator_history"):
            return
        ids = env_ids.to(torch.long)
        self.estimator_history.reset(env_ids)
        self._pre_reset_hands = self._pre_reset_hands.clone()
        self._pre_reset_hands[env_ids] = 0.
        self.slip_sum[ids] = 0.
        self.short_overloads[ids] = 0.
        self.drift_max[ids] = 0.
        self.pivot_max[ids] = 0.
        self.grasp_overload_kind[ids] = False
        self._load_f[ids] = 0.
        if not self._transition_active:
            self.him_next_target = self.him_next_target.clone()
            self.him_next_target[env_ids] = 0.

    def step(self, actions):
        self._transition_active = True
        try:
            return super().step(actions)
        finally:
            self._transition_active = False

    def observation_dict(self):
        out = {"policy": self.hl_obs, "estimator": self.estimator_obs,
               "critic": self.hl_critic_obs, "hand_target": self.hand_target,
               "him_next_target": self.him_next_target}
        if self.task_cfg.robot_velocity.enabled:
            out["robot_velocity_target"] = self.robot_velocity_target
        return out
