"""V5 reaction-aware push-cart RL over the frozen HOMIE.

``PushCartEnv`` extends ``PushCartSimEnv`` (scene, preset pose, cart) with

* an 18-dim high-level action: 3 residuals on the HOMIE velocity command (low-passed, clipped to HOMIE's
  training ranges) and 15 upper-body joint corrections on top of the grasp pose (slew-limited, fixed scale);
  a finite recovery gate also permits bounded correction after a stationary disturbance;
* a first-order reference ``cmd_ref`` of the step task command: the achievable velocity curve the cart is
  rewarded for following (and what the actor observes as its task command);
* the frozen HOMIE policy run inside ``step`` (50 Hz, same rate as the high-level policy);
* asymmetric observations (actor: proprioception, delayed/noisy pose and virtual wrist measurement,
  estimated mass and causal reaction features; critic: additional cart/contact/centroidal truth);
* interaction-centred rewards, push/pull-aware terminations, a grasp-invariant reset with whole-system
  pose randomisation, cart dynamics randomisation and a fine-grained multi-scale curriculum.

HOMIE's reward / termination / reset machinery (``HomieTaskCore``) is reused: our ``_reward_*`` methods
are installed through ``cfg.rewards.scales`` and ``check_termination`` / ``reset_idx`` are extended.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp

from homie_isaaclab.math_utils import euler_from_quaternion, quat_apply, quat_mul, quat_rotate_inverse, torch_rand_float

from .curriculum import CurriculumTracker, LevelParams
from .cart_quality import CartQualityTracker, QUALITY_REWARD_NAMES, rotate_planar
from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
from .paths import HOMIE_JOINTS, HOMIE_RUN_DIR, ROOT
from .sim_env import PushCartSimEnv
from .task_config import HL_FRAME, PRIVILEGED_FRAME, PushCartTaskCfg, num_actor_obs, num_critic_obs
from .reaction import ReactionController
from .reaction_observation import ReactionObservation

UPPER_JOINTS = HOMIE_JOINTS[12:]
WRIST_IDS = [i for i, n in enumerate(UPPER_JOINTS) if "wrist_pitch" in n or "wrist_yaw" in n]
TERMINATION_REASONS = ("fall", "grasp_overload", "cart_yaw", "cart_distance", "cart_tilt", "wheel_lift", "blowup")


def configure_homie_cfg(cfg, task_cfg: PushCartTaskCfg):
    """Turn a G1RoughCfg into the push-cart low-level profile: our rewards, no HOMIE curricula, no leg jitter."""
    cfg.env.episode_length_s = task_cfg.env.episode_length_s
    cfg.env.action_curriculum = False
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = 1.0e6          # commands come from the high-level policy every step
    cfg.rewards.scales = task_cfg.rewards.scales
    cfg.rewards.only_positive_rewards = False
    cfg.domain_rand.randomize_initial_joint_pos = False   # legs must stay at the calibrated stance
    cfg.domain_rand.push_robots = False                   # opened by the curriculum
    return cfg


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class PushCartEnv(PushCartSimEnv):
    def __init__(self, cfg, task_cfg: PushCartTaskCfg, project_root: Path = ROOT, device="cuda:0", headless=True, *,
                 checkpoint: Path | None = None, sim_context=None, level: int | None = None, cart_params=None,
                 allow_robot_cart_contact=False):
        self.task_cfg = task_cfg
        configure_homie_cfg(cfg, task_cfg)
        super().__init__(cfg, project_root, device, headless, with_cart=True, sim_context=sim_context,
                         cart_params=cart_params, allow_robot_cart_contact=allow_robot_cart_contact)
        self.policy = HimLowLevelPolicy(Path(checkpoint) if checkpoint else frozen_checkpoint(), device)
        self.curriculum = CurriculumTracker(task_cfg, level)
        self.num_hl_actions = task_cfg.env.num_hl_actions
        self.num_hl_obs = num_actor_obs(task_cfg)
        self.num_hl_critic_obs = num_critic_obs(task_cfg)
        self._init_task_constants()
        self._init_hl_buffers()
        self._apply_level(self.curriculum.params)
        print(f"[pushcart-env] high-level actor obs {self.num_hl_obs} ({HL_FRAME} x {task_cfg.env.actor_history}), "
              f"critic obs {self.num_hl_critic_obs}, actions {self.num_hl_actions}, level {self.curriculum.level}; "
              f"HOMIE {self.policy.checkpoint.name}", flush=True)

    # ------------------------------------------------------------------ set-up
    def _init_task_constants(self):
        tc, dev, N = self.task_cfg, self.device, self.num_envs
        self.residual_scale = torch.tensor(tc.actions.cmd_residual_scale, device=dev)
        lim = torch.tensor(tc.actions.homie_cmd_limits, device=dev)
        self.cmd_lo, self.cmd_hi = lim[:, 0], lim[:, 1]
        self.cmd_alpha = min(1.0, self.dt / tc.actions.cmd_filter_tau_s) if tc.actions.cmd_filter_tau_s > 0 else 1.0
        self.ref_alpha = min(1.0, self.dt / tc.commands.ref_tau_s) if tc.commands.ref_tau_s > 0 else 1.0
        tau_m = float(getattr(tc.rewards, "measure_filter_tau_s", 0.0))
        self.meas_alpha = min(1.0, self.dt / tau_m) if tau_m > 0 else 1.0
        self.upper_lo = self.hard_dof_pos_limits[12:, 0].unsqueeze(0)
        self.upper_hi = self.hard_dof_pos_limits[12:, 1].unsqueeze(0)
        self.hold_target_q = self.preset_dof_pos[:, 12:] + self.upper_hold_offset          # (1, 15)
        self.upper_range = (self.upper_hi - self.upper_lo)
        self.wrist_ids = torch.tensor(WRIST_IDS, device=dev, dtype=torch.long)
        # cart geometry
        derived = self.cart_params["derived"]
        yaw0 = float(derived["base_yaw"])
        d = torch.tensor(derived["bar_center_world"], dtype=torch.float32) - torch.tensor(derived["base_pos_world"], dtype=torch.float32)
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        self.bar_local = torch.tensor([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]], device=dev)
        self.wheel_radius = float(self.cart_params["wheels"]["radius"])
        self.cart_base_kg = float(self.cart_params["mass"]["base_kg"])
        names = list(self.cart.body_names)
        self.cart_base_body = names.index("base_link")
        self.wheel_body_ids = torch.tensor([i for i, n in enumerate(names) if n.startswith("wheel_")], device=dev, dtype=torch.long)
        # the two casters on the handle side (negative cart x) are the ones the feet can reach
        positions = self.cart_params["wheels"]["positions_xy"]
        near = [i for i, (x, _) in enumerate(positions) if x < 0]
        self.near_caster_body_ids = torch.tensor([names.index(f"caster_{i}_swivel") for i in near], device=dev, dtype=torch.long)
        swivel, _ = self.cart.find_joints("caster_.*_swivel_joint")
        wheels, _ = self.cart.find_joints("wheel_.*_joint")
        self.swivel_joint_ids = torch.tensor(swivel, device=dev, dtype=torch.long)
        self.wheel_joint_ids = torch.tensor(wheels, device=dev, dtype=torch.long)
        self.wheel_joint_ids_sim = self.wheel_joint_ids.to(torch.int32)      # Isaac Lab's write_*_index kernels need int32
        self.cart_all_joint_ids = torch.arange(self.cart.num_joints, device=dev, dtype=torch.int32)
        self.cart_default_mass = self.cart.data.body_mass.torch.clone()
        self.cart_default_inertia = self.cart.data.body_inertia.torch.clone()
        self.cart_default_com = self.cart.data.body_com_pose_b.torch.clone()
        self.grasp_base_thresholds = torch.tensor(
            [tc.termination.pull_n, tc.termination.up_n, tc.termination.tangential_n, tc.termination.torque_nm], device=dev)
        self.commands_scale3 = self.commands_scale                                          # [2, 2, 0.5]
        self.level_max = tc.curriculum.num_levels - 1

    def _init_hl_buffers(self):
        dev, N, A, H = self.device, self.num_envs, self.task_cfg.env.num_hl_actions, self.task_cfg.env.actor_history
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, device=dev, dtype=dtype)   # noqa: E731
        self.cmd_task, self.cmd_ref = z(N, 3), z(N, 3)
        self.cmd_ref_f, self.cart_vel_f, self.base_wz_f = z(N, 3), z(N, 3), z(N)   # low-passed reference / cart velocity / torso yaw rate (rewards)
        self.cmd_homie, self.cmd_homie_prev = z(N, 3), z(N, 3)
        self.cmd_timer = z(N, dtype=torch.long)
        self.hl_actions, self.hl_actions_prev, self.hl_actions_prev2, self.hl_action_raw_prev = z(N, A), z(N, A), z(N, A), z(N, A)
        self.hl_raw, self.hl_raw_prev, self.hl_raw_prev2 = z(N, A), z(N, A), z(N, A)      # unclipped policy output (regularisers)
        self.overload_dir = z(N, 4, dtype=torch.bool)                   # pull / up / tangential / torque over threshold
        # per-episode diagnostics, cleared with the episode
        self.track_err2, self.track_steps = z(N, 3), z(N)
        self.track_vr, self.track_rr = z(N, 3), z(N, 3)                 # sum v * ref, sum ref^2 (regression gain)
        self.turn_dq_sum, self.turn_steps = z(N), z(N)
        self.rel_yaw_abs_sum, self.sat_steps = z(N), z(N)
        self.hl_delay = z(N, dtype=torch.long)
        self.upper_delta_q = z(N, 15)
        self.vel_est = z(N, 3)
        self.actor_hist = z(N, H, HL_FRAME)
        self.hl_obs = z(N, H * HL_FRAME)
        self.hl_critic_obs = z(N, H * HL_FRAME + PRIVILEGED_FRAME)
        self.cart_pose_hist = z(N, 3, 4)
        self.cart_pose_delay = z(N, dtype=torch.long)
        from .command_schedule import V4CommandSchedule
        from .idle_correction import IdleCorrectionTracker
        self.command_schedule = (V4CommandSchedule(self.task_cfg.commands, self.dt, N, dev)
                                 if self.task_cfg.commands.sequence_enabled else None)
        self.idle_correction = (IdleCorrectionTracker(self.task_cfg.idle_correction, self.dt, N, dev)
                                if self.task_cfg.idle_correction.enabled else None)
        self.idle_gate_applied, self.idle_gate_sum = z(N), z(N)
        self.force_c_prev = z(N, 2, 3)
        # The first valid post-reset wrench is a baseline, not a force-rate event.
        self.force_c_prev_valid = z(N, dtype=torch.bool)
        self.overload_counter = z(N, dtype=torch.long)
        self.wheel_lift_counter = z(N, dtype=torch.long)
        self.cart_mass_total = torch.full((N,), self.cart_base_kg, device=dev)
        self.payload_com = z(N, 2)
        self.wheel_friction = torch.full((N,), float(self.cart_params["wheels"]["friction"]), device=dev)
        self.wheel_damping = torch.full((N,), float(self.cart_params["wheels"]["wheel_damping"]), device=dev)
        self.swivel_damping = torch.full((N,), float(self.cart_params["wheels"]["swivel_damping"]), device=dev)
        self.grasp_jitter = torch.ones(N, device=dev)
        self.stall_timer = z(N, dtype=torch.long)
        self.stall_wheel = z(N, dtype=torch.long)
        self.cart_push_timer = z(N, dtype=torch.long)
        self.push_force = z(N, 1, 3)                                  # pelvis force pulse (world frame)
        self.push_timer = z(N, dtype=torch.long)
        self.last_push_step = -10 ** 9
        self.pelvis_body_id = torch.tensor([self.robot.body_names.index("pelvis")], device=dev, dtype=torch.long)
        self.terminated_buf = z(N, dtype=torch.bool)             # HOMIE creates it in check_termination; reset() runs first
        self.term_reason_buf = {name: z(N, dtype=torch.bool) for name in TERMINATION_REASONS}
        self.term_counts = z(len(TERMINATION_REASONS) + 1)          # GPU accumulator, order: TERMINATION_REASONS + time_out
        self.episodes_finished = 0
        self._track_sum, self._fail_sum, self._count = z(3), z(()), z(())       # curriculum window: per-axis tracking
        self._gain_vr, self._gain_rr, self._gain_steps = z(3), z(3), z(())      # ... and regression gain of v on the reference
        self._next_curriculum_eval = 0
        self.ts = SimpleNamespace()
        self.level_tensor = z(())
        # evaluation hooks (None = sample as usual): fixed task command, payload and caster heading
        self.fixed_command = None
        self.fixed_payload_kg = None
        self.fixed_caster_angle = None
        self.cart_quality = None
        if self.task_cfg.quality.enabled:
            self.cart_quality = CartQualityTracker(self.task_cfg.quality, self.dt, N, dev)
        elif any(getattr(self.task_cfg.rewards.scales, key) != 0 for key in QUALITY_REWARD_NAMES):
            raise ValueError("cart quality rewards require task.quality.enabled=true")
        self.quality_sums = {name: z(N, 3) for name in ("velocity_band2", "wrench_band2", "raw_error2", "task_wrench")}
        self.quality_steps = z(N)
        self.response_error2_sum, self.response_steps = z(N, 3), z(N)
        self.quality_peaks = {name: z(N) for name in ("force_N", "moment_Nm", "cart_tilt_rad", "robot_tilt_rad")}
        self.cart_mass_estimate = self.cart_mass_total.clone()
        self.robot_mass_estimate = self.robot.data.body_mass.torch.sum(1).clone()
        self.reaction_observer = ReactionObservation(self.task_cfg.reaction, self.dt, N, dev,
            handle_offset_cart=self.bar_local.tolist())
        self.reaction = self.reaction_observer.controller
        # Privileged, current-step reward model is separate from the delayed actor sensor.
        from homie_isaaclab.math_utils import class_to_dict
        teacher_cfg = class_to_dict(self.task_cfg.reaction)
        teacher_cfg.update(sensor_delay_steps=0, sensor_force_noise_std_n=0., sensor_moment_noise_std_nm=0.,
                           sensor_force_bias_std_n=0., sensor_moment_bias_std_nm=0., compensation_enabled=False)
        self.reaction_teacher = ReactionController(teacher_cfg, self.dt, N, dev)
        self.cart_pulse_timer = z(N, dtype=torch.long)
        self.cart_pulse_wrench = z(N, 6)
        self.cart_applied_wrench = z(N, 6)
        self.cart_pulse_impulse = z(N, 6)
        self.centroidal_filtered = z(N, 6)
        self.centroidal_cost = z(N)
        self.support_cost = z(N)
        self.reaction_episode_sum = z(N, 4)
        self.reaction_episode_samples = z(N)

    def _apply_level(self, params: LevelParams):
        self.level_params = params
        self.upper_scale = torch.tensor([params.waist_scale] + [params.arm_scale] * 14, device=self.device).unsqueeze(0)
        self.cfg.domain_rand.push_robots = bool(params.robot_push and self.task_cfg.randomization.robot_push)
        self.level_tensor.fill_(float(params.level))

    # ------------------------------------------------------------------ public API
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        self.vel_est = self.policy.estimate_velocity(self.obs_buf) / self.obs_scales.lin_vel
        self._update_hl_observations()
        return self.hl_obs, self.hl_critic_obs

    def step(self, hl_actions: torch.Tensor):
        tc = self.task_cfg
        unclipped = hl_actions.to(self.device)
        if unclipped.shape != (self.num_envs, self.num_hl_actions):
            raise ValueError(f"V5 actions require shape [{self.num_envs},{self.num_hl_actions}]")
        if not torch.isfinite(unclipped).all():
            raise FloatingPointError("Non-finite V5 action rejected before advancing physics")
        self.hl_raw_prev2.copy_(self.hl_raw_prev)
        self.hl_raw_prev.copy_(self.hl_raw)
        self.hl_raw.copy_(unclipped)
        raw = unclipped.clamp(-tc.actions.clip, tc.actions.clip)
        applied = torch.where(self.hl_delay.unsqueeze(1) > 0, self.hl_action_raw_prev, raw)
        self.hl_action_raw_prev.copy_(raw)
        self.hl_actions_prev2.copy_(self.hl_actions_prev)
        self.hl_actions_prev.copy_(self.hl_actions)
        self.hl_actions.copy_(applied)
        # task command schedule; the cart is asked to follow the command's first-order response, not the step
        self.cmd_timer -= 1
        due = (self.cmd_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(due):
            self._sample_task_command(due)
        self.cmd_ref += self.ref_alpha * (self.cmd_task - self.cmd_ref)
        # Nominal idle fades corrections; detected disturbances permit finite recovery windows.
        gate = (self.cmd_ref.norm(dim=1, keepdim=True) / tc.actions.idle_gate_speed).clamp(0., 1.) if tc.actions.idle_gate_speed > 0 else 1.0
        if self.idle_correction is not None:
            idle_gate = self.idle_correction.gate(self.cmd_ref, self.cmd_task)
            self.idle_gate_applied.copy_(idle_gate[:, 0])
            self.idle_gate_sum += self.idle_gate_applied
            if isinstance(gate, torch.Tensor):
                gate = torch.maximum(gate, idle_gate)
        # HOMIE command = reference + residual, low-passed, clipped to the training ranges
        reaction_on = tc.reaction.enabled
        if reaction_on:
            gate = torch.maximum(torch.as_tensor(gate, device=self.device), self.reaction.recovery_gate)
        # Transform both heading AND reference point: a turning cart COM and
        # the pelvis behind it require different linear velocities.
        reference_b = self.reaction_observer.reference_base(self.cmd_ref)
        correction_b = self.reaction_observer.command_correction_base if reaction_on else 0.
        target = (reference_b + gate * applied[:, :3] * self.residual_scale + correction_b).clamp(self.cmd_lo, self.cmd_hi)
        self.cmd_homie_prev.copy_(self.cmd_homie)
        self.cmd_homie += self.cmd_alpha * (target - self.cmd_homie)
        self.command_buffer.copy_(self.cmd_homie)
        self.commands[:, :3] = self.cmd_homie
        self.commands[:, 3] = 0.
        self.commands[:, 4] = self.cfg.rewards.base_height_target
        # HOMIE observes the command it is about to follow (its frame layout starts with the 3 scaled commands)
        f = self.num_one_step_obs
        self.obs_buf[:, -f:-f + 3] = self.commands[:, :3] * self.commands_scale
        # upper-body correction: slew-limited, kept inside the hard joint limits
        step_lim = tc.actions.upper_rate_limit_rad_s * self.dt
        delta_target = gate * applied[:, 3:] * self.upper_scale
        delta = self.upper_delta_q + (delta_target - self.upper_delta_q).clamp(-step_lim, step_lim)
        q_target = (self.hold_target_q + delta).clamp(self.upper_lo, self.upper_hi)
        self.upper_delta_q = q_target - self.hold_target_q
        self.upper_delta.copy_(self.upper_delta_q / self.cfg.control.action_scale)
        # disturbances acting through the simulator
        self._apply_robot_push()
        self._apply_cart_disturbances()
        # frozen low-level policy
        leg_actions, _ = self.policy.act_and_estimate(self.obs_buf)
        super().step(leg_actions)
        # Pair the velocity estimate with the NEW pose/IMU, not the preceding action's observation.
        self.vel_est = self.policy.estimate_velocity(self.obs_buf) / self.obs_scales.lin_vel
        self._update_hl_observations()
        self.extras["time_outs"] = self.time_out_buf
        self.extras["v5"] = {"reaction_wrench": self.reaction.filtered_wrench,
                             "required_wrench": self.reaction.aligned_required_wrench,
                             "recovery_gate": self.reaction.recovery_gate,
                             "cart_applied_wrench": self.cart_applied_wrench,
                             "sensor_valid": self.reaction.sensor_valid}
        return self.hl_obs, self.hl_critic_obs, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------ per-step task state
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()          # HOMIE: command hook (no-op here) + robot pushes per curriculum
        self._compute_task_state()
        # measurement filter for the tracking rewards (once per control step; the caster contact ripple of a rolling
        # cart must not read as a tracking error, see task_config.rewards.measure_filter_tau_s)
        vel = torch.cat((self.ts.cart_v, self.ts.cart_wz.unsqueeze(1)), -1)
        self.cart_vel_f += self.meas_alpha * (vel - self.cart_vel_f)
        self.cmd_ref_f += self.meas_alpha * (self.cmd_ref - self.cmd_ref_f)
        # Match cart_wz and the planar reference about world vertical; body-z
        # also contains roll/pitch rotation when the robot is tilted.
        self.base_wz_f += self.meas_alpha * (self.root_states[:, 12] - self.base_wz_f)
        self._update_reaction_diagnostics()
        if self.cart_quality is not None:
            self.cart_quality.update(vel, self.cmd_ref, self.ts.cart_yaw,
                                     self.ts.cart_force_w, self.ts.cart_moment_w,
                                     valid=self.episode_length_buf > 0)
        self._update_persistence_counters()
        self._accumulate_diagnostics()

    def _compute_task_state(self):
        ts, tc, N = self.ts, self.task_cfg, self.num_envs
        cart_pos, cart_quat, cart_lin_w, cart_ang_w = self.cart_state()
        _, _, cart_yaw = euler_from_quaternion(cart_quat)
        ts.cart_yaw = cart_yaw.squeeze(1)
        c, s = torch.cos(ts.cart_yaw), torch.sin(ts.cart_yaw)
        ts.cart_v = torch.stack((c * cart_lin_w[:, 0] + s * cart_lin_w[:, 1], -s * cart_lin_w[:, 0] + c * cart_lin_w[:, 1]), -1)
        ts.cart_speed = cart_lin_w[:, :2].norm(dim=-1)
        ts.cart_wz = cart_ang_w[:, 2]
        ts.cart_tilt = quat_rotate_inverse(cart_quat, self.gravity_vec)[:, :2].norm(dim=-1)
        bar_w = cart_pos + quat_apply(cart_quat, self.bar_local.expand(N, -1))
        rel = bar_w - self.root_states[:, :3]
        ts.bar_b = quat_rotate_inverse(self.base_quat, rel)
        ts.rel_yaw = wrap_angle(ts.cart_yaw - self.yaw.squeeze(1))
        ts.dist = rel[:, :2].norm(dim=-1)
        force_w, torque_w, _ = self.grasp_wrench()
        valid = (self.episode_length_buf > 0).view(N, 1, 1)   # PhysX wrench is stale right after a reset
        # Multiplication does not mask a stale NaN/Inf (NaN * 0 is still NaN).
        force_w = torch.where(valid, force_w, torch.zeros_like(force_w))
        torque_w = torch.where(valid, torque_w, torch.zeros_like(torque_w))
        ts.force_w_raw, ts.torque_w_raw = force_w, torque_w
        ts.cart_force_w, ts.cart_moment_w, ts.cart_com_w = self.cart_hand_wrench(force_w, torque_w)
        wrench_w = torch.stack((ts.cart_force_w[:, 0], ts.cart_force_w[:, 1], ts.cart_moment_w[:, 2]), -1)
        ts.cart_task_wrench = rotate_planar(wrench_w, ts.cart_yaw, inverse=True)
        # raw magnitudes for the blow-up guard; everything downstream sees clipped forces so that one solver
        # event cannot dominate an episode return
        ts.force_norm_raw = force_w.norm(dim=-1)
        clip = tc.rewards.force_clip_n
        scale = (clip / ts.force_norm_raw.clamp_min(1e-6)).clamp_max(1.0).unsqueeze(-1)
        ts.force_w = force_w * scale
        ts.torque_w = torque_w * scale
        cc, ss = c.view(N, 1), s.view(N, 1)
        ts.force_c = torch.stack((cc * ts.force_w[..., 0] + ss * ts.force_w[..., 1],
                                  -ss * ts.force_w[..., 0] + cc * ts.force_w[..., 1], ts.force_w[..., 2]), -1)
        ts.torque_norm = ts.torque_w.norm(dim=-1)
        ts.force_norm = ts.force_w.norm(dim=-1)
        ts.budget = tc.rewards.force_budget_base_n + tc.rewards.force_budget_per_kg * self.cart_mass_total
        wheel_z = self.cart.data.body_link_pose_w.torch[:, self.wheel_body_ids, 2]
        ts.wheel_lift = (wheel_z - self.wheel_radius).clamp_min(0.)
        ts.caster_angles = self.cart.data.joint_pos.torch[:, self.swivel_joint_ids]
        ts.upper_tau = self.torques[:, 12:]
        ts.feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # horizontal distance from each foot to the nearest handle-side caster (no physical contact in sim)
        caster_xy = self.cart.data.body_link_pose_w.torch[:, self.near_caster_body_ids, :2]      # (N, 2, 2)
        feet_xy = self.feet_pos[:, :, :2]                                                         # (N, 2, 2)
        ts.feet_caster_dist = (feet_xy.unsqueeze(2) - caster_xy.unsqueeze(1)).norm(dim=-1).min(dim=2).values   # (N, 2)

    def _update_reaction_diagnostics(self):
        """Current-step physics for reward/critic only; never part of the deployed actor input."""
        ts, rc = self.ts, self.task_cfg.reaction
        robot, cart = self.centroidal_state(self.robot), self.centroidal_state(self.cart)
        ts.robot_centroidal, ts.cart_centroidal = robot, cart
        force = rotate_planar(ts.cart_force_w, ts.cart_yaw, inverse=True)
        moment = rotate_planar(ts.cart_moment_w, ts.cart_yaw, inverse=True)
        velocity = rotate_planar(robot["velocity"], ts.cart_yaw, inverse=True)
        # In a turn, robot and cart COMs have different nominal velocities.
        # Compare the robot with rigid-coupled task motion at its own point.
        offset = rotate_planar(robot["com"] - cart["com"], ts.cart_yaw, inverse=True)
        relative_nominal = torch.stack((-self.cmd_ref[:, 2] * offset[:, 1],
                                         self.cmd_ref[:, 2] * offset[:, 0],
                                         torch.zeros_like(self.cmd_ref[:, 2])), -1)
        robot_reference = torch.cat((self.cmd_ref[:, :2], torch.zeros_like(self.cmd_ref[:, :1])), -1) + relative_nominal
        omega = rotate_planar(self.root_states[:, 10:13], ts.cart_yaw, inverse=True)
        gravity = rotate_planar(self.projected_gravity, -ts.rel_yaw)
        valid = self.dynamics_wrench_valid & (self.episode_length_buf > 0)
        self.reaction_teacher.update(force, moment, torch.cat((ts.cart_v, ts.cart_wz[:, None]), -1),
            self.cmd_ref, velocity - relative_nominal, omega, gravity, cart["mass"], cart["inertia_world"][:, 2, 2],
            robot["mass"], valid)
        # Task-relative, low-pass momentum proxy: no penalty on all raw gait momentum.
        # Filter in fixed world axes: averaging cart-heading samples without
        # transporting their history invents momentum errors during yaw changes.
        omega_reference_w = torch.cat((torch.zeros_like(self.cmd_ref[:, :2]), self.cmd_ref[:, 2:]), -1)
        angular_reference_w = torch.bmm(robot["inertia_world"], omega_reference_w[:, :, None]).squeeze(-1)
        angular_error_w = (robot["angular_momentum"] - angular_reference_w) / (
            robot["mass"][:, None] * self.cfg.rewards.base_height_target)
        velocity_error_w = robot["velocity"] - rotate_planar(robot_reference, ts.cart_yaw)
        error_world = torch.cat((velocity_error_w, angular_error_w), -1)
        self.centroidal_filtered.lerp_(error_world, -math.expm1(-self.dt / rc.momentum_filter_tau_s))
        error = torch.cat((rotate_planar(self.centroidal_filtered[:, :3], ts.cart_yaw, inverse=True),
                           rotate_planar(self.centroidal_filtered[:, 3:], ts.cart_yaw, inverse=True)), -1)
        deadband = error.new_tensor([*rc.momentum_velocity_deadband, *rc.angular_momentum_deadband])
        self.centroidal_cost.copy_(((error.abs() - deadband).clamp_min(0.) /
                                   error.new_tensor([.3, .3, .3, .25, .25, .25])).square().sum(-1).clamp_max(rc.cost_cap) * valid)
        ground = self.ground_contact_wrenches()
        ts.ground_wrench = ground["feet"].sum(1)
        feet = ground["feet"]
        # CoP is defined at the sole plane, not the ankle-link origin. The
        # conservative common support box is centered ahead of that origin.
        sole_local = feet.new_tensor(rc.foot_support_center_m)
        sole_world = self.feet_pos + torch.stack([
            quat_apply(self.feet_quat[:, i], sole_local.expand(self.num_envs, -1)) for i in range(2)], 1)
        moment_at_foot = feet[..., 3:] + torch.linalg.cross(robot["com"][:, None] - sole_world, feet[..., :3])
        foot_force = torch.stack([quat_rotate_inverse(self.feet_quat[:, i], feet[:, i, :3]) for i in range(2)], 1)
        foot_moment = torch.stack([quat_rotate_inverse(self.feet_quat[:, i], moment_at_foot[:, i]) for i in range(2)], 1)
        normal = foot_force[..., 2].clamp_min(1.)
        support = foot_force[..., 2] > 10.
        utilization = foot_force[..., :2].norm(dim=-1) / (rc.support_friction * normal)
        cop = torch.stack((-foot_moment[..., 1], foot_moment[..., 0]), -1) / normal[..., None]
        cop_excess = (cop.abs() / cop.new_tensor(rc.foot_half_size_m) - .9).clamp_min(0.).square().sum(-1)
        self.support_cost.copy_((((utilization - .9).clamp_min(0.).square() + cop_excess) * support).sum(-1)
                                .clamp_max(rc.cost_cap) * ground["valid"])
        costs = torch.stack((self.reaction_teacher.costs["task_wrench"], self.reaction_teacher.costs["reaction_motion"],
                             self.centroidal_cost, self.support_cost), -1)
        self.reaction_episode_sum += costs * valid[:, None]
        self.reaction_episode_samples += valid.float()

    def _reward_task_wrench(self):
        return self.reaction_teacher.costs["task_wrench"].clamp_max(self.task_cfg.reaction.cost_cap) * self.task_cfg.reaction.enabled

    def _reward_reaction_motion(self):
        return self.reaction_teacher.costs["reaction_motion"].clamp_max(self.task_cfg.reaction.cost_cap) * self.task_cfg.reaction.enabled

    def _reward_centroidal_recovery(self):
        return self.centroidal_cost * self.task_cfg.reaction.enabled

    def _reward_support_feasibility(self):
        return self.support_cost * self.task_cfg.reaction.enabled

    def _grasp_thresholds(self):
        tc = self.task_cfg.termination
        thr = self.grasp_base_thresholds.unsqueeze(0) * self.level_params.grasp_threshold_mult * self.grasp_jitter.unsqueeze(1)
        extra = tc.per_kg_n * (self.cart_mass_total - self.cart_base_kg).clamp_min(0.).unsqueeze(1)
        thr = thr.clone()
        thr[:, :3] += extra
        return thr

    def _update_persistence_counters(self):
        ts, tc = self.ts, self.task_cfg.termination
        thr = self._grasp_thresholds()
        pull = (-ts.force_c[..., 0]).clamp_min(0.).max(dim=1).values
        up = ts.force_c[..., 2].clamp_min(0.).max(dim=1).values
        tangential = ts.force_c[..., 1].abs().max(dim=1).values
        torque = ts.torque_norm.max(dim=1).values
        self.overload_dir = torch.stack((pull > thr[:, 0], up > thr[:, 1], tangential > thr[:, 2], torque > thr[:, 3]), -1)
        over = self.overload_dir.any(dim=1)
        self.overload_counter = torch.where(over, self.overload_counter + 1, torch.zeros_like(self.overload_counter))
        lifted = ts.wheel_lift.max(dim=1).values > tc.wheel_lift_m
        self.wheel_lift_counter = torch.where(lifted, self.wheel_lift_counter + 1, torch.zeros_like(self.wheel_lift_counter))

    def _accumulate_diagnostics(self):
        """Per-episode statistics read out at reset: tracking RMSE against the reference, arm use while turning,
        cart-robot yaw offset, action saturation. Once per control step (this callback runs once per step)."""
        ts, tc = self.ts, self.task_cfg
        err = self.cmd_ref_f - self.cart_vel_f                          # same filtered signals as the rewards
        self.track_err2 += err * err
        self.track_vr += self.cart_vel_f * self.cmd_ref_f
        self.track_rr += self.cmd_ref_f * self.cmd_ref_f
        self.track_steps += 1.
        turning = (self.cmd_ref[:, 2].abs() > tc.rewards.turning_min_wz).float()
        self.turn_dq_sum += turning * self.upper_delta_q.abs().mean(dim=1)
        self.turn_steps += turning
        self.rel_yaw_abs_sum += ts.rel_yaw.abs()
        self.sat_steps += (self.hl_raw.abs() > 1.0).float().mean(dim=1)
        q = self.cart_quality
        if q is not None:
            self.response_error2_sum += q.response_error.square() * q.response_ready.float().unsqueeze(1)
            self.response_steps += q.response_ready.float()
            valid = q.ready.float().unsqueeze(1)
            self.quality_steps += q.ready.float()
            self.quality_sums["velocity_band2"] += q.velocity_band.square() * valid
            self.quality_sums["wrench_band2"] += q.wrench_band.square() * valid
            raw_error = torch.cat((ts.cart_v, ts.cart_wz.unsqueeze(1)), -1) - self.cmd_ref
            self.quality_sums["raw_error2"] += raw_error.square() * valid
            self.quality_sums["task_wrench"] += ts.cart_task_wrench * valid
            peaks = {"force_N": ts.cart_force_w.norm(dim=-1), "moment_Nm": ts.cart_moment_w.norm(dim=-1),
                     "cart_tilt_rad": torch.asin(ts.cart_tilt.clamp(0, 1)),
                     "robot_tilt_rad": torch.asin(self.projected_gravity[:, :2].norm(dim=-1).clamp(0, 1))}
            for name, value in peaks.items():
                self.quality_peaks[name] = torch.maximum(self.quality_peaks[name], value)

    # ------------------------------------------------------------------ terminations
    def check_termination(self):
        super().check_termination()            # torso contact + |g_xy| > 0.8 -> terminated_buf; time_out_buf
        ts, tc = self.ts, self.task_cfg.termination
        r = self.term_reason_buf
        r["fall"] = self.terminated_buf.clone()
        r["grasp_overload"] = self.overload_counter >= tc.persist_steps
        r["cart_yaw"] = ts.rel_yaw.abs() > tc.yaw_diff_rad
        r["cart_distance"] = (ts.dist < tc.distance_m[0]) | (ts.dist > tc.distance_m[1])
        r["cart_tilt"] = ts.cart_tilt > math.sin(tc.cart_tilt_rad)
        r["wheel_lift"] = self.wheel_lift_counter >= tc.wheel_lift_persist_steps
        r["blowup"] = ((ts.cart_speed > tc.blowup_cart_speed)
                       | (ts.force_norm_raw.max(dim=1).values > tc.blowup_force_n)
                       | self.dynamics_fault)
        terminated = r["fall"].clone()
        for name in TERMINATION_REASONS[1:]:
            terminated |= r[name]
        self.terminated_buf = terminated
        self.reset_buf = terminated | self.time_out_buf

    # ------------------------------------------------------------------ rewards (HOMIE machinery: scale x dt applied outside)
    # tracking: low-passed cart velocity against the equally low-passed reference (measure_filter_tau_s)
    def _reward_track_cart_vx(self):
        err = torch.square(self.cmd_ref_f[:, 0] - self.cart_vel_f[:, 0])
        return torch.exp(-err / self.task_cfg.rewards.tracking_sigma)

    def _reward_track_cart_vy(self):
        err = torch.square(self.cmd_ref_f[:, 1] - self.cart_vel_f[:, 1])
        return torch.exp(-err / self.task_cfg.rewards.tracking_sigma)

    def _reward_track_cart_wz(self):
        err = torch.square(self.cmd_ref_f[:, 2] - self.cart_vel_f[:, 2])
        return torch.exp(-err / self.task_cfg.rewards.tracking_sigma)

    def _reward_track_progress(self):
        """Linear tracking score per commanded axis: 1 - |v - r| / |r|, clipped to [0, 1] (standing still = 0, on the
        reference = 1, overshoot costs as much as undershoot). Complements the exp terms, which are flat for small
        commands and let "do not move" score 0.9 at 0.1 m/s."""
        rc = self.task_cfg.rewards
        r, v = self.cmd_ref_f, self.cart_vel_f
        active = (r.abs() >= rc.progress_min_cmd).float()
        score = (1. - (v - r).abs() / r.abs().clamp_min(rc.progress_min_cmd)).clamp(0., 1.)
        return (score * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.)

    def _reward_cart_yaw_lead(self):
        """Cart yaw rate closer to the reference than the torso's. Only the arms can earn it (they rotate the cart
        relative to the pelvis) and only in transients: in a steady turn both rates are equal, so it is zero."""
        rc = self.task_cfg.rewards
        ref = self.cmd_ref_f[:, 2]
        e_torso = (ref - self.base_wz_f).abs()
        e_cart = (ref - self.cart_vel_f[:, 2]).abs()
        turning = (ref.abs() > rc.turning_min_wz).float()
        return turning * (e_torso - e_cart).clamp(0., rc.cart_yaw_lead_cap)

    def _reward_stand_still(self):
        zero = (self.cmd_ref.norm(dim=1) < self.task_cfg.actions.idle_gate_speed).float()
        return zero * (self.ts.cart_speed + 0.5 * self.ts.cart_wz.abs() + 0.5 * (~self.ts.feet_contact).float().sum(dim=1))

    def _reward_force_budget(self):
        excess = (self.ts.force_norm - self.ts.budget.unsqueeze(1)).clamp_min(0.)
        return torch.sum(torch.square(excess), dim=1) / torch.square(self.ts.budget)

    def _reward_force_waste(self):
        """Force that cannot move the cart: vertical, and the two hands pulling the bar apart / squeezing it (the
        along-bar component the fingers carry). Net lateral force steers and is left to the budget; differential
        push / pull (the yaw moment) is free."""
        f = self.ts.force_c
        along_bar = f[:, 0, 1] - f[:, 1, 1]
        return (torch.sum(torch.square(f[..., 2]), dim=1) + 0.5 * torch.square(along_bar)) / torch.square(self.ts.budget)

    def _reward_force_rate(self):
        rate = torch.sum(torch.square(self.ts.force_c - self.force_c_prev), dim=(1, 2)) / torch.square(self.ts.budget)
        if self.task_cfg.rewards.force_rate_seed_on_reset:
            # Do not hide a non-finite first sample: the reward machinery must still fail fast.
            finite = torch.isfinite(rate)
            seed = ~self.force_c_prev_valid & finite
            rate = torch.where(seed, torch.zeros_like(rate), rate)
            # Only valid samples establish/update the history. The invalid reward above is
            # still returned to the existing fail-fast check in HomieTaskCore.compute_reward.
            self.force_c_prev.copy_(torch.where(finite[:, None, None], self.ts.force_c, self.force_c_prev))
            self.force_c_prev_valid |= finite
        else:
            self.force_c_prev.copy_(self.ts.force_c)
            self.force_c_prev_valid.fill_(True)
        return rate

    def _quality_reward_ramp(self):
        seconds = float(self.task_cfg.quality.reward_ramp_s)
        return min(1.0, self.common_step_counter * self.dt / seconds) if seconds > 0 else 1.0

    def _reward_cart_ripple(self):
        return self.cart_quality.ripple_cost * self._quality_reward_ramp()

    def _reward_cart_lin_ripple(self):
        return self.cart_quality.lin_ripple_cost * self._quality_reward_ramp()

    def _reward_cart_yaw_ripple(self):
        return self.cart_quality.yaw_ripple_cost * self._quality_reward_ramp()

    def _reward_cart_response_error(self):
        return self.cart_quality.response_cost * self._quality_reward_ramp()

    def _reward_cart_wrench_osc(self):
        return self.cart_quality.wrench_cost * self._quality_reward_ramp()

    def _reward_wrist_saturation(self):
        rc = self.task_cfg.rewards
        tau = self.ts.upper_tau[:, self.wrist_ids].abs()
        return torch.sum(torch.square((tau - rc.wrist_saturation_frac * rc.wrist_tau_max_nm).clamp_min(0.)), dim=1)

    def _reward_grasp_pull_up(self):
        up = self.ts.force_c[..., 2].clamp_min(0.)
        return torch.sum(torch.square((up - self.task_cfg.rewards.pull_up_free_n).clamp_min(0.)), dim=1) / 100.0 ** 2

    def _reward_wheel_lift(self):
        return torch.sum((self.ts.wheel_lift - self.task_cfg.rewards.wheel_lift_free_m).clamp_min(0.), dim=1)

    def _reward_robot_cart_alignment(self):
        excess = (self.ts.rel_yaw.abs() - self.task_cfg.rewards.alignment_deadzone_rad).clamp_min(0.)
        return torch.square(excess)

    def _reward_robot_cart_distance(self):
        lo, hi = self.task_cfg.rewards.distance_deadzone_m
        d = self.ts.dist
        return torch.square((lo - d).clamp_min(0.)) + torch.square((d - hi).clamp_min(0.))

    def _reward_joint_limit_proximity(self):
        margin = self.task_cfg.rewards.joint_limit_margin * self.upper_range
        q = self.dof_pos[:, 12:]
        lower = (self.upper_lo + margin - q).clamp_min(0.)
        upper = (q - (self.upper_hi - margin)).clamp_min(0.)
        return torch.sum((lower + upper) / margin, dim=1)

    def _reward_feet_caster_proximity(self):
        clear = self.task_cfg.rewards.feet_caster_clear_m
        return torch.sum((clear - self.ts.feet_caster_dist).clamp_min(0.) / clear, dim=1)

    def _reward_upper_torque(self):
        return torch.sum(torch.square(self.ts.upper_tau), dim=1)

    # The three action regularisers act on the *unclipped* policy output. On the clipped action a mean parked
    # outside the clip made every sample identical, so both rate terms were zero: that is how v2's arms froze.
    def _reward_hl_action_rate(self):
        return torch.sum(torch.square(self.hl_raw - self.hl_raw_prev), dim=1)

    def _reward_hl_smoothness(self):
        return torch.sum(torch.square(self.hl_raw - 2 * self.hl_raw_prev + self.hl_raw_prev2), dim=1)

    def _reward_hl_action_magnitude(self):
        return torch.sum(torch.square(self.hl_raw), dim=1)

    def _reward_homie_cmd_rate(self):
        return torch.sum(torch.square(self.cmd_homie - self.cmd_homie_prev), dim=1)

    def _reward_homie_cmd_residual(self):
        reference_b = self.reaction_observer.reference_base(self.cmd_ref)
        return torch.sum(torch.square(self.cmd_homie - reference_b), dim=1)

    def _reward_termination(self):
        # HOMIE multiplies every scale by dt; divide it back out so the scale is the penalty per event.
        return self.terminated_buf.float() / self.dt

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.int32)
        self._record_episode_stats(env_ids)
        self._randomize_cart(env_ids)
        self.command_buffer[env_ids] = 0.            # HOMIE starts the episode with a zero command
        self.cmd_homie[env_ids] = 0.
        self.cmd_homie_prev[env_ids] = 0.
        self.cmd_ref[env_ids] = 0.                   # the reference ramps up from rest
        self.cmd_ref_f[env_ids] = 0.
        self.cart_vel_f[env_ids] = 0.
        self.base_wz_f[env_ids] = 0.
        self.upper_delta[env_ids] = 0.
        super().reset_idx(env_ids)                   # HOMIE rebuilds extras["episode"] in here ...
        self._log_episode_diagnostics(env_ids)       # ... so the read-out comes after it and before the buffers are cleared
        self._reset_hl_state(env_ids)
        self._sample_task_command(env_ids)
        if self.command_schedule is not None and self.fixed_command is None:
            # step() decrements before testing due; the reset sample has not executed yet.
            self.cmd_timer[env_ids] += 1

    def _log_episode_diagnostics(self, env_ids):
        ep = self.extras.setdefault("episode", {})
        ep["curriculum_level"] = self.level_tensor.clone()
        lw = self.curriculum.last_window
        ep["curriculum_tracking_window"] = torch.tensor(lw["tracking_frac"] if math.isfinite(lw["tracking_frac"]) else 0., device=self.device)
        for axis, value in zip(("vx", "vy", "wz"), lw["tracking_axes"]):
            ep[f"curriculum_tracking_{axis}"] = torch.tensor(value if math.isfinite(value) else 0., device=self.device)
        ep["curriculum_fail_window"] = torch.tensor(lw["fail_rate"] if math.isfinite(lw["fail_rate"]) else 0., device=self.device)
        for axis, value in zip(("vx", "vy", "wz"), lw["gains"]):
            ep[f"curriculum_gain_{axis}"] = torch.tensor(value if math.isfinite(value) else 0., device=self.device)
        for name in TERMINATION_REASONS:
            ep[f"term_{name}"] = self.term_reason_buf[name][env_ids].float().mean()
        ep["term_time_out"] = self.time_out_buf[env_ids].float().mean()
        overload = self.term_reason_buf["grasp_overload"][env_ids]
        for i, name in enumerate(("pull", "up", "tangential", "torque")):
            ep[f"term_overload_{name}"] = (overload & self.overload_dir[env_ids, i]).float().mean()
        ep["cart_mass_kg"] = self.cart_mass_total[env_ids].mean()
        for index, name in enumerate(("task_wrench", "reaction_motion", "centroidal_recovery", "support_feasibility")):
            ep[f"v5_{name}_cost"] = (self.reaction_episode_sum[env_ids, index] /
                                      self.reaction_episode_samples[env_ids].clamp_min(1.)).mean()
        ep["v5_reaction_sensor_valid"] = self.reaction.sensor_valid[env_ids].mean()
        ep["v5_recovery_gate"] = self.reaction.recovery_gate[env_ids].mean()
        if self.command_schedule is not None:
            for mode, name in enumerate(("random", "stop_go", "turn_reversal")):
                ep[f"command_mode_{name}_frac"] = (self.command_schedule.mode[env_ids] == mode).float().mean()
        if self.idle_correction is not None:
            ep["idle_correction_gate_mean"] = (self.idle_gate_sum[env_ids]
                                               / self.track_steps[env_ids].clamp_min(1.)).mean()
        # tracking RMSE against the reference (m/s, m/s, rad/s) and how the actions were used, per finished episode
        steps = self.track_steps[env_ids].clamp_min(1.)
        rmse = torch.sqrt(self.track_err2[env_ids] / steps.unsqueeze(1))
        ep["track_rmse_vx"], ep["track_rmse_vy"], ep["track_rmse_wz"] = rmse[:, 0].mean(), rmse[:, 1].mean(), rmse[:, 2].mean()
        ep["hl_saturated_frac"] = (self.sat_steps[env_ids] / steps).mean()                  # |raw action| > 1
        ep["hl_abs_cmd"] = self.hl_raw[env_ids, :3].abs().mean()
        ep["hl_abs_arm"] = self.hl_raw[env_ids, 3:].abs().mean()
        turned = (self.turn_steps[env_ids] > 0).float()                                    # masked mean, no GPU sync
        per_env = self.turn_dq_sum[env_ids] / self.turn_steps[env_ids].clamp_min(1.)
        ep["upper_dq_turning_rad"] = (per_env * turned).sum() / turned.sum().clamp_min(1.)
        ep["rel_yaw_abs_rad"] = (self.rel_yaw_abs_sum[env_ids] / steps).mean()
        if self.cart_quality is not None:
            response_counts = self.response_steps[env_ids]
            response_valid = (response_counts > 0).float()
            response_rmse = (self.response_error2_sum[env_ids] / response_counts.clamp_min(1).unsqueeze(1)).sqrt()
            for i, axis in enumerate(("vx", "vy", "wz")):
                ep[f"quality_response_rmse_{axis}"] = (
                    (response_rmse[:, i] * response_valid).sum() / response_valid.sum().clamp_min(1))
            counts = self.quality_steps[env_ids]
            valid = (counts > 0).float()
            for name, values in self.quality_sums.items():
                values = values[env_ids] / counts.clamp_min(1).unsqueeze(1)
                if name.endswith("2"):
                    values = values.clamp_min(0).sqrt()
                axes = ("fx_N", "fy_N", "mz_Nm") if "wrench" in name else ("vx", "vy", "wz")
                for i, axis in enumerate(axes):
                    ep[f"quality_{name.removesuffix('2')}_{axis}"] = (values[:, i] * valid).sum() / valid.sum().clamp_min(1)
            for name, values in self.quality_peaks.items():
                ep[f"quality_peak_{name}"] = values[env_ids].max()
            ep["quality_valid_samples"] = counts.mean()
            ep["quality_reward_ramp"] = torch.tensor(self._quality_reward_ramp(), device=self.device)
        if hasattr(self.ts, "force_norm_raw"):                           # not yet computed at the very first reset
            ep["grasp_force_max_N"] = self.ts.force_norm_raw.max()      # over all envs this step: spikes show up here
            ep["feet_caster_min_dist_m"] = self.ts.feet_caster_dist.min()

    def _record_episode_stats(self, env_ids):
        """Episode-level statistics for the curriculum (GPU accumulators; the CPU sync happens every 50 steps)."""
        env_ids = env_ids[self.episode_length_buf[env_ids] > 0]      # the very first reset is not an episode
        if len(env_ids) == 0:
            return
        rs = self.reward_scales
        length = self.episode_length_buf[env_ids].clamp_min(1).float()
        fracs = []
        for name in ("track_cart_vx", "track_cart_vy", "track_cart_wz"):             # per-axis mean reward fraction
            w = rs.get(name, 0.)
            fracs.append(self.episode_sums[name][env_ids] / (length * w) if w > 0 and name in self.episode_sums else torch.ones_like(length))
        self._track_sum += torch.stack(fracs, -1).sum(dim=0)
        self._gain_vr += self.track_vr[env_ids].sum(dim=0)
        self._gain_rr += self.track_rr[env_ids].sum(dim=0)
        self._gain_steps += self.track_steps[env_ids].sum()
        failed = self.terminated_buf[env_ids]
        self._fail_sum += failed.float().sum()
        self._count += float(len(env_ids))
        self.episodes_finished += len(env_ids)
        self.term_counts += torch.stack([self.term_reason_buf[n][env_ids].float().sum() for n in TERMINATION_REASONS]
                                        + [self.time_out_buf[env_ids].float().sum()])
        interval = max(1, int(self.task_cfg.curriculum.eval_interval_s / self.dt))
        if self.common_step_counter >= self._next_curriculum_eval and self._count.item() >= self.curriculum.window:
            self._next_curriculum_eval = self.common_step_counter + interval
            count = self._count.item()
            tracking_axes = (self._track_sum / count).tolist()
            fail_rate = self._fail_sum.item() / count
            gains = (self._gain_vr / self._gain_rr.clamp_min(1e-6)).tolist()
            cmd_rms = torch.sqrt(self._gain_rr / self._gain_steps.clamp_min(1.)).tolist()
            before = self.curriculum.level
            changed = self.curriculum.record_window(tracking_axes, fail_rate, gains=gains, cmd_rms=cmd_rms)
            self._track_sum.zero_(); self._fail_sum.zero_(); self._count.zero_()
            self._gain_vr.zero_(); self._gain_rr.zero_(); self._gain_steps.zero_()
            if changed:
                self._apply_level(self.curriculum.params)
                print(f"[pushcart-env] curriculum {before} -> {self.curriculum.level} ({int(count)} episodes: tracking "
                      f"vx {tracking_axes[0]:.3f} vy {tracking_axes[1]:.3f} wz {tracking_axes[2]:.3f}, gain "
                      f"vx {gains[0]:.2f} vy {gains[1]:.2f} wz {gains[2]:.2f}, fail rate {fail_rate:.3f})", flush=True)

    def _randomize_cart(self, env_ids):
        n, lp, rz, dev = len(env_ids), self.level_params, self.task_cfg.randomization, self.device
        payload = torch_rand_float(0., lp.payload_max_kg, (n, 1), dev)
        if self.fixed_payload_kg is not None:
            payload = torch.full((n, 1), float(self.fixed_payload_kg), device=dev)
        total = self.cart_base_kg + payload[:, 0]
        com_xy = torch_rand_float(-rz.payload_com_offset_m, rz.payload_com_offset_m, (n, 2), dev)
        self.payload_com[env_ids] = com_xy
        masses = self.cart_default_mass[env_ids].clone()
        masses[:, self.cart_base_body] = total
        self.cart_mass_total[env_ids] = masses.sum(1)
        uncertainty = self.task_cfg.reaction.mass_estimate_error_fraction
        self.cart_mass_estimate[env_ids] = masses.sum(1) * torch_rand_float(1. - uncertainty, 1. + uncertainty, (n,), dev)
        self.cart.set_masses_index(masses=masses, env_ids=env_ids)
        inertias = self.cart_default_inertia[env_ids].clone()
        # Base and a compact payload box, both shifted to their combined COM.
        # Parallel-axis term is m_base*m_payload/(m_base+m_payload) * (|d|² I-ddᵀ).
        d = torch.cat((com_xy, torch.zeros(n, 1, device=dev)), -1)
        eye = torch.eye(3, device=dev).expand(n, -1, -1)
        parallel = d.square().sum(-1)[:, None, None] * eye - d[:, :, None] * d[:, None, :]
        payload_box = torch.diag(torch.tensor([.30**2 + .20**2, .40**2 + .20**2, .40**2 + .30**2], device=dev)) / 12.
        correction = payload[:, :, None] * payload_box + (self.cart_base_kg * payload[:, 0] / total)[:, None, None] * parallel
        inertias[:, self.cart_base_body] += correction.reshape(n, *inertias.shape[2:])
        self.cart.set_inertias_index(inertias=inertias, env_ids=env_ids)
        coms = self.cart_default_com[env_ids].clone()
        coms[:, self.cart_base_body, :2] += (payload / total.unsqueeze(1)) * com_xy
        self.cart.set_coms_index(coms=coms, env_ids=env_ids)
        wd = torch_rand_float(*rz.wheel_damping_range, (n, 1), dev)
        sd = torch_rand_float(*rz.swivel_damping_range, (n, 1), dev)
        self.wheel_damping[env_ids], self.swivel_damping[env_ids] = wd[:, 0], sd[:, 0]
        damping = torch.zeros(n, self.cart.num_joints, device=dev)
        damping[:, self.wheel_joint_ids] = wd
        damping[:, self.swivel_joint_ids] = sd
        self.cart.write_joint_damping_to_sim_index(damping=damping, env_ids=env_ids)
        mu = torch_rand_float(*rz.wheel_friction_range, (n, 1), dev)
        self.wheel_friction[env_ids] = mu[:, 0]
        ids_cpu = env_ids.to(device="cpu", dtype=torch.int32)
        materials = wp.to_torch(self.cart.root_view.get_material_properties())
        materials[ids_cpu, :, :2] = mu.cpu().to(materials.dtype).view(n, 1, 1)
        self.cart.root_view.set_material_properties(wp.from_torch(materials), wp.from_torch(ids_cpu))
        self.grasp_jitter[env_ids] = torch_rand_float(1. - rz.grasp_threshold_jitter, 1. + rz.grasp_threshold_jitter, (n,), dev)

    def _reset_root_states(self, env_ids):
        """Robot and cart move as one rigid system (grasp constraint stays satisfied); casters get random headings."""
        n, rz, dev = len(env_ids), self.task_cfg.randomization, self.device
        yaw = torch_rand_float(-math.pi, math.pi, (n, 1), dev) if rz.init_yaw else torch.zeros(n, 1, device=dev)
        xy = torch_rand_float(-rz.init_xy_m, rz.init_xy_m, (n, 2), dev)
        q_yaw = torch.cat((torch.zeros(n, 2, device=dev), torch.sin(yaw / 2), torch.cos(yaw / 2)), -1)
        for asset in (self.robot, self.cart):
            pose = asset.data.default_root_pose.torch[env_ids].clone()
            pos = quat_apply(q_yaw, pose[:, :3])
            pos[:, :2] += xy
            pos += self.env_origins[env_ids]
            quat = quat_mul(q_yaw, pose[:, 3:7])
            asset.write_root_pose_to_sim_index(root_pose=torch.cat((pos, quat), -1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        joints = torch.zeros(n, self.cart.num_joints, device=dev)
        r = self.level_params.caster_angle_range
        if self.fixed_caster_angle is not None:
            joints[:, self.swivel_joint_ids] = float(self.fixed_caster_angle)
        elif r > 0:
            joints[:, self.swivel_joint_ids] = torch_rand_float(-r, r, (n, len(self.swivel_joint_ids)), dev)
        self.cart.write_joint_position_to_sim_index(position=joints, joint_ids=self.cart_all_joint_ids, env_ids=env_ids)
        self.cart.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints), joint_ids=self.cart_all_joint_ids, env_ids=env_ids)

    def _reset_hl_state(self, env_ids):
        lp, tc, dev, n = self.level_params, self.task_cfg, self.device, len(env_ids)
        for buf in (self.hl_actions, self.hl_actions_prev, self.hl_actions_prev2, self.hl_action_raw_prev,
                    self.hl_raw, self.hl_raw_prev, self.hl_raw_prev2,
                    self.upper_delta_q, self.vel_est, self.actor_hist, self.cart_pose_hist, self.force_c_prev,
                    self.track_err2, self.track_steps, self.track_vr, self.track_rr,
                    self.turn_dq_sum, self.turn_steps, self.rel_yaw_abs_sum, self.sat_steps):
            buf[env_ids] = 0.
        self.overload_counter[env_ids] = 0
        self.wheel_lift_counter[env_ids] = 0
        self.overload_dir[env_ids] = False
        self.stall_timer[env_ids] = 0
        self.push_timer[env_ids] = 0
        self.push_force[env_ids] = 0.
        for buf in (self.cart_pulse_timer, self.cart_pulse_wrench, self.cart_applied_wrench, self.cart_pulse_impulse,
                    self.centroidal_filtered, self.centroidal_cost, self.support_cost,
                    self.reaction_episode_sum, self.reaction_episode_samples):
            buf[env_ids] = 0
        self.reaction_observer.reset(env_ids)
        self.reaction_teacher.reset(env_ids)
        if self.cart_quality is not None:
            self.cart_quality.reset(env_ids)
        self.quality_steps[env_ids] = 0.
        self.response_steps[env_ids] = 0.
        self.response_error2_sum[env_ids] = 0.
        for buf in (*self.quality_sums.values(), *self.quality_peaks.values()):
            buf[env_ids] = 0.
        self.force_c_prev_valid[env_ids] = False
        self.idle_gate_applied[env_ids] = 0.
        self.idle_gate_sum[env_ids] = 0.
        if self.idle_correction is not None:
            self.idle_correction.reset(env_ids)
        if self.command_schedule is not None:
            self.command_schedule.reset(env_ids, lp)
        noise = lp.obs_noise_frac
        self.cart_pose_delay[env_ids] = torch.randint(0, int(round(noise * tc.observations.cart_pose_delay_steps_max)) + 1, (n,), device=dev)
        self.hl_delay[env_ids] = torch.randint(0, int(round(noise * tc.actions.delay_steps_max)) + 1, (n,), device=dev)
        interval = int(tc.randomization.cart_push_interval_s / self.dt)
        self.cart_push_timer[env_ids] = torch.randint(interval // 2, interval + 1, (n,), device=dev)

    def _sample_task_command(self, env_ids):
        lp, cc, dev, n = self.level_params, self.task_cfg.commands, self.device, len(env_ids)
        if self.fixed_command is not None:
            self.cmd_task[env_ids] = torch.tensor(self.fixed_command, device=dev, dtype=torch.float32)
            self.cmd_timer[env_ids] = 10 ** 9
            return
        vx = torch_rand_float(lp.vx_range[0], lp.vx_range[1], (n, 1), dev)
        vy = torch_rand_float(-lp.vy_max, lp.vy_max, (n, 1), dev)
        wz = torch_rand_float(-lp.wz_max, lp.wz_max, (n, 1), dev)
        keep = (torch.rand(n, 1, device=dev) >= cc.zero_prob).float()
        self.cmd_task[env_ids] = torch.cat((vx, vy, wz), -1) * keep
        lo, hi = int(cc.resample_time_s[0] / self.dt), int(cc.resample_time_s[1] / self.dt)
        self.cmd_timer[env_ids] = torch.randint(lo, hi + 1, (n,), device=dev)
        if self.command_schedule is not None:
            commands, steps = self.command_schedule.sample(
                env_ids, lp, self.cmd_task[env_ids], self.cmd_timer[env_ids])
            self.cmd_task[env_ids] = commands
            self.cmd_timer[env_ids] = steps

    # ------------------------------------------------------------------ disturbances
    def _push_robots(self):
        """HOMIE's random push, delivered as a force pulse instead of a root-velocity jump.

        Same schedule (``push_interval_s``) and momentum (robot mass x U[-max_push_vel_xy, +]) as HOMIE, spread
        over ``robot_push_duration_s``; applied at the pelvis by ``_apply_robot_push`` before every step.
        """
        rz = self.task_cfg.randomization
        if rz.robot_push_duration_s <= 0:                 # diagnostics: HOMIE's original root-velocity jump
            self.last_push_step = self.common_step_counter
            return super()._push_robots()
        limit = self.cfg.domain_rand.max_push_vel_xy
        steps = max(1, int(round(rz.robot_push_duration_s / self.dt)))
        self.last_push_step = self.common_step_counter
        dv = torch_rand_float(-limit, limit, (self.num_envs, 2), self.device)
        mass = self.robot.data.body_mass.torch.sum(dim=1)
        self.push_force.zero_()
        self.push_force[:, 0, :2] = mass.unsqueeze(1) * dv / (steps * self.dt)
        self.push_timer[:] = steps

    def _apply_robot_push(self):
        active = (self.push_timer > 0).view(self.num_envs, 1, 1).float()
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.push_force * active, body_ids=self.pelvis_body_id, is_global=True)
        self.push_timer = (self.push_timer - 1).clamp_min(0)

    def inject_cart_wrench(self, force_w, torque_w, duration_s, env_ids=None, offset_w=None):
        """Queue force/couple at cart COM + offset; duration rounds to 5 ms physics steps."""
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None else
               torch.as_tensor(env_ids, device=self.device))
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("cart pulse env_ids must be a one-dimensional integer sequence")
        if ((ids < 0) | (ids >= self.num_envs)).any() or ids.unique().numel() != ids.numel():
            raise ValueError("cart pulse env_ids must be distinct and within the environment batch")
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("cart pulse duration must be finite and positive")
        def vector(value, name):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            if value.shape not in ((3,), (len(ids), 3)) or not torch.isfinite(value).all():
                raise ValueError(f"cart pulse {name} must be finite with shape [3] or [selected_envs,3]")
            return value.expand(len(ids), 3)
        force, torque = vector(force_w, "force"), vector(torque_w, "torque")
        if offset_w is not None:
            offset = vector(offset_w, "offset")
            torque = torque + torch.linalg.cross(offset, force)
        if not (torch.isfinite(force).all() and torch.isfinite(torque).all()):
            raise ValueError("cart pulse must be finite")
        self.cart_pulse_wrench[ids] = torch.cat((force, torque), -1)
        self.cart_pulse_timer[ids] = max(1, int(round(duration_s / self.cfg.sim.dt)))

    def _apply_action(self):
        super()._apply_action()
        if not hasattr(self, "cart_applied_wrench"):
            return
        self.cart_applied_wrench.copy_(torch.where((self.cart_pulse_timer > 0)[:, None], self.cart_pulse_wrench, 0.))
        # Composer torque is about selected body COM, not assembly COM.
        force = self.cart_applied_wrench[:, :3]
        base_com = self.cart.data.body_com_pose_w.torch[:, self.cart_base_body, :3]
        moment = self.cart_applied_wrench[:, 3:] + torch.linalg.cross(self.cart_com_world() - base_com, force)
        self.cart.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force[:, None].contiguous(), torques=moment[:, None].contiguous(),
            body_ids=torch.tensor([self.cart_base_body], device=self.device, dtype=torch.int32), is_global=True)
        self.cart_pulse_impulse += self.cart_applied_wrench * self.cfg.sim.dt
        self.cart_pulse_timer.sub_(1).clamp_(min=0)

    def _apply_cart_disturbances(self):
        frac = self.level_params.disturbance_frac
        rz, dev, N = self.task_cfg.randomization, self.device, self.num_envs
        # Random physical force/couple pulses; manual injections also work at level zero.
        self.cart_push_timer -= 1
        kick = (self.cart_push_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(kick) and frac > 0 and rz.cart_force_pulse:
            forces = torch.zeros(len(kick), 3, device=dev)
            torques = torch.zeros_like(forces)
            forces[:, :2] = torch_rand_float(-rz.cart_force_max_n * frac, rz.cart_force_max_n * frac, (len(kick), 2), dev)
            torques[:, 2] = torch_rand_float(-rz.cart_torque_max_nm * frac, rz.cart_torque_max_nm * frac, (len(kick),), dev)
            offset = torch_rand_float(-rz.cart_force_offset_m, rz.cart_force_offset_m, (len(kick), 3), dev)
            self.inject_cart_wrench(forces, torques, rz.cart_force_duration_s, kick, offset)
            interval = int(rz.cart_push_interval_s / self.dt)
            self.cart_push_timer[kick] = torch.randint(int(interval * 0.8), int(interval * 1.2) + 1, (len(kick),), device=dev)
        # single-wheel stall: multiply the stalled wheel's *joint damping* for a short time. The damping is part of
        # PhysX's implicit joint drive and therefore unconditionally stable. (An earlier version applied the brake
        # as an explicit effort target -tau = k * omega held over the 20 ms control step; for an airborne or
        # slipping wheel, I = 7e-4 kg m^2, k * dt / I > 2 -> the wheel spin flipped sign and grew every step, and on
        # touch-down the friction launched the cart: the 3.5 / 9.4 kN ... 1e5 N palm forces of the first training
        # run. Attribution runs attr_A..H (4096 env x 30 iter, level 12): every variant with the brake on peaked at
        # 1e5-8e5 N regardless of collision filter / solver iterations / push type; brake off -> 319 N. The damping
        # version (attr_I) reproduces the brake-off numbers: max 320 N, 0 blow-ups.)
        ending = (self.stall_timer == 1).nonzero(as_tuple=False).flatten()
        if len(ending):
            self.cart.write_joint_damping_to_sim_index(
                damping=self.wheel_damping[ending].unsqueeze(1).expand(-1, 4).contiguous(),
                joint_ids=self.wheel_joint_ids_sim, env_ids=ending.to(torch.int32))
        self.stall_timer = (self.stall_timer - 1).clamp_min(0)
        if frac <= 0:
            return
        start = (self.stall_timer <= 0) & (torch.rand(N, device=dev) < rz.wheel_stall_prob_per_s * self.dt * frac)
        if start.any():
            ids = start.nonzero(as_tuple=False).flatten()
            self.stall_timer[ids] = max(2, int(rz.wheel_stall_duration_s / self.dt))
            self.stall_wheel[ids] = torch.randint(0, 4, (len(ids),), device=dev)
            factor = 1.0 + (rz.wheel_stall_factor - 1.0) * frac
            damping = self.wheel_damping[ids].unsqueeze(1).expand(-1, 4).clone()
            damping[torch.arange(len(ids), device=dev), self.stall_wheel[ids]] *= factor
            self.cart.write_joint_damping_to_sim_index(damping=damping, joint_ids=self.wheel_joint_ids_sim, env_ids=ids.to(torch.int32))

    # ------------------------------------------------------------------ observations
    def _update_hl_observations(self):
        self._compute_task_state()
        ts, tc, N, dev = self.ts, self.task_cfg, self.num_envs, self.device
        lp = self.level_params
        noise = lp.obs_noise_frac
        f = self.num_one_step_obs
        homie_frame = self.obs_buf[:, -f:]
        task_cmd = self.cmd_ref * self.commands_scale3                  # the reference the rewards are computed against
        mass = ((self.cart_mass_estimate - tc.observations.cart_mass_norm[0]) / tc.observations.cart_mass_norm[1]).unsqueeze(1)
        tau = ts.upper_tau * tc.observations.torque_scale
        if noise > 0:
            tau = tau * (1. + tc.observations.upper_torque_noise_frac * noise * (2 * torch.rand_like(tau) - 1))
        est = self.vel_est + tc.observations.est_vel_noise_m_s * noise * (2 * torch.rand_like(self.vel_est) - 1)
        est_scaled = est * self.obs_scales.lin_vel
        diff = torch.stack((self.cmd_homie[:, 0] - est[:, 0], self.cmd_homie[:, 1] - est[:, 1],
                            self.cmd_homie[:, 2] - self.base_ang_vel[:, 2]), -1) * self.commands_scale3
        # cart pose relative to the base: true value into a 3-deep history, per-env delay, then noise
        true_pose = torch.cat((ts.bar_b, ts.rel_yaw.unsqueeze(1)), -1)
        fresh = (self.episode_length_buf == 0)
        self.cart_pose_hist[:, 1:] = self.cart_pose_hist[:, :-1].clone()
        self.cart_pose_hist[:, 0] = true_pose
        self.cart_pose_hist[fresh] = true_pose[fresh].unsqueeze(1)
        delayed = self.cart_pose_hist[torch.arange(N, device=dev), self.cart_pose_delay]
        if noise > 0:
            delayed = delayed + torch.cat((tc.observations.cart_pose_noise_pos_m * torch.randn(N, 3, device=dev),
                                           tc.observations.cart_pose_noise_yaw_rad * torch.randn(N, 1, device=dev)), -1) * noise
        scale = torch.full((N, 1), lp.arm_scale / tc.actions.arm_scale_range[1], device=dev)
        # Virtual wrist axes use the same measured cart heading as the pose
        # estimator; relative-yaw latency/noise must not be bypassed by truth.
        observed_cart_yaw = ts.cart_yaw - ts.rel_yaw + delayed[:, 3]
        force_cart = rotate_planar(ts.cart_force_w, observed_cart_yaw, inverse=True)
        moment_cart = rotate_planar(ts.cart_moment_w, observed_cart_yaw, inverse=True)
        virtual_wrist = torch.cat((force_cart, moment_cart), -1)
        self.reaction_observer.update(virtual_wrist, delayed, est,
            homie_frame[:, 4:7] / self.obs_scales.ang_vel, homie_frame[:, 7:10], self.cmd_ref,
            self.cart_mass_estimate, self.robot_mass_estimate,
            self.dynamics_wrench_valid & (self.episode_length_buf > 0))
        reaction_features = self.reaction.actor_features if tc.reaction.enabled else torch.zeros_like(self.reaction.actor_features)
        frame = torch.cat((homie_frame, task_cmd, mass, tau, est_scaled, diff, delayed,
                           self.hl_actions, scale, reaction_features), -1)
        self.actor_hist[:, :-1] = self.actor_hist[:, 1:].clone()
        self.actor_hist[:, -1] = frame
        self.actor_hist[fresh, :-1] = 0.
        self.hl_obs = self.actor_hist.reshape(N, -1).clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        robot = self.centroidal_state(self.robot)
        ground = self.ground_contact_wrenches()
        ground_total = ground["feet"].sum(1)
        # The task is invariant to absolute world yaw. Keep privileged vectors
        # in cart-heading axes too, rather than mixing unlabeled world/body axes.
        in_cart = lambda value: rotate_planar(value, ts.cart_yaw, inverse=True)
        dynamic_extra = torch.cat((in_cart(robot["linear_momentum"]) / robot["mass"][:, None],
            in_cart(robot["angular_momentum"]) / (robot["mass"][:, None] * .74), in_cart(robot["velocity"]),
            in_cart(ground_total[:, :3]) / 400., in_cart(ground_total[:, 3:]) / 100.,
            (self.dynamics_wrench_valid & ground["valid"])[:, None].float()), -1)
        hand_yaw = ts.cart_yaw[:, None].expand(-1, 2).reshape(-1)
        privileged_force = rotate_planar(ts.force_w.reshape(-1, 3), hand_yaw, inverse=True).reshape(N, 6)
        privileged_moment = rotate_planar(ts.torque_w.reshape(-1, 3), hand_yaw, inverse=True).reshape(N, 6)
        privileged = torch.cat((
            ts.cart_v * self.obs_scales.lin_vel, (ts.cart_wz * self.obs_scales.ang_vel).unsqueeze(1),
            privileged_force / 100., privileged_moment / 10.,
            self.payload_com / max(tc.randomization.payload_com_offset_m, 1e-6),
            self.wheel_friction.unsqueeze(1), (self.wheel_damping / tc.randomization.wheel_damping_range[1]).unsqueeze(1),
            (self.swivel_damping / tc.randomization.swivel_damping_range[1]).unsqueeze(1),
            torch.sin(ts.caster_angles), torch.cos(ts.caster_angles),
            (self.level_tensor / max(self.level_max, 1)).expand(N).unsqueeze(1),
            self.base_lin_vel * self.obs_scales.lin_vel,
            dynamic_extra,
        ), -1)
        # Torque-only failures can leave every reward finite while poisoning critic inputs.
        # Check before clipping, which would otherwise conceal infinities. Actor history is
        # checked in full because every stacked frame is passed to PPO on this step.
        finite = (torch.isfinite(self.actor_hist).all(dim=(1, 2))
                  & torch.isfinite(privileged).all(dim=1))
        if not finite.all():
            bad_ids = (~finite).nonzero(as_tuple=False).flatten().tolist()
            raise FloatingPointError(f"Non-finite push-cart observation in environments {bad_ids}")
        if getattr(self, "idle_correction", None) is not None:
            # These are precisely the actor's measurements, not ts.cart_v or a simulator root velocity.
            # Update once after the control step; the following action uses this history.
            self.idle_correction.update(delayed[:, :3], delayed[:, 3], est,
                                        homie_frame[:, 4:7] / self.obs_scales.ang_vel,
                                        homie_frame[:, 7:10], task_command=self.cmd_task)
        self.hl_critic_obs = torch.cat((self.hl_obs, privileged), -1).clamp(-self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

    # ------------------------------------------------------------------ diagnostics
    def termination_summary(self) -> dict:
        counts = dict(zip(TERMINATION_REASONS + ("time_out",), self.term_counts.tolist()))
        total = max(1.0, sum(counts.values()))
        return {"episodes": self.episodes_finished, "fractions": {k: v / total for k, v in counts.items()}, "counts": counts}
