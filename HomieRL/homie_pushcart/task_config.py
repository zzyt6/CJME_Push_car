"""Stage-one push-cart task configuration (high-level policy on top of the frozen HOMIE).

Plain nested classes in HOMIE's BaseConfig style so the whole thing can be exposed as a dict in the Isaac Lab
config (Hydra-overridable) and re-applied with ``_apply_dict``. All lengths in metres, forces in newtons,
angles in radians, times in seconds. Curriculum schedules live in ``curriculum.py``.
"""
from __future__ import annotations

import math

from homie_isaaclab.config import BaseConfig


class PushCartTaskCfg(BaseConfig):
    class env:
        episode_length_s = 20.0
        actor_history = 5                # high-level frames stacked for the actor
        num_hl_actions = 18              # 3 command residuals + 15 upper-body joint corrections
        # reserved switches (stage two); both must stay off for the frozen-HOMIE stage
        enable_height_action = False
        leg_residual_scale = 0.0

    class actions:
        clip = 1.0                                   # raw policy output is clipped to [-clip, clip] before scaling
        cmd_residual_scale = [0.4, 0.3, 0.4]         # m/s, m/s, rad/s per unit action
        cmd_filter_tau_s = 0.15                      # first-order low-pass on the HOMIE command
        homie_cmd_limits = [[-0.8, 1.2], [-0.5, 0.5], [-0.8, 0.8]]   # HOMIE training ranges
        upper_rate_limit_rad_s = 3.0                 # per-step slew limit of the upper-body correction
        arm_scale_range = [0.15, 0.5]                # curriculum start -> end, rad per unit action (14 arm joints)
        waist_scale_range = [0.1, 0.4]               # waist_yaw
        delay_steps_max = 1                          # random 0..N step delay of the high-level action

    class observations:
        cart_pose_noise_pos_m = 0.02
        cart_pose_noise_yaw_rad = math.radians(2.0)
        cart_pose_delay_steps_max = 2
        upper_torque_noise_frac = 0.05
        est_vel_noise_m_s = 0.05
        torque_scale = 0.02                          # upper-body torques observed as tau * 0.02 (50 N.m -> 1)
        cart_mass_norm = [12.0, 25.0]                # observed as (m_total - 12) / 25

    class commands:
        resample_time_s = [4.0, 6.0]
        zero_prob = 0.2
        # curriculum end points (75 % of HOMIE's ranges); the schedule ramps towards these
        vx_range_max = [-0.6, 0.9]
        vy_max = 0.35
        wz_max = 0.6

    class rewards:
        class scales:
            track_cart_lin_vel = 2.0
            track_cart_ang_vel = 1.0
            stand_still = -0.3
            force_budget = -0.5
            force_waste = -0.5
            force_rate = -0.2
            wrist_saturation = -1.0
            grasp_pull_up = -1.0
            wheel_lift = -2.0
            robot_cart_alignment = -0.3
            robot_cart_distance = -0.5
            joint_limit_proximity = -2.0
            feet_caster_proximity = -1.0             # feet near the two near-side casters (no physical contact in sim)
            upper_torque = -1.0e-4
            hl_action_rate = -0.01
            hl_smoothness = -0.005
            homie_cmd_rate = -0.05
            homie_cmd_residual = -0.02
            termination = -5.0
        tracking_sigma = 0.25
        force_budget_base_n = 30.0
        force_budget_per_kg = 2.0                    # budget = base + per_kg * cart total mass
        wrist_tau_max_nm = 5.0
        wrist_saturation_frac = 0.8
        pull_up_free_n = 20.0
        wheel_lift_free_m = 0.002
        distance_deadzone_m = [0.2, 0.5]             # pelvis -> handle bar, horizontal
        joint_limit_margin = 0.05                    # fraction of the upper-body joint range
        feet_caster_clear_m = 0.12                   # horizontal foot-caster distance below which the proximity penalty starts
        force_clip_n = 400.0                         # grasp forces are clipped here before any penalty / observation

    class termination:
        # finger-borne grasp directions only (cart frame): pulling the bar, lifting it, sliding along it, twisting
        pull_n = 80.0
        up_n = 60.0
        tangential_n = 50.0
        torque_nm = 10.0
        per_kg_n = 1.0                               # thresholds grow with cart total mass
        persist_steps = 10                           # 0.2 s at 50 Hz
        yaw_diff_rad = math.radians(45.0)
        distance_m = [0.15, 0.6]
        cart_tilt_rad = math.radians(15.0)
        wheel_lift_m = 0.03
        wheel_lift_persist_steps = 10
        fall_gravity_xy = 0.8                        # HOMIE's own fall test (|g_xy| > 0.8, about 53 deg)
        # numerical blow-up guard: ends the episode immediately so one solver event cannot poison a rollout
        blowup_cart_speed = 3.0
        blowup_force_n = 2000.0

    class randomization:
        init_xy_m = 0.5                              # whole robot+cart system: random planar offset
        init_yaw = True                              # ... and random heading (uniform)
        payload_com_offset_m = 0.10
        wheel_friction_range = [0.4, 1.2]
        wheel_damping_range = [0.002, 0.02]
        swivel_damping_range = [0.02, 0.15]
        grasp_threshold_jitter = 0.2
        cart_push_interval_s = 5.0
        cart_push_vel_max = 0.15
        wheel_stall_duration_s = 0.3
        wheel_stall_factor = 20.0
        wheel_stall_prob_per_s = 0.1
        robot_push = True                            # HOMIE's push schedule / magnitude, opened by the curriculum ...
        robot_push_duration_s = 0.25                 # ... but delivered as a pelvis force pulse carrying the same momentum
                                                     # (a finite-duration shove is the physical event; HOMIE's original
                                                     # root-velocity jump measured only ~275 N at the palms because the
                                                     # arm PD chain is compliant, so this is realism, not a stability fix)
        # wheel stall = the stalled wheel's joint damping x (1 + (wheel_stall_factor - 1) * disturbance_frac), i.e. an
        # implicit PhysX drive term (stable). Never apply a brake as an explicit effort target: see task_env.py.

    class curriculum:
        enabled = True
        num_levels = 18
        init_level = 0
        window_episodes = 50
        eval_interval_s = 10.0                   # a window is judged at most this often (HOMIE: once per episode length)
        advance_tracking_frac = 0.7
        advance_max_fail_rate = 0.05
        demote_after_windows = 2


# Observation layout helpers (kept next to the config so the Lab config and the env agree).
HOMIE_FRAME = 76                      # HOMIE's own noisy one-step observation, reused verbatim
HL_EXTRA_FRAME = 3 + 1 + 15 + 3 + 3 + 4 + 18 + 1   # task cmd, cart mass, upper torques, est vel, cmd-est, cart pose, last action, scale
HL_FRAME = HOMIE_FRAME + HL_EXTRA_FRAME             # 124
PRIVILEGED_FRAME = 3 + 12 + 2 + 1 + 1 + 1 + 8 + 1 + 3   # cart vel, grasp wrench, payload com, frictions/dampings, casters, level, base vel


def num_actor_obs(cfg: PushCartTaskCfg | None = None) -> int:
    history = cfg.env.actor_history if cfg is not None else PushCartTaskCfg.env.actor_history
    return HL_FRAME * history


def num_critic_obs(cfg: PushCartTaskCfg | None = None) -> int:
    return num_actor_obs(cfg) + PRIVILEGED_FRAME
