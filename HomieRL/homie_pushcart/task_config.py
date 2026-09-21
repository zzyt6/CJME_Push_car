"""Stage-one push-cart task configuration (high-level policy on top of the frozen HOMIE).

Plain nested classes in HOMIE's BaseConfig style so the whole thing can be exposed as a dict in the Isaac Lab
config (Hydra-overridable) and re-applied with ``_apply_dict``. All lengths in metres, forces in newtons,
angles in radians, times in seconds. Curriculum schedules live in ``curriculum.py``.

Third revision (docs/push_cart_rl_plan_v3.md): the command residual is a feed-forward trim on HOMIE (x / y as
in v2, wz wide enough to reach HOMIE's full yaw range from any task command), the arms have a fixed scale and
two jobs the residual cannot do (isolate the cart from gait / reaction transients, lead the torso in yaw
transients), the cart follows a first-order *reference* instead of the raw step command, and the whole
correction fades out at zero command. ``task_cfg_from_run`` restores the configuration a checkpoint was
trained with, so v2 and v3 checkpoints can be run side by side in MuJoCo.
"""
from __future__ import annotations

import math
from pathlib import Path

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
        # The raw policy output is clipped here before scaling: the residual is a trim (cannot invert the command)
        # and the arm offsets stay within their design range. Safe with a tight clip because the three action
        # regularisers act on the *unclipped* output (v2 penalised the clipped action, so parking the mean outside
        # the clip was free and the arms froze there; the first v3 run with clip 3.0 saturated 30 % of outputs).
        clip = 1.0
        # m/s, m/s, rad/s per unit action. x / y unchanged from v2; wz 0.4 -> 0.5 so |a| <= 1 already reaches
        # HOMIE's 0.8 rad/s from any task command (HOMIE turns at about half its command with the cart attached).
        cmd_residual_scale = [0.4, 0.3, 0.5]
        cmd_filter_tau_s = 0.15                      # first-order low-pass on the HOMIE command
        homie_cmd_limits = [[-0.8, 1.2], [-0.5, 0.5], [-0.8, 0.8]]   # HOMIE training ranges
        upper_rate_limit_rad_s = 3.0                 # per-step slew limit of the upper-body correction
        # rad per unit action; start == end: the curriculum no longer changes the correction scale (the ``scale``
        # observation stays in the layout and is constant 1.0)
        arm_scale_range = [0.4, 0.4]                 # 14 arm joints
        waist_scale_range = [0.3, 0.3]               # waist_yaw
        # |reference command| below this fades the residual and the arm correction to zero (linear ramp), so a
        # standing robot is exact HOMIE passthrough: no exploration noise can push the cart (bench: 0.41 m drift)
        idle_gate_speed = 0.05
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
        # curriculum end points = what the robot + cart can physically do, so every commanded target is trackable
        # and the curriculum gates need no per-axis excuses. vx: 75 % of HOMIE. wz: HOMIE with the cart delivers
        # 0.38-0.44 rad/s at its 0.8 rad/s command limit (MuJoCo bench); the residual supplies the command
        # headroom, the arms the transient. vy: passthrough crabs at 0.05 of 0.25 commanded, the v2 two-stage
        # policy reached 0.28 - reachable, but it has to be learned.
        vx_range_max = [-0.6, 0.9]
        vy_max = 0.25
        wz_max = 0.4
        # The task command is a step; the cart is asked to follow its first-order response (the "velocity
        # curve"). Rewards, curriculum statistics and the task-command observation all use this reference.
        ref_tau_s = 0.4

    class rewards:
        class scales:
            track_cart_vx = 1.5                      # split per axis so the curriculum can gate on each
            track_cart_vy = 0.75                     # lateral cart motion through a centre grasp is physically hard
            track_cart_wz = 3.0                      # steering is the point of this layer (HOMIE: x 1.5 / y 1.0 / ang 2.0)
            # linear tracking score 1 - |v - r| / |r| per commanded axis (0..1; overshoot costs like undershoot),
            # averaged over the commanded axes. The exp terms are flat for small commands (standing still scores
            # 0.9 at 0.1 m/s), so at the low curriculum levels "do not move" was within 0.1/s of tracking and the
            # first three v3 runs learned to cancel the command; this term separates them by 1/s at any magnitude.
            track_progress = 1.0
            cart_yaw_lead = 0.5                      # cart tracks the yaw reference better than the torso (arm lead)
            # (no explicit cart-acceleration term: the cart-velocity signal carries ~1 m/s^2 / 2.5 rad/s^2 of caster
            #  contact jitter whenever the cart rolls, so such a term taught the first v3 run not to roll; smoothness
            #  comes from the reference curve, tracking_sigma 0.1 and the raw-action regularisers)
            stand_still = -1.0
            force_budget = -0.5
            force_waste = -0.5
            force_rate = -0.2
            # V3.1 experimental terms: opt in with train --quality_profile; old runs stay unchanged.
            cart_ripple = 0.0
            cart_wrench_osc = 0.0
            wrist_saturation = -1.0
            grasp_pull_up = -1.0
            wheel_lift = -2.0
            robot_cart_alignment = -0.3
            robot_cart_distance = -0.5
            joint_limit_proximity = -0.5
            feet_caster_proximity = -1.0             # feet near the two near-side casters (no physical contact in sim)
            upper_torque = -1.0e-4
            hl_action_rate = -0.01                   # these three act on the *unclipped* policy output
            hl_smoothness = -0.005
            hl_action_magnitude = -0.02              # discourages parking the mean at the clip (|a| = 1 on 18 dims: 0.36/s);
                                                     # -0.05 shrank the std to 0.26 within 250 iterations
            homie_cmd_rate = -0.05
            homie_cmd_residual = -0.005
            termination = -2.0
        tracking_sigma = 0.1                         # exp(-err^2 / sigma): 0.1 m/s -> 0.90, 0.2 -> 0.67, 0.3 -> 0.41
        # The tracking terms compare a first-order low-passed cart velocity with the equally low-passed reference.
        # A rolling cart's measured yaw rate carries 0.18 rad/s RMS of caster contact / gait ripple (MuJoCo, 0.3 m/s):
        # on the raw signal that alone cost 0.8/s of the wz term - more than pushing at 0.3 m/s earns - and the
        # second v3 run learned to cancel the command and stand still. 0.2 s leaves 0.07 rad/s (0.14/s) and still
        # passes a 0.5-1 Hz oscillation of the policy at 60-85 %.
        measure_filter_tau_s = 0.2
        progress_min_cmd = 0.1                       # axes with |reference| below this do not count for track_progress
        alignment_deadzone_rad = 0.35                # |cart yaw - robot yaw| inside this is free (arm steering)
        cart_yaw_lead_cap = 0.3                      # rad/s of lead credited per step at most
        turning_min_wz = 0.1                         # |wz reference| above which the yaw-lead term is active
        force_budget_base_n = 30.0
        force_budget_per_kg = 2.0                    # budget = base + per_kg * cart total mass
        wrist_tau_max_nm = 5.0
        wrist_saturation_frac = 0.8
        pull_up_free_n = 20.0
        wheel_lift_free_m = 0.002
        distance_deadzone_m = [0.2, 0.5]             # pelvis -> handle bar, horizontal
        joint_limit_margin = 0.03                    # fraction of the upper-body joint range
        feet_caster_clear_m = 0.12                   # horizontal foot-caster distance below which the proximity penalty starts
        force_clip_n = 400.0                         # grasp forces are clipped here before any penalty / observation

    class quality:
        enabled = False                            # training diagnostics / filters; no change to actor inputs
        # Initial experimental band, to be checked against full-rate baseline traces before long training.
        low_hz = 0.5
        high_hz = 5.0
        warmup_s = 1.0                             # first valid sample seeds filters; exclude initial settling
        reward_ramp_s = 60.0                       # linear ramp over environment control time after launch
        velocity_scale = [0.05, 0.05, 0.10]        # m/s, m/s, rad/s
        velocity_deadband = [0.005, 0.005, 0.01]
        wrench_scale = [20.0, 20.0, 5.0]           # N, N, N.m; fixed scale, not a desired force
        wrench_deadband = [1.0, 1.0, 0.2]
        wrench_input_clip = [800.0, 800.0, 200.0]  # reward-filter input only; raw telemetry stays raw
        cost_cap = 4.0                             # caps new penalties; tracking/progress remains primary

    class termination:
        # finger-borne grasp directions only (cart frame): pulling the bar, lifting it, sliding along it, twisting
        pull_n = 80.0
        up_n = 60.0
        tangential_n = 50.0
        torque_nm = 10.0
        per_kg_n = 1.0                               # thresholds grow with cart total mass
        persist_steps = 20                           # 0.4 s at 50 Hz (start-up / shove transients do not terminate)
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
        window_episodes = 500                    # 4096 envs finish ~4000 episodes per 20 s: 50 had no statistical weight
        eval_interval_s = 20.0                   # a window is judged at most this often (one episode length)
        # mean per-episode tracking reward fraction each axis (vx, vy, wz) must reach; a strong straight push can
        # no longer hide a weak turn. Uniform: the command ranges above are already the physically reachable ones.
        advance_tracking_frac = [0.8, 0.8, 0.8]
        # ... and the regression gain of the cart velocity on the reference (sum v*r / sum r^2 over the window) each
        # axis must reach: the exp score is an absolute error, so at low levels (commands <= 0.2 m/s) a cart that does
        # not move at all still scores 0.8 - the first v3 run climbed the curriculum that way. Skipped on axes whose
        # reference RMS over the window is below gain_min_cmd_rms (e.g. wz before level 4).
        advance_min_gain = [0.7, 0.7, 0.7]
        gain_min_cmd_rms = 0.1
        advance_max_fail_rate = 0.05
        demote_after_windows = 3


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


# ---------------------------------------------------------------------------- per-run configuration

# Behaviour switches that did not exist when older runs were trained: when a saved config has the section but
# not the key, the *old* behaviour is restored (no reference filter, no idle gate) instead of today's default.
LEGACY_DEFAULTS = {("commands", "ref_tau_s"): 0.0, ("actions", "idle_gate_speed"): 0.0}


def run_config_path(checkpoint_or_run_dir) -> Path | None:
    """``<run>/params/env.yaml`` written by Isaac Lab for a checkpoint or a run directory, if it exists."""
    p = Path(checkpoint_or_run_dir)
    run = p.parent if p.suffix == ".pt" else p
    for candidate in (run / "params" / "env.yaml", run.parent / "params" / "env.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _apply_lenient(target, values: dict, path: list, report: dict):
    for key, value in values.items():
        if not hasattr(target, key):
            report["ignored"].append(".".join(path + [key]))
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and not isinstance(current, (dict, list, tuple)) and hasattr(current, "__dict__"):
            _apply_lenient(current, value, path + [key], report)
        else:
            setattr(target, key, list(value) if isinstance(value, tuple) else value)


def task_cfg_from_run(checkpoint_or_run_dir=None, verbose: bool = True) -> PushCartTaskCfg:
    """The task configuration a checkpoint was trained with (its run's ``params/env.yaml``), or the defaults.

    Used by every MuJoCo-side tool (viewer, bench, export): a checkpoint keeps its own residual scale, action
    clip, arm scale and command reference when the defaults have moved on. Keys that did not exist at training
    time fall back to the legacy behaviour (``LEGACY_DEFAULTS``); keys that no longer exist are ignored.
    """
    cfg = PushCartTaskCfg()
    path = run_config_path(checkpoint_or_run_dir) if checkpoint_or_run_dir is not None else None
    if path is None:
        return cfg
    try:
        import yaml
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.Loader)
        task = data.get("task") if isinstance(data, dict) else None
    except Exception as exc:                                             # noqa: BLE001 - fall back to the defaults
        if verbose:
            print(f"[pushcart-cfg] could not read {path} ({exc}); using the current defaults", flush=True)
        return cfg
    if not isinstance(task, dict):
        return cfg
    report = {"ignored": [], "legacy": []}
    _apply_lenient(cfg, task, [], report)
    for (section, key), value in LEGACY_DEFAULTS.items():
        if section in task and key not in task[section]:
            setattr(getattr(cfg, section), key, value)
            report["legacy"].append(f"{section}.{key}={value}")
    cfg.source_yaml = str(path)
    if verbose:
        extra = "".join(f"; {k}: {', '.join(v)}" for k, v in report.items() if v)
        print(f"[pushcart-cfg] task config from {path}{extra}", flush=True)
    return cfg
