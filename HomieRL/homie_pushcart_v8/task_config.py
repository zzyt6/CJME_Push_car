"""V8: high-level teacher-student training on the V7.1 plant, plus a gated residual on HOMIE's leg actions.

Plan and rationale: docs/push_cart_v8_teacher_student_plan.md.

The plant, rewards, curriculum and randomization are V7.1's (homie_pushcart_v7, implementation revision 10): the V8
configuration subclasses it and keeps its V7 fields unchanged (so the V7 environment it extends validates them), and
adds one ``v8`` section that V8 owns:

* observation frame (126, deployable): HOMIE's noisy 76 + task command 3 + upper-body torque 15 + camera-measured
  relative yaw of the cart 1 (latency, frame rate, noise, dropout) + previous 30 actions + residual scale 1. The cart
  mass, the handle position (not measurable on the robot, user decision 2026-10-08), HOMIE's velocity estimate and every
  cart velocity are NOT in it.
* ``policy`` = 5 frames (630); ``history`` = 32 frames (4032, the adaptation module of the ROA teacher / student);
  ``privileged`` = 62 true quantities (teacher encoder; its first 19 are the explicit physical context: cart velocity,
  cart mass, robot velocity, two hand wrenches); ``critic`` = policy + privileged (692).
* actions 30 = V7's 18 high-level actions + 12 residual leg actions added to HOMIE's output. Leg residual (2026-10-09,
  "tanh" mode): applied = low-pass(range * tanh(u)), range 0.3 HOMIE action units (0.075 rad), first-order filter
  40 ms, penalty on |u| beyond 2 (keeps u where tanh still responds). The first V8 teacher (run 2026-10-08_21-05-17)
  used "clip" mode: clamp(u, +-0.2) with a learnable gate; its outputs drifted to about +-10 (bang-bang at +-0.05 rad),
  docs/push_cart_v8_teacher_student_plan.md section 3.1.
* HOMIE input layer: the velocity HOMIE's leg actor receives comes from a copy of HOMIE's HIM encoder fine-tuned in
  the push-cart scene (supervised on the true pelvis velocity); HOMIE's latent and every HOMIE weight stay frozen.
  MuJoCo oracle (2026-10-08): giving HOMIE's actor the true velocity halved V6's 25 kg course error (0.090 -> 0.046 m).
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from homie_pushcart_v7.task_config import PushCartTaskCfg as V71TaskCfg
from homie_pushcart_v7.task_config import IMPLEMENTATION_REVISION as V71_REVISION

RELEASE = "8.0"
OBSERVATION_SCHEMA_V8 = "v8_frame126_policy5_history32_priv62_critic692_act30_v2"
FRAME_DIM = 126
CAMERA_DIM = 1                         # relative yaw only
POLICY_HISTORY = 5
LONG_HISTORY = 32
PRIVILEGED_DIM = 62
EXPLICIT_DIM = 19                      # cart velocity 3 | cart mass 1 | robot velocity 3 | hand wrenches 12
HL_ACTIONS = 18
LEG_ACTIONS = 12
ACTION_DIM = HL_ACTIONS + LEG_ACTIONS


class PushCartTaskCfg(V71TaskCfg):
    release = RELEASE

    class v8:
        observation_schema = OBSERVATION_SCHEMA_V8
        base_revision = V71_REVISION           # the V7.1 plant / reward base this V8 run is built on
        emit_history = False                   # the 32-frame student history (distillation phase only; 1.6 GB of storage)

        class camera:
            """Relative yaw of the cart from a head camera (e.g. AprilTag on the cart + RealSense D435, as SteadyTray).
            Only the yaw: the handle position is not available on the robot. Estimates pending the actual camera:
            a 30 Hz camera at 50 Hz control updates every 1-2 steps; detection + transport latency 40-100 ms; tag
            orientation error ~1.5 deg at 0.5-1 m; occasional missed detections. Scaled by the curriculum's
            observation-noise level like every V3.2 observation noise."""
            latency_steps = [2, 5]                 # per episode, control steps of 20 ms
            update_period_steps = [1, 2]           # per episode: 50 / 25 Hz effective
            noise_yaw_rad = math.radians(1.5)      # per new measurement
            dropout_prob = 0.03                    # per update: measurement missed, the last one is held
            history_steps = 8                      # ring buffer of true poses (>= max latency + 1)

        class homie_velocity:
            """Fine-tuned copy of HOMIE's velocity estimator feeding HOMIE's own leg actor (input-layer adaptation)."""
            enabled = True
            train = True                           # online supervised fine-tuning (teacher phase); frozen when False
            init_checkpoint = ""                   # "" = HOMIE's frozen weights; else a saved homie_velocity_estimator.pt
            learning_rate = 3.0e-4
            rollout_steps = 24                     # collect this many control steps, then update (one PPO rollout)
            updates_per_rollout = 4
            batch_size = 8192
            max_grad_norm = 1.0
            valid_after_steps = 6                  # HOMIE's six-frame history is complete
            save_every_updates = 200               # = runner save_interval; one update per PPO iteration (24 steps),
                                                   # homie_velocity_estimator_<N>.pt pairs with model_<N>.pt

        class estimate_noise:
            """Retired (2026-10-10 trial for the first student's RL fine-tuning; did not transfer). The fields stay only
            so that saved V8 run configurations load (apply_config is strict); enabling it is rejected."""
            enabled = False
            bias_cart_v = [0.06, 0.05, 0.05]       # vx, vy (m/s), wz (rad/s): uniform +-, per episode
            bias_mass_kg = 10.0
            bias_robot_v = [0.07, 0.05, 0.03]
            bias_hands = 0.15                      # scaled wrench units
            noise_cart_v = [0.03, 0.03, 0.04]      # per step, Gaussian std
            noise_robot_v = [0.03, 0.03, 0.02]
            noise_hands = 0.1

        class arm_drive:
            """Per-episode scale of the upper-body (waist, arms, hands) drive stiffness and damping (2026-10-10).
            The student's estimator read the cart's state from arm compliance that is specific to Isaac's implicit
            drive (MuJoCo uses an explicit PD; the robot differs again): at x0.5 gains the student's 25 kg course
            error rose 0.043 -> 0.062 m while the teacher's stayed 0.032 m. Randomising it forces estimates that
            do not depend on one arm compliance. Log-uniform in scale_range; 1 = HOMIE's gains."""
            enabled = False
            scale_range = [0.5, 2.0]

        class upper_drive:
            """Upper-body (waist + arms) drive (2026-10-12, V8.1). "implicit": PhysX joint drive with HOMIE's gains (V8.0
            and every saved V8.0 run). "explicit": the PD torque kp (q* - q) - kd dq is computed before every 200 Hz
            physics substep, like HOMIE's legs and like the MuJoCo bench / the robot's motor drivers, with Unitree's
            real-robot gains (unitree_rl_gym deploy/deploy_real/configs/g1.yaml, arm_waist_kps / arm_waist_kds); the
            PhysX drive of those joints is switched off. Keys match joint names (first match)."""
            mode = "implicit"
            kp = {"waist": 300.0, "shoulder_pitch": 100.0, "shoulder_roll": 100.0, "shoulder_yaw": 50.0, "elbow": 50.0,
                  "wrist": 20.0}
            kd = {"waist": 3.0, "shoulder_pitch": 2.0, "shoulder_roll": 2.0, "shoulder_yaw": 2.0, "elbow": 2.0,
                  "wrist": 1.0}

        class leg_residual:
            enabled = True
            mode = "tanh"                          # "tanh": range * tanh(u) (smooth bound); "clip": clamp(u, +-clip)
            range = 0.3                            # tanh mode, HOMIE action units; x action_scale 0.25 -> 0.075 rad
            clip = 0.2                             # clip mode (first V8 teacher): +-0.05 rad
            excess_threshold = 2.0                 # tanh mode: |u| above this is penalised (rewards.leg_residual_excess)
            filter_tau_s = 0.04                    # first-order low-pass of the applied residual (0 = off)
            # (initial exploration std and the zero-initialised output rows live in the Actor, see teacher_policy.py)

    class rewards(V71TaskCfg.rewards):
        class scales(V71TaskCfg.rewards.scales):
            # V8 leg residual: kept small and smooth (HOMIE action units, 12 joints)
            leg_residual = -1.0
            leg_residual_rate = -0.5
            leg_residual_excess = -0.2             # tanh mode: sum((|u| - excess_threshold)_+^2)
            # HOMIE's own locomotion regularizers at HOMIE's training weights (homie_isaaclab.config): the residual
            # must not buy cart tracking with a worse gait
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_acc = -2.5e-7
            action_rate = -0.01


def num_policy_obs() -> int:
    return FRAME_DIM * POLICY_HISTORY


def num_history_obs() -> int:
    return FRAME_DIM * LONG_HISTORY


def num_critic_obs() -> int:
    return num_policy_obs() + PRIVILEGED_DIM


def upper_drive_gains(cfg: PushCartTaskCfg, joint_names) -> tuple[list[float], list[float]]:
    """(kp, kd) of v8.upper_drive for the given upper-body joint names (first key contained in the name)."""
    ud = cfg.v8.upper_drive
    kp, kd = [], []
    for name in joint_names:
        key = next((k for k in ud.kp if k in name), None)
        if key is None:
            raise ValueError(f"v8.upper_drive has no gain for joint {name}")
        kp.append(float(ud.kp[key]))
        kd.append(float(ud.kd[key]))
    return kp, kd


def validate_v8(cfg: PushCartTaskCfg) -> None:
    if getattr(cfg, "release", None) != RELEASE or cfg.v8.observation_schema != OBSERVATION_SCHEMA_V8:
        raise ValueError("V8 requires release 8.0 and its observation schema; cross-version loading is not implicit")
    if cfg.implementation_revision != V71_REVISION or cfg.v8.base_revision != V71_REVISION:
        raise ValueError("V8 is built on the V7.1 plant (V7 implementation revision 10)")
    lr = cfg.v8.leg_residual
    if lr.mode not in ("tanh", "clip") or lr.range <= 0 or lr.clip <= 0 or lr.filter_tau_s < 0:
        raise ValueError(f"invalid V8 leg residual: mode {lr.mode!r}, range {lr.range}, clip {lr.clip}")
    if cfg.v8.upper_drive.mode not in ("implicit", "explicit"):
        raise ValueError(f"v8.upper_drive.mode must be implicit or explicit, got {cfg.v8.upper_drive.mode!r}")
    if cfg.v8.estimate_noise.enabled:
        raise ValueError("v8.estimate_noise is retired (kept only so that saved configurations load)")
    cam = cfg.v8.camera
    if max(cam.latency_steps) + 1 > cam.history_steps:
        raise ValueError("camera history must cover the maximum latency")


def run_config_path(checkpoint_or_run_dir) -> Path | None:
    path = Path(checkpoint_or_run_dir)
    run = path.parent if path.suffix == ".pt" else path
    candidate = run / "params" / "env.yaml"
    return candidate if candidate.is_file() else None


def task_cfg_from_run(checkpoint_or_run_dir=None, verbose: bool = True) -> PushCartTaskCfg:
    """New V8 defaults, or the complete saved V8 task of a run."""
    from homie_pushcart_v7.baseline import apply_config
    cfg = PushCartTaskCfg()
    if checkpoint_or_run_dir is None:
        return cfg
    source = run_config_path(checkpoint_or_run_dir)
    if source is None:
        raise FileNotFoundError(f"Saved V8 params/env.yaml not found for {checkpoint_or_run_dir}")
    saved = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    task = saved.get("task") if isinstance(saved, dict) else None
    if not isinstance(task, dict) or task.get("release") != RELEASE:
        raise ValueError(f"{source} is not a V8 run")
    apply_config(cfg, {k: v for k, v in task.items() if k != "source_yaml"})
    if "mode" not in ((task.get("v8") or {}).get("leg_residual") or {}):
        # runs saved before the tanh residual (2026-10-09): hard clip, no filter, no excess penalty
        cfg.v8.leg_residual.mode = "clip"
        cfg.v8.leg_residual.filter_tau_s = 0.0
        cfg.rewards.scales.leg_residual_excess = 0.0
    validate_v8(cfg)
    if verbose:
        print(f"[pushcart-v8-cfg] restored V8 task from {source}", flush=True)
    return cfg


def leg_residual_bound(cfg) -> float:
    """Largest applied leg residual (HOMIE action units)."""
    lr = cfg.v8.leg_residual
    return float(lr.range if lr.mode == "tanh" else lr.clip)


def leg_residual_map(u, cfg):
    """Network output u -> residual target (before the low-pass): range * tanh(u), or clamp(u, +-clip)."""
    lr = cfg.v8.leg_residual
    if lr.mode == "tanh":
        return lr.range * (u.tanh() if hasattr(u, "tanh") else __import__("numpy").tanh(u))
    return u.clamp(-lr.clip, lr.clip) if hasattr(u, "clamp") else __import__("numpy").clip(u, -lr.clip, lr.clip)


def leg_residual_filter_alpha(cfg, dt: float) -> float:
    """First-order low-pass weight per control step (1 = no filter)."""
    tau = float(cfg.v8.leg_residual.filter_tau_s)
    return 1.0 if tau <= 0 else float(1.0 - math.exp(-dt / tau))
