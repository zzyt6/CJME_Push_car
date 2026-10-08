"""V7: the V6 interface (Actor/estimators unchanged) on a plant where the Dex3 grasp can slip and the cart resists.

Everything V6 kept from V3.2 stays, except what the new grasp/cart physics makes physically wrong:

* grasp (square handle tube): per-hand chain on the handle axis -- slide with load-dependent elasto-plastic
  friction (anchor spring capped at the Coulomb limit),
  roll / tilt / yaw elastic up to the finger capacity and yielding beyond it (episode ends when the hand has
  pivoted past a break angle), pull / lift rigid with capacity termination; a lateral grip offset at reset;
* cart: Coulomb rolling resistance and swivel friction, base-mass error, random caster headings from level 0;
* disturbances: horizontal force bumps at a random deck point (replace V3.2's root-velocity kick, which is an
  infinite impulse through a frictional grasp) and a per-episode slope-equivalent bias force;
* rewards: tracking with a per-axis precision scale; the hand loads split into the minimum-load distribution that
  produces the cart's planar wrench and the internal remainder (penalised, less the load across the tube that
  the friction limit needs), plus grasp margin / plastic slip / drift and an action-clip hinge;
  V3.2's ``force_budget`` / ``force_waste`` / ``grasp_pull_up`` are superseded and switched off.

The Actor (605), cart HIM (462) and force head are unchanged; the Critic appends 30 privileged values.
Physical ranges are read from ``config/cart_params_v7.yaml`` once and saved with every run's env.yaml.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from homie_pushcart.task_config import PushCartTaskCfg as V3TaskCfg

from .baseline import apply_config, baseline_source, config_dict, task_baseline
from .paths import CART_PARAMS_V7_YAML

VERSION = 7
IMPLEMENTATION_REVISION = 10
RELEASE = "7.1"
# Revision 9 (2026-10-08): rolling-dependent caster swivel scrub (cart_physics.swivel_bearing_range /
# swivel_scrub_speed_m_s), V3.2 / V6 tracking kernel and weights, hardware-bounded grasp capacities (yaml).
# Revision-8 runs load through migrate_saved_task() with the physics they were trained on (constant swivel scrub).
# Revision 10 = V7.1 (2026-10-08, docs/push_cart_v7_improvement_plan.md): turn commands up to 0.6 rad/s, no action-clip
# penalty, entropy 0.005 (agent), a push-cart-tuned pelvis-velocity estimator replacing HOMIE's estimate in the policy
# frames, and a random cart yaw damping (steering-authority randomization).
SUPPORTED_REVISIONS = (8, 9, 10)
OBSERVATION_SCHEMA = "v7_v32_121x5_him77x6_critic695_v3"      # v3: critic level slot = per-env difficulty
HOMIE_FRAME = 76
LEGACY_HL_FRAME = 124
HL_FRAME = LEGACY_HL_FRAME - 3
HL_EXTRA_FRAME = HL_FRAME - HOMIE_FRAME
ACTOR_HISTORY = 5
ESTIMATOR_FRAME = HOMIE_FRAME + 1
ESTIMATOR_HISTORY = 6
LEGACY_PRIVILEGED_FRAME = 32
HAND_LABEL_FRAME = 12 + 1
GRASP_PRIVILEGED_FRAME = 2 * 12         # per hand: chain state (slide q / slip rate, roll/tilt/yaw q), friction, stiffness, capacities
CART_PRIVILEGED_FRAME = 6               # rolling coeff, swivel friction, bias force xy, bump force xy (cart axes)
PRIVILEGED_FRAME = LEGACY_PRIVILEGED_FRAME + HAND_LABEL_FRAME + GRASP_PRIVILEGED_FRAME + CART_PRIVILEGED_FRAME

_PHYS = yaml.safe_load(Path(CART_PARAMS_V7_YAML).read_text(encoding="utf-8"))
_G, _W = _PHYS["grasp"], _PHYS["wheels_v7"]


class PushCartTaskCfg(V3TaskCfg):
    version = VERSION
    implementation_revision = IMPLEMENTATION_REVISION
    release = RELEASE
    observation_schema = OBSERVATION_SCHEMA
    baseline_source = {}

    class env(V3TaskCfg.env):
        estimator_history = ESTIMATOR_HISTORY

    class rewards(V3TaskCfg.rewards):
        class scales(V3TaskCfg.rewards.scales):
            grasp_internal = -1.0        # internal hand load (does not move the cart), capacity-normalised, on the
                                         # loads low-passed over internal_filter_tau_s (the sustained part). The
                                         # filtered cost is ~1/4 of the instantaneous one (d 0.7: HOMIE alone 0.33,
                                         # model_1600 mean 0.15), so -1.0 (was -0.5 on raw loads): HOMIE alone ~-0.33 /s
            grasp_margin = -1.0          # finger-limited loads above margin_free_frac of the sampled capacity
            grasp_slip = -0.2            # hand sliding (plastic slip, not elastic deflection) / rolling on the bar
            grasp_drift = -0.5           # hand slipped along the bar from its reset grip, or rolled far
            # raw policy output beyond the action clip: a mean parked outside +-1 makes the executed action constant,
            # kills that dimension's policy gradient and lets the entropy bonus grow its std freely (first V7
            # run: 31 % of deterministic outputs and 45 % of samples beyond the clip, entropy 9 -> 19.6)
            # V7.1: 0 (was -0.2). It priced exactly the saturated corrections that keep V6 on the path when the
            # plant steers ~15 % weaker than in training (MuJoCo bench); entropy goes back to 0.005 with it.
            hl_action_clip = 0.0
        # tracking kernel exp(-(e_i / sigma_i)^2) per axis: vx, vy [m/s], wz [rad/s]. Revision 9 restores V3.2 / V6:
        # exp(-e^2 / 0.1), i.e. sigma = sqrt(0.1) on every axis. Revision 8's (0.15, 0.15, 0.2) has no gradient
        # left once an error exceeds ~2 sigma (wz 0.4 rad/s: exp(-4) = 0.02), exactly in the loaded corners
        # where V7 lost the path; together with weight 2.0 (V6: 3.0) it made giving up a turn cheap.
        tracking_sigma_axes = [0.1 ** 0.5, 0.1 ** 0.5, 0.1 ** 0.5]

    class robot_velocity:
        """V7.1: pelvis-velocity estimator fine-tuned in the push-cart scene (actor head, interaction_policy)."""
        enabled = True                   # emit the robot_velocity_target label (true pelvis velocity x 2, valid)
        replace_homie_estimate = True    # zero HOMIE's estimate (x, y) and the command-minus-estimate (x, y) slots
        valid_after_steps = 6            # the six-frame history is complete

    class grasp:
        normal_force_range = list(_G["normal_force_range"])
        friction_coeff_range = list(_G["friction_coeff_range"])
        load_dependent_friction = bool(_G["load_dependent_friction"])
        friction_update_interval = 1     # control steps between slide-limit (max force) writes (CPU model write)
        slide_stiffness_range = list(_G["slide_stiffness_range"])
        slide_damping_ratio = float(_G["slide_damping_ratio"])
        slide_damping_mass_kg = float(_G["slide_damping_mass_kg"])
        slide_offset_n = float(_G["slide_offset_n"])
        slide_min_n = float(_G["slide_min_n"])
        dynamic_over_static = list(_G["dynamic_over_static"])
        initial_offset_m = float(_G["initial_offset_m"])
        pull_capacity_range = list(_G["pull_capacity_range"])
        lift_capacity_range = list(_G["lift_capacity_range"])
        push_capacity_n = float(_G["push_capacity_n"])
        roll_capacity_range = list(_G["roll_capacity_range"])
        moment_capacity_range = list(_G["moment_capacity_range"])
        rotation_stiffness_range = list(_G["rotation_stiffness_range"])      # tilt / yaw
        roll_stiffness_range = list(_G["roll_stiffness_range"])
        rotation_damping_ratio = float(_G["rotation_damping_ratio"])
        rotation_armature = float(_G["rotation_armature"])
        roll_break_rad = float(_G["roll_break_rad"])
        tilt_break_rad = float(_G["tilt_break_rad"])
        slide_drift_break_m = float(_G["slide_drift_break_m"])
        capacity_mult_start = 2.5        # curriculum: capacities x this at level <= 2, minus 0.1 per level ...
        capacity_mult_floor = 1.0        # ... down to the physical value (level 17)
        margin_free_frac = 0.6           # no grasp_margin cost below 60 % of a capacity
        # grasp_internal acts on the hand loads low-passed with this time constant: the high-level policy (arm
        # offsets through the arm PD at 50 Hz) can shape the sustained internal load, not its gait-frequency
        # (~1.7 Hz) oscillation; an instantaneous penalty was 95 % such oscillation (2026-10-07 diagnosis)
        internal_filter_tau_s = 0.5
        slip_speed_scale_m_s = 0.05      # plastic slip rate (anchor motion)
        rotation_speed_scale_rad_s = 0.5
        rotation_slip_weight = 0.1       # elastic rotation rates are benign; sliding along the tube is not
        drift_free_m = 0.02
        drift_scale_m = 0.05
        rotation_free_rad = 0.05         # beyond the elastic range (capacity / stiffness ~ 0.01-0.08 rad)
        rotation_scale_rad = 0.1

    class cart_physics:
        rolling_coeff_range = list(_W["rolling_coeff_range"])
        breakaway_over_rolling = list(_W["breakaway_over_rolling"])
        wheel_viscous_range = list(_W["wheel_viscous_range"])
        swivel_friction_range = list(_W["swivel_friction_range"])        # tyre scrub at standstill
        swivel_bearing_range = list(_W["swivel_bearing_range"])          # what remains while the wheel rolls
        swivel_scrub_speed_m_s = float(_W["swivel_scrub_speed_m_s"])     # rim speed scale of the scrub decay
        # V7.1 steering-authority randomization: extra cart yaw damping tau = -c wz (floor / caster turning
        # resistance). c = 10 N.m.s/rad lowers the bare steering response by 18-29 % (MuJoCo, 0-25 kg, 0.5 m/s +
        # 0.6 rad/s command); the MuJoCo bench steers ~15 % weaker than PhysX. Physical, so nothing leaks into
        # the observations (scaling HOMIE's yaw command would show up in the HOMIE frames the policy sees).
        yaw_damping_range = [0.0, 10.0]
        swivel_viscous_range = list(_W["swivel_viscous_range"])
        wheel_friction_range = list(_W["wheel_friction_range"])
        base_mass_scale_range = list(_W["base_mass_scale_range"])
        caster_heading_range_rad = math.pi   # every level: real casters start wherever they were left

    class disturbance:
        bump_interval_s = [4.0, 8.0]
        bump_force_max_n = 60.0              # x disturbance_frac; a person / door frame bumping the cart
        bump_duration_s = [0.1, 0.4]
        bump_point_x_m = [-0.45, 0.45]       # application point on the cart (base frame): yaw moment included
        bump_point_y_m = [-0.25, 0.25]
        bump_point_z_m = 0.30
        slope_deg_max = 1.5                  # bias force m g sin(slope) in a random horizontal direction, x frac

    class curriculum_v7:
        """Per-environment continuous difficulty (curriculum.py). Ramps are [d_start, d_end] windows; the V3.2
        level schedule is reproduced with level L <-> d = L / 17."""
        enabled = True
        step_up = 0.03                       # success: no termination and every tracking axis >= success_tracking_frac;
                                             # ~35 successes to d = 1 (~1400 iterations, the V6 pace)
        step_down = 0.06                     # any termination
        success_rmse = [0.15, 0.15, 0.2]     # episode tracking RMSE per axis (vx, vy m/s; wz rad/s) for a success;
                                             # V3.2's 0.8 reward fraction under its kernel meant ~0.15 m/s
        vx_forward_ramp = [0.0, 0.53]        # 0 -> 0.9 m/s   (V3.2: +0.1 per level from L1, cap at L9)
        vx_backward_ramp = [0.0, 0.35]       # 0 -> -0.6 m/s  (cap at L6)
        wz_ramp = [0.18, 0.41]               # 0 -> 0.4 rad/s (L4..L7)
        vy_ramp = [0.41, 0.71]               # 0 -> 0.25 m/s  (L8..L12)
        payload_start_kg = 5.0
        payload_max_kg = 25.0
        payload_ramp = [0.29, 0.76]          # 5 -> 25 kg     (L6..L13)
        capacity_ramp = [0.12, 1.0]          # grasp capacity x2.5 -> x1.0 (L2..L17)
        noise_ramp = [0.41, 0.76]            # observation noise / delay, population mean (L8..L13)
        disturbance_ramp = [0.53, 0.88]      # cart bumps, slope, wheel stalls (L10..L15)
        robot_push_ramp = [0.65, 0.88]       # pelvis push magnitude; V3.2 switched it on at L12
        physics_ramp = [0.0, 0.6]            # V7 physical ranges widen around nominal ...
        physics_width_start = 0.25           # ... from 25 % of the full width at d = 0

    def __init__(self):
        super().__init__()
        apply_config(self, task_baseline())
        self.baseline_source = baseline_source()
        # These options did not exist in V3.2; do not inherit V4 behavior.
        self.commands.sequence_enabled = False
        self.idle_correction.enabled = False
        self.rewards.force_rate_seed_on_reset = False
        # V7 physics replaces these V3.2 mechanisms (the snapshot above re-applied the V3.2 values).
        self.randomization.cart_push_vel_max = 0.0         # velocity kick replaced by force bumps
        self.randomization.grasp_threshold_jitter = 0.0    # capacities are sampled physically instead
        self.randomization.wheel_friction_range = list(_W["wheel_friction_range"])
        self.termination.persist_steps = int(_G["overload_persist_steps"])   # 0.4 s over a finger capacity
        # Tracking weights: revision 9 keeps V3.2 / V6's (1.5, 0.75, 3.0) from the snapshot above -- the precise
        # V6 tracking was trained with them; revision 8 used (2.0, 1.0, 2.0), cutting the steering weight by a third.
        sc = self.rewards.scales
        sc.force_budget, sc.force_waste, sc.grasp_pull_up = 0.0, 0.0, 0.0      # superseded by grasp_internal / margin
        sc.termination = -5.0
        self.curriculum.enabled = False                    # V3.2 global levels: diagnostics only (curriculum_v7)
        # V7.1: turn commands up to 0.6 rad/s (the bench follower's limit; 75 % of HOMIE's range). V3.2 / V6 / V7
        # stopped at 0.4, below what a 1.25 m radius corner at 0.5 m/s needs once any error builds up.
        self.commands.wz_max = 0.6


def num_actor_obs(cfg: PushCartTaskCfg | None = None) -> int:
    """Only the policy observation group; the estimator group is independent."""
    return HL_FRAME * (cfg.env.actor_history if cfg is not None else ACTOR_HISTORY)


def num_critic_obs(cfg: PushCartTaskCfg | None = None) -> int:
    history = cfg.env.actor_history if cfg is not None else ACTOR_HISTORY
    return LEGACY_HL_FRAME * history + PRIVILEGED_FRAME


def num_estimator_obs(cfg: PushCartTaskCfg | None = None) -> int:
    history = cfg.env.estimator_history if cfg is not None else ESTIMATOR_HISTORY
    return ESTIMATOR_FRAME * history


def capacity_mult(level: int, cfg: PushCartTaskCfg) -> float:
    g = cfg.grasp
    return max(g.capacity_mult_floor, g.capacity_mult_start - 0.1 * max(0, int(level) - 2))


def run_config_path(checkpoint_or_run_dir) -> Path | None:
    path = Path(checkpoint_or_run_dir)
    run = path.parent if path.suffix == ".pt" else path
    for candidate in (run / "params" / "env.yaml", run.parent / "params" / "env.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _require_complete(values: dict, required: dict, path: str = "task") -> None:
    for key, expected in required.items():
        name = f"{path}.{key}"
        if key not in values:
            raise ValueError(f"Incomplete saved V7 configuration: missing {name}")
        if isinstance(expected, dict):
            if not isinstance(values[key], dict):
                raise ValueError(f"Saved V7 configuration requires a mapping at {name}")
            _require_complete(values[key], expected, name)


def validate_task_schema(cfg: PushCartTaskCfg) -> None:
    """Reject accidental cross-version loading or a different observation contract."""
    if getattr(cfg, "implementation_revision", None) not in SUPPORTED_REVISIONS:
        raise ValueError(f"V7 supports implementation revisions {SUPPORTED_REVISIONS}; checkpoint migration is not implicit")
    expected = {
        "version": VERSION,
        "observation_schema": OBSERVATION_SCHEMA,
        "baseline_source": baseline_source(),
    }
    for name, value in expected.items():
        if getattr(cfg, name, None) != value:
            raise ValueError(f"V7 requires the pinned {name}; checkpoint migration is not implicit")
    if (cfg.env.actor_history != ACTOR_HISTORY
            or cfg.env.estimator_history != ESTIMATOR_HISTORY
            or cfg.env.num_hl_actions != 18):
        raise ValueError("V7 requires Actor history 5, estimator history 6, and 18 actions")
    if (cfg.commands.sequence_enabled or cfg.idle_correction.enabled
            or cfg.rewards.force_rate_seed_on_reset):
        raise ValueError("V7 retains V3.2 behavior; V4 command/idle/force-reset switches must stay off")
    if cfg.randomization.cart_push_vel_max != 0.0:
        raise ValueError("V7 disturbs the cart with force bumps; the V3.2 velocity kick must stay off")


def migrate_saved_task(task: dict) -> dict:
    """Explicit migration of saved revision-8 / 9 task mappings to the current fields, with the physics they were
    trained on: revision 8 had a constant swivel scrub (the new law with no decay: bearing floor 0, infinite scrub
    speed); revisions 8 / 9 had no yaw damping randomization and no V7.1 velocity head (release 7.0). Their saved
    tracking kernel / weights and every other value are kept as saved."""
    rev = task.get("implementation_revision")
    if rev not in (8, 9):
        return task
    task = dict(task)
    cp = dict(task.get("cart_physics", {}))
    if rev == 8:
        cp.setdefault("swivel_bearing_range", [0.0, 0.0])
        cp.setdefault("swivel_scrub_speed_m_s", 1.0e9)
    cp.setdefault("yaw_damping_range", [0.0, 0.0])            # revisions 8 / 9: no steering randomization
    task["cart_physics"] = cp
    task.setdefault("release", "7.0")
    task.setdefault("robot_velocity", {"enabled": False, "replace_homie_estimate": False, "valid_after_steps": 6})
    return task


def task_cfg_from_run(checkpoint_or_run_dir=None, verbose: bool = True) -> PushCartTaskCfg:
    """New V7 defaults, or a complete saved V7 task; never fall back for a checkpoint."""
    cfg = PushCartTaskCfg()
    if checkpoint_or_run_dir is None:
        return cfg
    source = run_config_path(checkpoint_or_run_dir)
    if source is None:
        raise FileNotFoundError(f"Saved V7 params/env.yaml not found for {checkpoint_or_run_dir}")
    try:
        saved = yaml.load(source.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    except Exception as exc:
        raise ValueError(f"Cannot read saved V7 configuration {source}: {exc}") from exc
    task = saved.get("task") if isinstance(saved, dict) else None
    if not isinstance(task, dict):
        raise ValueError(f"Saved V7 configuration has no task mapping: {source}")
    if task.get("version") != VERSION:
        raise ValueError("V7 requires task.version=7; use the V6 entrypoint for older checkpoints")
    # A restored run may have saved the previous restore path as diagnostic metadata.
    task = {key: value for key, value in task.items() if key != "source_yaml"}
    task = migrate_saved_task(task)
    _require_complete(task, config_dict(cfg))
    apply_config(cfg, task)
    validate_task_schema(cfg)
    cfg.source_yaml = str(source)
    if verbose:
        print(f"[pushcart-v7-cfg] restored pinned V7 task from {source}", flush=True)
    return cfg
