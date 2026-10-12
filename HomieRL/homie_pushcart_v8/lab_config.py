"""Isaac Lab registration configs for V8: the environment, the V3.2 runner / PPO values and the ROA teacher / RMA
student model and algorithm (rma_policy.py). Phase 2 and student RL reuse the ROA task with algorithm overrides."""
from copy import deepcopy

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from homie_pushcart_v7.baseline import agent_baseline, config_dict, env_baseline, homie_baseline
from homie_pushcart_v7.paths import HOMIE_FROZEN_CHECKPOINT

from .task_config import ACTION_DIM, PushCartTaskCfg, num_critic_obs, num_policy_obs

_SAVED_ENV = env_baseline()
_SAVED_AGENT = agent_baseline()
_TASK = PushCartTaskCfg()
EXPERIMENT = "homie_pushcart_v8"


def _physics_cfg():
    saved = _SAVED_ENV["sim"]["physics"]
    return PhysxCfg(**{key: deepcopy(value) for key, value in saved.items() if key != "class_type"})


@configclass
class V8PhysicsCfg(PresetCfg):
    default: PhysxCfg = _physics_cfg()
    physx: PhysxCfg = _physics_cfg()


@configclass
class PushCartEnvCfg(DirectRLEnvCfg):
    decimation = _SAVED_ENV["decimation"]
    episode_length_s = _SAVED_ENV["episode_length_s"]
    action_space = ACTION_DIM
    observation_space = num_policy_obs()
    state_space = num_critic_obs()
    is_finite_horizon = _SAVED_ENV["is_finite_horizon"]
    seed = _SAVED_ENV["seed"]
    ui_window_class_type = None
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", dt=_SAVED_ENV["sim"]["dt"], gravity=tuple(_SAVED_ENV["sim"]["gravity"]),
        render_interval=_SAVED_ENV["decimation"], physics=V8PhysicsCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_SAVED_ENV["scene"]["num_envs"], env_spacing=_SAVED_ENV["scene"]["env_spacing"],
        replicate_physics=_SAVED_ENV["scene"]["replicate_physics"],
        filter_collisions=_SAVED_ENV["scene"]["filter_collisions"],
    )
    homie: dict = homie_baseline()
    task: dict = config_dict(_TASK)
    checkpoint: str = str(HOMIE_FROZEN_CHECKPOINT)
    curriculum_level: int | None = _SAVED_ENV["curriculum_level"]
    enable_randomization: bool = _SAVED_ENV["enable_randomization"]
    log_dir: str | None = None
    allow_robot_cart_contact: bool = _SAVED_ENV["allow_robot_cart_contact"]
    cart_solver_iterations: list[int] | None = _SAVED_ENV["cart_solver_iterations"]


@configclass
class TeacherActorCfg(RslRlMLPModelCfg):
    class_name = "homie_pushcart_v8.teacher_policy:TeacherActor"
    latent_dim = 24
    encoder_hidden_dims = [128, 64]
    leg_gate_init = 1.0                           # tanh residual (2026-10-09): fixed output gate, bound by 0.3 * tanh(u)
    leg_gate_learnable = False
    leg_init_std = 0.1                            # in u: applied ~ 0.3 * 0.1 = 0.03 HOMIE action units at u = 0


def _model_cfg(name, cls):
    saved = deepcopy(_SAVED_AGENT[name])
    saved.pop("class_name")
    distribution = saved.get("distribution_cfg")
    if isinstance(distribution, dict):
        saved["distribution_cfg"] = RslRlMLPModelCfg.GaussianDistributionCfg(**distribution)
    return cls(**saved)


def _algorithm_cfg():
    saved = deepcopy(_SAVED_AGENT["algorithm"])
    saved["entropy_coef"] = 0.005                  # V3.2 / V6 value (V7.1 restored it as well)
    return RslRlPpoAlgorithmCfg(**saved)


@configclass
class TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = _SAVED_AGENT["class_name"]
    seed = _SAVED_AGENT["seed"]
    device = _SAVED_AGENT["device"]
    num_steps_per_env = _SAVED_AGENT["num_steps_per_env"]
    max_iterations = _SAVED_AGENT["max_iterations"]
    save_interval = _SAVED_AGENT["save_interval"]
    experiment_name = EXPERIMENT
    run_name = ""
    logger = _SAVED_AGENT["logger"]
    empirical_normalization = _SAVED_AGENT["empirical_normalization"]
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["critic"]}
    clip_actions = _SAVED_AGENT["clip_actions"]
    check_for_nan = _SAVED_AGENT["check_for_nan"]
    actor = _model_cfg("actor", TeacherActorCfg)
    critic = _model_cfg("critic", RslRlMLPModelCfg)
    algorithm = _algorithm_cfg()


# ---------------------------------------------------------------------------------------------------- ROA teacher / RMA student
_ROA_TASK = PushCartTaskCfg()
_ROA_TASK.v8.emit_history = True                  # the adaptation module trains alongside the teacher (ROA)
_ROA_TASK.v8.upper_drive.mode = "explicit"         # V8.1: upper body as the legs / MuJoCo / robot (Unitree gains)
_ROA_TASK.v8.arm_drive.enabled = True              # V8.1: arm-drive randomisation from the start (V8.0: from 6000)
_ROA_TASK.v8.arm_drive.scale_range = [0.8, 1.2]    # V8.1: explicit PD matches MuJoCo / robot, so the wide 0.5-2 (needed
                                                   # for V8.0's implicit-vs-explicit gap; cost the teacher at 0 kg) shrinks


@configclass
class ROATeacherEnvCfg(PushCartEnvCfg):
    task: dict = config_dict(_ROA_TASK)


@configclass
class RMATeacherActorCfg(RslRlMLPModelCfg):
    class_name = "homie_pushcart_v8.rma_policy:RMATeacherActor"
    latent_dim = 20                               # ROA / Extreme Parkour latent size
    encoder_hidden_dims = [256, 128]              # RMA environment-factor encoder
    leg_gate_init = 1.0
    leg_gate_learnable = False
    leg_init_std = 0.1
    adaptation_channels = 64
    velocity_head = True                          # V8.1: phi also estimates cart vx/vy/wz, robot vx/vy (diagnostics)


@configclass
class V8ROAPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "homie_pushcart_v8.rma_policy:V8ROAPPO"
    roa_lambda_max: float = 0.1                   # V8.1: 0 -> 0.1 over 3000-5000, then 0.1 (cap = Extreme Parkour's;
    roa_start: int = 3000                         # ROA's appendix ramps to 1 over 5000-10000, which degraded the
    roa_ramp: int = 2000                          # V8.0 teacher above ~0.12)
    adapt_interval: int = 20                      # H: every 20th rollout acts on z_phi and only phi is updated
    adapt_lr: float = 5.0e-4
    adapt_only: bool = False                      # True = RMA phase 2 (base policy / encoder frozen)
    adapt_std: float = 0.05                       # phase 2 rollout exploration
    init_checkpoint: str = ""                     # phase 2 / continuation: a trained ROA run checkpoint
    adapt_every_update: bool = True               # also fit phi on every privileged rollout (measured need, 2026-10-11)
    start_update: int = 0                         # continuation: the checkpoint's iteration (lambda / H schedule)
    student_rl: bool = False                      # PPO fine-tuning of the deployable student (phi frozen)
    student_std: float = 0.2                      # its exploration std
    velocity_loss_coef: float = 1.0               # weight of the velocity head's supervision (if the actor has it)


def _roa_algorithm_cfg():
    saved = deepcopy(_SAVED_AGENT["algorithm"])
    saved.pop("class_name")
    saved["entropy_coef"] = 0.005                  # as the first V8 teacher
    return V8ROAPPOCfg(**saved)


@configclass
class ROATeacherRunnerCfg(TeacherRunnerCfg):
    obs_groups = {"actor": ["policy", "privileged", "history"], "critic": ["critic"]}
    actor = _model_cfg("actor", RMATeacherActorCfg)
    algorithm = _roa_algorithm_cfg()
