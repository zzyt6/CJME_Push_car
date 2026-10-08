"""Isaac Lab registration configs for the V8 teacher (phase 1): V3.2 runner / PPO values, V8 observation groups."""
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
    leg_gate_init = 0.1
    leg_init_std = 0.05


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


# ---------------------------------------------------------------------------------------------------- phase 2: student
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg  # noqa: E402

_STUDENT_TASK = PushCartTaskCfg()
_STUDENT_TASK.v8.emit_history = True              # the 32-frame deployable history for the adaptation module


@configclass
class StudentEnvCfg(PushCartEnvCfg):
    task: dict = config_dict(_STUDENT_TASK)


@configclass
class StudentActorCfg(RslRlMLPModelCfg):
    class_name = "homie_pushcart_v8.student_policy:StudentActor"
    latent_dim = 24
    freeze_body = True                            # RMA / SteadyTray phase 2: only the adaptation module learns
    adaptation_channels = 64


@configclass
class V8DistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name = "homie_pushcart_v8.distillation:V8Distillation"
    num_learning_epochs = 2
    learning_rate = 5.0e-4
    gradient_length = 1                           # one optimizer step per stored time step (4096 rows)
    max_grad_norm = 1.0
    loss_type = "mse"
    explicit_weight = 1.0
    latent_weight = 1.0
    dagger_std = 0.05                             # exploration noise of the student while it drives (DAgger)
    hl_clip = 1.0                                 # behaviour loss on the applied actions (task actions.clip)
    leg_clip = 0.2                                # (v8.leg_residual.clip)


def _student_model_cfg():
    saved = deepcopy(_SAVED_AGENT["actor"])
    saved.pop("class_name")
    dist = dict(saved["distribution_cfg"])
    dist["init_std"] = 0.05
    saved["distribution_cfg"] = RslRlMLPModelCfg.GaussianDistributionCfg(**dist)
    return StudentActorCfg(**saved)


@configclass
class StudentRunnerCfg(RslRlDistillationRunnerCfg):
    seed = _SAVED_AGENT["seed"]
    device = _SAVED_AGENT["device"]
    num_steps_per_env = _SAVED_AGENT["num_steps_per_env"]
    max_iterations = 3000
    save_interval = 200
    experiment_name = EXPERIMENT
    run_name = ""
    logger = _SAVED_AGENT["logger"]
    obs_groups = {"student": ["policy", "history"], "teacher": ["policy", "privileged"]}
    clip_actions = _SAVED_AGENT["clip_actions"]
    check_for_nan = _SAVED_AGENT["check_for_nan"]
    student = _student_model_cfg()
    teacher = _model_cfg("actor", TeacherActorCfg)
    algorithm = V8DistillationAlgorithmCfg()
    policy = None
