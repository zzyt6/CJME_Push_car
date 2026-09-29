"""Isaac Lab registration configs pinned to V3.2, with V6 estimator integration."""
from copy import deepcopy

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .baseline import agent_baseline, config_dict, env_baseline, homie_baseline
from .paths import HOMIE_FROZEN_CHECKPOINT
from .task_config import PushCartTaskCfg, num_actor_obs, num_critic_obs

_SAVED_ENV = env_baseline()
_SAVED_AGENT = agent_baseline()
_TASK = PushCartTaskCfg()


def _physics_cfg():
    saved = _SAVED_ENV["sim"]["physics"]
    return PhysxCfg(**{key: deepcopy(value) for key, value in saved.items() if key != "class_type"})


@configclass
class V6PhysicsCfg(PresetCfg):
    default: PhysxCfg = _physics_cfg()
    physx: PhysxCfg = _physics_cfg()


@configclass
class PushCartEnvCfg(DirectRLEnvCfg):
    decimation = _SAVED_ENV["decimation"]
    episode_length_s = _SAVED_ENV["episode_length_s"]
    action_space = _TASK.env.num_hl_actions
    observation_space = num_actor_obs(_TASK)
    state_space = num_critic_obs(_TASK)
    is_finite_horizon = _SAVED_ENV["is_finite_horizon"]
    seed = _SAVED_ENV["seed"]
    ui_window_class_type = None
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", dt=_SAVED_ENV["sim"]["dt"], gravity=tuple(_SAVED_ENV["sim"]["gravity"]),
        render_interval=_SAVED_ENV["decimation"], physics=V6PhysicsCfg(),
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
class InteractionActorCfg(RslRlMLPModelCfg):
    class_name = "homie_pushcart_v6.interaction_policy:InteractionActor"
    estimator_hidden_dims = [512, 256, 128]


@configclass
class InteractionPpoCfg(RslRlPpoAlgorithmCfg):
    class_name = "homie_pushcart_v6.interaction_ppo:InteractionPPO"
    estimator_learning_rate = 1.0e-3
    estimator_epochs = 5
    estimator_mini_batches = 8
    estimator_max_grad_norm = 1.0
    huber_delta = 1.0


def _model_cfg(name, cls):
    saved = deepcopy(_SAVED_AGENT[name])
    saved.pop("class_name")
    distribution = saved.get("distribution_cfg")
    if isinstance(distribution, dict):
        saved["distribution_cfg"] = RslRlMLPModelCfg.GaussianDistributionCfg(**distribution)
    return cls(**saved)


def _algorithm_cfg():
    saved = deepcopy(_SAVED_AGENT["algorithm"])
    saved.pop("class_name")
    return InteractionPpoCfg(**saved)


@configclass
class PushCartRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = _SAVED_AGENT["class_name"]
    seed = _SAVED_AGENT["seed"]
    device = _SAVED_AGENT["device"]
    num_steps_per_env = _SAVED_AGENT["num_steps_per_env"]
    max_iterations = _SAVED_AGENT["max_iterations"]
    save_interval = _SAVED_AGENT["save_interval"]
    experiment_name = "homie_pushcart_v6"
    run_name = ""
    logger = _SAVED_AGENT["logger"]
    empirical_normalization = _SAVED_AGENT["empirical_normalization"]
    obs_groups = {"actor": ["policy", "estimator"], "critic": ["critic"]}
    clip_actions = _SAVED_AGENT["clip_actions"]
    check_for_nan = _SAVED_AGENT["check_for_nan"]
    actor = _model_cfg("actor", InteractionActorCfg)
    critic = _model_cfg("critic", RslRlMLPModelCfg)
    algorithm = _algorithm_cfg()
