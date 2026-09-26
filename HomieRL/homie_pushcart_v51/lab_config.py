"""Isaac Lab typed configuration for the push-cart task (registered as ``Homie-G1-PushCart-V51-Direct-v0``).

``PushCartEnvCfg`` carries HOMIE's low-level profile (``homie``) and the stage-one task settings (``task``)
as plain dicts, so both are inspectable and overridable through Hydra (``env.task.rewards.scales...=``).
``PushCartRunnerCfg`` is RSL-RL 5 standard PPO with an asymmetric actor / critic.
"""
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from homie_isaaclab.config import G1RoughCfg
from homie_isaaclab.lab_config import HomiePhysicsCfg
from homie_isaaclab.math_utils import class_to_dict

from .task_config import PushCartTaskCfg, num_actor_obs, num_critic_obs

_ENV = G1RoughCfg()
_TASK = PushCartTaskCfg()


@configclass
class PushCartEnvCfg(DirectRLEnvCfg):
    decimation = _ENV.control.decimation
    episode_length_s = _TASK.env.episode_length_s
    action_space = _TASK.env.num_hl_actions
    observation_space = num_actor_obs(_TASK)
    state_space = num_critic_obs(_TASK)
    is_finite_horizon = False
    seed = 1
    ui_window_class_type = None
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", dt=_ENV.sim.dt, gravity=tuple(_ENV.sim.gravity),
        render_interval=_ENV.control.decimation, physics=HomiePhysicsCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=_ENV.env.env_spacing, replicate_physics=True, filter_collisions=True,
    )
    homie: dict = class_to_dict(_ENV)          # low-level profile (PD gains, HOMIE randomisation, noise, ...)
    task: dict = class_to_dict(_TASK)          # stage-one task: actions, rewards, terminations, curriculum
    checkpoint: str | None = None              # frozen HOMIE model_*.pt; None = paths.HOMIE_FROZEN_CHECKPOINT (model_60000)
    curriculum_level: int | None = None        # start level; None = task.curriculum.init_level
    enable_randomization: bool = True          # HOMIE's robot randomisation + observation noise
    log_dir: str | None = None
    # diagnostics only (attribution of numerical blow-ups); leave at the defaults for training
    allow_robot_cart_contact: bool = False     # undo the asset-level robot <-> cart collision filter
    cart_solver_iterations: list[int] | None = None   # override cart_params.solver [position, velocity]


@configclass
class InteractionActorCfg(RslRlMLPModelCfg):
    class_name = "homie_pushcart_v51.interaction_policy:InteractionActor"
    frame_dim = 124
    history_length = 20
    latent_dim = 16
    encoder_hidden_dims = [256, 128]
    interaction_hidden_dims = [64]


@configclass
class InteractionPpoCfg(RslRlPpoAlgorithmCfg):
    class_name = "homie_pushcart_v51.interaction_ppo:InteractionPPO"
    aux_coef = 0.1
    huber_delta = 1.0


@configclass
class PushCartRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    seed = 1
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "homie_pushcart_v51"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = None
    actor = InteractionActorCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = InteractionPpoCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )
