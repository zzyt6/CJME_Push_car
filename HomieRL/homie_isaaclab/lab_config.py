"""HOMIE author's settings exposed through Isaac Lab's typed configuration."""
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import PresetCfg

from .config import G1RoughCfg, G1RoughCfgPPO
from .math_utils import class_to_dict

_ENV = G1RoughCfg()
_TRAIN = G1RoughCfgPPO()


def homie_physx_cfg():
    return PhysxCfg(
        solver_type=_ENV.sim.physx.solver_type,
        max_position_iteration_count=_ENV.sim.physx.num_position_iterations,
        max_velocity_iteration_count=_ENV.sim.physx.num_velocity_iterations,
        bounce_threshold_velocity=_ENV.sim.physx.bounce_threshold_velocity,
        gpu_max_rigid_contact_count=_ENV.sim.physx.max_gpu_contact_pairs,
    )


@configclass
class HomiePhysicsCfg(PresetCfg):
    default: PhysxCfg = homie_physx_cfg()
    physx: PhysxCfg = homie_physx_cfg()


@configclass
class HomieDirectEnvCfg(DirectRLEnvCfg):
    decimation = _ENV.control.decimation
    episode_length_s = _ENV.env.episode_length_s
    action_space = _ENV.env.num_actions
    observation_space = _ENV.env.num_observations
    state_space = _ENV.env.num_privileged_obs
    is_finite_horizon = False
    seed = _TRAIN.seed
    ui_window_class_type = None
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", dt=_ENV.sim.dt, gravity=tuple(_ENV.sim.gravity),
        render_interval=_ENV.control.decimation, physics=HomiePhysicsCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_ENV.env.num_envs, env_spacing=_ENV.env.env_spacing,
        replicate_physics=True, filter_collisions=True,
    )
    # Rewards, curriculum, domain randomization, commands, PD gains, and
    # observation layout remain inspectable and overrideable in env.yaml.
    homie: dict = class_to_dict(_ENV)
    enable_randomization: bool = True
    log_dir: str | None = None


@configclass
class HomieAlgorithmCfg:
    class_name: str = "homie_isaaclab.rsl_rl:HimPpoAdapter"
    use_flip: bool = _TRAIN.algorithm.use_flip
    num_learning_epochs: int = _TRAIN.algorithm.num_learning_epochs
    num_mini_batches: int = _TRAIN.algorithm.num_mini_batches
    clip_param: float = _TRAIN.algorithm.clip_param
    gamma: float = _TRAIN.algorithm.gamma
    lam: float = _TRAIN.algorithm.lam
    value_loss_coef: float = _TRAIN.algorithm.value_loss_coef
    entropy_coef: float = _TRAIN.algorithm.entropy_coef
    learning_rate: float = _TRAIN.algorithm.learning_rate
    max_grad_norm: float = _TRAIN.algorithm.max_grad_norm
    use_clipped_value_loss: bool = _TRAIN.algorithm.use_clipped_value_loss
    schedule: str = _TRAIN.algorithm.schedule
    desired_kl: float = _TRAIN.algorithm.desired_kl
    symmetry_scale: float = _TRAIN.algorithm.symmetry_scale
    rnd_cfg: None = None


@configclass
class HomieRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    seed = _TRAIN.seed
    device = "cuda:0"
    num_steps_per_env = _TRAIN.runner.num_steps_per_env
    max_iterations = _TRAIN.runner.max_iterations
    save_interval = _TRAIN.runner.save_interval
    experiment_name = "homie_g1"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    # HIMPPO consumes this original network configuration directly. It must
    # not be translated into the standard RSL-RL MLP actor (which lacks HIM).
    him_policy: dict = class_to_dict(_TRAIN.policy)
    algorithm: HomieAlgorithmCfg = HomieAlgorithmCfg()
