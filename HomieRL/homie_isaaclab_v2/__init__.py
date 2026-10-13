"""HOMIE's 27-DoF G1 task on Isaac Lab 3, v2: port-vs-official fixes (see README.md)."""
def register_tasks():
    """Register the external task through Isaac Lab's --external_callback hook."""
    import sys
    import gymnasium as gym

    task_id = "Homie-G1-V2-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="homie_isaaclab_v2.direct_env:HomieDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "homie_isaaclab_v2.lab_config:HomieDirectEnvCfg",
                "rsl_rl_cfg_entry_point": "homie_isaaclab_v2.lab_config:HomieRunnerCfg",
            },
        )
    # The official entrypoint intersects this return value with unparsed Hydra
    # overrides. Keep the user's overrides, including physics=physx.
    return sys.argv[1:]
