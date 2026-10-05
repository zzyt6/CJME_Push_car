"""HOMIE's 27-DoF G1 task adapted to Isaac Lab 3 and Isaac Sim 6.0.1."""
def register_tasks():
    """Register the external task through Isaac Lab's --external_callback hook."""
    import sys
    import gymnasium as gym

    task_id = "Homie-G1-HandForce-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="homie_hand_force.direct_env:HomieDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "homie_hand_force.lab_config:HomieDirectEnvCfg",
                "rsl_rl_cfg_entry_point": "homie_hand_force.lab_config:HomieRunnerCfg",
            },
        )
    # The official entrypoint intersects this return value with unparsed Hydra
    # overrides. Keep the user's overrides, including physics=physx.
    return sys.argv[1:]
