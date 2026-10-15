"""HOMIE's 27-DoF G1 task on Isaac Lab 3, v2: port-vs-official fixes (see README.md)."""
TASKS = {
    # task id: (env cfg, runner cfg)
    "Homie-G1-V2-Direct-v0": ("HomieDirectEnvCfg", "HomieRunnerCfg"),
    "Homie-G1-V2-Balance-Direct-v0": ("HomieBalanceEnvCfg", "HomieBalanceRunnerCfg"),
}


def register_tasks():
    """Register the external tasks through Isaac Lab's --external_callback hook."""
    import sys
    import gymnasium as gym

    for task_id, (env_cfg, runner_cfg) in TASKS.items():
        if task_id not in gym.registry:
            gym.register(
                id=task_id,
                entry_point="homie_isaaclab_v2.direct_env:HomieDirectEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"homie_isaaclab_v2.lab_config:{env_cfg}",
                    "rsl_rl_cfg_entry_point": f"homie_isaaclab_v2.lab_config:{runner_cfg}",
                },
            )
    # The official entrypoint intersects this return value with unparsed Hydra
    # overrides. Keep the user's overrides, including physics=physx.
    return sys.argv[1:]
