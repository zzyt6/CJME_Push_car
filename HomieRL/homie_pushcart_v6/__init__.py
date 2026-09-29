"""V6: V3.2 physical/control baseline with explicitly added estimators."""


def register_tasks():
    import sys
    import gymnasium as gym
    task_id = "Homie-G1-PushCart-V6-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id, entry_point="homie_pushcart_v6.direct_env:PushCartDirectEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "homie_pushcart_v6.lab_config:PushCartEnvCfg",
                    "rsl_rl_cfg_entry_point": "homie_pushcart_v6.lab_config:PushCartRunnerCfg"},
        )
    return sys.argv[1:]
