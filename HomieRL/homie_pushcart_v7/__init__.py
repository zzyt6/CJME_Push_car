"""V7: V6 interface on the slip-capable grasp + Coulomb cart plant (see task_config.py)."""


def register_tasks():
    import sys
    import gymnasium as gym
    task_id = "Homie-G1-PushCart-V7-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id, entry_point="homie_pushcart_v7.direct_env:PushCartDirectEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "homie_pushcart_v7.lab_config:PushCartEnvCfg",
                    "rsl_rl_cfg_entry_point": "homie_pushcart_v7.lab_config:PushCartRunnerCfg"},
        )
    return sys.argv[1:]
