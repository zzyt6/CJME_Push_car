"""V5 reaction recovery task, isolated from the runnable homie_pushcart baseline.

Nothing in homie_isaaclab is modified. The low-level HOMIE policy, its
configuration, and the original g1.urdf are only read.
"""


def register_tasks():
    """Register the push-cart task through Isaac Lab's --external_callback hook."""
    import sys
    import gymnasium as gym

    task_id = "Homie-G1-PushCart-V51-Direct-v0"
    if task_id not in gym.registry:
        # Entry points stay strings: this callback runs before Kit starts, so nothing from isaaclab / pxr
        # may be imported here (a pre-Kit pxr import breaks every Kit extension afterwards).
        gym.register(
            id=task_id,
            entry_point="homie_pushcart_v51.direct_env:PushCartDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "homie_pushcart_v51.lab_config:PushCartEnvCfg",
                "rsl_rl_cfg_entry_point": "homie_pushcart_v51.lab_config:PushCartRunnerCfg",
            },
        )
    return sys.argv[1:]
