"""V8: high-level teacher-student on the V7.1 plant with a gated leg residual (docs/push_cart_v8_teacher_student_plan.md)."""

TEACHER_TASK_ID = "Homie-G1-PushCart-V8-Teacher-Direct-v0"
STUDENT_TASK_ID = "Homie-G1-PushCart-V8-Student-Direct-v0"


def register_tasks():
    import sys
    import gymnasium as gym
    if TEACHER_TASK_ID not in gym.registry:
        gym.register(
            id=TEACHER_TASK_ID, entry_point="homie_pushcart_v8.direct_env:PushCartDirectEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "homie_pushcart_v8.lab_config:PushCartEnvCfg",
                    "rsl_rl_cfg_entry_point": "homie_pushcart_v8.lab_config:TeacherRunnerCfg"},
        )
    if STUDENT_TASK_ID not in gym.registry:
        gym.register(
            id=STUDENT_TASK_ID, entry_point="homie_pushcart_v8.direct_env:PushCartDirectEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "homie_pushcart_v8.lab_config:StudentEnvCfg",
                    "rsl_rl_cfg_entry_point": "homie_pushcart_v8.lab_config:StudentRunnerCfg"},
        )
    return sys.argv[1:]
