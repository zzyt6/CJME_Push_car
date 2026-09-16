# HOMIE 原作者训练参数与 NVIDIA 官方入口

适用本机 `RLisaac601`、Isaac Sim 6.0.1.0、Isaac Lab 3 和 RSL-RL 5.0.1。

## 原作者的命令和配置

原作者 [HomieRL/README.md](../HomieRL/README.md) 中的命令如下，运行位置是 `HomieRL` 目录，适用于旧 Isaac Gym / Linux 环境：

```bash
python legged_gym/legged_gym/scripts/train.py --task g1 --num_envs 4096 --headless --run_name my_policy --rl_device cuda:0 --sim_device cuda:0
```

源文件：

- [原训练入口](../HomieRL/legged_gym/legged_gym/scripts/train.py)。
- [G1 任务、奖励、课程及训练配置](../HomieRL/legged_gym/legged_gym/envs/g1/g1_29dof_config.py)：`G1RoughCfg` / `G1RoughCfgPPO`。
- [继承的 PPO 默认参数](../HomieRL/legged_gym/legged_gym/envs/base/legged_robot_config.py)：`LeggedRobotCfgPPO`。
- [原算法](../HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py)与[原策略及估计器网络](../HomieRL/rsl_rl/rsl_rl/modules/him_actor_critic.py)。

下表是 G1 子类覆盖后最终生效的训练参数，不能只看基类中的 5000 轮和 20 轮保存间隔。

| 参数 | 原作者值 / 当前默认值 |
| --- | --- |
| 并行环境 `num_envs` | 4096 |
| 最大迭代 `max_iterations` | 100000 |
| 每环境每轮采样 `num_steps_per_env` | 50 |
| 模型保存间隔 `save_interval` | 200 轮 |
| 随机种子 | 1 |
| 算法 / 策略 | HIMPPO / HIMActorCritic |
| actor / critic 隐层 | 均为 `[512, 256, 256]` |
| 激活函数 / 初始动作标准差 | ELU / 1.0 |
| PPO 每轮更新遍数 / 小批次数 | 5 / 4 |
| 初始学习率 / 调度 | 0.001 / adaptive |
| `clip_param` / `desired_kl` | 0.2 / 0.01 |
| `gamma` / `lam` | 0.99 / 0.95 |
| 熵系数 / value loss 系数 | 0.01 / 1.0 |
| 梯度范数上限 / clipped value loss | 1.0 / True |
| 镜像增强 `use_flip` / `symmetry_scale` | True / 1.0 |
| actor 观测 / critic 观测 / 策略动作 | 456 / 79 / 12 |
| actor 历史 | 6 帧，每帧 76 维 |
| 物理 / 控制频率 | 200 Hz / 50 Hz（dt=0.005，decimation=4） |
| episode 长度 | 20 秒（1000 个控制步） |

原作者默认使用 W&B，且项目名和用户名留空。本机接入默认使用 TensorBoard、实验名 `homie_g1`。这是日志配置变化；策略、PPO 超参数、奖励、上肢课程和域随机化默认值均保留。迁移阶段已经做过的物理与重置修正见 [迁移记录](isaac_migration.md)。

## 在本机启动训练

本机已完成项目的 editable 安装。搬移项目或重新创建环境后，先安装本地外部任务包，保留现有 Isaac 和 RSL-RL 依赖：

```powershell
cd E:\CJME_Push_car
& E:\envs\RLisaac601\python.exe -m pip install -e . --no-deps --no-build-isolation
```

项目命令脚本 [train_homie.ps1](../train_homie.ps1) 使用原作者默认参数，并转交 NVIDIA 官方训练脚本：

```powershell
cd E:\CJME_Push_car

# 原作者规模：4096 个环境、100000 轮、每轮 50 步
.\train_homie.ps1 --run_name my_policy --viz none

# 同一配置的显式写法，兼容原作者的设备参数及 g1 名称
.\train_homie.ps1 --task g1 --num_envs 4096 --max_iterations 100000 --run_name my_policy --rl_device cuda:0 --sim_device cuda:0 --viz none

# 短程验证
.\train_homie.ps1 --num_envs 32 --max_iterations 3 --run_name my_check --viz none
```

也可以使用 `run_homie.ps1 train` 或 `python scripts/homie.py train`，三者最终进入同一官方训练入口。

直接调用本机 NVIDIA 的统一入口：

```powershell
conda activate RLisaac601
cd E:\CJME_Push_car
& E:\IsaacLab_601\isaaclab.bat train --rl_library rsl_rl --task Homie-G1-Direct-v0 --external_callback homie_isaaclab.register_tasks --num_envs 4096 --max_iterations 100000 --run_name my_policy --viz none physics=physx
```

`--external_callback` 是本机官方脚本提供的外部任务注册接口，需要保留。`--viz kit` 可请求 Isaac Sim 窗口，`--viz none` 用于无界面训练。旧 `--headless` 在本机仍可用，但官方会提示弃用。

## 修改参数与继续训练

CLI 参数覆盖配置默认值。其余参数使用官方 Hydra 覆盖方式，例如：

```powershell
.\train_homie.ps1 --num_envs 32 --max_iterations 3 --viz none agent.algorithm.learning_rate=0.0005 agent.save_interval=1
```

常用映射：

| HOMIE 参数 | 官方配置 / 参数 |
| --- | --- |
| `--task g1` | `--task Homie-G1-Direct-v0` |
| `--sim_device cuda:0` | `--device cuda:0` |
| `--rl_device cuda:0` | `agent.device=cuda:0` |
| `runner.num_steps_per_env` | `agent.num_steps_per_env` |
| `policy.actor_hidden_dims` | `agent.him_policy.actor_hidden_dims` |
| `algorithm.symmetry_scale` | `agent.algorithm.symmetry_scale` |
| 奖励，例如 `rewards.scales.tracking_base_height` | `env.homie.rewards.scales.tracking_base_height` |

项目脚本可直接接收模型文件路径；会转换为本机官方 resolver 的 `--resume --load_run --checkpoint` 参数。外部/旧目录的模型会先复制到日志根目录中的 `0000_imported_<hash>`，原文件保持不变。

```powershell
.\train_homie.ps1 --num_envs 32 --max_iterations 1000 --run_name continued --checkpoint .\logs\rsl_rl\homie_g1\2026-09-14_15-01-32_official_cli_check\model_2.pt --viz none
```

直接使用 NVIDIA 命令时，本机版本需要显式给出 `--resume` 与运行目录选择器：

```powershell
& E:\IsaacLab_601\isaaclab.bat train --rl_library rsl_rl --task Homie-G1-Direct-v0 --external_callback homie_isaaclab.register_tasks --resume --load_run 2026-09-14_15-01-32_official_cli_check --checkpoint model_2.pt --num_envs 32 --max_iterations 1000 --run_name continued --viz none physics=physx
```

同时恢复 PPO 与 HIM estimator 的优化器状态，并恢复自适应学习率。现代 runner 的 `iter` 使用其自身编号规则；额外的 `homie_updates` 记录累计实际更新次数。旧 HOMIE 的 `model_state_dict` 检查点结构仍可加载。原项目的回放入口仍可用于新模型：

```powershell
.\run_homie.ps1 play --num_envs 4 --steps 300 --checkpoint .\logs\rsl_rl\homie_g1\2026-09-14_15-01-32_official_cli_check\model_2.pt --viz none
```

## 接入结构与输出

实际调用链：

```text
NVIDIA train.py → train_rsl_rl.py
  → gym.make("Homie-G1-Direct-v0") → HomieDirectEnv
  → NVIDIA RslRlVecEnvWrapper
  → RSL-RL 5.0.1 OnPolicyRunner
  → HimPpoAdapter → 原作者 HIMPPO + HIMActorCritic + HIMEstimator
```

- [lab_config.py](../HomieRL/homie_isaaclab/lab_config.py)：将原训练参数提供给官方配置解析器。
- [direct_env.py](../HomieRL/homie_isaaclab/direct_env.py)：继承 DirectRLEnv；基类负责仿真上下文和 seed，复用已迁移的场景与步进逻辑，向外提供 Gymnasium 五项返回值。
- [rsl_rl.py](../HomieRL/homie_isaaclab/rsl_rl.py)：通过 RSL-RL 的 `algorithm.class_name` 扩展点接入原 HIM 算法。重置前的 critic 观测经 extras 传入，防止估计器误用下一 episode 的观测。
- 旧算法以 `homie_rsl_rl` 名称加载；原文件只改了 4 条包内 import，计算逻辑保持原样。环境中的现代 `rsl_rl` 与 NVIDIA 源码无需修改。

运行目录：`logs/rsl_rl/homie_g1/<时间>_<run_name>/`，其中：

- `params/env.yaml` / `params/agent.yaml`：本次训练全部生效配置。
- `model_*.pt`：策略、HIM 网络及两个优化器。
- `events.out.tfevents.*`：TensorBoard 曲线，包含 estimation、swap、actor_symmetry、critic_symmetry 损失。
- `physics_config.json`：实际物理参数。
- `integration.json`：runner/算法来源及观测尺寸。

```powershell
& E:\envs\RLisaac601\python.exe -m tensorboard.main --logdir .\logs\rsl_rl\homie_g1
```

## 验证与边界

8 项 CPU 测试覆盖原配置逐项一致、两个 RSL-RL 包共存、相同随机输入下原算法与适配算法的动作/损失/参数更新完全一致、两个优化器的保存恢复，以及此前的关节/四元数/对称/超时检查。

GPU 检查覆盖已注册任务与官方 wrapper 的 101 步运行、非零接触力、重复查询不推进历史、超时重置及重置前 critic 观测。官方入口完成 32 环境 / 3 轮训练与检查点续训。

另外直接使用 `isaaclab.bat train` 实测了原作者 **4096 环境规模**：仅将迭代次数改为 1，其余训练超参数保留默认值，完成 204800 个环境采样步，保存 `logs/rsl_rl/homie_g1/2026-09-14_15-04-56_author_4096_check/model_0.pt`，退出码 0。该轮采样约 4.19 秒、优化约 1.52 秒，不含首次场景初始化时间，单次测量不代表长期吞吐基准。证据位于 `logs/nvidia_author_4096.log`、其余 `logs/nvidia_*.log`、`logs/nvidia_env_check.json` 及训练目录。

本次接入面向单 GPU、PhysX、状态观测训练。视频录制和 NVIDIA 通用 play 的自动 JIT/ONNX 导出尚未接入；使用上面的项目回放命令。短程验证不表示策略已收敛。

官方参考：[原项目 README](https://github.com/InternRobotics/OpenHomie/blob/main/HomieRL/README.md)、[Isaac Lab 3 统一入口迁移说明](https://isaac-sim.github.io/IsaacLab/develop/source/migration/migrating_to_isaaclab_3-0.html)、[官方 RSL-RL 训练实现](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/scripts/reinforcement_learning/rsl_rl/train_rsl_rl.py)。运行接口以本机源码为准。
