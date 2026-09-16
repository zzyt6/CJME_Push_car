# 本机 RLisaac601 / Isaac Sim / Isaac Lab 运行说明

本项目已接入 NVIDIA 官方统一训练入口，使用原作者 HOMIE 的 HIM-PPO、网络与默认训练参数。使用现有 `RLisaac601` 环境。**原作者命令、完整参数对照及启动方式见 [HOMIE 训练说明](docs/homie_training.md)**；物理迁移记录见 [迁移核对记录](docs/isaac_migration.md)。

## 本机版本

| 组件 | 实际版本 / 位置 |
| --- | --- |
| Python | 3.12.13，`E:\envs\RLisaac601\python.exe` |
| Isaac Sim Python 包 | 6.0.1.0，位于 RLisaac601 环境内，项目实际使用此安装 |
| 本机 Isaac Sim 源码构建 | `E:\isaacsim`，6.0.1-rc.7+main.0.98701505.local |
| Isaac Lab | `E:\IsaacLab_601`，仓库 VERSION 为 3.0.0 |
| Isaac Lab 当前提交 | `2e44ddb2e19536579140496023b5ccb060bc4152`，本机原有 setup.py 修改保留 |
| Isaac Lab Python 包 | isaaclab 6.1.17、isaaclab_physx 1.1.3、isaaclab_assets 0.3.5、isaaclab_tasks 1.10.9、isaaclab_rl 0.5.7 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8；torchvision 0.26.0+cu128 |
| NumPy | 2.3.1 |
| GPU / 驱动 | RTX 4090 24 GB / 610.88 |
| 环境自带 RSL-RL | rsl-rl-lib 5.0.1，本项目使用其官方 OnPolicyRunner |

仓库版本、子包版本和 Isaac Sim 构建版本采用不同编号，不能把 `isaaclab==6.1.17` 当成 Isaac Sim 6.1。此适配针对上表的本机接口，尤其是 Lab 3 的 `ProxyArray.torch`、`*_index` 方法与 **xyzw** 四元数。

## 直接运行

在项目目录的 PowerShell 中执行。脚本会选择 RLisaac601，无需先激活 Conda。

```powershell
cd E:\CJME_Push_car

# 环境、CUDA、HOMIE 算法加载检查
.\run_homie.ps1 doctor

# 物理、接触力、观测、重置检查
.\run_homie.ps1 smoke --num_envs 4 --steps 100 --viz none

# NVIDIA 官方入口 + 原 HOMIE HIM-PPO 短程训练
.\run_homie.ps1 train --num_envs 32 --max_iterations 3 --run_name my_check --viz none

# 原作者默认规模：4096 环境 / 100000 轮
.\train_homie.ps1 --run_name my_policy --viz none
```

Lab 3 默认无界面，`--viz none` 明确关闭可视化；`--viz kit` 开启 Isaac Sim 窗口。不要照搬 Lab 2 的默认 GUI 行为。旧的 `--headless` 仍由本机 AppLauncher 兼容，但会提示弃用。

加载已有检查点：

```powershell
# 这里的模型来自官方入口 3 次迭代验证，尚未训练收敛
.\run_homie.ps1 play --num_envs 4 --steps 300 --checkpoint .\logs\rsl_rl\homie_g1\2026-09-14_15-01-32_official_cli_check\model_2.pt --viz none

# 想观察机器人时，将 --viz none 改为 --viz kit

# 播放训练完成难度的上半身随机姿态；只开启上半身课程，其他物理随机化保持关闭
.\run_homie.ps1 play --num_envs 1 --steps 100000 --checkpoint .\logs\rsl_rl\homie_g1\2026-09-14_15-10-07_my_policy\model_20000.pt --play_upper_randomization --viz kit

# 继续训练，包括加载 PPO 和 HIM estimator 的优化器状态
.\train_homie.ps1 --num_envs 32 --max_iterations 1000 --run_name continued --checkpoint .\logs\rsl_rl\homie_g1\2026-09-14_15-01-32_official_cli_check\model_2.pt --viz none
```

也可以在已激活的环境中运行 `python scripts/homie.py doctor`，或 `python scripts/homie.py train --num_envs 32 --max_iterations 3 --viz none`。迁移机器后，通过 `HOMIE_PYTHON` 指定解释器；适配代码仍要求上述 Isaac 包版本。

## 适配内容

- `scripts/homie.py` 与 `run_homie.ps1` 提供环境检查、仿真检查、训练及回放入口。`train` 已转交本机 NVIDIA `scripts/reinforcement_learning/train.py`；`train_homie.ps1` 是训练快捷脚本。
- `HomieRL/homie_isaaclab/` 使用本机 Isaac Lab 的 PhysX、InteractiveScene、Articulation 和接触传感器。原 `legged_gym` 的 Isaac Gym 训练入口保留。
- 使用项目自己的 `g1.urdf` 和本地 STL，首次运行生成 `.cache/g1_isaaclab/`。地面也在本地创建，不依赖下载示例机器人。
- 原机器人是 **27 个可动关节**：双腿 12、腰 1、双臂 14。策略输出 12 维下肢动作，上肢按原有课程生成目标。文件名中的“29dof”不代表此 URDF 有 29 个活动关节。
- 显式映射 Isaac Lab 的关节排列到原 HOMIE 顺序，保留每步 76 维观测、6 帧历史（456 维 actor）、79 维 critic、HIM estimator、PPO 和对称性损失。
- 按原 URDF 重建无质量 IMU 与 6 个脚底参考点；显式为左右脚及躯干配置接触报告，适配 Isaac Sim 6 的嵌套链接结构。
- 保留速度/高度命令、奖励尺度、上下肢混合控制、上肢课程及配置中启用的随机化。修正了原实现生成动作延迟却没有实际应用、重置后历史残留以及把上肢位置目标当作力矩统计的问题；质量随机化限制为正值并同步缩放惯量。
- 官方训练通过 `Homie-G1-Direct-v0` 注册任务、`RslRlVecEnvWrapper`、RSL-RL 5 的 `OnPolicyRunner` 和 HIM 算法适配器运行。原算法以 `homie_rsl_rl` 命名加载，不覆盖共享环境的 `rsl_rl`。回放/旧物理检查入口仍在独立进程使用旧接口。日志默认 TensorBoard。
- `.vscode/` 已指定本机 Python，并提供 doctor / smoke 调试配置。

原 `HomieRL/requirements.txt` 是 Python 3.8 / Isaac Gym Preview 4 的依赖表。当前任务包通过项目根目录 `pip install -e . --no-deps --no-build-isolation` 安装，本机已完成；不安装旧 `legged_gym` 或覆盖环境中的 RSL-RL。附加依赖列在 `requirements-isaaclab.txt`，本机均已满足。`constraints-rlisaac601.txt` 记录版本约束。

## 检查结果与输出

已完成的本机验证：

- NVIDIA 官方 `isaaclab.bat train` 完成 4096 环境、1 轮训练（204800 个环境采样步），保存模型，退出码为 0；仅将迭代次数缩短为 1，其余训练超参数采用原作者默认值。
- 官方入口完成 32 环境、3 轮训练及检查点续训，保留 HIM estimator 与对称性损失。
- 8 项 CPU 测试通过，包含原作者配置逐项对照、算法更新一致性、两个优化器恢复；注册任务通过官方 wrapper 的 101 步 GPU 检查。

此前物理迁移阶段还完成了以下检查（旧日志若已清理，以当前 `logs/nvidia_*` 验证记录为准）：

- `doctor` 与 `pip check` 通过。
- 4 个环境、100 步物理测试通过；覆盖接触力、有限数值、456/79 观测维度、自然重置和强制超时重置。
- 32 个环境完成 3 次 HIM-PPO 迭代（4,800 个采样步），参数保持有限，生成检查点和 TensorBoard 事件。
- 重新加载 `model_2.pt` 并回放 60 步通过。
- 按官方教程补齐物理参数后，重新通过 32 环境/3 次迭代训练；日志在 `logs/official_train.log`。
- 5 项 CPU 检查通过：URDF 关节顺序、xyzw 旋转、参考点旋转组合、对称变换的可逆性，以及 20 秒超时边界。
- 使用 `--record_ovd logs\ovd_migration_final` 已生成 OmniPVD 轨迹，并导出实际物理参数。

```powershell
# 单独运行策略兼容性检查
& E:\envs\RLisaac601\python.exe -m unittest discover -s tests -v

# 训练曲线
& E:\envs\RLisaac601\python.exe -m tensorboard.main --logdir .\logs\rsl_rl\homie_g1
```

当前运行证据位于 `logs/nvidia_*.log`、`logs/nvidia_env_check.json`；训练目录为 `logs/rsl_rl/homie_g1/<时间>_<run_name>/`，包含生效的 YAML 配置、模型和物理记录。详见 [训练说明](docs/homie_training.md)。

这些验证证明环境和短程训练链路已打通，不代表策略已经学会稳定行走或推车。当前仓库任务是 HOMIE G1 平地运动控制，未包含车辆资产或推车奖励。Isaac Gym 与 Isaac Sim 的物理及导入器有差别，长期训练效果和旧模型迁移效果仍需单独评估。本次验证使用无界面模式；GUI 入口已提供，但未进行人工视觉评估。

官方参考：[Isaac Lab 发布说明](https://github.com/isaac-sim/IsaacLab/releases)、[Isaac Sim 6.0 迁移说明](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/index.html)。具体调用以本机 `E:\IsaacLab_601\source` 的代码为准。
