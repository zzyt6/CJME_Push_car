# 按 NVIDIA 官方教程进行的 HOMIE 迁移核对

适用范围：本机 `RLisaac601`、Isaac Sim 6.0.1.0、`E:/IsaacLab_601` 的 Lab 3 开发版。检查日期：2026-09-14。

## 查阅的官方资料

1. [From IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)：任务结构、控制步长、场景克隆、关节顺序和环境步进迁移。本机也有对应的 `E:/IsaacLab_601/docs/source/migration/migrating_from_isaacgymenvs.rst`。
2. [Migrating To 3.0](https://isaac-sim.github.io/IsaacLab/develop/source/migration/migrating_to_isaaclab_3-0.html)：按当前版本处理 `ProxyArray`、索引写入、XYZW 四元数、PhysX 子包和可视化命令。
3. [Comparing Simulations Between Isaac Gym and Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html)：读取实际质量、惯量、驱动、阻尼、碰撞参数，并录制 PhysX 调试轨迹。本机对应 `comparing_simulation_isaacgym.rst`。

第一篇教程自身注明基础示例针对 Isaac Lab 1.0；不能直接把其中的张量 API 或旧版四元数约定用于本机 Lab 3。因此实际函数签名同时对照了本机源码。

## 已落实的迁移项

| 核对项 | 本项目实现与验证 |
| --- | --- |
| 物理与策略频率 | 原配置 `dt=0.005`、`substeps=1`、`decimation=4`；现为 200 Hz 物理和 50 Hz 策略，奖励仍按 0.02 秒缩放 |
| 关节顺序 | `find_joints(..., preserve_order=True)` 按 URDF 关节名恢复原策略顺序；测试检查 27 关节和对称映射 |
| 四元数与状态 | 当前 Lab 3 与原 Gym 张量均为 XYZW，直接保持该约定；读取 `.torch`，使用 `*_index` 写入和 int32 索引 |
| 单步时序 | `_pre_physics_step` 每个策略步处理一次动作；`_apply_action` 每个物理步重算 PD；随后计算终止、奖励、终止观测、重置、新观测 |
| 真正终止与超时 | 分开保留 `terminated` / `time_outs`；20 秒对应 1,000 个策略步，修正原 `>` 比较造成的额外一步 |
| PD / actuator | 下肢显式计算力矩、上肢使用原增益的隐式位置驱动；把原来的关节 armature 0.01 写入所有 27 个关节 |
| 刚体参数 | 从原配置传入线性/角阻尼 0.01、速度上限和最大去穿透速度；逐个设置嵌套刚体，并核实 49 个链接 |
| 碰撞参数 | 两只脚的碰撞网格显式设置 contact offset=0.01、rest offset=0；处理实例代理后读取生效值 |
| PhysX 与克隆 | 显式选择 PhysX TGS、4/0 求解迭代设置和 GPU 接触缓冲；使用 InteractiveScene 复制环境并过滤跨环境碰撞 |
| 无质量参考链接 | 按 URDF 计算 IMU 与六个脚底参考点的运动学；保留脚底平行度等原始奖励 |
| 实际接触数据 | 为嵌套导入结构中的左右脚及躯干分别启用接触报告；物理测试要求测到非零接触力 |
| 随机化 / 调试基线 | 训练保留原配置启用的随机化；`--no_randomization` 关闭域随机化、观测噪声与随机上肢动作，便于排查物理差异 |
| 物理审计记录 | 输出 `logs/physics_config.json`，含生效的关节参数、质量、质心、惯量和碰撞属性；训练目录也保存该记录 |
| PhysX 轨迹 | `--record_ovd` 按官方顺序先指定录制目录再启用录制；已实测生成非空 `.ovd` |

本项目以官方的 `SimulationContext`、`InteractiveScene`、`Articulation` 和传感器实现上述时序。现在已增加继承 `DirectRLEnv` 的 `Homie-G1-Direct-v0` 注册任务，向官方 wrapper 提供 Gymnasium 五项返回值。HIM estimator 所需的重置前 critic 观测通过 `extras` 传递；官方 RSL-RL 5 `OnPolicyRunner` 使用自定义算法扩展点调用原 HIMPPO。`run_homie.ps1 train` 已转交 NVIDIA 的统一入口，详细命令及原作者参数见 [训练说明](homie_training.md)。底层七项接口仅供场景复用和原项目回放使用。

## 复现检查和录制

```powershell
cd E:\CJME_Push_car
.\run_homie.ps1 doctor
.\run_homie.ps1 smoke --num_envs 4 --steps 100 --no_randomization --viz none --record_ovd logs\ovd_migration_final
.\run_homie.ps1 train --num_envs 32 --max_iterations 3 --run_name official_migration_check --viz none
& E:\envs\RLisaac601\python.exe -m unittest discover -s tests -v
```

OVD 目录必须不含空格，因为当前本机 AppLauncher 使用空白字符拆分 `kit_args`。录制目录同时保存物理配置和烟雾测试结果；`rollout.npz` 保存输入动作、命令、根状态、关节位置和脚部接触力，便于之后向旧 Gym 侧输入相同动作。

实际验证包括：5 项 CPU 兼容性测试，4 环境/100 步物理参数检查，32 环境/3 次 HIM-PPO 训练迭代，以及检查点加载和继续训练。具体命令输出在 `logs/official_smoke.log`、`logs/official_train.log` 及相应回放/续训日志中。

接入官方入口后增加到 8 项 CPU 测试，并验证官方 wrapper 的 101 步 GPU 运行、32 环境 / 3 轮训练、续训，以及原作者 4096 环境规模的一轮训练。当前证据在 `logs/nvidia_*.log` 和 `logs/rsl_rl/homie_g1/`。没有改动本机 NVIDIA 源码或现代 RSL-RL 包。

官方推荐的严格物理对比需要旧 Gym 侧的 `.pxd2` 和当前 Sim 侧的 `.ovd`。本机没有可运行的旧 Isaac Gym 环境，因此本次完成了 Sim 侧参数核对与轨迹录制，尚未完成两套引擎的逐帧对照。短程训练成功也不等于策略已收敛。
