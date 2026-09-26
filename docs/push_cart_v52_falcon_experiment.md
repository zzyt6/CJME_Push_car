# V5.2 显式 PD 与力估计器训练实验

日期：2026-09-27。用户要求参考 FALCON 实现可用于真机接口的显式 PD，实际训练力估计器，并比较 Isaac 与 MuJoCo 上的表现。本页记录实验设计与执行入口；结果完成后写入 `logs/bench/v52_falcon_explicit/comparison.md`。

用户后续指定估力输入按 FALCON 附录 A.3 实现：新增真实 HOMIE 下肢动作历史，不含力矩或 DLS 先验，详见 [A.3 估力改造](push_cart_v52_falcon_estimator.md)。下面的原 `proprio` 版本仅是早期特征对照，不能称作 A.3 复现。

## 实验范围

冻结 HOMIE `model_60000.pt` 和原 V5.2 行为策略 `model_19999.pt`，先独立监督训练估计器，使驱动与输入特征的效果可以分辨。原策略仅负责生成数据；采集时新估计器不影响行为。之后将训练好的估计器实际接回 Actor，分别验证两引擎闭环表现。这里不把单独训练估计器称为重新训练了整个 V5.2 策略。

FALCON 源码固定于 `a967a6d8494f57777cf8d266a644ac8e45833301`：

- [MuJoCo 执行器计算](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/sim_env/base_sim.py#L95)：`tau_ff + kp*(q_des-q) + kd*(dq_des-dq)`，对总力矩限幅。
- [Unitree 命令接口](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/utils/comm/command_sender/unitree/unitree_command_sender.py)：写入 SDK motor command，再设置 CRC 并发送。
- [五帧本体观测](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml)：关节、IMU、命令、上身参考和动作历史。
- [驱动随机化](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml)：Kp/Kd 为标称值的 0.9–1.1，控制延迟作用于上下身。

借鉴的是上述控制接口、短历史和可部署输入原则。我们的双手十二维估计器、冻结 HOMIE 和监督训练仍是本项目设计；它不是 FALCON 双策略联合训练主方法，也不是其附录三维估力基线的完整复现。

## 预先确定的训练比较

| 数据组 | 上身驱动 | 实际驱动随机化 |
|---|---|---|
| `training` | 原 PhysX 隐式位置驱动 | 关闭 |
| `explicit_pd` | 每子步显式力矩 | 关闭 |
| `explicit_pd_randomized` | 每子步显式力矩 | 上下身 Kp/Kd 0.9–1.1；原 HOMIE 子步动作延迟 |

三组使用同一行为策略、采样种子 101，各 64 个环境、4096 个控制步；自然命令序列、随机载荷，标称接触参数和固定初始脚轮方向。观测噪声关闭。这是执行器随机化的有限对照，没有声称复制了 FALCON 的全部动力学随机化。

每组训练两种估计器，各用初始化/优化种子 11、22、33，共 18 个模型：

- `full`：原 V5.2 最近 `5 × 136` 帧，保留力矩代理和静力先验。
- `proprio`：最近 `5 × 96`，保留关节位置/速度、陀螺仪、投影重力、实际 HOMIE 命令、上身目标和上一高层动作；移除力矩、静力先验、HIM 速度和 FK 派生特征。

两组都使用 ELU MLP `[512,256,128]` 和双手十二维输出、Adam `1e-3`、归一化后的 Huber 损失、梯度裁剪 1、batch 2048。最多 100 epoch，Isaac 验证集连续 15 epoch 没有改善则停止。归一化只从训练集拟合。架构保持相同隐藏层；输入维度不同，因此参数总数不是完全相等。

按整个并行环境划分：`env_id % 8 == 0` 为验证，`== 1` 为 Isaac 测试，其余用于训练。该环境的所有回合都属于同一集合，避免历史重叠。选取验证集十二维归一化 MSE 最低的 epoch；MuJoCo 不参与训练或早停。

## 两种评测

1. **固定轨迹估力比较**：每个引擎、每种驱动各采集原行为策略的 12 个工况（额外载荷 0/12.5/25 kg，直行 0.3/0.6 m/s 和左右转弯），每工况 500 步、去掉前 2 s。各新模型读取完全相同的该轨迹样本；仅比较预测，不让新估计改变动作。
2. **接回策略的闭环比较**：把训练模型安装到实际 Actor 力估计通道，记录估力误差、速度/转向跟踪误差和失败。与第一项分开报告，避免把轨迹改变混作纯估计器收益。

各工况独立报告单手十二维及把手合力 `[Fx,Fy,Mz]` 的 RMSE/R²/bias。三种训练随机种子反映优化变异；它们不是三套独立采样数据。跨引擎标签仍各由自身引擎重建，现有 RNE 自洽检查不等于实物力传感器标定。

## 部署实现

[pd_command.py](../HomieRL/homie_pushcart_v52/pd_command.py) 提供 `q_des,dq_des,kp,kd,tau_ff` 命令，以及映射至 Unitree SDK2 `LowCmd.motor_cmd` 的适配器。它和两端仿真共用 PD 公式，支持：

- `motor_pd`：FALCON 风格，把 PD 参数发送给电机控制器。实际总力矩饱和由硬件执行器实现，需与其固件设置核对。
- `host_effort`：主机每 5 ms 按新状态计算并裁剪总力矩，SDK 中 Kp/Kd 为零，避免叠加第二个 PD。该模式需要主机伺服时钟和新鲜状态保障。

沿用保存的 HOMIE/G1 增益；没有照搬 FALCON 不同关节配置的数值。27 个模型关节映射到 G1 29 个电机时跳过腰 roll/pitch（13、14），与仓库 `g1_control.cpp` 核对。27 自由度模型假设这两个关节固定；它们的保持控制属于实机集成层，适配器不擅自写入。命令做有限值、映射、时间戳和限幅检查。

导出包含控制 JSON、完整/选定特征 TorchScript、无需仿真器的 NumPy PD 与本体历史打包器。`proprio` 打包器已与 MuJoCo 实际观测构造函数逐历史帧核对。模型导出需复核预测一致性及 CPU 时延。软件接口可接入 Unitree；本实验不包含实机发送、固件限幅验证或实机性能声明。

## 运行入口

```powershell
& E:\envs\RLisaac601\python.exe scripts/run_v52_transfer_training.py
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine mujoco --profiles isaac --upper_drive explicit_pd --capture --output logs/bench/v52_falcon_explicit/captures
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine isaac --upper_drive explicit_pd --capture --output logs/bench/v52_falcon_explicit/captures --viz none
& E:\envs\RLisaac601\python.exe scripts/evaluate_pushcart_transfer_heads.py --captures logs/bench/v52_falcon_explicit/captures --models logs/bench/v52_falcon_explicit --output logs/bench/v52_falcon_explicit/shadow_comparison.json
```

训练入口拒绝覆盖已有数据/模型目录。`job.json` 和各阶段日志记录状态；判断任务是否运行应同时检查记录的进程，而不能只凭状态文件。闭环评测通过 `--estimator <estimator.pt>` 安装实际预测头。导出入口为 `scripts/export_pushcart_v52_transfer.py --estimator <estimator.pt> --output <新目录>`。
