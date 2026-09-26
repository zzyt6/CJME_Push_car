# V5.2 双手交互力估计器：改动与训练命令

日期：2026-09-25。代码在 `HomieRL/homie_pushcart_v52`，由 V5.1 复制而来。V5.1 的源码、入口和日志没有改动。V5.2 用独立的任务 ID（`Homie-G1-PushCart-V52-Direct-v0`）和实验目录（`logs/rsl_rl/homie_pushcart_v52`），不能加载 V5.1 检查点：帧宽和网络结构都变了。

## 上身力矩传感器模型（按 HOMIE 比例，2026-09-25 追加）

V5.2 初版的上身力矩观测接近理想 PD 输出，只有 ±5% 乘性噪声，估计器 R² 因此接近 0.99。HOMIE 的执行器随机化只作用于腿（上身走 PhysX 隐式 PD）。现在按 HOMIE 腿部的相同比例，给上身力矩的**观测**加上以下扰动。物理过程和力标签都不变。所有幅度都乘以课程的 `obs_noise_frac`：level ≤ 7 为 0，level 13 及以后为 1。

| 项目 | 幅度（满额） | 采样方式 | HOMIE 对应项 |
|---|---|---|---|
| 常值偏置 | U(−5%, 5%) × 力矩上限（肩肘 ±1.25 N·m，腕 pitch/yaw ±0.25 N·m，腰 ±4.4 N·m） | 每回合 | `actuation_offset_range = [-0.05, 0.05]` |
| 加性噪声 | U(−5%, 5%) × 力矩上限 | 每步 | `joint_injection_range = [-0.05, 0.05]` |
| 乘性噪声 | ±5%（沿用原设置） | 每步 | — |
| 读数延迟 | 0–2 个控制步 | 每回合 | `delay`（0–3 个物理子步） |
| 上身关节速度附加噪声 | ±0.3 rad/s | 每步 | HOMIE 为 ±2 rad/s，会破坏 FK 车速，这里取折中值 |

(Jᵀ)⁺τ 先验、帧内的力矩和关节速度，用的都是这同一份带噪样本，符合部署时的因果性。

## 为什么改

用 V5.1 `model_14600` 做的离线测试（`logs/bench/v51_force_predictor_eval_model14600/report.json`）结论如下：

| 问题 | 证据 | V5.2 的处理 |
|---|---|---|
| 预测头被 PPO 牵制 | 策略内预测头 R² 0.88 / 0.69 / 0.88，与线性回归相同；同样输入单独训练能到 0.94 / 0.88 / 0.97 | 估计器使用独立 Adam 优化器；Actor 读取的预测做 detach；估计器从 PPO 优化器中移除 |
| 合力不利于学习 | 单手力 R² 0.94–0.97，左右手 fy 相关系数 −0.47（对挤内力与合力量级相当） | 分别预测双手：[左 F, 左 M, 右 F, 右 M]，参考点为各自掌心，表达在车朝向坐标系；合力由固定抓握几何合成 |
| 标签混叠 | 50 Hz 标签中 10–25 Hz 能量占 52–70%；200 Hz 下有 19–35% 能量在 25 Hz 以上 | 标签取 20 ms 内 4 个有效子步的均值，与冲量一致 |
| 历史太长 | 5 帧 ≥ 20 帧 ≈ 1 帧 | 估计器只读最近 5 帧，即 0.1 s，与 FALCON 的本体历史长度相同 |
| 缺少物理先验 | 上身关节力矩近似满足 τ = g(q) + Jᵀw | 帧内新增 12 维 (Jᵀ)⁺τ（阻尼最小二乘） |
| 显式力通道没被使用 | 预测、真值、置零三种输入的闭环结果相同 | 保留该通道；新增 `force_to_policy=false` 等容量消融，专门检验这一点 |

## 结构

```
policy obs: 20 帧 × 136 维（V5.1 的 124 维 + 12 维静力先验）

history_encoder : 20×136 → 16 维 latent（PPO 训练）
force_estimator : 最近 5×136 → 运行归一化 → MLP[512,256,128] → 12 维（独立 Adam，lr 1e-3，不受 PPO 梯度影响）
compose         : 12 维 → 把手中心 [Fx, Fy, Mz]（固定抓握几何，无参数）
policy mlp      : [当前帧 136, latent 16, detach(双手 12), 合力 3] = 167 → 18 维动作
critic          : 136 + 46 特权量 = 182
```

- 估计器更新：每次 PPO 更新后，用同一批 rollout 训练 5 个 epoch × 8 个 minibatch，损失为 Huber（delta 1.0），梯度裁剪 1.0。
- 日志中的 `estimator_r2_*` 在拟合之前计算，数据是本轮新采集的 rollout，因此属于样本外 R²。
- 标签尺度为每只手 `[50, 50, 50 N, 5, 5, 5 N·m]`。
- 奖励、课程、终止条件、动作接口都与 V5.1 相同。`entropy_coef` 保持 0.005，因为双臂残差需要探索。

## 训练命令

从头训练：

```powershell
cd E:\CJME_Push_car

.\run_v52_pushcart_train.ps1 `
    -NumEnvs 4096 `
    -MaxIterations 20000 `
    -Level 0 `
    -Seed 1 `
    -RunName v52_hand_estimator_20k
```

等容量消融（估计器照常训练，但不把力输入给策略），用来回答"显式力是否有用"：

```powershell
.\run_v52_pushcart_train.ps1 `
    -NumEnvs 4096 -MaxIterations 20000 -Level 0 -Seed 1 `
    -RunName v52_no_force_to_policy `
    -ExtraArgs 'agent.actor.force_to_policy=false'
```

续训：

```powershell
.\run_v52_pushcart.ps1 train `
    --checkpoint 'E:/CJME_Push_car/logs/rsl_rl/homie_pushcart_v52/<运行目录>/model_2000.pt' `
    --num_envs 4096 --max_iterations 5000 --run_name v52_continue --headless
```

## 训练时看哪些量

| TensorBoard | 含义 | 期望 |
|---|---|---|
| `Loss/estimator_r2_L_fx` 等 8 项 | 单手各轴样本外 R² | 课程稳定后在 0.9 以上（离线参照 0.94–0.97） |
| `Loss/estimator_r2_handle_Fx/Fy/Mz` | 合成合力 R² | 参照 0.93 / 0.91 / 0.95（4 子步均值标签） |
| `Loss/estimator_eval_huber` | 拟合前 Huber 损失 | 课程每升一级会跳升，随后回落；不能和 V5.1 的 interaction loss 直接比较（标签和尺度都不同） |
| `Loss/estimator_valid` | 有效标签比例 | 约 0.99 |
| `Episode/v52_hand_label_valid` | 回合平均标签有效率 | 约 0.99 |

## 本次验证

- `tests/test_pushcart_v52_hand_estimator.py`：PPO 梯度不进入估计器；两个优化器互不重叠且覆盖全部参数；估计器学习率不受 KL 自适应调整影响；合力合成与通用 `resultant_wrench` 一致；(Jᵀ)⁺τ 能还原已知 wrench；检查点能恢复两个优化器和归一化器；JIT 导出一致；消融开关把 15 个力槽位置零。
- `tests/test_pushcart_v52_labels.py`：标签是有效子步的均值；预热样本被排除；reset 只清除选中的环境；奖励仍使用最后一个子步。
- `scripts/validate_pushcart_v52_runtime.py`：Isaac 实机短程检查。它输出先验与标签在各轴的相关系数，用来确认 (Jᵀ)⁺τ 的符号和坐标轴，不代表精度结论。

## 限制

- 标签仍是 Newton–Euler 手子树代理量（`dynamic_contact_wrench_calibrated=False`），不是标定过的传感器。
- 上身力矩读取 `ImplicitActuator.applied_torque` 对应的裁剪后 PD 估算值，并非直接测量的 PhysX 求解器驱动力矩。旧检查点只含 ±5% 乘性噪声，不能因当前源码新增了传感器扰动就认定它已经训练过。部署到真机需要核验电机反馈的物理含义；静力先验也依赖该读数，建议做无力矩/先验输入的独立训练消融。详见 [FALCON 与 sim2sim 驱动核查](push_cart_v52_falcon_transfer_audit.md)。
- 每个子步都做手部补偿，采样会变慢。V5.1 性能记录：只算最后一个子步约 5.87 万 transitions/s，每个子步都算约 5.5 万，相差约 6–7%。
- `mujoco_hl.py` 的帧构造仍是旧 V5 的布局。V5.2 sim2sim 现在通过独立的 `pushcart_bench.policies_v5x` 接入；默认使用按训练资产对齐的场景，并保留旧场景复现。场景修改、同条件评测及跨引擎误差见 [sim2sim 对齐记录](push_cart_v52_sim2sim_alignment.md)。
