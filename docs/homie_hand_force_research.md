# 第一阶段手部外力：独立任务、源码研究与推车数据依据

研究日期：2026-10-04。范围：复制可运行的 HOMIE 基线任务；核对 FALCON、UniFP 的论文与公开训练代码；按左右手分别统计已有推车载荷，形成后续实现方案。

**本轮新任务仍为基线等价副本，尚未启用或实现手部外力。** 下文的施力机制是研究建议，不是已经训练完成的功能。原 `homie_isaaclab`、旧推车入口和 `model_60000.pt` 保持原样。

2026-10-05 补充：[SteadyTray 托架连接与 V6 场景对照](steadytray_scene_review.md)。已核实其“腕—托架”固定安装与“托架—托盘”物理接触是两种连接。当前 V6 的“手—车”固定连接需要先与真实抓握接口核对；下文从旧场景得到的负载范围只代表该场景下的需求记录，不能直接当作真实抓握能力或未经场景验证的新训练上限。

## 当前方案修订：推车工作范围内微调，以及抓握约束对采样的影响

后续任务以已经训练好的 `model_60000.pt` 初始化，在推车上身工作范围和双手负载下适配低层；不重复原 HOMIE 的全范围上身预训练。本节更新后文先前将推车姿态视为可选消融的安排。目前只更新设计，尚未改变可执行副本。

V6 的手保持在车把上，首先来自环境中的双手固定抓握连接：Isaac 的两个 `UsdPhysics.FixedJoint` 连接 palm 与 cart，见 `assets/pack_usd.py:172`；MuJoCo 使用两个 `grasp_left/right` weld，当前 XML 第 449/450 行。约束维持手相对车把的位置和朝向，车仍能运动，肩肘仍可随全身调整。MuJoCo 是软数值约束，PhysX 也有求解误差，不能解释成几何上永远零误差。该模型没有通过手指摩擦学习抓紧/脱手；抓握过载触发回合终止，不是模拟连接被拉断。

V6 的 15 维上身动作是 PD **目标角修正**：`q_target = clip(q_grasp + upper_hold_offset + delta_q, hard_limits)`。双臂每关节修正尺度 0.4 rad，腰 yaw 0.3 rad；变化率上限 3 rad/s，并有 idle gate 和动作延迟。这些是命令空间的范围，不是实际关节角/末端位置的保证。目标与实际姿态存在误差时，PD 产生力矩；固定连接把部分负载传给车把，物理系统共同决定实际运动。残差因此既影响姿态，也能在很小的末端位移下改变受力。

原 HOMIE 的上身随机采样在 V6 中被 `PushCartSimEnv._pre_physics_step` 替换；训练时上身残差来自高层 PPO 策略及其探索，不是环境独立均匀采样 15 个关节。reset 使用预设抓握姿态，并对机器人和小车施加同一个平移/旋转，维持初始连接几何。

这对新任务有直接后果：

- 将无车环境中每个手臂目标独立采样于 ±0.4 rad，仅对齐了命令范围，不能保证实际双手几何符合推车。移除约束后同一目标会产生不同的实际姿态及力。
- 后续记录应成对保存 `q_target / q_actual / dq_actual / F_L / F_R`，连同基座姿态、左右手相对位姿、命令和时间。优先使用满足现有推车约束的实际轨迹作为姿态分布依据，同时保留目标—实际误差以研究负载。
- 若保留无车外力适配环境，只能把它称为推车工作范围的近似负载训练。对于采样的自由空间姿态，可检查两手相对距离/朝向、与躯干的工作区和速度；必要时用可达性投影或 IK。此筛选针对自由空间实际姿态，不应错误地剔除 V6 中用于产生接触力的非可达 PD 目标。
- 若要严格保留 V6 的目标—姿态—受力耦合，需要真实车/双手连接或一个明确的虚拟把手模型，再训练低层。不能把随机点力当成固定连接的等价替代；在已有车反力上另加扰动时也要区分两种负载，避免重复计算。
- 初步消融改为：原冻结 HOMIE；相同初始权重和微调预算下，推车范围无外力适配；推车范围有外力适配。原全范围加力可作为附加对照，不是主任务。

采样中心沿用 `initial_pose.yaml` 的抓握角度＋`upper_hold_offset`，原 HOMIE 的 `default_dof_pos` 观测/动作参考不改变。左右腕 pitch 因硬限位，正向修正实际约只能达到 +0.319 rad；不能将“所有关节对称 ±0.4”当成最终执行范围。

## 1. 已落地的独立任务

| 项目 | 原基线 | 独立副本 |
|---|---|---|
| 包 | `HomieRL/homie_isaaclab` | `HomieRL/homie_hand_force` |
| Gymnasium ID | `Homie-G1-Direct-v0` | `Homie-G1-HandForce-Direct-v0` |
| 训练入口 | `train_homie.ps1` | `train_homie_hand_force.ps1` |
| 回放入口 | `run_homie.ps1` | `run_homie_hand_force.ps1` |
| 训练日志 | `logs/rsl_rl/homie_g1` | `logs/rsl_rl/homie_g1_hand_force` |
| URDF 转换缓存 | `.cache/g1_isaaclab` | `.cache/g1_hand_force_isaaclab` |

复制全部十个任务 Python 文件，包括环境、奖励、观测、配置和算法适配器；共享只读机器人源资产和原始 HIM-PPO 算法。不是继承原环境后在运行时 monkey-patch。之后修改副本不会改到原任务。

复制时的逐文件 SHA-256、限定的命名/路径替换及冻结模型哈希见 [baseline_manifest.json](../HomieRL/homie_hand_force/baseline_manifest.json)。`pyproject.toml` 增加新包，并已使用 `--no-deps --no-build-isolation` 更新 editable 安装。

验证结果：4 项副本独立性/注册/配置/入口检查，以及原有 8 项 HOMIE 测试全部通过。GPU 运行完成 8 环境、2 次 PPO 更新；旧 `model_60000.pt` 在新回放入口完成 2 环境、100 步。

```powershell
# 当前命令训练的是复制后的无手部外力基线，用于验证新任务链路。
.\train_homie_hand_force.ps1 --num_envs 8 --max_iterations 2 --run_name baseline_copy_check --viz none physics=physx
.\run_homie_hand_force.ps1 play --num_envs 2 --steps 100 --checkpoint logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt --viz none
```

证据：[训练日志](../logs/hand_force_task_copy_train.log)、[训练产物目录](../logs/rsl_rl/homie_g1_hand_force/2026-10-04_18-29-23_baseline_copy_check)、[回放日志](../logs/hand_force_task_copy_play.log)。短程运行验证接口可用，不证明新策略收敛，也不声称已经验证跨进程物理轨迹逐值相等。

## 2. 一手来源与固定版本

| 方法 | 论文 | 本次实际审阅代码 |
|---|---|---|
| FALCON | [PMLR 331, Zhang et al., 2026](https://proceedings.mlr.press/v331/zhang26a.html)，§3.1、§3.2、§4.2、Table 1 | [LeCAR-Lab/FALCON](https://github.com/LeCAR-Lab/FALCON/tree/a967a6d8494f57777cf8d266a644ac8e45833301)，`a967a6d8494f57777cf8d266a644ac8e45833301` |
| UniFP | [Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation](https://arxiv.org/html/2505.20829)，§3、附录 A | [unified-force/UniFP](https://github.com/unified-force/UniFP/tree/68847a070f88d731058c3d8476929bc3b205f5bd)，`68847a070f88d731058c3d8476929bc3b205f5bd` |

已从固定提交重新下载 FALCON 18 个、UniFP 14 个关键源文件，并逐文件验证 Git blob SHA-1。完整 URL、SHA-256 和字节数见 [.cache/hand_force_research_20261004/source_manifest.json](../.cache/hand_force_research_20261004/source_manifest.json)。代码分析为静态审阅，没有在本机运行这两套旧 Isaac Gym 训练器。

## 3. FALCON：外力进入物理，控制器学习抗扰动

实际调用链如下：

```text
_resample_force_settings
  → 左右手相位、方向份额、零力掩码、施力点、变化时长
_calculate_max_ee_forces
  → 当前手部 Jacobian + 关节力矩限制 → 各轴力范围
_scale_forces / _calculate_ee_forces
  → 课程尺度 + 相位 + 噪声 + 分量裁剪 + 行走反向投影
_physics_step
  → 每个物理子步同时施加末端外力和电机 PD 力矩
```

关键文件是 [diff_force 环境](../.cache/hand_force_research_20261004/FALCON/humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py)，重点行号 53、138、146、223、259、326、342、499、590；以及 [上身残差控制](../.cache/hand_force_research_20261004/FALCON/humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py) 的 `_compute_torques`（88 行）。

- **力施加到哪里：** `left_rubber_hand` / `right_rubber_hand` 刚体，施力点在线段上随机，线段连接当前手刚体位置与扩展末端位置。G1 扩展点相对 wrist-yaw 为 `[0.18,-0.025,0] m`；不能直接照搬到我们的 Dex3 手掌。
- **坐标系与频率：** simulator wrapper 第 487 行使用 `apply_rigid_body_force_at_pos_tensors(..., ENV_SPACE)`。外力在策略步计算，物理 decimation 的每个子步施加。`ENV_SPACE` 是环境坐标，不是随手旋转的手局部坐标。
- **与电机控制的关系：** 下肢/上肢两个 RL agent 联合训练。上肢动作是参考关节角上的残差，最终仍通过 PD 输出并裁剪关节力矩。外力独立进入刚体动力学，不等于往 actor 动作上加噪声，也不是默认启用的 `J^T F` 前馈补偿。
- **观测与奖励：** 五帧本体感觉历史供 actor；左右末端外力出现在 critic 的特权观测中。actor 不读取外力真值。下肢奖励主要管行走、站立和高度，上肢奖励管参考角跟踪及关节约束。
- **课程：** 公开配置初始 scale=0.1，范围 0–1；每环境 episode 长度大于 210 步增加 0.02、小于 200 步减少 0.02。按 50 Hz 分别约 4.2/4 s。不是“75% 完成率”，也不是固定训练轮数递增。
- **时间形状：** `randomize_force_duration=[150,250]` 是策略步，约 3–5 s。代码中三角相位只推进 stance 环境，完整来回周期为该时长的两倍；walking 环境仍因姿态/Jacobian变化及重采样改变外力，不能描述成全部环境都固定持续 3–5 s 的脉冲。
- **公开训练配置幅值：** x/y 分量裁剪 `[-40,40] N`，z `[-50,5] N`；还受姿态、方向份额和课程缩放影响。不是每手合力 40 N，也不是论文实机拉车展示的 100 N。

配置证据：[环境 YAML](../.cache/hand_force_research_20261004/FALCON/humanoidverse/config/env/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.yaml)、[奖励与课程 YAML](../.cache/hand_force_research_20261004/FALCON/humanoidverse/config/rewards/dec_loco/reward_dec_loco_stand_height_ma_diff_force.yaml)、[观测 YAML](../.cache/hand_force_research_20261004/FALCON/humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml)。README 的 G1 训练命令另将 effort-limit scale 设为 0.9。

### 论文公式和代码必须区分

论文用 `-τ_lim ≤ τ_g + J_EEᵀ f ≤ τ_lim` 讨论力矩可行性；公开实现存在这些差别：

1. 第 158–160 行取 `ref_pd_torques` 后原地乘零，计算力限时实际没有预留论文公式中的重力/保持力矩余量。
2. 训练 YAML 是 `use_lpf=False`。评测 `set_is_evaluating()` 才显式设为 True；滤波的是四组轴向上下界，且随后还会加随机力。不能说默认训练始终对最终施力做低通。
3. 第 234–235 行在 scale 之后加 `[0,1)` 随机量，再乘零力掩码。因此 scale=0 不保证严格零力。
4. 双手对共享腰关节的联合限制整段被注释；各自手臂约束不是完整全身接触可行性证明。
5. 行走时将水平力投影到命令运动反向。投影、噪声及施力点变化后没有再次做完整的受力点 Jacobian 联合校验。
6. `_resample_commands` 第 415 行对子集 `env_ids` 使用 `torch.where(...)[0]`，存下的是子集局部编号；直接当全局环境编号使用存在索引风险。移植必须单独检查部分环境 reset。

这些是固定版本的源码事实，不代表已在本机测出其最终策略性能问题。我们应借鉴机制并验证，不直接整段复制。

FALCON 的 `Upper-PD-w/o-Force-Curr.` 和 `Upper-PD` 两个 baseline 最贴近当前 HOMIE：上身由 PD 跟踪，双腿 RL；只增加力课程。Table 1 大力条件下 root tracking error 为 1.14→0.46，而上身误差 1.44→1.42，说明只训练腿部对稳定行走有效，上肢精确受力补偿仍受 PD 控制结构限制。不能把整个 FALCON 的效果归到“加了外力”一个因素上。

## 4. UniFP：施力、力指令和控制目标共同定义任务

本次重点读的是公开 README 指向的 B2-Z1 训练链路。论文还展示 G1，但不能据此声称这一份 B2-Z1 配置就是论文 G1 的完整训练设置。

关键文件：[legged_robot_b2z1_pos_force.py](../.cache/hand_force_research_20261004/UniFP/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py)。

- `step`（100 行）：每个物理子步先写关节力矩，再生成 gripper 外力并用 `apply_rigid_body_force_tensors(..., GLOBAL_SPACE)` 注入。**基座外力函数调用被注释**，即使配置中 `push_robot_base=True` 也不代表这条路径实际启用。
- `_push_gripper`（1037 行）有两套独立变量：`current_Fxyz_gripper_cmd` 是机器人期望施力命令，写入 command；`self.forces[...,gripper_idx]` 是真正施加到刚体的外力。力指令本身不是仿真外力。
- 两套采样幅值都是各轴 `[-60,60] N`，发生概率各 0.8，间隔各 3.5–9 s，上升/下降时长各 1–3 s，中间保持约 1 s；代码按总间隔进一步限制斜坡时长。这是渐增→保持→渐减→零力的时序，命令与扰动各自调度。
- 物理施力没有显式位置参数，按此 API 的刚体质心施力契约处理；若移植到我们手掌原点或抓握点，需要明确指定位置。
- `force_start_step=8000`，代码条件是 `global_steps > 8000*24`；默认每轮 24 步，相当于约 8000 轮后启用。`global_steps` 从 0 初始化，不能不检查恢复逻辑就把它当可续训的课程状态。
- `_compute_torques`（865 行）是全身 17 个动作（12 腿＋5 臂）的关节位置偏移，经 PD、motor strength 和力矩限幅。没有在这条执行路径中添加显式 `J^T F` 前馈。

最重要的是 `_reward_tracking_ee_force_world`（1891 行）：

```text
F_cmd_world = R_yaw F_cmd
x_target = x_command + (F_applied_world + F_cmd_world) / K
reward = exp(-2 * ||x_ee - x_target||_1 / tracking_ee_sigma)
```

公开配置 `K=[200,200,200] N/m`。这与论文静态阻抗形式一致：机器人面对外力、期望施力及环境反力时允许产生有定义的位置偏移。例如单轴净力 20 N 对应 0.1 m 的目标偏移。**这个偏移进入奖励目标，不是环境立刻把手传送过去，也不是将外力由虚拟弹簧计算后施加。** 当前公开训练真正施加的是独立采样的外力。

相应的基座速度奖励（2016 行）有 `v_target = v_command + (F_base_local + F_base_cmd)/D`；但这条训练 `step` 中基座施力调用关闭，应区分“有函数”和“执行中使用”。

actor 包含期望力 command 及 32 帧本体感觉历史。encoder 的 latent 输入 actor；decoder 用 MSE 监督预测机体速度、末端位置、末端力、基座力共 12 维。**代码中不是把 decoder 的力真值直接拼给 actor**。critic 使用特权信息。见 [actor_critic.py](../.cache/hand_force_research_20261004/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py)、[ppo.py](../.cache/hand_force_research_20261004/UniFP/legged_gym/b2_gym_learn/ppo_cse_pf/ppo.py)。

实现细节也不能照抄：`_push_gripper` 在四个物理子步均被调用，而触发条件使用尚未递增的 `episode_length_buf`；恰逢采样步时会在多个子步重复重采样。我们应将事件采样放在 50 Hz，把明确的力时序放到 200 Hz 执行。该函数 docstring 所说“修改 gripper velocity”也与实际写 force tensor 不符。

## 5. 两种方法对当前第一阶段的含义

| 方面 | FALCON | UniFP | 当前 HOMIE 第一阶段建议 |
|---|---|---|---|
| 外力的作用 | 干扰跟踪，训练补偿 | 同时影响阻抗/力控制任务目标 | 先研究保持原速度/姿态任务时的抗扰 |
| 上身 | RL 残差＋参考姿态＋PD | 全身 RL，包括手臂动作 | 当前 15 个上身目标仍由环境给定、PD 跟踪 |
| 期望力指令 | 不是主 actor 指令 | actor 明确接收 | 第一轮外力消融无需新增 |
| actor 外力信息 | 本体感觉隐式推断 | 历史 encoder＋力监督 latent | 保留 456 维输入，先隔离施力因素 |
| 最适合借鉴 | 关节能力约束、施力点、课程 | 分开命令/扰动、连续时间波形 | 不把两套算法混称为同一种随机扰动 |

若把 UniFP 的末端阻抗奖励直接加入当前仅控制腿部的 actor，机器人只能主要通过移动躯干/腿来追手目标，上肢控制权限仍在环境。这会改变第一阶段任务并可能与速度/高度奖励竞争。要做完整主动力控，应另设扩展动作/参考上身目标/力命令的消融，不作为本次简单抗扰变体默认。

## 6. 左右手范围：重新计算，而不是读取图轴或双手合力

复算脚本：[analyze_hand_force_training_ranges.py](../scripts/analyze_hand_force_training_ranges.py)。输出：[ranges.json](../logs/research/hand_force_20261004/ranges.json)、[per_hand_ranges.csv](../logs/research/hand_force_20261004/per_hand_ranges.csv)。每个输入数据和 metadata 均记录 SHA-256。

统一力方向为 **车对手**，因此原始“手对车”的 force/moment 全部取负。幅值不受符号影响，轴向范围会受影响。分析不使用图像平滑、central-95% 截尾或整体凸包代替原始样本；同时给出全轨迹、每段 1 s 后、路线前 19 m 等明确窗口。

| 数据集 | 采样/坐标 | 左手 P95 / P99 / 最大值 N | 右手 P95 / P99 / 最大值 N |
|---|---|---|---|
| 历史 5 工况 MuJoCo（载荷 0/5/15/25 kg） | 200 Hz，world | 52.06 / 67.50 / 116.98 | 60.06 / 75.90 / 109.70 |
| 当前 Isaac 推车 baseline，全轨迹 | 50 Hz，cart-heading | 37.05 / 43.58 / 52.48 | 37.46 / 42.92 / 49.19 |
| 当前 Isaac CWC 路线，全轨迹 | 50 Hz，cart-heading | 61.13 / 76.11 / 118.48 | 62.80 / 82.40 / 131.66 |
| 当前 Isaac CWC 路线，前 19 m | 50 Hz，cart-heading | 60.76 / 75.47 / 93.27 | 62.28 / 81.16 / 101.94 |
| 当前 Isaac CWC，vx 阶跃 | 50 Hz，cart-heading | 49.79 / 64.53 / 94.25 | 49.55 / 62.37 / 82.69 |
| 当前 Isaac CWC，vy 阶跃 | 50 Hz，cart-heading | 54.07 / 72.44 / 90.19 | 52.65 / 65.71 / 85.23 |
| 当前 Isaac CWC，wz 阶跃 | 50 Hz，cart-heading | 62.72 / 80.33 / 108.12 | 54.68 / 68.25 / 87.14 |

另复算当前 MuJoCo 的 vx/vy/wz、baseline/CWC 六组 50 Hz 轴向试验：当前 CWC 三组峰值分别左 52.28/53.28/54.68 N，右 52.61/46.28/55.48 N。这些与 Isaac 动态标签以及历史多工况载荷/动作条件不同，不能合并成一个“不分场景的真实最大值”。

### 关键峰值事件

当前 CWC 路线 t=40.2 s、接近终点时，车对手力为：

```text
左手 [-117.25, -16.71, -3.27] N，模长 118.48 N
右手 [+124.51, +21.87,+36.79] N，模长 131.66 N
双手合力 [+7.26, +5.16,+33.52] N，模长仅 34.68 N
```

该时刻命令约 `[0.06,0,-0.52]`。这说明两手存在明显反向负载；双手合力不能除以二作为各手范围。全路线约 81% 样本的左右力点积为负。反向点积并不严格等价于刚性双手系统的“纯内力”，还需力矩和抓握几何才能分解，但已足以证明必须保留双手相关性。

这也是为什么前 19 m 图表会漏掉最大峰值：峰值发生在终点阶段。停止、转向与启动的峰值应分层报告，不能以“只是瞬态”删除后再声称已覆盖最大力。

### 分量范围与持续时间

当前 Isaac 路线和三轴数据的联合分量极值（仍是仿真观测值，不是独立可任意组合的 box）：

| 手 | Fx N | Fy N | Fz N |
|---|---|---|---|
| 左 | [-117.25,106.96] | [-48.71,44.41] | [-52.51,50.43] |
| 右 | [-97.70,124.51] | [-47.62,35.27] | [-54.58,53.75] |

历史 MuJoCo world 分量包络另为左 `[-90.48,97.62] / [-98.92,71.71] / [-97.89,87.14] N`，右 `[-97.07,77.08] / [-79.74,106.12] / [-94.51,87.73] N`。缺少历史原始姿态轨迹时不能把这些 world x/y 直接当作机器人前后/左右。

当前 CWC 路线 0.5 s 窗口平均力向量的最大模长只有左 52.47、右 59.60 N；1 s 对应 39.77/42.68 N。超过 100 N 的最长连续记录为左 0.02、右 0.04 s。历史 200 Hz 数据超过 100 N 最长分别 0.02/0.015 s，0.5 s 窗口平均最大为 56.47/69.79 N。因此不能把 120–132 N 的瞬时峰值直接变成持续 3–5 s 的常规训练负载。

### 数据可信边界

- 当前 Isaac 数据是 incoming joint wrench 减手子树 Newton–Euler 负载的**模型动态标签**；不是独立校准的力传感器真值。左右手 moment 最大还达到约 16.19/15.23 N·m，需与手掌原点、物理驱动及抓握约束共同解释。
- 历史 200 Hz 原始父文件 `verified_policy_multiscenario_20260923/physics_trace_200Hz_multiscenario.npz` 已不在项目中；保留的 `force_points_and_hulls.npz` 确实包含未平滑逐手采样、符号对照、分段时间和命令，本次直接重算了它。父文件哈希只有历史 metadata 记录，不能重新核验；不得声称已复查历史完整物理轨迹。
- 当前 MuJoCo 有完整保留的轴向 trace，可核对 weld wrench；但命令和工况不同，不构成 Isaac 每个力尖峰的逐帧独立验证。
- 50 Hz 采样可能遗漏 200 Hz 峰值；已有路线单 seed，不能称为整个任务域的最大承载能力。
- 观测到的最大力、关节姿态条件下的准静态可承受力、同时满足脚底摩擦/平衡的全身可行力，是三个不同概念。

## 7. 建议的后续实现契约

第一版选择 **FALCON 式抗扰目的＋UniFP 式明确的时间调度**，保留 HOMIE 腿部动作与上身 PD 接口。具体约束如下：

1. **单独 task / log / checkpoint。** 所有环境改动只进入 `homie_hand_force`。关闭外力时不额外抽随机数，也不调用施力 API；保证与本轮复制快照可比较。
2. **成对采样左右手。** 优先从同一工况、同一时刻的 `(F_L,F_R)` 或短连续片段采样，保留同向/反向负载与转向联系。作为消融再比较独立采样。左右每轴边界并非任意组合，各轴都取极值会造出记录中没有的超大合力。
3. **分开常规和瞬态分布。** 常规负载参照每手 P95/P99 与 0.5–1 s 窗口；终点/换向峰值以短暂脉冲或原序列保留。当前证据支持常规覆盖几十 N、尾部覆盖左约 118.5 N/右约 131.7 N 的评测目标；它们不是已获准在任意姿态施加的固定训练上限。
4. **幅值由两个条件相交。** 从任务需求分布得到候选，再用当前姿态、真实施力点、双手共享腰部和现有力矩余量限制缩放。若定义 `f_ext` 为作用在机器人上的力，准静态保持力矩为 `tau_req = tau_hold - J_L^T f_L - J_R^T f_R`（点 Jacobian、坐标与符号必须一致），检查 `|tau_req|≤tau_limit`。还需独立监视地面摩擦、支撑和平衡；不宣称仅靠此式证明全身稳定。
5. **施力点明确。** 当前手掌 link 原点/抓握点与刚体 CoM 不同。在 palm 点施力要用对应 Jacobian；转成 CoM wrench 时包含 `(p_palm-p_com)×F`。用 `F` 和该位置调用物理 API 后不能再次重复加同一 `r×F`。
6. **先做 3D 点力消融，记录 6D 差异。** 推车刚性抓握可传自由力矩；任意 6D wrench 不能总用一个偏移点力代替（点力产生的力矩与 F 正交，不能生成沿 F 的扭矩）。后续若加随机自由力矩，应据逐手 moment 数据单列，不把已有未校准 16 N·m 直接写成默认值。
7. **采样和执行分频。** 50 Hz 更新事件/课程；200 Hz 施加外力。Isaac Lab 3 使用 `instantaneous_wrench_composer.add_forces_and_torques_index`，明确 world 坐标和实际受力点，每子步一次。避免 `set` 清空其他施力源；reset 只清目标环境，并检查无残留、无跨环境索引错误。
8. **课程不只数训练轮次。** 用一定窗口内的完整 episode 成功率、速度误差、手臂饱和率调整；升/降速率属于待标定超参数，应先列为方案值。课程尺度、时间计数、随机状态与来源 manifest 要能保存/恢复。
9. **先不增加 actor 力输入或阻抗奖励。** 将“外力随机化”“力感知/监督”“主动力控”拆成不同实验因素；critic 若加逐手力，需要同时更新 79 维布局、镜像变换及 HIM estimator 的下一帧目标索引，不能仅拼六个数字。最初纯外力消融可以保持 actor/critic 不变。
10. **改为推车工作范围的上身课程。** 从已有 HOMIE 权重微调，主训练替换原全范围姿态采样，围绕当前抓握保持目标及高层残差的实际执行方式构造轨迹。命令范围不能替代实际姿态与双手几何约束，详见文首修订。记录姿态条件下的力裁剪比例；原全范围训练作为基线/附加对照保留。

## 8. 公平对比安排

建议形成 2×2：原低层/外力训练低层 × passthrough/我们的高层。比较低层时先统一手姿、速度指令和外力序列；比较高层时在相同低层下分别训练或明确冻结后的零样本替换，不能混淆。

主任务从 `model_60000.pt` 微调，并建立相同推车姿态分布、同预算继续无外力训练的对照，分开评估姿态特化与外力适配的收益。原冻结模型作为参考保留。正式结论用多个 seed、相同最终 checkpoint 选择规则与未用于选幅值的保留工况；当前已看过的终点/25 kg 工况不能再标为完全未知泛化测试。

低层指标：无外力与受力下速度/高度误差、跌倒率、手姿偏差、关节饱和、恢复时间；高层指标：路线完成率、轨迹误差、末端双手分别的峰值/持续负载及相互抵消程度。外力本身是环境输入，控制器需要产生的地面反力和电机力矩是响应；这两者必须分别记录。

本轮已经完成研究与可运行副本。下一步工程实现应依据上述契约，先验证受力点、符号、作用时间、力矩限制和关闭等价性，再开展长训练。原先“每手 50 N、总 100 N”的建议在当前数据下不成立，已撤回。
