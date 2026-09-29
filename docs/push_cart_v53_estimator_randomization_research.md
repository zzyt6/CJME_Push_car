# V5.3 力／速度估计器随机化：真机方案与源码对照

核查日期：2026-09-28。核查对象：当前 V5.3、FALCON、Learning Force Control for Legged Manipulation、UniFP、HIMLoco。本文仅新增研究记录，没有修改训练配置、网络或权重，也没有运行新的训练或真机实验。

**结论：不能因为下层 HOMIE 用这些幅度，就认定力估计器和车速估计器也应使用相同幅度。但“要求精确，所以估计器必须无噪声／所有随机化都缩小”也不符合已核查的真机方案。**

真正需要分别设计的是物理变化、传感器误差和策略训练增强。公开方案中，估计器与 Actor 共用带噪历史很常见；Critic 和估计监督则额外使用仿真特权信息。没有查到这些方法普遍采用“策略大噪声、估计器小噪声”的统一规则。此前把 HOMIE 的观测幅度作为两个新估计器的默认依据，证据不足，应通过传感器标定和固定物理分布的消融确定。

**一、先把“随机化”分清楚**

| 项目 | 改变什么 | 对估计器意味着什么 |
|---|---|---|
| 物理随机化 | 真实参与积分的质量、摩擦、增益、驱动力矩、控制延迟、接触工况 | 改变真实运动和真实受力；输入、标签应来自同一组随机物理。估计器也需要覆盖实际会遇到的物理变化 |
| 传感器随机化 | 编码器／IMU／力矩读数的噪声、偏置、滤波、时间戳和读取延迟 | 改变网络看见的信息，不应把同一份观测噪声加到真实监督目标上。幅度应有测量依据 |
| 策略增强与探索 | PPO 动作标准差、策略额外输入扰动、估计输出误差增强 | 服务控制鲁棒性；PPO 动作标准差不是传感器精度，也不是估计器输出误差 |

大量且不可观测的传感器偏置或未知动力学参数，会使不同真实力／速度对应近似相同的输入历史。这时单纯加大网络不能保证逐时刻精确恢复。增加合理历史、校准、时间同步或额外可观测信号，才可能消除部分歧义。这个解释是基于估计问题的分析，并非下列论文给出的本项目误差归因。

**二、输入噪声的实际数值**

下表全部换回输入归一化之前的量纲。`U±a` 表示均匀分布 `U(-a,a)`；`Nσ` 表示零均值高斯噪声的标准差。数值来自指定版本的公开代码，不代表作者实测的传感器规格，也不保证等于每个真机演示使用的最终检查点配置。

| 方案／代码路径 | 关节位置 rad | 关节速度 rad/s | IMU 角速度 rad/s | 姿态输入 | 谁使用这些噪声 |
|---|---:|---:|---:|---|---|
| 当前 V5.3 完整课程 | U±0.02 | U±2.0 | U±0.5 | 投影重力每分量 U±0.05 | 高层 Actor、车速估计器、双手估力器共用同一可观测历史 |
| FALCON 官方 G1 主方法 | U±0.01 | **U±0.1** | **0** | 投影重力 0 | 上下身主策略使用本体历史；这不是附录显式估力基线的独立配置 |
| Portela 等，Learning Force Control，B1+Z1 | U±0.01 | U±1.5 | 主输入列表中没有角速度传感器 | 投影重力每分量 U±0.05 | 显式力／速度等状态估计模块和 Actor 共享带噪历史 |
| UniFP，公开 B2+Z1 配置 | Nσ=0.01 | Nσ=1.5 | Nσ=0.2 | 两维机体姿态噪声 σ=0.05；不是投影重力三维输入 | 历史编码器供状态估计和 Actor 共用；Critic 另取特权观测 |
| HIMLoco，公开 Go1／基础配置 | U±0.01 | U±1.5 | U±0.2 | 投影重力每分量 U±0.05 | HIM 速度／潜变量编码器与 Actor 使用同源带噪输入 |

两个容易误读的细节：

- FALCON 的 YAML 写着 `add_noise: False`，但 README 的 G1 训练命令明确覆盖为 `obs.add_noise=True`。实际噪声公式是 `(observation + U(-1,1)*noise_scale)*obs_scale`，所以关节速度确实是物理单位 ±0.1 rad/s，不能再除一次 `dof_vel` 缩放。
- 均匀 ±2.0 rad/s 的标准差约为 **1.155 rad/s**，而 UniFP 的高斯 σ=1.5 rad/s 标准差就是 **1.5 rad/s**。因此不能仅比较配置里的“2.0”和“1.5”，就说我们比所有真机方案噪声都大。相对 FALCON 主方法，我们的关节速度均匀幅度确实是 20 倍，但没有证据证明改成 0.1 就一定更准。

**三、FALCON：主方法与附录估力基线必须分开**

论文：[FALCON arXiv v2](https://arxiv.org/html/2505.06776v2)，重点为附录 A.2、A.3。源码固定版本：`LeCAR-Lab/FALCON@a967a6d8494f57777cf8d266a644ac8e45833301`。

主方法训练上下身两个策略，从五帧本体历史隐式适应外力。Actor 不读取手外力真值；Critic 读取真实基座速度、左右末端施加力等特权信息。公开的观测构造对 Actor 和 Critic 的共有 q/dq 字段都调用噪声函数，不能笼统说“整个 Critic 完全无噪声”。外力特权通道的噪声配置为零。

附录 A.3 的 `Upper-PD-ID` 对照才训练显式三维力估计器，输入为五帧 q、dq、IMU 角速度、投影重力和已执行的下身动作。估计力供下身策略使用，并通过 `JᵀF` 补偿上身。附录指出：改变施力点会改变有效雅可比，运动状态还会破坏准静态补偿假设。**该附录没有单独披露估力器与 Actor 的不同噪声数值，也不能把主方法真机演示解释为该估力基线的真机精度验证。**

公开 G1 主方法的物理随机化如下：

| 参数 | 代码配置 |
|---|---|
| 连杆质量倍率 | 0.9–1.2 |
| 额外基座质量 | −1 至 +3 kg |
| Kp、Kd 倍率 | 各 0.9–1.1 |
| 摩擦系数 | 0.25–1.25 |
| 控制延迟 | 0 或 1 个控制周期；名义 50 Hz 下为 0 或 20 ms |
| 额外随机驱动力矩 RFI | 关闭 |
| 基座 CoM 随机化 | 关闭 |
| 通用推机器人开关 | 关闭；任务中的末端外力课程是另一条机制 |

论文表 6 的摩擦为 0.5–1.25，并列出每 5 s、1 m/s 的推扰；与公开配置并非逐项相同。不能把论文、默认 YAML、训练命令任意拼成一份“实际配置”。

源码证据：[训练命令](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/README.md#L92)、[观测及噪声](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml)、[噪声公式](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/utils/helpers.py#L82)、[物理随机化](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml)。

**四、Portela 等 ICRA 2024：实际部署显式力／速度估计，但没有实现精密测力**

论文：[Learning Force Control for Legged Manipulation, arXiv v2](https://arxiv.org/html/2405.01402v2)。这是 FALCON 估力对照引用的工作。平台为 Unitree B1+Z1。源码版本：`Improbable-AI/learning-compliance@c760e1d74ad165d3c069d4f57ab5d066f6a41eb6`。

估计模块从本体历史预测基座速度、末端力等状态；Actor 读取同一历史和估计值；Critic 读取历史和真实特权状态。估计模块用特权状态做 MSE 监督，观测噪声没有等量施加到特权目标上。这里没有一条专门给估计器提供干净 q/dq 的输入支路。论文写 H=30，当前公开训练脚本为 10 帧，二者需要注明版本差异。

按 `config_b1_plus_z1`、基础配置和 `scripts/train.py` 后续覆盖顺序核查，训练包含摩擦 0.6–5.0、CoM 每轴 ±0.05 m、电机强度 0.9–1.1、Kd 倍率 0.5–1.5、动作延迟和外力等变化；最终 Kp 随机化关闭，Kd 和电机强度随机化开启。脚本中较早出现的相反开关不是最终值。

该论文的真机测力计实验覆盖 0–70 N、五个末端位置，报告的是**力跟踪误差约 5–10 N**。作者另明确写到估计力偏高，估力存在显著差异，策略可能部分忽略估计力反馈。不能把上述 5–10 N 跟踪误差偷换成估力误差，更不能据此称其达到了高精度力传感器水平。

源码证据：[训练输入与覆盖配置](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/scripts/train.py)、[基础噪声](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/b1_gym/envs/base/legged_robot_config.py#L692)、[关节速度噪声的单位换算](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/b1_gym/sensors/joint_velocity_sensor.py)、[Actor／Critic／估计模块](https://github.com/Improbable-AI/learning-compliance/blob/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6/b1_gym_learn/ppo_cse/actor_critic.py)。

**五、UniFP，CoRL 2025：明确测过真机估力，任务仍以粗粒度交互为主**

论文：[Learning a Unified Policy for Position and Force Control in Legged Loco-Manipulation, arXiv v2](https://arxiv.org/html/2505.20829v2)。源码版本：`unified-force/UniFP@68847a070f88d731058c3d8476929bc3b205f5bd`，本文核查其 B2+Z1 路径，不把它直接当作 G1 专属噪声配置。

32 帧带噪历史进入共享编码器，状态解码器估计末端力、基座速度等；Actor 使用当前观测和编码特征，Critic 另读真实状态及物理参数。尤其要注意：公开 Actor 代码直接消费的是编码潜变量，并非只依赖解码出的力数值，因此任务成功不能单独证明力解码精度。公开实现中的估计监督为 MSE；共享编码也不等同于本项目两个估计头与 PPO 的完全梯度隔离。

物理随机化包括摩擦 0.3–2.0、基座附加质量 0–15 kg、CoM 每轴 ±0.15 m、腿和臂电机强度 0.85–1.15、外部推扰等。论文表 A.2 的夹爪载荷为 0–0.5 kg，公开配置是 0–0.2 kg，应分开报告。

论文报告六档力水平下的**估力误差约 5–10 N**；X 轴测试覆盖 ±60 N，Y/Z 因硬件限制在 40 N 内。附录 E 明确将应用定位为粗粒度操作，Y 轴仍有 sim2real 偏差。C.6 讨论了驱动和接触模型差异，并把真机数据微调和系统辨识作为改进方向。作者还指出较低的 PD 增益可通过增大可观测位移改善估力，但过低会损害位置跟踪；这不是建议直接降低我们冻结 HOMIE／闭链系统的增益。

源码证据：[基础噪声参数](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/base/legged_robot_config.py#L147)、[实际高斯噪声与监督目标构造](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/legged_robot_b2z1_pos_force.py#L405)、[动力学配置](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/envs/b2/b2z1_pos_force_config.py#L70)、[共享编码器及状态解码](https://github.com/unified-force/UniFP/blob/68847a070f88d731058c3d8476929bc3b205f5bd/legged_gym/b2_gym_learn/ppo_cse_pf/actor_critic.py)。

**六、HIMLoco，ICLR 2024：速度估计用带噪历史，速度监督保留真值**

论文：[Hybrid Internal Model, arXiv v3](https://arxiv.org/html/2312.11460v3)。平台为 A1、Go1、Aliengo，均有真机部署；这里估计的是机器人基座线速度，不是轮车速度。源码版本：`OpenRobotLab/HIMLoco@ef289acaa62795009363b7b819c9186690630441`。

HIM 从与策略同源的本体历史预测三维速度和潜变量；Actor 读取当前带噪观测、预测速度、潜变量。速度使用 MSE 真值监督，潜变量使用对比目标。源码先给 q/dq/IMU 加噪，再拼入真实基座速度，因此 Critic 里的速度监督通道未被同一观测噪声污染，但其共有本体观测部分仍有噪声。配置里虽然还定义了 `lin_vel=0.1`，不能据此宣称真实速度标签被加了 ±0.1 m/s 噪声；实际构造没有这样做。

公开 Go1 所继承的基础物理配置包含附加载荷 −1 至 +2 kg、CoM ±0.05 m、摩擦 0.2–1.25、电机强度与 Kp/Kd 各 0.9–1.1，以及延迟／扰动；连杆质量随机化和恢复系数随机化关闭。论文表 6 列出更宽的若干范围，例如质量倍率 0.8–1.2、摩擦 0.2–2.75。两者不可混称同一版本配置。

这支持“学习式速度估计需要在含噪、随机物理中训练”，但真机越障、抗扰实验不能替代车速估计精度验收。我们的 V5.3 车速头只借鉴其监督思路，没有复制完整 HIM 对比学习架构，也不能自动继承它的迁移结果。

源码证据：[观测噪声配置](https://github.com/OpenRobotLab/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot_config.py#L199)、[物理随机化配置](https://github.com/OpenRobotLab/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot_config.py#L124)、[带噪输入与真实速度分开构造](https://github.com/OpenRobotLab/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/legged_gym/legged_gym/envs/base/legged_robot.py#L244)、[速度监督](https://github.com/OpenRobotLab/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/rsl_rl/rsl_rl/modules/him_estimator.py#L76)。

**七、对当前项目的具体判断**

当前代码在 [task_config.py](../HomieRL/homie_pushcart_v53/task_config.py) 定义噪声，在 [task_env.py](../HomieRL/homie_pushcart_v53/task_env.py) 采样一次可观测帧，在 [interaction_policy.py](../HomieRL/homie_pushcart_v53/interaction_policy.py) 给 Actor 和两个估计器分配历史。当前两个估计器各使用最近 5 帧，Actor 历史编码使用 20 帧；估力还读取 detach 后的预测车速。因此估计质量既受本身输入影响，也受串联的车速误差影响。

最需要重新论证的是**驱动力矩随机化与力矩读数误差共用比例**。当前完整课程同时存在：逐步乘性误差 ±5%、每回合固定偏置 ±5% 关节力矩上限、逐步加性误差 ±5% 上限，以及 0–40 ms 的读数延迟。力矩上限 25 N·m 的关节，后两种幅度分别为 ±1.25 N·m；上限 5 N·m 的腕关节分别为 ±0.25 N·m。固定偏置不会像独立白噪声一样靠短历史平均消失，未知延迟还会破坏力矩与运动的时间对应。

本次核查的 FALCON A.3、Portela、UniFP、HIM 输入都不能为这个 G1 全身力矩读数误差模型提供直接标定依据。HOMIE 的驱动扰动力矩作用于真实动力学，而读数误差只污染测量；二者不是同一物理量的同一不确定性。

关节速度 ±2、陀螺仪 ±0.5 也不宜继续仅以“HOMIE 就这么设”为理由。相对 FALCON 的公开 G1 主路径明显更强；但相对 Portela/HIM 和 UniFP，不能推断所有字段都过强，也不能仅凭跨论文数值判断应当降到哪个值。

已有 [名义评测](../logs/bench/v53_trained_grasp_20260928/nominal_comparison.md) 的原始 `nominal.json` 明确为 `noise_effective=false`、`randomization_effective=false`。该检查点在此评测中的单手各轴力 MAE 约 1.61–2.65 N，车速 MAE 为 `[0.0282, 0.0257, 0.0389]`（前两项 m/s，第三项 rad/s）。这些是同域名义仿真成绩，既不是完整输入噪声下的结果，也不能与其他机器人的真机 5–10 N 直接排高低。保存的训练配置确实采用 revision 3 和 `homie_relative`；本次未重建训练期间课程的逐时分布。

**八、建议的修改方向与可判定实验**

1. **保留物理变化的合理覆盖，给传感器误差单独配置。** 质量、摩擦、负载和驱动误差与测量噪声分开设定，避免一个 HOMIE 比例同时决定所有网络输入。也不应为提高估计器同域分数而把物理随机化全部关闭。
2. **Actor 与估计器共享真实测量模型；额外增强可单独控制。** 同一传感器的基础误差应一致。如果验证发现 Actor 需要额外鲁棒增强，可在其分支另加；“Actor 大噪声、估计器小噪声”是待验证的工程选项，不是文献已经证明的最优方案。Actor 的训练始终应覆盖部署时真实估计误差，不能用理想真值替代后直接部署。
3. **优先校准力矩链路和时钟。** 从静止、空载动作、已知负载记录中辨识逐关节增益、偏置、温漂／摩擦残差、滤波与读取延迟，再决定力矩输入和 DLS 特征的误差范围。关节角、速度和 IMU 同样用静止与运动数据估计误差；静止方差不能替代动态误差。没有实测前的候选范围只能标注为消融设置。
4. **固定物理分布，分别改变输入噪声。** 至少比较当前 HOMIE 观测幅度、较小本体噪声组、力矩偏置／白噪声／延迟分别改变的组，以及有实测后建立的传感器模型组。FALCON 的 q=±0.01、dq=±0.1 可作为一个本体消融参考，不应把其零 IMU 噪声直接当真机默认。先在固定行为与配对轨迹上诊断，再接回闭环评测；消融模型需要重新训练，单纯关闭已训练模型的测试噪声不能回答训练分布是否合适。
5. **对估计器设独立精度与延迟验收。** 力报告左右手分轴 MAE、RMSE、bias、P95 和力矩误差；速度报告 vx/vy/ωz，区分静止、起步、稳态、转弯、制动；同时报告响应时延。再分别看策略跟踪、稳定性和交互负荷。双手单独受力、合力和闭链内部力不可互相代替。若用于在线快速补偿，不应靠过度低通换取好看的 MAE。

若目标是稳定的亚牛顿估力或厘米每秒级车速精度，以上论文的控制演示不能作为充分保证。应先明确所需带宽和误差阈值，再用独立测力计／力传感器、运动捕捉或轮车外部速度基准验证；必要时增加测量信息，而不是仅继续扩大 MLP 或随机化幅度。

原始网页和本轮补充的源码副本位于 `.cache/estimator_dr_research/`；FALCON 固定版本已有副本位于 `.cache/falcon_reference/repo/`。本文关于公开代码的结论来自静态路径核查，没有声称运行复现这些外部训练工程。
