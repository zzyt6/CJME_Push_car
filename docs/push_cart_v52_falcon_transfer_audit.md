# V5.2 力估计迁移与 FALCON 实现核查

日期：2026-09-27。核查对象是 `model_19999.pt` 的实际配置、项目代码、已完成的三场景评测，以及 FALCON 官方源码。本页记录最初的源码与资料核查；后续已增加共同显式 PD 分支并完成实测，见 [冻结 HOMIE 控制对齐报告](../logs/bench/v52_control_alignment/comparison.md)。两轮均未更新模型权重。

## 当前判断

优先排查 sim2sim 的驱动响应、力矩观测含义及闭链标签标定。已确认参数差异的修正不等于驱动和约束求解数值等价；现有结果不足以把误差归因为估计器网络结构错误，也不足以认定全部是评测程序错误。可以确定的是，当前估计器对 MuJoCo 的泛化不够好。

三组完整评测见 [comparison.md](../logs/bench/v52_sim2sim_alignment/comparison.md)。Isaac 的高分仅证明模型能预测该引擎和标签定义下的新数据，不能独立证明标签的物理准确性或实机测力能力。

## FALCON 实际怎样迁移

官方仓库核查版本：`LeCAR-Lab/FALCON`，commit `a967a6d8494f57777cf8d266a644ac8e45833301`。源码只读副本保存在 `.cache/falcon_reference/repo`。

| 环节 | FALCON 官方 G1 路径 | 当前项目 |
|---|---|---|
| 训练上身控制 | Isaac Gym，每个物理子步显式计算 PD，再施加并裁剪关节力矩 | PhysX 隐式位置驱动；腿部才接收 Python 计算的显式力矩 |
| MuJoCo 上身控制 | 读取与实机相同格式的目标位置、速度、Kp、Kd、前馈力矩，逐子步计算显式 PD 并限幅 | 新配置使用 MuJoCo 原生位置 PD + implicitfast；不是 PhysX 隐式驱动的数值复刻 |
| 名义时钟 | 训练 200 Hz 物理、4 子步/动作；部署 0.005 s 物理、50 Hz 策略 | 同为 200 Hz/50 Hz；频率相同尚不能保证采样相位、驱动离散化等价 |
| 部署输入 | 主策略读取关节位置/速度、IMU、命令、上一动作，五帧历史；不读取手力真值或 PD 力矩代理 | 还输入上身 PD 力矩代理、由其计算的静力先验、HOMIE 速度估计和 FK 派生量 |
| 驱动随机化 | Kp、Kd 均为标称值的 0.9–1.1，按所有自由度实际参与力矩计算；上/下身控制延迟开启 | 随机 Kp/Kd 和附加力矩虽被计算，但只把显式结果写给腿；上身隐式驱动增益没有经这条路径随机化 |
| 交互任务 | 训练时向双手施加已知三维外力，随机作用点，按力矩可行性做课程 | 双手固定抓握小车，存在闭链内力；监督双手三维力和三维力矩 |
| sim2real 接口 | 同一策略程序经 SDK 状态/命令接口连接 MuJoCo 或真实机器人 | 当前主要是直接调用各仿真环境的独立评测接口 |

上表的项目驱动描述对应旧检查点的 `training` 路径。后续新增的 `explicit_pd` 路径在两端逐子步提交相同显式力矩，上身原生 PD 关闭，并实际使用上身增益/附加力矩随机化因子；该分支没有改变旧检查点的训练历史。

FALCON 训练中的核心计算为 `tau = Kp * (q_target - q) - Kd * dq`，MuJoCo 端为 `tau_ff + Kp * (q_target - q) + Kd * (dq_target - dq)`，在零目标速度和零前馈时对应同一个控制律。两端都向引擎提交显式力矩。其 nominal gains 也在训练和部署配置中对应。

这并不意味着 FALCON 消除了所有物理引擎差异；它同时使用动力学随机化、外力课程和面向部署的观测接口。官方公开主方法的迁移成功也不是“双手六维估计器在 MuJoCo 上保持高 R²”的证明。

源码依据（固定版本）：

- [实际训练类的 PD 计算](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma.py#L88)；[逐物理步施加力矩](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/envs/decoupled_locomotion/decoupled_locomotion_stand_height_waist_wbc_ma_diff_force.py#L326)。
- [MuJoCo PD、限幅及积分](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/sim_env/base_sim.py#L107)；[FALCON 交互场景调用同一个 PD](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/sim_env/loco_manip.py#L34)。
- [部署说明](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/README.md)；[G1 部署参数](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/sim2real/config/g1/g1_29dof_falcon.yaml)。
- [训练随机化](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml)；[主策略观测](https://github.com/LeCAR-Lab/FALCON/blob/a967a6d8494f57777cf8d266a644ac8e45833301/humanoidverse/config/obs/dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma.yaml)。

## 我们的估计器参考了 FALCON 什么

仓库可确认的参考是短本体历史和外力适应的研究思路，不能据此称作 FALCON 估计器复现。

需要区分论文的两个方案：

1. FALCON **主方法**：上、下身双策略联合训练，以本体历史隐式适应外力；手部外力作为 critic 的特权信息。
2. arXiv v2 **附录 A.3 的对照方案**：显式估计末端三维力，把估计值送入下身策略，并以 `Jᵀ F` 补偿上身。附录同时讨论作用点改变和准静态假设的局限。这是对照方案，不能与主方法混称。

V5.2 为 `5 × 136 -> 独立归一化 -> MLP[512,256,128] -> 双手12维wrench`；使用独立 Adam/Huber 监督训练，策略读取 detach 后的预测。新增的上身力矩输入、`(Jᵀ)⁺ tau` 先验、双手六维标签和冻结 HOMIE 都是本项目的具体选择。其网络结构没有因为用了五帧历史就自动继承 FALCON 的迁移性质。

HOMIE 的角色也不同：提供冻结的腿部策略和 HIM 基座速度估计；我们的双手力估计器不是直接复用 HIM 网络。

论文来源：[正式出版页](https://proceedings.mlr.press/v331/zhang26a.html)；[arXiv v2 附录 A.3](https://arxiv.org/html/2505.06776v2#A1.SS3)。

## 为什么 PD 差异可能放大估力误差

实际力矩平衡需要包含惯性、科氏/离心项、重力、被动摩擦以及接触负载。V5.2 的静力先验只解 `Jᵀ w ≈ tau`，并未完整扣除这些项，网络需要从历史中学习其影响。

更关键的是，Isaac Lab 的 `ImplicitActuator.compute()` 明确把 `applied_effort` 描述为 **approximate torques**。它根据子步开始的目标位置、关节位置/速度计算并裁剪 PD 估算值；位置目标和增益仍交由 PhysX 隐式求解。这不是直接测量的求解器电机力矩。

项目在 `PushCartSimEnv.step()` 读取 `robot.data.applied_torque`，随后把上身部分送入观测和静力先验。MuJoCo 当前复现了相同的 PD 观测公式，但观测公式相同并不保证“该读数与实际手力之间的关系”相同。双手闭链中的微小变形、约束刚度和驱动离散化尤其可能改变内部负载分配。

后续同引擎驱动对照支持模型对驱动方式敏感：Isaac 单工况 Fx RMSE 从原驱动的 0.989 N 上升到显式驱动的 14.379 N。但该干预也改变了闭环轨迹，尚未量化各因素贡献。旧 MuJoCo 也是显式 PD，但它训练端仍对应隐式驱动，且旧模型另有物理参数错误；旧场景分数较高不能单独证明显式 PD 一定解决迁移。

目前使用的 `model_19999.pt` 保存配置只有上身力矩 ±5% 乘性噪声，没有后来加入的常值偏置、加性噪声和读数延迟字段。源码里新增默认项不等于旧模型已经训练过这些扰动。即使增加观测噪声，也不等于已经随机化实际驱动和约束动力学。

## 新增的有限诊断

在旧/新 MuJoCo 各复跑一个工况：额外载荷 12.5 kg、直行 0.3 m/s、500 步、舍弃前 2 s。脚本为 `.cache/v52_transfer_probe.py`，输出为 [transfer_probe.json](../logs/bench/v52_sim2sim_alignment/transfer_probe.json) 和 `probe_legacy.npz` / `probe_isaac.npz`。不改变物理参数、不训练、不替换策略输入。

- **整数控制步错位**：扫描预测相对标签的 -5 到 +5 个控制步（±100 ms），使用统一的中间目标窗口。两场景三个合力/力矩轴的最小 RMSE 都在 0 步；不支持“整体错了一帧”作为该工况的主要解释。该检查不能排除物理子步内的采样差异。
- **粗略输入量程**：用检查点的训练归一化器查看每组输入的绝对 z 值，第 99 百分位在对齐场景均不超过 2.71；力矩组为 1.73，先验组为 1.70。没有看到普遍的量纲/幅值爆炸。这不证明联合分布、历史相关性和输入到标签的映射相同。
- **高频误差**：在共同的中间 350 个样本上，对齐场景原始 RMSE 为 `[21.063 N, 14.272 N, 3.021 N·m]`；对预测和标签同时作离线 3 Hz 零相位低通后，为 `[12.198 N, 4.747 N, 1.223 N·m]`。部分误差位于高频，但低频误差仍明显。零相位低通是诊断工具，不是可在线部署的精度修复。
- **误差不是统一常值偏移**：完整 12 工况中对齐场景合力 bias 仅约 `[1.23 N, -0.72 N, 0.027 N·m]`，RMSE 却约 `[19.44 N, 15.00 N, 3.08 N·m]`。统一加减常数不足以消除误差。
- **并非所有输出都坏**：对齐场景单手 Mx/My 的 R² 仍约 0.94–0.96，但单手 Fy 已为负，不能把合力 Fy 的负 R² 全归于左右手相消。

已有的 2720 维观测拼接一致性测试降低了字段顺序/缩放错误的可能；MuJoCo RNE 与 weld 一致性检查只核验该引擎内的力学映射，不能替代 PhysX 与 MuJoCo 闭链标签的直接标定。

## 建议的验证次序

执行进展：下述第 3 项的共同显式 PD 分支、相同样本的 HOMIE/PD 检查和单工况冻结模型对照已经完成；另完成了自由基座、双手闭链下的相同目标/已知车体外力重放。其结果没有显示物理响应更接近。第 1 项的固定基座隔离实验、第 2 项的跨引擎标签标定和第 4 项训练消融仍未完成。

1. **先验证驱动，不用估计器打分。** 固定基座或使用可控支撑，给相同关节目标阶跃/扫频及已知手部负载，分别记录 q、dq、目标、PD 估算值、实际积分响应和可独立核验的外力。检查时钟、限幅和上下身目标映射。
2. **再验证标签。** 单手、双手、无外力、已知外力/力矩逐级增加复杂度；统一作用方向、参考点、坐标系和20 ms冲量平均。特别单独验证双手闭链内力及 PhysX incoming-joint/subtree 补偿。
3. **设立两端显式 PD 的独立分支作为 FALCON 式对照。** 在 Isaac 和 MuJoCo 同时采用同一控制律、限幅和子步时钟，而非只改 MuJoCo 或照抄 FALCON 的不同机型增益。当前检查点是在隐式上身驱动下训练的，切换训练侧驱动会改变其环境，不能直接把新分支测评分数当作原训练环境基准。
4. **物理和标签核验后再做估计器消融。** 分别比较含力矩/先验、无力矩/先验、补偿重力/动力学残差的估计器；训练与验证分开。若需要重训，实际随机化上身驱动增益、延迟和负载，而不只给观测加噪。

判别标准：已知外力都无法一致重建时先修标签/接口；简单关节响应不一致时先修驱动；这些都通过后，同等输入仍预测不准，再定位训练覆盖和估计器结构。当前尚未完成上述固定基座与标签隔离实验，不能给各原因分配确定比例。
