# V5.2：Isaac 固定抓握在 MuJoCo 中的对应实现

日期：2026-09-27。范围：官方语义、项目资产与关节坐标系核查；未采用新的抓握配置，未重训或重新进行闭环评测。

## 已确认的方案

本项目 Isaac 采用左右两个 `PhysicsFixedJoint`，分别连接掌部与车体。对于保留机器人、动态推车以及双手闭链的 MuJoCo 模型，适合继续使用两个 `equality/weld`。推荐以 `site1/site2` 明确描述每个固定关节的两侧局部坐标系，使连接点位置和坐标轴直接对应 USD 的 `localPos0/1`、`localRot0/1`。

这是关节语义与坐标系的映射方案，不是已经验证的跨引擎受力等价配置。`body1/body2 + anchor + relpose` 也可以表达相同约束；选择 site 的原因是两侧坐标系更直接，便于审查。

## 官方依据

| 来源 | 可直接确认的事实 |
|---|---|
| [PhysX 5.4.1 Joints](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Joints.html) 的 Joint Basics 与 Fixed Joint | 每个对象具有一个局部约束坐标系；FixedJoint 约束两者的位置与姿态重合。文档明确允许数值求解漂移。这里引用关节语义，不声称本机 Isaac 内置的 PhysX 版本就是 5.4.1。 |
| [MuJoCo 3.8.0 weld](https://mujoco.readthedocs.io/en/3.8.0/XMLreference.html#equality-weld) | weld 软约束全部六个相对自由度；支持 body 与 site 两种表达。site 形式同时对齐两个 site 的位置和姿态。 |
| [MuJoCo equality](https://mujoco.readthedocs.io/en/3.8.0/computation/index.html#equality) | equality 可表达运动学树以外的闭环关节；单个 connect 只有三条位置约束。 |
| [MuJoCo 3.8.0 约束源码](https://github.com/google-deepmind/mujoco/blob/3.8.0/src/engine/engine_core_constraint.c) 的 `mj_equalityAnchors` 与 weld 分支 | body weld 使用 `eq_data[3:6]` 作为 body1 锚点、`eq_data[:3]` 作为 body2 锚点；site weld 使用 `site_xpos` 并经 `site_bodyid` 找到刚体。平移残差及 Jacobian 在这些点上构建。 |

当前安装的 MuJoCo 为 **3.8.0**。下载的官方原文、网址与文件哈希保存在 `.cache/grasp_mapping_reference/sources.json`。

## 项目实际采用的 Isaac 抓握

V5.2 的 `paths.py` 指向 `assets/g1_pushcart_v5/g1_pushcart.usd`，`sim_env.py` 用该组合资产建立场景。直接读取组合 USD 得到：

- `/Scene/GraspJoints/left` 与 `/Scene/GraspJoints/right` 均为 `PhysicsFixedJoint`。
- `body0` 是对应掌部，`body1` 是车体 `base_link`。
- 关节启用，`excludeFromArticulation=True`，连接体碰撞关闭，断裂力和断裂力矩为无穷。
- 资产生成代码还过滤机器人与车的碰撞；抓握靠固定约束维持，不靠手指摩擦。

MuJoCo 将固定掌部折叠进 wrist body，因此需要做完整位姿变换：

`T_wrist_grasp = T_wrist_palm × T_palm_grasp`

不能将掌部局部位置原样填入腕部。当前资产的对应结果如下，单位 m，四元数顺序为 **wxyz**：

| 手 | wrist 侧 site 位置 | cart 侧 site 位置 | cart 侧 site 四元数 |
|---|---|---|---|
| 左 | `[0.100, -0.027, 0]` | `[-0.490, 0.071690, 0.917218745]` | `[0.740357159, 0.669517754, 0.057419993, -0.017894108]` |
| 右 | `[0.100, 0.027, 0]` | `[-0.490, -0.071690, 0.917218745]` | `[0.740357159, -0.669517754, 0.057419993, 0.017894108]` |

wrist 侧姿态在数值精度内为单位四元数。完整精度与来源层哈希见 [joint_frames.json](../logs/bench/v52_grasp_mapping_review/joint_frames.json)。未来表达形式是 `<weld name="grasp_left" site1="left_hand_grasp" site2="left_cart_grasp" .../>`，右手同理；site 本身只定义坐标系，不提供额外刚度。

## 现有 MuJoCo 中遗漏的核查项

原 XML 使用 `body1/body2 + relpose`，没有指定 `anchor`。MuJoCo weld 的默认 anchor 在 **body2（车体）局部原点**。Isaac 的连接点在手与把手的连接处，二者在车体局部坐标中的距离约 **1.04237 m**。

这不表示手与把手相隔一米。根据两侧局部关节坐标系反算，Isaac 与原 weld 要求的理想相对位姿差只有约 `1.87e-8 m` 和 `7.44e-9 rad`。原来的初始姿态/FK 检查可以通过，但没有验证有限柔顺性下的平移残差究竟定义在哪个点。

锚点会改变平移与转动的耦合以及约束 Jacobian。双手闭链中的逐手内力因此需要重新验证；本次没有实验量化这一差别对预测误差的贡献，不能把全部差异归因于它。

## 为什么不能只改成“更硬”

PhysX 的固定关节也由数值求解器执行，并非绝对无漂移。MuJoCo 的 `solref`、`solimp` 与 `torquescale` 不是 PhysX 参数的一一对应项。此前将阻抗设为 `.9999` 的实验可以说明结果对约束数值设置敏感，但不能充当官方兼容预设或证明两种实现已经一致。

单个 connect 会允许相对转动；摩擦抓握会增加滑移与接触自由度；把车固定到世界或 mocap 会改变推车动力学。把车简单作为一只手的无关节子 body，也不能同时表达另一只手的树结构连接。保留当前拓扑、以两个 weld 闭合连接，是本项目最直接的对应方案。

## 已做的核验与当前边界

`scripts/audit_v52_grasp_frames.py` 从实际 USD 提取坐标系，并在内存副本中用四个 site 与两个 weld 编译完整 MuJoCo 模型。未保存该模型，未执行 `mj_step`。每手产生六条 equality 约束，reset 时连接点差约 **0.27 μm**、姿态差约 **1.39 μrad**。这验证表达形式和初始几何，不验证动态受力一致。

主标签代码 `grasp_cart_wrenches()` 按 equality ID 经 `Jᵀ` 映射到车体广义力，没有把 `eq_obj1id` 当 body ID，因而没有发现这一处 site 转换障碍。资产诊断脚本 `audit_v52_sim2sim_assets.py` 的 equality 输出则假定对象为 body，正式转换时需要适配 `eq_objtype/site_bodyid`。力与力矩的符号、参考点和控制周期平均仍需重新核对。

默认入口已回到 `nominal + urdf`；用实际 V5.2 检查点初始化确认了默认选择和显式实验选择，见 [default_selection.json](../logs/bench/v52_grasp_mapping_review/default_selection.json)。12 项相关既有测试通过。

后续应依次验证：连接坐标系、单手已知载荷、双手相同目标下的内力与车侧平衡，再进入固定网络的真实推车预测验证。保持时间步长、驱动、惯量与接触设置可追踪，避免同时更改多个因素。

用户随后要求查看已有修正配置的力预测准确度，该独立统计见 [准确度报告](../logs/bench/v52_corrected_force_accuracy/accuracy.md)。它使用此前 `rigid_limit + physx` 的存档轨迹，未使用本次提出的 site 映射。
