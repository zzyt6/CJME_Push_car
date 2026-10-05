# Dex3 双手—横向车把接触：近两年文献与场景修订依据

核查日期：2026-10-05；检索时间范围：2024-10-05 至 2026-10-05。用户已明确实机方案为 **Dex3 双手闭合手指握住横向车把**。本轮只研究文献与设计，不改机器人资产、V6 或训练入口。

## 结论与筛选边界

当前最需要补齐的是 **有闭合力、有限摩擦、允许滑移/旋转/失去接触的手—把手交互**。原 V6 的双手固定关节预先满足了相对抓握关系，却没有证明真实 Dex3 能维持这项关系。额外训练手部抗外力不能单独修复这个问题。

但“固定连接必然不能 sim-to-real”过于绝对：真实机械锁紧或经过验证的无滑移抓握，在有限工况内可以用固定连接近似。当前 Dex3 配置缺少的正是这种适用范围的验证，不能直接宣称其固定连接结果能迁移。

本轮核查了 OpenAlex、arXiv、PMLR、RSS 正式论文集、Crossref、作者项目页和可获取源码。普通网页检索返回了挑战页或不相关结果，没有用这些结果支持结论。**在已核实文献中，尚未找到同时满足“近两年顶刊顶会＋G1/Dex3 双手握横杆＋显式小车接触训练＋实机迁移验证”的完整同构方案。** 下面明确区分局部机制参考、直接场景参考与不匹配工作，不能把它们拼成已有完整解决方案。

HAIC 按用户要求不列入直接参考。它与本项目的抓握姿态/接触需求不匹配；不能仅凭 RSS 和推拉车展示就把它当作 Dex3—车把连接模型的依据。本文也不以未经核实的“它完全没有仿真推车”断言作为排除理由。

## A. 最值得采用的正式发表工作

### 1. DexEvolve — RSS 2026：把手抓握要在接触仿真中承受加载

René Zurbrügg, Andrei Cramariuc, Marco Hutter. **DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis.**

- 正式会议来源：[RSS XXII, paper 59](https://www.roboticsproceedings.org/rss22/p059.html)。
- 全文：[arXiv 2602.15201](https://arxiv.org/html/2602.15201)。
- 作者项目：[dexevolve.github.io](https://dexevolve.github.io/)。
- 硬件：Franka Panda＋XHand；实物把手/柜体操作，有实机验证。不是双手人形推车，也不是 Dex3。

与我们连接建模最相关的细节：

1. 在 Isaac Sim 内优化实际手—物体相互作用，不仅检查几何是否把手包围住。抓握表示包含腕位姿、手指关节角和闭合指令。
2. Handles 数据包含 90 个不同把手（纹理变体共 190 个），提供碰撞几何。作者为 SDF 接触细化网格，三角形最大边长约 1 cm，并明确报告粗网格会造成可见穿透和接触不稳定。这是其资产处理设置，不能未经测试当作 Dex3 的最佳网格分辨率。
3. 把手具有环境侧 prismatic 运动模型，并加入 x/y 小行程（±1.5 cm）以测试顺应与抓握稳健性。**这些是把手/环境运动自由度，不是把手与手指焊接。** 推车不能照搬此抽屉运动模型，但可以借用其单把手抓握试验思想。
4. 逐方向施加 `+x,-x,+y,-y,+z,-z` 外力，按存活时间、接触距离和穿透评估抓握。不能因为手看起来环住把手就判定成功。
5. 金属与高摩擦软材料分别建材质（论文示例 μ=0.2/0.8）；不同接触表面不会统一设成一个任意大摩擦值。
6. 实机使用关节阻抗执行抓握与操作轨迹；接触能力由指面几何、手指作用和摩擦共同形成。

**对 Dex3 的启发：** 先建立一个“手＋横杆”的独立物理测试场景，验证抓握姿态、闭合指令、指面材料与可承受方向，再将通过验证的接口接到人形/小车系统。论文没有证明我们的 118/132 N 峰值对 Dex3 可行。

### 2. Learning Wrist Policies — IEEE T-RO 2025：闭合抓握还需要腕部柔顺

Florian Voigt, Abdeldjallil Naceri, Sami Haddadin. **Learning Wrist Policies for Anthropomorphic Soft Power Grasping in Handle and Door Manipulation.** IEEE Transactions on Robotics, 41:3738–3759, 2025.

- [正式 DOI](https://doi.org/10.1109/TRO.2025.3576950)。
- 发表信息已由 Crossref 核实；原摘要由 OpenAlex 与 Semantic Scholar 元数据交叉检索。
- **证据范围：本轮未取得可读全文或作者源码，仅使用原摘要。不能推断其精确仿真接触参数、关节约束类型或具体控制实现。**

摘要明确描述在统一 hand-arm 系统中引入 wrist compliance，结合 impedance-force 学习应对把手位置/朝向误差；策略迁移到服务型移动人形机器人，使用全驱动及欠驱动手完成真实把手/开门操作。其价值在于：在受约束把手上，固定抓握目标存在定位误差时，应让手腕/手臂按受力适当调整，而不是不断追求两个不相容的刚性位置目标。

**对 Dex3 的启发：** 握住横杆之后仍需处理两臂位置误差和内部负载。真实抓握不仅取决于手指闭合，也取决于两只手腕怎样协调。它是接触控制的局部参考，不是已经公开验证的双手推车场景模板。

### 3. Robust Dexterous Grasping of General Objects — CoRL 2025：手指本身是受控的执行系统

Hui Zhang, Zijian Wu, Linyi Huang, Sammy Christen, Jie Song.

- [正式 PMLR 305:3035–3050](https://proceedings.mlr.press/v305/zhang25h.html)。
- [全文 arXiv 2504.05287](https://arxiv.org/html/2504.05287)。
- 硬件：UR5＋Allegro，真实抓取；不是人形推车。

具体做法：

- teacher 使用物体几何、手—物体 contacts/impulses 等特权量；student 使用可部署的单视角点云及带噪关节状态。
- 策略输出手臂与手指关节目标，低层 PD 执行；保持 grasp 是控制结果，而不是将物体绑在掌心。
- 在训练中随机化摩擦、关节观测和 PD 增益，允许策略适应滑动、碰撞和执行器不确定性。
- 其外力鲁棒性实验中，仿真施加随机方向 2.5 N，实机用约 250 g 附加载荷在不同位置加载。这些量级说明其研究范围，不能外推为大力推车能力。

**对 Dex3 的启发：** 恢复真实可动手指和闭合伺服，不等于一定要立即把全部手指动作交给 RL。可先用固定闭合目标＋真实限力/PD 维持把手，让接触求解决定实际指角和夹紧力；之后再判断是否需要手指策略。

### 4. Guided Reinforcement Learning for Robust Multi-Contact Loco-Manipulation — CoRL 2024

Jean Pierre Sleiman, Mayank Mittal, Marco Hutter.

- [正式 PMLR 270:531–546](https://proceedings.mlr.press/v270/sleiman25a.html)。会议为 2024-11-06 至 11-09；论文集出版年为 2025，均在本轮时间范围内。
- [全文 arXiv 2410.13817](https://arxiv.org/html/2410.13817)、[作者项目与实机演示](https://leggedrobotics.github.io/guided-rl-locoma/)。
- 平台：ANYmal-D＋6-DoF 手臂；推/拉弹簧门、开关洗碗机，成功 sim-to-real。不是双手推车。

论文明确将交互分为无接触、抓取接触、非抓取接触；训练随机化把手几何、对象动力学、摩擦和扰动。状态跟踪包含末端与把手位置的误差，任务推进相位能够调整或回退。发生意外接触丢失时，策略可回退并重新建立接触；实机报告了打滑和错过把手后的恢复。

**能据此确认：** 接触建立、丢失和恢复是被学习/评估的行为，不能预先假定抓握永久存在。**不能据此确认：** 本轮没有得到可核验的作者场景源码，不能声称其具体每个接触面形状、摩擦锥离散化或底层约束 API 已经审计。

### 5. GraspQP — CoRL 2025：先测手—横杆的抓握力能力

René Zurbrügg, Andrei Cramariuc, Marco Hutter. **GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping.**

- [正式 PMLR 305:2583–2602](https://proceedings.mlr.press/v305/zurbrugg25a.html)、[全文](https://arxiv.org/html/2508.15002)、[官方代码](https://github.com/leggedrobotics/graspqp)。
- 与 DexEvolve 是同一研究路线中的抓握生成/评价基础；不是双手推车策略。

论文以接触点法向、摩擦锥及有界接触力评估 force closure，再在 Isaac Lab 做外力测试。附录对手指使用实际机械规格的力矩/速度限制和 PD 执行。它直接说明“包住物体的姿态”与“能抵抗指定力/力矩的抓握”并不等价；而 force closure 也不等于任意强度的实际承载能力。

本轮下载并校验官方当前提交 `e2de49478429f77c6b2a989e6d33a1af96266ede` 的 10 个相关源文件。这个提交是本次查看的代码状态，不保证与论文实验提交完全一致。

- `agents/static.py` 提供静态抓握与可选闭合指令；`assets/allegro.py` 用 finger actuators 配置有限力矩/速度以及刚度/阻尼。
- `handle_mining_env.py` 将机器人手和 handle object 分开建模，使用 `pull_drawer` 对对象侧加载、根据对象位移评价保持能力。
- 其 current Allegro 示例有 1.6 N·m、2 rad/s 等参数，**只能作为代码如何表达限力的证据，不能复制为 Dex3 参数**。
- 源文件清单/哈希：[graspqp_code/source.json](../.cache/contact_literature_20261005/graspqp_code/source.json)。

### 6. 已有力控论文的正确位置

[UniFP，CoRL 2025](https://proceedings.mlr.press/v305/zhi25a.html) 可以指导受力时调整位置目标；[Minimalist Compliance Control，RSS 2026](https://www.roboticsproceedings.org/rss22/p123.html) 通过标定电机信号估计 wrench 并做任务空间 admittance，展示了实机接触任务；[TACT，RA-L 2025](https://doi.org/10.1109/LRA.2025.3580329) 通过实机示教与视觉/触觉反馈学习持物，并明确讨论刚性物体内部力对位置控制机器人的负担。

它们分别涉及控制柔顺和接触反馈，不能代替 Dex3 指面—横杆的接触模型；TACT 的主要实机训练路线是模仿学习，也不能称为仿真 RL 推车迁移证据。

## B. 场景较接近，但不能当作顶会实机证明

| 工作 | 已核实内容 | 不应外推的部分 |
|---|---|---|
| [Partial Motion Imitation for Learning Cart Pushing with Legged Manipulators, 2026](https://arxiv.org/abs/2603.26659) | Go2＋单臂 WidowX AI 与真实建模的车交互；明确报告 gripper 与车把接触丢失；Isaac Lab→MuJoCo | 本轮未核实顶会/顶刊录用，也没有论文中的实机迁移验证；不能称为 G1 双手 Dex3 方案 |
| [SteadyTray, 2026](https://arxiv.org/abs/2603.10306) | 已核验实际源码/USD：腕—托架固定，托架—托盘接触；有实机 | 本轮未核实顶刊顶会发表；托架支撑不是闭合 Dex3 握横杆 |
| [FALCON, L4DC 2026](https://proceedings.mlr.press/v331/zhang26a.html) | 已核验末端施力代码，真实拉车展示 | 训练用末端外力近似，没有提供可照搬的双手—小车显式接触训练方案 |
| [CHIP, 2025/2026 预印本](https://arxiv.org/abs/2512.14689) | 作者展示双手持物中刚度过高引发竞争接触力，柔顺可改善；有实机 | 本轮未核实正式顶会发表，作者页面代码仍标 Coming Soon；不是场景连接模型的已审计实现 |
| [CAIMAN, 2025](https://arxiv.org/abs/2502.00835) | 非抓取式全身推物；附加小车实验 | 不是闭合双手握把手，不能用来证明该接口；实机推物与附加小车仿真要区分 |

过旧但常见的引用 `Learning Force Control for Legged Manipulation (ICRA 2024, arXiv 2405.01402)` 不在本轮近两年窗口内，未列为近期证据。也没有把后来上传 arXiv 的 2021 年旧论文误算为 2025 新工作。

## C. 对现有 V6 的具体判断

已审计的当前资产：双手 palm—cart-base 固定连接的 breakForce/breakTorque 为无限；手指角度被烘焙为固定关节；机器人手掌/手指没有对应接触碰撞体；机器人与车的碰撞对被过滤。实机 Dex3 则需要手指闭合力通过指面接触产生法向力和摩擦。

这里有四件不同的事：

1. 手模型能否形成包络横杆的几何；
2. 手指电机能否持续提供足够夹紧力；
3. 该接触能否抵抗沿横杆滑移、绕横杆旋转、拉开与扭转；
4. 行走与双臂策略能否在这些有限能力之内工作。

V6 当前把前三件事用永久连接近似了，然后主要优化第四件事。因此，当前仿真中末端不能脱离，并不表示物理 Dex3 不会脱手。现有 118/132 N 单手峰值及抵消负载只说明旧模型的运行需求，不能视为真实闭合力或抓握能力。

仅增大手臂 PD、增加 force penalty、改成一个随意设置刚度的弹簧连接，都没有自动解决这个差别。力矩约束也必须包括 **手指夹紧力→接触力可行性**；之前只检验手臂 `J^T F` 的方案不够。

## D. 建议先改哪一层

### 第一步：Dex3—横杆独立接触验证

- 使用真实横杆半径、位置和表面材料；恢复真实可动的 Dex3 指关节、掌面/指面碰撞体、手指质量和机械限位。
- 以位置/电流/力矩接口中实际可用的一种维护闭合目标，配置实机测得的增益、限流/限矩和速度；高层 RL 暂时可以不控制手指。
- 不建立 hand—cart 的永久固定关节；闭合时让有限电机作用与接触决定实际指角。**固定“闭合指令”可以，固定“手—物体关系”是另一种模型。**
- 用已知加载分别测试沿横杆滑移、绕横杆转动、推拉、上下负载、偏心扭矩及左右共同加载，记录位移、转角、指关节、电机信号和接触力。
- 摩擦/顺应/阻尼范围由实机拉力与扭矩测试标定，保留一部分测试用于独立验证。不得仅把摩擦调大到仿真“不掉”。

### 第二步：加入腕部/双臂协调

在抓握已经可复现的基础上，检查原 ±0.4 rad 目标残差是否会让两个手腕互相顶住。可研究降低不必要方向的刚度、方向性 admittance 或较小腕目标残差；参数和通路要与真实控制接口一致，不预设“越软越好”。此处参考 Wrist Policies、UniFP 或 Minimalist Compliance，而不是让手—车连接负责吸收所有不一致。

### 第三步：再接 HOMIE 和推车策略

先做冻结 HOMIE 的低速启动、停车、转弯，观察接触是否稳定；然后从旧权重进行推车范围低层适配与高层残差训练。新增指标包括每手轴向滑移、绕杆角、接触持续性、夹紧力/饱和率和恢复行为，不能只看路线完成率或双手合力。

若完整 Dex3 接触计算代价过高，可研究从上述独立试验拟合有方向性、有限承载、允许滑移/释放的等效模型。但它是需要校准与验证的代理，不是把 FixedJoint 改名成弹簧就能宣称 sim-to-real。

## 证据可复核性

全文 HTML、官方发表页、Crossref/OpenAlex 元数据以及 GraspQP 源码保存在 `.cache/contact_literature_20261005/`。论文中没有披露的细节未补写成事实；本轮没有运行这些论文的训练器，也没有验证它们对 Dex3 双手推车的效果。
