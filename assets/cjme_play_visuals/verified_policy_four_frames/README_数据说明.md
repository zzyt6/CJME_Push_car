# 四帧抓握受力原始数据

本包保存训练策略在 MuJoCo 中重放时实际产生的数据，四帧取自同一段推车轨迹的 2、3、4、5 秒。力数据未平滑、裁剪或插值。

这不是原始 Isaac/PhysX play 的录屏或传感器记录。提取时没有运行中的 play，使用项目 run_two_stage_viewer.ps1 / mj view 所使用的加载器与控制管线，新启动独立 MuJoCo 进程重放。本任务未修改源程序、原 XML/URDF/USD 或 checkpoint 文件。

采集前后被核验的源文件哈希一致；采集结束后，工作区另有源文件发生变化，已在 validation.json 中如实记录并保留。本任务没有回滚这些修改。模型与权重文件仍匹配采集时哈希，最终再次用原 XML 独立复算四帧，约束力与 Jacobian 完全一致。

## 当前模型的范围

左右手通过 grasp_left / grasp_right 两条 weld 固定约束把力传给车。手指关节固定在已有抓握姿态，手指和辅助握杆碰撞被禁用；掌部部分碰撞 mesh 仍存在，但本次四帧没有手—车点接触。

本包有每只手对车的三维合力和三维力矩，以及求解器中的完整原始约束行。没有逐指、逐表面点的真实抓握压力/摩擦力分布。每帧 8 个接触记录实际属于脚—地面和轮—地面，不能把它们当作手部数据。

每个真实接触点的一个法向和两个切向是三维空间中的一组基。任意切平面方向的力都能表示为 f_t1 * t1 + f_t2 * t2，并非只允许沿两条切线方向施力。本模型 condim=3，只启用了法向和两个切向力；mj_contactForce 后三个力矩槽位为 0，是模型未启用扭转/滚动摩擦力矩的结果。

如果参考图表示“所有方向上的可承受力/力矩包络”，还需要具体的数学定义、接触几何、摩擦系数、执行器限制和各方向求解。四个瞬时合力不能唯一重建这个包络。本包未伪造这样的包络，也未新增接触模型。

## 数据来源

- 上层：logs/rsl_rl/homie_pushcart/2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt
- 下层：logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt
- 模型：assets/g1_pushcart/mujoco/g1_pushcart.xml
- MuJoCo 3.8.0，implicitfast，物理步长 0.005 s，控制步长 0.02 s。
- 场景：项目标准 push_03，任务命令 [0.3,0,0]（m/s、m/s、rad/s），附加载荷 0 kg，初始脚轮角度 0 rad，课程级别 17，seed=0。
- 全段运行无跌倒、几何终止、自动复位。
- 文件路径、SHA-256、关节/几何体地址映射在 snapshots.json 的 metadata 中。

## 文件

| 文件 | 内容 |
|---|---|
| snapshots.json | 四帧主数据：原始 efc_force、Jacobian、约束类型/ID、接触记录、状态、控制量、各 body 世界位姿、左右握持力/力矩、广义力。JSON 保留浮点精度。 |
| hand_wrenches.csv | 8 行（4 帧 × 左右手）；世界系/掌部轴向合力、关于车总成质心和抓握点的力矩、握持点位置、6 条原始 weld 乘子。 |
| contacts_all_points.csv | 32 行（4 帧 × 8 接触）；接触点、法向、两切向基、局部力/力矩、世界力/力矩、摩擦系数、两个接触 body。is_hand_cart 全为 False。 |
| constraint_solver_rows.csv | 340 行；四帧所有活动约束的求解器乘子、约束残差与速度；不同类型的乘子不混称为 N 或 N·m。 |
| actuator_joint_forces.csv | 108 行（4 × 27 执行器）；关节位置/速度、控制量、执行器输出、约束广义力、被动力、偏置项。 |
| physics_trace_200Hz.npz | 0～5.015 s 共 1004 个物理求解样本，包含双手六维量、原始 weld 乘子、qpos/qvel/ctrl 和策略动作；可用 NumPy 加载，未平均。 |
| images/ | 12 张 1600×1000 PNG：每帧双手、左手、右手各一张，纯绿背景；对应本包该帧真实状态。 |
| validation.json | 数量与源文件哈希核验。哈希核验范围为 Git 已跟踪的代码、配置和资产文件。 |
| images/photo_metadata.json | 截图来源、相机、可见几何体与裁切定义。 |
| images/photo_validation.json | 截图位姿与原始帧对齐、尺寸、背景与哈希核验。 |

## 坐标、正负号和单位

世界系右手坐标，+Z 向上，初始机器人前方 +X、左方 +Y。力单位 N，力矩单位 N·m，位置单位 m，时间单位 s。

左右手数据：

- force_on_cart_world_N：该手作用在车上的合力，世界系表达。
- moment_on_cart_about_assembly_com_world_Nm：该手约束对车的力矩，参考点为车总成质心。
- moment_on_cart_about_cart_grip_world_Nm：同一个约束的力矩平移到车侧名义抓握点，M_grip = M_COM - (p_grip - p_COM) × F。
- force_on_cart_in_palm_axes_N：同一个力在掌部坐标轴中的表达。它不是新测到的一组力。
- 软约束两侧可能存在微小位姿残差，所以同时提供 grasp_point_hand_world_m 与 grasp_point_cart_world_m。
- 要得到同一参考点上作用到手的反力/反力矩，两者同时取负。
- constraint_multipliers_raw 的后三项是 weld 求解器乘子，不能直接当成 Mx/My/Mz。笛卡尔力矩经过 J.T 映射恢复。

逐点接触数据：

- 局部六项顺序：[法向力, 切向1力, 切向2力, 法向扭转力矩, 切向1滚动力矩, 切向2滚动力矩]。
- contact_basis_rows_world 三行依次为世界系中的法向 n、切向 t1、切向 t2。
- 对列向量：F_world = R.T @ F_local；力矩同样旋转。旧版本错用了 R @ F_local，本包已修正。
- 正方向为作用在 geom2 上，geom1 在同一接触点受到反号的作用。
- 扭转和滚动力矩为 0，并不否认多个空间分布接触力关于远端点可以形成合力矩。

## 时间与重现

mj_step 返回后，状态已积分到下一步，但本步求解器力和 body 位姿仍对应积分前状态。本包在每次物理步之前保存 qpos/qvel/ctrl/time/warm start，然后配对该物理步求解器输出：

- solver_time_s：数据和照片对应时刻，分别约 2.000、3.000、4.000、5.000 s。
- integration_end_time_s：对应 2.005、3.005、4.005、5.005 s，不能用来标注该帧照片。
- state 保存积分前状态；body_poses_world 保存同一求解状态的位姿。
- JSON 的 high_level_action_after_command_clip、NPZ 的 high_level_action 是项目已有动作限幅后的策略命令；原始力值未限幅。
- NPZ 的 hand_wrench_world_about_cart_grip 形状 (1004,2,6)，第二维按 hand_order=[left,right] 排列，最后一维 [Fx,Fy,Fz,Mx,My,Mz]，力矩关于车侧抓握点。
- 从原 XML 创建独立 MjData，恢复 state 各字段以及 solver_time_s 后，运行 mj_forward 可复核。四帧已独立重放核验，约束力、Jacobian、全部导出广义力、body 位姿与双手笛卡尔量最大绝对差为 0。

照片在独立渲染副本中只显示原手/腕 mesh 和原车握把表面，隐藏其他身体与碰撞辅助几何，使用深度/分割遮罩得到精确 #00FF00 背景。没有用同一张静态图冒充四帧，也没有镜像替代另一只手。手指为模型预设固定关节，照片展示各帧腕部/车体运动下的握持姿态，不是新模拟的张手—闭手动作。

## 旧结果

上一次零动作试验未运行训练策略，时间未与求解器严格对齐，且世界接触力的坐标旋转存在错误。旧原始局部数值保留在项目 .cache/grip_force_superseded_20260922 供追溯，不纳入本包。旧力场图同样撤出交付。请使用本目录的文件绘图。
