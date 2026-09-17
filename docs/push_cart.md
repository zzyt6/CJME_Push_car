# 推车任务（homie_pushcart）—— 第一阶段强化学习

独立于 `homie_isaaclab` 的任务包。**第一阶段**的定义：冻结已训练的 HOMIE 作为底层运控，G1 以预设抓握位姿握住推车把手，只训练一个上层修正/调度策略（HOMIE 速度命令 + 腰/双臂修正）。后续阶段（高度动作、腿部残差、解冻 HOMIE 联合微调等）不在本入口范围内。所有命令通过 `run_one_stage_pushcart.ps1`（自动选择 RLisaac601 解释器）。

## 资产（阶段 A）

| 文件 | 内容 |
| --- | --- |
| `HomieRL/homie_pushcart/config/initial_pose.yaml` | 27 关节预设位姿（臂 14 来自轨迹 JSON 最后一帧，腿/腰为 HOMIE 默认）、14 个手指角、pelvis 落地高度（脚部网格最低点贴地）、掌坐标系抓握中心 |
| `HomieRL/homie_pushcart/config/cart_params.yaml` | 推车尺寸、质量、轮/车身摩擦、关节阻尼、课程、交互参数；`derived` 段由位姿推导 |
| `HomieRL/legged_gym/resources/robots/g1_description/g1_grasp.urdf` | `g1.urdf` 的副本，14 个 fixed 手指关节的 origin 烘焙了抓握角；27 个可动关节完全一致 |
| `HomieRL/legged_gym/resources/object/cart/cart.urdf` | 4 万向轮被动关节体推车，OBJ 视觉 + 基本体碰撞，视觉横杆 |
| `assets/g1_pushcart/g1_pushcart.usd` | 打包场景：`/Scene/Robot` + `/Scene/Cart` + 两处掌↔把手 FixedJoint + 预设关节状态与链接位姿 + 物理材质 |
| `assets/g1_pushcart/asset_report.json` | 关节集合/限位/质量核对、固定关节局部坐标、材质绑定、静态一致性误差 |
| `assets/g1_pushcart/mujoco/` | `g1_grasp.xml`、`g1_pushcart.xml`（MuJoCo sim2sim 场景）、`mjcf_report.json` |

```powershell
.\run_one_stage_pushcart.ps1 pose import .\initial_pose\<轨迹>.json   # 生成 initial_pose.yaml + g1_grasp.urdf
.\run_one_stage_pushcart.ps1 cart build                                # 推导把手位姿, 写 cart.urdf
.\run_one_stage_pushcart.ps1 pose view --port 8080                     # viser 静态检查 (可调抓握中心/杆半径并保存)
.\run_one_stage_pushcart.ps1 pack --viz none                           # Kit 内打包 USD, 输出 PACK PASS
.\run_one_stage_pushcart.ps1 usd view --port 8082                      # viser 渲染打包 USD
& E:\envs\RLisaac601\python.exe -m unittest tests.test_pushcart -v
```

## sim2sim 遥控（阶段 T）：MuJoCo 物理 + viser 渲染 + Xbox 手柄

与 `MujocoDeploy/` 同一条路：MuJoCo 跑物理、HOMIE 检查点直接推理，CPU 上 20–35 倍实时。场景由 `mj build` 从 HOMIE 的 `g1.xml` 生成（手指烘焙为抓握姿态并关掉手指碰撞），带车版本追加推车（自由底盘 + 4 被动万向轮）和两处 `weld`（`*_wrist_yaw_link` ↔ `cart_base`，MuJoCo 已把掌 link 合并进腕 body）。

```powershell
.\run_one_stage_pushcart.ps1 mj build                        # assets/g1_pushcart/mujoco/{g1_grasp.xml, g1_pushcart.xml, mjcf_report.json}
.\run_one_stage_pushcart.ps1 mj smoke --mode cart --steps 600 --vx 0.3      # 无界面冒烟, 写 logs/pushcart_mj_smoke.json
.\run_one_stage_pushcart.ps1 mj teleop --mode robot --port 8080             # 只有 G1（上肢保持抓握姿态）
.\run_one_stage_pushcart.ps1 mj teleop --mode cart  --port 8080             # G1 + 推车
```

浏览器打开 http://localhost:8080。手柄经 Windows XInput 直接读取（启动前接好），映射同宇树遥控：左摇杆上下 → vx（前 +1.2 / 后 −0.8 m/s），左摇杆左右 → vy（±0.5 m/s），右摇杆左右 → ωz（±0.8 rad/s）；不控制高度（固定 0.74 m）。A 复位，B 命令归零。没有手柄时页面里的 vx/vy/wz 滑条直接生效。页面显示命令、基座速度、推车速度/偏航、两手 weld 约束力、摔倒次数；"镜头跟随"默认开启，"速度倍率"可放慢/加快。

## Isaac Lab 冒烟（阶段 C）

```powershell
.\run_one_stage_pushcart.ps1 smoke --mode robot --num_envs 2 --steps 200 --viz none
.\run_one_stage_pushcart.ps1 smoke --mode cart  --num_envs 2 --steps 300 --vx 0.3 --viz none
```

Isaac 版 `teleop --viz kit` 也保留，但 GPU 被训练占用时只有 ~0.3 倍实时，遥控请用 MuJoCo 版。

## 推车资产的一次修正：Isaac 里推车"蹦跳"的原因

诊断命令（只有推车、给 0.5 m/s 初速度、记录四轮离地与车身竖向速度）：

```powershell
.\run_one_stage_pushcart.ps1 cartdiag --colliders cylinder --v0 0.5 --viz none
.\run_one_stage_pushcart.ps1 cartdiag --colliders sphere   --v0 0.5 --viz none
```

| 轮碰撞体（Isaac Lab / PhysX） | 最大离地 | 车身起伏 | 竖向速度 RMS | 2 s 后速度 |
| --- | --- | --- | --- | --- |
| 圆柱，求解迭代 4/0 | 187 mm | 105 mm | 0.32 m/s | 0.49 → 0.02 |
| 圆柱，求解迭代 16/4 | 8 mm | 7 mm | 0.09 m/s | 0.48 → 0.02 |
| **球体**，求解迭代 4/0 | 0.0006 mm | 0.001 mm | 7e-5 m/s | 0.47 → 0.22（只剩滚阻） |

原因：Isaac Lab 的 PhysX 管理器为 GPU 管线关闭了圆柱精确几何（`/physics/collisionCylinderCustomGeometry=False`），圆柱碰撞体被烘成多面体凸包，滚动时逐面颠簸并耗散动能；NVIDIA 官方建议就是改用球形碰撞体或柔性接触。现已默认 `wheels.collider: sphere`（视觉仍是圆柱），Isaac 与 MuJoCo 使用同一种碰撞体。修正后 Isaac 里机器人+推车零命令下推车速度从 0.24–0.32 m/s 降到 0.04–0.07 m/s，四轮离地 ≤0.015 mm；命令 0.3 m/s 时 Isaac 基座/推车 0.26/0.26，MuJoCo 0.21/0.21。

球形轮之后在 Isaac 里补做的验收（4 env，HOMIE `model_48200`，均无摔倒、四轮离地 ≤0.03 mm）：

| 场景 | 基座速度 | 推车速度 | 备注 |
| --- | --- | --- | --- |
| 零命令 | 0.044 m/s | 0.04–0.07 m/s | 与单独站立同量级 |
| vx=0.3 | 0.26 | 0.26 | MuJoCo 同场景 0.21 / 0.21 |
| vx=0.3, ωz=0.3 | 0.30 | 0.32 | 偏航率 0.16，车载削弱了转向 |
| vx=−0.3（拉车） | 0.18 | 0.18 | |
| vy=0.3（横移） | 0.23 | 0.19 | 万向轮随动 |
| 200 kg 重车、零命令 | 0.024 | 0.014 | 圆柱轮时会摔倒，球形轮后站稳 |

机器人物理与 HOMIE 训练环境的逐项对照（`smoke` 自动输出）：关节顺序、armature、PD 刚度/阻尼、49 个 body、接触偏移、dt、decimation 全部相等；质量与 URDF 差 2e-7 kg（训练目录里的 `physics_config.json` 因开启了质量随机化而与 URDF 相差最多 0.44 kg，属正常）。

仍未验证 / 留待阶段 B：推车真实质量、惯量与万向轮参数（目前是估值，靠随机化覆盖）；Isaac 固定关节刚性 vs MuJoCo 软 weld（可选加柔性，重车测试表明刚性已不再导致摔倒）；分钟级长时程与 1.0 m/s 高速；推车随机化尚未实现。

参考的开源做法：
- FALCON（LeCAR，L4DC 2026）：不仿真物体，训练时在末端施加随机外力并做力课程（实机拉车 0–100 N），下肢/上肢双智能体。
- Partial AMP 推车（Go2 + WidowX，Isaac Lab → MuJoCo，arXiv 2603.26659）：仿真中放真实带被动轮的推车，夹爪与把手靠**接触**保持（最常见失败是脱手），高层控制器给推车 (v, ω) 命令，随机化摩擦 0.4–1.2、车质量 ×0.5–1.5、轮阻尼 0–0.02，并做跨仿真器压力测试。
- CAIMAN（Go2 推箱/推车，Isaac Sim）：分层策略，推车为 2 定向 + 2 万向轮，随机化质量与摩擦，实机用 mocap 观测。
- Humanoid-Gym：Isaac Gym 训练 + MuJoCo sim2sim 的框架，附 60 kg 推车演示。

## 阶段 B1：预设位姿校准与抓握力测量

问题：预设腿角原来取 HOMIE 的**默认**关节角（−0.1 / 0.3 / −0.2），而策略实际站姿更屈膝、骨盆低 3.3 cm；双手被把手固定后，这个高度差变成每手约 50 N 的静态预载（MuJoCo 里 43 N）。

工具（Isaac Lab，只有机器人）：

```powershell
.\run_one_stage_pushcart.ps1 stance --num_envs 8 --settle_s 3 --average_s 3 --viz none          # 只测量, 写 config/homie_stance.yaml
.\run_one_stage_pushcart.ps1 stance --num_envs 8 --settle_s 3 --average_s 3 --apply --viz none  # 写回 initial_pose.yaml + 重建 cart.urdf
.\run_one_stage_pushcart.ps1 pack --viz none ; .\run_one_stage_pushcart.ps1 mj build           # 之后必须重建 USD / MJCF
```

`stance --apply` 做四件事：(1) 腿角 = 零命令站立 3 s 的均值，左右对称化、踝关节用 FK 把脚底在世界系校平；(2) 骨盆高度 = 实测站高，骨盆俯仰 = 实测均值烘进 `base_quat_xyzw`（`model_60000` 前倾 1.1°，手在骨盆前 0.3 m，忽略它会让把手偏 6 mm ≈ 10 N 预载）；(3) FK 贴地高度比实测低 3 mm，脚在 reset 时悬空这么多、一步物理即落地；(4) `robot.upper_hold_offset` = 无车时上肢在重力下的下垂量（腕 roll 0.054 rad、腕 yaw 0.034、肩 pitch −0.028、肘 −0.020），环境把 PD 目标抬高这个量，使手在预设位置静止。HOMIE 检查点更换后重跑一次即可（幂等）。

**冻结的底层检查点**：`paths.HOMIE_FROZEN_CHECKPOINT = model_60000.pt`（2026-09-17 决定），所有工具默认用它；`model_60000` 站高 0.7592 m（比 59200 高 8 mm），把手高度 0.9172 m。校准后零命令：Isaac 每手 5–6 N（静态 < 3 N，根高度 0.7590 vs 预设 0.7592），MuJoCo 1.8 N。

抓握力的测量方式（`PushCartSimEnv.grasp_wrench()`）：读 PhysX 掌部 link 的 incoming joint force（母关节传给掌的六维力），减去掌 + 手指子树的重力（0.70 kg），就是手施加给推车的力；无车时自检约 2.4 N（HOMIE 原地踏步的惯性项）。校准前后（Isaac，零命令，4 env，最后 2 s）：

| | 骨盆高度 | 每手平均力 | 静态分量（世界系） | 基座漂移 |
| --- | --- | --- | --- | --- |
| 校准前 | 0.764（预设 0.785） | 50 N | (−0.2, ±11.8, −48) N | 0.049 m/s |
| 校准后 | 0.751（预设 0.751） | 8 N | (0.4, −0.6, 0.5) N | 0.049 → 评估 0.026 m/s |

MuJoCo 同场景 weld 力从 43–66 N 降到 3 N，且机器人完全静止。CPU 一致性测试 `tests/test_pushcart_parity.py`：MuJoCo 观测帧与 `HomieTaskCore.compute_observations` 逐位一致（2e-6）、PD/限位/动作缩放一致、`PushCartSimEnv` 未覆写 HOMIE 的观测/力矩/步进方法、上肢保持目标在限位内、检查点严格加载。

## 阶段 B2/B3：上层策略环境与训练入口

`homie_pushcart/task_env.py` 的 `PushCartEnv(PushCartSimEnv)`（设计见计划文档《阶段B上层策略RL设计》）：

- 动作 18：HOMIE 命令残差 3（×[0.4, 0.3, 0.4]，τ=0.15 s 低通，裁到训练范围）+ 上肢修正 15（臂 0.15→0.5 rad、腰 0.1→0.4 rad 随课程放开，3 rad/s 限速，硬限位裁剪）。零动作 = passthrough。
- actor 观测 124 × 5 帧 = 620：HOMIE 自己的带噪 76 维帧（命令/IMU/关节/腿动作）+ 任务命令 3 + 推车总质量 1 + 上肢力矩 15 + HIM 速度估计 3 + 命令与估计之差 3 + 把手相对基座位姿 4（噪声/延迟随课程）+ 上一动作 18 + 修正刻度 1。critic 追加 32 维特权量（车速、双手六维力、载荷质心、摩擦/阻尼、万向轮角、级别、真实基座速度）= 652。
- 奖励/终止/reset/随机化/18 级课程按计划实现；HOMIE 的 `_reward_*`/`episode_sums`/`reset_idx` 机制复用，`termination` 权重按"每事件"生效。
- 冒烟与基线：

```powershell
.\run_one_stage_pushcart.ps1 envsmoke --num_envs 16 --steps 300 --policy random --level 0 --viz none   # 复位不变量 + 终止统计
.\run_one_stage_pushcart.ps1 eval --num_envs 8 --level 6 --no_randomization --viz none                  # passthrough 基线 (无 --checkpoint)
.\run_one_stage_pushcart.ps1 train --num_envs 4096 --max_iterations 20000 --headless --run_name b4 --homie_checkpoint <HOMIE model_*.pt>
.\run_one_stage_pushcart.ps1 play  --checkpoint logs\rsl_rl\homie_pushcart\<run>\model_N.pt --num_envs 4
.\run_one_stage_pushcart.ps1 eval  --checkpoint logs\rsl_rl\homie_pushcart\<run>\model_N.pt --num_envs 16 --level 6 --viz none
```

`train` / `play` 走 NVIDIA 官方 `scripts/reinforcement_learning/{train,play}.py --rl_library rsl_rl`（子进程），任务 `Homie-G1-PushCart-Direct-v0` 由 `homie_pushcart.register_tasks` 注册；Hydra 覆盖照常可用（如 `env.task.rewards.scales.force_waste=-1.0`、`env.curriculum_level=4`）。passthrough 基线（level 6 刻度、无机器人随机化）：保持漂移 0.026 m/s；直推 0.3/0.6 偏差 −0.05；后退 0.3 只到 0.14；转向 0.3 rad/s 时偏航率 RMSE 0.23；25 kg 载荷与万向轮反向起步均无终止；抓握力 p95 ≤ 40 N。上层策略要补的就是速度偏差、转向、后拉与急停。

### 可学性验证（1024 env × 400 iter，HOMIE `model_59200` 冻结，18 min）

课程 0 → 12 级（后期在 10–12 级之间因 grasp_overload 8–30% 往返），最终每轮回报 28、跟踪奖励 1.73/2.0 与 0.84/1.0。与 passthrough 在同一评估设置（level 6 刻度、无机器人随机化、8 env）对比：

| 场景 | passthrough vx RMSE (偏差) / ωz RMSE | model_399 vx RMSE (偏差) / ωz RMSE |
| --- | --- | --- |
| 保持 | 0.026 (−0.01) / 0.072 | 0.055 (−0.02) / 0.054 |
| 直推 0.3 | 0.055 (−0.05) / 0.161 | 0.084 (+0.06) / 0.092 |
| 直推 0.6 | 0.066 (−0.05) / 0.196 | 0.108 (+0.08) / 0.108 |
| 后退 0.3 | 0.168 (−0.16) / 0.126 | 0.110 (−0.09) / 0.075 |
| 左转 / 右转 | 0.052 / 0.227 · 0.059 / 0.230 | 0.105 / 0.119 · 0.087 / 0.165 |
| 25 kg | 0.062 / 0.142，无终止 | 0.093 / 0.104，切向力 p95 111 N → 全部 grasp_overload |

结论：偏航跟踪与后拉明显改善（转向 RMSE 减半），速度偏差从 −0.05 变成 +0.06 的过补偿，零命令下有抖动，抓握力 p95 从 10–25 N 升到 50–70 N（策略在用手臂，但沿杆切向分量偏大，25 kg 时触发终止）。这是 1000 万步的短训，用来确认管线可学而不是验收；正式训练要看 `force_waste`/切向力惩罚是否足够，以及 `stand_still` 能否压掉零命令抖动。同一检查点在 MuJoCo 里（`mj eval`）给出同量级的结果（直推偏差 +0.04/+0.06、转向 ωz RMSE 0.13/0.15、25 kg 抓握 108 N），说明上层管线在两个仿真器里是一致的。

课程判定节奏已改为最多每 10 s 仿真时间评一次窗口（`curriculum.eval_interval_s`）：验证训练里 1024 个 env 每 0.5 s 就填满 50 轮窗口，级别在一轮迭代内来回跳。

### 正式训练首跑暴露的数值爆炸（已定位、已修）

第一次 4096 env 训练（run `two_stage`）在 11–12 级出现单次迭代回报 −1e5、`rew_force_budget` −3e4/s、value loss 1e9 的尖峰，随后策略 std 从 0.31 一路涨到 1.0、回报从 38 跌到 20。用 `envsmoke --hl_checkpoint <model_1200> --hl_noise_std 1.0 --level 12 --force_trace 500` 复现：某一步双掌约束力 3.5 kN / 9.4 kN，推车被甩到 4 m/s、翘轮 11 cm。

**先说物理量级**：这套系统里没有任何机构能产生 kN 级的手—车作用力。G1 臂关节力矩上限 5–25 N·m、肩到手 0.3–0.5 m，臂能施加的力 ≤ 100 N/轴；机器人 35 kg、推车 12–62 kg，重力 350–600 N；驱动器、重力、摩擦都到不了 kN。刚体仿真里出现远超这些量级的约束力，只有一种来源：**某个量在一个 5 ms 子步内被强行不连续地改变**，约束力 = 冲量 / Δt。所以正确的做法是找到并去掉那个不连续的输入，而不是把力裁掉。

**归因实验**（`attr_A..I`，4096 env × 30 iter，12 级，同一随机种子）：

| run | 碰撞过滤 | 推车迭代 | 推力形式 | 轮卡死 | 车速踢 | `grasp_force_max_N` 峰值 | blowup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 全部旧行为 | 关 | 4/0 | 速度阶跃 | 力矩刹车 | 开 | 4.1e5 | 0.11 % |
| B 仅碰撞过滤 | 开 | 4/0 | 速度阶跃 | 力矩刹车 | 开 | 4.1e5（与 A 逐位相同） | 0.11 % |
| F 三项"修复"全开 | 开 | 16/4 | 力脉冲 | 力矩刹车 | 开 | **8.2e5** | 0.14 % |
| H F + 关车速踢 | 开 | 16/4 | 力脉冲 | 力矩刹车 | 关 | 1.4e5 | 0.09 % |
| G F + 关轮卡死 | 开 | 16/4 | 力脉冲 | **关** | 开 | **319 N** | **0** |
| I F + 卡死改为阻尼 | 开 | 16/4 | 力脉冲 | 阻尼 ×(1+19·frac) | 开 | **320 N** | **0** |

结论只有一条：**根因是"单轮卡死"扰动的实现方式**。旧实现把刹车写成显式力矩目标 τ = −k·ω，在 20 ms 控制步内零阶保持；轮子绕轴惯量 I = 7.2e-4 kg·m²，k·dt/I ≫ 2，一旦轮子离地或打滑（轻车、转向时常见），显式积分发散：转速每步变号并放大，落地瞬间摩擦把这个转速变成车速（几十到几百 m/s），焊在把手上的双手于是读到 1e4–1e5 N。碰撞过滤、求解迭代、推力形式对爆炸**没有任何影响**（B 与 A 的曲线逐位相同；F 比 A 更差）。之前文档里"机器人—抓握关节—推车刚性闭环 / 速度阶跃推力导致 kN 冲量"的解释是错的：`envsmoke --legacy_push` 实测 HOMIE 原生的 0.5 m/s 根速度阶跃在掌上只有 ~275 N（臂 PD 链是柔性的，速度失配由臂关节吸收，手端等效质量只有 1–2 kg）。

**修复**：卡死改为把该轮关节阻尼乘以 `1 + (wheel_stall_factor − 1)·disturbance_frac`，持续 0.3 s 后恢复。阻尼属于 PhysX 隐式关节驱动，任何数值都稳定；效果是滚阻增大 10–20 倍（0.3 m/s 时约 17–33 N 的拖滞），是"轮子卡到东西"的合理量级。首版阻尼实现有一处 int64 关节索引传给了需要 int32 的 Isaac Lab 写入核（第一次卡死事件即崩溃），已修。验证：`attr_I`（训练路径）与 `attr_G` 逐项一致；smoke 复现（同一 `model_1200` + std 1.0 噪声，1500 步 × 32 env，53 次卡死、236 次车速踢、7 轮推力）原始力最大 292 N，0 次 blowup；零动作 12 级 600 步各变体（基线 / `--legacy_push` / `--no_kick` / 全关）最大都在 275–278 N，即回合开始起步时的瞬态，与扰动无关。

**保留但重新定性的改动**（都不是修 bug，是建模选择或保险）：机器人—推车碰撞过滤 + `feet_caster_proximity`（脚在惩罚下始终 > 0.12 m，`--allow_contact` 实测无异常；若要保留"脚踢到轮子"这一真实失效模式可以去掉过滤）；推车求解迭代 16/4（对爆炸无影响，保留作为闭环收敛的余量）；推力 0.25 s 力脉冲（有限时长的推更符合物理，非稳定性需要）；抓握力裁剪 400 N 与 `blowup` 终止（保险丝，正常训练里不应触发，`term_blowup` 应恒为 0）。`envsmoke` 的归因开关：`--allow_contact --cart_iters POS VEL --legacy_push --no_force_clip --no_stall --no_kick --force_trace N`，结果里的 `disturbance_events` 给出本次运行实际发生的卡死 / 踢车 / 推力次数。之前跑坏的检查点不要 resume（value function 已被污染），重新开始训练。

### 正式训练（B4/B5）

`model_60000` 的 passthrough 基线（level 6、无机器人随机化、8 env）：保持漂移 0.008 m/s；直推 0.3/0.6 偏差 −0.05/−0.04；后退 0.3 只到 0.15；转向 0.3 rad/s 时 ωz RMSE 0.23；25 kg 无终止；抓握力 p95 ≤ 40 N（`logs/pushcart_eval_passthrough.json`，MuJoCo 版 `logs/pushcart_mj_eval_passthrough.json`）。

正式训练命令（首跑 `two_stage` 因上述卡死 bug 作废，修复后由用户重新启动，不要 resume 旧检查点）：`train --num_envs 4096 --max_iterations 20000 --headless --run_name b4_homie60000`，2.2 s/iter（约 12.5 h），GPU 10 GB。TensorBoard：`tensorboard --logdir logs/rsl_rl/homie_pushcart`；开跑后盯 `Episode/grasp_force_max_N`（应在几百 N，`two_stage` 里是 1e4–1e5）、`Episode/term_blowup`（应恒为 0）、`Policy/mean_noise_std`（应停在 0.3–0.5，不上爬）。验收：`eval --checkpoint <run>/model_N.pt --num_envs 16 --level 6`（B4 门：保持漂移 < 0.03、直推 RMSE < 0.05；B5 门：转向 RMSE < 0.1、25 kg 终止率 < 5%），再 `mj eval` 做 sim2sim 门禁（退化 < 30%），`export` 导出。

## 阶段 B6/B7：MuJoCo sim2sim 与导出

`homie_pushcart/mujoco_hl.py` 的 `MujocoHighLevelRunner` 在 MuJoCo 里复现 Isaac 端的整条上层管线（命令残差 + 低通 + 裁剪、上肢修正限速/限位、124 维 actor 帧的逐项拼装，观测噪声/延迟关闭），检查点通过 `hl_policy.HighLevelPolicy` 直接从 RSL-RL 5 的 `actor_state_dict` 重建：

```powershell
.\run_one_stage_pushcart.ps1 mj eval --level 6                                                   # MuJoCo passthrough 基线
.\run_one_stage_pushcart.ps1 mj eval --checkpoint logs\rsl_rl\homie_pushcart\<run>\model_N.pt --level 6
.\run_one_stage_pushcart.ps1 mj teleop --mode cart --hl logs\rsl_rl\homie_pushcart\<run>\model_N.pt --level 6 --port 8080
.\run_one_stage_pushcart.ps1 export --checkpoint logs\rsl_rl\homie_pushcart\<run>\model_N.pt      # TorchScript + ONNX + hl_interface.json
```

`mj eval` 与 Isaac `eval` 用同一份场景表（`evaluate.SCENARIOS`），结果分别写到 `logs/pushcart_mj_eval_<tag>.json` / `logs/pushcart_eval_<tag>.json`，sim2sim 门禁即两者跟踪 RMSE 的比值。带 `--hl` 的遥控里手柄给的是**任务命令**（推车坐标系，范围为 HOMIE 的 75%），页面额外显示上层输出给 HOMIE 的命令和上肢修正幅度。`export` 不依赖 Kit：把归一化层折进网络，输入 620 维张量、输出 18 维动作，`hl_interface.json` 逐段说明 124 维帧的含义/缩放，以及动作到 HOMIE 命令和 15 个上肢 PD 目标的换算（与 `PushCartEnv.step` 一致）；部署时与 HOMIE 的 ONNX 并行运行。

两个实现细节：`register_tasks` 不能 import 任何 isaaclab/pxr 模块（在 Kit 启动前执行，提前 import 会让所有 Kit 扩展加载失败）；HOMIE 的 `HIMActorCritic` 会把 `torch.distributions.Normal.set_default_validate_args` 覆盖成布尔值，`lowlevel.py` 加载后立即恢复，否则 RSL-RL 5 的高斯分布构造失败。

### 两级策略查看器：`run_two_stage_viewer.ps1`（= `mj view`）

看训练效果用这个，不用 Kit：MuJoCo 物理 + viser 渲染，CPU 上实时。默认加载 `logs/rsl_rl/homie_pushcart` 下**最新一次运行的最新 `model_*.pt`**，HOMIE 用冻结的 `model_60000`。

```powershell
.\run_two_stage_viewer.ps1                                  # 手动命令（手柄 / 滑条），浏览器打开 http://localhost:8080
.\run_two_stage_viewer.ps1 --auto                           # 依次跑完 9 个 eval 场景，页面上出统计表
.\run_two_stage_viewer.ps1 --scenario turn_left             # 只跑一个场景（结束后停在零命令）
.\run_two_stage_viewer.ps1 --run <运行目录名> --checkpoint <run>\model_1200.pt --level 17 --port 8090
```

页面里能做的：切换同一运行下的任意检查点；**"上层策略开启"开关**——关掉就是纯 HOMIE 直通，同一场景开/关各跑一遍就能看出策略学到了什么；场景下拉 + "运行所选场景" / "自动巡演"；手动模式的载荷与万向轮初始角；课程级别（决定臂/腰修正尺度）。三张 10 s 滚动曲线：任务命令 / 上层给 HOMIE 的命令 / 推车实际速度（vx 与 wz 各一张），以及两手抓握力的 |F| 与车坐标系分量（推 x / 侧 y / 竖 z，方向定义与 Isaac 的 `grasp_wrench` 一致：手对车的力）。状态栏给出残差、跟踪误差、当前级别下的过载阈值是否超限、机器人—车距离/偏航差/车倾斜；折叠面板列出 15 个上肢关节的修正量、原始动作和力矩；腕部的小球按 |F| 从绿到红（0–100 N）。每个场景结束在页面表格和终端各打一行与 `mj eval` 同定义的 vx/wz RMSE、偏差、距离、力 p95、失败次数。摔倒或几何终止（偏航差 > 45°、距离出界、车倾斜 > 15°）自动复位并计数。

首个可看的结果（`b4_homie60000` 的 `model_1800`，MuJoCo，上层开）：直推 0.3 / 0.6 的 vx RMSE 0.042 / 0.059（偏差 −0.02 / −0.03），转向 wz RMSE 0.091 / 0.106，25 kg 无终止，抓握力 p95 40–54 N。

## 实现说明

- `sim_env.py` 的 `PushCartSimEnv` 继承 `HomieIsaacLabEnv`，只覆写场景构建、上肢目标（保持抓握姿态 + 修正量）、外部命令、预设姿态复位；观测、PD 力矩、步进逻辑全部继承，`homie_isaaclab/` 不改。`task_env.py` 的 `PushCartEnv` 再在其上加上层策略的动作/观测/奖励/终止/课程；`direct_env.py` + `lab_config.py` 把它接到 Isaac Lab 的 `DirectRLEnv` 与 RSL-RL 5。
- `mujoco_env.py` 在 MuJoCo 中复现同一套观测/PD 管线（`G1RoughCfg` 的缩放、增益、动作尺度），`mujoco_teleop.py` 负责 viser 渲染、XInput 手柄与滑条回退。
- `lowlevel.py` 用原 `HIMActorCritic` 严格加载检查点。
- 机器人只有双脚有碰撞体（HOMIE 原样）；推车把手横杆为纯视觉件；机器人与推车的全部作用力经两处固定关节/weld 传递。
