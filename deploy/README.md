# V8.1 推车策略真机部署（Unitree G1 + Dex-3，HOMIE 低层）

本目录是 V8.1 推车高层策略在真机上的完整部署：PC 端导出与验证，以及拷到机器人 Orin 上运行的代码。
V8.1 = 学生 `logs/rsl_rl/homie_pushcart_v8/2026-10-13_04-57-30_v8_fp_p2_v81m6000/model_299.pt`
+ 推车微调的 HOMIE 速度估计器 `2026-10-12_20-52-19_v8_1_roa_teacher_8k/homie_velocity_estimator_6000.pt`
+ 冻结的 HOMIE `logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt`
（三者的 sha256 写在 `robot/models/v81_config.json` 的 `provenance` 里）。

---

## 1. 为什么不能沿用 HomieDeploy 的 `deploy_policy.py` 加一个外挂高层

HomieDeploy 原来的结构是“HOMIE 进程从 `pedal_command` 读速度指令，手臂由遥操作经 `arm_action` 给”。V8.1 不能做成一个往 `pedal_command` 发指令的外挂进程，原因有四个：

1. **V8.1 改了 HOMIE 的输入。** HOMIE 行走网络收到的速度估计来自推车场景微调过的估计器，不是 HOMIE 自带的那个（HOMIE 自己的隐变量保持不变）。
2. **V8.1 改了 HOMIE 的输出。** 学生网络输出 12 维腿部残差，滤波后直接加在 HOMIE 的腿部动作上；HOMIE 下一帧观测到的是加完之后实际执行的动作。
3. **两层必须在同一个 20 ms 控制步里按固定顺序计算**（见第 3 节），否则观测和训练对不上。
4. **原部署有两处和训练不一致**（2026-10-12 已在 MuJoCo 里复现出原地碎步）：
   - `g1_control.cpp` 的增益和训练不同，见下表；
   - `lcm_agent.py` 把 wz 指令乘 0.25，训练时是 0.5。

| 关节 | 原 `g1_control.cpp` Kp / Kd | V8.1 训练与验证用的 Kp / Kd |
|---|---|---|
| 髋 pitch / roll / yaw | 150 / 2 | **100 / 2**（HOMIE 训练增益） |
| 膝 | 300 / 4 | **150 / 4** |
| 踝 pitch / roll | 40 / 2 | 40 / 2 |
| 腰 yaw / roll / pitch | 300 / 5 | **300 / 3**（Unitree 实机增益） |
| 肩 pitch / roll | 150 / 4 | **100 / 2** |
| 肩 yaw | 150 / 4 | **50 / 2** |
| 肘 | 100 / 1 | **50 / 2** |
| 腕 roll / pitch / yaw | 10, 10, 5 / 0.5 | **20 / 1** |

所以真机上 V8.1 的结构是：**一个 50 Hz 的 Python 进程同时跑 V8.1 学生和改过输入的 HOMIE**，再加一个改过增益的 C++ 桥接程序。HOMIE 网络和 V8.1 网络都被导出成 ONNX，机器人上不需要 torch。

---

## 2. 真机进程与通信

```
              ┌──────────────────────── G1 Orin ───────────────────────────────────────────────┐
 Unitree      │                                                                                 │
 遥控器 ──DDS─┼─► g1_control_v81 (C++, 200 Hz)  ◄──DDS rt/lowstate, rt/lowcmd──► 电机 / IMU     │
              │      │  ▲   训练增益 PD；看门狗 100 ms；L2+B 硬急停（阻尼）                        │
              │  LCM │  │ LCM                                                                     │
              │      ▼  │  body_control_data (q, dq ×29)  state_estimator_data (rpy, gyro)        │
              │      │  │  rc_command (摇杆 + 按键位图)                                              │
              │      │  │  ◄── pd_plustau_targets (腿 12 + 腰 3)   ◄── arm_action (双臂 14)          │
              │      ▼  │                                                                         │
              │  run_v81.py (Python, 50 Hz)                                                     │
              │    状态机 / 安全 / 日志                                                           │
              │    V81Controller: V8.1 学生 (ONNX) → HOMIE (ONNX, 微调速度估计) → 关节目标           │
              │      ▲ UDP 127.0.0.1:9870              │ LCM hand_action (手指 14)               │
              │      │                                 ▼                                         │
              │  tag_camera.py (头部 RealSense + AprilTag)   hand_control (HomieDeploy, 原样)  ──DDS─► Dex-3 │
              └─────────────────────────────────────────────────────────────────────────────────┘
```

| 进程 | 来源 | 作用 |
|---|---|---|
| `g1_control_v81` | `robot/unitree_sdk2/g1_control_v81.cpp`，由 HomieDeploy 的 `g1_control.cpp` 改出 | DDS 和 LCM 之间的桥接，200 Hz 发布状态、执行 PD。相对原版只改了 5 处：训练增益；100 ms 看门狗；L2+B 硬急停；`rc_command.mode` 带出按键位图；状态里带上四元数 |
| `hand_control` | HomieDeploy 原样 | 订阅 `hand_action`，控制 Dex-3（kp 0.5） |
| `tag_camera.py` | 本目录 | RealSense 彩色图 → AprilTag 36h11 → 位姿（solvePnP IPPE_SQUARE）→ UDP |
| `run_v81.py` | 本目录 | 主控：状态机、50 Hz 推理、安全检查、日志 |

关节顺序：电机 0–28 是 G1 29 自由度的顺序，HOMIE 的 27 个关节对应电机 `0–12, 15–28`。电机 13、14（腰 roll、pitch）不在 HOMIE 模型里，固定在 0。

---

## 3. 每个 20 ms 控制步做什么（`pushcart_v81/controller.py`）

`controller.py` 是 `HomieRL/homie_pushcart_v8/mujoco_runner.py` 的逐行移植，同一个类在 PC 上的 MuJoCo 里验证，再原样在机器人上运行。第 k 步：

1. **HOMIE 帧（76 维）**：由本步传感器加上一步的指令和一步实际执行的腿部动作组成，压入 6 帧历史。
2. **V8 帧（126 维）**：HOMIE 帧 + 任务参考 + 上肢力矩 + 相机相对偏航 + 上一步动作，压入 5 帧（策略输入 630 维）和 32 帧（适应模块输入 4032 维）两个历史。
3. 任务参考做一阶滤波（τ = 0.4 s）。空闲门控 gate = min(1, |参考| / 0.05)。
4. V8.1 学生 → 30 维动作。
5. HOMIE 指令 = clip(参考 + gate × 残差 × [0.4, 0.3, 0.5])，再做一阶滤波（τ = 0.15 s），写回 HOMIE 最新帧的指令位。
6. 上肢目标 = 抓握保持姿态 + gate × 修正量（腰 0.3、臂 0.4 rad），限速 3 rad/s，并限制在关节限位内。
7. HOMIE 行走网络（输入微调速度估计 + HOMIE 自己的隐变量）→ 12 维腿部动作；加上腿部残差 0.3·tanh(u)（经 40 ms 低通）→ 腿部目标 = 默认角 + 0.25 × 动作。

**HOMIE 帧（76）**：`[0:3]` 指令 × [2, 2, 0.5] | `[3]` 高度 0.74 | `[4:7]` 骨盆角速度 × 0.5 | `[7:10]` 投影重力 | `[10:37]` q − 默认角 | `[37:64]` dq × 0.05 | `[64:76]` 实际执行的腿部动作

**V8 帧（126）**：`[0:76]` HOMIE 帧 | `[76:79]` 任务参考 × [2, 2, 0.5] | `[79:94]` 腰 + 双臂 PD 力矩 × 0.02 | `[94]` 相机测得的小车相对偏航（rad） | `[95:113]` 上一步高层动作 | `[113:125]` 实际施加的腿部残差 | `[125]` 尺度 1.0

**动作（30）**：`[0:3]` HOMIE 指令残差 | `[3:18]` 腰 + 双臂修正 | `[18:30]` 腿部残差

**上肢力矩观测的定义。** 训练时这一项是最后一个 5 ms 物理子步的 PD 力矩，用的是观测前 5 ms 的关节状态。机器人上用桥接程序上一条 200 Hz 关节消息（正好早 5 ms）来算 kp(q* − q) − kd·dq。MuJoCo 实测：如果改用最新一条消息，0 kg 巡线误差会从 0.046 m 升到 0.050 m。

**相机相对偏航。** 训练定义是 `wrap(yaw(小车) − yaw(骨盆))`，两个都是世界系偏航。机器人上按下式组合：

```
R_world_cart = R_imu(rpy) · R_腰(yaw, roll, pitch) · R_torso_cam · R_cam_tag · R_tag_cart
rel_yaw      = wrap(yaw(R_world_cart) − IMU yaw − 标定偏置)
```

两个偏航都来自同一个 IMU，IMU 的偏航漂移会抵消。IMU 和腰部角度取的是图像拍摄时刻的值（环形缓冲），所以观测量等于“被相机延迟推迟了的真实相对偏航”，和训练的相机模型一致：延迟 40–100 ms，更新率 25–50 Hz，噪声 1.5°，丢帧 3%，丢帧时保持上一个值。

所有常数（增益、缩放、滤波、限位、保持姿态等）都由 `pc/export_v81.py` 直接从 MuJoCo 基准用的同一个 `MujocoV8Runner` 对象读出，写进 `v81_config.json`。机器人代码没有手抄任何常数。

---

## 4. 文件

```
deploy/
├── README.md
├── pc/                                 在训练 PC 上运行（RLisaac601 环境）
│   ├── export_v81.py                   导出 ONNX + v81_config.json，并与 torch 逐值比对
│   ├── verify_sim2sim.py               部署核心 vs 发布管线：逐步对拍 + 36 次闭环基准
│   ├── test_run_v81_mujoco.py          用 MuJoCo 假机器人 + 脚本化遥控器，完整跑一遍 run_v81.py
│   └── test_cart_yaw.py                合成 AprilTag 图像 → 检测 → 位姿 → 相对偏航，与真值对比
└── robot/                              ← 整个目录拷到机器人，例如 ~/pushcart_v81
    ├── run_v81.py                      主控程序
    ├── tag_camera.py                   相机进程
    ├── calibrate_cart_yaw.py           小车偏航零位标定
    ├── setup_robot.sh                  一次性安装：pip 依赖 + 在 HomieDeploy 里编译 g1_control_v81
    ├── start_v81_tmux.sh               一键开 4 个 tmux 窗口
    ├── requirements.txt
    ├── config/robot_v81.json           真机参数：摇杆限幅、相机外参、标签安装方式、手指姿态、安全阈值
    ├── models/                         由 export_v81.py 生成：v81_student.onnx, homie_v81.onnx, v81_config.json
    ├── pushcart_v81/
    │   ├── controller.py               推理核心（只依赖 numpy + onnxruntime）
    │   ├── robot_io.py                 LCM 收发
    │   ├── cart_yaw.py                 相对偏航的组合、时间对齐、异常值过滤
    │   └── lcm_types/                  从 HomieDeploy 拷来的 Python LCM 类型
    └── unitree_sdk2/
        ├── g1_control_v81.cpp          C++ 桥接程序
        └── CMakeLists_v81_append.txt   追加到 HomieDeploy/unitree_sdk2/CMakeLists.txt 的编译规则
```

---

## 5. PC 端：导出与验证（换模型后必须重跑）

```powershell
E:\envs\RLisaac601\python.exe deploy\pc\export_v81.py              # -> deploy\robot\models\
E:\envs\RLisaac601\python.exe deploy\pc\verify_sim2sim.py --full   # 约 8 分钟，结果写到 logs\validation\v81_deploy_verify.json
E:\envs\RLisaac601\python.exe deploy\pc\test_run_v81_mujoco.py     # 约 25 秒，实时运行
E:\envs\RLisaac601\python.exe deploy\pc\test_cart_yaw.py
```

2026-10-15 的结果：

| 检查 | 结果 |
|---|---|
| ONNX 与 torch 对比 | 学生 max 7.9e-6，HOMIE max 9.5e-6，HOMIE 图与 runner 的 HOMIE 调用误差为 0 |
| 逐步对拍（2131 步，12.5 kg） | 关节目标 max 3.0e-6 rad，V8 帧 1.2e-5，高层动作 5.3e-6：移植是精确的 |
| 闭环基准，36 次（composite course，0 / 12.5 / 25 kg × 4 个脚轮角 × 3 个种子） | 见下表 |
| `run_v81.py` 预演 | 状态机完整走通（RAMP → STAND → HOMIE → PUSH → HOMIE → L2+B），小车前进 2.9 m 并左转，无摔倒；周期 20.00 ms（p99 20.16），计算 1.6 ms（p99 2.6）；相机偏航和拍摄时刻真值的误差为 0 |
| 相机链路（合成图像，157 次检测） | 相对偏航误差平均 0.12°，p95 0.29°，最大 0.53°（训练噪声 1.5°）。漏检全部是标签出了视野 |
| 相机外参敏感度（偏置标定后） | 外参 pitch 错 5° → 偏航误差最大 0.39°；roll 错 3° → 0.18°；yaw 错 3° → 0.02° |

闭环基准，巡线 RMSE（m），格式为 平均 / 最大，均无失败：

| 驱动 | 0 kg | 12.5 kg | 25 kg |
|---|---|---|---|
| 发布管线（PyTorch runner） | 0.0457 / 0.0508 | 0.0219 / 0.0241 | 0.0215 / 0.0251 |
| **部署核心（真机代码）** | **0.0459 / 0.0518** | **0.0221 / 0.0256** | **0.0218 / 0.0254** |
| （对照）部署核心，力矩用最新关节样本 | 0.0496 / 0.0542 | 0.0233 / 0.0275 | 0.0205 / 0.0248 |

预演中发现并已修复 `run_v81.py` 的一个 bug：按键沿检测读到的是已经更新过的“上一帧按键”，导致任何按键都不会触发。

---

## 6. 机器人上的安装

1. 把 `deploy/robot/` 整个拷到机器人，例如 `scp -r deploy/robot unitree@<ip>:~/pushcart_v81`。`models/` 必须是最新导出的版本。
2. 运行 `HOMIE_DEPLOY=~/HomieDeploy bash setup_robot.sh`。它会装 Python 依赖，把 `g1_control_v81.cpp` 拷进 `HomieDeploy/unitree_sdk2/`，追加编译规则，然后编译出 `build/bin/g1_control_v81`。
   - **注意：`g1_control_v81.cpp` 没有在本仓库里编译过**（这里没有 unitree_sdk2 的库）。它只用到原 `g1_control.cpp` 已经用过的 API，外加 `WirelessController_::keys()`。第一次编译如果报错，以原文件为准修改。
   - onnxruntime：Jetson 上用 pip 的 aarch64 包或 NVIDIA 提供的版本，只用 CPU 推理。
3. 在小车上贴 AprilTag 36h11、id 0，量出黑色方块的边长。默认安装方式是 `vertical_rear`：贴在小车朝向机器人的那一面，竖直、正立。也可以平贴在车板上，标签上沿朝外，此时在 `config/robot_v81.json` 里把 `tag_mount` 改成 `horizontal_top`。
   - 头部相机向下倾约 48°。合成测试里，腰部转动 ±0.3 rad 时标签会出视野。贴之前用 `tag_camera.py --show` 在推车姿态下转一转腰确认，也要确认手不会挡住标签。
4. 相机外参 `cam_rpy_in_torso` 的默认值取自 g1.urdf 的 `d435_joint` (0, 0.8308, 0)。如果你的 D455 装法不同就改这里。外参误差的影响很小（见第 5 节），偏航零位靠第 7 节第 5 步标定。

---

## 7. 第一次上机：逐步检查

每一步通过了再做下一步。全程挂龙门架或有人保护。

1. **按键映射**：只开 `g1_control_v81`，运行 `python3 run_v81.py --print_buttons`，逐个按 START / R2 / A / B / X / Y / L1 / L2，确认显示的名字和实际按键一致。按键位图是按 unitree 的 `xKeySwitchUnion` 解码的；对不上就改 `robot_io.py` 里的 `KEYS`。
2. **增益与急停**：挂着机器人启动 `g1_control_v81`。3 s 后关节回零位并保持。按 L2+B，应当进入阻尼（关节变软），终端打印 `DAMPING`。之后重启程序。
3. **空跑（不发指令）**：`python3 run_v81.py --dry_run --no_camera`，按 START、R2。看终端上的计算时间（Orin 上应远小于 20 ms）和 IMU 读数是否合理。
4. **相机**：运行 `python3 tag_camera.py --tag_size <边长>`，确认输出的帧率 ≥ 25 Hz，capture→send 的延迟加上曝光大约 40–100 ms（训练范围）。
5. **偏航标定**：机器人以推车姿态握着小车，小车摆正在机器人正前方，运行 `python3 calibrate_cart_yaw.py`，结果写到 `calib/cart_yaw.json`。它会打印测量噪声（训练值 1.5°）和更新率。
6. **挂着跑 HOMIE**：`python3 run_v81.py`，按 START（3 s 内缓慢到站立 / 抓握保持姿态），再按 R2（进入 HOMIE，零指令）。腿开始踏步是正常的：model_60000 在零指令下会原地小步踏（已知现象，V8.1 就是在这个低层上训练的）。
7. **落地站立**：放下龙门架，HOMIE 零指令站稳。按 X 合手（默认 `fist`，即 hand_control 的握拳姿态，手指卡在把手上产生握力），把小车把手送进手里。
8. **PUSH**：按 A 进入 V8.1。按住 L1，左摇杆轻推向前。首测限幅在 `config/robot_v81.json` 里：vx ≤ 0.4、vy ≤ 0.1、wz ≤ 0.3。训练范围是 vx −0.6…0.9、vy 0.25、wz 0.6，代码会自动截到训练范围以内。松开 L1，参考值在约 1 s 内平滑回零。按 B 回 HOMIE（必须先停稳）。
9. 每次运行的日志在 `logs/v81_<时间>.npz`（每步的传感器、相机、指令、126 维帧、动作、目标、计算时间），可以拿回 PC 和仿真对比。

**遥控器总表**

| 按键 | 状态 | 作用 |
|---|---|---|
| START | WAIT | 3 s 缓慢到站立 / 抓握保持姿态（挂着） |
| R2 | STAND | 进入 HOMIE（零指令） |
| X / Y | STAND, HOMIE | 合手 / 张手（0.5 s 渐变） |
| A | HOMIE | 进入 PUSH（要求已合手、相机数据新鲜） |
| 按住 L1 + 摇杆 | PUSH | 左摇杆上下 = vx，左右 = vy；右摇杆左右 = wz；松开 L1 即指令归零 |
| B | PUSH | 回 HOMIE（参考值已归零时才允许） |
| L2 + B | 任意 | 急停：Python 停发，C++ 自己也锁存阻尼 |

**自动急停**（停发指令，C++ 看门狗 100 ms 后进入阻尼）：机器人倾斜 |g_xy| > 0.8（与仿真的摔倒判据相同），关节数据超过 50 ms 未更新，Ctrl-C。阻尼的意思是机器人瘫软，所以一定要挂龙门架或有人保护。

在 PUSH 中，相机数据超过 0.3 s 没更新时，任务指令被强制为 0（策略继续运行，用保持的偏航值）。所有关节目标在发出前都会截到 URDF 限位内。

---

## 8. 与训练 / 仿真不同或仍需实测的地方

| 项目 | 现状 | 怎么确认 |
|---|---|---|
| `g1_control_v81.cpp` | 未在本仓库编译 | 第 6 节第 2 步，第 7 节第 2 步 |
| 按键位图 | 按 unitree `xKeySwitchUnion` 写的 | 第 7 节第 1 步 |
| 相机延迟、更新率、噪声 | 训练是 40–100 ms、25–50 Hz、1.5° | `tag_camera.py` 和 `calibrate_cart_yaw.py` 的输出 |
| 相机外参 | 取自 URDF 的 D435 安装位 | 影响 < 0.4°（第 5 节），偏航零位由标定吸收 |
| 标签可见性 | 腰转动或手遮挡时可能丢失 | 第 6 节第 3 步；丢失时自动零指令 |
| 握力 | 默认 `fist`；仿真里握持被建模为弹塑性锚点，不会脱手（与用户的实机经验一致） | 实测小车是否在手里转动或滑动；可以改 `hands.grasp` |
| IMU | 按 G1 `rt/lowstate` 的 imu_state 是骨盆 IMU 来用（HomieDeploy、unitree_rl_gym 的 G1 都这样用） | 第 7 节第 3 步看 rpy 和 gyro 是否合理 |
| 腰 roll / pitch | 训练模型里没有这两个关节，真机上用 300 / 3 固定在 0 | — |
| 小车 | 训练覆盖载荷 0–25 kg，以及 V7 的脚轮、轮摩擦随机化范围 | 先空车，再加载 |
| HOMIE 零指令踏步 | model_60000 的已知现象，训练和基准里都存在 | — |

## 9. 换模型

只要重跑 `pc/export_v81.py --checkpoint <新学生.pt>` 和第 5 节的验证，然后把新的 `models/` 拷到机器人上即可。机器人代码不用改：所有常数都从 `v81_config.json` 读取，而这个文件来自 checkpoint 自己保存的 `params/env.yaml`。
