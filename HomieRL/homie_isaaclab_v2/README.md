# HOMIE v2：修正移植版与官方的差异

2026-10-13 从 `homie_isaaclab` 复制而来，逐段对照官方 `legged_gym/envs/base/legged_robot.py` 做了修改。
原来的 `homie_isaaclab` 以及 `model_60000.pt` 都没有动。

- 任务：`Homie-G1-V2-Direct-v0`
- 训练：`.\train_homie_v2.ps1 --headless`（默认 4096 env / 100000 iter，与原版相同）
- smoke / 回放：`.\run_homie_v2.ps1 smoke|play ...`
- 日志：`logs/rsl_rl/homie_g1_v2/`、`logs/homie_isaaclab_v2/`

## 相对 baseline 的修改

| # | 内容 | baseline（homie_isaaclab） | 官方实际行为 | v2 |
|---|---|---|---|---|
| 1 | 载荷随机化 | 所有 link 先 ×U(0.8,1.2)，再给 torso 加 U(-5,10) kg、palm 加 U(-0.1,0.3) kg | 先写入载荷，随后 link-mass 循环用 `scale * default` 覆盖，载荷不生效 | 关闭（`randomize_payload_mass = False`） |
| 2 | pelvis 质量 | 一起缩放 | 循环从 body 1 开始，pelvis 不缩放 | 不缩放 |
| 3 | `max_angular_velocity` | 1000 原样写入 deg/s 字段，相当于把 link 角速度限制在 17.45 rad/s | 1000 rad/s | 换算成 `degrees(1000)` deg/s |
| 4 | `joint_power` 奖励 | 上肢使用真实力矩 | 上肢一项是 \|速度×目标角\|（单位不对，数值很小） | 只算腿部 12 个关节 |

在 Isaac Sim 里用 512 个 env 实测：

| | torso_link | pelvis | 整机 |
|---|---|---|---|
| baseline | 0.72 – 17.8 kg | 3.06 – 4.58 kg | 27.6 – 47.4 kg（均值 36.9） |
| v2 | 5.43 – 8.13 kg | 3.813 kg（固定） | 31.4 – 36.8 kg（均值 34.3） |

## 有意保留的与官方的差异

- **动作延迟**：官方计算了 `delayed_actions`，但实际用的是 `self.actions`，所以延迟没有生效。这里每个 env 每步按 0–3 个子步（0–15 ms）生效，符合 `delay = use_random` 的配置本意，也贴近真机的通信延迟。
- **reset 时**：会清零观测历史和 `actions`，并在 reset 后刷新刚体状态。官方会沿用上一回合的历史，IMU 也还是旧值。
- **超时**：用 `>=`，一回合正好 1000 步（官方是 1001 步）。
- **摩擦/恢复系数**：从预先采样的 64 个值（bucket）里选（PhysX 材质数量有限），官方是连续采样。

除以上几项外，奖励、观测、命令、上肢课程、PD 增益和 HIM-PPO 都与 baseline 相同（`tests/test_homie_v2.py` 会检查配置只差载荷开关这一项）。

## 平衡（站立不动）模型：`Homie-G1-V2-Balance-Direct-v0`

用途：二阶段抓握推车时站着不动。部署时 |速度命令| < 0.05 就从行走模型切换到平衡模型（和官方 GR00T Balance/Walk 的切换阈值一样）。它从行走模型 model_60000 微调而来，在 v2 物理下训练。

**高度**：和行走模型用同一套定义，高度命令都是 0.74。奖励要求"骨盆 − 脚踝 = 0.72"，而脚踝实际离地 0.0385 m，所以骨盆离地 0.759 m，正好是二阶段的起始站姿（0.7592）。按切换状态库统计，切换瞬间的骨盆高度均值是 0.7590 m，所以切换时不会掉高度。

**回合起点**（`G1RoughCfg.balance`）：
- 70% 从切换状态库开始：model_60000 走 1.5–4 s 随机命令，再经 0.4 s 滤波减速，在 |命令| < 0.05 时录下快照。快照包括根状态、关节、6 帧观测历史、上两帧动作、上肢课程状态，脚落在哪里就是哪里。
- 30% 从二阶段抓握站姿开始（initial_pose.yaml），腿角加 ±0.02 rad 噪声，双臂保持抓握姿态。
- 不约束腿角：走着停下时两只脚的位置各不相同，只要求脚不再动。

**奖励**：HOMIE 原有的各项全部保留，保证姿态风格和行走模型一致。另外：
- `stand_still`：−0.15 → −1.0（脚离地）
- 新增 `feet_still` −2.0：两只脚的速度之和，脚着不着地都算。只惩罚速度、不惩罚相对起点的位移，这样脚一旦移动过，就不会诱导策略再迈一步把它挪回去。
- 迈一步（脚抬起约 0.4 s、移动约 0.3 m）大约扣 1.0 分。一直错位站着到回合结束大约多扣 0.24 分。摔倒的损失远大于这两者。所以策略会尽量不迈步，但真要摔的时候仍然允许迈一步救回来。

**姿态项（第 2 轮加入）**：第 1 轮只用了上面两项防踏步的奖励。200 轮时踏步就基本消失了，但到 600 轮时站姿已经跑偏：骨盆 0.7552 m（低了 4 mm），前倾 −0.081 rad（向后仰 4.6°）。原有的 `orientation` 和 `tracking_base_height` 在这个误差量级上几乎不起约束作用，所以第 2 轮加了四项，权重按 model_60200 站立时的典型值来定：
- `stance_height` +2.0：exp(−(e/5 mm)²)，e 是骨盆**绝对高度**与推车起始站姿 0.759214 的差（第 3 轮改的）。第 2 轮用的是和 tracking_base_height 相同的"相对脚踝"定义，结果站高只有 0.755。原因是行走模型在踏步，脚踝平均离地 0.0385 m；平衡模型把脚放平以后，脚踝离地只有 0.0353 m，同样的相对高度就落低了 3–4 mm；
- `com_centering` −20：整机质心水平投影到两脚中心连线中点的距离的平方；
- `load_balance` −1.0：两脚都着地时的 |F左 − F右| / (F左 + F右)；
- `pitch_band` −5.0：前倾超出行走模型站姿（0.019）± 0.03 rad 的部分，按线性扣分。

前倾不硬钉在某个值上：训练完成后用 `stance --apply` 按平衡模型重新测量站姿，写回 initial_pose.yaml。

**随机化**：物理随机化与 v2 相同。另外：
- 推扰改为 0.2 m/s。依据是 LIPM 捕获点：ω₀ = √(9.81/0.7) = 3.7 s⁻¹，脚跟在脚踝后方 0.06 m，所以不迈步时向后最多能接住约 0.22 m/s。
- 上肢课程比例为 1.0（model_60000 在第 16300 轮达到 1.0）。
- 每次换上肢目标时，有 30% 的概率换成抓握姿态，模拟抓着把手的情况。

**流程**：
```powershell
.\run_homie_v2.ps1 bank --headless                                   # 录切换状态库 → .cache/homie_v2_balance/switch_bank.pt
.\train_homie_v2_balance.ps1 --headless                              # 从 model_60000 微调，默认 3000 轮
.\run_homie_v2.ps1 balance_eval --checkpoint <model.pt> --condition nominal   --num_envs 512 --seconds 8 --headless
.\run_homie_v2.ps1 balance_eval --checkpoint <model.pt> --condition disturbed --num_envs 512 --seconds 8 --headless
.\run_homie_v2_balance_viewer.ps1 -Port 8092                         # 浏览器打开 http://localhost:8092
```

查看器（`scripts/view_homie_v2_balance.py`）在 Isaac 里跑 1 个 env（CPU，无随机化），用 Viser 在浏览器中显示，操作方式和 one stage 的 teleop 一样：
- **速度命令**：用 Xbox 手柄（XInput）或滑条给出。左摇杆上下控制 vx、左右控制 vy，右摇杆左右控制 wz；A 键复位，B 键把命令归零。命令先经过 0.4 s 平滑（和录切换状态库时一样）。
- **自动切换**：平滑后 |命令| ≥ 0.05 时用行走模型（model_60000），< 0.05 时切换到平衡模型，并把速度命令置零。取消勾选自动切换可以做对照，看原模型在零命令下原地踏步。
- **起始状态**：从二阶段抓握站姿开始，手臂保持抓握姿态。可以打开手臂随机动作，也可以用按钮推一下机器人。
- **实时显示**：当前使用的策略、骨盆高度（与 0.7592 对比）、前倾、进入平衡后的迈步次数和脚掌漂移，以及相应的曲线。
- `-Demo`：自测模式，自动循环"以 0.6 m/s 前进 4 s，再归零 6 s"。

**基线**：model_60000 不微调，直接在零命令下跑 8 s，512 个 env：

| | 迈步次数 | 脚掌漂移 | 骨盆高度：起始 → 稳定 | 前倾 |
|---|---|---|---|---|
| nominal / 切换 | 32 | 0.59 m | 0.7591 → 0.7590 | 0.020 |
| nominal / 站姿 | 27 | 0.44 m | 0.7592 → 0.7589 | 0.020 |
| disturbed / 切换 | 29 | 0.55 m | 0.7592 → 0.7592 | 0.021 |
| disturbed / 站姿 | 29 | 0.55 m | 0.7592 → 0.7595 | 0.021 |

可以看到，行走模型在零命令下一直在原地踏步。微调的验收目标是：迈步次数和漂移接近 0，同时高度和前倾保持不变。

## 已做的验证

- `tests/test_homie_v2.py`：配置差异、质量随机化语义、joint_power、任务注册、训练入口
- `run_homie_v2.ps1 smoke --num_envs 16 --steps 200`：PASS，并断言了每个刚体的 `maxAngularVelocity` 等于 `degrees(1000)`
- `train_homie_v2.ps1 --num_envs 64 --max_iterations 2`：通过官方 train.py 跑完 2 次 PPO 更新
- 以上只说明流程能跑通，还没有训练出 v2 策略
