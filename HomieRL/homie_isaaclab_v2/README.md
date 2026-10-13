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

## 已做的验证

- `tests/test_homie_v2.py`：配置差异、质量随机化语义、joint_power、任务注册、训练入口
- `run_homie_v2.ps1 smoke --num_envs 16 --steps 200`：PASS，并断言了每个刚体的 `maxAngularVelocity` 等于 `degrees(1000)`
- `train_homie_v2.ps1 --num_envs 64 --max_iterations 2`：通过官方 train.py 跑完 2 次 PPO 更新
- 以上只说明流程能跑通，还没有训练出 v2 策略
