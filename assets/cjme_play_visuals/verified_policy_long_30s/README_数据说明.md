# 连续 30 秒双手整手受力原始数据

这份数据是从同一初始状态连续重放当前两级训练策略得到的 MuJoCo 原始轨迹，用于生成左右手各自的三维整体受力包络。它不是把原来的 5 秒数据复制或拼接，而是重新从 `t=0` 开始运行；运行期间没有自动复位、跌倒或几何终止。

## 采集条件

- MuJoCo 3.8.0，`implicitfast`。
- 物理步长 0.005 s（200 Hz），控制步长 0.02 s（50 Hz，每个控制步 4 个物理步）。
- 场景 `push_03`，任务命令 `[0.3, 0, 0]`，附加载荷 0 kg，万向轮角度 0，课程级别 17。
- 上层：`logs/rsl_rl/homie_pushcart/2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt`。
- 下层：`logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt`。
- 模型：`assets/g1_pushcart/mujoco/g1_pushcart.xml`。

共 6000 个物理求解样本，每只手均有 6000 个同步样本，时间为 `0.000–29.995 s`。每条记录对应一个 `mj_step`：`time_s/qpos/qvel/ctrl` 保存该步积分前状态，wrench 是该步求解后的笛卡尔整手 wrench。

## 力的含义和符号

`hand_wrench_world_about_cart_grip` 形状为 `(6000, 2, 6)`，第二维顺序为 `[left, right]`，最后一维为世界坐标系中的 `[Fx, Fy, Fz, Mx, My, Mz]`。它表示每只手通过 weld 抓握约束**作用在车上**的整手合力和关于车侧抓握点的力矩，单位分别为 N 和 N·m。

若需要画“车对手”的反力包络，取这个六维量整体乘以 `-1`；本目录的包络图就是这样处理的。它是 weld 约束传递的整手合力，不是逐指压力图，也不是手指表面接触力分布或理论最大承载能力。

## 文件

- `physics_trace_200Hz.npz`：原始轨迹。除上述 wrench 外，还保留 `time_s`、`qpos`、`qvel`、`ctrl`、`weld_multipliers_raw`、`task_command`、`high_level_action`、`low_level_leg_action` 和 `hand_order`。
- `trace_metadata.json`：模型/权重哈希、采样同步、坐标约定和运行终止状态。

原始数据未平滑、裁剪、插值或添加虚构零点。若需要把包络解释为“这段连续运动中观察到的力向量范围”，请使用同目录的三维包络结果：`../whole_hand_force_envelopes_long_30s/`。
