# 双手整体受力三维包络（多工况）

本目录对应左右手**整体**三维合力包络。每个 200 Hz 物理采样点是一个世界坐标系中的 `(Fx, Fy, Fz)` 力向量；对所有真实采样点做三维凸包，没有平滑、补点或虚构方向。

## 数据来源

- MuJoCo 中回放已训练的 HL/LL 策略，原始记录是“手作用于车”的六维 wrench `[Fx,Fy,Fz,Mx,My,Mz]`，单位 N / N·m。
- 图中默认显示“车对手”的反力，即六维量整体取负；下拉菜单可切换回“手对车”。
- 共 5 个独立连续工况段、总 162.995 s、200 Hz、每只手 32,600 点；段之间调用 runner reset，不能把它们理解为单条连续物理轨迹。
- 工况包括前后/横向/转向/停车换向、15/25 kg 载荷以及不同脚轮朝向。详细命令、载荷、脚轮角度见 `envelope_metadata.json` 与源目录 `trace_metadata.json`。

## 文件

- `force_envelope_on_hand_full.png` / `.svg`：车对手反力的静态三维凸包。
- `interactive_force_envelopes.html`：可旋转、悬停查看原始点、切换受力符号与时间筛选。
- `force_points_and_hulls.npz`：原始点、凸包三角面，以及 `time_s`、`segment_time_s`、`scenario_id`、`task_command`、`payload_kg`、`caster_angle_rad` 等字段。
- `envelope_metadata.json`：方法、符号约定、统计、源哈希和工况信息。

## 如何解释

该凸包表示这些已采集工况中出现过的整体受力范围；凸包内部是几何包络，不表示每个内部向量都被实际采样，也不代表机器人的理论最大承载能力。
