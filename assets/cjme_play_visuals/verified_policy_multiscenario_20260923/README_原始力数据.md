# 多工况双手整手原始力数据

`physics_trace_200Hz_multiscenario.npz` 保存五个独立、分别标注的 MuJoCo 策略重放段，共 32,600 个 200 Hz 样本/手，时间覆盖 162.995 s。段之间调用 `runner.reset`；不能把全体样本理解为一条没有复位的连续物理轨迹。

`hand_wrench_world_about_cart_grip` 的形状是 `(32600, 2, 6)`，左右顺序 `[left, right]`，末维为世界坐标系 `[Fx,Fy,Fz,Mx,My,Mz]`。它表示每只手通过抓握 weld **作用于车**的整体六维 wrench，力单位 N、力矩单位 N·m。要画车对手的反力，六个量整体取负。

NPZ 同时保存 `time_s`、`segment_time_s`、`task_command`、`scenario_id`、`payload_kg`、`caster_angle_rad` 和 `hand_order`，可按工况筛选后自行绘制。五段的命令、载荷、脚轮角度和失败状态见 `trace_metadata.json`。

数据来自当前 HL/LL checkpoint 在 MuJoCo 中的实际重放，不是原始 Isaac/PhysX play 传感器记录；没有平滑、插值、补零或合成力点。
