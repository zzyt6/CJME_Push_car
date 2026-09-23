# 左右手整体受力三维包络（连续 30 秒）

这组图把连续重放中每个时刻的整手 `(Fx, Fy, Fz)` 作为一个三维点，分别对左手和右手的实际点云计算凸包。采样来自 200 Hz、`0.000–29.995 s` 的连续轨迹，每只手 6000 点；没有把旧 5 秒数据复制到后面。

图中默认显示的是**车对手的反力**，也就是原始 `hand_wrench_world_about_cart_grip` 的前三项整体取负。两个子图使用同一坐标范围 `[-70, 70] N`，颜色表示合力大小 `|F|`，便于比较左右手。

## 结果

完整 30 秒：

- 左手：52 个凸包顶点、100 个三角面、体积约 `191957.800 N³`。
- 右手：53 个凸包顶点、102 个三角面、体积约 `204392.790 N³`。

另提供“去掉前 1 秒”的筛选版本（5800 点/手），用于单独查看启动段对外壳的影响；这不等于已经证明稳态。

凸包是这段轨迹观测到的外层几何范围。凸包内部包含几何插值，并不表示内部每个向量都被实际采样；它也不代表机器人的理论最大承载能力或所有接触方向的可行域。

## 文件

- `force_envelope_on_hand_full.png/.svg`：完整 30 秒的左右手静态图。
- `force_envelope_on_hand_after_1s.png/.svg`：去掉前 1 秒的静态图。
- `interactive_force_envelopes.html`：可旋转、缩放、悬停查看原始点坐标的交互图；下拉菜单可切换“车对手 / 手对车”和时间筛选。
- `force_points_and_hulls.npz`：原始点、车对手/手对车六维数据、凸包三角面和时间，可自行用 Python/Matlab 绘制。
- `envelope_metadata.json`：来源哈希、方法、符号、样本数和每个变体的统计。

原始六维轨迹在 [`../verified_policy_long_30s/physics_trace_200Hz.npz`](../verified_policy_long_30s/physics_trace_200Hz.npz)，说明见 [`../verified_policy_long_30s/README_数据说明.md`](../verified_policy_long_30s/README_数据说明.md)。
