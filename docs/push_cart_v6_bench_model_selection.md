# Bench viewer 的 V6 / V3 / passthrough 模型切换

运行 `run_v6_bench_viewer.ps1` 默认使用最新 V6 和 `composite_course` 综合赛道，物理环境固定为 legacy 参数＋四球足底。

```powershell
.\run_v6_bench_viewer.ps1 -Payload 15 -Speed 0.5
```

打开终端显示的 Viser 地址。在「模型选择」里先选 **模型 / 训练运行**，再选 **检查点**，点击绿色 **开始**。运行名称前标明 V6 或 V3 / 更早版本；切换运行时默认选该运行的最新检查点，也可选择相同训练轮数进行比较。

同一下拉框也提供 **passthrough · 直通 HOMIE**：不使用高层策略，命令交给冻结的 HOMIE，保持上身抓握姿态。选中后检查点显示「无需检查点」并禁用，点击 **开始** 生效；也可随时选回 V3/V6，再点击开始。

浏览下拉列表不会加载全部网络，也不会打断正在执行的控制；点击开始后才按需构建所选模型、从赛道起点复位。V3 使用原 `two_stage` 的 620 维历史，V6 使用自己的 605 维 Actor 历史和 462 维估计器历史。二者不混用网络结构或输入。切换后清空实时曲线、轨迹和策略历史，沿用界面选定的载荷与速度。已完成结果仍保留各自模型标识。

加载失败会在「模型选择」显示错误，并保留当前模型和场景。选择有效检查点后再次点击开始即可。

也可以直接指定旧模型或训练运行：

```powershell
.\run_v6_bench_viewer.ps1 -Checkpoint 'logs\rsl_rl\homie_pushcart\2026-09-21_16-56-46_v3_1_speed_quality_scratch\model_12000.pt'
.\run_v6_bench_viewer.ps1 -Run '2026-09-21_16-56-46_v3_1_speed_quality_scratch'
```

`-Checkpoint latest` 仍表示最新 V6。`-Run` 可使用完整目录或唯一运行名称。推车继续只显示视觉资产；模型选择不改变仿真物理、训练网络或奖励。

验证：`python -m unittest tests.test_bench_viewer_model_selection -v` 包含真实 V6 → V3 → passthrough → V6 加载、各自控制推理、物理参数一致性、轨迹清空及损坏检查点回滚检查；运行该测试时会自动使用空闲端口并关闭临时 Viser。
