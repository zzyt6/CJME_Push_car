# V5.3 的 bench Viser 查看入口

原来的 `run_pushcart_bench.ps1 path --view` 入口仍然保留。现在已接入 V5.3，沿用原有浏览器界面、路径跟踪器、速度／载荷滑条和开始按钮。

**最方便的启动方式**

```powershell
.\run_v53_bench_viewer.ps1
```

打开 **http://localhost:8080**。默认选择最新 V5.3 检查点，载荷 15 kg、巡线速度 0.3 m/s、直线路径、L13。模型会停在起点，点击绿色“开始”才运行。

“策略”下拉框有同一检查点的两个选项：**V5.3 原始场景**和**V5.3 新版四球足底**。待命时切换会重新摆位；运行中改好选项后再次点“开始”会按新选项重新运行。其余路径（S 弯、直角弯、U 形等）仍在“路径”下拉框里。

页面显示 vx、vy、wz 的**命令／实际／估计**曲线，以及路径误差、真实抓握力、HOMIE 实际指令和上身修正。路径任务中的命令来自原来的外层巡线控制器，不等同于独立恒定速度阶跃测试。

**指定我们当前一直对照的 model_15000**

```powershell
.\run_v53_bench_viewer.ps1 -Checkpoint 'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-37-10_v53_final_20k/model_15000.pt'
```

只看新版，或改端口、载荷、速度和初始路径：

```powershell
.\run_v53_bench_viewer.ps1 -Scene new -Port 8081 -Payload 15 -Speed 0.3 -PathName s_large
```

`-Scene original` 只加载原始场景，`-Scene both` 加载两套。默认 `latest` 仅搜索 `logs/rsl_rl/homie_pushcart_v53`，不会误选 V3/V4/V5.2；专用脚本先选定一个检查点，再给两套场景共用。

**继续使用以前的 bench 命令也可以**

```powershell
.\run_pushcart_bench.ps1 path --policies v53_original:latest v53_new:latest --view --level 13 --payload 15 --v_nom 0.3
```

或单独选择场景：

```powershell
.\run_pushcart_bench.ps1 path --policies v53:latest --scene_version new --view --level 13
```

`v53:<路径>`、`v53_original:<路径>`、`v53_new:<路径>` 也接受指定检查点，Windows 盘符会完整保留。

相同 V5.3 适配器也能运行原有的速度阶跃和鲁棒性 bench。例如不打开浏览器，比较一条直行阶跃：

```powershell
.\run_pushcart_bench.ps1 step --policies v53_original:latest v53_new:latest --steps vx_0_to_03 --payloads 15 --level 13
```

完成的运行写入 `logs/bench/<时间>_path/runs.json` 等目录，包含实际场景版本、检查点和三轴速度预测。专用 V5.3 查看脚本保存 50 Hz 控制采样。不同场景的比较明确记录物理配置，不把它们描述成完全相同的场景。

原来 `run_v53_pushcart.ps1 mujoco ... --viewer` 打开的是 MuJoCo 原生窗口；需要以前的 **bench Viser 浏览器界面**时，使用本页这两个入口。

验证：新增 V5.3 适配器通过两种场景的实际 bench 步进／遥测测试，并回归已有 bench 与场景测试。页面已核验场景下拉框、开始按钮、三轴预测曲线和完成结果；没有修改模型权重或训练配置。
