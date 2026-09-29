# 推车 MuJoCo：原始场景与新版四球足底场景

已保留两个独立版本，适配当前 27 关节机器人、冻结 HOMIE 和 V5.3 高层。默认仍为原始版本。

要使用原来的 **bench Viser 浏览器界面**，运行 `.\run_v53_bench_viewer.ps1`，详见 [V5.3 bench Viser 说明](push_cart_v53_bench_viewer.md)。下文的 `--viewer` 指 MuJoCo 原生窗口。

| 选择 | 场景文件 | 足底接触 |
|---|---|---|
| `original` | [original.xml](../assets/g1_pushcart_v53/mujoco_scenes/original.xml) | 原始每脚一个凸网格 |
| `new` 或 `four_spheres` | [new_four_spheres.xml](../assets/g1_pushcart_v53/mujoco_scenes/new_four_spheres.xml) | 每脚四个半径 5 mm 的接触球 |

新版只更换脚底接触表示。它保持足地有效摩擦 0.75、implicitfast、5 ms 物理／20 ms 控制周期、机器人质量惯量、PD、双手 nominal weld 和推车参数。原始 XML 的 SHA256 仍为 `fcc619d0f8e07b007594f2142703830736efd4c3c5d9fe3b4f0e9067d10175a9`。

这些 XML 由当前 model_15000 保存的配置生成；运行入口会按指定检查点自动选择相应缓存文件。[文件清单](../assets/g1_pushcart_v53/mujoco_scenes/manifest.json)记录两版的来源、哈希与物理周期。

**查看或运行两套场景**

在项目根目录 PowerShell 中，先指定同一个已训练检查点：

```powershell
$ckpt = 'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-37-10_v53_final_20k/model_15000.pt'
```

运行原始场景，显示 MuJoCo 窗口：

```powershell
.\run_v53_pushcart.ps1 mujoco --scene_version original --checkpoint $ckpt --level 13 --payload 15 --command 0.3 0 0 --steps 1500 --viewer
```

运行新版四球足底：

```powershell
.\run_v53_pushcart.ps1 mujoco --scene_version new --checkpoint $ckpt --level 13 --payload 15 --command 0.3 0 0 --steps 1500 --viewer
```

`--command` 的三个数是车速任务 `[vx,vy,wz]`，`--payload` 是额外载荷 kg。去掉 `--viewer` 就是无窗口评测。省略 `--output` 时，结果自动写入带场景名和时间的独立目录，不覆盖旧评测；启动日志和 `report.json` 都标明实际场景版本。

如需固定输出目录，可添加 `--output logs/bench/my_original_run` 或另一条新目录。目录已存在时会拒绝覆盖。`train`／`play` 仍是原有 Isaac 入口，场景版本参数用于 `mujoco`。

**完整的 12 工况对照**

```powershell
& E:/envs/RLisaac601/python.exe scripts/eval_v53_estimator_sim2sim_status.py --checkpoint $ckpt --scene_version original --output logs/bench/v53_original_compare
& E:/envs/RLisaac601/python.exe scripts/eval_v53_estimator_sim2sim_status.py --checkpoint $ckpt --scene_version new --output logs/bench/v53_new_compare
```

代码中也可直接指定：

```python
runner = V53Runner(checkpoint, level=13, scene_version="original")
runner = V53Runner(checkpoint, level=13, scene_version="four_spheres")
```

**本轮已完成的验证**

固定相同 V5.3 权重，在 12 个推车工况中，新足底使车速估计 MAE 从 `[0.0944,0.0276,0.0920]` 降为 `[0.0313,0.0271,0.0505]`，前两项 m/s、第三项 rad/s；实际跟踪 MAE 从 `[0.1121,0.0347,0.1120]` 变为 `[0.0451,0.0440,0.0648]`。两版均完成程序、无跌倒，MuJoCo runner 的终止监督不等同于完整 Isaac 任务标准。

逐手力估计仍有约 5.6–9.8 N 的分轴 MAE，不能把本次速度改善解释成所有 sim2sim 问题已经解决。完整版本差异、单因素实验与曲线见[场景排查报告](../logs/bench/homie_mujoco_scene_versions_20260928_complete/comparison.md)。

23 项相关测试通过；原始版本和新版分别通过正式入口的 400 步无窗口运行。当前没有改动训练场景或权重，也没有将官方 v2 的 29 关节接口替换进推车系统。

需要按其他检查点配置生成两份 XML 时，写到一个新的输出目录：

```powershell
& E:/envs/RLisaac601/python.exe scripts/build_pushcart_mujoco_versions.py --checkpoint $ckpt --output assets/g1_pushcart_v53/mujoco_scenes_another_config
```
