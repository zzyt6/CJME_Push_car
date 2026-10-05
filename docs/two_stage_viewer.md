# 两级策略查看器：选择V3／V6模型

统一入口：

```powershell
.\run_two_stage_viewer.ps1
```

打开 **http://localhost:8080**。默认在 `logs/rsl_rl/homie_pushcart` 和 `logs/rsl_rl/homie_pushcart_v6` 中选择最新训练运行的最新检查点。当前完整V6运行默认加载 `model_19999.pt`。

在左侧“策略”中：

1. **训练运行**：选择V6或旧版训练目录，V3.2对应 `2026-09-21_16-56-46_v3_1_speed_quality_scratch`。
2. **检查点**：选择该运行的 `model_12000.pt`、`model_19999.pt`等。
3. 模型切换成功后会清空旧曲线和场景统计、命令归零并复位；载荷、课程级别和上层开关保留。
4. 手柄／滑条控制、固定评测场景、上层关闭的HOMIE直通对照均可继续使用。

V6使用自己的观测和独立HIM历史；旧模型使用原控制器。加载错误会显示提示并保留原模型运行，不会把不同版本的权重强行加载到同一个网络。

也可启动时指定模型：

```powershell
.\run_two_stage_viewer.ps1 -Checkpoint 'logs/rsl_rl/homie_pushcart_v6/2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt'
```

指定V3.2训练运行，自动选择该运行最后一个检查点：

```powershell
.\run_two_stage_viewer.ps1 -Run '2026-09-21_16-56-46_v3_1_speed_quality_scratch' -Port 8090
```

原Python形式参数继续支持，例如 `--checkpoint`、`--run`、`--port`、`--payload 25`、`--scenario turn_left`、`--auto`。指定检查点优先于指定运行。默认课程级别为该模型配置的顶级，不把训练初始级别误当作模型保存时的能力。

## 推车显示

推车只显示原始OBJ可视化网格，采用不透明显示。碰撞盒、辅助把手圆柱和额外车轮基本体不再绘制；原始OBJ本身包含四个车轮。

“显示机器人碰撞体”只影响机器人，不会重新显示推车碰撞体。该修改只作用于Viser显示节点，没有改MJCF、USD、接触、质量、惯量或模型RGBA。共用此渲染器的bench页面和MuJoCo手动控制也使用同样显示规则。

## 验证

- 实际模型V3.2→V6→V3.2、同运行检查点切换、上层关闭／重开、载荷／级别保留通过。
- 无效V6模型加载失败后，原runner、renderer和物理状态保留且能继续步进。
- 显示前后所有MuJoCo模型数组逐值一致；车体可见节点只有非碰撞的 `cart_visual` 网格。
- 全量测试122项、2个子测试通过。
- 查看器HTTP服务已启动；本次浏览器自动化连接缺少授权token，未完成浏览器点击／截图验收。上述切换测试通过实际Viser服务端GUI事件和真实MuJoCo／网络运行完成。

此入口适合手动命令和固定速度场景。需要S弯／直线等路径巡线界面时，继续使用 `run_v6_bench_viewer.ps1` 或通用bench的`path --view`。
