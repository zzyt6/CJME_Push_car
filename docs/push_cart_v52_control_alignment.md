# V5.2：保留冻结 HOMIE 的控制对齐实现

日期：2026-09-27。目标是让 Isaac 与 MuJoCo 使用相同的冻结 HOMIE、保存配置、观测/动作含义和逐子步 PD 计算。两引擎的接触和闭链求解仍有差异；此次没有重新训练。

实际结果见 [控制对齐评测报告](../logs/bench/v52_control_alignment/comparison.md)。最关键的实测是：同一 Isaac 工况中，原上身隐式驱动的 Fx RMSE 为 0.989 N，切换共同显式 PD 后为 14.379 N。旧模型依赖训练时的驱动；控制律相同不等于物理响应相同，也不保证旧估计器迁移成功。

## 两种驱动模式

| `upper_drive` | Isaac 上身 | MuJoCo 上身（`scene_profile="isaac"`） | 用途 |
|---|---|---|---|
| `training` | 保留旧训练的 PhysX 隐式位置驱动 | 原生位置 PD / implicitfast | 旧检查点默认；尽量匹配原训练接口，但不声称数值等价 |
| `explicit_pd` | 原生 Kp/Kd 置零，逐子步提交显式力矩 | motor，逐子步提交显式力矩 | FALCON 式共同力矩接口的独立对照 |

两种模式都保留原 HOMIE 腿部控制和权重。共同控制律为：

```text
q_target = q_default + action_scale * action
tau = clip(Kp * (q_target - q) - Kd * dq + effort_offset, -tau_limit, tau_limit)
```

冻结网络每 20 ms 提供腿部动作，PD 每 5 ms 重算一次。相同输入下已核验完整 456 维 HOMIE 历史、动作和 HIM 速度估计；V5.2 的双手 12 维估计器为独立网络，不是 HOMIE 的 HIM。

在 `training` 中，上身观测的 `applied_torque` 是子步开始时裁剪的 PD 近似，不是 PhysX 隐式求解后的电机力矩。在 `explicit_pd` 中，它对应向执行器提交的裁剪力矩。V5.2 使用该读数和静力先验，两种驱动下这些输入与手力标签的关系可能改变。

## 代码入口与配置恢复

- [control_contract.py](../HomieRL/homie_pushcart_v52/control_contract.py)：共享 PD 公式、驱动模式校验、诊断用名义配置和 JSON 控制报告。
- [policies_v5x.py](../pushcart_bench/policies_v5x.py)：对齐场景从检查点保存配置恢复动作/观测缩放、默认角度、增益、时钟及实际 HOMIE 路径；校验低层历史宽度与网络一致。显式传入的 HOMIE 路径优先于保存路径。
- [mujoco_env.py](../HomieRL/homie_pushcart/mujoco_env.py)：增加可选 `cfg` 注入；未传入时保持原接口行为。
- [sim2sim_scene.py](../pushcart_bench/sim2sim_scene.py)：按保存的时间步、重力、增益与模式生成隔离 MJCF；缓存键包含驱动模式和关键控制参数，当前 revision 为 5。
- [task_config.py](../HomieRL/homie_pushcart_v52/task_config.py) / [task_env.py](../HomieRL/homie_pushcart_v52/task_env.py) / [sim_env.py](../HomieRL/homie_pushcart_v52/sim_env.py)：保存并恢复 `actions.upper_drive`，旧配置缺字段时使用 `training`；显式模式关闭上身原生 PD 并实际施加随机化后的上身力矩。运行报告记录所选模式。
- [eval_pushcart_v52_sim2sim.py](../scripts/eval_pushcart_v52_sim2sim.py)：冻结模型评测，两端读取同一名义配置和保存的 HOMIE 路径，报告记录实际路径与控制参数。
- [validate_pushcart_control_transfer.py](../scripts/validate_pushcart_control_transfer.py)：绕过学习动作，重放相同目标与已知车体外力，保存状态、力矩观测和标签，检查两端输入相同后比较物理响应。
- [test_pushcart_control_contract.py](../tests/test_pushcart_control_contract.py)：完整 HOMIE 输入/输出、生产 PD 路径、限幅、配置恢复及旧检查点行为测试。

这组 MuJoCo 评测通过 `V5xPolicy` / `V5xRunner` 进入；旧版 `mujoco_hl.py` 播放器不等同于此桥接路径。`legacy` 保留历史场景，仅用于复现原跨版本基准，不与新控制分支混合比较。

## 复现本轮评测

在仓库根目录的 PowerShell 中执行。默认模型是本报告的 `model_19999.pt`；四条命令在新目录写入独立报告，不覆盖之前 12 工况的数据。

```powershell
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine mujoco --profiles isaac --upper_drive training --payloads 12.5 --commands forward03 --steps 500 --output logs/bench/v52_control_alignment
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine isaac --upper_drive training --payloads 12.5 --commands forward03 --steps 500 --output logs/bench/v52_control_alignment --viz none
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine mujoco --profiles isaac --upper_drive explicit_pd --payloads 12.5 --commands forward03 --steps 500 --output logs/bench/v52_control_alignment
& E:\envs\RLisaac601\python.exe scripts/eval_pushcart_v52_sim2sim.py --engine isaac --upper_drive explicit_pd --payloads 12.5 --commands forward03 --steps 500 --output logs/bench/v52_control_alignment --viz none
```

不传 `--upper_drive` 时恢复检查点保存的模式；旧检查点恢复为 `training`。实际 Isaac 输出为 `physx*.json`，MuJoCo 训练参数场景输出为 `isaac*.json`。

相同目标序列的物理响应检查：

```powershell
& E:\envs\RLisaac601\python.exe scripts/validate_pushcart_control_transfer.py --engine mujoco --upper_drive training
& E:\envs\RLisaac601\python.exe scripts/validate_pushcart_control_transfer.py --engine isaac --upper_drive training --viz none
& E:\envs\RLisaac601\python.exe scripts/validate_pushcart_control_transfer.py --engine mujoco --upper_drive explicit_pd
& E:\envs\RLisaac601\python.exe scripts/validate_pushcart_control_transfer.py --engine isaac --upper_drive explicit_pd --viz none
& E:\envs\RLisaac601\python.exe scripts/validate_pushcart_control_transfer.py --engine compare
```

相关测试共 48 项通过：

```powershell
& E:\envs\RLisaac601\python.exe -m pytest tests/test_pushcart_control_contract.py tests/test_pushcart_sim2sim_alignment.py tests/test_pushcart_parity.py tests/test_pushcart_v52_hand_estimator.py tests/test_pushcart_v52_labels.py tests/test_pushcart_v52_torque_sensor.py -q --disable-warnings
```

## 对后续训练的含义

训练配置已支持 `env.task.actions.upper_drive=explicit_pd`，会写入新 run 的配置；本轮未启动训练。旧检查点切换驱动后属于环境迁移测试。若采用这条路线，应先继续核验物理响应与跨引擎标签，再在选定驱动下适配上层和估计器，验证上身增益/延迟/负载随机化的作用；不能仅凭两端 PD 公式一致宣布具备 sim2sim 或 sim2real 能力。
