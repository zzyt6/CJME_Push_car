# V6：V3.2 基线上的车速与双手载荷估计

V6 按 [实施计划](push_cart_v6_plan.md) 独立实现，基线是 V3.2 `2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt` 的保存配置。原 V3/V5 代码和结果保留为历史，V6 不加载旧版本高层检查点。

## 唯一物理与原奖励

MuJoCo 固定 `assets/g1_pushcart/mujoco/g1_pushcart.xml`。它原本就是每脚四个半径5 mm的球，位置与后来恢复四球的版本相同。V6同时保留该legacy资产的关节被动damping、干摩擦、armature、轮子／脚轮参数、碰撞、惯量、weld及原PD步进。V6没有scene/profile/inertia/grasp候选开关；资产哈希和物理标识写入结果。

Isaac直接复用V3.2：静态地面，腿部显式力矩，上身PhysX隐式位置驱动，原Kp/Kd和力矩限幅。载荷质量、COM、惯量写入顺序、随机化、课程、动作滤波、18维动作映射和所有28项奖励均使用原方法。完整配置固化在 `HomieRL/homie_pushcart_v6/config/baseline_v32.json`，运行不依赖历史日志存在。

上身位置驱动可以对接实机位置目标和电机PD接口；原HOMIE也是这套训练接口。隐式积分与实机伺服并不数值等价，V6保留已指定的基线，不擅自改成另一套官方增益。

奖励继续读原COM速度和重力补偿手部扳手代理。新动态手部标签只用于独立监督及新增Critic特权量，既不覆盖旧奖励信号，也不改变终止判断或地面类型。

## 确定的网络接口

| 通路 | V6约定 |
|---|---|
| 原观测投影 | 每帧124维仅删原索引101:104车把xyz，其他121维保持；相对yaw仍是原标量 |
| `policy`输入 | 121×5=605，原五帧直接拼接；质量、原上身力矩、HIM身体估速等仍保留 |
| `estimator`输入 | HOMIE76＋同一个相对yaw，77×6=462；无位置、真实车速或受力 |
| 小车HIM | 462→256→256→35，输出速度3和L2归一化latent32；target77→256→256→32，64原型 |
| 双手估计器 | 独立归一化的605维原历史＋预测车速3；608→512→256→128→12 |
| Actor | 605＋估速3＋HIM latent32＋双手12=652；整体统计归一化后512→256→128→18 |
| Critic | 原620完整历史＋原32特权量＋独立动态手标签12＋有效位1=665；保留统计归一化 |

Critic中的原车把位置仅用于训练。两个估计器和Actor均无法读取它。相对yaw的定义、噪声、延迟保持V3.2，未改成sin/cos，也未添加V5.3的视觉有效性／丢帧输入。

小车速度输出是原V3.2车根COM参考点的 `[vx,vy,wz]`，内部尺度 `[2,2,.5]`。双手输出为左、右掌原点处的 `[Fx,Fy,Fz,Mx,My,Mz]`，车头航向坐标，手对车方向，尺度每手 `[50,50,50,5,5,5]`。

HIM保留原ELU、Adam初始lr=1e−3、每个PPO minibatch跟随自适应lr、grad clip10、MSE+swap、temperature3和Sinkhorn .05/3次。按原HIM的t历史→t+1监督，终止用reset前的标签。新HIM不改变原五帧主策略历史。查看器把速度预测绘制在t+20 ms处，manifest也记录该预测步长；估速误差应与下一采样点对齐，手部估计仍对应当前采样。

双手头保留既有512/256/128主干、Huber、独立Adam1e−3和clip1，始终消费预测车速。标签由原物理上的手子树Newton–Euler只读计算获得，在50Hz控制采样时读取最后物理子步；不声称它是独立力传感器真值。原奖励仍使用旧代理。

PPO主干沿V3.2，两个监督优化器不接受PPO梯度。保存／恢复包含三个优化器、HIM原型和全部归一化统计。课程恢复保持原V3.2入口语义，未额外移植V5.3课程窗口恢复逻辑。当前实现按原单设备训练流程运行。

## 训练与运行

正式训练入口（不会自动加载V3.2权重；新网络需要训练）：

```powershell
.\run_v6_pushcart_train.ps1
```

默认4096环境、20000次更新、seed1、初始课程0。短程运行检查：

```powershell
.\run_v6_pushcart_train.ps1 -NumEnvs 16 -MaxIterations 2 -RunName v6_smoke
```

恢复已有V6检查点：

```powershell
.\run_v6_pushcart.ps1 train --checkpoint 'logs/rsl_rl/homie_pushcart_v6/<run>/model_1000.pt' --max_iterations 1000 --headless
```

训练足够的V6模型后，打开原bench页面：

```powershell
.\run_v6_bench_viewer.ps1 -Checkpoint 'logs/rsl_rl/homie_pushcart_v6/<run>/model_19999.pt' -Payload 29 -Speed 0.5 -PathName s_large
```

同一唯一场景可运行原path/step/robust测评：

通用bench当前默认策略也已改为`v6:latest`；显式指定V3/V5仍可运行历史对照。

```powershell
.\run_pushcart_bench.ps1 path --policies 'v6:logs/rsl_rl/homie_pushcart_v6/<run>/model_19999.pt' --paths s_large --payload 29 --v_nom 0.5
.\run_v6_pushcart.ps1 mujoco --checkpoint 'logs/rsl_rl/homie_pushcart_v6/<run>/model_19999.pt' --steps 500 --payload 29 --command 0.3 0 0
```

导出包含估计器和归一化器的JIT：

```powershell
.\run_v6_pushcart.ps1 export --checkpoint 'logs/rsl_rl/homie_pushcart_v6/<run>/model_19999.pt' --output logs/export/v6
```

部署输入分别为 `[N,605]` policy历史和 `[N,462]` HIM历史；`forward`输出18维动作，`physical_estimates`输出物理单位车速3维和双手载荷12维。导出脚本保存manifest并核验加载前后输出一致。

## 验证与效果边界

实际Isaac配对记录位于 [基线等价报告](../logs/bench/v6_runtime_contract_clean_20260929/report.json)。两独立进程各8环境×64步、两次局部重置，共141组原始数组逐值一致，包括质量／COM／完整惯量、620／652历史、奖励、状态、CPU/CUDA随机数。静态地面和上身隐式位置驱动通过实际检查。

MuJoCo检查覆盖编译参数及原步进一致、反复载荷重置；V6零残差头与旧passthrough在64控制步中的qpos/qvel/ctrl逐值相同。HIM与原实现、主PPO与原实现均有同权重／同输入的逐值更新对照。检查点、标签时序、特权隔离、归一化和导出有独立测试。V6及相关bench回归共58项通过。

实际训练完成16环境、2次PPO更新，再恢复同一检查点继续1次。PPO/力估计/HIM优化器步数分别40→60、80→120、40→60，三个模块权重继续变化；见 [续训核验](../logs/bench/v6_implementation_smoke_20260929/resume_checkpoint_check.json)。实际检查点完成160步MuJoCo运行和正式bench阶跃入口，均无跌倒；导出32组输入的动作及两估计输出最大差均为0，见 [导出manifest](../logs/export/v6_implementation_smoke_20260929/manifest.json)。

短程训练检查点仅用于验证代码能运行与恢复；不作为训练完毕的V6控制器，不能据此声称已经达到V3.2的S弯效果。正式效果应在同一唯一物理、路径、载荷和速度下，与固定V3.2检查点比较完成率、横向误差、侧漂、转向响应与双手负荷。
