# V5.2 按 FALCON 附录 A.3 改造估力输入

用户进一步指定使用**全身观测和全身控制历史**。因此保留下面 360 维 A.3 输入版作为对照，新增 `falcon_fullbody`：每帧再加入已经实际提交的 15 维上身关节目标，共 `5×87=435` 维。
关节位置/速度始终覆盖 27 个全身关节；两版差别在于是否包含上身控制目标历史，并非是否观测上身。
上身目标记录为 `(q_des-q_default)*dof_pos_scale`，下肢记录为裁剪后的 HOMIE 动作；均是已知控制信号，不是力矩观测。
这批数据已记录上身目标，能够直接配对真实下肢历史训练，不需要猜测或补造动作。数据审计确认新旧训练观测和标签逐项一致。

来源：[FALCON 附录 A.3](https://arxiv.org/html/2505.06776v2#A1.SS3)。该节的显式估力对照方案读取
`[q_(t-4:t), dq_(t-4:t), omega_(t-4:t), gravity_(t-4:t), a_lower_(t-5:t-1)]`，
预测末端三维力，然后把预测力送入下身策略，并用于上身 `Jᵀ F` 补偿。
它没有把关节力矩或雅可比伪逆先验作为估力输入。该附录引用 Portela 等人的
*Learning Force Control for Legged Manipulation*, ICRA 2024, pp. 15366–15372。

## 本项目实现

以下第一组列出 360 维 A.3 输入对照；435 维全身版本在每帧追加实际执行的 15 维上身目标，其他输入规则相同。

- 每帧 `q27 + dq27 + gyro3 + projected_gravity3 + previous_lower_action12 = 72` 维，最近五帧共 360 维。
- 关节与 IMU 使用 V5.2 同一份传感样本和固定缩放；动作来自实际裁剪后的 HOMIE 下肢输出，未乘 `action_scale`。
- 在观测时刻 t 配对刚执行的 `a_(t-1)`；重置时历史和上一动作清零，不使用未来动作。
- 估力输入不含关节力矩、DLS 先验、HIM 速度、FK、任务命令、上身目标或高层动作。
- 网络保持 ELU MLP `[512,256,128]`，独立 Adam/Huber 监督训练；归一化、早停和种子选择只使用 Isaac 训练/验证数据。
- 为推车任务保留双手掌原点处的 12 维力/力矩监督标签，再合成为把手中心 `[Fx,Fy,Mz]`。FALCON 附录是三维力，因此这里是其**输入和因果时序的任务适配**，不是完整复现。
- 冻结 HOMIE 和高层 Actor 的其他权重，只替换估力通道；高层 Actor 自身的旧观测仍包含原先验。这不代表整个控制器都移除了先验。
- 新估力输出通过独立 `falcon` TensorDict 键实际进入 Actor；遗漏该历史会报错，不能把旧 680 维帧误当成新 360 维输入。

FALCON 的 `Jᵀ F` 是**预测之后的力矩映射**；原 V5.2 的 DLS 是**预测之前从力矩构造输入**，二者作用方向不同。
本次按用户“优先仿真验证”的要求先验证估力输入，不同时加入未经验证的力矩前馈，以免把估力和补偿效果混在一起。

## 旧先验意味着什么

原实现求解 `w = (J Jᵀ + 0.001 I)^(-1) J tau`，且没有先扣除完整手臂重力、惯性、摩擦项。
因此它是力矩等效载荷提示，运动时还混有控制瞬态；不能视为准确末端测力。
网络输出由独立 MLP 产生，并不是直接输出先验或被先验精度限制。单独测先验误差只能判断先验本身质量，
不能证明其作为网络特征有益或有害。用户随后指定按 A.3 改造，所以本轮改为检验完整 A.3 输入方案，
未执行仅删除 12 维先验的消融；不得把 A.3 与全输入的差异归因于单一 DLS 特征。

## 数据与验证

用户要求重点区分网络和 sim2sim 环境的影响。本轮包含两类证据：大样本 Isaac 训练、留出测试与固定模型跨引擎/闭环测试；另有独立的等样本量两域拟合诊断。后者在每个引擎取同样的 8 个工况训练、2 个验证、2 个留出测试，分别在本域和另一域评分，按完整工况分组，避免相邻历史帧泄漏。该诊断允许使用 MuJoCo 标签，但产生的模型不接入控制、不用于主模型选择、不用于调整物理参数；主迁移网络仍只在 Isaac 上训练。

若两域分别训练都能达到较低留出误差，而固定源域网络交叉评测明显恶化，则支持存在域差异；这仍不能单独区分接触/约束动力学、输入分布、标签或模型对域特征的依赖，不能宣称已经定位某个环境参数错误。

旧数据没有存 HOMIE 下肢动作，不能用于训练此版本。重新在显式 PD Isaac 中采集 64 环境 × 4096 步，
种子 101；按完整环境划分训练/验证/测试，分别用优化种子 11/22/33 训练。
MuJoCo 不参与拟合、归一化、早停或模型选择。

1. 在两引擎各重新采集原行为策略的 12 个显式 PD 工况，同时保存旧输入和新历史；全输入模型与 A.3 输入模型在同一轨迹上评分。
2. 用 Isaac 验证集选择一个新模型，分别在两引擎实际接回 Actor，记录力误差、速度/转向跟踪误差、失败与有效样本数。
3. 源码测试验证输入隔离、时间对齐、reset、导出一致性、同状态两端观测一致性和预测确实进入动作。

结果目录：`logs/bench/v52_falcon_explicit/falcon_a3/`。完成的结果由该目录 `comparison.md` 和 JSON 记录。
原 `proprio` 480 维版本没有真实下肢动作历史，不能称为 FALCON A.3 输入版本。

```powershell
& E:\envs\RLisaac601\python.exe scripts/run_v52_falcon_estimator.py --output logs/bench/v52_falcon_reproduction
& E:\envs\RLisaac601\python.exe scripts/run_v52_transfer_validation.py --engine mujoco --groups explicit_pd_falcon explicit_pd_falcon_fullbody --output logs/bench/v52_falcon_reproduction
& E:\envs\RLisaac601\python.exe scripts/run_v52_transfer_validation.py --engine isaac --groups explicit_pd_falcon explicit_pd_falcon_fullbody --output logs/bench/v52_falcon_reproduction
```

训练入口现在默认训练两种输入各三个种子，共六个模型。`--features falcon_fullbody` 可只训练全身版本，`--dataset <已有配对数据目录>` 可复用真实数据。输出目录必须不存在，避免重跑覆盖记录。本轮早先的 `falcon_a3/training_complete.json` 记录首次三个 A.3 模型；后来三个全身模型各有独立目录和报告，历史清单未伪改成一次运行。

完整 Actor 导出暂不支持这个额外历史输入，会明确报错；独立估力网络支持 TorchScript。已实际导出全身模型种子 11：`logs/bench/v52_falcon_explicit/falcon_a3/export_fullbody_11/`，两种 JIT 接口与原模型随机输入核对最大差为 0，另有真实评测样本核查。包含可移植 PD 参数和 SDK 字段适配器；现成 NumPy 历史打包器仅适用于旧 `proprio` 480 维，不能拿来喂 435 维全身头。没有实机测试，也没有将独立头导出说成完整控制器部署。
