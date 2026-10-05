# 2026-09-30 项目收敛清理

按用户要求移除独立V4/V5版本、训练模型和日志、专属bench／资产／工具，保留V3及更早版本、V6，以及它们仍需的HOMIE和共享测试设施。

## 已删除

- 独立包：`homie_pushcart_v5`、`v51`、`v52`、`v53`。
- 旧资产：`assets/g1_pushcart_v5`、`assets/g1_pushcart_v53`及旧候选MuJoCo场景缓存。
- 上述V5系列全部训练目录；原`homie_pushcart`中只删除明确的`2026-09-22_18-22-27_v4_speed_quality_scratch`，其他运行保留。
- V4/V5启动器、专项测试、bench、脚本及文档；含未按版本命名但专属旧实现的控制迁移、场景比对工具。
- 旧V5.3查看器（原PID39208）已关闭；该查看器原占用8080端口。

清单共443个不重叠目标、2337个文件，文件逻辑大小 **11,265,491,937 bytes，约10.49 GiB**。这个数字是删除文件的逻辑大小，不是承诺文件系统净释放量；本轮另保留了必要测试和清理记录。

清单与执行记录：[deletion_plan.json](../logs/maintenance/cleanup_v4_v5_20260930/deletion_plan.json)、[deleted_file_inventory.json](../logs/maintenance/cleanup_v4_v5_20260930/deleted_file_inventory.json)、[deletion_result.json](../logs/maintenance/cleanup_v4_v5_20260930/deletion_result.json)。所有删除目标先解析为工作区内绝对路径并检查reparse point，再由PowerShell `Remove-Item -LiteralPath`删除。

## 明确保留

- `HomieRL/homie_pushcart`、`homie_pushcart_v6`、`homie_isaaclab`、原HOMIE及RSL-RL。
- `assets/g1_pushcart`及原机器人／小车资源、初始抓握姿态、V3.2演示素材。
- 全部V6训练与评测；全部V3及更早训练；原HOMIE `model_60000`及其训练日志。
- 本轮前后核验878个保留训练／资产文件，大小和修改时间均相同；V6资产锁定哈希检查通过。
- 官方HOMIE权重、裸机测试及相关外部研究参考。

原共享包内有少量V4时代加入的兼容代码，例如命令调度和静止修正模块。V3/V6仍复用该包及其测试，且V6相关开关固定关闭；本次保留这些共用基础设施，没有为了消除版本字符串而改动V3/V6任务语义。

## 混合结果与共享脚本处理

旧`v3_v53_scurve_audit_20260928`中的V3数据已保全到 [v3_baseline_audit_20260928](../logs/bench/v3_baseline_audit_20260928/README.md)：21份独立数据逐字节复制、4份JSON仅保留V3条目，哈希与筛选内容均核验。原混合目录已删除。迁移清单见该目录的`manifest.json`。

`logs/bench/versions_20k`保留passthrough、V3、V3.1、V3.2的原始结果，删除V4/V5独立数据并重新生成摘要和图表。`scripts/bench_versions.py`保持早期版本测试能力，200Hz只读记录已迁入独立`pushcart_bench/instrumentation.py`，不再借用V5。

HOMIE本体／官方权重工具已取消V5导入及默认V5.3检查点依赖。裸机评测使用原legacy模型；推车明确指定V3.2或V6。原裸机数据保留，混合报告中V5.3推车组已删除并重建汇总。历史裸机结果保留其原物理元数据，不能把历史其他场景数字当作当前legacy场景的新结果。

## 清理后验证

- **114项测试、2个子测试通过**，包括V3／更早兼容性、V6网络、物理、奖励、导出、bench和新记录器。
- 实际V3.2快速bench：大S弯12.5kg成功，横向RMSE约3.208cm；两个阶跃和两个外扰测试无失败。
- 实际V6最新检查点：25kg、160控制步MuJoCo运行完成，无几何失败／跌倒。
- 清理后HOMIE裸机6个短程序完成；工具迁移还分别验证了官方v1、v2及V3.2/V6推车接口，共42个最终短程序，权重哈希未变。
- 清理后Isaac：V3.2/V6各4环境×32步，两次局部重置；141组原状态／奖励／观测／随机数／物理参数逐值一致，静态地面及隐式上身驱动仍通过。

完整记录位于 `logs/maintenance/cleanup_v4_v5_20260930/`；HOMIE工具迁移记录位于 `logs/bench/cleanup_20260930_homie_validation/`。

全量回归最初发现一个清理前就存在的测试断言不一致：本地HOMIE增加了默认关闭的`upper_body_randomization`可视化开关。已记录原文件与Git HEAD一致的证据；测试现在明确检查该扩展必须为False，并继续要求其他作者配置逐项相同。没有修改HOMIE、V3或V6的训练／仿真配置。

V6使用入口仍为 `run_v6_pushcart_train.ps1`、`run_v6_pushcart.ps1`、`run_v6_bench_viewer.ps1`；早期版本使用原 `run_one_stage_pushcart.ps1`、`run_two_stage_viewer.ps1`或bench的`two_stage:<检查点>`。
