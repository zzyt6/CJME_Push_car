# SteadyTray 托架连接与 V6 场景对照

核查日期：2026-10-05。SteadyTray 固定提交 `ae781f9f43f9505118c2c434733c57ed1c09dbc3`，配套 IsaacLab fork 固定提交 `3e14846319ac07b5c430c8b6982745a8e7d1d95e`。

后续用户已明确采用 Dex3 双手闭合握住横向车把；近两年顶刊顶会的抓握接触与 sim-to-real 依据见 [Dex3—车把接触文献研究](dex3_cart_contact_literature_20261005.md)。该方案不把托架或 HAIC 的不同姿态交互当作直接替代。

结论：**机器人手腕与托架固定连接；托架与独立托盘通过接触相互作用。** 两种连接不能混淆。V6 则把机器人手掌直接固定连接到小车，省略了手/把手接触形成和滑脱的过程。

本次核查包括 README、训练任务注册、动作/PPO、环境事件、奖励、配套终止管理器、实际训练 USD 资产、URDF 和 MuJoCo XML。没有执行上游训练或实机测试，也没有修改 V6 物理或模型。

## 1. 托架究竟连接在哪里

URDF 的左侧定义如下，右侧对应相同结构：

```xml
<joint name="left_tray_holder_joint" type="fixed">
  <origin xyz="0.0414958 -0.0000038 -0.0000005" ... />
  <parent link="left_wrist_yaw_link"/>
  <child link="left_tray_holder_link"/>
</joint>
```

位置是腕部最后一个旋转关节之后的 `wrist_yaw_link`。托架原点相对该 link 约沿 x 正方向 4.15 cm；这是 link 坐标下的安装偏移，不是托盘尺寸或两手间距。托架没有独立电机、没有新增可动关节，相当于固定安装在腕部的工具，随肩肘腕运动，不是固定到世界地面。

来源：[固定提交 URDF 第 813 行](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/deploy/deploy_mujoco/g1_xml/g1_tray_holder.urdf#L813)，右侧第 1040 行；[本地副本](../.cache/steadytray_research_20261005/repo/deploy/deploy_mujoco/g1_xml/g1_tray_holder.urdf)。URDF 中每个托架质量 0.2 kg，训练环境另将托架质量随机化为 0.05–0.3 kg，因此不能把 URDF 数值当作整个训练中的固定质量。

更重要的是，本次实际下载并校验了训练使用的 `g1_side_tray_holder.usd`，不是只读部署 URDF 推断。23,083,236 字节的 Git LFS 资产 SHA-256 为 `ebd60f775c1bdc22c3d4997b9922138ccd3b91bd8632731a8adceb99e11fc89c`，与 LFS pointer 完全一致。

实际 USD 中两条 `PhysicsFixedJoint` 也分别连接 wrist-yaw 与对应 tray-holder，breakForce/breakTorque 为 `inf`。此处无限断裂阈值作用于**机器人工具安装**。第三条固定关节是相机安装，未发现把独立托盘固定到机器人的关节。

机器可读证据：[holder_asset_audit.json](../.cache/steadytray_research_20261005/holder_asset_audit.json)。

## 2. 托架与托盘不是同一个刚体

训练场景的连接关系为：

```text
左臂 → 左腕 → 固定连接 → 左托架 ── 碰撞/摩擦 ──┐
                                              独立托盘
右臂 → 右腕 → 固定连接 → 右托架 ── 碰撞/摩擦 ──┘
```

`TraySceneCfg` 单独创建 `{ENV_REGEX_NS}/Tray` 的 `RigidObjectCfg`，托盘是 0.254 × 0.352 × 0.018 m 的盒子，受重力和接触作用。spawn mass=0.1 kg 随后被 startup 随机质量覆盖为 0.3–0.7 kg。材料还配置 compliant contact stiffness=80000、damping=400，并随机化托盘摩擦参数。托架材料本身也采用较高摩擦（静摩擦范围 2–3、动摩擦 1.5–2.5）。实际摩擦合成取决于 PhysX material combine 规则，不能简单称某一个数为最终接触摩擦系数。

`reset_tray_pos` 是 **mode="reset"** 的放置事件：相对 pelvis 设置托盘初始位置/速度，然后交给物理仿真。不是每个控制步把托盘位置强行跟随机器人。托盘因此能移动、倾斜、滑动、失去接触或跌落。

来源：[steady_tray_env_cfg.py](../.cache/steadytray_research_20261005/repo/source/steadytray/steadytray/tasks/envs/steady_tray_env_cfg.py)，23 行创建托盘，38 行建立接触传感器，72 行 reset 放置；[events.py](../.cache/steadytray_research_20261005/repo/source/steadytray/steadytray/tasks/mdp/events.py) 的 `set_rigid_object_relative_to_robot`。

## 3. 托架不是一块普通手掌：实际碰撞几何有挡边

训练 USD 的每侧托架有独立可视网格，但真正启用碰撞的是 **3 个盒状凸包**。将节点旋转转换到资产根坐标后，可辨认为：

- 下方承托板：约 20 × 12 × 0.5 cm；
- 外侧挡板：约 20 × 0.5 × 3 cm；
- 端部挡板：约 0.5 × 4 × 3 cm。

两侧对称承托盘子，并通过底板、挡边与摩擦形成约束。因此不能把它描述为“没有机械辅助，仅靠裸手托盘”，也不能说只存在向上的支撑力：挡边接触能提供其他方向的接触力。但这些力必须来自实际碰撞几何，不能在接触已分离时继续由一个托盘—手腕固定关节传递。

MuJoCo 部署另将托架固连部分合并进 wrist-yaw body，使用 4 个盒子近似托架，含额外上沿结构；托盘 `plate` 有自己的 `<freejoint/>`，未建立到手腕的 weld。训练 USD 的 3 盒与 MuJoCo 的 4 盒并非逐项相同，不能称两套接触模型完全等价。

来源：[g1_tray_holder.xml](../.cache/steadytray_research_20261005/repo/deploy/deploy_mujoco/g1_xml/g1_tray_holder.xml) 第 237–240 行及 292–295 行，独立 plate 第 467–470 行。托架可视网格的形状不能替代碰撞体审计。

## 4. 其“第一阶段”要按官方命名区分

| 阶段 | 机器人/场景 | 训练内容 |
|---|---|---|
| Stage 1 `Pre-Locomotion` | 带左右托架的 G1；没有单独托盘 | 基础行走，上肢动作均值和标准差受额外正则约束 |
| Stage 1* 可选 `Locomotion` | 同一个无托盘机器人场景 | 全身行走，取消上述特殊正则 |
| Stage 2 `Steady-Tray` | 加入独立托盘和托架接触 | 从行走权重微调全身策略，加入托盘稳定/接触/对齐奖励 |
| Stage 3 `Steady-Object` | 再加托盘上的独立物体 | 冻结基础网络权重，训练 encoder/adapter；默认是 FiLM，对全身动作产生影响 |
| Stage 4 | 相同物理交互 | 蒸馏为可部署观测的 student encoder |

README 所谓 Stage 1 “upper body frozen” 在 PPO 源码中是对 14 个臂关节动作分布 `mean² + 0.5 std²` 加正则，不是将手臂关节改成 fixed，也不是输出维度缩成 12。Stage 2 没有沿用这项特殊正则，学习全身关节动作。该方法与 HOMIE 的“环境给上身目标、RL 仅输出腿部动作”不同。

Stage 2 中持续接触、低冲击、托盘水平、托架/托盘朝向相近等由奖励鼓励，并非几何硬锁。托盘跌到 0.7 m 以下被记录，训练配置 `track_only=True, track_only_delay=1.0`；配套 fork 会在连续违反达到 1 s 后 reset，并将其归入 time_outs。不能仅按 `track_only=True` 推断永不终止，也不能说触发即立刻终止。play 的 delay=0 则仅跟踪，不产生该延迟 reset。

来源：[任务注册](../.cache/steadytray_research_20261005/repo/source/steadytray/steadytray/tasks/envs/__init__.py)、[PPO 配置](../.cache/steadytray_research_20261005/repo/source/steadytray/steadytray/tasks/agents/rsl_rl_ppo_cfg.py)、[base/ppo.py](../.cache/steadytray_research_20261005/repo/scripts/rsl_rl/base/ppo.py) 第 346–374 行、[fork termination_manager.py](../.cache/steadytray_research_20261005/isaaclab_fork/source/isaaclab/isaaclab/managers/termination_manager.py)。

## 5. 对 V6 的直接含义

本次也打开了 V6 当前实际使用的 packed USD：左右 `GraspJoints` 直接连接 hand-palm 与 cart-base，breakForce/breakTorque 都为 `inf`；Cart 对 Robot 的碰撞对被过滤。当前 `g1_grasp.urdf` 只有左右脚踝 link 有 collision，手掌/手指没有物理接触几何。MuJoCo 构建器也禁用手指碰撞并建立手腕—车的 weld。

证据：[v6_connection_audit.json](../.cache/steadytray_research_20261005/v6_connection_audit.json)、[pack_usd.py](../HomieRL/homie_pushcart/assets/pack_usd.py) 第 172/192 行、[mjcf_scene.py](../HomieRL/homie_pushcart/assets/mjcf_scene.py) 的 `bake_fingers` 和 weld 构造。

| 比较项 | SteadyTray | 当前 V6 |
|---|---|---|
| 固定连接对象 | 腕部与托架工具 | 手掌与小车 |
| 操作物体与机器人 | 可分离的多点接触 | 预先建立的双手固定连接 |
| 力的产生 | 真实碰撞体、材料及运动共同决定 | PD、闭环固定约束及车轮/地面动力学共同决定 |
| 接触丢失/滑移 | 物理模型允许，策略需维持 | 当前固定连接没有滑脱机制；过载通过任务终止处理 |
| 姿态/受力能力 | 受托架几何和接触可行性约束 | 残差目标误差能在固定连接内积累负载，连接本身没有显式抓握力包络 |

**可以确定的建模局限：** V6 不验证双手是否能靠真实抓握承受其六维载荷。尤其沿把手滑移、绕把手转动、摩擦不足、松脱与真实接触压强分布都被固定连接简化掉了。任务中的力惩罚/过载终止没有把这种固定连接变成有限抓握接触。

**尚不能据此断言的结论：** 不能仅因用了固定连接就认定所有 V6 路径跟踪结果错误。若研究明确以稳定无滑移抓握为假设，它可以作为受约束推车控制模型。真正需要验证的是该假设是否匹配最终硬件，以及策略是否依赖真实手/把手无法承受的力矩或互相抵消的负载。

更不能直接把旧焊接场景峰值当作真实抓握最大承载能力，再用它标定新外力训练上限：那会把旧模型可能放大的负载带入新模型。先前整理的 118/132 N 等仍是旧场景下的需求记录，应待抓握模型验证后决定如何使用。

## 6. 建议的修订顺序

1. 先明确最终机械接口：裸手推、闭合 Dex3 抓把手、或腕部固定专用夹具。三者允许的接触方向、滑移和力矩不同。
2. 借鉴 SteadyTray 的分界：可以把真实工具固定安装在腕部，但工具/手指与把手之间采用符合硬件的接触；或者明确使用经过实验标定的有限刚度、有限承载、允许滑移/脱开的简化抓握模型。
3. **不能只删 V6 的 FixedJoint/weld**：当前手没有碰撞体，删除后并不会自动出现正常抓握。需一并建立接触几何、碰撞对、手指/夹具保持方式和初始化。
4. 先做静态已知方向加载、侧向滑移、扭矩、启停和转向验证，比较目标角/实际角/双手力及接触状态。原 V6 保留为固定连接基线，新场景使用独立任务和资产。
5. 在通过验证的新场景中重新采集推车姿态与负载分布，再确定 HOMIE 微调的上肢采样和外力课程。SteadyTray 的方法支持“预训练行走→带任务物体/接触适配→更高层残差”的思路，但不能照搬托盘的高摩擦与挡边参数到车把。

来源总清单：[source_manifest.json](../.cache/steadytray_research_20261005/source_manifest.json)。正文是源码与资产层面的判断，尚不包含新场景运行效果或实机抓握验证。
