"""Summarize fixed-weight scene interventions and the two supported scene versions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/bench/homie_mujoco_scene_versions_20260928_complete'
CART_OLD = ROOT/'logs/bench/v53_estimator_sim2sim_20260928'
CART_NEW = ROOT/'logs/bench/homie_mujoco_scene_versions_20260928_cart_four_spheres'


def vec(v):
    return ' / '.join(f'{x:.4f}' for x in v)


def main():
    results = json.loads((OUT/'results.json').read_text())
    validation = json.loads((OUT/'validation.json').read_text())
    assert validation['result'] == 'PASS_EXECUTION'
    old, new = [json.loads((p/'report.json').read_text()) for p in (CART_OLD, CART_NEW)]
    assert old['checkpoint_sha256'] == new['checkpoint_sha256']
    assert old['complete_cases'] == new['complete_cases'] == 12
    summary = {'body_interventions': results, 'cart_before': old, 'cart_after': new,
               'runtime_scene_versions': json.loads((ROOT/'assets/g1_pushcart_v53/mujoco_scenes/manifest.json').read_text())}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey='row')
    for col, name in enumerate(('ours_baseline', 'ours_geometry')):
        d = np.load(OUT/name/'reverse.npz'); t = d['time']
        axes[0, col].plot(t, d['truth'][:, 0], label='Actual body vx', lw=1)
        axes[0, col].plot(t, d['prediction'][:, 0], label='HIM estimate', lw=1)
        axes[0, col].plot(t, d['command'][:, 0], 'k--', label='Command', lw=1)
        axes[0, col].set_title(('Original foot convex mesh', 'Only foot contact geometry changed')[col])
        axes[0, col].set_ylabel('Body vx (m/s)'); axes[0, col].legend(fontsize=8)
        axes[1, col].plot(t, d['contact_point_speed'], color='#9467bd', lw=.8)
        axes[1, col].set_ylabel('Loaded contact horizontal speed (m/s)')
        axes[1, col].set_xlabel('Simulation time (s)')
        for row in (0, 1): axes[row, col].axvspan(0, 2, color='gray', alpha=.1)
    fig.suptitle('Fixed local HOMIE weights, same gains / friction / solver: reverse program\nContact speed is a load-weighted contact-point proxy, not an independent hardware measurement', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT/'foot_geometry_reverse.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey='row')
    for col, folder in enumerate((CART_OLD, CART_NEW)):
        d = np.load(folder/'payload_15_reverse.npz')
        for axis in range(3):
            ax = axes[axis, col]
            ax.plot(d['tracking_time'], d['tracking_velocity'][:, axis], label='Actual cart velocity', lw=1)
            ax.plot(d['estimator_time'], d['velocity_prediction'][:, axis], label='Estimated cart velocity', lw=1)
            ax.plot(d['tracking_time'], d['tracking_reference'][:, axis], 'k--', label='Filtered command', lw=1)
            ax.set_ylabel(('vx (m/s)', 'vy (m/s)', 'wz (rad/s)')[axis])
            if axis == 0:
                ax.set_title(('Original scene', 'New four-sphere scene')[col]); ax.legend(fontsize=8)
            if axis == 2: ax.set_xlabel('Simulation time (s)')
    fig.suptitle('Fixed V5.3 model_15000, 15 kg payload: only foot contact geometry changed', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT/'cart_scene_reverse.png', dpi=150); plt.close(fig)

    lines = ['# MuJoCo 场景版本差异：固定权重排查结果', '',
             '日期：2026-09-28。先核查实际编译模型，再做固定权重的单因素干预。已将原始场景和四球足底场景保留为两个可选择版本；使用方法见 [场景切换说明](../../../docs/push_cart_mujoco_scene_versions.md)。', '',
             '**当前速度迁移问题的首要已确认因素是足底接触表示。保持本地权重、关节数、增益、惯量、足地摩擦和积分器不变，仅把足底凸网格换成四球接触，本体估速误差显著下降；反向替换也破坏官方 v2 的稳定性。**', '',
             '| 项目 | 我们原始场景 | 官方 v2 原生场景 |', '|---|---|---|',
             '| 实际运行 MuJoCo 版本 | 3.8.0 | 同一个 3.8.0 |',
             '| 实际物理／控制周期 | 5 ms / 20 ms | 5 ms / 20 ms；其 XML 默认 2 ms 会被入口覆盖 |',
             '| 机器人运动关节／策略动作 | 27／HOMIE 腿部 12 | 29／腿部＋腰部 15 |',
             '| 单脚足底接触 | 一个脚部凸网格 | 四个半径 5 mm 的球 |',
             '| 足地滑动摩擦 | 显式 pair：0.75 | 默认组合：1.0 |',
             '| 积分器 | implicitfast | Euler |',
             '| 主求解器、锥、迭代次数 | Newton、pyramidal、100 | 相同 |',
             '| 机器人质量 | 34.394234 kg | 34.3942352 kg |',
             '| 活跃机器人碰撞体 | 两只脚 | 足底＋腿、躯干、手臂等；含地面共 53 个活跃 geom |',
             '| 关节被动 damping / frictionloss | 0 / 0 | 0.001 / 0.1 |',
             '| hinge armature | 0.01 | 0.01；官方 free joint 也继承了 0.01 |',
             '| 额外刚体阻尼 | 模拟 PhysX 的 linear/angular damping，各 0.01 | standalone 入口无同一外力阻尼路径 |',
             '| 腰 yaw 范围 | ±0.6 rad | ±2.618 rad，另有主动 roll/pitch |',
             '| hip / knee Kp | 100 / 150 | 150 / 200 |',
             '| 上身控制 | V5.3 为全身显式 PD；前轮裸机对照为上身 native position PD | 全身逐子步显式 PD |',
             '| 小车、双手闭链 | 推车版本有载荷、脚轮和两个 weld；裸机干预先移除了车 | 官方原生模型没有车／抓握 weld |', '',
             '两端总质量几乎完全一致。比较完整惯量张量而非主轴排序后，主要微小差异来自 1 g 固定 IMU 链接的合并；本轮没有发现一个“大质量／大惯量差”可直接解释此前现象。', '',
             '## 第一组：固定我们自己的 HOMIE，只改变一个场景因素', '',
             '每组 6 个程序 × 24 s，含前进、左右转、倒退、左右侧移；舍弃最初 2 s。三项是本体线速度 `[vx,vy,vz]` MAE，单位 m/s。各行均完成 6/6，无跌倒。', '',
             '| 设置 | 本体估速 MAE vx/vy/vz | 倒退 8–12 s 实际 vx |', '|---|---|---:|']
    labels = {'baseline': '原始场景', 'geometry': '**仅改四球足底**', 'friction': '仅将摩擦 0.75 改为 1.0',
              'integrator': '仅换 Euler', 'passive': '仅采用官方被动关节／自由关节参数',
              'geometry_friction_integrator': '四球＋摩擦 1.0＋Euler'}
    for suffix, label in labels.items():
        r = results['ours_'+suffix]; reverse = next(c for c in r['cases'] if c['scenario']=='reverse')
        lines.append(f'| {label} | {vec(r["estimation"]["mae"])} | {reverse["steady_vx"]:.4f} |')
    lines += ['', '倒退指令稳态约 −0.350 m/s。此前 Isaac 裸机本体 MAE 为 `[0.0408,0.0164,0.0257]`；只改足底后的 MuJoCo 已接近这个量级。继续把摩擦和积分器一起改掉没有进一步明显收益。', '',
              '编译后质量、COM、完整刚体惯性表示、关节限制、被动参数、执行器参数逐项验证完全相同，见 [geometry_only_checks.json](geometry_only_checks.json)。该实验改变的是脚底接触表示及相应接触点约束数量，不是把所有物理参数换成官方值。', '',
              '倒退程序中，每脚承重接触点（单点法向力 >5 N）的时间平均数由约 0.96/1.03 增至 2.87/2.88；对应的承重接触点水平速度代理从约 **0.167 降至 0.0096 m/s**。这支持接触支撑与滑移响应是重要机制。该统计使用控制时刻最后物理子步的接触记录，包含腾空时刻，不是独立硬件足底滑移测量。', '',
              '![本体单因素对照](foot_geometry_reverse.png)', '',
              '## 第二组：固定官方 v2，反向替换场景因素', '',
              '- 官方原生：6/6 完成，估速诊断 MAE `[0.0266,0.0309,0.0320]`。',
              '- 只将其四球足底替换为我们的凸网格：6 个程序均在约 2.14 s 失稳终止。这里不把短暂存活样本的 MAE 当作完整程序精度。',
              '- 单独把足地摩擦改为 0.75、积分器改为 implicitfast、或被动参数改为我们的值，仍全部完成；对应估速约 0.03 m/s 量级。',
              '- “网格＋0.75 摩擦＋implicitfast”组合更早失稳，没有足够的 2 s 后有效样本，明确不报告精度。', '',
              '两条方向的结果共同支持足底表示是实质因素。官方 v2 的原模型、权重和 29 关节接口与本地方案不同，因此不做跨模型的单纯精度排名。', '',
              '## 第三组：保持当前 V5.3 推车权重和抓握，仅改足底', '',
              '5/15/25 kg × 四种指令，共 12 个程序 × 24 s。权重固定 model_15000，保持原 nominal weld、URDF 惯量、显式 PD、0.75 足地摩擦及 implicitfast。两组均完成 12/12，无跌倒；runner 不包含完整 Isaac 终止监督。', '',
              '| 指标 | 原始场景 | 四球足底场景 |', '|---|---|---|',
              f'| 车速估计 MAE vx/vy/wz | {vec(old["speed"]["mae"])} | {vec(new["speed"]["mae"])} |',
              f'| 实际车速跟踪 MAE vx/vy/wz | {vec(old["tracking"]["mae"])} | {vec(new["tracking"]["mae"])} |',
              f'| 左手力 MAE Fx/Fy/Fz，N | {vec(old["force"]["mae"][:3])} | {vec(new["force"]["mae"][:3])} |',
              f'| 右手力 MAE Fx/Fy/Fz，N | {vec(old["force"]["mae"][6:9])} | {vec(new["force"]["mae"][6:9])} |', '',
              '车速前两项为 m/s，第三项为 rad/s。倒推主阶段实际车速从约 −0.074～−0.102 m/s 改善到 **−0.279～−0.291 m/s**，滤波参考均约 −0.3405 m/s。', '',
              '**速度估计与控制明显改善，但逐手估力仍有约 5.6–9.8 N 的分轴 MAE。足底差异没有解释全部闭链估力误差，后续仍须独立核查抓握、驱动与标签。**', '',
              '![推车版本对照](cart_scene_reverse.png)', '',
              '## 已落地的两个版本及边界', '',
              '用户要求保留两版后，新增正式 `scene_version` 选择：`original` 为默认原始场景；`four_spheres`／`new` 为四球足底。新版本使用本项目 27 关节机器人及推车，不是直接换成官方 29 关节机器人。', '',
              '两份 XML 位于 `assets/g1_pushcart_v53/mujoco_scenes/`。正式新场景与本轮推车探针的物理数组逐项一致；差别仅为诊断 geom／pair 名字。原场景文件哈希保持不变。23 项相关测试通过，两个正式命令行入口各运行 400 步、输出 300 个评分样本且无跌倒。', '',
              '四球方案是可选择的 MuJoCo 接触表示，不能仅因分数改善就宣称它已与 PhysX 或真机的足底接触完全等价。本轮没有改 Isaac 训练碰撞、没有重训，也没有扩大或缩小输入噪声。', '',
              '原始证据：[完整编译模型 inventory.json](inventory.json)、[12 组干预 results.json](results.json)、[执行验证](validation.json)、[推车新场景原始结果](../homie_mujoco_scene_versions_20260928_cart_four_spheres/report.json)、[正式版本运行核验](../mujoco_scene_versions_smoke_20260928/compiled_equivalence.json)。', '',
              '初次排查在 XML 球体省略 type 属性的读取处停止；修正评测脚本后完整重跑，本文仅使用 `_complete` 目录完成的全部结果，没有合并失败准备记录。']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(OUT/'comparison.md')


if __name__ == '__main__': main()
