"""Render measured reward contributions from the fresh Isaac reward audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=ROOT/'logs/bench/v53_reward_audit/report.json')
args = parser.parse_args()
r = json.loads(args.report.read_text(encoding='utf-8'))
assert r['result'] == 'PASS_EXECUTION' and r['parent_verified'] and r['child_exit_code'] == 0
d = np.load(args.report.with_suffix('.npz'))
t = np.arange(r['steps'])*r['dt']
fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True, layout='constrained')
for i, label in enumerate(('Forward velocity (m/s)', 'Lateral velocity (m/s)', 'Yaw rate (rad/s)')):
    v = d['velocity'][..., i]
    axes[i].fill_between(t, v.min(1), v.max(1), color='#3178ae', alpha=.15, label=f"{r['num_envs']}-env range")
    axes[i].plot(t, v.mean(1), color='#3178ae', linewidth=1., label='Cart mean')
    axes[i].plot(t, d['reference'][..., i].mean(1), color='#d66a22', linewidth=1.5, label='Reference')
    axes[i].set_ylabel(label)
    axes[i].grid(alpha=.2)
axes[0].legend(loc='upper right', ncols=3)
axes[3].plot(t, d['gate'].mean(1), color='#33885e', label='Guard mean')
axes[3].set_ylabel('Force-cost gate')
axes[3].set_ylim(-.05, 1.05)
axes[3].legend(loc='upper right')
for name, label in (('cart_vx_ripple', 'Forward ripple'), ('cart_vy_ripple', 'Lateral ripple'),
                    ('cart_yaw_ripple', 'Yaw ripple'), ('cart_response_error', 'Response error')):
    axes[4].plot(t, -d['reward_'+name].mean(1), linewidth=.9, label=label)
axes[4].set_ylabel('Motion penalties / s')
axes[4].legend(loc='upper right', ncols=2)
axes[5].plot(t, -d['reward_cart_wrench_guarded'].mean(1), color='#814890')
axes[5].set_ylabel('Guarded force penalty / s')
axes[5].set_xlabel('Time (s)')
for ax in axes:
    ax.grid(alpha=.15)
    for seconds, _, _ in r['protocol'][1:]:
        ax.axvline(seconds, color='grey', linewidth=.6, linestyle=':')
fig.suptitle('Fresh Isaac reward audit: frozen HOMIE, zero high-level residual\nExecution check; no claim of trained-policy improvement')
fig.savefig(args.report.parent/'reward_audit.png', dpi=140)
plt.close(fig)

names = {'start_cruise':'起步／直线', 'pulse_recovery':'外力脉冲／恢复', 'stop':'停车',
         'restart':'重新起步', 'turn':'转弯', 'turn_reverse':'转向换向', 'reverse_translation':'前后换向'}
o = r['overall']
lines = ['# V5.3 奖励实际运行审计', '',
    f"Isaac 新轨迹：{r['num_envs']} 个环境 × {r['steps']} 步，每步 {r['dt']:.2f} 秒，共 {r['num_envs']*r['steps']} 个样本。冻结 HOMIE、零高层残差；关闭自动随机扰动，在第 5 秒施加持续 0.2 秒的 −8 N 世界 x 方向车体力。无完整 RL 重训，以下不是新旧策略性能对比。", '',
    '采用条件力实验配置，使该项确实参与本次奖励核算；默认训练仍关闭条件力项。审计关闭 60 秒权重渐增，直接检查最终强度。', '',
    f"执行通过：父进程核验为 {r['parent_verified']}，仿真子进程退出码 {r['child_exit_code']}；重置事件 {o['reset_events']}。逐步核验实际总奖励与各项求和一致。", '',
    '| 实测量 | 均值，奖励／秒 |', '|---|---:|',
    f"| 跟踪＋运动进度正奖励 | {o['tracking_plus_progress_mean']:.6f} |",
    f"| 新分轴波纹＋持续响应惩罚绝对值 | {o['motion_penalty_mean']:.6f} |",
    f"| 同一轨迹按旧质量系数／尺度重算的惩罚绝对值 | {o['revision1_motion_penalty_same_trace']:.6f} |",
    f"| 条件力辅助惩罚绝对值 | {o['guarded_force_penalty_mean']:.6f} |",
    f"| 同权重、不加条件的力波纹惩罚绝对值 | {o['unguarded_force_penalty_same_weight']:.6f} |", '',
    f"新质量惩罚约为跟踪／进度奖励的 **{100*o['motion_to_tracking_ratio']:.2f}%**。这是这批轨迹上的平均占比，不保证任意课程级别或状态下的优化梯度占比。各阶段的运动惩罚触顶比例见 report.json 的 axis_cap_fraction。", '',
    '| 阶段 | 力门控均值 | 力门控最大值 | 前向波纹惩罚／秒 | 侧向波纹惩罚／秒 | 偏航波纹惩罚／秒 |',
    '|---|---:|---:|---:|---:|---:|']
for name, s in r['stages'].items():
    w = s['reward_per_s']
    lines.append(f"| {names[name]} | {s['gate_mean']:.3f} | {s['gate_max']:.3f} | {w['cart_vx_ripple']:.5f} | {w['cart_vy_ripple']:.5f} | {w['cart_yaw_ripple']:.5f} |")
lines += ['', '起步／恢复／换向阶段的表格均值包含其后参考已稳定的直线指令区间，实际运动仍可能存在跟踪误差。停车、转弯和转向换向阶段门控始终为 0；施加外力的首步也有即时关闭断言。门控恢复不代表物理接触已被证明完全稳定，仍需不同载荷和接触状态的消融。', '',
    '![车速参考、实际运动、条件力门控和奖励占比](reward_audit.png)', '',
    '原始数据：[report.json](report.json)、[report.npz](report.npz)。', '',
    '设计、公式、参数与训练判据：[奖励设计说明](../../../docs/push_cart_v53_reward_design.md)。']
(args.report.parent/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(args.report.parent/'comparison.md')
