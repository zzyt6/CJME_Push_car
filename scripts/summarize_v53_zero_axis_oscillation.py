"""Explain zero-axis drift and gait-frequency ripple from recorded trials."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/bench/v53_zero_axis_oscillation_20260928'


def main():
    report = json.loads((OUT/'report.json').read_text())
    assert json.loads((OUT/'validation.json').read_text())['result'] == 'PASS_EXECUTION'
    filters = {}
    for scenario in ('straight', 'reverse'):
        group = [v for k,v in report.items() if k.startswith(scenario+'_')]
        assert len({v['initial_state_sha256'] for v in group}) == 1
        assert len({v['checkpoint_sha256'] for v in group}) == 1
        assert all(v['steps']==1200 and not v['fall'] for v in group)
        x = np.load(OUT/f'{scenario}_full.npz')
        v, ref = np.zeros(3), np.zeros(3)
        vf, rf = [], []
        for velocity, reference in zip(x['velocity'], x['reference']):
            v += .1*(velocity-v); ref += .1*(reference-ref)
            vf.append(v.copy()); rf.append(ref.copy())
        vf, rf = np.stack(vf), np.stack(rf)
        sel = (x['time']>=8)&(x['time']<12)
        filters[scenario] = dict(raw_std=x['velocity'][sel].std(0).tolist(),
                                 filtered_std=vf[sel].std(0).tolist(),
                                 tracking_score=np.exp(-np.square(vf[sel]-rf[sel])/.1).mean(0).tolist())
    (OUT/'filter_analysis.json').write_text(json.dumps(filters,indent=2),encoding='utf8')

    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey='row')
    for col,mode in enumerate(('full','no_yz_residual')):
        x=np.load(OUT/f'reverse_{mode}.npz')
        for row,axis in enumerate((1,2)):
            ax=axes[row,col]
            ax.plot(x['time'],x['velocity'][:,axis],label='Actual cart velocity',lw=.9)
            ax.plot(x['time'],x['prediction'][:,axis],label='Prediction',lw=.8)
            ax.plot(x['time'],x['reference'][:,axis],'k--',label='Task reference',lw=1)
            ax.set_ylabel(('vy (m/s)','wz (rad/s)')[row])
            if row==0:ax.set_title(('Full policy','Only lateral/yaw high-level residuals set to zero')[col]);ax.legend(fontsize=8)
            else:ax.set_xlabel('Simulation time (s)')
    fig.suptitle('Diagnostic only: fixed V5.3 model, same reset, 15 kg reverse program\nUpper-body actions and normal command-coordinate conversion are retained',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'zero_axis_action_probe.png',dpi=150);plt.close(fig)

    x=np.load(OUT/'reverse_full.npz');t=x['time'];sel=(t>=8)&(t<10)
    signals=[x['foot_load'][:,0]-x['foot_load'][:,1],x['cart_hand_moment'][:,2],x['velocity'][:,2]]
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True)
    for ax,v,label in zip(axes,signals,['Left-right foot normal load (N)','Hand resultant Mz (N m)','Actual cart wz (rad/s)']):
        ax.plot(t[sel],v[sel],lw=1);ax.set_ylabel(label)
    axes[-1].set_xlabel('Simulation time (s)')
    fig.suptitle('Full policy: foot loading, hand wrench and cart yaw share a 3 Hz peak\nSignals shown at recorded times, without phase alignment or smoothing',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'gait_hand_cart_signals.png',dpi=150);plt.close(fig)

    lines=['# 零侧向／转向指令为何仍有漂移和振荡', '',
           '日期：2026-09-28。固定 V5.3 model_15000、四球足底、15 kg 载荷、无观测噪声／随机扰动。先检查原图存档，再新采集直行和倒推各四种控制通道诊断，共 8 个 24 s 程序。没有重训或修改正式策略。', '',
           '**结论：实际存在两类问题——步态节律经双手闭链传递的运动波纹，以及高层侧向／转向残差造成的额外稳态偏差。预测误差还会使反馈偏小。零指令是优化目标，不是锁死相应运动轴的物理约束。**', '',
           '原图蓝线为真实车体 base-link 速度，橙线为估计器输出，黑虚线为任务参考。因此蓝线振荡不是只发生在预测器里的噪声。', '',
           '## 原图和新采集数据的口径', '',
           '新版场景原存档的 15 kg 倒推、8–12 s：真实 vy 均值约 0.0529 m/s、标准差 0.0163 m/s；wz 均值约 0.0498 rad/s、标准差 0.1147 rad/s。存在非零平均偏差，不应全部叫作零均值波纹。', '',
           '本轮配对诊断在每个试验前完整 reset MuJoCo 状态，四种模式的初始状态哈希相同。与原存档的连续多工况 runner 初始化路径不同，波形和数字会略有差别；下表只在本轮配对组内部比较。', '',
           '## 步态传递的证据', '',
           '- 完整策略直行时，左右脚载荷差、本体侧向速度、双手横向合力／合力矩与小车 vy/wz 都有约 2.25 Hz 主峰；倒推时相应主峰约 3.0 Hz。频率来自 8–12 s 窗口的 periodogram，分辨率 0.25 Hz。',
           '- 即使把高层所有残差置零，仅保留 HOMIE 和固定抓握目标，周期性偏航波纹仍存在。说明它不是纯粹由高层采样噪声生成。',
           '- 把上身残差固定为零，倒推 wz 标准差反而从约 0.114 增到 0.190 rad/s。这说明现有上身协同已在缓解部分摆动，不能把固定上身作为直接解决方案。',
           '- 频率同峰本身不能给各物理路径分配因果百分比。结合上述动作干预和固定双手抓握结构，证据支持步态受力通过机器人—双手—小车耦合传递。', '',
           '![步态与手力同频](gait_hand_cart_signals.png)', '',
           '## 高层零指令轴并没有自动输出零修正', '',
           '当前残差门控使用整个任务命令向量的模长。只要前后运动命令非零，侧向、转向及上身残差通道也会开启。任务参考在车体航向坐标；HOMIE 接受的是本体坐标命令，所以正常坐标转换也会产生非零本体 vy。下面关闭的是额外残差，保留了这项坐标转换。', '',
           '倒推完整策略在 8–12 s 发给 HOMIE 的平均命令约为 `[-0.460, +0.044, +0.057]`；其中高层侧向／转向残差均值约为 `+0.028 m/s / +0.058 rad/s`。虽然任务 vy/wz 为零，底层实际收到的命令不全为零。', '',
           '| 倒推诊断模式 | 实际 vx 均值 | vy 均值 | wz 均值 | wz 标准差 |',
           '|---|---:|---:|---:|---:|']
    labels={'full':'完整策略','no_yz_residual':'仅关闭高层 y/wz 残差','fixed_upper':'仅固定上身残差','passthrough':'全部高层残差置零'}
    for mode,label in labels.items():
        r=report['reverse_'+mode];v=r['velocity']
        lines.append(f'| {label} | {v["mean"][0]:.4f} | {v["mean"][1]:.4f} | {v["mean"][2]:.4f} | {v["std"][2]:.4f} |')
    lines += ['', '只关闭 y/wz 残差时，倒退速度基本保持，平均偏差和偏航波纹降低。这是这两路残差在当前工况加重偏差的证据，不是训练好的通用替代策略。两路一起关闭不能分别归因给单个通道，其他动作也会随新的闭环状态变化。', '',
              '直行同样出现侧漂降低（vy 均值约 0.098→0.013 m/s），但 wz 标准差约 0.094→0.100 rad/s，未降低。因此硬关这两路不能保证解决全部工况。', '',
              '![通道诊断](zero_axis_action_probe.png)', '',
              '## 为什么训练奖励没有压住这些波动', '',
              '从检查点保存配置和代码确认：主要速度跟踪项使用 tau=0.2 s 的一阶低通，alpha=0.1，而图里画的是原始物理速度；评分形式为 `exp(-error^2/0.1)`。', '',
              '| 完整策略 | 原始 wz 标准差 | 滤波后 wz 标准差 | vy 跟踪归一化得分 | wz 跟踪归一化得分 |',
              '|---|---:|---:|---:|---:|']
    for scenario in ('straight','reverse'):
        f=filters[scenario]
        lines.append(f'| {scenario} | {f["raw_std"][2]:.4f} | {f["filtered_std"][2]:.4f} | {f["tracking_score"][1]:.4f} | {f["tracking_score"][2]:.4f} |')
    lines += ['', '这些分数只是用本轮 MuJoCo 原始轨迹按保存公式离线计算的单项得分，不是训练日志或完整总奖励。它说明较明显的原始快速摆动仍可能获得很高的主要跟踪分数。', '',
              '此外，`track_progress` 只对绝对指令至少 0.1 的轴计分；vy/wz 指令为零时不进入这个加强项。`stand_still` 只在整个三维指令接近零时启用，直行或倒推期间不会单独约束零指令的 y/wz。', '',
              '不能说训练完全没有波纹惩罚：另有 0.5–5 Hz 速度波纹项，保存的 vy／wz 权重分别为 −0.10／−0.15。现有训练结果说明它们与其他奖励的组合尚未把原始摆动压到足够小；本轮没有重新训练来单独证明某个权重的因果作用。', '',
              '## 估计反馈与参考点检查', '',
              '倒推完整策略的真实均值约 vy=0.051、wz=0.047，而预测均值约 vy=0.022、wz=0.013；wz 预测标准差约 0.080，真实约 0.114。策略可见的估计反馈低估了部分偏差和波动。尚未做真值替换干预，因此不把全部控制偏差归因于估计头。', '',
              '同时检查了车体 COM 与 base-link 原点速度：两者在稳态横向平均绝对差约 0.0018 m/s，远小于约 0.05 m/s 的侧漂，不能用参考点选择解释主要问题。', '',
              '此前 Isaac 名义轨迹本身也有摆动：15 kg 倒推稳态 wz 标准差约 0.072–0.078 rad/s，侧向均值约 0.023–0.027 m/s。这不是新场景独有的问题，但当前 MuJoCo 倒推的幅度更大。', '',
              '后续应针对零指令轴的平均偏差、步频波纹和估计反馈分别优化，并保留有效的上身协同。需要验证多载荷、直行／倒推／转弯，避免只在一个方向硬关残差或仅让输出曲线看起来平滑。本轮没有修改正式场景、奖励或模型权重。', '',
              '原始数据：[report.json](report.json)、[滤波分析](filter_analysis.json)、[执行验证](validation.json)及本目录 8 份 NPZ。复现脚本为 `scripts/diagnose_v53_zero_axis_oscillation.py` 和 `scripts/summarize_v53_zero_axis_oscillation.py`。']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(filters,indent=2))


if __name__=='__main__':main()
