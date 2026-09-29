"""Compare raw cart velocity in the matched 24-second nominal protocol."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/bench/v52_v53_nominal_tracking_20260928'
KEYS = ('v52_full_aligned', 'v53_aligned', 'v52_original_native')
REPORTS = {k: json.loads((OUT / (k+'.json')).read_text(encoding='utf8')) for k in KEYS}
DATA = {k: np.load(OUT / (k+'.npz')) for k in KEYS}
DT = REPORTS[KEYS[0]]['control_dt']
T = np.arange(REPORTS[KEYS[0]]['steps'])*DT
CASES = REPORTS[KEYS[0]]['cases']
OLD, NEW = KEYS[:2]
LABELS = {OLD: 'v5.2 全输入估力头＋显式 PD（车体对齐）', NEW: 'v5.3 当前策略',
          KEYS[2]: 'v5.2 原训练版（原驱动、原车体惯量）'}

for k, r in REPORTS.items():
    d = DATA[k]
    assert r['parent_verified'] and r['result'] == 'PASS_EXECUTION'
    assert r['cases'] == CASES and r['steps'] == len(T) and r['control_dt'] == DT
    assert not r['noise_effective'] and not r['randomization_effective'] and not r['action_exploration']
    assert not d['cart_wrench'].any() and not d['robot_wrench'].any() and not d['stall'].any()
    np.testing.assert_allclose(d['command'], DATA[OLD]['command'], atol=0, rtol=0)
    # Refuse a survivor-only comparison or comparisons whose reset reference differs.
    assert not d['terminated'].any() and not d['timeout'].any(), 'Report failures explicitly before computing matched metrics'
    np.testing.assert_allclose(d['reference'], DATA[OLD]['reference'], atol=1e-6, rtol=0)

checks = {}
for field in ('mass_kg', 'wheel_friction', 'wheel_damping', 'swivel_damping',
              'kp_factors', 'kd_factors', 'p_gains', 'd_gains', 'thresholds',
              'torque_delay_steps', 'pose_delay_steps', 'cart_body_mass',
              'cart_inertia_link_frame', 'cart_com_pose', 'robot_body_mass',
              'robot_inertia', 'cart_materials', 'robot_materials'):
    a, b = [np.array(REPORTS[k]['initial_parameters'][field]) for k in (OLD, NEW)]
    np.testing.assert_allclose(a, b, atol=5e-6, rtol=1e-6, err_msg=field)
    checks[field] = float(np.max(np.abs(a-b)))
assert REPORTS[OLD]['control_contract'] == REPORTS[NEW]['control_contract']
assert REPORTS[OLD]['termination_config'] == REPORTS[NEW]['termination_config']
assert REPORTS[OLD]['reference_tau_s'] == REPORTS[NEW]['reference_tau_s'] == .4
assert REPORTS[OLD]['lowlevel_checkpoint'] == REPORTS[NEW]['lowlevel_checkpoint']

previous = np.load(ROOT/'logs/bench/v53_trained_grasp_20260928/nominal.npz')
repeat_difference = float(abs(previous['velocity']-DATA[NEW]['velocity']).max())
np.testing.assert_allclose(previous['command'], DATA[NEW]['command'], atol=0, rtol=0)

def window(a, b):
    return (T >= a) & (T < b)

def metrics(k):
    d = DATA[k]
    v, ref = d['velocity'].astype(float), d['reference'].astype(float)
    e = v-ref
    steady = ((8.,12.), (18.,21.))
    selected = window(8,12) | window(18,21)
    moving = np.isin(d['stages'], ['start','main','restart_turn_reverse'])
    stop = window(14,16) | window(23,24)
    # Demean each environment and each constant-command plateau separately.
    # Do not count a commanded velocity transition or a steady bias as ripple.
    std = sum((b-a)*v[window(a,b)].std(axis=0).mean(axis=0) for a,b in steady)/7
    spread = sum((b-a)*(np.quantile(v[window(a,b)],.95,axis=0)-np.quantile(v[window(a,b)],.05,axis=0)).mean(0)
                 for a,b in steady)/7
    case_rows = []
    for payload in (5.,15.,25.):
        for scenario in ('straight','left','right','reverse'):
            ids = [c['env'] for c in CASES if c['payload_kg'] == payload and c['scenario'] == scenario]
            row = dict(payload_kg=payload, scenario=scenario,
                overall_mae=abs(e[:,ids]).mean((0,1)).tolist(),
                main_steady_mean=v[window(8,12)][:,ids].mean((0,1)).tolist(),
                main_steady_reference=ref[window(8,12)][:,ids].mean((0,1)).tolist(),
                main_steady_std=v[window(8,12)][:,ids].std(0).mean(0).tolist(),
                restart_steady_mean=v[window(18,21)][:,ids].mean((0,1)).tolist())
            case_rows.append(row)
    return dict(overall_mae=abs(e).mean((0,1)).tolist(),
        moving_mae=abs(e[moving]).mean((0,1)).tolist(),
        steady_mae=abs(e[selected]).mean((0,1)).tolist(),
        steady_wave_std=std.tolist(), steady_wave_p95_p05=spread.tolist(),
        stopped_abs_velocity=abs(v[stop]).mean((0,1)).tolist(),
        within_tolerance=(abs(e)<=np.array([.05,.05,.05])).mean((0,1)).tolist(),
        cases=case_rows, terminations=int(d['terminated'].sum()))

STATS = {k: metrics(k) for k in KEYS}
increase = (np.array(STATS[NEW]['overall_mae'])/STATS[OLD]['overall_mae']-1)*100

# Plot the same left-turn environment as the user's figure, plus straight motion.
fig, axes = plt.subplots(2,2,figsize=(14,7),sharex=True,layout='constrained')
colors = {OLD:'#2878b5', NEW:'#e67924'}
labels = {OLD:'v5.2 full head, explicit PD, aligned cart', NEW:'v5.3 model_15000'}
for col, (env_id, scenario) in enumerate(((5,'Left turn, then right turn'),(4,'Straight / stop / restart'))):
    for row, (axis,name,unit) in enumerate(((0,'vx','m/s'),(2,'wz','rad/s'))):
        ax = axes[row,col]
        for k in (OLD,NEW):
            ax.plot(T+DT, DATA[k]['velocity'][:,env_id,axis],lw=.85,alpha=.88,color=colors[k],label=labels[k])
        ax.plot(T+DT,DATA[NEW]['reference'][:,env_id,axis],color='black',ls='--',lw=1.25,label='Filtered reference (0.4 s)')
        for t in (2,6,12,16,21): ax.axvline(t,color='gray',alpha=.3,lw=.6)
        for a,b in ((8,12),(18,21)): ax.axvspan(a,b,color='#9dccab',alpha=.10)
        ax.set_ylabel(f'{name} ({unit})');ax.grid(alpha=.18)
        if row == 0: ax.set_title(f'{scenario} | payload 15 kg | env {env_id}')
        else: ax.set_xlabel('Simulation time (s)')
        ax.set_xlim(0,24)
handles,names=axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,loc='outside upper center',ncol=3,fontsize=9)
fig.savefig(OUT/'velocity_overlay.png',dpi=170)
plt.close(fig)

with (OUT/'velocity_overlay.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['time_s','env','payload_kg','scenario','reference_vx','reference_vy','reference_wz',
                     *[f'{k}_{axis}' for k in KEYS for axis in ('vx','vy','wz')]])
    for i in (4,5):
        for step,t in enumerate(T+DT):
            writer.writerow([t,i,15,CASES[i]['scenario'],*DATA[NEW]['reference'][step,i],
                             *[x for k in KEYS for x in DATA[k]['velocity'][step,i]]])

summary = dict(axes=['vx_m_s','vy_m_s','wz_rad_s'], stats=STATS,
    comparison='v52_full_aligned versus v53_aligned; original_native is historical context',
    parameter_readback_max_differences=checks,
    v53_previous_trace_max_difference=repeat_difference,
    overall_mae_increase_percent=increase.tolist(),
    steady_windows_s=[[8,12],[18,21]], stopped_windows_s=[[14,16],[23,24]],
    description='Mean absolute tracking deviation, no RMSE. Ripple is window/environment-demeaned velocity standard deviation.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')

lines=['# v5.3 与 v5.2：正常环境前进速度、转向角速度实测对比','',
       '**当前 v5.3 在这套正常环境中没有超过此前 v5.2 全输入估力头＋显式 PD 方案。** '
       f'全程前进速度平均绝对偏差增加 {increase[0]:.1f}%，转向角速度偏差增加 {increase[2]:.1f}%；'
       '稳态波动接近，停车残余运动和侧向漂移更明显。三个方案均无终止，不能用“能完成”代替“跟得更准”。','',
       '## 如何保证这次比较有意义','',
       '- 新版使用 `v53_final_20k/model_15000.pt`，与之前图中的检查点相同；旧版 actor 使用 `v52_hand_estimator_20k/model_19999.pt`。HOMIE 均为 `model_60000.pt`，确定性推理，无训练或优化器更新。',
       '- 主要旧版基线使用此前的 `model_explicit_pd_full_22/estimator.pt` 全输入估力头，腰臂显式 PD。只替换估力头不代表旧 actor 重新做过 PPO 训练。',
       '- 正常平地；随机化、观测噪声、传感器/动作随机延迟、外力脉冲、轮子卡滞与动作探索全部关闭。地面摩擦 1.0，轮子材料摩擦 0.8，轮轴/转向阻尼 0.005/0.05。',
       '- 固定载荷 5/15/25 kg，空车 14.4 kg；每种载荷分别直行、左转、右转、倒推，共 12 工况，每工况 2 个环境；两个名义重复不视为独立随机样本。',
       '- 每个环境 24 s，控制周期 20 ms、物理周期 5 ms，冻结课程 L13 和相同过载阈值。0–2 s 静止，2–6 s 起步，6–12 s 主运动，12–16 s 制动，16–21 s 再起步/转弯反向，21–24 s 停车。',
       '- 主运动直行指令 0.6 m/s、倒推 −0.35 m/s；转向工况 0.35 m/s、±0.25 rad/s。两版保留相同 0.4 s 指令参考滤波，指标使用真实物理车速，不使用奖励中的额外车速滤波。',
       '- **车体惯量也做了对齐：**v5.2 原重置先设置惯量再恢复 COM 旋转，会改变实际惯量。主要基线仅在此次测试实例中改用 v5.3 的设置顺序；从 PhysX 回读确认两版车体质量、COM、惯量、材料，以及机器人质量/惯量、PD 增益、延迟、阈值匹配。训练源代码和配置未修改。',
       '- 另跑 v5.2 原训练版作为历史参考，保留其混合驱动和旧惯量；该行不能当作物理条件完全一致的网络消融。','',
       '## 全程跟踪结果','',
       '以下每个时刻都计入，包括起步、转向、制动、停车，没有只筛选表现好的时间段。单位分别为 m/s、rad/s。','',
       '| 方案 | vx 平均绝对偏差 | wz 平均绝对偏差 | 非期望侧向速度平均绝对值 | 失败终止 |',
       '|---|---:|---:|---:|---:|']
for k in KEYS:
    m=STATS[k]['overall_mae']
    lines.append(f'| {LABELS[k]} | {m[0]:.4f} | {m[2]:.4f} | {m[1]:.4f} | {STATS[k]["terminations"]} |')
lines += ['', '这里统计的是平均绝对偏差，而非 RMSE。正常环境中整体跟踪尚未改善，尤其侧向速度指令为 0，却存在更明显的漂移。','',
          '## 对照图里的具体数值','',
          '下表取 15 kg 载荷，平均同工况的两个环境；用 8–12 s 稳定运动段说明持续偏差，'
          '不把 6 s 指令切换的必需过渡过程算成稳态问题。参考此时约等于指令。','',
          '| 工况 | 轴与参考 | v5.2 对齐基线实际均值 | v5.3 实际均值 |','|---|---|---:|---:|']
for scenario,cn in [('straight','直行'),('left','左转'),('right','右转'),('reverse','倒推')]:
    rows=[next(c for c in STATS[k]['cases'] if c['payload_kg']==15 and c['scenario']==scenario) for k in (OLD,NEW)]
    for axis,name,unit in [(0,'vx','m/s'),(2,'wz','rad/s')]:
        lines.append(f'| {cn} | {name} = {rows[1]["main_steady_reference"][axis]:.3f} {unit} | '
                     f'{rows[0]["main_steady_mean"][axis]:.3f} | {rows[1]["main_steady_mean"][axis]:.3f} |')
lines += ['', '![相同正常工况下的前进、转向叠图](velocity_overlay.png)','',
          '左列是之前图中的 15 kg 左转/反向转弯环境 5；右列是 15 kg 直行环境 4。'
          '每条曲线为该环境的原始物理车速，没有为绘图再加平滑。绿色浅区为稳态统计区间。','',
          '## 波动与停车表现','',
          '波动在 8–12 s、18–21 s 两段内，先按各环境、各段去掉均值，计算车速标准差，再按段长平均。'
          '因此持续欠速不会被误算成波动，起步/转向指令跳变也不直接作为波动惩罚。它仍包含步态振动和段内缓慢变化，不是特定频带的波纹能量。','',
          '| 方案 | 稳态 vx 波动标准差 m/s | 稳态 wz 波动标准差 rad/s | 停车后绝对 vx m/s | 停车后绝对 wz rad/s |',
          '|---|---:|---:|---:|---:|']
for k in (OLD,NEW):
    s=STATS[k];w=s['steady_wave_std'];stop=s['stopped_abs_velocity']
    lines.append(f'| {LABELS[k]} | {w[0]:.4f} | {w[2]:.4f} | {stop[0]:.4f} | {stop[2]:.4f} |')
lines += ['', '停车列仅取制动指令发出至少 2 s 后的 14–16 s 和 23–24 s，统计真实速度绝对值。'
          '两版稳态波动非常接近，当前证据不支持“加了波纹奖励就已经更平滑”；新版停车后的残余运动反而更大。','',
          '## 各载荷、各工况全程偏差','',
          '| 载荷 kg | 工况 | vx 偏差 v5.2 / v5.3 m/s | wz 偏差 v5.2 / v5.3 rad/s |','|---|---|---:|---:|']
for a,b in zip(STATS[OLD]['cases'],STATS[NEW]['cases']):
    x,y=a['overall_mae'],b['overall_mae']
    lines.append(f'| {a["payload_kg"]:g} | {a["scenario"]} | {x[0]:.4f} / {y[0]:.4f} | {x[2]:.4f} / {y[2]:.4f} |')
lines += ['', '## 能得出的结论和下一步','',
          '这是当前整套闭环方案的比较。v5.3 同时改变了输入、车速估计方式、奖励和训练分布，检查点也分别是 15000 与 19999 次更新；'
          '不能单凭这次结果认定某一个网络或某一个奖励项导致退步，也不能把两个训练阶段当作相等训练预算。','',
          '下一步应在相同正常环境中拆分车速估计偏差与控制策略响应，重点看直行欠速、左右转向偏差和停车残余运动。'
          '如果做真值车速替换，应明确标为仿真诊断消融，保持其他输入和控制不变；随后再决定调整奖励或随机化。'
          '本次没有继续加大力变化惩罚、放宽抓握阈值或修改训练。','',
          f'新评测与之前 `nominal.npz` 相同 v5.3 检查点的速度逐帧最大差为 {repeat_difference:.6g}；'
          '数值见 [summary.json](summary.json)，叠图逐帧数据见 [velocity_overlay.csv](velocity_overlay.csv)。','',
          '原始数据：[v5.2 对齐基线](v52_full_aligned.json) / [数组](v52_full_aligned.npz)，'
          '[v5.3](v53_aligned.json) / [数组](v53_aligned.npz)，'
          '[v5.2 历史参考](v52_original_native.json) / [数组](v52_original_native.npz)。'
          '各报告保存运行 ID、父进程完成校验、物理参数回读以及策略/源文件 SHA256。','']
(OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(overall_mae={k:STATS[k]['overall_mae'] for k in KEYS},
    steady_wave_std={k:STATS[k]['steady_wave_std'] for k in KEYS},
    stopped_abs_velocity={k:STATS[k]['stopped_abs_velocity'] for k in KEYS},
    mae_increase_percent=increase.tolist(), repeat_difference=repeat_difference),indent=2))
print(OUT/'comparison.md')
