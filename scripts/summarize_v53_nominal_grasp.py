"""Focused nominal-environment assessment; no randomized traces enter this report."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'logs/bench/v53_trained_grasp_20260928'
r = json.loads((p/'nominal.json').read_text(encoding='utf8'))
d = np.load(p/'nominal.npz')
assert r['parent_verified'] and r['mode'] == 'nominal'
assert not r['randomization_effective'] and not r['noise_effective']
assert not d['terminated'].any() and not d['timeout'].any()
assert not np.any(d['cart_wrench']) and not np.any(d['robot_wrench']) and not np.any(d['stall'])
for name in ('heading_delay_steps', 'torque_delay_steps'):
    assert not np.any(r['initial_parameters'][name])
for name in ('kp_factors', 'kd_factors'):
    assert np.allclose(r['initial_parameters'][name], 1.)
assert np.allclose(r['initial_parameters']['wheel_friction'], .8)
assert np.allclose(r['initial_parameters']['wheel_damping'], .005)
assert np.allclose(r['initial_parameters']['swivel_damping'], .05)

# Physical trace k ends at (k+1)dt; estimator pair k+1 is at that exact time.
physical = np.concatenate((d['force_mean'], d['moment_mean']), axis=-1).reshape(len(d['time']), -1, 12)
mask = d['est_valid'][1:]
alignment_error = float(np.abs(physical[:-1]-d['est_label'][1:])[mask].max())
assert alignment_error < 3e-5, alignment_error
i = next(c['env'] for c in r['cases'] if c['payload_kg'] == 15 and c['scenario'] == 'straight')
dt = r['control_dt']
times = np.arange(len(d['time']))*dt
truth = d['est_label'][:, i].reshape(-1,2,6)
prediction = d['est_prediction'][:,i].reshape(-1,2,6)
valid = d['est_valid'][:,i]
fig, axes = plt.subplots(3,2,figsize=(13,9),sharex=True,layout='constrained')
for axis_index, name in enumerate(('Fx','Fy','Fz')):
    for hand in (0,1):
        ax=axes[axis_index,hand]
        ax.plot(times[valid],truth[valid,hand,axis_index],lw=.8,label='Simulation label (20 ms mean)')
        ax.plot(times[valid],prediction[valid,hand,axis_index],lw=.8,alpha=.8,label='Force estimator')
        for t in (2,6,12,16,21): ax.axvline(t,color='gray',alpha=.35,lw=.6)
        ax.set_ylabel(f'{name} (N)');ax.grid(alpha=.2)
        if axis_index == 0:
            ax.set_title(('Left','Right')[hand]);ax.legend(fontsize=8)
        if axis_index == 2:ax.set_xlabel('Simulation time (s)')
fig.suptitle('Nominal environment, deterministic model_15000; straight/stop/restart; payload 15 kg (total 29.4 kg)')
fig.savefig(p/'nominal_straight_axes.png',dpi=150)
plt.close(fig)
with (p/'nominal_straight_axes.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f)
    names=[f'{h}_{v}_{axis}_N' for h in ('L','R') for v in ('label','prediction') for axis in ('fx','fy','fz')]
    w.writerow(['time_s','valid',*names])
    for k in range(len(times)):
        w.writerow([times[k],valid[k],*truth[k,0,:3],*prediction[k,0,:3],*truth[k,1,:3],*prediction[k,1,:3]])

longest=int(d['overload_counter'].max())
valid_physical=d['label_valid']
moment_peak=float(d['moment_norm_used'][valid_physical].max())
phases={'settle':'初始静止','start':'起步','main':'主运动','brake':'制动',
        'restart_turn_reverse':'再起步/反向转弯','final_stop':'最终停车'}
fmt=lambda a:'['+', '.join(f'{v:.2f}' for v in a)+']'
summary={'mode':'nominal_only','checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],
         'num_envs':r['num_envs'],'seconds_each':len(times)*dt,'terminations':int(d['terminated'].sum()),
         'longest_any_overload_s':longest*dt,'hand_moment_sample_peak_Nm':moment_peak,
         'paired_label_alignment_max_error':alignment_error,'selected_env':i,
         'source_report':'nominal.json','source_arrays':'nominal.npz'}
(p/'nominal_assessment.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
lines=['# 正常环境推车：策略、抓握负荷和估力实测','',
       '**本报告只使用名义环境、确定性策略的数据，不使用随机化组解释正常推车。**','',
       '本次 24 个环境均完成 24 s 程序，无过载终止、跌倒、车轮离地终止或数值保护终止。当前阈值没有阻止这些工况完成动作。但存在欠速、侧向漂移及明显的双手内部作用力，不能把“无终止”写成“已达到高精度推车”。','',
       '## 实际运行条件','',
       '- 策略：`v53_final_20k/model_15000.pt`，网络和归一化器固定；HOMIE 底层仍为 `model_60000.pt`。',
       '- 正常环境：机器人和车体参数随机化关闭，观测噪声关闭，随机延迟关闭，外力脉冲和轮子卡滞关闭；高层使用动作均值。',
       '- 地面摩擦 1.0，车轮材料摩擦 0.8，轮轴阻尼 0.005、脚轮转向阻尼 0.05；Kp/Kd 为名义值。',
       '- 固定载荷 5/15/25 kg，空车 14.4 kg，总重分别 19.4/29.4/39.4 kg。每种载荷覆盖直行、左转、右转、倒推，以及制动和再起步；每类 2 个环境，重复环境不作为独立随机样本。',
       '- 0–2 s 静止，2–6 s 起步，6–12 s 主运动，12–16 s 制动，16–21 s 再起步/转向反转，21–24 s 停车。',
       '- 保留原有固定抓握、显式 PD、参考滤波和过载持续时间；课程冻结 L13，仅决定固定阈值等调度参数，并没有打开随机化。','',
       '## 阈值是否妨碍正常推车','',
       f'当前各手接触力矩阈值为 12 N·m；所有工况中控制时刻的力矩最大值为 {moment_peak:.2f} N·m。确实存在瞬时越界，但任一项连续超限最长只有 {longest} 步（{longest*dt:.2f} s），低于 20 步（0.4 s）的终止条件。因此本次零过载终止并不意味着没有任何瞬时峰值。','',
       '原阈值在本次载荷和速度范围内可以保留作为基线；数据不支持仅因训练课程回退就先放宽正常环境的阈值。瞬态负荷要与持续过载区分，也不能据此对所有力变化直接加重惩罚。','',
       '## 实际速度跟踪','',
       '| 载荷 kg | 工况 | 主运动 vx 实际 / 参考，m/s | 主运动 wz 实际 / 参考，rad/s | 主运动平均 vy，m/s（指令为 0） |',
       '|---|---|---:|---:|---:|']
for c in r['case_statistics']:
    v,ref=c['main']['mean_velocity'],c['main']['mean_reference']
    lines.append(f'| {c["payload"]:g} | {c["scenario"]} | {v[0]:.3f} / {ref[0]:.3f} | {v[2]:.3f} / {ref[2]:.3f} | {v[1]:.3f} |')
lines += ['', '上表参考是主运动阶段的滤波参考均值。直行指令设为 0.6 m/s，该阶段参考均值约 0.581 m/s，实际约 0.493–0.497 m/s；并有约 0.076–0.085 m/s 的平均侧向速度。正常环境中也需要改善跟踪，不应只盯着过载终止率。','',
          '## 受力在哪些阶段增加','',
          '下表为各轴力绝对值的 95% 分位数，顺序 [Fx,Fy,Fz]，单位 N；使用重置前最新物理子步负荷。与估计器监督的 20 ms 平均标签分开统计。','',
          '| 阶段 | 左手 | 右手 | 沿两掌连线的内部力绝对值均值 / N |','|---|---|---|---:|']
for name,s in r['stages'].items():
    lines.append(f'| {phases[name]} | {fmt(s["force_abs_p95"][0])} | {fmt(s["force_abs_p95"][1])} | {s["internal_axial_abs_mean"]:.2f} |')
lines += ['', '运动阶段的内部力明显高于静止阶段。这里的“内部力”严格取沿两掌连线的一对等大反向分量，其合力与合力矩均为零；不能把左右手所有差动力都视为无用，因为差动前后推力可以帮助转向。当前数据说明存在可优化的内部作用力，但尚未证明它全部可消除或全部由高层策略造成；固定双手约束和步态也会影响。','',
          '腕 pitch/yaw 四个关节在记录的控制时刻没有达到 99% 力矩上限；左右肩 pitch 到达该比例的时刻分别约 0.53% / 1.64%。这是最后物理子步的驱动力矩统计，不能冒充全部 200 Hz 峰值或真实硬件承载验证。','',
          '## 正常环境中的估力准确度','',
          '| 手 | Fx 平均绝对偏差 / N | Fy / N | Fz / N |','|---|---:|---:|---:|']
for hand,name in [(0,'左'),(1,'右')]:
    e=r['estimator']['mae'][hand*6:hand*6+3]
    lines.append(f'| {name} | {e[0]:.3f} | {e[1]:.3f} | {e[2]:.3f} |')
lines += ['', '固定示例：15 kg 载荷、直行环境（总重 29.4 kg），下表展示 4/9/14/19 s；向量均为 [Fx,Fy,Fz] N。标签为同一 20 ms 周期的四个物理子步平均，手对车方向、车体航向坐标系。','',
          '| 时刻 s | 手 | 仿真标签 | 网络预测 | 预测−标签 |','|---|---|---|---|---|']
for seconds in (4.,9.,14.,19.):
    k=round(seconds/dt)
    assert valid[k]
    for h,name in [(0,'左'),(1,'右')]:
        t,pr=truth[k,h,:3],prediction[k,h,:3]
        lines.append(f'| {seconds:g} | {name} | {fmt(t)} | {fmt(pr)} | {fmt(pr-t)} |')
lines += ['', '![正常环境逐手逐轴预测](nominal_straight_axes.png)','',
          '[完整逐帧数据](nominal_straight_axes.csv)。原始数组分别在步进前和步进后记录；按实际仿真时间对应后，估计器使用的标签与物理周期均值最大数值差为 7.63e−6，未改动预测或标签。','',
          '![正常环境转弯受力与阈值](nominal_grasp_trace.png)','',
          '这张转弯图使用 15 kg 载荷左转/反向转弯环境；它与上面的直行估力图是两个明确区分的工况。','',
          '## 本次判断与下一步','',
          '在本次正常环境和动作范围内，原有阈值允许策略完成起步、直行、转向、制动及倒推，可以保留为基线。优先改善正常环境中的欠速、侧向漂移和运动阶段内部作用力。若后续改奖励，应区分有用的加减速/转向负荷与内部作用力，再验证跟踪是否改善；本次没有修改奖励、阈值或随机化配置。','',
          '范围：单一检查点、平地、24 s 程序；固定连接没有检验真实手指滑移，手部负荷仍为子树动力学代理量。没有把本次结果提升为真机可用结论。原始报告 [nominal.json](nominal.json)，完整数组 [nominal.npz](nominal.npz)，校验 [nominal_assessment.json](nominal_assessment.json)。','']
(p/'nominal_comparison.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False))
print(p/'nominal_comparison.md')
