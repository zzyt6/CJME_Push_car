"""Summarize pre-reset Isaac traces and render the actual grasp loads."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory', type=Path, default=Path('logs/bench/v53_trained_grasp_20260928'))
a = p.parse_args()
folder = a.directory
modes = ('nominal', 'full', 'sampled')
reports, extras = {}, {}
labels = {'nominal': '名义环境／确定性动作', 'full': 'L13 随机化／确定性动作', 'sampled': 'L13 随机化／采样动作'}
for mode in modes:
    r = json.loads((folder/f'{mode}.json').read_text(encoding='utf8'))
    assert r['parent_verified'] and r['result'] == 'PASS_EXECUTION'
    reports[mode] = r
    d = np.load(folder/f'{mode}.npz')
    valid = d['label_valid']
    primary = d['episode_id'] == 0
    wrist_ids = [i for i,n in enumerate(r['joint_names']) if 'wrist_pitch' in n or 'wrist_yaw' in n]
    ratios = np.abs(d['torques']) / np.array(r['torque_limits'])
    failures = d['terminated']
    extra = dict(failed_environments=int(failures.any(0).sum()),
        completed_first_protocols=int((~failures.any(0)).sum()),
        primary_episode_sample_fraction=float(primary.mean()),
        wrist_any_saturation_fraction=float((ratios[..., wrist_ids] >= .99).any(-1)[valid].mean()),
        events_with_recent_disturbance=0, events_with_recent_stall=0,
        termination_details=[], exemplar_rows=[])
    for event in r['events']:
        t, i = event['step'], event['env']
        start = max(0,t-49)
        same = d['episode_id'][start:t+1,i] == d['episode_id'][t,i]
        force_pulse = (np.abs(d['cart_wrench'][start:t+1,i]).sum(-1) > 0) | (np.abs(d['robot_wrench'][start:t+1,i]).sum(-1) > 0)
        recent = bool((force_pulse & same).any())
        stalled = bool((d['stall'][start:t+1,i] & same).any())
        extra['events_with_recent_disturbance'] += int(recent)
        extra['events_with_recent_stall'] += int(stalled)
        extra['termination_details'].append(dict(time=event['time'], env=i, case=event['case'],
            reason=event['reason'], stage=event['stage'], pulse_previous_1s=recent, stall_previous_1s=stalled,
            internal_axial_abs_mean_previous_04s=float(np.abs(d['internal_axial_force'][max(0,t-19):t+1,i]).mean()),
            over_directions_previous_04s=d['over_directions'][max(0,t-19):t+1,i].mean(0).tolist()))
    # A common preselected case, rather than selecting the most flattering error.
    selected = next(c['env'] for c in r['cases'] if c['payload_kg'] == 15 and c['scenario'] == 'left')
    for seconds in (5., 9., 14., 19., 23.):
        k = round(seconds/r['control_dt'])
        if k >= len(d['time']): continue
        extra['exemplar_rows'].append(dict(time=seconds, env=selected,
            valid=bool(d['est_valid'][k,selected]),
            label=d['est_label'][k,selected].reshape(2,6).tolist(),
            prediction=d['est_prediction'][k,selected].reshape(2,6).tolist()))
    # Use the first actual failure, or the same 15-kg left-turn case if none.
    i = r['events'][0]['env'] if r['events'] else selected
    times = d['time'][:,i]
    fig, ax = plt.subplots(3,2,figsize=(13,10),sharex=True,layout='constrained')
    for idx, axis, unit in [(0,ax[0,0],'vx (m/s)'),(2,ax[0,1],'yaw rate (rad/s)')]:
        axis.plot(times,d['reference'][:,i,idx],label='Filtered reference',color='black',lw=1.3)
        axis.plot(times,d['velocity'][:,i,idx],label='Actual',lw=.9)
        axis.set_ylabel(unit)
    for hand,color in [(0,'tab:blue'),(1,'tab:orange')]:
        ax[1,0].plot(times,d['force'][:,i,hand,1],label=['Left Fy','Right Fy'][hand],color=color,lw=.8)
        ax[1,1].plot(times,d['moment_norm_used'][:,i,hand],label=['Left |M|','Right |M|'][hand],color=color,lw=.8)
    ax[1,0].plot(times,d['thresholds'][:,i,2],'k--',label='Side-force limit',lw=1)
    ax[1,0].plot(times,-d['thresholds'][:,i,2],'k--',lw=1)
    ax[1,1].plot(times,d['thresholds'][:,i,3],'k--',label='Moment limit',lw=1)
    ax[1,0].set_ylabel('Hand lateral force (N)')
    ax[1,1].set_ylabel('Palm moment magnitude (N m)')
    ax[2,0].plot(times,d['internal_axial_force'][:,i],label='Axial internal force',color='tab:purple',lw=.8)
    ax[2,0].set_ylabel('Equal/opposite internal force (N)')
    for hand,color in [('left','tab:blue'),('right','tab:orange')]:
        ids=[j for j in wrist_ids if r['joint_names'][j].startswith(hand)]
        ax[2,1].plot(times,ratios[:,i,ids].max(-1),label=hand+' wrist pitch/yaw',color=color,lw=.8)
    ax[2,1].axhline(1.,color='black',ls='--')
    ax[2,1].set_ylabel('Drive effort / joint limit')
    for axis in ax.flat:
        for t in times[failures[:,i]]: axis.axvline(t,color='red',ls=':',lw=1)
        for t in (2,6,12,16,21): axis.axvline(t,color='gray',alpha=.3,lw=.6)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8,loc='upper right')
    ax[2,0].set_xlabel('Simulation time (s)');ax[2,1].set_xlabel('Simulation time (s)')
    fig.suptitle(f'{mode}: model_15000, env {i}, payload {r["cases"][i]["payload_kg"]:g} kg, '
                 f'{r["cases"][i]["scenario"]}; red lines = termination/reset')
    fig.savefig(folder/f'{mode}_grasp_trace.png',dpi=150)
    plt.close(fig)
    extra['trace_env'] = i
    with (folder/f'{mode}_trace.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['time_s','episode','ref_vx','actual_vx','ref_wz','actual_wz',
                    'L_fx_N','L_fy_N','L_fz_N','R_fx_N','R_fy_N','R_fz_N',
                    'L_mx_Nm','L_my_Nm','L_mz_Nm','R_mx_Nm','R_my_Nm','R_mz_Nm',
                    'pull_limit_N','up_limit_N','side_limit_N','moment_limit_Nm','internal_axial_N','terminated'])
        for k in range(len(times)):
            w.writerow([times[k],d['episode_id'][k,i],d['reference'][k,i,0],d['velocity'][k,i,0],
                        d['reference'][k,i,2],d['velocity'][k,i,2],*d['force'][k,i].flatten(),
                        *d['moment'][k,i].flatten(),*d['thresholds'][k,i],d['internal_axial_force'][k,i],failures[k,i]])
    extras[mode]=extra

assert len({r['checkpoint_sha256'] for r in reports.values()}) == 1
(folder/'analysis.json').write_text(json.dumps(extras,indent=2,ensure_ascii=False),encoding='utf8')
lines=['# 已训练 v5.3 策略：Isaac 实际抓握负荷诊断','',
       '固定 `v53_final_20k/model_15000.pt`，沿用保存的网络、归一化器、显式 PD 和 HOMIE `model_60000.pt`。本次没有训练或修改过载阈值。','',
       '每组 24 个环境 × 1200 个控制步 × 20 ms = 576 环境秒；载荷 5/15/25 kg（不含空车），每个载荷含直行、左转、右转、倒推 4 类，每类 2 个环境。名义空车 14.4 kg，因此三组名义总重为 19.4/29.4/39.4 kg。名义环境重复并非独立随机试验。','',
       '程序：0–2 s 静止；2–6 s 起步；6–12 s 主运动；12–16 s 制动；16–21 s 再起步/反向转弯；21–24 s 停车。直行主指令 0.6 m/s，转弯 0.35 m/s、±0.25 rad/s，倒推 −0.35 m/s。参考保留原有 0.4 s 滤波。失败后仍自动重置并接入当时的全局程序；最长回合设为 30 s，避免 24 s 程序中出现纯时间终止。','',
       '随机化组固定载荷工况，其他 L13 参数、噪声、扰动和终止逻辑保留；命令程序与训练随机命令不同。采样组使用检查点每维高斯标准差，不额外捏造统一噪声。不同闭环轨迹会改变重置与随机数使用顺序，因此三组差值不是严格的单因素因果估计。','',
       '| 设置 | 终止事件 | 发生失败的环境 / 24 | 侧向超限时刻比例 | 力矩超限时刻比例 | 轴向内力绝对值 P95 / N |','|---|---:|---:|---:|---:|---:|']
for mode,r in reports.items():
    s=r['overall'];e=extras[mode]
    lines.append(f'| {labels[mode]} | {s["terminations"]} | {e["failed_environments"]} | {s["over_fraction"][2]*100:.2f}% | {s["over_fraction"][3]*100:.2f}% | {s["internal_axial_abs_p95"]:.2f} |')
lines += ['', '“超限时刻”不同于“终止”：任一检查项超限连续维持 20 个控制步才触发过载终止。发生失败的环境比例也不能当作 20 s 随机训练回合失败率。', '',
          '## 各工况实际表现','', '| 设置 | 载荷 kg | 工况 | 终止事件 | 主运动平均 vx / 指令参考 | 主运动平均 wz / 指令参考 | 制动阶段平均绝对 vx 误差 |',
          '|---|---:|---|---:|---:|---:|---:|']
for mode,r in reports.items():
    for c in r['case_statistics']:
        s=c['main'];b=c['brake']
        lines.append(f'| {labels[mode]} | {c["payload"]:g} | {c["scenario"]} | {c["terminations"]} | '
            f'{s["mean_velocity"][0]:.3f} / {s["mean_reference"][0]:.3f} | '
            f'{s["mean_velocity"][2]:.3f} / {s["mean_reference"][2]:.3f} | {b["tracking_abs_mean"][0]:.3f} |')
lines += ['', '## 估力误差：同一时刻输入与标签配对','',
          '单位为 N；列为各轴平均绝对偏差，非 RMSE。这是固定权重在新采集轨迹上的结果。有效重置前观察不包含已被自动重置替代的终止观察；物理过载现场通过单独的重置前回调完整保留。', '',
          '| 设置 | L Fx | L Fy | L Fz | R Fx | R Fy | R Fz |','|---|---:|---:|---:|---:|---:|---:|']
for mode,r in reports.items():
    errors=r['estimator']['mae']
    lines.append('| '+labels[mode]+' | '+' | '.join(f'{errors[i]:.3f}' for i in (0,1,2,6,7,8))+' |')
lines += ['', '## 真实记录的轨迹与限制','',
          '图中优先展示该组第一次失败的环境；没有失败时展示预先指定的 15 kg 载荷左转环境。红竖线是终止/重置，灰竖线是程序阶段切换。', '',
          '内力只取沿两掌连线的一对等大反向力：该分量合力和合力矩均为零。不能把任意左右手力差都当成无效内力，差动前后推力可以产生有用的转向力矩。', '',
          '手部负荷是子树动力学代理量，尚非独立标定接触传感器。过载使用最新物理子步负荷，估计器标签使用同一控制周期的四子步平均。关节力矩是控制时刻记录的最后子步驱动值；没有宣称覆盖全部 200 Hz 极值。本测试没有检验真实手指滑动或硬件承载能力。','']
for mode in modes:
    lines += [f'### {labels[mode]}','',f'![{mode}]({mode}_grasp_trace.png)','',
              f'逐时刻数据：[{mode}_trace.csv]({mode}_trace.csv)。完整 24 环境数组：[{mode}.npz]({mode}.npz)。', '']
lines += ['所有 `.json` 均包含本次唯一 run_id、检查点哈希、源码哈希和父进程完成验证；`PASS_EXECUTION` 只表示测评完整执行。结论需结合负荷、跟踪和终止结果判断。','']
(folder/'comparison.md').write_text('\n'.join(lines),encoding='utf8')
for mode,r in reports.items():
    print(mode,json.dumps(dict(terminations=r['overall']['termination_reasons'], extra=extras[mode],
                              force_mae=r['estimator']['mae']),ensure_ascii=False))
print(folder/'comparison.md')
