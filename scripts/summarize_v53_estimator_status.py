"""Summarize existing Isaac evaluations and the fixed V5.3 MuJoCo evaluation."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_v53_estimator_sim2sim_status import metrics, protocol

FOLDER = ROOT / 'logs/bench/v53_estimator_sim2sim_20260928'
ISAAC = ROOT / 'logs/bench/v53_trained_grasp_20260928'


def vector(x, digits=3):
    return ' / '.join(f'{v:.{digits}f}' for v in x)


def main():
    mj = json.loads((FOLDER / 'report.json').read_text())
    summary = {'checkpoint': mj['checkpoint'], 'checkpoint_sha256': mj['checkpoint_sha256'], 'conditions': {}}
    for mode in ('nominal', 'full', 'sampled'):
        report = json.loads((ISAAC / f'{mode}.json').read_text())
        assert report['checkpoint_sha256'] == mj['checkpoint_sha256']
        data = np.load(ISAAC / f'{mode}.npz')
        valid = data['est_valid'].astype(bool)
        sv = data['est_speed_valid'].astype(bool)
        stat = {'force': metrics(data['est_prediction'][valid], data['est_label'][valid]),
                'speed': metrics(data['est_speed_prediction'][sv], data['est_speed_label'][sv]),
                'tracking': metrics(data['velocity'].reshape(-1, 3), data['reference'].reshape(-1, 3)),
                'num_envs': report['num_envs'], 'steps_per_env': report['steps'],
                'terminations': report['overall']['terminations'], 'noise': report['noise_effective'],
                'physical_randomization': report['randomization_effective']}
        np.testing.assert_allclose(stat['force']['mae'], report['estimator']['mae'], atol=3e-5)
        np.testing.assert_allclose(stat['speed']['mae'], report['estimator']['speed_mae'], atol=2e-6)
        summary['conditions']['isaac_' + mode] = stat
    summary['conditions']['mujoco_nominal'] = {k: mj[k] for k in ('force', 'speed', 'tracking', 'complete_cases')}
    isa = np.load(ISAAC / 'nominal.npz')
    case = next(c for c in mj['cases'] if c['payload_kg'] == 15. and c['scenario'] == 'straight')
    md = np.load(FOLDER / (case['name'] + '.npz'))
    case_id = 4
    # Commands and filtered references agree; physical trajectories need not.
    np.testing.assert_allclose(md['command'], isa['command'][:, case_id], atol=1e-7)
    np.testing.assert_allclose(md['tracking_reference'], isa['reference'][:, case_id], atol=1e-6)
    for c in mj['cases']:
        d = np.load(FOLDER / (c['name'] + '.npz'))
        ci = (5., 15., 25.).index(c['payload_kg']) * 4 + ('straight', 'left', 'right', 'reverse').index(c['scenario'])
        np.testing.assert_allclose(d['command'], isa['command'][:, ci], atol=1e-7)
        np.testing.assert_allclose(d['tracking_reference'], isa['reference'][:, ci], atol=1e-6)
        assert np.all(d['substep_count'] == 4)
    summary['checks'] = {'same_checkpoint': True, 'all_12_commands_and_references_match': True,
                         'mujoco_all_labels_have_four_physics_samples': True,
                         'mujoco_prediction_repeat_max_abs': max(c['prediction_repeat_max_abs'] for c in mj['cases'])}
    (FOLDER / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .22})
    valid = isa['est_valid'][:, case_id].astype(bool)
    it = isa['time'][:, case_id][valid] - .02
    fig, ax = plt.subplots(6, 2, figsize=(13, 12), sharex=True)
    for row, axis in enumerate((0, 1, 2, 6, 7, 8)):
        for col, (t, truth, pred) in enumerate((
            (it, isa['est_label'][valid, case_id, axis], isa['est_prediction'][valid, case_id, axis]),
            (md['estimator_time'], md['force_label'][:, axis], md['force_prediction'][:, axis]))):
            ax[row, col].plot(t, truth, label='Physics label', lw=.8, color='#1f77b4')
            ax[row, col].plot(t, pred, label='Prediction', lw=.8, alpha=.85, color='#d95f02')
            ax[row, col].set_ylabel(f'{"Left" if axis < 6 else "Right"} F{"xyz"[axis % 6]} (N)')
        lo = min(a.get_ylim()[0] for a in ax[row]); hi = max(a.get_ylim()[1] for a in ax[row])
        for a in ax[row]: a.set_ylim(lo, hi)
    ax[0, 0].set_title('Isaac nominal'); ax[0, 1].set_title('MuJoCo nominal: current deployment scene')
    for a in ax[0]: a.legend(loc='upper right')
    for a in ax[-1]: a.set_xlabel('Simulation time (s)')
    fig.suptitle('Fixed V5.3 model_15000: 15 kg payload, straight/start/stop program\nSeparate closed-loop trajectories; labels are 20 ms interval means', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(FOLDER / 'force_prediction_comparison.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    sv = isa['est_speed_valid'][:, case_id].astype(bool)
    for axis in range(3):
        series = [(isa['time'][:, case_id][sv] - .02, isa['est_speed_label'][sv, case_id, axis], isa['est_speed_prediction'][sv, case_id, axis], isa['time'][:, case_id], isa['reference'][:, case_id, axis]),
                  (md['estimator_time'], md['velocity_label'][:, axis], md['velocity_prediction'][:, axis], md['tracking_time'], md['tracking_reference'][:, axis])]
        for col, (t, truth, pred, rt, ref) in enumerate(series):
            ax[axis, col].plot(t, truth, label='Actual cart velocity', color='#1f77b4', lw=.8)
            ax[axis, col].plot(t, pred, label='Estimated cart velocity', color='#d95f02', lw=.8)
            ax[axis, col].plot(rt, ref, label='Filtered command', color='black', lw=1, ls='--')
            ax[axis, col].set_ylabel(('vx (m/s)', 'vy (m/s)', 'wz (rad/s)')[axis])
        lo = min(a.get_ylim()[0] for a in ax[axis]); hi = max(a.get_ylim()[1] for a in ax[axis])
        for a in ax[axis]: a.set_ylim(lo, hi)
    ax[0, 0].set_title('Isaac nominal'); ax[0, 1].set_title('MuJoCo nominal')
    for a in ax[0]: a.legend(fontsize=8)
    for a in ax[-1]: a.set_xlabel('Simulation time (s)')
    fig.suptitle('Velocity estimation and actual tracking: 15 kg payload, straight program', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(FOLDER / 'velocity_prediction_tracking.png', dpi=140); plt.close(fig)

    names = {'isaac_nominal': 'Isaac 名义／确定性', 'isaac_full': 'Isaac L13 随机化／确定性',
             'isaac_sampled': 'Isaac L13 随机化／采样动作', 'mujoco_nominal': 'MuJoCo 名义／确定性（本轮新测）'}
    lines = ['# V5.3 显式估力、车速估计与闭环控制：当前实测', '',
             '固定 `v53_final_20k/model_15000.pt`，HOMIE 为 `model_60000.pt`。本轮补充 MuJoCo 实测，并从既有 Isaac 原始数组重新计算指标。没有训练、调参、改变力标签、修改抓握或按分数调整物理。', '',
             '12 种载荷／指令组合：5/15/25 kg × 直行、左转、右转、倒推；每个程序 24 s，含静止、起步、主运动、制动、再起步／反向转弯、停车。Isaac 每组合两个环境，MuJoCo 每组合一个；不是跨引擎相同轨迹，也不是多个独立训练种子。全部 12 个命令及滤波参考序列已逐帧对照通过。', '',
             '## 力估计', '', '下表是各手各轴 MAE，单位 N，不是合力，不是 RMSE。标签均为手对车、车体航向坐标、四个物理子步的周期均值。Isaac 为手子树动力学代理，MuJoCo 为 weld 约束重建；跨引擎物理／标签等价性不能仅凭统一命名保证。', '',
             '| 条件 | 左 Fx | 左 Fy | 左 Fz | 右 Fx | 右 Fy | 右 Fz |', '|---|---:|---:|---:|---:|---:|---:|']
    for k, st in summary['conditions'].items():
        lines.append('| ' + names[k] + ' | ' + ' | '.join(f'{st["force"]["mae"][i]:.3f}' for i in (0, 1, 2, 6, 7, 8)) + ' |')
    lines += ['', '## 车速估计与实际控制分开统计', '',
              '估计误差：预测车速减真实车速；跟踪误差：真实车速减滤波参考。前两维单位 m/s，第三维 rad/s。都报告全程 MAE。', '',
              '| 条件 | 估计 vx / vy / wz | 实际跟踪 vx / vy / wz |', '|---|---|---|']
    for k, st in summary['conditions'].items():
        lines.append(f'| {names[k]} | {vector(st["speed"]["mae"])} | {vector(st["tracking"]["mae"])} |')
    lines += ['', '## 结果边界', '',
              '- MuJoCo 完成 12/12 程序、14,376 个估计评分时刻，未触发其跌倒判断。当前部署 runner 不含完整 Isaac 任务终止监督，因此不能把它写成“按相同过载／轮离地标准全部通过”。',
              '- MuJoCo 沿用当前正式部署入口的 `nominal + urdf + explicit_pd`，没有采用历史 `rigid_limit + physx` 优化分数配置。',
              '- Isaac 名义和 L13 组同时改变了物理分布、观测噪声和外部扰动，二者的误差差不能全部归因于输入噪声。',
              '- 名义 MuJoCo 已关闭观测噪声，估力仍明显变差。继续增加白噪声不是已经验证的迁移修复；减小训练噪声的收益也尚需重新训练消融。',
              '- 当前模型的原训练范围并未确保跨引擎精度。零跌倒不等于速度跟踪合格：倒推主阶段实际约 −0.074 至 −0.102 m/s，参考约 −0.341 m/s。',
              '- 最新可见训练目录另有 model_15200；此处有意固定为已有 Isaac 成绩的 model_15000，避免把不同检查点混成 sim2sim 差异。', '',
              '## 轨迹', '', '固定展示 15 kg 直行工况，未按预测误差选择片段。', '',
              '![逐手估力](force_prediction_comparison.png)', '', '![车速估计与控制](velocity_prediction_tracking.png)', '',
              '完整数值：[summary.json](summary.json)；MuJoCo 配置、逐工况成绩、源文件与权重哈希：[report.json](report.json)。原始逐帧数据在本目录的 `payload_*.npz`；Isaac 原始数据来自 `../v53_trained_grasp_20260928/`。', '',
              '复现：`python scripts/eval_v53_estimator_sim2sim_status.py --checkpoint <同一检查点> --output <新目录>`；脚本禁止覆盖已有目录。汇总脚本：`scripts/summarize_v53_estimator_status.py`。']
    (FOLDER / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    for k, st in summary['conditions'].items():
        print(k, 'force', vector([st['force']['mae'][i] for i in (0, 1, 2, 6, 7, 8)]),
              'speed', vector(st['speed']['mae']), 'tracking', vector(st['tracking']['mae']))


if __name__ == '__main__':
    main()
