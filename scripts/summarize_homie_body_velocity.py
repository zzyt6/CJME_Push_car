"""Summarize the measured HOMIE/HIM body velocity audit without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval_v53_estimator_sim2sim_status import metrics

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'logs/bench/homie_body_velocity_20260928'
NAMES = {'isaac_free': 'Isaac、不带车', 'mujoco_free_final': 'MuJoCo、不带车',
         'isaac_cart': 'Isaac、V5.3 推车', 'mujoco_cart': 'MuJoCo、V5.3 推车'}


def vec(values):
    return ' / '.join(f'{x:.4f}' for x in values)


def main():
    reports, data, statistics = {}, {}, {}
    for name in NAMES:
        reports[name] = r = json.loads((FOLDER / name / 'report.json').read_text())
        data[name] = x = np.load(FOLDER / name / 'samples.npz')
        assert r['result'] == 'PASS_EXECUTION'
        if name.startswith('isaac'):
            assert r['parent_verified'] and r['child_exit_code'] == 0
        valid = x['valid'].astype(bool) & (x['time'] >= 2.)
        current = metrics(x['prediction'][valid], x['truth_com'][valid])
        np.testing.assert_allclose(current['mae'], r['current']['mae'], atol=1e-12)
        case_stats = []
        for i, case in enumerate(r['cases']):
            selection = valid[:, i]
            stable = selection & (x['time'][:, i] >= 8.) & (x['time'][:, i] < 12.)
            case_stats.append({**case, 'current': metrics(x['prediction'][selection, i], x['truth_com'][selection, i]),
                               'steady_true': x['truth_com'][stable, i].mean(0).tolist(),
                               'steady_prediction': x['prediction'][stable, i].mean(0).tolist(),
                               'steady_command': x['command'][stable, i].mean(0).tolist()})
        statistics[name] = dict(current=current, action_time_to_next=r['action_time_to_next'],
                                origin=r['origin'], reference_point_difference=r['reference_point_difference'],
                                cases=case_stats,
                                gyro_rms=np.sqrt(np.square(x['gyro'][valid]).mean(0)).tolist(),
                                joint_velocity_rms=float(np.sqrt(np.square(x['dq'][valid]).mean())))
    assert len({r['low_sha256'] for r in reports.values()}) == 1
    np.testing.assert_allclose(data['isaac_free']['command'], data['mujoco_free_final']['command'], atol=1e-6)
    checks = json.loads((FOLDER / 'replay_checks.json').read_text())
    for check in checks.values():
        assert check['cpu_replay_max_abs'] < 2e-6
        assert check['latest_frame_reconstruction_max_abs'] < 1e-6
        assert check['noncommand_history_shift_max_abs'] == 0.
        assert check['com_velocity_identity_max_abs'] < 1e-6
    summary = dict(low_sha256=reports['isaac_free']['low_sha256'], conditions=statistics, replay_checks=checks,
                   free_commands_match=True, all_runs_without_termination=all(r['falls_or_terminations'] == 0 for r in reports.values()))
    (FOLDER / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey='row')
    for row, (case, label) in enumerate(((0, 'Forward'), (3, 'Reverse'))):
        for col, name in enumerate(('isaac_free', 'mujoco_free_final')):
            x, ax = data[name], axes[row, col]
            t = x['time'][:, case]
            ax.axvspan(0, 2, color='gray', alpha=.12)
            ax.plot(t, x['truth_com'][:, case, 0], label='Actual body vx', color='#1f77b4', lw=1.)
            ax.plot(t, x['prediction'][:, case, 0], label='HIM estimate', color='#d95f02', lw=1.)
            ax.plot(t, x['command'][:, case, 0], label='Applied velocity command', color='black', ls='--', lw=1.)
            ax.set_title(f'{label}: {"Isaac" if col == 0 else "MuJoCo"}')
            ax.set_ylabel('Pelvis COM vx (m/s)')
            if row == 0: ax.legend(fontsize=8)
            if row == 1: ax.set_xlabel('Simulation time (s)')
    fig.suptitle('Frozen HOMIE model_60000, no cart, no observation noise\nActual body velocity and HIM estimate are different signals', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(FOLDER / 'free_forward_reverse.png', dpi=150)
    plt.close(fig)

    lines = ['# 冻结 HOMIE 本体速度：Isaac 与 MuJoCo 实测', '',
             '日期：2026-09-28。问题是检查当前高层 sim2sim 误差是否已经出现在底层 HIM。四组均为本轮新采集，无学习、无参数更新、无观测替换。', '',
             '**结果：在本项目当前仿真链路中，HOMIE/HIM 的本体速度预测也存在明显 sim2sim 差距；不带车时已经出现，不能只归因于新高层或双手闭链。**', '',
             '固定低层 `homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt`，四组权重哈希一致。推车组固定 V5.3 `model_15000.pt`。', '',
             '## 指标定义与范围', '',
             '- 输出为 `[vx, vy, vz]`，三项均为 m/s，表达在 pelvis 本体坐标轴；第三项不是小车角速度 wz。',
             '- 主真值使用 pelvis 刚体质心的线速度，匹配 HOMIE 训练目标。它不是全机器人系统质心速度。另记录 pelvis link 原点速度，评估参考点差异。',
             '- 物理周期 5 ms，控制 20 ms；每个程序 24 s，舍弃最初 2 s，并屏蔽历史未填满／重置样本。两端关闭观测噪声、参数随机化、随机延迟和推扰，策略用确定性输出。',
             '- 不带车：6 个独立场景，直行、左右转、后退、左右侧移，HOMIE 标称上身姿态，直接向 HOMIE 给相同滤波指令；每引擎 6,606 个评分时刻。',
             '- 推车：5/15/25 kg × 直行、左右转、倒推，共 12 个场景；高层产生实际 HOMIE 指令，每引擎 13,212 个评分时刻。',
             '- 不带车 Isaac 使用原 `HomieIsaacLabEnv`。MuJoCo 使用项目已有 URDF 参数对齐机器人场景，移除车体、抓握和相关接触对，恢复 HOMIE 标称姿态；保留原有 native upper PD 模式。推车组使用当前 `nominal + urdf + explicit_pd` 路径。不是同一物理引擎的轨迹重放，也不是 HOMIE 论文真机精度测量。', '',
             '## 本体速度估计 MAE', '', '| 条件 | vx | vy | vz |', '|---|---:|---:|---:|']
    for name, label in NAMES.items():
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.4f}' for v in statistics[name]['current']['mae']) + ' |')
    lines += ['', '所有值都是预测减物理真值的误差，不是实际速度减指令的跟踪误差。四组记录均无终止／跌倒；MuJoCo 与 Isaac 的终止监督并不完全相同，不能据此认定相同成功标准全部通过。', '',
              '## 方向并非同等恶化', '',
              '| 不带车工况 | Isaac vx MAE | MuJoCo vx MAE |', '|---|---:|---:|']
    for i, c in enumerate(statistics['isaac_free']['cases']):
        lines.append(f'| {c["scenario"]} | {c["current"]["mae"][0]:.4f} | {statistics["mujoco_free_final"]["cases"][i]["current"]["mae"][0]:.4f} |')
    lines += ['', '不带车倒退工况的 8–12 s 均值：', '',
              '| 引擎 | 指令 vx | 实际 vx | HIM 预测 vx |', '|---|---:|---:|---:|']
    for name in ('isaac_free', 'mujoco_free_final'):
        c = statistics[name]['cases'][3]
        lines.append(f'| {NAMES[name]} | {c["steady_command"][0]:.4f} | {c["steady_true"][0]:.4f} | {c["steady_prediction"][0]:.4f} |')
    lines += ['', 'MuJoCo 中机器人实际只以约 −0.103 m/s 后退，HIM 却预测约 −0.336 m/s。这是带明显方向／工况依赖的失配，不能用一个统一偏置就保证修复。它与“身体运动确实发生了变化”同时出现，不能仅凭预测接近指令就断言网络直接复制指令。', '',
              '![不带车前进与倒退](free_forward_reverse.png)', '',
              '## 核对了哪些可能的统计错误', '',
              '1. **检查点／缩放**：模型 strict 加载，同一低层 SHA256；训练保存配置与当前默认的增益、动作缩放、观测缩放、历史长度、标称关节位置、armature、阻尼、IMU link、高度参考和 5 ms 时间步一致。速度输出除以 lin_vel=2，恢复 m/s。',
              '2. **输入重构**：从实际记录的 q/dq、IMU、指令等重构最新 76 维帧，最大差小于 6e−8。六帧历史除有意更新的命令槽外，移位差为 0。',
              '3. **推理一致性**：四组真实观测分别在 CPU 批量重放，共 3,960 条；相对在线预测最大差约 1.16e−6 m/s，远低于被讨论的 0.1 m/s。',
              '4. **参考点**：`v_COM = v_link + omega × r_COM` 在本体坐标验证，最大差小于 2.4e−7 m/s。两端 pelvis COM 的 z 偏移约 −0.07605 m。参考点确实影响部分速度，但不能解释整个前向误差。',
              '5. **时间目标**：HIM 实际监督代码用当前历史预测下一控制步 critic 的线速度。故同时记录了“动作前预测 vs 动作后真值”，不只按当前时刻评分。', '',
              '按训练的一步目标评估，MAE 为：', '', '| 条件 | vx / vy / vz，m/s |', '|---|---|']
    for name in NAMES:
        lines.append(f'| {NAMES[name]} | {vec(statistics[name]["action_time_to_next"]["mae"])} |')
    lines += ['', 'Isaac 不带车 vz 在一步预测口径下变好，但 MuJoCo 前向误差仍约 0.111／0.121 m/s。将输出错看成当前估计并不能解释主要跨域差距；也没有为了降低误差而选择任意最佳时间平移。', '',
              '## 为什么无输入噪声也会差', '',
              '关闭观测噪声只移除了人为叠加的测量误差，没有使两引擎的实际状态历史、接触／驱动响应和“本体历史到速度”的映射相同。HIM 从关节、IMU、动作和命令历史间接估速，它并非直接读取仿真速度真值。', '',
              '本次不带车时，两引擎的实际运动已有差别；例如倒退速度明显不同。所有自由运动工况汇总，关节速度 RMS 约从 Isaac 的 0.675 rad/s 变为 MuJoCo 的 1.267 rad/s，机体角速度也变化。这支持存在动力学／执行轨迹差异，但尚未把原因独立定位到接触、摩擦、增益、惯量或某个单一参数。', '',
              '高层车速头和估力头又使用 HIM 速度输入，因此上游失配可能继续影响高层。当前证据足以否定“差距只出在新高层”的归因，但尚不能计算高层误差中有多少百分比由 HIM 造成；这需要同轨迹的 HIM 输入替换诊断或其他受控干预。', '',
              '后续应先核验不带车的实际步态／速度响应与低层估计，再定位推车额外引入的差异。没有证据支持此时单靠扩大高层白噪声来补救，也不应先按 MuJoCo 分数改物理参数后直接声称迁移成功。', '',
              '## 原始文件与执行核验', '',
              '- 汇总：[summary.json](summary.json)；重放／历史／参考点核验：[replay_checks.json](replay_checks.json)。',
              '- 每组目录含 `report.json` 和 `samples.npz`，保存 q/dq、IMU、完整 456 维低层输入、HIM 输出、两种参考点真值、动作前预测、指令及时间。',
              '- 两次 Isaac 子进程均返回 0，完成报告中的 `parent_verified=true`。PowerShell 外层重定向把 Kit 的 stderr 警告包装为非零 shell 状态；数据是否完成依据子进程返回码、运行 ID、报告和完整数组共同核验。',
              '- MuJoCo 不带车场景裁剪准备阶段有两次失败，目录 `mujoco_free`／`mujoco_free_complete` 不含成功评测；统计仅采用 `mujoco_free_final`。',
              '- 复现入口：`scripts/eval_homie_body_velocity.py --engine isaac|mujoco --mode free|cart --output <新目录>`；汇总：`scripts/summarize_homie_body_velocity.py`。', '',
              '监督时序源码：[HIM estimator](../../../HomieRL/rsl_rl/rsl_rl/modules/him_estimator.py)、[HIM PPO transition](../../../HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py)。本次未修改这些训练或控制实现。']
    (FOLDER / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({k: v['current']['mae'] for k, v in statistics.items()}, indent=2))


if __name__ == '__main__':
    main()
