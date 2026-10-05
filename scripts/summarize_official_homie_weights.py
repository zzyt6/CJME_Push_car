"""Summarize downloaded official HOMIE weights and isolated evaluations."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from eval_homie_body_velocity import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/bench/homie_official_weights_20260928'
SOURCE = ROOT / '.cache/homie_official_weights'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vec(v):
    return ' / '.join(f'{n:.4f}' for n in v)


def main():
    models = {
        'deploy.onnx': ('OpenHomie v1', '33071f79f1884bb122bd45631c8c60d61dac1eb8',
                        'https://raw.githubusercontent.com/InternRobotics/OpenHomie/33071f79f1884bb122bd45631c8c60d61dac1eb8/HomieDeploy/deploy.onnx', 456, 12),
        'GR00T-WholeBodyControl-Walk.onnx': ('HOMIE v2 / Decoupled WBC Walk', 'b042411fae38ee4d1af9aac82a37a1f8d14d6dd0',
                        'https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx', 516, 15),
        'GR00T-WholeBodyControl-Balance.onnx': ('HOMIE v2 / Decoupled WBC Balance', 'b042411fae38ee4d1af9aac82a37a1f8d14d6dd0',
                        'https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx', 516, 15)}
    manifest = {'checked_on': '2026-09-28', 'models': {}}
    for name, (label, commit, url, inputs, actions) in models.items():
        f = SOURCE / name
        manifest['models'][name] = dict(label=label, commit=commit, url=url, path=str(f), bytes=f.stat().st_size,
                                       sha256=sha(f), input_dim=inputs, action_dim=actions)
    assert manifest['models']['deploy.onnx']['sha256'] == sha(ROOT / 'HomieDeploy/deploy.onnx')
    assert manifest['models']['GR00T-WholeBodyControl-Walk.onnx']['sha256'] == '7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407'
    assert manifest['models']['GR00T-WholeBodyControl-Balance.onnx']['sha256'] == 'f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18'
    (SOURCE / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    reports = {}
    for name in ('v1_isaac_complete', 'v1_mujoco_complete', 'v1_isaac_local_gains', 'v1_mujoco_local_gains'):
        r = json.loads((OUT / name / 'report.json').read_text()); x = np.load(OUT / name / 'samples.npz')
        valid = x['valid'].astype(bool) & (x['time'] >= 2.)
        measured = metrics(x['prediction'][valid], x['truth_com'][valid])
        np.testing.assert_allclose(measured['mae'], r['current']['mae'], atol=1e-12)
        assert r['onnx_action_diagnostic_max_abs'] == 0.
        if 'isaac' in name:
            assert r['parent_verified'] and r['child_exit_code'] == 0
        reports[name] = r
    v2 = json.loads((OUT / 'v2_native_mujoco/report.json').read_text())
    assert v2['complete_cases'] == 6 and v2['action_output_unchanged_max_abs'] < 2e-6
    reports['v2_native_mujoco'] = v2
    local = {name: json.loads((ROOT / 'logs/bench/homie_body_velocity_20260928' / name / 'report.json').read_text())
             for name in ('isaac_free', 'mujoco_free_final')}
    summary = dict(models=manifest, official_reports=reports,
                   local_reference={k: v['current'] for k, v in local.items()},
                   interpretation='V1 controlled interface checks use project 27-DoF plant. V2 native test changes plant and DoFs; do not attribute difference to weights alone.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    # Visualize unmodified measurements; no smoothing or per-model tuning.
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey='row')
    lx = np.load(ROOT / 'logs/bench/homie_body_velocity_20260928/mujoco_free_final/samples.npz')
    ox = np.load(OUT / 'v1_mujoco_complete/samples.npz')
    for row, (idx, case) in enumerate(((0, 'straight'), (3, 'reverse'))):
        nx = np.load(OUT / 'v2_native_mujoco' / f'{case}.npz')
        data = [(lx['time'][:, idx], lx['truth_com'][:, idx, 0], lx['prediction'][:, idx, 0], lx['command'][:, idx, 0]),
                (ox['time'][:, idx], ox['truth_com'][:, idx, 0], ox['prediction'][:, idx, 0], ox['command'][:, idx, 0]),
                (nx['time'], nx['truth_com'][:, 0], nx['prediction'][:, 0], nx['command'][:, 0])]
        for col, (t, truth, pred, cmd) in enumerate(data):
            ax = axes[row, col]
            ax.axvspan(0, 2, color='gray', alpha=.12)
            ax.plot(t, truth, lw=1, color='#1f77b4', label='Actual body vx')
            ax.plot(t, pred, lw=1, color='#d95f02', label='Velocity-head diagnostic')
            ax.plot(t, cmd, lw=1, ls='--', color='black', label='Command')
            if row == 0:
                ax.set_title(('Local weights / project plant', 'Official v1 / project plant + official gains', 'Official v2 / official 29-DoF plant')[col])
                ax.legend(fontsize=7, loc='lower right')
            if col == 0: ax.set_ylabel(f'{case}: vx (m/s)')
            if row == 1: ax.set_xlabel('Simulation time (s)')
    fig.suptitle('Official HOMIE weight diagnostics in MuJoCo\nV2 uses a different robot and control setup: this is not a weights-only comparison', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(OUT / 'mujoco_official_comparison.png', dpi=140); plt.close(fig)

    lines = ['# HOMIE 官方权重核查与实际仿真测试', '',
             '日期：2026-09-28。已获取旧版官方 ONNX 和新版 Walk／Balance ONNX，执行独立仿真测试；当前推车训练、冻结 HOMIE 默认路径及权重均未替换。', '',
             '**结论：官方权重值得测试，而且已测试。旧版官方权重没有消除当前 MuJoCo 链路的误差；新版 v2 在其官方原生 29 关节场景中表现较好，但不是当前接口的直接替换件。**', '',
             '## 官方到底发布了什么', '',
             '- OpenHomie README 的 2026-07 更新说明：HOMIE v2 RL policy 位于 NVIDIA 的 Decoupled WBC，并另外推荐 SONIC。这里核查的是作者指向的 Decoupled WBC；SONIC 属于不同的全身运动参考架构，未混入本次 HIM 对照。',
             '- 旧版 `HomieDeploy/deploy.onnx` 是官方说明中的示例部署权重。仓库没有同时提供这个文件对应的完整训练 `.pt`／优化器记录；不能据此宣称它是作者所有实验的最优检查点。',
             '- 当前项目其实已经附带同一个 `HomieDeploy/deploy.onnx`。本轮从固定官方 commit 重新下载，SHA256 完全一致。',
             '- 新版权重有 `GR00T-WholeBodyControl-Walk.onnx` 与 `GR00T-WholeBodyControl-Balance.onnx`；实际下载了 LFS 内容，核验大小和 SHA256，不是 132 字节指针文件。', '',
             '| 模型 | 输入 | 动作 | 文件大小 |', '|---|---:|---:|---:|']
    for name, m in manifest['models'].items():
        lines.append(f'| {m["label"]} | {m["input_dim"]} | {m["action_dim"]} | {m["bytes"]:,} bytes |')
    lines += ['', '下载文件及来源／哈希清单位于 `.cache/homie_official_weights/manifest.json`。模型保留原始文件，读取内部估计输出只在内存 ONNX 图中追加诊断输出，没有改变权重。', '',
              '## 旧版：同一项目机器人链路中的实测', '',
              '不带车、六种行走程序、每种 24 s，舍弃前 2 s，无输入噪声和域随机化。真值为 pelvis 刚体质心本体坐标速度；下表三项均是 m/s，第三项为 vz，非转向角速度。', '',
              '| 权重／增益设置 | Isaac 估速 MAE vx/vy/vz | MuJoCo 估速 MAE vx/vy/vz | 终止／跌倒记录 |', '|---|---|---|---|',
              f'| 本地 model_60000／本地增益（前轮） | {vec(local["isaac_free"]["current"]["mae"])} | {vec(local["mujoco_free_final"]["current"]["mae"])} | 0 / 0 |',
              f'| 官方 v1／本地增益 | {vec(reports["v1_isaac_local_gains"]["current"]["mae"])} | {vec(reports["v1_mujoco_local_gains"]["current"]["mae"])} | **3 / 0** |',
              f'| 官方 v1／官方 C++ 部署增益 | {vec(reports["v1_isaac_complete"]["current"]["mae"])} | {vec(reports["v1_mujoco_complete"]["current"]["mae"])} | 0 / 0 |', '',
              '官方 v1 的两组输入均按公开 LCMAgent 部署接口处理：转向命令缩放为 0.25，而项目历史帧使用 0.5，因此只对历史中的对应命令槽做确定的单位转换。其余 q/dq、IMU、历史顺序和默认关节角一致。不是按评测分数搜索输入倍率。', '',
              'PD 以真正发送给硬件的 `g1_control.cpp` 为依据，而不是仅用 Python 中声明但未发送的数组。官方部署 hip Kp=150、knee Kp=300；本地训练为 100／150。官方肩 Kp=150、腕 roll/pitch/yaw Kp=10/10/5，与当前训练配置也有差别。Kd 按 C++ 数组恢复。', '',
              '“官方 v1／本地增益”在 Isaac 出现 3 次终止并自动重置，其 MAE 使用有效非重置样本，不能把该数字当作无失败模型的完整性能排名。两引擎终止监督也不完全一致。采用官方增益后没有终止，说明权重与执行配置必须配套检查。', '',
              '官方 v1 配官方增益仍存在 MuJoCo 倒退失配：8–12 s 实际本体 vx 约 −0.063 m/s，估计约 −0.293 m/s，指令约 −0.350 m/s。不能把上一轮差距简单解释为“本地模型没训练好，换作者权重自然解决”。但这也不是对作者真机效果的否定：这里使用的是本项目 MuJoCo 机器人物理。', '',
              '## 新版 v2：在其官方原生 MuJoCo 模型上测试', '',
              '新版使用 29 个运动关节，动作控制 12 腿关节＋3 腰关节，观测为 86×6=516。我们现有模型为 27 个运动关节、12 腿动作、76×6=456，腰 roll/pitch 没有相同的控制接口。不能通过随意补零输入或丢弃两维动作，把它当作无差别替换。', '',
              '因此单独使用官方 `g1_gear_wbc.xml`、其 49 个网格、官方 standalone 模拟器的 PD 和观测方法；49 个网格的本地文件均经官方 Git LFS 指针 blob 哈希验证后复用。输入构造直接复用了已核查的官方纯函数，未启动键盘、网络或机器人硬件接口。速度程序沿用同样六工况及 0.4 s 指令滤波。', '',
              '使用当前发布的 Balance／Walk 文件；按官方规则在指令模长不超过 0.05 时用 Balance，否则用 Walk。旧 standalone YAML 中的 ft92/ft109 文件名已过时，本轮用当前控制配置指向的正式发布文件。', '',
              f'- 完成 **{v2["complete_cases"]}/6** 程序，无跌倒。',
              '- 倒退稳态（8–12 s）：参考约 −0.350 m/s，真实本体 vx 约 **−0.325 m/s**；直行真实 vx 约 0.561 m/s，参考约 0.600 m/s。',
              f'- 内部速度头诊断 MAE：**{vec(v2["estimation"]["mae"])} m/s**。这是读取同类 HIM 结构的编码器前三通道并按公开 base_lin_vel 缩放 2 换算；原 ONNX 对外只输出动作，因此该数值不是官方公开的测速指标或新增传感器接口。',
              '- 该试验改变了权重、关节自由度、机器人 XML、接触／驱动配置和策略切换。结果显示这套官方组合能够在 MuJoCo 运行良好，不能据此计算“仅更换权重提升多少”，也不能代替 v2 在 Isaac 或双手推车中的验收。', '',
              '![MuJoCo 三组轨迹](mujoco_official_comparison.png)', '',
              '## 对当前任务的建议', '',
              '保留当前推车基线，使用这些官方文件作为独立对照。旧版的实测说明单换权重不够；新版原生场景提供了一个值得研究的、倒退表现较好的参考系统。应先比较我们机器人仿真与该参考系统的执行／接触响应，并明确是继续修复当前 27 关节链路，还是将新版完整的 29 关节接口适配后重新验收高层。当前没有完成后者，不应直接覆盖冻结 HOMIE 或把现有高层检查点视为已适配。', '',
              '## 证据与复现', '',
              '- [汇总原始数字](summary.json)，各组子目录保存 `report.json` 与完整逐帧数组；两次正式 Isaac 子进程均 `exit=0`、`parent_verified=true`。',
              '- 旧版内存图追加估计输出后，在 330 条真实观测上的动作与原 ONNX 最大差为 **0**；新版在运行样本上也逐段验证动作不变。',
              '- 初始旧版试验曾因评测器增益字典缺少 wrist_roll 键而未运行；这两个准备失败目录不进入统计。修复只在新增评测脚本中，不涉及生产控制器。',
              '- 复现旧版：`scripts/eval_official_homie_velocity.py --engine isaac|mujoco --gain_profile local|official_cpp --output <新目录>`。',
              '- 复现新版原生模拟器：`scripts/eval_homie_v2_native_mujoco.py --output <新目录>`。',
              '- 汇总：`scripts/summarize_official_homie_weights.py`。', '',
              '官方来源：[OpenHomie 更新说明](https://github.com/InternRobotics/OpenHomie/blob/33071f79f1884bb122bd45631c8c60d61dac1eb8/README.md#L29)、[v1 部署权重说明](https://github.com/InternRobotics/OpenHomie/blob/33071f79f1884bb122bd45631c8c60d61dac1eb8/HomieDeploy/README.md#L31)、[实际 C++ 增益](https://github.com/InternRobotics/OpenHomie/blob/33071f79f1884bb122bd45631c8c60d61dac1eb8/HomieDeploy/unitree_sdk2/g1_control.cpp#L85)、[v2 模型目录](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/decoupled_wbc/sim2mujoco/resources/robots/g1/policy)、[v2 观测与策略切换](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/decoupled_wbc/control/policy/g1_gear_wbc_policy.py)。']
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('Saved', OUT/'comparison.md')


if __name__ == '__main__':
    main()
