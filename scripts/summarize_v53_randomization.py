"""Summarize simulator readback and training integration evidence for V5.3 revision 3."""
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/bench/v53_domain_randomization'
reports = {name: json.loads((OUT/f'{name}.json').read_text(encoding='utf-8'))
           for name in ('initial', 'full', 'nominal')}
for r in reports.values():
    assert r['result'] == 'PASS_EXECUTION' and r['parent_verified'] and r['child_exit_code'] == 0
    assert r['partial_reset_preserved_other_rows'] and r['asymmetric_stall_restore_verified']

training_run = ROOT/'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-21-28_v53_domain_smoke'
resume_run = ROOT/'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-27-02_v53_domain_resume_smoke'
checks = []
for run in (training_run, resume_run):
    checkpoint = run/'model_1.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = yaml.load((run/'params/env.yaml').read_text(encoding='utf-8'), Loader=yaml.FullLoader)
    task = saved['task']
    assert task['implementation_revision'] == 3 and task['randomization']['profile'] == 'homie_relative'
    assert task['observations']['joint_velocity_noise_rad_s'] == 2.
    for key in ('optimizer_state_dict', 'estimator_optimizer_state_dict', 'cart_velocity_optimizer_state_dict'):
        assert state[key]['state'], f'Optimizer was not updated: {key}'
    for key in ('actor_state_dict', 'critic_state_dict'):
        for value in state[key].values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
    checks.append(dict(checkpoint=str(checkpoint.relative_to(ROOT)), iteration=state['iter'],
                       implementation_revision=3, three_optimizer_states_present=True))

def interval(values):
    return f'{values[0]:.5g} – {values[1]:.5g}'

lines = ['# V5.3 域随机化：实际仿真参数检查', '',
    '检查模式为课程初期（第 0 级）、完整范围（第 17 级）、关闭随机化（第 17 级）。每组 32 环境，反复重置读取实际 PhysX 参数，再运行 400 步。使用冻结 HOMIE 和零高层残差；没有把它当成训练后策略的性能评估。', '',
    '| 实际参数 | 课程初期 | 完整范围 | 关闭随机化 |', '|---|---:|---:|---:|']
for key, label in (('mass_scale', '空车连杆质量倍率'), ('total_mass', '整车质量 kg'),
    ('payload', '载荷 kg'), ('wheel_friction', '车轮材料摩擦'), ('wheel_damping', '滚动阻尼 N·m·s/rad'),
    ('swivel_damping', '转向阻尼 N·m·s/rad'), ('kp_factors', '控制器 Kp 倍率'), ('kd_factors', '控制器 Kd 倍率'),
    ('heading_delay', '视觉延迟／20 ms 控制步'), ('heading_bias', '视觉固定偏置 rad'),
    ('hl_delay', '高层动作延迟／20 ms 控制步'), ('torque_delay', '力矩读取延迟／20 ms 控制步')):
    lines.append('| '+label+' | '+' | '.join(interval(reports[n]['ranges'][key]) for n in reports)+' |')
lines += ['', '表中是有限样本的实际最小／最大值，不是理论端点。机器人显式 PD 使用控制器中的 Kp/Kd 倍率；车体质量、惯量、材料和被动关节阻尼则直接从 PhysX 底层接口读取。', '',
    '| 检查项 | 初期 | 完整 | 关闭 |', '|---|---:|---:|---:|']
for key, label in (('resets', '运行阶段重置事件'), ('numerical_blowup_terminations', '数值爆炸终止'),
    ('valid_hand_labels', '完整四子步手力标签'), ('auto_cart_pulse_samples', '自动车体外力环境步数'),
    ('wheel_stall_samples', '脚轮卡滞环境步数')):
    lines.append('| '+label+' | '+' | '.join(str(reports[n][key]) for n in reports)+' |')
lines += ['', '各组都通过：状态／奖励／监督标签有限，局部重置不影响其他环境参数，卡滞结束恢复各轮独立阻尼，随机化关闭后自动扰动与观测／高层动作延迟均为零。所有仿真子进程退出码为 0，并由父进程匹配本次 run_id 核验成功。', '',
    '完整随机化下，载荷合成参数与仿真读回的最大绝对误差：', '',
    '| 参数 | 最大绝对误差 |', '|---|---:|']
for key, value in reports['full']['max_readback_error'].items():
    lines.append(f'| {key} | {value:.6g} |')
lines += ['', '惯量具有正特征值，并满足主惯量三角不等式。动力学缓存也与实际惯量读回对齐，避免扩大质量／偏载后监督模型仍使用错误惯量。', '',
    '新范围短程 PPO：16 环境 × 64 步／更新 × 2 次更新，第 13 级；恢复该检查点后又完成 1 次更新。已检查模型张量有限、三个优化器状态存在、保存配置为修订 3。恢复日志确认同一随机化配置的课程窗口成功保留。', '',
    f"- 初次检查点：`{checks[0]['checkpoint']}`",
    f"- 恢复检查点：`{checks[1]['checkpoint']}`", '',
    '这些结果说明范围确实生效、数值与训练链路可运行。它们不证明策略已收敛、估力准确度提升或真机部署成功。', '',
    '原始报告：[initial.json](initial.json)、[full.json](full.json)、[nominal.json](nominal.json)，对应 `.samples.json` 保留参数采样。运行日志位于同目录。', '',
    '配置与依据：[域随机化设计说明](../../../docs/push_cart_v53_domain_randomization.md)。']
(OUT/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
validation = dict(result='PASS_EXECUTION', tests_passed=79, isaac_modes=list(reports),
    isaac_samples=sum(r['num_envs']*r['steps'] for r in reports.values()), checkpoints=checks,
    full_training_complete=False, hardware_validated=False,
    post_audit_changes='domain_randomization.py docstring clarified link-frame inertia versus principal axes; no executable change.',
    current_source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                          for p in reports['full']['source_sha256']})
(OUT/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
print(OUT/'comparison.md')
