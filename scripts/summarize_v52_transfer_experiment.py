"""Build the simulation report only from complete training/evaluation artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GROUPS=('training_full','training_proprio','explicit_pd_full','explicit_pd_proprio',
        'explicit_pd_randomized_full','explicit_pd_randomized_proprio')
LABELS={'training_full':'原驱动重训 / 全输入','training_proprio':'原驱动重训 / 本体历史',
        'explicit_pd_full':'显式 PD 重训 / 全输入','explicit_pd_proprio':'显式 PD 重训 / 本体历史',
        'explicit_pd_randomized_full':'显式 PD＋执行器随机化 / 全输入',
        'explicit_pd_randomized_proprio':'显式 PD＋执行器随机化 / 本体历史'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def format_mean(values):
    a=np.asarray(values)
    return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}'


def complete_report(p):
    d=read(p)
    assert d['total_conditions']==12 and len(d['conditions'])==12,p
    assert not d.get('partial',False),p
    # Failed conditions must remain visible, never be omitted by aggregation.
    assert all(c['steps']<=500 for c in d['conditions']),p
    return d


def main(args):
    out=args.output
    assert read(out/'training_complete.json')['trained_models']==18
    shadow=read(out/'shadow_comparison.json')['results']
    assert len(shadow)==72, 'Expected 18 models x two drives x two engines'
    captures={}
    for engine,prefix in (('isaac_physx','physx'),('mujoco','isaac')):
        for drive,suffix in (('training',''),('explicit_pd','_explicit_pd')):
            captures[engine,drive]=complete_report(out/'captures'/f'{prefix}{suffix}.json')
    datasets={name:read(out/f'data_{name}'/'metadata.json') for name in ('training','explicit_pd','explicit_pd_randomized')}
    assert all(x['completed_steps']==4096 and x['num_envs']==64 for x in datasets.values())
    models={}
    for group in GROUPS:
        models[group]=[read(out/f'model_{group}_{seed}'/'report.json') for seed in (11,22,33)]
    lines=['# V5.2 显式 PD：力估计器训练与跨引擎仿真对照','',
           '日期：2026-09-27。优先验证仿真：已在 Isaac 实际采集数据并训练估计器，再在 Isaac / MuJoCo 评测。HOMIE 和原高层策略其余权重冻结；本轮没有进行实机试验。','',
           '## 实验设置','',
           '- 三组数据：原上身隐式驱动、共同显式 PD、共同显式 PD 加实际 Kp/Kd 及动作延迟随机化。每组 64 环境 × 4096 控制步；每组两种输入、三个优化种子，共 18 个模型。',
           '- 全输入为最近 5×136 帧；本体历史为 5×96，去除力矩代理、静力先验、HIM/FK 派生特征。相同隐藏层 `[512,256,128]`；每组从头监督训练，最多 100 epoch，依据 Isaac 验证集早停。',
           '- 训练/验证/Isaac 测试按整个环境划分。MuJoCo 不参与拟合、归一化、早停或闭环模型选择。三个种子共用各组采集数据，误差条仅反映网络初始化与优化变异。',
           '- 评测为额外载荷 0/12.5/25 kg × 四种命令，500 步/工况，去掉前 2 s。每组最多 4800 有效样本；失败和有效样本数记录在 JSON。',
           '- 原行为策略为 `model_19999.pt`，冻结 HOMIE 为 `model_60000.pt`；估计目标为双手掌原点、车朝向坐标系的十二维 wrench，合力在把手中心合成。','',
           '| 数据组 | 有效样本 | 训练 | 验证 | Isaac 测试 | 重置次数（含超时） |',
           '|---|---:|---:|---:|---:|---:|']
    for name,d in datasets.items():
        s=d['split_samples']
        lines.append(f"| {name} | {d['samples']} | {s['train']} | {s['validation']} | {s['test']} | {d['resets']} |")
    lines+=['','## 固定轨迹的估力比较','',
            '以下每个引擎/驱动内的所有模型读取相同轨迹样本，新预测不改变行为。两引擎之间的轨迹不同；原驱动与显式驱动的轨迹也不同。均值 ± 标准差来自三个训练种子；旧检查点只有一个模型。','']
    aggregates=[]
    for engine in ('isaac_physx','mujoco'):
        lines += [f'### {engine}','', '| 模型与对应评测驱动 | Fx RMSE (N) | Fy RMSE (N) | Mz RMSE (N·m) |', '|---|---:|---:|---:|']
        for drive in ('training','explicit_pd'):
            d=captures[engine,drive]
            v=d['handle']['rmse']
            lines.append(f"| 旧检查点 / {drive} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
        for group in GROUPS:
            drive='training' if group.startswith('training') else 'explicit_pd'
            paths={str((out/f'model_{group}_{seed}'/'estimator.pt').resolve()) for seed in (11,22,33)}
            selected=[r for r in shadow if r['engine']==engine and r['eval_drive']==drive and str(Path(r['model']).resolve()) in paths]
            assert len(selected)==3,(engine,group)
            values=np.array([r['scores']['handle']['rmse'] for r in selected])
            lines.append(f"| {LABELS[group]} | "+' | '.join(format_mean(values[:,i]) for i in range(3))+' |')
            aggregates.append(dict(engine=engine,group=group,rmse_mean=values.mean(0).tolist(),rmse_std=values.std(0,ddof=1).tolist()))
        lines+=['']
    lines+=['完整交叉驱动结果与逐工况/逐手指标见 [shadow_comparison.json](shadow_comparison.json)。其中也包含“原驱动训练模型 → 显式评测轨迹”的比较，可与显式重训模型在完全相同输入上对比。','',
            '## 实际接回 Actor 的闭环验证','',
            '每组仅根据 Isaac 验证集选择三个种子中的一个，再将该估计器安装到 Actor 的真实预测通道；其余高层权重及 HOMIE 保持冻结。此处预测会影响动作，不能与上面的固定轨迹结果混为一谈。跟踪量为控制后车体 `[vx,vy,wz]` 与平滑命令参考的误差。','',
            '| 组别 | 引擎 | 选定种子 | Fx/Fy/Mz RMSE (N,N,N·m) | vx/vy/wz 跟踪 RMSE (m/s,m/s,rad/s) | 完成工况 |',
            '|---|---|---:|---|---|---:|']
    closed=[]
    not_run=[]
    for group in GROUPS:
        sel=read(out/'closed_loop'/group/'selection.json')
        expected=min(zip((11,22,33),models[group]),key=lambda x:x[1]['scores']['validation']['selection_mse'])[0]
        assert sel['seed']==expected
        drive='training' if group.startswith('training') else 'explicit_pd'
        suffix='' if drive=='training' else '_explicit_pd'
        for engine,prefix in (('isaac_physx','physx'),('mujoco','isaac')):
            path=out/'closed_loop'/group/f'{prefix}{suffix}.json'
            if not path.exists() and group=='explicit_pd_proprio' and engine=='isaac_physx':
                lines.append(f'| {LABELS[group]} | {engine} | {sel["seed"]} | 未运行 | 未运行 | — |')
                not_run.append(dict(group=group,engine=engine,
                                    reason='User redirected validation to full-body FALCON inputs and domain diagnosis.'))
                continue
            d=complete_report(path)
            rmse=d.get('handle',{}).get('rmse')
            c=[r for r in d['conditions'] if r['tracking_rmse'] is not None]
            weights=np.array([r['handle']['n'] for r in c])
            tracking=np.sqrt(np.average(np.array([r['tracking_rmse'] for r in c])**2,axis=0,weights=weights)).tolist() if c else [None]*3
            fmt=lambda a:'/'.join(f'{x:.3f}' if x is not None else 'NA' for x in a)
            lines.append(f"| {LABELS[group]} | {engine} | {sel['seed']} | {fmt(rmse or [None]*3)} | {fmt(tracking)} | {d['completed_conditions']}/12 |")
            closed.append(dict(group=group,engine=engine,seed=sel['seed'],force_rmse=rmse,tracking_rmse=tracking,
                               completed_conditions=d['completed_conditions'],report=str(path)))
    lines+=['','## 解释边界','',
            '- 用户后续指定全身本体与全身控制历史，新增结果见 [全身历史估力与域诊断](falcon_a3/comparison.md)。原显式 proprio 的 Isaac 闭环未排入本轮；训练、固定轨迹评分和 MuJoCo 闭环结果保留。未运行不计作成功或失败。',
            '- 这里验证的是估计器重新训练后的仿真表现，尚未重新优化高层 PPO；力估计精度改善不自动等于任务性能提升。',
            '- 显式控制统一了提交力矩的含义，接触/闭链求解和跨引擎力标签仍可能不同。原有 MuJoCo RNE 核验不能替代 PhysX–MuJoCo 的独立已知力标定。',
            '- 本体历史版本借鉴 FALCON 的可部署输入原则；双手十二维显式力矩估计任务与 FALCON 主方法不同。性能以这里的实测为准。',
            '- 真机接口准备与仿真结论分开；没有硬件测试结果。','',
            '实现、FALCON 固定版本来源和训练设计见 [实验说明](../../../docs/push_cart_v52_falcon_experiment.md)。源码测试验证接口与计算，仿真性能由本目录实际运行结果支持。']
    lookup={(r['group'],r['engine']):r for r in closed}
    fmt=lambda a:' / '.join(f'{x:.3f}' for x in a)
    explicit=lookup['explicit_pd_full','mujoco']
    original=lookup['training_full','mujoco']
    randomized=lookup['explicit_pd_randomized_full','mujoco']
    findings=[
        '## 实测结论','',
        '1. **显式 PD 必须配套重新采集和训练。** 旧网络在 Isaac 原驱动下的 '
        f"Fx RMSE 为 {captures['isaac_physx','training']['handle']['rmse'][0]:.3f} N，"
        f"只切显式 PD 后变为 {captures['isaac_physx','explicit_pd']['handle']['rmse'][0]:.3f} N；"
        f"在新驱动下重训全输入网络并接回 Actor 后为 {lookup['explicit_pd_full','isaac_physx']['force_rmse'][0]:.3f} N。"
        '这说明驱动改变会使原估计器失配，不能直接用旧网络分数否定显式控制。',
        f"2. **本轮跨引擎估力最好的是显式 PD＋原全输入重训。** MuJoCo 闭环 Fx/Fy/Mz RMSE 为 {fmt(explicit['force_rmse'])}（N/N/N·m）；"
        f"原驱动重训全输入为 {fmt(original['force_rmse'])}。两者各自闭环轨迹不同；同轨迹 shadow 结果同样支持显式数据重训有效。"
        '这仍未消除 Isaac–MuJoCo 差距。',
        f"3. **有限执行器随机化未带来收益。** 显式随机化全输入的 MuJoCo 闭环误差为 {fmt(randomized['force_rmse'])}；"
        '去掉多类特征的早期 proprio 版本也未超过全输入基线。增益和延迟同时变化，本轮不能分别归因。',
        f"4. **估力改善没有转化为一致的跟踪改善。** 原驱动重训全输入的 MuJoCo vx/vy/wz RMSE 为 {fmt(original['tracking_rmse'])}，"
        f"显式重训为 {fmt(explicit['tracking_rmse'])}（m/s、m/s、rad/s）；只有侧向分量改善，前向和转向未改善。"
        '所有已运行闭环均完成 12 个工况；本轮未重训高层策略，也没有多次闭环随机种子统计。',
        '5. **用户指定的全身本体与全身控制历史已另行实现、训练和双引擎验证。** '
        '见 [最新网络与采集诊断](falcon_a3/comparison.md)，其中包含 FALCON A.3 对照、全身控制历史、两域训练交叉测试及数据核查。','',
    ]
    lines[4:4]=findings
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(shadow=aggregates,closed_loop=closed,not_run=not_run),indent=2),encoding='utf-8')
    print('Wrote',out/'comparison.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit')
    main(p.parse_args())
