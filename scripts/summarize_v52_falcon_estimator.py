"""Report the actual FALCON A.3-input simulation experiment, including failures."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def tracking(report):
    rows=[c for c in report['conditions'] if c['tracking_rmse'] is not None]
    return np.sqrt(np.average([np.square(c['tracking_rmse']) for c in rows],axis=0,
                             weights=[c['handle']['n'] for c in rows])).tolist()


def valid_report(path):
    r=read(path)
    assert len(r['conditions'])==r['total_conditions']==12,path
    assert not r.get('partial',False),path
    return r


def fmt(values):
    return ' / '.join(f'{v:.3f}' for v in values)


def main(args):
    out=args.output
    meta=read(out/'data_explicit_pd/metadata.json')
    audit=read(out/'dataset_audit.json')
    assert audit['same_training_samples'] and audit['measurement_history_exact']
    assert meta['completed_steps']==4096 and meta['num_envs']==64
    models={f:[read(out/f'model_explicit_pd_{f}_{s}/report.json') for s in (11,22,33)]
            for f in ('falcon','falcon_fullbody')}
    selections={f:read(out/f'closed_loop/explicit_pd_{f}/selection.json') for f in models}
    for f,reports in models.items():
        expected=min(zip((11,22,33),reports),key=lambda x:x[1]['scores']['validation']['selection_mse'])[0]
        assert selections[f]['seed']==expected
    new=read(out/'shadow_comparison.json')['results']
    old=read(out/'baseline_shadow.json')['results']
    assert len(new)==12 and len(old)==36
    lines=['# 全身本体历史估力：FALCON 参考训练与 sim2sim 诊断','',
           '按用户要求优先在仿真中验证。新的全身版估力网络使用全身关节/IMU 及实际上下身控制历史，估力输入不含关节力矩和 DLS 先验；另保留 A.3 动作输入作为对照。没有进行真机实验。','',
           '## 输入和训练','',
           '- 输入为最近五帧：关节位置 27、速度 27、IMU 角速度 3、投影重力 3、上一下肢动作 12，共 360 维。观测时刻 t 对应下肢动作 t−1；两引擎同样处理 reset 和固定缩放。',
           '- 按用户后续要求增加全身控制历史版本：在上述每帧 72 维基础上加入 15 维实际上身控制目标，共 5×87=435 维。上身目标来自已经记录的 PD 目标；没有加入力矩、DLS、HIM 或 FK。两版的关节/IMU 观测一直覆盖全身，并非只看下肢。',
           '- 按 [FALCON A.3](https://arxiv.org/html/2505.06776v2#A1.SS3) 的输入及时间顺序实现。为推车任务保留双手 12 维力/力矩输出；原论文该对照方案是三维力，且联合训练下身策略，因此本实验不是其完整复现。',
           '- 新采集显式 PD Isaac 数据，64 环境 × 4096 步；按整个环境划分，MuJoCo 不参与训练、归一化、早停或模型选择。高层其余权重及 HOMIE 冻结。',
           '- 数据审计确认重新采集的原 680 维观测、12 维标签、环境/回合编号及时间步与全输入训练集逐项完全相同；只增加真实下肢动作历史。见 `dataset_audit.json`。',
           f"- 有效样本 {meta['samples']}：训练 {meta['split_samples']['train']}，验证 {meta['split_samples']['validation']}，独立 Isaac 测试 {meta['split_samples']['test']}。三个优化种子 11/22/33 共用采样数据。",'',
           '| 输入 | 训练种子 | 最优 epoch | Isaac 留出测试 Fx/Fy/Mz RMSE (N,N,N·m) |',
           '|---|---|---:|---|']
    for f,reports in models.items():
        for s,r in zip((11,22,33),reports):
            lines.append(f"| {f} | {s} | {r['best_epoch']} | {fmt(r['scores']['test']['handle']['rmse'])} |")
    lines+=['','## 同一评测轨迹的网络对照','',
            '各引擎内部，各模型读取同一批原行为策略轨迹；两引擎间不是逐状态相同。12 个工况为载荷 0/12.5/25 kg × 直行 0.3/0.6 m/s 与左右转弯，每工况 500 步，去除前 2 s。新预测在本节不影响动作。',
            '数值为三个训练种子的均值 ± 样本标准差，仅反映初始化/优化变异。以下重训模型都使用显式 PD 数据。','',
            '| 引擎 | 输入 | Fx RMSE (N) | Fy RMSE (N) | Mz RMSE (N·m) |',
            '|---|---|---:|---:|---:|']
    aggregate=[]
    for engine in ('isaac_physx','mujoco'):
        for feature,label,source in (('full','原全输入，含力矩/先验',old),
                                     ('proprio','早期本体版，无下肢动作历史',old),
                                     ('falcon','A.3：全身观测＋下肢动作历史',new),
                                     ('falcon_fullbody','全身观测＋全身控制历史',new)):
            rows=[r for r in source if r['engine']==engine and r['features']==feature and
                  r['training_drive']=='explicit_pd' and not r['actuator_randomization']]
            assert len(rows)==3,(engine,feature)
            a=np.array([r['scores']['handle']['rmse'] for r in rows])
            cells=[f'{a[:,i].mean():.3f} ± {a[:,i].std(ddof=1):.3f}' for i in range(3)]
            lines.append(f'| {engine} | {label} | '+' | '.join(cells)+' |')
            aggregate.append(dict(engine=engine,features=feature,mean=a.mean(0).tolist(),std=a.std(0,ddof=1).tolist()))
    lines+=['','## 实际接回 Actor 后的闭环验证','',
            f"新网络仅按 Isaac 验证集选择：A.3 输入种子 {selections['falcon']['seed']}，全身控制历史种子 {selections['falcon_fullbody']['seed']}。估力输出实际参与高层动作计算；标签只用于评分。这里没有增加 JᵀF 前馈，也未重训高层 PPO。",'',
            '| 引擎 | 模型 | Fx/Fy/Mz RMSE (N,N,N·m) | vx/vy/wz 跟踪 RMSE (m/s,m/s,rad/s) | 完成工况 | 有效样本 |',
            '|---|---|---|---|---:|---:|']
    closed=[]
    for engine,prefix in (('isaac_physx','physx'),('mujoco','isaac')):
        for name,path in (
            ('旧检查点，显式驱动',out/'captures'/f'{prefix}_explicit_pd.json'),
            ('显式 PD 重训，全输入',out.parent/'closed_loop/explicit_pd_full'/f'{prefix}_explicit_pd.json'),
            ('FALCON A.3 输入',out/'closed_loop/explicit_pd_falcon'/f'{prefix}_explicit_pd.json'),
            ('全身观测＋全身控制历史',out/'closed_loop/explicit_pd_falcon_fullbody'/f'{prefix}_explicit_pd.json')):
            r=valid_report(path)
            t=tracking(r)
            lines.append(f"| {engine} | {name} | {fmt(r['handle']['rmse'])} | {fmt(t)} | {r['completed_conditions']}/12 | {r['handle']['n']} |")
            closed.append(dict(engine=engine,name=name,force_rmse=r['handle']['rmse'],tracking_rmse=t,
                               completed_conditions=r['completed_conditions'],samples=r['handle']['n'],path=str(path)))
    diagnostic=read(out/'domain_diagnostic/comparison.json')
    assert len(diagnostic['results'])==12
    lines+=['','## 区分网络与跨引擎影响：等样本量的两域诊断','',
            '此节独立于主迁移训练。每域各 3200 个训练、800 个验证、800 个测试样本；按完整工况划分。'
            '验证工况为 0 kg/forward06 和 12.5 kg/left，留出测试为 0 kg/right 和 25 kg/forward03，其他八个工况训练。'
            '网络结构与监督目标相同，分别在两域训练，再交叉测试；归一化和模型选择只看各自源域训练/验证数据。'
            '这组诊断使用 MuJoCo 标签，但模型不接入控制、不用于调物理参数，也不替换只在 Isaac 训练的主模型。','',
            '| 输入 | 训练引擎 | 测试引擎 | Fx RMSE (N) | Fy RMSE (N) | Mz RMSE (N·m) |',
            '|---|---|---|---:|---:|---:|']
    domain_aggregates=[]
    for f in ('full','falcon_fullbody'):
        for source in ('isaac_physx','mujoco'):
            rows=[r for r in diagnostic['results'] if r['features']==f and r['training_engine']==source]
            assert len(rows)==3
            for target in ('isaac_physx','mujoco'):
                a=np.array([r['test'][target]['handle']['rmse'] for r in rows])
                cells=[f'{a[:,i].mean():.3f} ± {a[:,i].std(ddof=1):.3f}' for i in range(3)]
                lines.append(f'| {f} | {source} | {target} | '+' | '.join(cells)+' |')
                domain_aggregates.append(dict(features=f,source=source,target=target,mean=a.mean(0).tolist(),std=a.std(0,ddof=1).tolist()))
    lines+=['','两个对角项衡量各自域内的可学习性，交叉项衡量固定网络换引擎后的误差。'
            '同域训练较好而交叉较差支持存在域差异，但不能单独定位为 PD、接触参数、约束求解或标签差异。'
            '这里的样本规模和留出工况与前面大数据实验不同，不应跨表直接比较绝对精度。']
    known=read(out/'known_load_v52_substeps/report.json')
    assert known['result']=='MEASURED' and known['recorded_every_physics_substep']
    known_hands=[h for p in known['phases'].values() for e in p['environment_hand'] for h in e if h]
    collection=read(out/'collection_audit/audit.json')
    assert collection['finite'] and collection['upper_target_max_abs_error']==0
    assert collection['hand_substep_count_values']==[4.0]
    coverage=read(out/'feature_coverage.json')
    coverage_by_field={(r['dataset'],r['field']):r for r in coverage['fields']}
    outside=lambda d,f:coverage_by_field[d,f]['outside_train_q01_q99_pct']
    lines+=['','## 采集与标签检查','',
            '- 主数据的 245,888 对同回合相邻样本历史完全连续，无环境/回合/age 重复；旧输入、标签与对照训练集逐项一致。',
            '- 旧主数据没有逐样本保存实际载荷，无法据此声称载荷覆盖已完整验证。已增加载荷、原始命令、参考、子步数和 PD 目标记录，并用额外 1656 条真实采样检查：有效子步均为 4，上身目标记录与实际 PD 目标差为 0。',
            f"- 按训练集逐坐标 1%–99% 分位区间核查最新观测：IMU 角速度标量超出比例在 Isaac 评测为 {outside('isaac_eval','imu_gyro'):.2f}%，MuJoCo 为 {outside('mujoco_eval','imu_gyro'):.2f}%；"
            f"关节速度分别为 {outside('isaac_eval','joint_velocity'):.2f}% / {outside('mujoco_eval','joint_velocity'):.2f}%，上身控制目标为 {outside('isaac_eval','upper_target_applied'):.2f}% / {outside('mujoco_eval','upper_target_applied'):.2f}%。"
            '输入分布差异有直接记录，但边缘分位统计不能证明因果或多维覆盖；见 [输入覆盖报告](feature_coverage.md)。',
            f"- V5.2 显式 PD、开链运动手臂已知载荷检查完成六阶段：最大力向量 RMS {max(h['force_vector_rms_N'] for h in known_hands):.6f} N，最大力矩向量 RMS {max(h['moment_vector_rms_Nm'] for h in known_hands):.6f} N·m；动态力/力矩测得零时延。它不等于推车闭链的跨引擎标定。",
            '- 核对了 FALCON、Learning Force Control for Legged Manipulation 和 UniFP 的原始论文及官方代码。后两者使用全身控制/观测历史、额外状态监督和已知仿真施加力；我们的闭链反力重建标签、冻结策略采样和短历史与其有实质差别。',
            '- 已知载荷首次诊断因遗漏 V5.2 任务配置，只采末子步而不满足 200 Hz 数量要求，保留为无效记录；修正诊断采样配置后另存有效结果，没有放宽误差阈值。','',
            '源码出处、采集时序和完整边界见 [外部方案与采集核查](../../../../docs/push_cart_v52_force_sources_and_data_audit.md)。']
    lines+=['','## 如何解释','',
            '- A.3 输入替换的是估力通道。冻结高层 Actor 的原 136 维帧与历史编码仍在使用原先验，因此不是整个控制器的“去先验”实验。',
            '- 相比全输入，此版本同时移除了多类特征并加入下肢动作；结果不能单独归因为 DLS。原本计划的仅删除先验消融因用户指定 A.3 输入方案而未执行。',
            '- 先验单独计算存在重力/惯性和驱动瞬态误差；网络的独立监督输出并不受先验精度上限约束。',
            '- [原 DLS 提示单独评分](../static_prior.json)在显式 PD 的 Isaac / MuJoCo 轨迹上，Fx RMSE 分别约 25.15 / 29.80 N。这是输入提示自身的误差；开链已知载荷的约 0.0027 N 是标签重建校核误差，两者都不是训练后估力网络的误差。',
            '- PD 计算一致并不保证接触、闭链约束和力标签跨引擎完全一致。本实验不能独立分离输入分布变化、输出可辨识性和标签差异。',
            '- 估力精度和推车跟踪效果分别报告；不能因为预测更准就宣称任务控制更好。',
            '- FALCON 附录的 JᵀF 是预测后的补偿映射，并非输入侧伪逆。是否接入前馈需要后续独立的仿真消融。','',
            '实现和复现入口见 [说明](../../../../docs/push_cart_v52_falcon_estimator.md)。完整逐手、逐工况指标见 [新模型固定轨迹结果](shadow_comparison.json)、[原模型固定轨迹结果](baseline_shadow.json) 和 `closed_loop/explicit_pd_falcon/`。']
    by_shadow={(r['engine'],r['features']):r for r in aggregate}
    by_domain={(r['features'],r['source'],r['target']):r for r in domain_aggregates}
    by_closed={(r['engine'],r['name']):r for r in closed}
    isaac_a3=by_shadow['isaac_physx','falcon']['mean']
    isaac_fullbody=by_shadow['isaac_physx','falcon_fullbody']['mean']
    mj_a3=by_shadow['mujoco','falcon']['mean']
    mj_fullbody=by_shadow['mujoco','falcon_fullbody']['mean']
    domain=lambda s,t:by_domain['falcon_fullbody',s,t]['mean'][0]
    main_full=by_closed['mujoco','显式 PD 重训，全输入']
    fullbody=by_closed['mujoco','全身观测＋全身控制历史']
    a3=by_closed['mujoco','FALCON A.3 输入']
    conclusions=[
        '## 对三个问题的回答','',
        f"1. **输入设计有影响，不能只保留下肢动作历史。** 同一 Isaac 评测轨迹、三种子平均 Fx RMSE 从 A.3 版的 {isaac_a3[0]:.3f} N 降到全身控制历史版的 {isaac_fullbody[0]:.3f} N；"
        f"MuJoCo 同轨迹从 {mj_a3[0]:.3f} N 降到 {mj_fullbody[0]:.3f} N。"
        '两者都观测全身关节，差别是后者补上了实际执行的上身目标。该结果支持保留全身控制信息；尚不能证明当前网络容量或五帧历史已足够。',
        f"2. **sim2sim 域差异有实验证据，但尚未定位成某一个环境参数错误。** 在独立等样本量诊断中，全身历史网络的 Fx RMSE："
        f"Isaac 训练→Isaac 测试 {domain('isaac_physx','isaac_physx'):.3f} N，→MuJoCo {domain('isaac_physx','mujoco'):.3f} N；"
        f"MuJoCo 训练→MuJoCo 测试 {domain('mujoco','mujoco'):.3f} N，→Isaac {domain('mujoco','isaac_physx'):.3f} N。"
        '两域各自能学到更好的预测，固定网络跨域显著变差。PD 公式统一后，闭链/接触物理、标签重建和训练分布仍需分别核验。',
        '3. **没发现主数据的明显拼接/时序错误，但采集覆盖和标签可比性仍有缺口。** '
        '24.6 万条主样本通过有限值、相邻历史、回合和划分核查；开链已知小载荷验证了标签基础计算。'
        '主数据只有一次采集种子、冻结行为策略和标称物理，原记录缺逐样本实际载荷，不能据此证明完整覆盖。'
        '已补全采集字段并做真实短采集核查；双手推车闭链标签的独立跨引擎标定仍未完成。','',
        f"**当前选择：** 显式 PD 下重训的原全输入网络仍作为本轮仿真基线；其 MuJoCo 闭环 Fx/Fy/Mz RMSE 为 {fmt(main_full['force_rmse'])}，"
        f"全身本体/控制历史版为 {fmt(fullbody['force_rmse'])}，A.3 输入版为 {fmt(a3['force_rmse'])}（N/N/N·m）。"
        '保留全身历史版继续研究，但本轮证据不支持把它直接定为更准确的替代方案。所有这些闭环均完成 12/12 工况。','',
        f"**控制表现单独看：** MuJoCo 中全身历史版的 vx/vy/wz RMSE 为 {fmt(fullbody['tracking_rmse'])}，"
        f"A.3 版为 {fmt(a3['tracking_rmse'])}（m/s、m/s、rad/s）；全身版前向更好，侧向和转向反而较差。"
        '因此不能用估力误差下降推导推车控制全面改善。','',
        '## 下一轮仿真优先级','',
        '1. 先做闭链标签对照：相同机械模型、相同已知载荷和上身目标下，分别记录已施加力、重建反力、约束误差、实际驱动力矩，明确两引擎标签的比较范围。',
        '2. 借鉴 Learning Force Control / UniFP 的力激励与分阶段采样，覆盖零载、正负载荷、不同方向/频率、启停、转向和上身姿态。使用新增的实际载荷/命令/PD 字段按工况统计覆盖，并用独立采集种子和完整工况留出。',
        '3. 在同一批校核后的数据上比较 5/10/30 或 32 帧历史及状态辅助监督；先报告同域和跨域估力，再单独验证或重训高层闭环。不要同时改网络、标签和物理参数后把收益归给某一个因素。','',
    ]
    lines[4:4]=conclusions
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(selected_seeds={f:s['seed'] for f,s in selections.items()},shadow=aggregate,closed_loop=closed,domain_diagnostic=domain_aggregates),indent=2),encoding='utf-8')
    print('Wrote',out/'comparison.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit/falcon_a3')
    main(p.parse_args())
