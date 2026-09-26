"""Describe marginal training support without fitting models or physics."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def current_features(data):
    return torch.cat((data['x'][:,-136:],data['falcon_x'][:,-12:]),dim=-1).float()


def main(args):
    from homie_pushcart_v52.observation import frame_layout
    from homie_pushcart_v52.fixed_grasp import palm_offsets_from_handle
    from homie_pushcart_v52.transfer_estimator import resultants
    torch.set_num_threads(4)
    data=torch.load(args.output/'data_explicit_pd/dataset.pt',map_location='cpu',weights_only=True)
    train_mask=data['env_id']%8>=2
    test_mask=data['env_id']%8==1
    x=current_features(data)
    train=x[train_mask]
    quantiles=torch.quantile(train,torch.tensor([.01,.99]),dim=0)
    mean,std=train.mean(0),train.std(0)
    lower,upper=train.min(0).values,train.max(0).values
    fields=frame_layout()+[dict(name='previous_lower_action',start=136,dim=12)]
    offsets=torch.tensor(palm_offsets_from_handle(),dtype=torch.float32)
    datasets={'training':(train,data['y'][train_mask]),'isaac_heldout':(x[test_mask],data['y'][test_mask])}
    for engine,prefix in (('isaac_eval','physx'),('mujoco_eval','isaac')):
        capture=torch.load(args.output/f'captures/{prefix}_explicit_pd_samples.pt',map_location='cpu',weights_only=False)
        datasets[engine]=(current_features(capture),capture['y'])
    rows=[]
    labels={}
    for name,(values,y) in datasets.items():
        assert torch.isfinite(values).all() and torch.isfinite(y).all()
        for field in fields:
            s=slice(field['start'],field['start']+field['dim'])
            v=values[:,s]
            variable=std[s]>1.e-6
            shifted=(v.mean(0)-mean[s]).abs()/std[s].clamp_min(1.e-6)
            row=dict(dataset=name,field=field['name'],samples=len(v),dimensions=field['dim'],
                     variable_dimensions=int(variable.sum()),
                     outside_train_q01_q99_pct=100*float(((v<quantiles[0,s])|(v>quantiles[1,s])).float().mean()),
                     outside_train_minmax_pct=100*float(((v<lower[s]-1.e-6)|(v>upper[s]+1.e-6)).float().mean()),
                     max_mean_shift_in_train_std=float(shifted[variable].max()) if variable.any() else None)
            rows.append(row)
        handle=resultants(y,offsets)
        labels[name]=dict(samples=len(handle),mean=handle.mean(0).tolist(),std=handle.std(0).tolist(),
                          q01=torch.quantile(handle,.01,dim=0).tolist(),q99=torch.quantile(handle,.99,dim=0).tolist())
    report=dict(schema='v52_marginal_coverage_v1',training_samples=len(train),latest_frame_only=True,
                definition='Percentage of scalar entries outside each coordinate training band; not percentage of independent trajectories.',
                threshold_source='Isaac training environments only; no changes to normalization, weights, model selection or physical parameters.',
                caveat='Marginal statistics do not establish multivariate coverage or causal attribution. Correlated frames are not independent trials.',
                fields=rows,handle_labels=labels)
    (args.output/'feature_coverage.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 训练覆盖与跨域输入分布','',
           '仅使用 Isaac 训练环境确定每个坐标的 1%–99% 分位区间和最小/最大值；检查最新一帧，不重复统计五帧重叠。',
           '百分比的分母是样本数×该字段维数，表示标量超出比例；不代表独立轨迹比例或因果贡献。训练集自身约有 2% 超出分位区间。',
           '未据此调整网络、归一化、模型选择或物理参数。完整字段、均值移动和标签分布见 [JSON](feature_coverage.json)。','',
           '| 字段 | Isaac 留出：超出分位区间 % | Isaac 评测 % | MuJoCo 评测 % | MuJoCo 超出训练最小/最大 % |',
           '|---|---:|---:|---:|---:|']
    lookup={(r['dataset'],r['field']):r for r in rows}
    for field in fields:
        name=field['name']
        values=[lookup[d,name]['outside_train_q01_q99_pct'] for d in ('isaac_heldout','isaac_eval','mujoco_eval')]
        values.append(lookup['mujoco_eval',name]['outside_train_minmax_pct'])
        lines.append(f'| {name} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','这是单坐标分布描述：即使每个坐标都在训练范围内，组合状态仍可能超出训练分布；反之，少量超出也不必然导致预测错误。',
            '评测的载荷/命令网格与自然命令采集不同，因此需同时比较 Isaac 留出和独立 Isaac 评测，不能把所有变化都归给引擎。']
    (args.output/'feature_coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit/falcon_a3')
    main(parser.parse_args())
