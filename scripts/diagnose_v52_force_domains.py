"""Separate diagnostic: matched-size within/cross-engine estimator fitting.

These models never enter the transfer experiment or control. MuJoCo labels
are used ONLY here to test whether the same architecture can learn each
domain locally. Splits hold out complete command/payload episodes in both
engines; validation/normalization use only the source domain.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def main(args):
    import torch
    from homie_pushcart_v52.transfer_estimator import TransferEstimator
    from homie_pushcart_v52.falcon_history import fullbody_history
    from homie_pushcart_v52.fixed_grasp import palm_offsets_from_handle
    from train_pushcart_v52_transfer import score
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    offsets=torch.tensor(palm_offsets_from_handle(),dtype=torch.float32,device=args.device)
    # Whole episodes: validation=(0kg forward06,12.5kg left),
    # test=(0kg right,25kg forward03); other eight are training.
    val_ids,test_ids=(1,6),(3,8)
    domains={}
    for engine,prefix in (('isaac_physx','physx'),('mujoco','isaac')):
        path=args.captures/f'{prefix}_explicit_pd_samples.pt'
        d=torch.load(path,map_location='cpu',weights_only=False)
        assert not d['report'].get('estimator_override') and len(d['report']['conditions'])==12
        assert all(c['failure'] is None and c['steps']==500 for c in d['report']['conditions'])
        c=d['condition_id'].to(args.device)
        validation=(c==val_ids[0])|(c==val_ids[1])
        test=(c==test_ids[0])|(c==test_ids[1])
        masks=dict(train=~(validation|test),validation=validation,test=test)
        domains[engine]=dict(full=d['x'].to(args.device),
                             falcon_fullbody=fullbody_history(d['x'],d['falcon_x']).to(args.device),
                             y=d['y'].to(args.device),masks=masks,report=d['report'],path=str(path))
    for split in ('train','validation','test'):
        assert int(domains['isaac_physx']['masks'][split].sum())==int(domains['mujoco']['masks'][split].sum())
    results=[]
    for features in ('full','falcon_fullbody'):
        for source,d in domains.items():
            for seed in (11,22,33):
                torch.manual_seed(seed)
                model=TransferEstimator(features).to(args.device)
                masks=d['masks']
                x,y=d[features],d['y']
                tx,ty=x[masks['train']],y[masks['train']]
                vx,vy=x[masks['validation']],y[masks['validation']]
                model.fit_normalization(tx)
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
                best,best_mse,best_epoch,stale=None,float('inf'),-1,0
                for epoch in range(200):
                    model.train()
                    for idx in torch.randperm(len(tx),device=args.device).split(512):
                        loss=torch.nn.functional.huber_loss(model(tx[idx]),ty[idx],delta=1.)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                    model.eval()
                    with torch.inference_mode():
                        mse=float((model(vx)-vy).square().mean())
                    if mse<best_mse:
                        best,best_mse,best_epoch,stale=copy.deepcopy(model.state_dict()),mse,epoch+1,0
                    else:
                        stale+=1
                    if stale>=30:
                        break
                model.load_state_dict(best)
                row=dict(features=features,training_engine=source,seed=seed,best_epoch=best_epoch,
                         trained_epochs=epoch+1,validation_mse=best_mse,
                         source_train=score(model,tx,ty,offsets),test={})
                for target,other in domains.items():
                    m=other['masks']['test']
                    row['test'][target]=score(model,other[features][m],other['y'][m],offsets)
                results.append(row)
                torch.save(dict(schema='v52_transfer_estimator_v1',architecture=dict(features=features),
                                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                diagnostic_only=True,scores=row),args.output/f'{features}_{source}_{seed}.pt')
                print(features,source,seed,{k:v['handle']['rmse'] for k,v in row['test'].items()},flush=True)
    report=dict(schema='v52_domain_diagnostic_v1',results=results,
                split=dict(validation=list(val_ids),test=list(test_ids),train=[i for i in range(12) if i not in (*val_ids,*test_ids)]),
                samples_per_engine={k:int(m.sum()) for k,m in domains['isaac_physx']['masks'].items()},
                conditions=domains['isaac_physx']['report']['conditions'],
                scope='Diagnostic only; no model is installed, promoted, or used to tune physics. Main transfer heads still train on Isaac only.',
                limitation='Tests held-out condition combinations in these fixed nominal trajectories; does not calibrate cross-engine labels or prove unique causal attribution.')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit/falcon_a3/captures')
    p.add_argument('--output',type=Path,default=ROOT/'logs/bench/v52_falcon_explicit/falcon_a3/domain_diagnostic')
    p.add_argument('--device',default='cuda:0')
    main(p.parse_args())
