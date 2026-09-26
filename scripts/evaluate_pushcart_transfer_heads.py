"""Evaluate all trained heads on identical captured trajectories in each engine."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]


def main(args):
    import torch
    from homie_pushcart_v52.transfer_estimator import load_estimator
    from homie_pushcart_v52.fixed_grasp import palm_offsets_from_handle
    from train_pushcart_v52_transfer import score
    torch.set_num_threads(1)
    offsets=torch.tensor(palm_offsets_from_handle(),dtype=torch.float32,device=args.device)
    captures=sorted(args.captures.glob('*_samples.pt'))
    models=sorted(args.models.glob('model_*/estimator.pt'))
    if not captures or not models:
        raise ValueError('Both completed evaluation captures and trained models are required')
    results=[]
    for path in captures:
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['schema']!='v52_transfer_eval_v1':
            raise ValueError('Unexpected capture schema')
        y=data['y'].to(args.device)
        original=data['report']
        if original.get('estimator_override'):
            raise ValueError('The controlled shadow comparison requires original behavior trajectories')
        for model_path in models:
            model,bundle=load_estimator(model_path,args.device)
            input_key = 'falcon_x' if model.features.startswith('falcon') else 'x'
            if input_key not in data:
                raise ValueError(f'{path} lacks actual FALCON action history; collect it again separately')
            x=data[input_key].to(args.device)
            if model.features == 'falcon_fullbody':
                from homie_pushcart_v52.falcon_history import fullbody_history
                x=fullbody_history(data['x'].to(args.device),x)
            result=dict(capture=str(path),engine=original['engine'],eval_drive=original['upper_drive'],
                        model=str(model_path),training_drive=bundle['training']['data_metadata']['upper_drive'],
                        actuator_randomization=bundle['training']['data_metadata']['actuator_randomization'],
                        features=bundle['architecture']['features'],seed=bundle['training']['seed'],
                        scores=score(model,x,y,offsets),conditions=[])
            for i,condition in enumerate(original['conditions']):
                mask=data['condition_id'].to(args.device)==i
                if mask.any():
                    result['conditions'].append(dict(condition=condition['condition'],payload_kg=condition['payload_kg'],
                                                     scores=score(model,x[mask],y[mask],offsets)))
            results.append(result)
            print(original['engine'],original['upper_drive'],model_path.parent.name,result['scores']['handle']['rmse'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(schema='v52_shadow_comparison_v1',results=results,
        note='All heads see identical samples per engine/drive. New predictions do not affect behavior here. No MuJoCo training or selection.'),indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    main(p.parse_args())
