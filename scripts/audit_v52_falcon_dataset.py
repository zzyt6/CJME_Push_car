"""Verify newly collected FALCON features and equivalence to the prior data run."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]
from homie_pushcart_v52.observation import frame_layout


def main():
    torch.set_num_threads(4)
    base=ROOT/'logs/bench/v52_falcon_explicit'
    old=torch.load(base/'data_explicit_pd/dataset.pt',map_location='cpu',weights_only=True)
    new=torch.load(base/'falcon_a3/data_explicit_pd/dataset.pt',map_location='cpu',weights_only=True)
    indices=[i for f in frame_layout() if f['name'] in ('joint_position','joint_velocity','imu_gyro','projected_gravity')
             for i in range(f['start'],f['start']+f['dim'])]
    features=new['falcon_x'].reshape(-1,5,72)
    measurements=new['x'].reshape(-1,5,136)[:,:,indices]
    torch.testing.assert_close(features[:,:,:60],measurements,atol=0,rtol=0)
    result=dict(samples=len(new['x']),falcon_shape=list(new['falcon_x'].shape),
                all_finite=all(bool(torch.isfinite(new[k]).all()) for k in ('x','falcon_x','y')),measurement_history_exact=True,
                lower_action_max_abs=float(features[:,:,60:].abs().max()),previous_run={})
    for key in ('x','y','env_id','episode_id','age'):
        a,b=old[key],new[key]
        r=dict(old_shape=list(a.shape),new_shape=list(b.shape),exact=torch.equal(a,b))
        if a.shape==b.shape:
            r.update(max_abs=float((a-b).abs().max()),rms=float((a-b).float().square().mean().sqrt()))
        result['previous_run'][key]=r
    result['same_training_samples']=all(r['exact'] for r in result['previous_run'].values())
    temporal=dict(adjacent_pairs=0,policy_history_max_abs_error=0.,falcon_history_max_abs_error=0.,
                  duplicate_env_episode_age=0,minimum_age=int(new['age'].min()),environments={})
    for env_id in new['env_id'].unique().tolist():
        m=new['env_id']==env_id
        episode,age=new['episode_id'][m],new['age'][m]
        consecutive=(episode[1:]==episode[:-1])&(age[1:]==age[:-1]+1)
        temporal['adjacent_pairs']+=int(consecutive.sum())
        temporal['duplicate_env_episode_age']+=len(age)-len(torch.unique(torch.stack((episode,age),-1),dim=0))
        temporal['environments'][str(env_id)]=dict(samples=int(m.sum()),episodes=len(episode.unique()),
                                                  first_age=int(age[0]),last_age=int(age[-1]))
        for key,width,metric in (('x',136,'policy_history_max_abs_error'),('falcon_x',72,'falcon_history_max_abs_error')):
            a=new[key][m].reshape(-1,5,width)
            if consecutive.any():
                err=float((a[1:,:4][consecutive]-a[:-1,1:][consecutive]).abs().max())
                temporal[metric]=max(temporal[metric],err)
    assert temporal['minimum_age']>=50 and temporal['duplicate_env_episode_age']==0
    assert temporal['policy_history_max_abs_error']==temporal['falcon_history_max_abs_error']==0
    result['temporal_audit']=temporal
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE
    labels=new['y']*torch.tensor(HAND_WRENCH_SCALE).repeat(2)
    result['label_summary']=dict(all_finite=bool(torch.isfinite(labels).all()),mean=labels.mean(0).tolist(),
                                 std=labels.std(0).tolist(),min=labels.min(0).values.tolist(),max=labels.max(0).values.tolist(),
                                 q01=torch.quantile(labels,.01,dim=0).tolist(),q99=torch.quantile(labels,.99,dim=0).tolist())
    result['recording_limitations']=['Per-sample payload and raw task command were not persisted; smoothed references ARE present in x.',
                                    'Labels are reconstructed reactions, not independent force ground truth.',
                                    'One data collection seed and a nominal scene do not establish broad load/pose coverage.']
    (base/'falcon_a3/dataset_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    printed={k:v for k,v in result.items() if k not in ('label_summary','temporal_audit')}
    printed['temporal_audit']={k:v for k,v in temporal.items() if k!='environments'}
    print(json.dumps(printed,indent=2))


if __name__=='__main__':
    main()
