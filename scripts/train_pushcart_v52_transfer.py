"""Collect Isaac data and train force heads without changing the behavior policy.

Data splits are by entire parallel environment (all its episodes), not adjacent
frames. MuJoCo data is never used by the optimizer or early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL')]
DEFAULT_CKPT = ROOT / 'logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt'


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def collect(args):
    import numpy as np
    import torch
    from homie_pushcart_v52.control_contract import nominal_configs
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.hl_policy import HighLevelPolicy
    from homie_isaaclab.math_utils import class_to_dict
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg, task = nominal_configs(args.checkpoint, args.upper_drive, 20.)
    cfg.env.num_envs = args.num_envs
    if args.actuator_randomization:
        cfg.domain_rand.randomize_kp = True
        cfg.domain_rand.randomize_kd = True
        cfg.domain_rand.kp_range = [.9, 1.1]
        cfg.domain_rand.kd_range = [.9, 1.1]
        cfg.domain_rand.delay = True
    # Natural command schedules and random payloads remain enabled. Other
    # disturbances/noise are explicitly off for the matched drive experiment.
    low = Path(cfg.pushcart_saved_checkpoint)
    if not low.is_absolute():
        low = ROOT / low
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=low)
    env.fixed_caster_angle = 0.
    hl = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema='v52_force_dataset_v1', engine='isaac_physx', upper_drive=args.upper_drive,
                    behavior_checkpoint=str(args.checkpoint), homie_checkpoint=str(env.policy.checkpoint),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    seed=args.seed, num_envs=args.num_envs, requested_steps=args.steps,
                    actuator_randomization=args.actuator_randomization, physics=env.physics_report,
                    homie=class_to_dict(cfg), task=class_to_dict(task), history=5,
                    split='env_id modulo 8: 0 validation, 1 Isaac test, 2..7 training',
                    note='Frozen original behavior policy. Estimator data never feeds actions during collection.')
    write_json(args.output/'metadata.json', metadata)
    metadata['falcon_input'] = '5x[q27,dq27,gyro3,gravity3,previous clipped HOMIE lower action12]; no torque/prior'
    metadata['audit_columns'] = dict(cart_mass_kg='actual total cart mass including wheels, kg',
                                    command_task='raw task vx,vy,wz; m/s,m/s,rad/s',
                                    command_reference='smoothed task reference; same units',
                                    hand_substep_count='valid physics samples in the preceding control interval',
                                    pd_target='last submitted joint position target, 27 joints in HOMIE order, rad')
    rows = {k: [] for k in ('x', 'falcon_x', 'y', 'env_id', 'episode_id', 'age',
                           'cart_mass_kg','command_task','command_reference','hand_substep_count','pd_target')}
    episode = torch.zeros(env.num_envs, dtype=torch.int32, device=args.device)
    ids = torch.arange(env.num_envs, device=args.device)
    resets, start = 0, time.monotonic()
    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(args.steps):
            obs, _, _, done, _ = env.step(hl(obs))
            episode += done.to(episode.dtype)
            resets += int(done.sum())
            # Post-step features and the same interval's labels, matching the
            # production evaluation. Exclude reset/warmup and invalid labels.
            valid = (~done.bool()) & (env.episode_length_buf >= 50) & (env.interaction_target[:, 12] > .5)
            x, y = obs[:, -680:], env.interaction_target[:, :12]
            valid &= torch.isfinite(x).all(-1) & torch.isfinite(y).all(-1)
            for k, v in dict(x=x, falcon_x=env.falcon_obs, y=y, env_id=ids, episode_id=episode, age=env.episode_length_buf,
                             cart_mass_kg=env.cart_mass_total,command_task=env.cmd_task,
                             command_reference=env.cmd_ref,hand_substep_count=env.hand_wrench_count,
                             pd_target=env.joint_pos_target).items():
                rows[k].append(v[valid].detach().cpu().clone())
            if (step+1) % 128 == 0:
                progress = dict(steps=step+1, resets=resets, samples=sum(len(x) for x in rows['x']),
                                elapsed_s=time.monotonic()-start)
                write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    data = {k: torch.cat(v) for k,v in rows.items()}
    torch.save(data, args.output/'dataset.pt')
    metadata.update(completed_steps=args.steps, samples=len(data['x']), resets=resets,
                    elapsed_s=time.monotonic()-start,
                    split_samples={name:int(mask.sum()) for name,mask in split_masks(data['env_id']).items()})
    write_json(args.output/'metadata.json', metadata)
    print('COLLECTION COMPLETE', json.dumps({k:metadata[k] for k in ('samples','resets','elapsed_s','split_samples')}), flush=True)


def split_masks(ids):
    return {'train':ids % 8 >= 2, 'validation':ids % 8 == 0, 'test':ids % 8 == 1}


def score(model, x, y, offsets):
    import torch
    from eval_pushcart_v52_sim2sim import metrics
    from homie_pushcart_v52.transfer_estimator import resultants
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE
    with torch.inference_mode():
        pred = torch.cat([model(a) for a in x.split(4096)])
        scale = torch.tensor(HAND_WRENCH_SCALE, device=y.device).repeat(2)
        return dict(hands=metrics((pred*scale).cpu().numpy(),(y*scale).cpu().numpy()),
                    handle=metrics(resultants(pred, offsets).cpu().numpy(),resultants(y, offsets).cpu().numpy()),
                    selection_mse=float((pred-y).square().mean()))


def train(args):
    import copy
    import torch
    from homie_pushcart_v52.transfer_estimator import TransferEstimator, PhysicalWrenchExport
    from homie_pushcart_v52.fixed_grasp import palm_offsets_from_handle
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    meta = json.loads((args.dataset/'metadata.json').read_text(encoding='utf-8'))
    data = torch.load(args.dataset/'dataset.pt', map_location='cpu', weights_only=True)
    masks = split_masks(data['env_id'])
    if not all(int(m.sum()) for m in masks.values()):
        raise ValueError('Every split needs samples; collect at least eight environments')
    input_key = 'falcon_x' if args.features.startswith('falcon') else 'x'
    if input_key not in data:
        raise ValueError('FALCON training requires newly collected actual lower-action histories')
    if args.features == 'falcon_fullbody':
        from homie_pushcart_v52.falcon_history import fullbody_history
        data['falcon_x'] = fullbody_history(data['x'],data['falcon_x'])
    splits = {k:(data[input_key][m].to(args.device),data['y'][m].to(args.device)) for k,m in masks.items()}
    del data
    model = TransferEstimator(args.features).to(args.device)
    tx, ty = splits['train']
    model.fit_normalization(tx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    offsets = torch.tensor(palm_offsets_from_handle(), dtype=torch.float32, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    best_loss, best, best_epoch, stale = float('inf'), None, -1, 0
    logs, start = [], time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx), device=args.device)
        total, count = 0., 0
        for idx in order.split(args.batch_size):
            pred = model(tx[idx])
            loss = torch.nn.functional.huber_loss(pred, ty[idx], delta=1.)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(idx)
            count += len(idx)
        model.eval()
        val = score(model, *splits['validation'], offsets)
        row = dict(epoch=epoch+1, loss=total/count, validation=val, elapsed_s=time.monotonic()-start)
        logs.append(row)
        if val['selection_mse'] < best_loss:
            best_loss, best, best_epoch, stale = val['selection_mse'], copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            stale += 1
        print(f"epoch={epoch+1} loss={total/count:.6f} val_rmse={val['handle']['rmse']} best={best_epoch}", flush=True)
        write_json(args.output/'training.json', logs)
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    reports={k:score(model, *v, offsets) for k,v in splits.items()}
    bundle=dict(schema='v52_transfer_estimator_v1', architecture=dict(features=args.features,hidden=model.hidden,history=5),
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                training=dict(dataset=str(args.dataset),data_metadata=meta,seed=args.seed,epochs=len(logs),best_epoch=best_epoch,
                              batch_size=args.batch_size,lr=args.lr,selection='Isaac validation normalized per-hand MSE only'),
                scores=reports)
    torch.save(bundle,args.output/'estimator.pt')
    export=torch.jit.script(PhysicalWrenchExport(model))
    export.save(str(args.output/'estimator_physical.jit'))
    write_json(args.output/'report.json',dict(features=args.features,upper_drive=meta['upper_drive'],seed=args.seed,
                                            best_epoch=best_epoch,epochs=len(logs),scores=reports,
                                            elapsed_s=time.monotonic()-start))
    print('TRAINING COMPLETE',json.dumps(reports['test']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect')
    c.add_argument('--checkpoint',type=Path,default=DEFAULT_CKPT)
    c.add_argument('--upper_drive',choices=('training','explicit_pd'),required=True)
    c.add_argument('--num_envs',type=int,default=64)
    c.add_argument('--steps',type=int,default=2048)
    c.add_argument('--actuator_randomization',action='store_true')
    c.add_argument('--seed',type=int,default=101)
    c.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train')
    t.add_argument('--dataset',type=Path,required=True)
    t.add_argument('--features',choices=('full','proprio','falcon','falcon_fullbody'),default='full')
    t.add_argument('--seed',type=int,default=11)
    t.add_argument('--epochs',type=int,default=100)
    t.add_argument('--patience',type=int,default=15)
    t.add_argument('--batch_size',type=int,default=2048)
    t.add_argument('--lr',type=float,default=1.e-3)
    t.add_argument('--device',default='cuda:0')
    t.add_argument('--output',type=Path,required=True)
    preliminary,_=p.parse_known_args()
    if preliminary.command=='collect':
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(c)
        args=p.parse_args()
        if args.num_envs < 8:
            p.error('Use at least 8 environments for disjoint splits')
        launcher=AppLauncher(args)
        try:
            collect(args)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            launcher.app.close()
    else:
        train(p.parse_args())


if __name__=='__main__':
    main()
