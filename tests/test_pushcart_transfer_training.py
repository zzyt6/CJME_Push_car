"""Deployment and estimator contract checks, without launching Isaac Kit."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]
from homie_pushcart_v52.pd_command import PDCommand, HOMIE_TO_UNITREE_G1
from homie_pushcart_v52.transfer_estimator import TransferEstimator, PhysicalWrenchExport, feature_indices
from homie_pushcart_v52.observation import frame_layout
sys.path.insert(0,str(ROOT/'scripts'))
from train_pushcart_v52_transfer import split_masks


def packet():
    return SimpleNamespace(motor_cmd=[SimpleNamespace(mode=0,q=0.,dq=0.,kp=0.,kd=0.,tau=0.) for _ in range(35)])


def test_sdk_hybrid_reconstruction_matches_explicit_effort_and_mapping():
    rng=np.random.default_rng(71)
    q,dq,target,velocity,ff=rng.normal(size=(5,27))
    cmd=PDCommand(target,velocity,np.full(27,30.),np.full(27,2.),ff,np.full(27,20.),1.)
    low=packet()
    cmd.fill_unitree(low,now_s=1.01)
    # Reconstruct as FALCON's actual MuJoCo/SDK bridge does, in robot motor order.
    reconstructed=[]
    for j,i in enumerate(HOMIE_TO_UNITREE_G1):
        m=low.motor_cmd[i]
        reconstructed.append(np.clip(m.tau+m.kp*(m.q-q[j])+m.kd*(m.dq-dq[j]),-20,20))
    np.testing.assert_allclose(reconstructed,cmd.effort(q,dq),atol=1e-12)
    assert all(low.motor_cmd[i].mode==0 for i in (13,14,29,30,31,32,33,34))
    # Confirm the mapping against the hardware source, not just a second constant.
    source=(ROOT/'HomieDeploy/unitree_sdk2/g1_control.cpp').read_text(encoding='utf-8')
    assert 'LeftShoulderPitch = 15' in source and 'RightShoulderPitch = 22' in source
    assert HOMIE_TO_UNITREE_G1[13]==15 and HOMIE_TO_UNITREE_G1[20]==22
    cmd.fill_unitree(low,now_s=1.01,execution='host_effort',q=q,dq=dq)
    np.testing.assert_allclose([low.motor_cmd[i].tau for i in HOMIE_TO_UNITREE_G1],cmd.effort(q,dq))
    assert all(low.motor_cmd[i].kp==low.motor_cmd[i].kd==0 for i in HOMIE_TO_UNITREE_G1)


def test_invalid_or_stale_command_rejected_before_writing():
    n=27
    with pytest.raises(ValueError):
        PDCommand(np.full(n,np.nan),np.zeros(n),np.ones(n),np.ones(n),np.zeros(n),np.ones(n),1.)
    cmd=PDCommand(np.zeros(n),np.zeros(n),np.ones(n),np.ones(n),np.zeros(n),np.ones(n),1.)
    low=packet()
    with pytest.raises(ValueError):
        cmd.fill_unitree(low,now_s=2.)
    assert all(m.mode==0 for m in low.motor_cmd)
    with pytest.raises(ValueError):
        cmd.fill_unitree(low,now_s=1.01,motor_indices=(0,)*27)
    assert all(m.mode==0 for m in low.motor_cmd)


def test_proprio_head_ignores_torque_prior_and_privileged_like_derived_fields():
    torch.manual_seed(9)
    model=TransferEstimator('proprio').eval()
    x=torch.randn(3,680)
    changed=x.clone().reshape(3,5,136)
    ignored=('upper_torque','static_hand_wrench','him_velocity','handle_position','cart_twist_estimate')
    for f in frame_layout():
        if f['name'] in ignored:
            changed[:,:,f['start']:f['start']+f['dim']]+=1000.
    torch.testing.assert_close(model(x),model(changed.reshape(3,680)),atol=0,rtol=0)
    jit=torch.jit.script(PhysicalWrenchExport(model))
    torch.testing.assert_close(jit(x),PhysicalWrenchExport(model)(x),atol=0,rtol=0)


def test_data_split_is_by_environment_and_normalization_is_train_only():
    ids=torch.arange(16).repeat_interleave(100)
    masks=split_masks(ids)
    assert torch.stack(list(masks.values())).sum(0).eq(1).all()
    groups={k:set(ids[m].tolist()) for k,m in masks.items()}
    assert not groups['train'] & groups['validation'] and not groups['train'] & groups['test']
    model=TransferEstimator('full')
    x=torch.ones(len(ids),680)
    x[~masks['train']]=1000.
    model.fit_normalization(x[masks['train']])
    torch.testing.assert_close(model.mean,torch.ones(680))


def test_falcon_history_has_causal_lower_actions_and_no_torque_or_prior():
    from homie_pushcart_v52.falcon_history import FalconHistory
    history=FalconHistory(2)
    frames=[]
    for t in range(7):
        frame=torch.full((2,136),float(t))
        for f in frame_layout():
            if f['name'] not in ('joint_position','joint_velocity','imu_gyro','projected_gravity'):
                frame[:,f['start']:f['start']+f['dim']]=9999.
        frames.append(history.push(frame,torch.full((2,12),float(t-1)),torch.tensor([t==0,t in (0,6)])))
    out=frames[-1].reshape(2,5,72)
    torch.testing.assert_close(out[0,:,:60],torch.arange(2.,7.)[:,None].expand(5,60))
    torch.testing.assert_close(out[0,:,60:],torch.arange(1.,6.)[:,None].expand(5,12))
    assert out[1,:4].eq(0).all() and out[1,-1,60:].eq(0).all()
    assert frames[0].eq(0).all()  # Saved snapshots were not overwritten.
    model=TransferEstimator('falcon').eval()
    assert model.input_width==360 and len(model.indices)==360
    with pytest.raises(ValueError,match='360D'):
        model(torch.zeros(2,680))
    jit=torch.jit.script(PhysicalWrenchExport(model))
    torch.testing.assert_close(jit(frames[-1]),PhysicalWrenchExport(model)(frames[-1]))


def test_falcon_head_reads_separate_history_and_enters_actor_actions(tmp_path):
    from homie_pushcart_v52.hl_policy import HighLevelPolicy
    from homie_pushcart_v52.transfer_estimator import install_estimator
    checkpoint=ROOT/'logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt'
    hl=HighLevelPolicy(checkpoint,2720,182,18)
    model=TransferEstimator('falcon')
    path=tmp_path/'estimator.pt'
    torch.save(dict(schema='v52_transfer_estimator_v1',architecture=dict(features='falcon'),
                    state_dict=model.state_dict()),path)
    install_estimator(hl.actor,path)
    obs=torch.randn(1,2720)
    a=torch.zeros(1,360)
    b=torch.ones(1,360)
    from tensordict import TensorDict
    first=hl.actor.predict_hand_wrench(TensorDict(dict(policy=obs,falcon=a),batch_size=[1]))
    second=hl.actor.predict_hand_wrench(TensorDict(dict(policy=obs*100,falcon=a),batch_size=[1]))
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert not torch.allclose(hl(obs,estimator_obs=a),hl(obs,estimator_obs=b))
    with pytest.raises(KeyError):
        hl(obs)
    with pytest.raises(ValueError,match='separate history'):
        hl.actor.as_jit()


def test_fullbody_history_uses_upper_commands_but_never_torque_or_prior():
    from homie_pushcart_v52.falcon_history import fullbody_history
    original=torch.randn(2,680)
    lower=torch.randn(2,360)
    merged=fullbody_history(original,lower)
    assert merged.shape==(2,435)
    torch.testing.assert_close(merged.reshape(2,5,87)[:,:,:72],lower.reshape(2,5,72))
    upper=next(f for f in frame_layout() if f['name']=='upper_target_applied')
    torch.testing.assert_close(merged.reshape(2,5,87)[:,:,72:],
                              original.reshape(2,5,136)[:,:,upper['start']:upper['start']+15])
    changed=original.clone().reshape(2,5,136)
    for f in frame_layout():
        if f['name'] != 'upper_target_applied':
            changed[:,:,f['start']:f['start']+f['dim']]+=1000.
    torch.testing.assert_close(merged,fullbody_history(changed.reshape(2,680),lower),atol=0,rtol=0)
    model=TransferEstimator('falcon_fullbody').eval()
    exported=torch.jit.script(PhysicalWrenchExport(model))
    torch.testing.assert_close(exported(merged),PhysicalWrenchExport(model)(merged))


def test_deployment_proprio_history_matches_production_observation_builder():
    from homie_pushcart_v52.deploy_proprio import ProprioHistory
    from pushcart_bench.policies_v5x import V5xRunner
    from pushcart_bench.sim2sim_scene import model_report
    checkpoint=ROOT/'logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt'
    runner=V5xRunner('homie_pushcart_v52',checkpoint,upper_drive='explicit_pd')
    runner.reset(caster_angle=0.)
    history=ProprioHistory(model_report(runner.env)['control'])
    indices=feature_indices('proprio').numpy()
    for step in range(8):
        if step:
            runner.step([.3,0.,.1])
        e=runner.env
        actual=history.push(e.dof_pos,e.dof_vel,e.base_ang_vel,e.projected_gravity,runner.cmd_homie,
                            runner.hold_target_q+runner.upper_delta_q,runner.hl_actions)
        np.testing.assert_allclose(actual,runner.hl_obs[-680:][indices],atol=2e-6,rtol=0.)
        f=runner.falcon_obs.reshape(5,72)
        measured=torch.cat((torch.from_numpy(runner.hl_obs[-136:-82]),
                            torch.from_numpy(runner.hl_obs[-67:-61])))
        torch.testing.assert_close(f[-1,:60],measured)
        np.testing.assert_allclose(f[-1,60:].numpy(),e.actions[:12],atol=2e-6,rtol=0.)
        if step:
            upper=next(field for field in frame_layout() if field['name']=='upper_target_applied')
            logged=runner.hl_obs[-136:][upper['start']:upper['start']+15]
            actual_target=(e.joint_pos_target[12:]-e.default_dof_pos[12:])*e.obs_scales['dof_pos']
            np.testing.assert_allclose(logged,actual_target,atol=2e-6,rtol=0.)
