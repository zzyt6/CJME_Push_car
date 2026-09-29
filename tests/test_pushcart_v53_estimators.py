"""V5.3 deployability, gradient isolation, supervision, resume and export."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))
from homie_pushcart_v53.hl_policy import import_rsl_rl5
import_rsl_rl5()
from rsl_rl.runners import OnPolicyRunner
from homie_pushcart_v53.observation import FRAME_DIM, CART_INPUT_FRAME_DIM, frame_layout
from homie_pushcart_v53.deploy_observation import ObservableHistory, visual_heading
from homie_pushcart_v53.heading_sensor import SimulatedVisualHeading
from homie_pushcart_v53.task_config import PushCartTaskCfg


def observations(n=8):
    x = torch.randn(n, FRAME_DIM*20)
    speed = .1*x[:, -FRAME_DIM:-FRAME_DIM+3]
    hands = .1*x[:, -12:]
    return TensorDict(dict(policy=x, critic=torch.randn(n, 202),
        interaction_target=torch.cat((hands, torch.ones(n, 1)), -1),
        cart_velocity_target=torch.cat((speed, torch.ones(n, 1)), -1)), batch_size=[n])


class SyntheticEnv:
    num_envs, num_actions, device, cfg = 8, 18, 'cpu', {}
    def __init__(self): self.obs = observations()
    def get_observations(self): return self.obs
    def step(self, actions):
        self.obs = observations()
        return self.obs, actions[:, 0]*.2, torch.zeros(8, dtype=torch.bool), {}


def runner():
    cfg = dict(num_steps_per_env=4, save_interval=100, obs_groups={'actor':['policy'], 'critic':['critic']},
        actor=dict(class_name='homie_pushcart_v53.interaction_policy:InteractionActor',
                   hidden_dims=[32], encoder_hidden_dims=[24], estimator_hidden_dims=[32],
                   cart_velocity_hidden_dims=[24], obs_normalization=False,
                   distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.4}),
        critic=dict(class_name='rsl_rl.models:MLPModel', hidden_dims=[32], obs_normalization=False),
        algorithm=dict(class_name='homie_pushcart_v53.interaction_ppo:InteractionPPO',
                       estimator_epochs=2, estimator_mini_batches=3, num_learning_epochs=1,
                       num_mini_batches=2, schedule='fixed'))
    with contextlib.redirect_stdout(io.StringIO()):
        result = OnPolicyRunner(SyntheticEnv(), cfg, device='cpu')
    return result


class EstimatorTests(unittest.TestCase):
    def test_targets_and_critic_never_change_predictions(self):
        a = runner().alg.actor.eval()
        obs = observations()
        with torch.no_grad():
            action, speed, force = a(obs), a.predict_cart_velocity(obs), a.predict_hand_wrench(obs)
            obs['cart_velocity_target'][:] = 999.
            obs['interaction_target'][:] = -999.
            obs['critic'][:] = 1000.
            torch.testing.assert_close(action, a(obs), rtol=0, atol=0)
            torch.testing.assert_close(speed, a.predict_cart_velocity(obs), rtol=0, atol=0)
            torch.testing.assert_close(force, a.predict_hand_wrench(obs), rtol=0, atol=0)

    def test_gradient_and_optimizer_isolation(self):
        alg = runner().alg
        a, obs = alg.actor, observations()
        a(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in a.estimator_parameters()))
        a.zero_grad(set_to_none=True)
        a.predict_hand_wrench(obs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in a.cart_velocity_parameters()))
        self.assertTrue(any(p.grad is not None for p in a.force_parameters()))
        groups = [{id(p) for g in opt.param_groups for p in g['params']} for opt in
                  (alg.optimizer, alg.estimator_optimizer, alg.cart_velocity_optimizer)]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_training_updates_both_heads_and_restores_optimizers(self):
        r = runner(); alg = r.alg
        old = [p.detach().clone() for p in alg.actor.estimator_parameters()]
        with torch.inference_mode():
            obs = r.env.get_observations()
            for _ in range(4):
                act = alg.act(obs)
                obs, reward, done, extra = r.env.step(act)
                alg.process_env_step(obs, reward, done, extra)
            alg.compute_returns(obs)
        metrics = alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertIn('cart_velocity_mae_vx', metrics)
        self.assertIn('hand_mae_L_fx', metrics)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(old, alg.actor.estimator_parameters())))
        saved = alg.save(); other = runner().alg
        other.load(saved, None, True)
        for key in ('estimator_optimizer_state_dict', 'cart_velocity_optimizer_state_dict'):
            self.assertTrue(saved[key]['state'])
            self.assertEqual(len(saved[key]['state']), len(other.save()[key]['state']))
        with torch.no_grad():
            alg.actor.eval(); other.actor.eval()
            torch.testing.assert_close(alg.actor.predict_cart_velocity(obs), other.actor.predict_cart_velocity(obs))
            torch.testing.assert_close(alg.actor.predict_hand_wrench(obs), other.actor.predict_hand_wrench(obs))

    def test_export_contains_both_heads_and_matches_eager(self):
        a = runner().alg.actor.eval(); obs = observations()
        export = torch.jit.script(a.as_jit().eval())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'actor.pt'; export.save(str(path)); export = torch.jit.load(str(path))
            with torch.no_grad():
                torch.testing.assert_close(export(obs['policy']), a(obs))
                speed, hands = export.physical_estimates(obs['policy'])
                torch.testing.assert_close(speed, a.predict_cart_velocity(obs)/a.cart_velocity_scale)
                torch.testing.assert_close(hands, a.predict_hand_wrench(obs)*a.hand_scale)

    def test_invalid_targets_skip_both_optimizers_even_with_momentum(self):
        r=runner(); alg=r.alg
        with torch.inference_mode():
            obs=r.env.get_observations()
            for _ in range(4):
                act=alg.act(obs)
                obs,reward,done,extra=r.env.step(act)
                alg.process_env_step(obs,reward,done,extra)
        alg.update_estimator()  # Populate Adam moments with valid samples first.
        before=[p.detach().clone() for p in alg.actor.estimator_parameters()]
        for key in ('interaction_target','cart_velocity_target'):
            alg.storage.observations[key][..., :-1]=float('nan')
            alg.storage.observations[key][..., -1]=0.
        metrics=alg.update_estimator()
        self.assertEqual(metrics['estimator'],0.)
        self.assertEqual(metrics['cart_velocity_supervised'],0.)
        for p,q in zip(before,alg.actor.estimator_parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_cart_encoder_excludes_prior_but_uses_whole_body_torque(self):
        a = runner().alg.actor.eval(); x = observations()['policy']; changed = x.clone()
        changed.reshape(-1,20,FRAME_DIM)[:, :, -12:] += 100
        torch.testing.assert_close(a.cart_velocity_input(x), a.cart_velocity_input(changed), rtol=0, atol=0)
        self.assertEqual(a.cart_velocity_input(x).shape[-1], 5*CART_INPUT_FRAME_DIM)
        changed.reshape(-1,20,FRAME_DIM)[:, -1, 54] += 1
        self.assertFalse(torch.equal(a.cart_velocity_input(x), a.cart_velocity_input(changed)))


class ObservableTests(unittest.TestCase):
    def test_version_and_old_experiment_guards(self):
        from homie_pushcart_v53.training import _validate_experiment
        from homie_pushcart_v53.task_config import task_cfg_from_run
        with self.assertRaises(ValueError): _validate_experiment('homie_pushcart_v52')
        with tempfile.TemporaryDirectory() as folder:
            params=Path(folder)/'params'; params.mkdir()
            (params/'env.yaml').write_text('task:\n  version: 52\n',encoding='utf-8')
            with self.assertRaises(ValueError): task_cfg_from_run(folder)

    def test_measurement_only_builder_reset_and_leg_torques(self):
        n=2; z=lambda width: torch.zeros(n,width)
        h=ObservableHistory(n)
        kwargs=dict(q=z(27),dq=z(27),torque=torch.ones(n,27),gyro=z(3),
            gravity=torch.tensor([[0.,0.,-1.]]).repeat(n,1),him_velocity=z(3),
            relative_yaw=torch.tensor([.3,-.2]),heading_valid=torch.ones(n,dtype=torch.bool),
            heading_age_s=torch.zeros(n),default_q=z(27),task_reference=z(3),homie_command=z(3),
            upper_target=z(15),previous_high_action=z(18),previous_lower_action=torch.ones(n,12),
            fresh=torch.zeros(n,dtype=torch.bool))
        first=h.push(**kwargs).clone(); second=h.push(**kwargs)
        fields={f['name']:slice(f['start'],f['start']+f['dim']) for f in frame_layout()}
        self.assertNotIn('handle_position',fields); self.assertNotIn('cart_twist_estimate',fields)
        frame=second.reshape(n,20,FRAME_DIM)[:,-1]
        torch.testing.assert_close(frame[:,fields['joint_torque']],torch.full((n,27),.02))
        kwargs['fresh']=torch.tensor([True,False]); third=h.push(**kwargs).reshape(n,20,FRAME_DIM)
        self.assertEqual(third[0,:-1].abs().sum().item(),0.)
        self.assertGreater(third[1,:-1].abs().sum().item(),0.)
        self.assertEqual(third[0,-1,fields['joint_torque']].abs().sum().item(),0.)
        self.assertTrue(torch.isfinite(third).all())
        self.assertTrue(torch.equal(first.reshape(n,20,FRAME_DIM)[:,-1],frame))

    def test_vision_dropout_staleness_and_wrap(self):
        yaw,valid=visual_heading(torch.tensor([3.1416,-3.1416,float('nan'),.4]),
            torch.ones(4,dtype=torch.bool),torch.tensor([.02,.02,.02,.5]))
        self.assertEqual(valid.tolist(),[True,True,False,False])
        self.assertTrue(torch.isfinite(yaw).all())
        cfg=PushCartTaskCfg().observations; cfg.heading_dropout_probability=1.
        sensor=SimulatedVisualHeading(2,.02,cfg)
        sensor.reset(slice(None),False)
        self.assertFalse(sensor.sample(torch.zeros(2),True)[1].any())
        sensor.reset(slice(None),False)
        self.assertTrue(sensor.sample(torch.zeros(2),False)[1].all())


if __name__=='__main__': unittest.main()
