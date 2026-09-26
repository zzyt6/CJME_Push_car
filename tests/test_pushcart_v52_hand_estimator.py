"""V5.2 per-hand estimator: isolation from PPO, own optimizer, composition, prior, checkpoint, export."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart_v52.hl_policy import import_rsl_rl5
import_rsl_rl5()

from homie_pushcart_v52.interaction_policy import InteractionActor, compose_handle_wrench
from homie_pushcart_v52.interaction_ppo import InteractionPPO, masked_interaction_huber
from homie_pushcart_v52.observation import FRAME_DIM, HAND_WRENCH_SCALE, body_to_cart_heading, static_hand_wrench
from rsl_rl.runners import OnPolicyRunner

FRAME, HISTORY = 136, 20
OFFSETS = [[0.06, 0.18, 0.02], [0.06, -0.18, 0.02]]


def observations(n=8, *, label_offset=0.0):
    history = torch.randn(n, FRAME * HISTORY)
    # Learnable synthetic label: a fixed function of the current frame.
    target = torch.tanh(history[:, -12:] + label_offset)
    values = {"policy": history, "critic": torch.randn(n, 41),
              "interaction_target": torch.cat((target, torch.ones(n, 1)), -1)}
    return TensorDict(values, batch_size=[n])


def config(force_to_policy=True, epochs=2):
    return dict(
        num_steps_per_env=4, save_interval=100,
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=dict(class_name="homie_pushcart_v52.interaction_policy:InteractionActor",
                   hidden_dims=[32, 24], encoder_hidden_dims=[32, 24], estimator_hidden_dims=[32, 24],
                   frame_dim=FRAME, history_length=HISTORY, estimator_history=5,
                   force_to_policy=force_to_policy, palm_offsets=OFFSETS, obs_normalization=False,
                   distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.4}),
        critic=dict(class_name="rsl_rl.models:MLPModel", hidden_dims=[32, 24], obs_normalization=False),
        algorithm=dict(class_name="homie_pushcart_v52.interaction_ppo:InteractionPPO",
                       estimator_learning_rate=1e-3, estimator_epochs=epochs, estimator_mini_batches=2,
                       huber_delta=1.0, num_learning_epochs=2, num_mini_batches=2,
                       schedule="adaptive", desired_kl=0.01, rnd_cfg=None, symmetry_cfg=None),
    )


class SyntheticEnv:
    num_envs = 8
    num_actions = 18
    device = "cpu"
    cfg = {}

    def __init__(self):
        self.obs = observations(self.num_envs)

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.obs = observations(self.num_envs)
        return self.obs, 0.2 * actions[:, 0] + torch.randn(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}


def make_runner(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(SyntheticEnv(), config(**kwargs), device="cpu")
    runner.logger.init_logging_writer()
    return runner


def populate(algorithm, env):
    with torch.inference_mode():
        obs = env.get_observations()
        for _ in range(algorithm.storage.num_transitions_per_env):
            actions = algorithm.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            algorithm.process_env_step(obs, rewards, dones, extras)
        algorithm.compute_returns(obs)


def grad_size(module):
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


class SchemaTests(unittest.TestCase):
    def test_frame_width_and_hand_scale(self):
        self.assertEqual(FRAME_DIM, FRAME)
        self.assertEqual(len(HAND_WRENCH_SCALE), 6)


class CompositionTests(unittest.TestCase):
    def test_compose_sums_forces_and_transports_moments(self):
        hands = torch.zeros(1, 12)
        hands[0, 0:3] = torch.tensor([10., 2., 0.])      # left force
        hands[0, 5] = 1.                                  # left Mz
        hands[0, 6:9] = torch.tensor([6., -1., 0.])      # right force
        out = compose_handle_wrench(hands, torch.tensor(OFFSETS))
        rl, rr = OFFSETS
        mz = 1. + (rl[0] * 2. - rl[1] * 10.) + (rr[0] * -1. - rr[1] * 6.)
        torch.testing.assert_close(out, torch.tensor([[16., 1., mz]]))

    def test_matches_general_resultant_about_handle(self):
        from homie_pushcart_v52.cart_quality import resultant_wrench
        torch.manual_seed(3)
        f, m = torch.randn(5, 2, 3), torch.randn(5, 2, 3)
        points = torch.tensor(OFFSETS).expand(5, -1, -1)
        force, moment = resultant_wrench(f, m, points, torch.zeros(5, 3))
        hands = torch.cat((f[:, 0], m[:, 0], f[:, 1], m[:, 1]), -1)
        torch.testing.assert_close(compose_handle_wrench(hands, torch.tensor(OFFSETS)),
                                   torch.stack((force[:, 0], force[:, 1], moment[:, 2]), -1))


class StaticPriorTests(unittest.TestCase):
    def test_recovers_wrench_from_jacobian_transpose_torque(self):
        torch.manual_seed(5)
        n = 4
        jac = torch.zeros(n, 2, 6, 27)
        jac[:, 0, :, 13:20] = torch.randn(n, 6, 7)
        jac[:, 1, :, 20:27] = torch.randn(n, 6, 7)
        wrench = torch.randn(n, 2, 6)
        tau = torch.zeros(n, 15)
        tau[:, 1:8] = (jac[:, 0, :, 13:20].transpose(-1, -2) @ wrench[:, 0, :, None])[..., 0]
        tau[:, 8:15] = (jac[:, 1, :, 20:27].transpose(-1, -2) @ wrench[:, 1, :, None])[..., 0]
        identity = torch.eye(3).expand(n, -1, -1)
        prior = static_hand_wrench(jac, tau, identity, damping=1e-9)
        torch.testing.assert_close(prior, wrench.reshape(n, 12), atol=1e-3, rtol=1e-3)

    def test_rotation_to_cart_axes_and_finite_output(self):
        yaw = torch.tensor([0.5])
        rotation = body_to_cart_heading(torch.eye(3)[None], yaw)
        # Body x expressed in axes rotated by +yaw has components (cos, -sin).
        torch.testing.assert_close(rotation[0, :2, 0], torch.cat((torch.cos(yaw), -torch.sin(yaw))))
        prior = static_hand_wrench(torch.zeros(1, 2, 6, 27), torch.full((1, 15), float("nan")), rotation)
        self.assertTrue(torch.isfinite(prior).all())


class ActorIsolationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)
        self.runner = make_runner()
        self.actor = self.runner.alg.actor
        self.obs = self.runner.env.get_observations()

    def test_no_privileged_or_label_dependency_and_latent_layout(self):
        expected = self.actor(self.obs)
        poisoned = self.obs.clone()
        poisoned["critic"].fill_(float("nan"))
        poisoned["interaction_target"].fill_(float("nan"))
        torch.testing.assert_close(self.actor(poisoned), expected, rtol=0, atol=0)
        latent = self.actor.get_latent(self.obs)
        self.assertEqual(latent.shape, (8, FRAME + 16 + 15))
        hands = self.actor.predict_hand_wrench(self.obs)
        torch.testing.assert_close(latent[:, FRAME + 16:FRAME + 28], hands)
        torch.testing.assert_close(latent[:, -3:], self.actor.compose_resultant(hands))

    def test_estimator_reads_only_last_frames(self):
        changed = self.obs.clone()
        changed["policy"][:, :FRAME * (HISTORY - 5)] += 100.
        torch.testing.assert_close(self.actor.predict_hand_wrench(changed), self.actor.predict_hand_wrench(self.obs))

    def test_ppo_gradient_does_not_reach_estimator(self):
        self.actor(self.obs, stochastic_output=True)
        actions = self.actor.output_mean.detach() + torch.linspace(-.2, .3, 18)[None]
        (-self.actor.get_output_log_prob(actions).mean()).backward()
        self.assertGreater(grad_size(self.actor.history_encoder), 0)
        self.assertGreater(grad_size(self.actor.mlp), 0)
        self.assertEqual(grad_size(self.actor.force_estimator), 0)

    def test_force_to_policy_false_zeros_the_load_slots(self):
        runner = make_runner(force_to_policy=False)
        latent = runner.alg.actor.get_latent(runner.env.get_observations())
        self.assertTrue((latent[:, FRAME + 16:] == 0).all())


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)

    def test_optimizers_are_disjoint_and_cover_all_parameters(self):
        alg = make_runner().alg
        ppo = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
        est = {id(p) for g in alg.estimator_optimizer.param_groups for p in g["params"]}
        self.assertFalse(ppo & est)
        self.assertEqual(est, {id(p) for p in alg.actor.force_estimator.parameters()})
        everything = {id(p) for p in alg.actor.parameters()} | {id(p) for p in alg.critic.parameters()}
        self.assertEqual(ppo | est, everything)

    def test_update_trains_both_and_estimator_lr_is_not_adaptive(self):
        runner = make_runner()
        alg = runner.alg
        before = copy.deepcopy(alg.actor.state_dict())
        populate(alg, runner.env)
        losses = alg.update()
        self.assertEqual(alg.storage.step, 0)
        for key in ("estimator", "estimator_eval_huber", "estimator_r2_L_fx", "estimator_r2_handle_Fx"):
            self.assertIn(key, losses)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        for prefix in ("history_encoder.", "force_estimator.", "mlp."):
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in alg.actor.state_dict().items()
                                if k.startswith(prefix)), prefix)
        self.assertEqual(alg.estimator_optimizer.param_groups[0]["lr"], 1e-3)
        # The running input normalizer saw rollout data.
        self.assertGreater(int(alg.actor.estimator_normalizer.count), 0)

    def test_estimator_learns_synthetic_labels(self):
        runner = make_runner(epochs=20)
        alg = runner.alg
        obs = observations(512)
        alg.actor.estimator_normalizer.update(obs["policy"][:, -FRAME * 5:])
        target = obs["interaction_target"]
        first = masked_interaction_huber(alg.actor.predict_hand_wrench(obs), target)[0].item()
        for _ in range(300):
            loss, _ = masked_interaction_huber(alg.actor.predict_hand_wrench(obs), target)
            alg.estimator_optimizer.zero_grad()
            loss.backward()
            alg.estimator_optimizer.step()
        self.assertLess(loss.item(), 0.3 * first)

    def test_masked_loss_ignores_invalid_rows_and_nans(self):
        prediction = torch.zeros(3, 12, requires_grad=True)
        target = torch.zeros(3, 13)
        target[0, :12], target[0, 12] = 1., 1.
        target[1, :12], target[1, 12] = float("nan"), 1.
        target[2, :12], target[2, 12] = 999., 0.
        loss, fraction = masked_interaction_huber(prediction, target)
        self.assertAlmostEqual(loss.item(), 0.5)
        self.assertAlmostEqual(fraction.item(), 1 / 3)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[1:] == 0).all())

    def test_checkpoint_restores_both_optimizers_normalizer_and_export(self):
        source = make_runner()
        with contextlib.redirect_stdout(io.StringIO()):
            source.learn(1)
        evaluation = source.env.obs.clone()
        expected = source.get_inference_policy()(evaluation).detach()
        est_state = copy.deepcopy(source.alg.estimator_optimizer.state_dict())
        self.assertTrue(est_state["state"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            source.save(path, {"test": "v52"})
            target = make_runner()
            target.load(path, map_location="cpu")
            torch.testing.assert_close(target.alg.actor(evaluation), expected, rtol=0, atol=0)
            restored = target.alg.estimator_optimizer.state_dict()
            for param, state in est_state["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(restored["state"][param][name], value, rtol=0, atol=0)
            source.export_policy_to_jit(directory)
            exported = torch.jit.load(str(Path(directory) / "policy.pt"))
            torch.testing.assert_close(exported(evaluation["policy"]), expected, atol=1e-6, rtol=1e-6)
            populate(target.alg, target.env)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in target.alg.update().values()))


if __name__ == "__main__":
    unittest.main()
