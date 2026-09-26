"""CPU integration tests using the installed RSL-RL 5 runner and real PPO updates."""
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

from homie_pushcart_v51.hl_policy import import_rsl_rl5
import_rsl_rl5()

from homie_pushcart_v51.interaction_policy import InteractionActor
from homie_pushcart_v51.interaction_ppo import InteractionPPO, masked_interaction_huber
from rsl_rl.runners import OnPolicyRunner


def observations(n=8, *, labels=True, label_offset=0.0):
    history = torch.randn(n, 2480)
    values = {"policy": history, "critic": torch.randn(n, 41)}
    if labels:
        values["interaction_target"] = torch.cat((history[:, -3:] + label_offset, torch.ones(n, 1)), dim=-1)
    return TensorDict(values, batch_size=[n])


def config(aux_coef=0.1):
    return dict(
        num_steps_per_env=4, save_interval=100,
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=dict(class_name="homie_pushcart_v51.interaction_policy:InteractionActor",
                   hidden_dims=[32, 24], encoder_hidden_dims=[32, 24], interaction_hidden_dims=[16],
                   obs_normalization=False, distribution_cfg={
                       "class_name": "rsl_rl.modules:GaussianDistribution", "init_std": 0.4}),
        critic=dict(class_name="rsl_rl.models:MLPModel", hidden_dims=[32, 24], obs_normalization=False),
        algorithm=dict(class_name="homie_pushcart_v51.interaction_ppo:InteractionPPO",
                       aux_coef=aux_coef, huber_delta=1.0, num_learning_epochs=2, num_mini_batches=2,
                       schedule="adaptive", desired_kl=0.01, rnd_cfg=None, symmetry_cfg=None),
    )


class SyntheticEnv:
    num_envs = 8
    num_actions = 18
    device = "cpu"
    cfg = {}

    def __init__(self, labels=True, label_offset=0.0):
        self.labels = labels
        self.label_offset = label_offset
        self.obs = self.make_obs()

    def make_obs(self):
        return observations(self.num_envs, labels=self.labels, label_offset=self.label_offset)

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.obs = self.make_obs()
        rewards = 0.2 * actions[:, 0] + torch.randn(self.num_envs)
        return self.obs, rewards, torch.zeros(self.num_envs, dtype=torch.bool), {}


def make_runner(aux_coef=0.1, labels=True, label_offset=0.0):
    with contextlib.redirect_stdout(io.StringIO()):
        runner = OnPolicyRunner(SyntheticEnv(labels, label_offset), config(aux_coef), device="cpu")
    # Normal initialization with log_dir=None (no fake logger or monkeypatch).
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


class InteractionActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.runner = make_runner()
        self.actor = self.runner.alg.actor
        self.obs = self.runner.env.get_observations()

    def test_no_privileged_or_label_dependency_and_current_frame_order(self):
        expected = self.actor(self.obs)
        policy_only = self.obs.select("policy")
        torch.testing.assert_close(self.actor(policy_only), expected, rtol=0, atol=0)
        poisoned = self.obs.clone()
        poisoned["critic"].fill_(float("nan"))
        poisoned["interaction_target"].fill_(float("nan"))
        torch.testing.assert_close(self.actor(poisoned), expected, rtol=0, atol=0)
        latent = self.actor.get_latent(self.obs)
        self.assertEqual(latent.shape, (8, 143))
        self.assertEqual(expected.shape, (8, 18))
        torch.testing.assert_close(latent[:, :124], self.obs["policy"][:, -124:])
        torch.testing.assert_close(latent[:, -3:], self.actor.predict_interaction(policy_only))

    def test_ppo_gradient_reaches_encoder_and_force_head_without_supervision(self):
        self.actor(self.obs.select("policy"), stochastic_output=True)
        actions = self.actor.output_mean.detach() + torch.linspace(-0.2, 0.3, 18)[None, :]
        policy_loss = -self.actor.get_output_log_prob(actions).mean()
        policy_loss.backward()
        self.assertGreater(grad_size(self.actor.history_encoder), 0)
        self.assertGreater(grad_size(self.actor.interaction_head), 0)
        self.assertGreater(grad_size(self.actor.mlp), 0)

    def test_huber_gradient_reaches_encoder_and_force_head_independently(self):
        loss, fraction = masked_interaction_huber(self.actor.predict_interaction(self.obs),
                                                 self.obs["interaction_target"])
        loss.backward()
        self.assertEqual(fraction.item(), 1)
        self.assertGreater(grad_size(self.actor.history_encoder), 0)
        self.assertGreater(grad_size(self.actor.interaction_head), 0)
        self.assertEqual(grad_size(self.actor.mlp), 0)

    def test_actor_rejects_target_group_and_running_history_normalization(self):
        with self.assertRaisesRegex(ValueError, "policy"):
            InteractionActor(self.obs, {"actor": ["policy", "interaction_target"]}, "actor", 18)
        with self.assertRaisesRegex(ValueError, "obs_normalization"):
            InteractionActor(self.obs, {"actor": ["policy"]}, "actor", 18, obs_normalization=True)

    def test_official_jit_export_roundtrip_and_onnx_wrapper_parity(self):
        expected = self.actor(self.obs).detach()
        with tempfile.TemporaryDirectory() as directory:
            self.runner.export_policy_to_jit(directory)
            exported = torch.jit.load(str(Path(directory) / "policy.pt"))
            torch.testing.assert_close(exported(self.obs["policy"]), expected)
            exported.reset()
        onnx_wrapper = self.actor.as_onnx()
        torch.testing.assert_close(onnx_wrapper(self.obs["policy"]), expected)
        self.assertEqual(onnx_wrapper.get_dummy_inputs()[0].shape, (1, 2480))


class InteractionLossTests(unittest.TestCase):
    def test_valid_mask_uses_label_count_and_invalid_nans_do_not_contaminate_gradients(self):
        predictions = torch.zeros(4, 3, requires_grad=True)
        targets = torch.tensor([[1., 2., 3., 1.], [999., 999., 999., 0.],
                                [float("nan"), 0., 0., 1.], [1., 1., 1., float("nan")]])
        loss, fraction = masked_interaction_huber(predictions, targets)
        self.assertAlmostEqual(loss.item(), 1.5)
        self.assertAlmostEqual(fraction.item(), 0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertTrue((predictions.grad[1:] == 0).all())
        self.assertTrue((predictions.grad[0] < 0).all())

    def test_all_invalid_labels_give_differentiable_zero(self):
        predictions = torch.randn(8, 3, requires_grad=True)
        targets = torch.full((8, 4), float("nan"))
        loss, fraction = masked_interaction_huber(predictions, targets)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(fraction.item(), 0)
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions))


class InteractionPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)

    def test_actual_ppo_minibatch_update_changes_both_networks_and_recomputes_history(self):
        runner = make_runner()
        alg = runner.alg
        self.assertIsInstance(alg, InteractionPPO)
        initial = {name: value.clone() for name, value in alg.actor.state_dict().items()}
        populate(alg, runner.env)
        calls = []
        handle = alg.actor.history_encoder.register_forward_hook(lambda module, args, output: calls.append(output.shape))
        losses = alg.update()
        handle.remove()
        self.assertEqual(len(calls), 8)  # 2 epochs * 2 batches * (policy + auxiliary).
        self.assertEqual(alg.storage.step, 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertGreater(losses["interaction"], 0)
        for prefix in ("history_encoder.", "interaction_head.", "mlp."):
            self.assertTrue(any(not torch.equal(value, initial[name])
                                for name, value in alg.actor.state_dict().items() if name.startswith(prefix)))
        optimized = {id(p) for group in alg.optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in optimized for p in alg.actor.history_encoder.parameters()))
        self.assertTrue(all(id(p) in optimized for p in alg.actor.interaction_head.parameters()))

    def test_pre_action_label_is_stored_not_following_transition_label(self):
        runner = make_runner()
        before = runner.env.get_observations()
        before["interaction_target"].fill_(1.)
        after = before.clone()
        after["interaction_target"].fill_(9.)
        with torch.inference_mode():
            runner.alg.act(before)
            runner.alg.process_env_step(after, torch.zeros(8), torch.zeros(8), {})
        torch.testing.assert_close(runner.alg.storage.observations[0]["interaction_target"],
                                   before["interaction_target"])

    def test_aux_zero_retains_capacity_and_is_identical_without_labels_or_with_changed_labels(self):
        runs = []
        for labels, offset in ((True, 0.), (True, 1000.), (False, 0.)):
            torch.manual_seed(88)
            runner = make_runner(aux_coef=0., labels=labels, label_offset=offset)
            populate(runner.alg, runner.env)
            losses = runner.alg.update()
            self.assertEqual(losses["interaction"], 0)
            runs.append(runner)
        for other in runs[1:]:
            for name, value in runs[0].alg.actor.state_dict().items():
                torch.testing.assert_close(other.alg.actor.state_dict()[name], value, rtol=0, atol=0)
        supervised = make_runner(aux_coef=0.1)
        self.assertEqual(sum(p.numel() for p in supervised.alg.actor.parameters()),
                         sum(p.numel() for p in runs[0].alg.actor.parameters()))

    def test_official_runner_learn_and_checkpoint_restore_models_optimizer_iteration(self):
        source = make_runner()
        with contextlib.redirect_stdout(io.StringIO()):
            source.learn(1)
        source.current_learning_iteration = 17
        evaluation_obs = source.env.obs.clone()
        expected = source.get_inference_policy()(evaluation_obs).detach()
        optimizer_before = copy.deepcopy(source.alg.optimizer.state_dict())
        self.assertTrue(optimizer_before["state"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            source.save(path, {"test": "v51"})
            target = make_runner()
            info = target.load(path, map_location="cpu")
            self.assertEqual(info, {"test": "v51"})
            self.assertEqual(target.current_learning_iteration, 17)
            torch.testing.assert_close(target.alg.actor(evaluation_obs), expected, rtol=0, atol=0)
            self.assertEqual(target.alg.learning_rate, source.alg.learning_rate)
            for name, value in source.alg.critic.state_dict().items():
                torch.testing.assert_close(target.alg.critic.state_dict()[name], value, rtol=0, atol=0)
            optimizer_after = target.alg.optimizer.state_dict()
            self.assertEqual(optimizer_after["param_groups"], optimizer_before["param_groups"])
            for param, state in optimizer_before["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(optimizer_after["state"][param][name], value, rtol=0, atol=0)
            # Continue from the restored optimizer through another actual update.
            populate(target.alg, target.env)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in target.alg.update().values()))


if __name__ == "__main__":
    unittest.main()
