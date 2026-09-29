"""V6 Actor normalization, estimator isolation, t+1 storage, resume and export."""
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
from homie_pushcart_v6.hl_policy import HighLevelPolicy, import_rsl_rl5
import_rsl_rl5()
from rsl_rl.runners import OnPolicyRunner
from homie_pushcart_v6.interaction_policy import InteractionActor


def observations(n=8, invalid=False):
    policy, estimator = torch.randn(n, 605), torch.randn(n, 462)
    hands = .1 * policy[:, :12]
    target = torch.cat((torch.randn(n, 77), .1 * estimator[:, -3:]), -1)
    valid = torch.zeros(n, 1) if invalid else torch.ones(n, 1)
    if invalid:
        hands[:] = float("nan")
        target[:] = float("nan")
    return TensorDict({"policy": policy, "estimator": estimator, "critic": torch.randn(n, 665),
                       "hand_target": torch.cat((hands, valid), -1),
                       "him_next_target": torch.cat((target, valid), -1)}, batch_size=[n])


class SyntheticEnv:
    num_envs, num_actions, device, cfg = 8, 18, "cpu", {}

    def __init__(self):
        self.invalid = False
        self.obs = observations()

    def get_observations(self):
        return self.obs

    def step(self, actions):
        self.obs = observations(invalid=self.invalid)
        return self.obs, .2 * actions[:, 0], torch.zeros(8, dtype=torch.bool), {}


def config():
    return {
        "num_steps_per_env": 4, "save_interval": 100,
        "obs_groups": {"actor": ["policy", "estimator"], "critic": ["critic"]},
        "actor": {"class_name": "homie_pushcart_v6.interaction_policy:InteractionActor",
                  "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True,
                  "estimator_hidden_dims": [512, 256, 128],
                  "distribution_cfg": {"class_name": "rsl_rl.modules:GaussianDistribution", "init_std": .4}},
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [512, 256, 128],
                   "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "homie_pushcart_v6.interaction_ppo:InteractionPPO",
                      "num_learning_epochs": 2, "num_mini_batches": 2, "entropy_coef": .005,
                      "learning_rate": 1.e-3, "schedule": "adaptive", "desired_kl": .01,
                      "estimator_epochs": 2, "estimator_mini_batches": 3},
    }


def runner(cfg=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return OnPolicyRunner(SyntheticEnv(), cfg or config(), device="cpu")


def populate(runner):
    with torch.inference_mode():
        obs = runner.env.get_observations()
        for _ in range(runner.cfg["num_steps_per_env"]):
            actions = runner.alg.act(obs)
            obs, reward, done, extras = runner.env.step(actions)
            runner.alg.process_env_step(obs, reward, done, extras)
        runner.alg.compute_returns(obs)


def assert_nested_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_actor_preserves_full_history_and_normalizes_all_652_inputs(self):
        actor, obs = runner().alg.actor, observations()
        self.assertEqual(actor.mlp[0].in_features, 652)
        self.assertEqual(actor.force_estimator[0].in_features, 608)
        self.assertEqual(tuple(actor.obs_normalizer._mean.shape), (1, 652))
        self.assertEqual(tuple(actor.estimator_normalizer._mean.shape), (1, 605))
        raw = actor.unnormalized_actor_input(obs)
        torch.testing.assert_close(raw[:, :605], obs["policy"], rtol=0, atol=0)
        speed, latent = actor.cart_him(obs["estimator"])
        torch.testing.assert_close(raw[:, 605:608], speed, rtol=0, atol=0)
        torch.testing.assert_close(raw[:, 608:640], latent, rtol=0, atol=0)
        actor.obs_normalizer._mean.copy_(torch.linspace(-1., 1., 652)[None])
        actor.obs_normalizer._std.fill_(2.)
        expected = (raw - actor.obs_normalizer._mean) / 2.01
        torch.testing.assert_close(actor.get_latent(obs), expected, rtol=0, atol=0)
        expected_action = actor.distribution.deterministic_output(actor.mlp(expected))
        torch.testing.assert_close(actor(obs), expected_action, rtol=0, atol=0)
        actor.update_normalization(obs)
        self.assertEqual(actor.obs_normalizer.count.item(), 8)
        self.assertEqual(actor.estimator_normalizer.count.item(), 8)

    def test_privileges_and_targets_never_enter_actor_or_force_head(self):
        actor, obs = runner().alg.actor.eval(), observations()
        with torch.no_grad():
            actions, estimates = actor(obs), actor.physical_estimates(obs)
            obs["critic"].fill_(500.)
            obs["hand_target"].fill_(float("nan"))
            obs["him_next_target"].fill_(float("nan"))
            torch.testing.assert_close(actor(obs), actions, rtol=0, atol=0)
            for original, changed in zip(estimates, actor.physical_estimates(obs)):
                torch.testing.assert_close(original, changed, rtol=0, atol=0)
            hands = actor.predict_hand_wrench(obs)
            actor.cart_him.encoder[-1].bias[3:].add_(1.)
            torch.testing.assert_close(actor.predict_hand_wrench(obs), hands, rtol=0, atol=0)
            self.assertFalse(torch.equal(actor(obs), actions))  # HIM latent is actually used by the Actor.

    def test_three_optimizers_and_gradients_are_isolated(self):
        alg, obs = runner().alg, observations()
        actor = alg.actor
        actor(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in actor.estimator_parameters()))
        actor.zero_grad(set_to_none=True)
        actor.predict_hand_wrench(obs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in actor.cart_velocity_parameters()))
        self.assertTrue(any(p.grad is not None for p in actor.force_parameters()))
        groups = [{id(p) for group in optimizer.param_groups for p in group["params"]}
                  for optimizer in (alg.optimizer, alg.estimator_optimizer, alg.cart_velocity_optimizer)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_next_target_backfill_preserves_t_hand_labels_and_does_not_alias(self):
        r = runner()
        initial = r.env.get_observations()
        before = initial.clone()
        with torch.inference_mode():
            r.alg.act(initial)
            initial["policy"].fill_(999.)  # Simulate an environment reusing its observation buffers.
            following = observations()
            following["policy"][0].zero_()  # Post-reset observation for a terminal transition.
            following["him_next_target"].fill_(7.)
            following["him_next_target"][:, 80] = 1.
            target = following["him_next_target"].clone()
            r.alg.process_env_step(following, torch.ones(8), torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]), {})
        stored = r.alg.storage.observations[0]
        for key in ("policy", "estimator", "critic", "hand_target"):
            torch.testing.assert_close(stored[key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(stored["him_next_target"], target, rtol=0, atol=0)
        with torch.inference_mode():
            following["him_next_target"].zero_()
        torch.testing.assert_close(stored["him_next_target"], target, rtol=0, atol=0)

    def test_training_updates_both_estimators_and_him_follows_each_ppo_rate(self):
        r = runner()
        him_before = copy.deepcopy(r.alg.actor.cart_him.state_dict())
        force_before = copy.deepcopy(r.alg.actor.force_estimator.state_dict())
        rates = []
        original = r.alg.actor.cart_him.update

        def recorded_update(*args, **kwargs):
            rates.append(kwargs["lr"])
            return original(*args, **kwargs)

        r.alg.actor.cart_him.update = recorded_update
        populate(r)
        metrics = r.alg.update()
        self.assertEqual(len(rates), r.alg.num_learning_epochs * r.alg.num_mini_batches)
        self.assertEqual(rates[-1], r.alg.learning_rate)
        self.assertEqual(r.alg.cart_velocity_optimizer.param_groups[0]["lr"], rates[-1])
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        self.assertIn("cart_him_swap", metrics)
        self.assertIn("hand_mae_L_fx", metrics)
        self.assertTrue(any(not torch.equal(value, r.alg.actor.cart_him.state_dict()[key])
                            for key, value in him_before.items()))
        self.assertTrue(any(not torch.equal(value, r.alg.actor.force_estimator.state_dict()[key])
                            for key, value in force_before.items()))

    def test_invalid_auxiliary_batches_cannot_advance_existing_adam_momentum(self):
        r = runner()
        populate(r)
        r.alg.update()
        r.env.invalid = True
        r.env.obs = observations(invalid=True)
        saved_him = copy.deepcopy(r.alg.actor.cart_him.state_dict())
        saved_force = copy.deepcopy(r.alg.actor.force_estimator.state_dict())
        saved_him_opt = copy.deepcopy(r.alg.cart_velocity_optimizer.state_dict())
        saved_force_opt = copy.deepcopy(r.alg.estimator_optimizer.state_dict())
        populate(r)
        metrics = r.alg.update()
        self.assertEqual(metrics["cart_velocity_supervised"], 0.)
        self.assertEqual(metrics["cart_him_swap"], 0.)
        self.assertEqual(metrics["estimator"], 0.)
        assert_nested_equal(self, saved_him, r.alg.actor.cart_him.state_dict())
        assert_nested_equal(self, saved_force, r.alg.actor.force_estimator.state_dict())
        assert_nested_equal(self, saved_him_opt, r.alg.cart_velocity_optimizer.state_dict())
        assert_nested_equal(self, saved_force_opt, r.alg.estimator_optimizer.state_dict())

    def test_main_ppo_update_matches_installed_v32_ppo_when_auxiliary_labels_are_invalid(self):
        from rsl_rl.algorithms import PPO
        cfg = config()
        cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, estimator_epochs=0)
        r = runner(cfg)
        r.env.invalid, r.env.obs = True, observations(invalid=True)
        populate(r)
        reference_modules = runner().alg
        reference_modules.actor.load_state_dict(r.alg.actor.state_dict())
        reference_modules.critic.load_state_dict(r.alg.critic.state_dict())
        reference = PPO(reference_modules.actor, reference_modules.critic, copy.deepcopy(r.alg.storage),
                        num_learning_epochs=1, num_mini_batches=1, entropy_coef=.005,
                        learning_rate=1.e-3, schedule="adaptive", desired_kl=.01)
        torch.manual_seed(991)
        actual = r.alg.update()
        torch.manual_seed(991)
        expected = reference.update()
        for key in ("value", "surrogate", "entropy"):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(r.alg.learning_rate, reference.learning_rate)
        assert_nested_equal(self, r.alg.actor.state_dict(), reference.actor.state_dict())
        assert_nested_equal(self, r.alg.critic.state_dict(), reference.critic.state_dict())

    def test_checkpoint_normalizers_all_optimizers_and_export_round_trip(self):
        r = runner()
        populate(r)
        r.alg.update()
        saved = copy.deepcopy(r.alg.save())
        restored = runner().alg
        self.assertTrue(restored.load(saved, None, True))
        assert_nested_equal(self, saved, restored.save())
        self.assertEqual(restored.learning_rate, r.alg.learning_rate)
        self.assertEqual(restored.actor.cart_him.learning_rate, r.alg.actor.cart_him.learning_rate)
        actor, obs = restored.actor.eval(), observations()
        exported = torch.jit.script(actor.as_jit().eval())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actor.pt"
            exported.save(str(path))
            exported = torch.jit.load(str(path))
            with torch.no_grad():
                torch.testing.assert_close(exported(obs["policy"], obs["estimator"]), actor(obs), rtol=0, atol=0)
                for actual, expected in zip(exported.physical_estimates(obs["policy"], obs["estimator"]),
                                            actor.physical_estimates(obs)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        bad = copy.deepcopy(saved)
        bad["v6_network_metadata"]["version"] = 53
        with self.assertRaisesRegex(ValueError, "metadata"):
            restored.load(bad, None, True)

    def test_standalone_loader_uses_saved_architecture_and_rejects_old_models(self):
        import yaml
        from homie_pushcart_v6.baseline import config_dict
        from homie_pushcart_v6.task_config import PushCartTaskCfg
        r = runner()
        obs = observations()
        r.alg.actor.update_normalization(obs)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "params").mkdir()
            (folder / "params/env.yaml").write_text(yaml.safe_dump({"task": config_dict(PushCartTaskCfg())}), encoding="utf-8")
            agent_file = folder / "params/agent.yaml"
            agent_file.write_text(yaml.safe_dump(config()), encoding="utf-8")
            checkpoint = folder / "model_1.pt"
            torch.save({**r.alg.save(), "iter": 1}, checkpoint)
            loaded = HighLevelPolicy(checkpoint)
            with torch.no_grad():
                torch.testing.assert_close(loaded(obs["policy"], obs["estimator"]), r.alg.actor(obs), rtol=0, atol=0)
                torch.testing.assert_close(loaded(obs), r.alg.actor(obs), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "independent"):
                loaded(obs["policy"])
            old_config = config()
            old_config["actor"]["class_name"] = "homie_pushcart_v53.interaction_policy:InteractionActor"
            agent_file.write_text(yaml.safe_dump(old_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "earlier versions"):
                HighLevelPolicy(checkpoint)

    def test_actor_rejects_unapproved_history_or_normalization_changes(self):
        obs, cfg = observations(), config()
        kwargs = cfg["actor"].copy()
        kwargs.pop("class_name")
        kwargs["obs_normalization"] = False
        with self.assertRaises(ValueError):
            InteractionActor(obs, cfg["obs_groups"], "actor", 18, **kwargs)
        kwargs["obs_normalization"] = True
        obs["policy"] = torch.randn(8, 620)
        with self.assertRaises(ValueError):
            InteractionActor(obs, cfg["obs_groups"], "actor", 18, **kwargs)


if __name__ == "__main__":
    unittest.main()
