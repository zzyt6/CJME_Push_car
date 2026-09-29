"""V6 cart HIM: parity with original HIM, temporal labels, masks and export."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v6.him_estimator import (
    CartHIMEstimator, make_next_critic_observation, mirror_cart_frame,
    mirror_cart_velocity,
)


def original_estimator():
    path = ROOT / "HomieRL/rsl_rl/rsl_rl/modules/him_estimator.py"
    spec = importlib.util.spec_from_file_location("v6_test_original_him", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HIMEstimator(temporal_steps=6, num_one_step_obs=77, num_height_points=0)


def samples(n=8):
    return torch.randn(n, 462), make_next_critic_observation(torch.randn(n, 77), torch.randn(n, 3))


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


class CartHIMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_exact_original_him_forward_loss_and_adam_update(self):
        reference, estimator = original_estimator(), CartHIMEstimator()
        estimator.load_state_dict(reference.state_dict(), strict=True)
        history, target = samples()
        for actual, expected in zip(estimator(history), reference(history)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Repeated adaptive rates also verify Adam moments, not just one SGD-like step.
        for rate in (1.0e-3, 2.0e-4, 3.0e-4):
            expected_loss = reference.update(history, target, lr=rate)
            actual_loss = estimator.update(history, target, lr=rate)
            self.assertEqual(actual_loss, expected_loss)
            assert_nested_equal(self, estimator.state_dict(), reference.state_dict())
            assert_nested_equal(self, estimator.optimizer.state_dict(), reference.optimizer.state_dict())
            self.assertEqual(estimator.learning_rate, rate)

    def test_dedicated_next_target_scales_and_him_target_slice(self):
        frame = torch.arange(77, dtype=torch.float32).expand(3, -1).clone()
        velocity = torch.tensor([[1., 2., 3.]]).expand(3, -1).clone()
        packed = make_next_critic_observation(frame, velocity)
        torch.testing.assert_close(packed[:, 77:], torch.tensor([[2., 4., 1.5]]).expand(3, -1))
        estimator = CartHIMEstimator()
        seen = []
        hook = estimator.target.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        estimator.losses(torch.randn(3, 462), packed)
        hook.remove()
        expected = torch.cat((frame[:, 3:], packed[:, 77:]), -1)
        torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
        # Relative yaw survives as target column 73; no command is retained.
        torch.testing.assert_close(seen[0][:, 73], frame[:, 76])

    def test_forward_detaches_while_encode_trains_only_estimator(self):
        estimator = CartHIMEstimator()
        history = torch.randn(4, 462, requires_grad=True)
        velocity, latent = estimator(history)
        self.assertFalse(velocity.requires_grad or latent.requires_grad)
        velocity, latent = estimator.encode(history)
        (velocity.square().mean() + latent[:, 0].mean()).backward()
        self.assertIsNone(history.grad)
        self.assertTrue(any(p.grad is not None for p in estimator.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in estimator.target.parameters()))
        torch.testing.assert_close(latent.norm(dim=-1), torch.ones(4))

    def test_invalid_rows_match_valid_subset_and_empty_batch_cannot_advance_adam(self):
        estimator, selected = CartHIMEstimator(), CartHIMEstimator()
        selected.load_state_dict(estimator.state_dict())
        history, target = samples(7)
        valid = torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        history[~valid] = float("nan")
        target[~valid] = float("inf")
        result = estimator.update(history, target, valid=valid[:, None])
        expected = selected.update(history[valid], target[valid])
        self.assertEqual(result, expected)
        assert_nested_equal(self, estimator.state_dict(), selected.state_dict())
        state = copy.deepcopy(estimator.state_dict())
        optimizer = copy.deepcopy(estimator.optimizer.state_dict())
        result = estimator.update(history.fill_(float("nan")), target.fill_(float("nan")),
                                  valid=torch.zeros(7), lr=1.e-5)
        self.assertEqual(result, (0., 0.))
        assert_nested_equal(self, state, estimator.state_dict())
        assert_nested_equal(self, optimizer, estimator.optimizer.state_dict())
        self.assertEqual(estimator.learning_rate, 1.e-3)

    def test_bad_validity_and_unmasked_nonfinite_samples_fail_before_mutation(self):
        estimator = CartHIMEstimator()
        history, target = samples(3)
        state = copy.deepcopy(estimator.state_dict())
        with self.assertRaises(ValueError):
            estimator.update(history, target, valid=torch.tensor([1., .2, 0.]))
        target[0, 77] = float("nan")
        with self.assertRaises(ValueError):
            estimator.update(history, target)
        assert_nested_equal(self, state, estimator.state_dict())
        self.assertFalse(estimator.optimizer.state)

    def test_checkpoint_resume_and_scripted_inference_preserve_outputs(self):
        estimator = CartHIMEstimator()
        history, target = samples()
        estimator.update(history, target, lr=3.e-4)
        restored = CartHIMEstimator()
        restored.load_state_dict(copy.deepcopy(estimator.state_dict()))
        restored.optimizer.load_state_dict(copy.deepcopy(estimator.optimizer.state_dict()))
        # Caller supplies the current PPO rate, including the first resumed minibatch.
        self.assertEqual(estimator.update(history, target, lr=2.e-4),
                         restored.update(history, target, lr=2.e-4))
        assert_nested_equal(self, estimator.state_dict(), restored.state_dict())
        scripted = torch.jit.script(restored.eval())
        for actual, expected in zip(scripted(history), restored(history)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        metadata = restored.metadata()
        self.assertEqual(metadata["encoder_dims"], [462, 256, 256, 35])
        self.assertEqual(metadata["target_dims"], [77, 256, 256, 32])
        self.assertFalse(metadata["symmetry_augmentation_enabled"])

    def test_opt_in_reflection_is_an_involution_and_distinguishes_yaw_rate(self):
        frame = torch.arange(77, dtype=torch.float32).expand(2, 6, -1).clone()
        mirrored = mirror_cart_frame(frame)
        torch.testing.assert_close(mirror_cart_frame(mirrored), frame, rtol=0, atol=0)
        torch.testing.assert_close(mirrored[..., :10], frame[..., :10] * frame.new_tensor(
            [1, -1, -1, 1, -1, 1, -1, 1, -1, 1]))
        torch.testing.assert_close(mirrored[..., 10], frame[..., 16])
        torch.testing.assert_close(mirrored[..., 76], -frame[..., 76])
        velocity = torch.tensor([[1., 2., 3.]])
        torch.testing.assert_close(mirror_cart_velocity(velocity), torch.tensor([[1., -2., -3.]]))
        torch.testing.assert_close(mirror_cart_velocity(mirror_cart_velocity(velocity)), velocity)


if __name__ == "__main__":
    unittest.main()
