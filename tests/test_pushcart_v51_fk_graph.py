"""The optional FK CUDA graph must observe the current sample and reset rows."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HomieRL"))
from homie_pushcart_v51.fixed_grasp import FixedGraspObserver


class FixedGraspGraphTests(unittest.TestCase):
    def inputs(self, observer):
        q = observer.initial_joint_pos.expand(observer.num_envs, -1).clone()
        dq = torch.zeros_like(q)
        v = torch.zeros(observer.num_envs, 3, device=q.device, dtype=q.dtype)
        gyro = torch.zeros_like(v)
        gravity = torch.zeros_like(v)
        gravity[:, 2] = -1.
        return q, dq, v, gyro, gravity

    def assert_outputs(self, actual, expected):
        for name in actual._outputs:
            with self.subTest(output=name):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        torch.testing.assert_close(actual.valid, expected.valid, rtol=0, atol=0)
        self.assertIs(actual.ready, actual.valid)

    def test_cpu_fallback_matches_eager_with_changed_inputs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                fast = FixedGraspObserver(3, dtype=dtype)
                eager = FixedGraspObserver(3, dtype=dtype)
                values = self.inputs(fast)
                for step in range(3):
                    values[0][:, 15] += .002 * step
                    values[1][:, 14] = .1 * step
                    values[2][:, 0] = .2 * step
                    values[3][:, 2] = .3 * step
                    fast.update_fast(*values)
                    eager.update(*values)
                    self.assert_outputs(fast, eager)
                self.assertFalse(fast.cuda_graph_ready)

    def test_bad_shape_uses_existing_validation(self):
        observer = FixedGraspObserver(3)
        values = list(self.inputs(observer))
        values[1] = values[1][:, :-1]
        with self.assertRaisesRegex(ValueError, "dq must be"):
            observer.update_fast(*values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_replay_changes_all_live_inputs_and_preserves_outputs(self):
        fast = FixedGraspObserver(8, device="cuda")
        eager = FixedGraspObserver(8, device="cuda")
        values = self.inputs(fast)
        output_ids = {name: id(getattr(fast, name)) for name in (*fast._outputs, "valid")}
        for step in range(5):
            # Replacing tensors, not only editing the original ones, detects
            # accidental replay against a stale input address or old sample.
            values = tuple(value.clone() for value in values)
            values[0][:, 15] += .001 * (step + 1)
            values[1][:, 14] = .04 * (step + 1)
            values[2][:, 0] = .1 * (step + 1)
            values[3][:, 2] = .2 * (step + 1)
            values[4][:, 0] = .01 * (step + 1)
            fast.update_fast(*values)
            eager.update(*values)
            self.assert_outputs(fast, eager)
            self.assertEqual(output_ids, {name: id(getattr(fast, name)) for name in output_ids})
            self.assertTrue(fast.cuda_graph_ready)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_reset_invalid_and_general_update_between_replays(self):
        fast = FixedGraspObserver(8, device="cuda")
        eager = FixedGraspObserver(8, device="cuda")
        values = self.inputs(fast)
        fast.update_fast(*values)
        eager.update(*values)
        ids = torch.tensor([1, 4], device="cuda")
        fast.reset(ids)
        eager.reset(ids)
        self.assert_outputs(fast, eager)
        fast.update_fast(*values)
        eager.update(*values)
        self.assert_outputs(fast, eager)
        self.assertTrue(bool(fast.valid.all()))

        mask = torch.ones(8, device="cuda", dtype=torch.bool)
        mask[3] = False
        fast.update(*values, timestamp_s=3.5, valid=mask)
        eager.update(*values, timestamp_s=3.5, valid=mask)
        self.assert_outputs(fast, eager)
        values[0][0, 0] = float("nan")
        values[1][1, 14] = float("inf")
        values[2][2, 0] = float("nan")
        values[3][3, 2] = float("inf")
        values[4][4] = 0.
        fast.update_fast(*values)
        eager.update(*values)
        self.assert_outputs(fast, eager)
        self.assertFalse(bool(fast.valid[:5].any()))
        self.assertTrue(bool(fast.valid[5:].all()))
        fast.reset()
        eager.reset()
        self.assert_outputs(fast, eager)
        values = self.inputs(fast)
        fast.update_fast(*values)
        eager.update(*values)
        self.assert_outputs(fast, eager)
        self.assertTrue(bool(fast.valid.all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_capture_and_replay_in_training_inference_mode(self):
        # RSL-RL collects rollouts under inference_mode. Environment output
        # buffers already exist before that scope; graph storage is created
        # inside it on the first sample.
        fast = FixedGraspObserver(8, device="cuda")
        eager = FixedGraspObserver(8, device="cuda")
        values = self.inputs(fast)
        for step in range(3):
            with torch.inference_mode():
                values = tuple(value.clone() for value in values)
                values[2][:, 0] = .3 * step
                values[1][:, 14] = .1 * step
                fast.update_fast(*values)
                eager.update(*values)
                self.assert_outputs(fast, eager)
                if step == 1:
                    ids = torch.tensor([2, 5], device="cuda")
                    fast.reset(ids)
                    eager.reset(ids)
                    self.assert_outputs(fast, eager)
        self.assertTrue(fast.cuda_graph_ready)
        # The graph input storage must remain writable when a later caller
        # leaves inference_mode; no stale sample or inference-tensor error.
        values = self.inputs(fast)
        values[2][:, 0] = -.4
        fast.update_fast(*values)
        eager.update(*values)
        self.assert_outputs(fast, eager)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_producer_current_stream_and_noncontiguous_inputs(self):
        fast = FixedGraspObserver(8, device="cuda")
        eager = FixedGraspObserver(8, device="cuda")
        caller = torch.cuda.current_stream()
        producer = torch.cuda.Stream()
        producer.wait_stream(caller)
        with torch.cuda.stream(producer):
            values = self.inputs(fast)
            values = tuple(torch.stack((value, value), -1)[..., 0] for value in values)
            self.assertFalse(values[0].is_contiguous())
            fast.update_fast(*values)
            values[2][:, 0] = .8
            fast.update_fast(*values)
            eager.update(*values)
        caller.wait_stream(producer)
        self.assert_outputs(fast, eager)


if __name__ == "__main__":
    unittest.main()
