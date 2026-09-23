"""V4 runner timing, report validity and condition forwarding contracts."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'HomieRL', ROOT / 'HomieRL/rsl_rl'):
    sys.path.insert(0, str(p))

from pushcart_bench.runner import Condition, run_step
from pushcart_bench.report import _cell, _short


class ReferencePolicy:
    """Ideal plant tracking a causal reference, without simulator dependence."""
    dt = .02

    def reset(self, **kwargs):
        self.ref = np.zeros(3)
        self.xy = np.zeros(2)
        self.commands = []

    def step(self, cmd):
        self.commands.append(np.asarray(cmd).copy())
        self.ref += (1 - np.exp(-self.dt / .4)) * (cmd - self.ref)
        self.xy += self.ref[:2] * self.dt

    def pose(self):
        return [*self.xy, 0.]

    def velocity(self):
        return self.ref

    def failure(self):
        return None

    def telemetry(self):
        return dict(zip(('cart_ref_vx', 'cart_ref_vy', 'cart_ref_wz'), self.ref))

    def describe(self):
        return {'name': 'ideal_reference', 'kind': 'test'}


class V4IntegrationTests(unittest.TestCase):
    def test_command_switch_uses_exact_ticks_and_reference_tracking_is_distinct(self):
        policy = ReferencePolicy()
        step = dict(cmd0=[0., 0., 0.], cmd1=[.6, 0., 0.], pre_s=5., seconds=8., signal='vx')
        result = run_step(policy, step, Condition(), {'trace_stride': 1})
        first_new = np.flatnonzero(np.asarray(policy.commands)[:, 0] > 0)[0]
        self.assertEqual(first_new, 250)
        self.assertEqual(result['metrics']['command_switch_s'], 5.)
        self.assertEqual(result['trace']['cmd'][249][0], 0.)
        self.assertEqual(result['trace']['cmd'][250][0], .6)
        m = result['metrics']
        self.assertGreater(m['reference_rise_time'], .7)
        self.assertAlmostEqual(m['reference_tracking_full_rmse'], 0.)
        self.assertAlmostEqual(m['reference_tracking_transient_iae'], 0.)

    def test_unconfirmed_stop_cannot_have_a_finite_stop_distance(self):
        policy = ReferencePolicy()
        step = dict(cmd0=[.6, 0., 0.], cmd1=[0., 0., 0.], pre_s=5., seconds=5.1,
                    signal='vx', stopping=True)
        m = run_step(policy, step, Condition(), {})['metrics']
        self.assertFalse(m['stopping_confirmed'])
        self.assertIsNone(m['stopping_distance'])

    def test_boolean_confirmation_is_a_rate_and_legacy_distance_is_not_relabelled(self):
        runs = [{'kind': 'step', 'metrics': {'stopping_confirmed': x}} for x in (True, False)]
        self.assertEqual(_cell(runs, 'stopping_confirmed', None), '50 % (1/2)')
        old = [{'kind': 'step', 'metrics': {'stopping_distance': .2}}]
        self.assertEqual(_cell(old, 'stopping_distance', 3), '-')

    def test_level_zero_is_not_replaced_by_default_level(self):
        from pushcart_bench.experiments import _policies
        args = SimpleNamespace(level=0, policies=['passthrough'], homie_checkpoint=None)
        with patch('pushcart_bench.experiments.make_policy') as factory:
            _policies(args, {'defaults': {'level': 17}})
        self.assertEqual(factory.call_args.kwargs['level'], 0)

    def test_chart_labels_distinguish_runs_at_the_same_iteration(self):
        old = _short('two_stage:2026-09-20_21-22-19_v3_1_combined_scratch/model_19999')
        new = _short('two_stage:2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999')
        self.assertNotEqual(old, new)
        self.assertIn('combined', old)
        self.assertIn('speed_quality', new)

    def test_v4_gate_is_exported_as_required_runtime_state(self):
        # The network keeps its shape, but the deployment runner needs the same gate.
        import tempfile
        from homie_pushcart.export import export_checkpoint
        from homie_pushcart.cart_quality import apply_quality_profile
        from homie_pushcart.task_config import PushCartTaskCfg
        from types import SimpleNamespace
        import torch
        cfg = apply_quality_profile(PushCartTaskCfg(), 'v4')
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_normalizer = torch.nn.Identity()
                self.mlp = torch.nn.Linear(620, 18)
        policy = SimpleNamespace(actor=Actor(), num_actor_obs=620, num_actions=18, iteration=0)
        with tempfile.TemporaryDirectory() as output, \
                patch('homie_pushcart.export.task_cfg_from_run', return_value=cfg), \
                patch('homie_pushcart.hl_policy.HighLevelPolicy', return_value=policy):
            interface = export_checkpoint(Path('model_0.pt'), Path(output))
        runtime = interface['action_to_command']['idle_correction']
        self.assertTrue(runtime['config']['enabled'])
        self.assertFalse(runtime['network_contains_gate'])
        self.assertEqual(runtime['config']['max_gate'], .25)
        self.assertEqual(interface['actor_obs_dim'], 620)
        self.assertEqual(interface['self_check_max_abs_diff'], 0.)


if __name__ == '__main__':
    unittest.main()
