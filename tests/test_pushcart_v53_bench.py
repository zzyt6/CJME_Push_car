"""V5.3 must use its own observation/velocity pipeline through the real bench."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'HomieRL')]
from pushcart_bench.policies import make_policy, parse_policy_spec
from pushcart_bench.config_io import load_config
from pushcart_bench.runner import Condition, run_step

CKPT=ROOT/'logs/rsl_rl/homie_pushcart_v53/2026-09-28_00-37-10_v53_final_20k/model_15000.pt'


class V53BenchTests(unittest.TestCase):
    def test_specs_preserve_windows_paths_and_explicit_scene(self):
        p=parse_policy_spec(r'v53_new:E:\runs\model_15000.pt')
        self.assertEqual(p['checkpoint'],r'E:\runs\model_15000.pt')
        self.assertEqual(p['scene_version'],'four_spheres')
        self.assertEqual(parse_policy_spec('v53_original')['checkpoint'],'latest')

    @unittest.skipUnless(CKPT.is_file(),'Local V5.3 checkpoint required')
    def test_real_bench_step_and_telemetry_in_each_scene(self):
        torch.set_num_threads(1)
        for tag,scene in [('v53_original','original'),('v53_new','four_spheres')]:
            policy=make_policy(f'{tag}:{CKPT}',level=13)
            result=run_step(policy,dict(name='v53_bridge',cmd0=[0.,0.,0.],cmd1=[.2,0.,0.],pre_s=.2,seconds=1.4,signal='vx'),
                            Condition(payload_kg=15.),load_config())
            self.assertEqual(result['policy']['kind'],'v53')
            self.assertEqual(result['policy']['scene_version'],scene)
            self.assertEqual(result['policy']['frame_dim'],156)
            tel=policy.telemetry()
            self.assertTrue(tel['quality_ready'])
            np.testing.assert_allclose([tel['cart_est_vx'],tel['cart_est_vy'],tel['cart_est_wz']],policy.runner.pred_cart_velocity)
            self.assertTrue(np.isfinite(list(tel[k] for k in ('est_left_fx','est_right_fx','cart_response_error_cost'))).all())
            self.assertAlmostEqual(policy.cart_mass_total(),29.4,places=6)
            self.assertIsNone(policy.failure())


if __name__=='__main__':unittest.main()
