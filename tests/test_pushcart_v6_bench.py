"""V6 bench binding preserves the baseline control path and rejects scene choices."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"HomieRL")]
import numpy as np
import torch
import yaml
from homie_pushcart.hl_policy import import_rsl_rl5
import_rsl_rl5()
from tensordict import TensorDict
from homie_pushcart_v6.baseline import agent_baseline,config_dict
from homie_pushcart_v6.task_config import PushCartTaskCfg
from homie_pushcart_v6.interaction_policy import InteractionActor
from pushcart_bench.policies import PassthroughPolicy,make_policy,parse_policy_spec
from pushcart_bench.policies_v6 import V6Policy


class V6BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory=tempfile.TemporaryDirectory()
        path=Path(cls.directory.name)
        (path/"params").mkdir()
        cfg=agent_baseline()
        cfg["actor"]["class_name"]="homie_pushcart_v6.interaction_policy:InteractionActor"
        cfg["obs_groups"]={"actor":["policy","estimator"],"critic":["critic"]}
        (path/"params/env.yaml").write_text(yaml.safe_dump({"task":config_dict(PushCartTaskCfg())}),encoding="utf-8")
        (path/"params/agent.yaml").write_text(yaml.safe_dump(cfg),encoding="utf-8")
        acfg=dict(cfg["actor"]);acfg.pop("class_name")
        actor=InteractionActor(TensorDict({"policy":torch.zeros(1,605),"estimator":torch.zeros(1,462)},batch_size=[1]),
                               cfg["obs_groups"],"actor",18,**acfg)
        for parameter in actor.mlp.parameters():
            parameter.data.zero_()
        cls.checkpoint=path/"model_0.pt"
        torch.save({"actor_state_dict":actor.state_dict(),"v6_network_metadata":actor.metadata(),"iter":0},cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_zero_residual_v6_is_bit_exact_original_physics_and_control(self):
        v6=V6Policy(self.checkpoint)
        baseline=PassthroughPolicy(17)
        for policy in (v6,baseline):policy.reset(payload_kg=29.,seed=1)
        for tick in range(64):
            command=np.array([.3,0.,.1 if tick>=32 else 0.])
            for policy in (v6,baseline):policy.step(command)
            for key in ("qpos","qvel","ctrl"):
                np.testing.assert_array_equal(getattr(v6.env.data,key),getattr(baseline.env.data,key))
            self.assertEqual(v6.runner.hl_obs.shape,(605,))
            self.assertEqual(tuple(v6.runner.estimator_obs.shape),(1,462))
        telemetry=v6.telemetry()
        self.assertTrue(np.isfinite([telemetry[k] for k in ("cart_est_vx","cart_est_vy","cart_est_wz")]).all())
        self.assertEqual(v6.describe()["physics"]["foot_contact"]["spheres_per_foot"],4)

    def test_formal_policy_spec_preserves_windows_path_and_rejects_other_scenes(self):
        spec="v6:E:\\runs\\model_1.pt"
        self.assertEqual(parse_policy_spec(spec),{"kind":"v6","checkpoint":"E:\\runs\\model_1.pt"})
        with self.assertRaisesRegex(ValueError,"fixed legacy"):
            make_policy(f"v6:{self.checkpoint}",scene_version="new")


if __name__=="__main__":unittest.main()
