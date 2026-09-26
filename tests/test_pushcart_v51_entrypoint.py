"""Version/schema isolation and concrete launcher selection for V5.1."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v51 import training
from homie_pushcart_v51.observation import FRAME_DIM, PRIVILEGED_DIM
from homie_pushcart_v51.task_config import PushCartTaskCfg, num_actor_obs, num_critic_obs


class EntryTests(unittest.TestCase):
    def test_scratch_selects_new_profile_and_preserves_explicit_aux_ablation(self):
        with patch.object(training, '_lab_scripts', return_value=Path('official')), \
             patch.object(training, '_run', return_value=0) as call:
            training.train(['--num_envs', '4096', '--level', '0', 'agent.algorithm.aux_coef=0'])
        args = call.call_args.args[1]
        self.assertIn('Homie-G1-PushCart-V51-Direct-v0', args)
        self.assertIn('homie_pushcart_v51', args)
        self.assertIn('env.task.reaction.enabled=false', args)
        self.assertIn('agent.algorithm.aux_coef=0', args)
        self.assertIn('env.curriculum_level=0', args)

    def test_legacy_experiments_cannot_be_overwritten(self):
        for name in ('homie_pushcart', 'homie_pushcart_v5', '../homie_pushcart_v51'):
            with self.assertRaises(ValueError):
                training._validate_experiment(name)

    def test_dimensions_and_default_physics_dependencies(self):
        cfg = PushCartTaskCfg()
        self.assertEqual((FRAME_DIM, PRIVILEGED_DIM, num_actor_obs(cfg), num_critic_obs(cfg)), (124, 46, 2480, 170))
        self.assertEqual(cfg.version, 51)
        self.assertFalse(cfg.reaction.enabled)
        self.assertFalse(cfg.diagnostics.full_dynamics)
        self.assertEqual(cfg.env.num_hl_actions, 18)

    def test_preserved_v4_snapshot(self):
        snapshot = json.loads((ROOT / 'HomieRL/homie_pushcart_v5/legacy_snapshot.json').read_text(encoding='utf-8'))
        for name, checksum in snapshot['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), checksum, name)


if __name__ == '__main__':
    unittest.main()
