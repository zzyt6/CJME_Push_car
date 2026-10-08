"""V7.1 (revision 10): push-cart-tuned pelvis-velocity head, HOMIE-estimate masking, config, backward loading."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
V7_RUNS = ROOT / "logs/rsl_rl/homie_pushcart_v7"
OLD = [V7_RUNS / "2026-10-07_04-28-54_v7_persist04/model_17200.pt",                     # revision 8
       V7_RUNS / "2026-10-08_02-16-36_v7r9_rollscrub_v6track_hwgrip_20k/model_12000.pt"]  # revision 9


def make_actor(head):
    from homie_pushcart.hl_policy import import_rsl_rl5
    import_rsl_rl5()
    from tensordict import TensorDict
    from homie_pushcart_v7.interaction_policy import InteractionActor
    probe = TensorDict({"policy": torch.zeros(4, 605), "estimator": torch.zeros(4, 462)}, batch_size=[4])
    dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    try:
        from rsl_rl.models import MLPModel  # noqa: F401
        actor = InteractionActor(probe, {"actor": ["policy", "estimator"]}, "actor", 18,
                                 distribution_cfg=dist, robot_velocity_head=head)
    except TypeError:
        actor = InteractionActor(probe, {"actor": ["policy", "estimator"]}, "actor", 18, robot_velocity_head=head)
    return actor, TensorDict


class RobotVelocityHead(unittest.TestCase):
    def test_initialised_from_homie_encoder(self):
        from homie_pushcart.lowlevel import HimLowLevelPolicy, frozen_checkpoint
        from homie_pushcart_v7.interaction_policy import homie_history
        actor, TensorDict = make_actor(True)
        low = HimLowLevelPolicy(frozen_checkpoint(), "cpu")
        est = torch.randn(4, 462)
        hist = homie_history(est)
        self.assertEqual(hist.shape, (4, 456))
        with torch.no_grad():
            ours = actor.predict_robot_velocity({"estimator": est})
            homie, _ = low.model.estimator(hist)
        torch.testing.assert_close(ours, homie, rtol=0, atol=1e-6)
        # frames [:76] of every 77-wide estimator frame, oldest -> newest
        torch.testing.assert_close(hist.reshape(4, 6, 76), est.reshape(4, 6, 77)[..., :76])

    def test_actor_dims_metadata_and_export(self):
        actor, TensorDict = make_actor(True)
        md = actor.metadata()
        self.assertEqual(md["schema"], "v7_1_v32_policy605_him462_robotvel3_actor655_critic695_v1")
        self.assertEqual(md["actor_mlp"][0], 655)
        obs = TensorDict({"policy": torch.randn(4, 605), "estimator": torch.randn(4, 462)}, batch_size=[4])
        self.assertEqual(actor.unnormalized_actor_input(obs).shape, (4, 655))
        actor.eval()
        with torch.no_grad():
            a = actor(obs)
            b = actor.as_jit()(obs["policy"], obs["estimator"])
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        estimator_ids = {id(p) for p in actor.estimator_parameters()}
        self.assertTrue(all(id(p) in estimator_ids for p in actor.robot_velocity.parameters()))
        self.assertTrue(all(not p.requires_grad for p in actor.robot_velocity_homie.parameters()))

    def test_v7_actor_unchanged_without_head(self):
        actor, _ = make_actor(False)
        self.assertEqual(actor.metadata()["schema"], "v7_v32_policy605_him462_actor652_critic695_v2")
        self.assertFalse(hasattr(actor, "robot_velocity"))


class MaskAndConfig(unittest.TestCase):
    def test_mask_slots(self):
        from homie_pushcart_v7.observation import mask_homie_velocity
        x = torch.randn(3, 620)
        f = mask_homie_velocity(x).reshape(3, 5, 124)
        self.assertTrue((f[..., 95:100] == 0).all())
        torch.testing.assert_close(f[..., 100], x.reshape(3, 5, 124)[..., 100])   # gyro-based yaw difference kept

    def test_config(self):
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        c = PushCartTaskCfg()
        self.assertEqual((c.implementation_revision, c.release), (10, "7.1"))
        self.assertEqual(c.commands.wz_max, 0.6)
        self.assertEqual(c.rewards.scales.hl_action_clip, 0.0)
        self.assertEqual(c.cart_physics.yaw_damping_range, [0.0, 10.0])
        self.assertTrue(c.robot_velocity.enabled and c.robot_velocity.replace_homie_estimate)

    def test_agent_entropy_and_head(self):
        from homie_pushcart_v7.lab_config import PushCartRunnerCfg
        r = PushCartRunnerCfg()
        self.assertEqual(r.algorithm.entropy_coef, 0.005)
        self.assertTrue(r.actor.robot_velocity_head)


class OldCheckpoints(unittest.TestCase):
    def test_revision_8_and_9_still_load(self):
        from homie_pushcart_v7.hl_policy import HighLevelPolicy
        from homie_pushcart_v7.task_config import task_cfg_from_run
        for ck in OLD:
            if not ck.is_file():
                self.skipTest(f"missing {ck}")
            pol = HighLevelPolicy(ck)
            self.assertFalse(pol.actor.robot_velocity_head)
            task = task_cfg_from_run(ck, verbose=False)
            self.assertEqual(task.release, "7.0")
            self.assertFalse(task.robot_velocity.replace_homie_estimate)
            self.assertEqual(task.cart_physics.yaw_damping_range, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
