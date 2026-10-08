"""V8 teacher phase: configuration, teacher Actor (zero leg residual at start, small leg std), export."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def make_teacher():
    from homie_pushcart.hl_policy import import_rsl_rl5
    import_rsl_rl5()
    from tensordict import TensorDict
    from homie_pushcart_v8.teacher_policy import TeacherActor
    probe = TensorDict({"policy": torch.zeros(4, 630), "privileged": torch.zeros(4, 62)}, batch_size=[4])
    dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
    actor = TeacherActor(probe, {"actor": ["policy", "privileged"]}, "actor", 30, distribution_cfg=dist)
    return actor, TensorDict


class TeacherActorTests(unittest.TestCase):
    def test_leg_residual_starts_at_zero_and_small_std(self):
        actor, TensorDict = make_teacher()
        obs = TensorDict({"policy": torch.randn(8, 630), "privileged": torch.randn(8, 62)}, batch_size=[8])
        with torch.no_grad():
            out = actor(obs)
        self.assertEqual(out.shape, (8, 30))
        self.assertTrue(torch.all(out[:, 18:] == 0))
        self.assertTrue(torch.any(out[:, :18] != 0))
        std = actor.distribution.std_param.detach()
        torch.testing.assert_close(std[18:], torch.full((12,), 0.05))
        torch.testing.assert_close(std[:18], torch.full((18,), 0.4))

    def test_gradient_reaches_leg_rows_and_export_matches(self):
        actor, TensorDict = make_teacher()
        obs = TensorDict({"policy": torch.randn(8, 630), "privileged": torch.randn(8, 62)}, batch_size=[8])
        actor(obs)[:, 18:].sum().backward()
        last = [m for m in actor.mlp[0].modules() if isinstance(m, torch.nn.Linear)][-1]
        self.assertGreater(last.weight.grad[18:].abs().sum().item(), 0)       # zero-init rows still learn
        actor.zero_grad()
        actor.eval()
        with torch.no_grad():
            a = actor(obs)
            b = actor.as_jit()(obs["policy"], obs["privileged"])
        torch.testing.assert_close(a, b)
        md = actor.metadata()
        self.assertEqual((md["actions"], md["latent"], md["explicit"]), (30, 24, 19))


class ConfigTests(unittest.TestCase):
    def test_v8_config_on_v71_base(self):
        from homie_pushcart_v8.task_config import PushCartTaskCfg, validate_v8, num_critic_obs, num_policy_obs
        from homie_pushcart_v7.task_config import validate_task_schema
        c = PushCartTaskCfg()
        validate_v8(c)
        validate_task_schema(c)                           # the V7 environment it extends must accept it
        self.assertEqual((c.release, c.implementation_revision), ("8.0", 10))
        self.assertEqual((num_policy_obs(), num_critic_obs()), (630, 692))
        self.assertEqual(c.commands.wz_max, 0.6)
        self.assertTrue(c.v8.leg_residual.enabled)
        sc = c.rewards.scales
        self.assertEqual((sc.feet_slip, sc.leg_residual, sc.hl_action_clip), (-0.25, -1.0, 0.0))

    def test_runner_cfg(self):
        from homie_pushcart_v8.lab_config import TeacherRunnerCfg, PushCartEnvCfg
        r, e = TeacherRunnerCfg(), PushCartEnvCfg()
        self.assertEqual(r.algorithm.entropy_coef, 0.005)
        self.assertEqual(r.obs_groups, {"actor": ["policy", "privileged"], "critic": ["critic"]})
        self.assertEqual((e.action_space, e.observation_space, e.state_space), (30, 630, 692))


if __name__ == "__main__":
    unittest.main()


def make_student():
    from homie_pushcart.hl_policy import import_rsl_rl5
    import_rsl_rl5()
    from tensordict import TensorDict
    from homie_pushcart_v8.student_policy import StudentActor
    probe = TensorDict({"policy": torch.zeros(4, 630), "history": torch.zeros(4, 4032)}, batch_size=[4])
    dist = {"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"}
    return StudentActor(probe, {"student": ["policy", "history"]}, "student", 30, distribution_cfg=dist), TensorDict


class StudentTests(unittest.TestCase):
    def test_body_copy_freeze_and_export(self):
        teacher, TensorDict = make_teacher()
        student, _ = make_student()
        with torch.no_grad():                                  # make the teacher body non-trivial
            for p in teacher.mlp.parameters():
                p.add_(0.01 * torch.randn_like(p))
        student.load_body_from(teacher)
        for (n, a), (_, b) in zip(teacher.mlp.state_dict().items(), student.mlp.state_dict().items()):
            torch.testing.assert_close(a, b, msg=n)
        self.assertTrue(all(not p.requires_grad for p in student.mlp.parameters()))
        self.assertTrue(all(p.requires_grad for p in student.adaptation.parameters()))
        obs = TensorDict({"policy": torch.randn(6, 630), "history": torch.randn(6, 4032)}, batch_size=[6])
        explicit, z = student.context(obs)
        self.assertEqual((explicit.shape, z.shape), ((6, 19), (6, 24)))
        student.eval()
        with torch.no_grad():
            a = student(obs)
            b = student.as_jit()(obs["policy"], obs["history"])
        self.assertEqual(a.shape, (6, 30))
        torch.testing.assert_close(a, b)
        # same body + same context => same action as the teacher would give
        with torch.no_grad():
            x = teacher.obs_normalizer(torch.cat((obs["policy"], explicit), -1))
            t = teacher.distribution.deterministic_output(teacher.mlp(torch.cat((x, z), -1)))
        torch.testing.assert_close(a, t)
