"""V8: configuration, teacher Actor base (zero leg residual at start), ROA teacher / RMA student, final models."""
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
        from homie_pushcart_v8.lab_config import TeacherRunnerCfg, PushCartEnvCfg, ROATeacherRunnerCfg
        r, e = TeacherRunnerCfg(), PushCartEnvCfg()
        self.assertEqual(r.algorithm.entropy_coef, 0.005)
        self.assertEqual(r.obs_groups, {"actor": ["policy", "privileged"], "critic": ["critic"]})
        self.assertEqual((e.action_space, e.observation_space, e.state_space), (30, 630, 692))
        roa = ROATeacherRunnerCfg()
        self.assertEqual(roa.obs_groups, {"actor": ["policy", "privileged", "history"], "critic": ["critic"]})
        self.assertEqual((roa.actor.latent_dim, roa.actor.encoder_hidden_dims), (20, [256, 128]))


class LegResidualTanhTests(unittest.TestCase):
    """2026-10-09: smooth bound (0.3 * tanh), 40 ms low-pass, excess penalty, fixed gate; first teacher stays clip."""

    def test_defaults_and_mapping(self):
        from homie_pushcart_v8.task_config import (PushCartTaskCfg, leg_residual_bound, leg_residual_filter_alpha,
                                                   leg_residual_map, validate_v8)
        c = PushCartTaskCfg()
        validate_v8(c)
        lr = c.v8.leg_residual
        self.assertEqual((lr.mode, lr.range, lr.filter_tau_s, lr.excess_threshold), ("tanh", 0.3, 0.04, 2.0))
        self.assertEqual(c.rewards.scales.leg_residual_excess, -0.2)
        self.assertAlmostEqual(leg_residual_bound(c) * 0.25, 0.075)                 # rad
        u = torch.tensor([-10., -1., 0., 0.1, 1., 10.])
        out = leg_residual_map(u, c)
        self.assertTrue(torch.all(out.abs() <= 0.3 + 1e-6))
        self.assertTrue(torch.all(out[1:] > out[:-1]))                                # strictly monotonic, no flat part
        torch.testing.assert_close(out[3], torch.tensor(0.3 * torch.tanh(torch.tensor(0.1)).item()))
        self.assertAlmostEqual(leg_residual_filter_alpha(c, 0.02), 1 - torch.exp(torch.tensor(-0.5)).item(), places=6)

    def test_clip_mode_for_first_teacher(self):
        from homie_pushcart_v8.task_config import PushCartTaskCfg, leg_residual_filter_alpha, leg_residual_map
        c = PushCartTaskCfg()
        c.v8.leg_residual.mode, c.v8.leg_residual.filter_tau_s = "clip", 0.0
        torch.testing.assert_close(leg_residual_map(torch.tensor([-5., 0.1, 5.]), c), torch.tensor([-0.2, 0.1, 0.2]))
        self.assertEqual(leg_residual_filter_alpha(c, 0.02), 1.0)

    def test_teacher_gate_fixed_and_leg_std(self):
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from tensordict import TensorDict
        from homie_pushcart_v8.teacher_policy import TeacherActor
        from homie_pushcart_v8.lab_config import TeacherRunnerCfg
        cfg = TeacherRunnerCfg().actor
        probe = TensorDict({"policy": torch.zeros(4, 630), "privileged": torch.zeros(4, 62)}, batch_size=[4])
        dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
        actor = TeacherActor(probe, {"actor": ["policy", "privileged"]}, "actor", 30, distribution_cfg=dist,
                             leg_gate_init=cfg.leg_gate_init, leg_init_std=cfg.leg_init_std,
                             leg_gate_learnable=cfg.leg_gate_learnable)
        gate = actor.mlp[1].alpha
        self.assertEqual((float(gate), gate.requires_grad), (1.0, False))
        torch.testing.assert_close(actor.distribution.std_param.detach()[18:], torch.full((12,), 0.1))
        with torch.no_grad():
            self.assertTrue(torch.all(actor(TensorDict({"policy": torch.randn(3, 630), "privileged": torch.randn(3, 62)},
                                                       batch_size=[3]))[:, 18:] == 0))

def make_rma():
    from homie_pushcart.hl_policy import import_rsl_rl5
    import_rsl_rl5()
    from tensordict import TensorDict
    from homie_pushcart_v8.rma_policy import RMATeacherActor
    probe = TensorDict({"policy": torch.zeros(4, 630), "privileged": torch.zeros(4, 62), "history": torch.zeros(4, 4032)},
                       batch_size=[4])
    dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
    actor = RMATeacherActor(probe, {"actor": ["policy", "privileged", "history"]}, "actor", 30, distribution_cfg=dist)
    obs = TensorDict({"policy": torch.randn(8, 630), "privileged": torch.randn(8, 62), "history": torch.randn(8, 4032)},
                     batch_size=[8])
    return actor, obs


class ROATeacherStudentTests(unittest.TestCase):
    def test_latent_only_policy_and_modes(self):
        actor, obs = make_rma()
        self.assertEqual(actor.latent_dim, 20)
        self.assertEqual(actor.mlp[0][0].in_features, 630 + 20)            # no raw explicit input
        with torch.no_grad():
            torch.nn.init.normal_([m for m in actor.mlp[0].modules() if isinstance(m, torch.nn.Linear)][-1].weight)
            a_priv = actor(obs)
            actor.use_estimate = True
            a_est = actor(obs)
        self.assertEqual(a_priv.shape, (8, 30))
        self.assertFalse(torch.allclose(a_priv, a_est))                     # z_phi replaces z_mu
        # exports: teacher (policy, privileged) and deployable student (policy, history)
        actor.use_estimate = False
        torch.testing.assert_close(actor.as_jit()(obs["policy"], obs["privileged"]), a_priv)
        actor.use_estimate = True
        torch.testing.assert_close(actor.as_jit()(obs["policy"], obs["history"]), a_est)

    def test_roa_schedules(self):
        from types import SimpleNamespace
        from homie_pushcart_v8.rma_policy import V8ROAPPO
        alg = SimpleNamespace(adapt_only=False, adapt_interval=20, roa_lambda_max=1.0, roa_start=5000, roa_ramp=5000)
        est = [k for k in range(60) if V8ROAPPO._estimate_rollout(alg, k)]
        self.assertEqual(est, [19, 39, 59])                                  # every H-th rollout acts on z_phi
        lam = lambda k: V8ROAPPO.roa_lambda(SimpleNamespace(**vars(alg), n_updates=k))      # noqa: E731
        self.assertEqual((lam(0), lam(5000), lam(7500), lam(10000), lam(12000)), (0.0, 0.0, 0.5, 1.0, 1.0))
        alg.adapt_only = True                                                # RMA phase 2
        self.assertTrue(all(V8ROAPPO._estimate_rollout(alg, k) for k in range(5)))
        self.assertEqual(lam(8000), 0.0)

    def test_student_stages_schedules(self):
        from types import SimpleNamespace
        from homie_pushcart_v8.rma_policy import V8ROAPPO
        srl = SimpleNamespace(student_rl=True, adapt_only=False, adapt_interval=20)
        self.assertTrue(all(V8ROAPPO._estimate_rollout(srl, k) for k in range(5)))     # student RL acts on z_phi
        lam = SimpleNamespace(adapt_only=False, roa_lambda_max=0.1, roa_start=0, roa_ramp=1, n_updates=6000)
        self.assertAlmostEqual(V8ROAPPO.roa_lambda(lam), 0.1)                          # the final teacher's cap

    def test_velocity_head_and_explicit_upper_gains(self):
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from tensordict import TensorDict
        from homie_pushcart_v8.rma_policy import RMATeacherActor
        from homie_pushcart_v8.task_config import PushCartTaskCfg, upper_drive_gains
        probe = TensorDict({"policy": torch.zeros(4, 630), "privileged": torch.zeros(4, 62), "history": torch.zeros(4, 4032)},
                           batch_size=[4])
        dist = {"class_name": "GaussianDistribution", "init_std": 0.4, "std_type": "scalar"}
        actor = RMATeacherActor(probe, {"actor": ["policy", "privileged", "history"]}, "actor", 30, distribution_cfg=dist,
                                velocity_head=True)
        v, z = actor.adaptation(torch.randn(3, 4032))
        self.assertEqual((v.shape, z.shape), ((3, 5), (3, 20)))
        actor.use_estimate = True
        e = actor.as_jit().explicit(torch.randn(3, 4032))
        self.assertEqual(e.shape, (3, 19))
        self.assertTrue(torch.all(e[:, [3] + list(range(6, 19))] == 0))
        c = PushCartTaskCfg()
        self.assertEqual(c.v8.upper_drive.mode, "implicit")                       # saved V8.0 runs load unchanged
        kp, kd = upper_drive_gains(c, ["waist_yaw_joint", "left_shoulder_yaw_joint", "right_elbow_joint", "left_wrist_pitch_joint"])
        self.assertEqual((kp, kd), ([300.0, 50.0, 50.0, 20.0], [3.0, 2.0, 2.0, 1.0]))   # Unitree deploy_real g1.yaml

    def test_retired_estimate_noise_rejected(self):
        from homie_pushcart_v8.task_config import PushCartTaskCfg, validate_v8
        c = PushCartTaskCfg()
        c.v8.estimate_noise.enabled = True
        with self.assertRaises(ValueError):
            validate_v8(c)


FINAL = ROOT / "logs/rsl_rl/homie_pushcart_v8"
FINAL_STUDENT = FINAL / "2026-10-12_07-56-35_v8_fp_rl_armdr800/model_200.pt"
FINAL_TEACHER = FINAL / "2026-10-12_05-58-10_v8_roa_lam01_armdr_from6000/model_800.pt"


@unittest.skipUnless(FINAL_STUDENT.is_file() and FINAL_TEACHER.is_file(), "final V8 checkpoints not present")
class FinalModelTests(unittest.TestCase):
    def test_final_student_and_teacher_load(self):
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from homie_pushcart_v8.student_policy import is_student_run, load_student
        from homie_pushcart_v8.teacher_policy import load_teacher
        from homie_pushcart_v8.training import homie_velocity_file
        self.assertTrue(is_student_run(FINAL_STUDENT))
        self.assertFalse(is_student_run(FINAL_TEACHER))
        s, t = load_student(FINAL_STUDENT), load_teacher(FINAL_TEACHER)
        with torch.no_grad():
            self.assertEqual(s(torch.zeros(2, 630), torch.zeros(2, 4032)).shape, (2, 30))
            self.assertEqual(t(torch.zeros(2, 630), torch.zeros(2, 62)).shape, (2, 30))
        self.assertEqual(homie_velocity_file(FINAL_STUDENT).name, "homie_velocity_estimator_6800.pt")
        self.assertEqual(homie_velocity_file(FINAL_TEACHER).name, "homie_velocity_estimator_6800.pt")


V81_STUDENT = FINAL / "2026-10-13_04-57-30_v8_fp_p2_v81m6000/model_299.pt"
V81_TEACHER = FINAL / "2026-10-12_20-52-19_v8_1_roa_teacher_8k/model_6000.pt"


@unittest.skipUnless(V81_STUDENT.is_file() and V81_TEACHER.is_file(), "V8.1 checkpoints not present")
class V81ModelTests(unittest.TestCase):
    def test_v81_student_teacher_and_settings(self):
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from homie_pushcart_v8.student_policy import is_student_run, load_student
        from homie_pushcart_v8.task_config import task_cfg_from_run
        from homie_pushcart_v8.teacher_policy import load_teacher
        from homie_pushcart_v8.training import homie_velocity_file
        from pushcart_bench.policies_v8 import resolve_v8_checkpoint
        self.assertTrue(is_student_run(V81_STUDENT))
        s, t = load_student(V81_STUDENT), load_teacher(V81_TEACHER)
        with torch.no_grad():
            self.assertEqual(s(torch.zeros(2, 630), torch.zeros(2, 4032)).shape, (2, 30))
            self.assertEqual(t(torch.zeros(2, 630), torch.zeros(2, 62)).shape, (2, 30))
            self.assertTrue(s.velocity_head)
            self.assertEqual(s.explicit(torch.zeros(2, 4032)).shape, (2, 19))
        v8 = task_cfg_from_run(V81_STUDENT, verbose=False).v8
        self.assertEqual((v8.upper_drive.mode, list(v8.arm_drive.scale_range), v8.arm_drive.enabled),
                         ("explicit", [0.8, 1.2], True))
        self.assertEqual(homie_velocity_file(V81_STUDENT).name, "homie_velocity_estimator_6000.pt")
        self.assertEqual(Path(resolve_v8_checkpoint("latest")).resolve(), V81_STUDENT.resolve())


if __name__ == "__main__":
    unittest.main()
