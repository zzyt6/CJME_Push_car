"""Bounded integration checks for V5.1 sensing, privileged loads, reset and PPO inputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--check_dynamics_cache", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
try:
    import torch
    from homie_pushcart_v51.env_smoke import make_configs
    from homie_pushcart_v51.task_env import PushCartEnv
    from homie_pushcart_v51.observation import PRIVILEGED_FIELDS

    torch.manual_seed(25)
    cfg, task = make_configs(args.num_envs, True, 17, 3.0)
    task.curriculum.enabled = False
    task.randomization.robot_push = False
    task.randomization.cart_force_pulse = False
    task.randomization.wheel_stall_prob_per_s = 0.
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=17)
    env.fixed_command, env.fixed_payload_kg = [.2, 0., .1], 5.
    env.reset()
    assert not env.full_dynamics and not env.physics_samples and not env._ground_contact_views
    counts = dict(valid_fk=0, valid_label=0, resets=0, physical_external_visible=0)
    maxima = dict(fk_position_error_m=0., label_normalized=0., peak_hand_N=0.)
    cache_checks = 0
    offsets, cursor = {}, 124
    for name, size in PRIVILEGED_FIELDS:
        offsets[name] = slice(cursor, cursor + size)
        cursor += size
    for step in range(args.steps):
        if step == 35:
            env.inject_cart_wrench([12., 0., 0.], [0., 0., .5], .20, env_ids=[0])
        if step == 80:
            env.fixed_command = [0., 0., 0.]
            env.cmd_task[:] = 0.
        obs, critic, reward, done, extra = env.step(torch.zeros(args.num_envs, 18, device=args.device))
        assert obs.shape == (args.num_envs, 2480) and critic.shape == (args.num_envs, 170)
        assert env.interaction_target.shape == (args.num_envs, 4)
        assert torch.isfinite(obs).all() and torch.isfinite(critic).all() and torch.isfinite(reward).all()
        assert not env.ground_wrench_valid.any()
        counts['valid_fk'] += int(env.kinematic_valid.sum())
        counts['valid_label'] += int(env.interaction_target[:, 3].sum())
        counts['resets'] += int(done.sum())
        counts['physical_external_visible'] += int((critic[:, offsets['cart_external_wrench_at_handle']].abs().sum(-1) > 0).sum())
        # FK truth is used ONLY by this validation code, not by actor construction.
        error = (env.fixed_grasp.handle_pos_b - env.ts.bar_b).norm(dim=-1)
        maxima['fk_position_error_m'] = max(maxima['fk_position_error_m'], float(error.max()))
        maxima['label_normalized'] = max(maxima['label_normalized'], float(env.interaction_target[:, :3].abs().max()))
        maxima['peak_hand_N'] = max(maxima['peak_hand_N'], float(env.hand_force_peak.max()))
        fresh = env.episode_length_buf == 0
        if fresh.any():
            assert not env.actor_hist[fresh, :-1].any()
            assert not env.interaction_target[fresh].any()
            assert not env.applied_robot_wrench[fresh].any()
        if args.check_dynamics_cache and step in (10, 70, 100, 170, args.steps - 1):
            # Independent full-state getters are deliberately slow; validation
            # only. Covers both initial and post-reset randomized properties.
            full = env._body_dynamics(env.robot)
            cached = env._body_dynamics(env.robot, env._hand_body_ids)
            for actual, reference in zip(cached, full):
                torch.testing.assert_close(actual, reference[:, env._hand_body_ids], atol=2e-5, rtol=2e-5)
            direct, _, _ = env._measure_hand_wrench(*full, env.robot.data.body_com_acc_w.torch,
                                                   env._hand_subtree_ids)
            valid = env.dynamics_wrench_valid
            torch.testing.assert_close(env._hand_wrench[valid], direct[valid], atol=2e-3, rtol=2e-4)
            mass = env.cart.data.body_mass.torch
            reference = (env.cart.data.body_com_pose_w.torch[..., :3] * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
            torch.testing.assert_close(env.cart_com_world(), reference, atol=2e-5, rtol=2e-5)
            cache_checks += 1
    assert counts['physical_external_visible'] > 0
    assert counts['resets'] > 0
    assert counts['valid_fk'] > .9 * args.steps * args.num_envs
    assert counts['valid_label'] > .9 * args.steps * args.num_envs
    report = dict(result="PASS_EXECUTION", steps=args.steps, num_envs=args.num_envs,
                  counts=counts, maxima=maxima, actor_history=2480, critic=170,
                  dynamics_cache_checks=cache_checks,
                  fk_cuda_graph=env.fixed_grasp.cuda_graph_ready,
                  note="Runtime correctness only; zero actor, not training convergence or recovery advantage.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
finally:
    launcher.app.close()
