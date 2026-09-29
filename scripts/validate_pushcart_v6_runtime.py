"""Compare real Isaac V3.2 and V6 execution in isolated, nonce-checked children.

This validates preservation of the physical task, rewards and legacy signals.
It does not train an estimator or establish V6 policy performance. Import and
--help do not start Kit. Run with the project Isaac Python and --headless.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
NONCE_ENV = "PUSHCART_V6_RUNTIME_RUN_ID"
SUCCESS = "PASS_EXECUTION"
VARIANTS = ("v32", "v6")


def make_runtime_configs(variant, num_envs=8, level=8):
    """Identical pinned V3.2 settings and explicitly shared probe overrides."""
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart.task_config import PushCartTaskCfg as V32TaskCfg
    from homie_pushcart_v6.baseline import apply_config, homie_baseline, task_baseline
    from homie_pushcart_v6.task_config import PushCartTaskCfg as V6TaskCfg

    if variant not in VARIANTS:
        raise ValueError(f"Unknown runtime child: {variant}")
    cfg = G1RoughCfg()
    apply_config(cfg, homie_baseline())
    cfg.env.num_envs = num_envs
    task = V6TaskCfg() if variant == "v6" else V32TaskCfg()
    if variant == "v32":
        apply_config(task, task_baseline())
        task.commands.sequence_enabled = False
        task.idle_correction.enabled = False
        task.rewards.force_rate_seed_on_reset = False
    task.curriculum.enabled = False
    task.curriculum.init_level = level
    task.env.episode_length_s = 20.
    # The old noise, latency and reset randomization remain enabled. Only
    # episodic pushes/stalls are replaced by this common deterministic script.
    task.randomization.robot_push = False
    task.randomization.cart_push_vel_max = 0.
    task.randomization.wheel_stall_prob_per_s = 0.
    return cfg, task


def prescribed_action(step, num_envs):
    """No RNG consumption: cover all 18 residual channels at small amplitude."""
    env = np.arange(num_envs, dtype=np.float64)[:, None]
    channel = np.arange(18, dtype=np.float64)[None, :]
    return (.025 * np.sin(.13 * step + .21 * channel + .17 * env)).astype(np.float32)


def compare_traces(baseline, upgraded):
    """Require exact finite equality; never conceal drift with a chosen tolerance."""
    missing = sorted(set(baseline) - set(upgraded))
    added = sorted(set(upgraded) - set(baseline))
    differences = {}
    if not baseline or not upgraded:
        differences["<empty>"] = {"reason": "Missing trace arrays"}
    for name in sorted(set(baseline) & set(upgraded)):
        a, b = np.asarray(baseline[name]), np.asarray(upgraded[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            differences[name] = {"baseline_shape": list(a.shape), "v6_shape": list(b.shape),
                                 "baseline_dtype": str(a.dtype), "v6_dtype": str(b.dtype)}
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            differences[name] = {"reason": "Non-finite trace"}
        elif not np.array_equal(a, b):
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            index = tuple(int(i) for i in np.argwhere(a != b)[0])
            differences[name] = {"max_abs_error": float(delta.max()),
                                 "unequal_values": int(np.count_nonzero(a != b)),
                                 "first_index": list(index),
                                 "baseline_first": a[index].item(), "v6_first": b[index].item()}
    return {"equal": not (missing or added or differences), "compared_arrays": len(set(baseline) & set(upgraded)),
            "missing_in_v6": missing, "added_in_v6": added, "differences": differences,
            "comparison": "exact (rtol=0, atol=0), non-finite values rejected"}


def validate_child_artifacts(output, variant, run_id, returncode):
    """Kit may exit zero after an exception; require this invocation's artifacts."""
    output = Path(output)
    report_path, trace_path = output / f"{variant}.json", output / f"{variant}.npz"
    failure_path = output / f"{variant}.failure.json"
    if returncode != 0:
        raise RuntimeError(f"{variant} child exited {returncode}; see {variant}.log")
    if not report_path.is_file() or not trace_path.is_file():
        raise RuntimeError(f"{variant} child did not write complete success artifacts")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id or report.get("variant") != variant or report.get("result") != SUCCESS:
        raise RuntimeError(f"{variant} child success artifact is stale or invalid")
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("run_id") == run_id:
            raise RuntimeError(f"{variant} child reported a failure in this invocation")
    if report.get("trace_sha256") != hashlib.sha256(trace_path.read_bytes()).hexdigest():
        raise RuntimeError(f"{variant} trace hash does not match its success report")
    return report


def _physics_contract(env):
    import omni.usd
    from pxr import UsdPhysics
    import torch

    ground = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Ground")
    assert ground.IsValid(), "Ground prim is missing"
    assert not ground.HasAPI(UsdPhysics.RigidBodyAPI), "V6 requires V3.2 static ground"
    kp = env.robot.data.joint_stiffness.torch[:, env.joint_ids]
    kd = env.robot.data.joint_damping.torch[:, env.joint_ids]
    torch.testing.assert_close(kp[:, :12], torch.zeros_like(kp[:, :12]), rtol=0, atol=0)
    torch.testing.assert_close(kd[:, :12], torch.zeros_like(kd[:, :12]), rtol=0, atol=0)
    torch.testing.assert_close(kp[:, 12:], env.p_gains[12:].expand_as(kp[:, 12:]), rtol=0, atol=0)
    torch.testing.assert_close(kd[:, 12:], env.d_gains[12:].expand_as(kd[:, 12:]), rtol=0, atol=0)
    assert env.cart_quality is not None, "V3.2 quality tracker must remain enabled"
    assert type(env.cart_quality).__module__ == "homie_pushcart.cart_quality"
    return {"static_ground": True, "implicit_upper_position_drive": True,
            "quality_tracker": type(env.cart_quality).__module__ + "." + type(env.cart_quality).__name__,
            "reward_names": list(env.reward_names), "scaled_reward_coefficients": dict(env.reward_scales),
            "joint_names": list(env.dof_names), "robot_body_names": list(env.robot.body_names),
            "cart_body_names": list(env.cart.body_names), "physics_dt_s": float(env.cfg.sim.dt),
            "control_dt_s": float(env.dt)}


def _snapshot(env, variant, raw_rewards):
    import torch

    def array(value):
        return value.detach().cpu().numpy().copy()

    sample = {
        "legacy_actor_obs": array(env.legacy_actor_obs if variant == "v6" else env.hl_obs),
        "legacy_critic_obs": array(env.legacy_critic_obs if variant == "v6" else env.hl_critic_obs),
    }
    names = ("root_states", "rigid_body_states", "dof_pos", "dof_vel", "torques", "joint_pos_target",
             "contact_forces", "obs_buf", "actions", "last_actions", "rew_buf", "reset_buf",
             "episode_length_buf", "terminated_buf", "time_out_buf", "cmd_task", "cmd_ref",
             "cmd_homie", "cmd_ref_f", "cart_vel_f", "upper_delta_q", "upper_delta", "vel_est",
             "cart_mass_total", "payload_com", "wheel_friction", "wheel_damping", "swivel_damping",
             "cart_pose_hist", "cart_pose_delay", "hl_delay", "force_c_prev", "force_c_prev_valid")
    for name in names:
        sample["state." + name] = array(getattr(env, name))
    for prefix, obj in (("task", env.ts), ("quality", env.cart_quality)):
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                sample[f"{prefix}.{name}"] = array(value)
    for name, value in env.term_reason_buf.items():
        sample["termination." + name] = array(value)
    for name, value in raw_rewards.items():
        sample["reward.raw." + name] = array(value)
    for prefix, articulation in (("robot", env.robot), ("cart", env.cart)):
        for name in ("body_mass", "body_inertia", "body_com_pose_b", "body_link_pose_w", "body_com_pose_w",
                     "body_com_vel_w", "joint_pos", "joint_vel", "joint_stiffness", "joint_damping", "joint_armature"):
            sample[f"physics.{prefix}.{name}"] = array(getattr(articulation.data, name).torch)
    sample["rng.cpu"] = torch.get_rng_state().cpu().numpy().copy()
    if str(env.device).startswith("cuda"):
        sample["rng.cuda"] = torch.cuda.get_rng_state(env.device).cpu().numpy().copy()
    return sample


def _check_v6_outputs(env, expected_next_velocity, expected_next_frame):
    import torch
    from homie_pushcart_v6.observation import without_handle_position

    n = env.num_envs
    for name, width in (("hl_obs", 605), ("estimator_obs", 462), ("hl_critic_obs", 665),
                        ("hand_target", 13), ("him_next_target", 81),
                        ("legacy_actor_obs", 620), ("legacy_critic_obs", 652)):
        tensor = getattr(env, name)
        assert tuple(tensor.shape) == (n, width), (name, tensor.shape)
        assert torch.isfinite(tensor).all(), name
    torch.testing.assert_close(env.hl_obs, without_handle_position(env.legacy_actor_obs), rtol=0, atol=0)
    torch.testing.assert_close(env.hl_critic_obs[:, :652], env.legacy_critic_obs, rtol=0, atol=0)
    torch.testing.assert_close(env.hl_critic_obs[:, 652:], env.hand_target, rtol=0, atol=0)
    torch.testing.assert_close(env.estimator_obs[:, -1], env.legacy_actor_obs[:, -124 + 104], rtol=0, atol=0)
    valid = env.him_next_target[:, -1] > .5
    if valid.any():
        torch.testing.assert_close(env.him_next_target[valid, :77], expected_next_frame[valid], rtol=0, atol=0)
        torch.testing.assert_close(env.him_next_target[valid, 77:80], expected_next_velocity[valid], rtol=0, atol=0)
    return int(valid.sum()), int((env.hand_target[:, -1] > .5).sum())


def _run_child(args):
    # The simulator must be started before importing env.py or USD/PhysX modules.
    from isaaclab.app import AppLauncher

    launcher = None
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get(NONCE_ENV)
    if not run_id:
        raise RuntimeError("Runtime child requires its parent-generated nonce")
    try:
        launcher = AppLauncher(headless=True, device=args.device)
        import torch
        from homie_isaaclab.math_utils import quat_rotate_inverse
        from homie_pushcart_v6.baseline import baseline_source, config_dict
        from homie_pushcart_v6.physics import physics_manifest

        if args.variant == "v6":
            from homie_pushcart_v6.task_env import PushCartEnv
        else:
            from homie_pushcart.task_env import PushCartEnv
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg, task = make_runtime_configs(args.variant, args.num_envs, args.level)
        task_reward_config = config_dict(task.rewards)
        env = PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level)
        env.fixed_command, env.fixed_payload_kg, env.fixed_caster_angle = [.3, 0., .15], 29., .2
        env.reset()
        contract = _physics_contract(env)
        raw_rewards = {name: torch.zeros(args.num_envs, device=args.device) for name in env.reward_names}
        if "termination" in env.reward_scales:
            raw_rewards["termination"] = torch.zeros(args.num_envs, device=args.device)
        wrapped = []
        for name, fn in zip(env.reward_names, env.reward_functions):
            def capture(fn=fn, name=name):
                value = fn()
                raw_rewards[name] = value.detach().clone()
                return value
            wrapped.append(capture)
        env.reward_functions = wrapped
        if "termination" in raw_rewards:
            termination = env._reward_termination
            def capture_termination():
                value = termination()
                raw_rewards["termination"] = value.detach().clone()
                return value
            env._reward_termination = capture_termination
        next_velocity = torch.zeros(args.num_envs, 3, device=args.device)
        next_frame = torch.zeros(args.num_envs, 77, device=args.device)
        callback = env._post_physics_step_callback
        def capture_next_state():
            callback()
            next_velocity.copy_(torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1)
                                * torch.tensor([2., 2., .5], device=args.device))
            # Independent clean next-frame construction. Calling HOMIE's
            # compute_termination_observations would consume its noise RNG.
            imu = env.rigid_body_states[:, env.imu_index]
            next_frame.copy_(torch.cat((
                env.commands[:, :3] * env.commands_scale, env.commands[:, 4:5],
                quat_rotate_inverse(imu[:, 3:7], imu[:, 10:13]) * env.obs_scales.ang_vel,
                quat_rotate_inverse(imu[:, 3:7], env.gravity_vec),
                (env.dof_pos - env.default_dof_pos) * env.obs_scales.dof_pos,
                env.dof_vel * env.obs_scales.dof_vel, env.actions[:, :12], env.ts.rel_yaw[:, None],
            ), -1))
        env._post_physics_step_callback = capture_next_state
        samples = [_snapshot(env, args.variant, raw_rewards)]
        valid_velocity = valid_hand = manual_resets = automatic_resets = 0
        with torch.inference_mode():
            for step in range(args.steps):
                if not launcher.app.is_running():
                    raise RuntimeError("Kit closed before the requested trace completed")
                if step in (args.steps // 2, 3 * args.steps // 4):
                    env.fixed_payload_kg = 0. if step == args.steps // 2 else 15.
                    ids = torch.tensor([0 if step == args.steps // 2 else min(1, args.num_envs - 1)],
                                       device=args.device, dtype=torch.long)
                    env.reset_idx(ids)
                    env._update_hl_observations()
                    if args.variant == "v6":
                        assert (env.hand_target[ids, -1] == 0).all()
                        assert (env.him_next_target[ids, -1] == 0).all()
                        assert (env.estimator_obs[ids, :-77] == 0).all()
                    manual_resets += len(ids)
                command = [.3, 0., .15 if step < args.steps // 2 else -.15]
                env.fixed_command = command
                env.cmd_task[:] = torch.tensor(command, device=args.device)
                action = torch.as_tensor(prescribed_action(step, args.num_envs), device=args.device)
                _, _, _, done, _ = env.step(action)
                automatic_resets += int(done.sum())
                if args.variant == "v6":
                    velocity, hand = _check_v6_outputs(env, next_velocity, next_frame)
                    valid_velocity += velocity
                    valid_hand += hand
                samples.append(_snapshot(env, args.variant, raw_rewards))
        if args.variant == "v6" and not (valid_velocity > 0 and valid_hand > 0):
            raise AssertionError("V6 did not produce valid velocity and hand supervision")
        keys = set(samples[0])
        assert all(set(sample) == keys for sample in samples), "Trace field coverage changed during execution"
        trace = {name: np.stack([sample[name] for sample in samples]) for name in sorted(keys)}
        for name, value in trace.items():
            assert np.isfinite(value).all(), f"Non-finite {name}"
        trace_path = args.output / f"{args.variant}.npz"
        np.savez_compressed(trace_path, **trace)
        report = {"result": SUCCESS, "run_id": run_id, "variant": args.variant, "steps": args.steps,
                  "num_envs": args.num_envs, "seed": args.seed, "level": args.level,
                  "contract": contract, "reward_configuration": task_reward_config,
                  "baseline_source": baseline_source(), "physics": physics_manifest(),
                  "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  "trace_arrays": len(trace), "manual_resets": manual_resets,
                  "automatic_resets": automatic_resets, "valid_velocity_labels": valid_velocity,
                  "valid_hand_labels": valid_hand, "observation_noise_preserved": bool(cfg.noise.add_noise)}
        (args.output / f"{args.variant}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[V6 runtime {args.variant}] PASS_EXECUTION: {args.steps} steps, {len(trace)} trace arrays", flush=True)
    except BaseException:
        error = traceback.format_exc()
        (args.output / f"{args.variant}.failure.json").write_text(json.dumps(
            {"result": "FAIL", "run_id": run_id, "variant": args.variant, "error": error}, indent=2), encoding="utf-8")
        print(error, flush=True)
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


def _run_parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    report = {"result": "FAIL", "run_id": run_id, "steps": args.steps, "num_envs": args.num_envs,
              "seed": args.seed, "level": args.level, "children": {}}
    try:
        for variant in VARIANTS:
            command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant,
                       "--output", str(args.output.resolve()), "--steps", str(args.steps),
                       "--num_envs", str(args.num_envs), "--seed", str(args.seed),
                       "--level", str(args.level), "--device", args.device, "--headless"]
            print(f"[V6 runtime] starting {variant}; log: {args.output / (variant + '.log')}", flush=True)
            with (args.output / f"{variant}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, env=dict(os.environ, **{NONCE_ENV: run_id}),
                                        stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_s, cwd=ROOT)
            report["children"][variant] = validate_child_artifacts(args.output, variant, run_id, result.returncode)
        old, new = report["children"]["v32"], report["children"]["v6"]
        for name in ("contract", "reward_configuration", "baseline_source", "physics", "steps", "num_envs",
                     "seed", "level", "manual_resets", "automatic_resets", "observation_noise_preserved"):
            if old[name] != new[name]:
                raise AssertionError(f"V3.2/V6 runtime metadata differs: {name}")
        with np.load(args.output / "v32.npz", allow_pickle=False) as baseline, np.load(args.output / "v6.npz", allow_pickle=False) as upgraded:
            report["comparison"] = compare_traces(baseline, upgraded)
        if not report["comparison"]["equal"]:
            raise AssertionError("V6 changed V3.2 runtime arrays; inspect report.json comparison")
        report["result"] = "PASS_BASELINE_PARITY"
        report["limitations"] = ["Prescribed high-level actions, frozen HOMIE; no V6 actor training or performance claim.",
                                 "V6 label checks cover shape, finiteness, privilege projection and velocity timing; they do not calibrate force accuracy."]
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[V6 runtime] {report['result']}: {args.output / 'report.json'}", flush=True)
    return 0 if report["result"] == "PASS_BASELINE_PARITY" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", help="Accepted for clarity; this validator always runs headless")
    parser.add_argument("--timeout_s", type=float, default=900.)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/bench" / datetime.now().strftime("v6_runtime_contract_%Y%m%d_%H%M%S"))
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps < 8 or args.num_envs < 1 or args.timeout_s <= 0:
        parser.error("Require steps>=8, num_envs>=1 and timeout_s>0")
    if args.variant:
        _run_child(args)
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
