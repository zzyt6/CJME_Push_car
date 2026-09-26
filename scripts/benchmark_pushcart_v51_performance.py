"""Compare bounded V4/V5.1 environment throughput without training or checkpoints.

Timing excludes startup, reset, warmup and profiling. Both variants use their
actual training quality profile, the same seed, level, environment count and
zero high-level actions. This measures environment collection cost, not PPO or
learned-policy performance. Run variants serially on an otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import hashlib
import importlib
import io
import json
from pathlib import Path
import pstats
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def source_hashes(package, package_root):
    return {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((package_root / package).rglob("*.py"))
    }


def record_sections(env, torch):
    """Install labels only for the optional instrumented pass, after timing."""
    objects = [(env, ("step", "_sample_dynamics", "_sample_training_dynamics",
                      "_body_dynamics", "_measure_hand_wrench", "_publish_hand_wrench",
                      "_compute_task_state", "_update_hl_observations", "_apply_action",
                      "_accumulate_diagnostics", "check_termination", "reset_idx"))]
    observer = getattr(env, "fixed_grasp", None)
    if observer is not None:
        objects.append((observer, ("update",)))
        if hasattr(observer, "kinematics"):
            objects.append((observer.kinematics, ("forward",)))
    for obj, methods in objects:
        for name in methods:
            if not hasattr(obj, name):
                continue
            original = getattr(obj, name)
            label = f"pushcart/{type(obj).__name__}.{name}"

            @functools.wraps(original)
            def recorded(*a, _original=original, _label=label, **kw):
                with torch.profiler.record_function(_label):
                    return _original(*a, **kw)

            setattr(obj, name, recorded)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v4", "v51"), required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--torch_profile_steps", type=int, default=0)
    parser.add_argument("--trace", action="store_true", help="Export optional torch Chrome trace")
    parser.add_argument("--label", default="")
    parser.add_argument("--package-root", type=Path, default=ROOT / "HomieRL",
                        help="Directory containing the selected package (optional source snapshot)")
    parser.add_argument("--run_checkpoint", type=Path, help="Load saved environment configuration only")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("num_envs/steps/repeats must be positive; warmup must be nonnegative")
    if args.profile_steps < 0 or args.torch_profile_steps < 0:
        parser.error("profiling step counts must be nonnegative")
    if args.variant == "v4" and args.run_checkpoint:
        parser.error("V4 make_configs does not restore a run; omit --run_checkpoint")
    package = "homie_pushcart" if args.variant == "v4" else "homie_pushcart_v51"
    args.package_root = args.package_root.resolve()
    if not (args.package_root / package / "__init__.py").is_file():
        parser.error(f"Missing package {package} below {args.package_root}")
    sys.path.insert(0, str(args.package_root))
    hashes = source_hashes(package, args.package_root)
    launcher = AppLauncher(args)
    try:
        import numpy as np
        import torch
        from homie_isaaclab.math_utils import class_to_dict

        module = lambda name: importlib.import_module(f"{package}.{name}")
        paths = module("paths")
        if args.package_root != ROOT / "HomieRL":
            # Snapshot source/config stays isolated while assets/checkpoints are
            # the same read-only project inputs. No snapshot file is rewritten.
            snapshot_root = paths.ROOT
            local_config = {"PACKAGE_DIR", "CONFIG_DIR", "INITIAL_POSE_YAML", "CART_PARAMS_YAML"}
            for name, value in list(vars(paths).items()):
                if isinstance(value, Path) and name not in local_config and value.is_relative_to(snapshot_root):
                    setattr(paths, name, ROOT / value.relative_to(snapshot_root))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cfg, task = module("env_smoke").make_configs(
            args.num_envs, True, args.level,
            **({"checkpoint": args.run_checkpoint} if args.variant == "v51" else {}))
        # make_configs alone does not apply the CLI's scratch-training profile.
        module("cart_quality").apply_quality_profile(task, args.variant)
        task.curriculum.enabled = False
        checkpoint = getattr(cfg, "pushcart_saved_checkpoint", None)
        env = module("task_env").PushCartEnv(
            cfg, task, ROOT, args.device, bool(getattr(launcher, "_headless", True)),
            checkpoint=Path(checkpoint) if checkpoint else None, level=args.level)
        env.reset()
        actions = torch.zeros(args.num_envs, env.num_hl_actions, device=args.device)
        is_cuda = torch.device(args.device).type == "cuda"

        def synchronize():
            if is_cuda:
                torch.cuda.synchronize(args.device)

        def step_once():
            return env.step(actions)

        report = {
            "variant": args.variant, "label": args.label,
            "package_root": str(args.package_root),
            "num_envs": args.num_envs, "level": args.level, "seed": args.seed,
            "warmup_steps": args.warmup, "steps_per_repeat": args.steps,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device) if is_cuda else None,
            "torch_version": torch.__version__, "control_dt_s": env.dt,
            "physics_dt_s": cfg.sim.dt, "decimation": cfg.control.decimation,
            "actor_observation_size": env.num_hl_obs, "critic_observation_size": env.num_hl_critic_obs,
            "homie_checkpoint": str(env.policy.checkpoint),
            "task_config": class_to_dict(task),
            "source_sha256": hashes,
            "method": "Zero high-level actions; normal frozen HOMIE and environment logic; CUDA synchronized only at timing boundaries; no PPO or high-level network inference.",
            "timing": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def save():
            args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        with torch.inference_mode():
            for _ in range(args.warmup):
                step_once()
            synchronize()
            print(f"[benchmark] ready: {args.variant}, N={args.num_envs}, level={args.level}, "
                  f"warmup={args.warmup}, steps={args.steps}, repeats={args.repeats}", flush=True)
            for repeat in range(args.repeats):
                # Keep reset accounting on the GPU until after the timed region.
                reset_count = torch.zeros((), dtype=torch.long, device=args.device)
                synchronize()
                start = time.perf_counter()
                for _ in range(args.steps):
                    _, _, _, done, _ = step_once()
                    reset_count.add_(done.count_nonzero())
                synchronize()
                elapsed = time.perf_counter() - start
                result = {
                    "repeat": repeat, "elapsed_s": elapsed,
                    "ms_per_control_step": 1000 * elapsed / args.steps,
                    "environment_transitions_per_s": args.num_envs * args.steps / elapsed,
                    "resets": int(reset_count),
                }
                report["timing"].append(result)
                print("[benchmark] timing " + json.dumps(result), flush=True)
                save()
            total = sum(row["elapsed_s"] for row in report["timing"])
            report["aggregate_environment_transitions_per_s"] = (
                args.num_envs * args.steps * args.repeats / total)
            report["aggregate_ms_per_control_step"] = 1000 * total / (args.steps * args.repeats)
            if args.profile_steps:
                profiler = cProfile.Profile()
                synchronize()
                profiler.enable()
                for _ in range(args.profile_steps):
                    step_once()
                synchronize()
                profiler.disable()
                profile_path = args.output.with_suffix(".pstats")
                profiler.dump_stats(str(profile_path))
                text = io.StringIO()
                for sort in ("tottime", "cumtime"):
                    text.write(f"\nSorted by {sort} (Python wall time, not CUDA kernel duration)\n")
                    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats(sort).print_stats(100)
                args.output.with_suffix(".profile.txt").write_text(text.getvalue(), encoding="utf-8")
                report["cprofile_steps"] = args.profile_steps
            if args.torch_profile_steps:
                record_sections(env, torch)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if is_cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                synchronize()
                with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                    for _ in range(args.torch_profile_steps):
                        step_once()
                        profiler.step()
                    synchronize()
                averages = profiler.key_averages()
                text = averages.table(sort_by="self_cpu_time_total", row_limit=100)
                if is_cuda:
                    text += "\n" + averages.table(sort_by="self_cuda_time_total", row_limit=100)
                args.output.with_suffix(".torch.txt").write_text(text, encoding="utf-8")
                if args.trace:
                    profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
                report["torch_profile_steps"] = args.torch_profile_steps
            synchronize()
            obs, critic = env.hl_obs, env.hl_critic_obs
            report["finite_observations"] = bool(torch.isfinite(obs).all() & torch.isfinite(critic).all())
            report["source_unchanged_during_run"] = hashes == source_hashes(package, args.package_root)
            if is_cuda:
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            save()
            print("[benchmark] complete " + str(args.output), flush=True)
    except BaseException:
        # Kit can terminate the interpreter during close; emit the failure first.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
