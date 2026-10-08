"""Profile one V7 environment step: where the time goes and which lines force a CPU <-> GPU synchronisation.

    E:\\envs\\RLisaac601\\python.exe scripts/profile_pushcart_v7.py --headless --num_envs 4096

1. baseline   step time with no instrumentation (one synchronize at the end of the block)
2. syncs      torch.cuda sync-debug mode: every torch op that blocks on the GPU, grouped by the project line
              that issued it (PhysX tensor-API / warp CPU round trips are not torch ops: see 3)
3. sections   each env method wrapped with synchronize-before/after timers (inclusive times; nested sections
              overlap, so read them as "time spent inside", not as a partition)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

SECTIONS = ("step", "_apply_cart_disturbances", "_apply_robot_push", "_post_physics_step_callback", "_compute_task_state",
            "_update_hl_observations", "_v7_privileged", "compute_reward", "check_termination", "reset_idx",
            "_randomize_cart", "_reset_root_states", "_record_episode_stats", "_log_episode_diagnostics",
            "_write_slide_limit", "_update_slide_anchor", "_update_persistence_counters", "compute_observations", "_internal_load",
            "_reward_grasp_internal", "_reward_track_cart_vx")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--level", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/validation/v7_profile.json")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    launcher = AppLauncher(args)
    try:
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv

        cfg = G1RoughCfg()
        apply_config(cfg, homie_baseline())
        cfg.env.num_envs = args.num_envs
        task = PushCartTaskCfg()
        task.curriculum.init_level = args.level
        task.env.episode_length_s = 4.0              # frequent resets, as in a population mid-training
        env = PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level)
        env.reset()
        act = lambda: 0.3 * torch.randn(args.num_envs, 18, device=args.device)   # noqa: E731
        for _ in range(30):
            env.step(act())
        # 1. baseline
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(act())
        torch.cuda.synchronize()
        base_ms = (time.perf_counter() - t0) / args.steps * 1e3
        print(f"[profile] baseline {base_ms:.1f} ms/step at {args.num_envs} envs", flush=True)
        # 2. synchronising torch ops, grouped by the first project frame of their stack
        sites = collections.Counter()
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(10):
                env.step(act())
        torch.cuda.set_sync_debug_mode(0)
        for w in caught:
            if "synchroniz" not in str(w.message).lower():
                continue
            stack = traceback.extract_stack()
            sites[f"{Path(w.filename).name}:{w.lineno}"] += 1
        # warnings carry the user frame that triggered them (warn stacklevel); also record by message head
        sync_sites = [{"site": k, "per_step": v / 10} for k, v in sites.most_common(40)]
        for s in sync_sites:
            print("[profile] sync", s, flush=True)
        # 3. sections
        times, calls = collections.defaultdict(float), collections.Counter()
        originals = {}
        for name in SECTIONS:
            fn = getattr(env, name, None)
            if fn is None:
                continue
            originals[name] = fn

            def wrap(fn=fn, name=name):
                def timed(*a, **k):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    finally:
                        torch.cuda.synchronize()
                        times[name] += time.perf_counter() - t
                        calls[name] += 1
                return timed
            setattr(env, name, wrap())
        policy = env.policy.act_and_estimate

        def timed_policy(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = policy(*a, **k)
            torch.cuda.synchronize()
            times["homie_policy"] += time.perf_counter() - t
            calls["homie_policy"] += 1
            return out
        env.policy.act_and_estimate = timed_policy
        sim_step = env.sim.step

        def timed_sim(*a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = sim_step(*a, **k)
            torch.cuda.synchronize()
            times["physx_sim_step(x4)"] += time.perf_counter() - t
            calls["physx_sim_step(x4)"] += 1
            return out
        env.sim.step = timed_sim
        # nested pieces of _compute_task_state
        import homie_pushcart.task_env as v32_task
        for owner, attr, label in ((env, "grasp_wrench", "grasp_wrench(V3.2 proxy)"), (env, "cart_hand_wrench", "cart_hand_wrench"),
                                   (env, "cart_state", "cart_state"), (env._labels, "sample", "hand_labels.sample(V6)"),
                                   (env, "_get_feet_heights", "_get_feet_heights")):
            fn = getattr(owner, attr)

            def wrap2(fn=fn, label=label):
                def timed(*a, **k):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    finally:
                        torch.cuda.synchronize()
                        times[label] += time.perf_counter() - t
                        calls[label] += 1
                return timed
            setattr(owner, attr, wrap2())
        v32_cts = v32_task.PushCartEnv._compute_task_state

        def timed_v32(self, *a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return v32_cts(self, *a, **k)
            finally:
                torch.cuda.synchronize()
                times["_compute_task_state(V3.2 part)"] += time.perf_counter() - t
                calls["_compute_task_state(V3.2 part)"] += 1
        v32_task.PushCartEnv._compute_task_state = timed_v32
        n = 50
        for _ in range(n):
            env.step(act())
        sections = sorted(({"section": k, "ms_per_step": v / n * 1e3, "calls_per_step": calls[k] / n}
                           for k, v in times.items()), key=lambda r: -r["ms_per_step"])
        for s in sections:
            print(f"[profile] {s['section']:32s} {s['ms_per_step']:8.2f} ms/step  ({s['calls_per_step']:.2f} calls/step)", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"num_envs": args.num_envs, "baseline_ms_per_step": base_ms,
                                           "sync_sites": sync_sites, "sections": sections}, indent=2), encoding="utf-8")
    finally:
        launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
