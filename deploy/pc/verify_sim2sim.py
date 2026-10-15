"""Validate the robot-side V8.1 inference core (deploy/robot/pushcart_v81/controller.py + the exported ONNX files)
against the released V8.1 pipeline in MuJoCo. Run on the training PC after export_v81.py.

    python deploy/pc/verify_sim2sim.py                       # lockstep + quick closed loop (3 runs)
    python deploy/pc/verify_sim2sim.py --full                # closed loop on the release bench: 36 runs

1. Lockstep (shadow): MujocoV8Runner (PyTorch, the released bench pipeline) drives the plant; at every control step the
   deploy core receives only what the robot will have (joint q / dq, pelvis gyro, projected gravity, camera yaw, task
   command) and its joint targets, frames and actions are compared with the runner's. The core is given the runner's
   upper-body torque (MuJoCo's last-substep PD torque), so any difference is a porting error (expected: float32 /
   ONNX rounding only, ~1e-5). The torque the core computes itself on the robot -- PD from the bridge's previous
   200 Hz joint sample, i.e. 5 ms before the observation, like MuJoCo's last substep -- is compared with MuJoCo's
   separately. (2026-10-15: PD from the newest sample instead raised the 0 kg cross-track RMS 0.046 -> 0.050 m; with
   MuJoCo's torque the core gave 0.0459 = the runner's 0.0457.)
2. Closed loop: the deploy core alone drives the plant along the release bench (composite course, follower, payload
   x caster x seed) and its cross-track RMS is compared with the runner's on the same conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "deploy" / "robot")]

from pushcart_v81.controller import MODE_PUSH, OnnxModels, Sensors, V81Controller, load_config   # noqa: E402

MODELS = ROOT / "deploy" / "robot" / "models"


def sensors(env, prev=None) -> Sensors:
    """What the robot delivers: joint state now and one 5 ms sample earlier (prev, from step_recording)."""
    return Sensors(q=env.dof_pos.copy(), dq=env.dof_vel.copy(), ang_vel=env.base_ang_vel.copy(),
                   gravity=env.projected_gravity.copy(), q_tau=None if prev is None else prev[0],
                   dq_tau=None if prev is None else prev[1])


def step_recording(env, fn, *args):
    """fn(*args) (a control step), also returning the joint state before its last 5 ms physics substep (= the robot
    bridge's previous 200 Hz sample at the next tick)."""
    import mujoco
    real, seen = mujoco.mj_step, []

    def rec(m, d):
        seen.append((env.dof_pos.copy(), env.dof_vel.copy()))
        real(m, d)
    mujoco.mj_step = rec
    try:
        out = fn(*args)
    finally:
        mujoco.mj_step = real
    return out, seen[-1]


def make_policies():
    from pushcart_bench.policies_v8 import V8Policy

    class ShadowPolicy(V8Policy):
        """The runner drives; the deploy core shadows it on the same inputs."""

        def __init__(self, core):
            super().__init__("final")
            self.core = core
            self.err = {k: 0.0 for k in ("q_target", "frame", "hl_actions", "leg_residual", "cmd_homie")}
            self.err_mean = {k: [] for k in self.err}
            self.tau_diff = []                                  # robot torque definition - MuJoCo's, N.m

        def reset(self, *a, **kw):
            super().reset(*a, **kw)
            self.core.reset_homie()
            self.core.reset_task()
            self.prev = None

        def step(self, cmd):
            r, env = self.runner, self.env
            cam = float(r.frames[-1, 94])                       # what the runner's camera model delivered
            frame_ref = r.frames[-1].copy()
            s = sensors(env, self.prev)
            if self.core.q_target_prev is not None:
                self.tau_diff.append(self.core.upper_torque(s) - r.ts["upper_tau"])
            out = self.core.step(s, cmd, cam, MODE_PUSH, upper_tau=r.ts["upper_tau"])
            frame_core = self.core.frames[-1].copy()
            _, self.prev = step_recording(env, super().step, cmd)
            for k, a, b in (("q_target", out.q_target, env.joint_pos_target), ("frame", frame_core, frame_ref),
                            ("hl_actions", out.hl_actions, r.hl_actions), ("leg_residual", out.leg_residual, r.leg_residual),
                            ("cmd_homie", out.cmd_homie, r.cmd_homie)):
                d = np.abs(np.asarray(a, float) - np.asarray(b, float))
                self.err[k] = max(self.err[k], float(d.max()))
                self.err_mean[k].append(float(d.mean()))

    class CorePolicy(V8Policy):
        """The deploy core drives the plant (the runner only provides MuJoCo, the grasp / caster models, the camera
        model and the bench bookkeeping)."""

        def __init__(self, core, lagged_tau=True):
            super().__init__("final")
            self.core, self.lagged_tau = core, lagged_tau

        def reset(self, *a, **kw):
            super().reset(*a, **kw)
            self.core.reset_homie()
            self.core.reset_task()
            self.cam = float(self.runner.frames[-1, 94])
            self.prev = None

        def step(self, cmd):
            import torch
            r, env = self.runner, self.env
            out = self.core.step(sensors(env, self.prev if self.lagged_tau else None), cmd, self.cam, MODE_PUSH)
            n = env.num_lower
            applied = (out.q_target[:n] - env.default_dof_pos[:n]) / env.action_scale
            env.upper_delta[:] = (out.q_target[n:] - env.default_dof_pos[n:]) / env.action_scale - env.upper_hold_actions
            env.set_command(*out.cmd_homie)
            anchor_prev = None if r.grasp.anchor is None else r.grasp.anchor.copy()
            (_, fallen), self.prev = step_recording(env, env.step, applied)
            r.grasp.update(r.grasp.hand_loads())
            r.casters.update()
            r.slip_v = np.zeros(2) if anchor_prev is None else (r.grasp.anchor - anchor_prev) / r.dt
            r.episode_step += 1
            r.task_state()
            # runner fields the bench telemetry reads
            r.cmd_task, r.cmd_ref, r.cmd_homie = np.asarray(cmd, float), out.cmd_ref, out.cmd_homie
            r.hl_actions, r.leg_residual, r.homie_leg_actions = out.hl_actions, out.leg_residual, out.homie_legs
            r.upper_delta_q = out.q_target[n:] - r.hold_target_q
            r.vel_est = out.vel_used
            r.frames = self.core.frames
            r.quality.update(torch.tensor([[*r.ts["cart_v"], r.ts["cart_wz"]]], dtype=torch.float64),
                             torch.from_numpy(r.cmd_ref[None]), torch.tensor([r.ts["cart_yaw"]], dtype=torch.float64),
                             torch.from_numpy(r.ts["cart_force_w"][None]), torch.from_numpy(r.ts["cart_moment_w"][None]))
            self.cam = float(r.camera(r.ts["rel_yaw"]))
            self._fallen = bool(fallen)

    return ShadowPolicy, CorePolicy, V8Policy


def bench_setup():
    from pushcart_bench.config_io import load_config as bench_config
    from pushcart_bench.follower import FollowerConfig
    from pushcart_bench.paths import build_path
    cfg = bench_config()
    fcfg = FollowerConfig.from_dict({**cfg["follower"], "v_nom": 0.5, "cruise": True})
    return cfg, fcfg, build_path("composite_course", cfg["paths"]["composite_course"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="closed loop on all 36 release-bench conditions")
    p.add_argument("--skip_reference", action="store_true", help="closed loop: do not re-run the runner")
    p.add_argument("--out", default=str(ROOT / "logs/validation/v81_deploy_verify.json"))
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.runner import Condition, run_path

    cfg_d = load_config(MODELS)
    core = V81Controller(cfg_d, OnnxModels(MODELS, threads=1))
    ShadowPolicy, CorePolicy, V8Policy = make_policies()
    cfg, fcfg, path = bench_setup()
    report = {"config_provenance": cfg_d["provenance"], "export_check": cfg_d.get("export_check")}

    # ---------------------------------------------------------------- 1. lockstep
    pol = ShadowPolicy(core)
    t0 = time.perf_counter()
    res = run_path(pol, path, fcfg, Condition(payload_kg=12.5, caster_angle_deg=90.0, seed=1), cfg)
    steps = len(pol.err_mean["q_target"])
    tau = np.abs(np.array(pol.tau_diff))
    report["lockstep"] = {"steps": steps, "max_abs": pol.err,
                          "mean_abs": {k: float(np.mean(v)) for k, v in pol.err_mean.items()},
                          "runner_cross_track_rms": float(res["metrics"]["cross_track_rms"]),
                          "upper_torque_robot_definition_minus_mujoco_Nm": {
                              "mean_abs_per_joint": tau.mean(0).round(4).tolist(), "p99_abs": float(np.percentile(tau, 99)),
                              "max_abs": float(tau.max())}}
    print(f"[lockstep] {steps} steps ({time.perf_counter() - t0:.0f}s) max |core - runner|: " +
          ", ".join(f"{k} {v:.2e}" for k, v in pol.err.items()), flush=True)
    print(f"[lockstep] upper torque, robot definition (PD from the previous 5 ms joint sample) - MuJoCo last substep: "
          f"mean {tau.mean():.3f}, p99 {np.percentile(tau, 99):.3f}, max {tau.max():.3f} N.m", flush=True)

    # ---------------------------------------------------------------- 2. closed loop
    if a.full:
        conds = [(pl, c, s) for pl in (0.0, 12.5, 25.0) for c in (0.0, 90.0, 180.0, 270.0) for s in (0, 1, 2)]
    else:
        conds = [(0.0, 0.0, 0), (12.5, 90.0, 1), (25.0, 180.0, 2)]
    runs = {"core": CorePolicy(core)}
    if not a.skip_reference:
        runs["runner"] = V8Policy("final")
    rows = []
    for name, pol in runs.items():
        t0 = time.perf_counter()
        for payload, caster, seed in conds:
            m = run_path(pol, path, fcfg, Condition(payload_kg=payload, caster_angle_deg=caster, seed=seed), cfg)["metrics"]
            rows.append({"driver": name, "payload": payload, "caster": caster, "seed": seed,
                         "success": bool(m["success"]), "xt": float(m["cross_track_rms"]), "failure": m.get("failure")})
            print(f"[closed/{name}] {payload:4.1f} kg caster {caster:5.1f} seed {seed}: xt {rows[-1]['xt']:.4f} m "
                  f"success {rows[-1]['success']}", flush=True)
        print(f"[closed/{name}] {len(conds)} runs in {time.perf_counter() - t0:.0f}s", flush=True)
    summary = {}
    for name in runs:
        summary[name] = {}
        for payload in sorted({c[0] for c in conds}):
            xt = [r["xt"] for r in rows if r["driver"] == name and r["payload"] == payload]
            fails = sum(not r["success"] for r in rows if r["driver"] == name and r["payload"] == payload)
            summary[name][str(payload)] = {"mean": float(np.mean(xt)), "max": float(np.max(xt)), "fails": fails}
    report["closed_loop"] = {"rows": rows, "summary": summary,
                             "release_reference_36_runs": {"0.0": 0.046, "12.5": 0.022, "25.0": 0.022}}
    print("\ncross-track RMS (m) per payload, mean / max / failures:")
    for name, s in summary.items():
        print(f"  {name:7s} " + " | ".join(f"{pl} kg {v['mean']:.4f} / {v['max']:.4f} / {v['fails']}" for pl, v in s.items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(a.out)
    ok = report["lockstep"]["max_abs"]["q_target"] < 1e-3 and all(r["success"] for r in rows if r["driver"] == "core")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
