"""Policy x plant cross bench (MuJoCo, CPU): separates "the V7 policy tracks worse" from "the V7 plant is harder".

Every policy (passthrough HOMIE, V6, V7) runs on every plant (the frozen V3.2 legacy plant used by every bench
before V7, and the V7 plant), on the same path, follower, payload and seed as the official bench. The Actor
interface is identical in V6 and V7 (policy 605 + HIM 462 -> 18 actions), so the swap is exact.

Plant variants of the V7 scene isolate single physical changes (``--v7_variants``):
    v7               the V7 plant (nominal grasp, Coulomb rolling, swivel scrub fading with rolling speed)
    v7_constswivel   the V7 plant before 2026-10-08: standstill swivel scrub applied at every speed
    v7_noswivel      V3.2 viscous swivels      v7_noroll   V3.2 viscous wheels      v7_viscous   both
    v7_strong        nominal grasp compliance, capacities that never bind (load demand)
(variants live in pushcart_bench/policies_cross.py)

    python scripts/cross_plant_bench_v7.py --paths composite_course --payloads 0 12.5 25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

import mujoco
import numpy as np
import torch

V6_DEFAULT = ROOT / "logs/rsl_rl/homie_pushcart_v6/2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt"


def make_policy(version, plant, checkpoint, level=17):
    if version in ("v8", "v8_legres0"):                 # V8 teacher: V7 plant only (its own runner)
        if plant != "v7":
            return None
        from pushcart_bench.policies_v8 import V8Policy
        return V8Policy(checkpoint, level, zero_leg_residual=(version == "v8_legres0"))
    from pushcart_bench.policies_cross import CrossPolicy
    return CrossPolicy(version, plant, checkpoint, level)


# ------------------------------------------------------------------------------------------------ main

def summarize(result, loads):
    m = result["metrics"]
    tr = result["trace"]
    keep = ("success", "failure", "time_to_goal", "cross_track_rms", "cross_track_max", "heading_rms",
            "progress", "mean_speed", "cmd_wz_rmse", "cmd_vx_rmse", "duration")
    out = {k: m.get(k) for k in keep if k in m}
    for k, v in m.items():
        if ("cross" in k or "heading" in k or "rmse" in k) and k not in out and isinstance(v, (int, float)):
            out[k] = v
    L = np.array(loads) if loads else np.zeros((0, 20))
    if len(L):
        F = L[:, :12].reshape(-1, 2, 6)
        out["load_p99"] = {n: float(np.percentile(np.abs(F[:, :, i]), 99)) for i, n in enumerate(("Fx", "Fy", "Fz", "Mx", "My", "Mz"))}
        out["pull_p99_N"] = float(np.percentile(np.maximum(-F[:, :, 0], 0), 99))
        out["pivot_max_rad"] = {n: float(np.abs(L[:, 12 + 2 * i:14 + 2 * i]).max()) for i, n in enumerate(("roll", "tilt", "yaw"))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v6", default=str(V6_DEFAULT))
    p.add_argument("--v7", default="latest")
    p.add_argument("--v8", default="latest")
    p.add_argument("--policies", nargs="+", default=["homie", "v6", "v7"])
    p.add_argument("--plants", nargs="+", default=["legacy", "v7"])
    p.add_argument("--paths", nargs="+", default=["composite_course"])
    p.add_argument("--payloads", nargs="+", type=float, default=[0.0, 12.5, 25.0])
    p.add_argument("--v_nom", type=float, default=0.5)
    p.add_argument("--caster_deg", type=float, default=0.0)
    p.add_argument("--out", default=str(ROOT / "logs/validation/v7_cross_plant"))
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    from pushcart_bench.config_io import load_config
    from pushcart_bench.follower import FollowerConfig
    from pushcart_bench.paths import build_path
    from pushcart_bench.runner import Condition, run_path
    from pushcart_bench.policies_v7 import resolve_v7_checkpoint
    cfg = load_config()
    cfg["trace_stride"] = 1
    fcfg = FollowerConfig.from_dict({**cfg["follower"], "v_nom": a.v_nom, "cruise": True})
    v7 = str(resolve_v7_checkpoint(a.v7)) if "v7" in a.policies else None
    from pushcart_bench.policies_v8 import resolve_v8_checkpoint
    v8 = str(resolve_v8_checkpoint(a.v8)) if any(x.startswith("v8") for x in a.policies) else None
    ckpt = {"homie": None, "v6": a.v6, "v7": v7, "v8": v8, "v8_legres0": v8}
    ver = {"homie": None, "v6": 6, "v7": 7, "v8": "v8", "v8_legres0": "v8_legres0"}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for plant in a.plants:
        for pol in a.policies:
            policy = make_policy(ver[pol], plant, ckpt[pol])
            if policy is None:
                continue
            for pname in a.paths:
                path = build_path(pname, cfg["paths"][pname])
                for payload in a.payloads:
                    t0 = time.perf_counter()
                    res = run_path(policy, path, fcfg, Condition(payload_kg=payload, caster_angle_deg=a.caster_deg), cfg)
                    s = summarize(res, policy.loads)
                    row = {"policy": pol, "plant": plant, "path": pname, "payload": payload, **s}
                    rows.append(row)
                    print(f"[{pol:5s}@{plant:16s}] {pname} {payload:5.1f} kg  ok={s.get('success')}  "
                          f"xtrack_rms={s.get('cross_track_rms'):.4f}  head_rms_deg={math.degrees(s.get('heading_rms')):.2f}  wz_rmse={s.get('cmd_wz_rmse'):.3f}  "
                          f"t={s.get('time_to_goal')}  fail={s.get('failure')}  ({time.perf_counter() - t0:.0f}s)", flush=True)
                    extra = {"gait": np.array(policy.gait)} if getattr(policy, "gait", None) else {}
                    np.savez_compressed(out_dir / f"trace_{pol}_{plant}_{pname}_{payload:g}{a.tag}.npz",
                                        **{k: np.asarray(v) for k, v in res["trace"].items() if isinstance(v, list) and v and not isinstance(v[0], (dict, str))},
                                        loads=np.array(policy.loads), **extra)
    meta = {"v6": a.v6, "v7": v7, "v8": v8, "v_nom": a.v_nom, "caster_deg": a.caster_deg, "rows": rows}
    (out_dir / f"summary{a.tag}.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(out_dir / f"summary{a.tag}.json")


if __name__ == "__main__":
    main()
