"""MuJoCo bench evaluation of V8 checkpoints (V7 plant, composite course, follower as the official bench): ROA teachers
(privileged) and deployable students; also used to pick checkpoints.

Every checkpoint runs every condition: payload x caster angle x seed (the seed drives the camera latency / noise /
dropout draws). Reports mean, worst case and failures per checkpoint and payload, and the overall ranking
(mean cross-track RMS over all runs; ties broken by the worst run).

    python scripts/select_v8_teacher.py --checkpoints <run>/model_6000.pt <run>/model_9000.pt ... \
        [--payloads 0 12.5 25] [--casters 0 90 180 270] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]

import numpy as np
import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--payloads", nargs="+", type=float, default=[0.0, 12.5, 25.0])
    p.add_argument("--casters", nargs="+", type=float, default=[0.0, 90.0, 180.0, 270.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--v_nom", type=float, default=0.5)
    p.add_argument("--out", default=str(ROOT / "logs/validation/v8_teacher_selection.json"))
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    from pushcart_bench.config_io import load_config
    from pushcart_bench.follower import FollowerConfig
    from pushcart_bench.paths import build_path
    from pushcart_bench.runner import Condition, run_path
    from pushcart_bench.policies_v8 import V8Policy
    cfg = load_config()
    fcfg = FollowerConfig.from_dict({**cfg["follower"], "v_nom": a.v_nom, "cruise": True})
    path = build_path("composite_course", cfg["paths"]["composite_course"])
    rows = []
    for ck in a.checkpoints:
        pol = V8Policy(ck)
        name = f"{Path(ck).parent.name[:19]}/{Path(ck).stem}"
        t0 = time.perf_counter()
        for payload in a.payloads:
            for caster in a.casters:
                for seed in a.seeds:
                    res = run_path(pol, path, fcfg, Condition(payload_kg=payload, caster_angle_deg=caster, seed=seed), cfg)
                    m = res["metrics"]
                    rows.append({"checkpoint": str(ck), "name": name, "payload": payload, "caster": caster, "seed": seed,
                                 "success": bool(m["success"]), "xt": float(m["cross_track_rms"]),
                                 "heading_deg": float(np.degrees(m["heading_rms"])), "failure": m.get("failure")})
        print(f"[select] {name}: {len(a.payloads) * len(a.casters) * len(a.seeds)} runs ({time.perf_counter() - t0:.0f}s)", flush=True)
    summary = {}
    for ck in a.checkpoints:
        r = [x for x in rows if x["checkpoint"] == str(ck)]
        name = r[0]["name"]
        per = {}
        for payload in a.payloads:
            q = [x for x in r if x["payload"] == payload]
            xt = np.array([x["xt"] for x in q])
            per[payload] = {"mean": float(xt.mean()), "max": float(xt.max()), "fails": sum(not x["success"] for x in q),
                            "heading_mean_deg": float(np.mean([x["heading_deg"] for x in q]))}
        xt = np.array([x["xt"] for x in r])
        summary[name] = {"mean": float(xt.mean()), "max": float(xt.max()), "fails": sum(not x["success"] for x in r),
                         "per_payload": per}
    ranking = sorted(summary, key=lambda n: (summary[n]["fails"], summary[n]["mean"], summary[n]["max"]))
    print("\ncheckpoint                                   all mean / max   fails | " +
          " | ".join(f"{pl:g} kg mean / max" for pl in a.payloads))
    for n in ranking:
        s = summary[n]
        print(f"{n:44s} {s['mean']:.4f} / {s['max']:.4f}  {s['fails']:3d} | " +
              " | ".join(f"{s['per_payload'][pl]['mean']:.4f} / {s['per_payload'][pl]['max']:.4f}" for pl in a.payloads))
    Path(a.out).write_text(json.dumps({"ranking": ranking, "summary": summary, "rows": rows,
                                       "conditions": {"payloads": a.payloads, "casters": a.casters, "seeds": a.seeds,
                                                      "v_nom": a.v_nom}}, indent=1), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
