"""MuJoCo test diagnosis for an RMA / ROA student: on the bench course, log the adaptation module's z_phi (what the student
uses) next to the teacher encoder's z_mu from the MuJoCo truth, plus the true / linearly-decoded cart velocity.

    python scripts/v8_rma_mujoco_latent_trace.py --checkpoint <phase-2 / student-RL model.pt> --payloads 0 25
Reports per payload: relative latent error, its steady (mean) part vs fluctuating part, and the cart vy decoded from
z_phi vs z_mu (decoder fitted on z_mu of the same runs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--payloads", nargs="+", type=float, default=[0.0, 25.0])
    p.add_argument("--caster", type=float, default=0.0)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from pushcart_bench.config_io import load_config
    from pushcart_bench.follower import FollowerConfig
    from pushcart_bench.paths import build_path
    from pushcart_bench.runner import Condition, run_path
    from pushcart_bench.policies_v8 import V8Policy
    from homie_pushcart_v8.rma_policy import load_rma
    cfg = load_config()
    fcfg = FollowerConfig.from_dict({**cfg["follower"], "v_nom": 0.5, "cruise": True})
    path = build_path("composite_course", cfg["paths"]["composite_course"])
    pol = V8Policy(a.checkpoint)
    r = pol.runner
    assert r.student, "the checkpoint must load as a deployable student"
    teacher = load_rma(a.checkpoint, deploy=False)
    log = []
    step0 = pol.step

    def step(cmd):
        step0(cmd)
        with torch.no_grad():
            h = torch.from_numpy(r.history_obs).unsqueeze(0)
            pv = torch.from_numpy(r.privileged).unsqueeze(0)
            log.append((r.policy.adaptation(h)[1][0].numpy(), teacher.encoder(teacher.privileged_normalizer(pv))[0].numpy(),
                        r.privileged[:3].copy()))
    pol.step = step
    out = {}
    for payload in a.payloads:
        log.clear()
        res = run_path(pol, path, fcfg, Condition(payload_kg=payload, caster_angle_deg=a.caster, seed=0), cfg)
        zp = np.array([x[0] for x in log]); zm = np.array([x[1] for x in log]); cv = np.array([x[2] for x in log])
        skip = 100
        zp, zm, cv = zp[skip:], zm[skip:], cv[skip:]
        err = zp - zm
        rel = float((err ** 2).mean() / (zm ** 2).mean())
        steady = float((err.mean(0) ** 2).sum() / (err ** 2).mean(0).sum())     # share of the error that is a constant bias
        X = np.c_[zm, np.ones(len(zm))]
        out[payload] = {"xt": float(res["metrics"]["cross_track_rms"]), "latent_rel_err": rel, "bias_share": steady,
                        "z_std": float(zm.std(0).mean())}
        for k, name in ((0, "vx"), (1, "vy"), (2, "wz")):
            w, *_ = np.linalg.lstsq(X, cv[:, k], rcond=None)
            dec_mu, dec_phi = X @ w, np.c_[zp, np.ones(len(zp))] @ w
            out[payload][name] = {"true_std": float(cv[:, k].std()), "dec_mu_r2": float(1 - ((dec_mu - cv[:, k]) ** 2).mean() / cv[:, k].var()),
                                  "phi_bias": float((dec_phi - dec_mu).mean()), "phi_err_rms": float(np.sqrt(((dec_phi - dec_mu) ** 2).mean()))}
        print(f"[trace] {payload:g} kg: {json.dumps(out[payload])}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
