"""Plan B diagnosis: is the ROA adaptation module under-trained (it updates once per 20 iterations while the encoder
moves), or is the teacher's latent not recoverable from the deployable history?

Data: scripts/v8_probe_collect.py with the ROA checkpoint as teacher and student (the deployable mode drives).
On held-out environments, compares z_mu (teacher encoder) with z_phi from (a) the run's adaptation module, (b) the same
module trained to convergence on these data, (c) a fresh module trained to convergence; reports the relative latent
error and the applied-action error of the base policy fed with each z.

    python scripts/v8_roa_estimator_check.py --checkpoint <roa model.pt> --data logs/validation/v8_probe/roa3000_deploy.npz
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import numpy as np
    import torch
    from homie_pushcart.hl_policy import import_rsl_rl5
    import_rsl_rl5()
    from homie_pushcart_v8.rma_policy import load_rma
    from homie_pushcart_v8.student_policy import AdaptationModule
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_rma(a.checkpoint, deploy=False, device=dev)
    student = load_rma(a.checkpoint, deploy=True, device=dev)
    d = np.load(a.data)
    frames = torch.from_numpy(d["frames"].astype(np.float32))           # [T, N, 126]
    priv = torch.from_numpy(d["privileged"])                            # [T, N, 62]
    if teacher.extra and "priv_extra" in d:                             # arm-scale privileged teacher: 62 + 1
        priv = torch.cat((priv, torch.from_numpy(d["priv_extra"])), -1)
    step = torch.from_numpy(d["step"])                                  # [T, N] episode step after the env step
    T, N, F = frames.shape
    H = 32
    # samples whose 32-frame history lies inside one episode (no zero padding); envs split 80 / 20
    t_idx, n_idx = torch.nonzero((step >= H) & (torch.arange(T)[:, None] >= H - 1), as_tuple=True)
    test_env = torch.zeros(N, dtype=torch.bool)
    test_env[torch.randperm(N)[: N // 5]] = True
    is_test = test_env[n_idx]
    print(f"[check] {len(t_idx)} samples ({int(is_test.sum())} held-out), T={T}, N={N}", flush=True)
    clip = 100.0

    def history(ids):
        t, n = t_idx[ids], n_idx[ids]
        win = torch.stack([frames[t - k, n] for k in range(H - 1, -1, -1)], 1)      # [B, 32, 126], oldest first
        return win.reshape(len(ids), -1).clamp(-clip, clip).to(dev)

    def batch(ids):
        t, n = t_idx[ids], n_idx[ids]
        return history(ids), priv[t, n].to(dev), frames[t, n]

    def z_mu(pv):
        return teacher.encoder(teacher.privileged_normalizer(pv))

    train_ids = torch.nonzero(~is_test).squeeze(1)
    test_ids = torch.nonzero(is_test).squeeze(1)

    def policy_input(ids):
        """the 5-frame policy input of each sample (last 5 frames of its history)"""
        t, n = t_idx[ids], n_idx[ids]
        win = torch.stack([frames[t - k, n] for k in range(4, -1, -1)], 1)
        return win.reshape(len(ids), -1).clamp(-clip, clip).to(dev)

    def applied(act):
        return torch.cat((act[:, :18].clamp(-1, 1), 0.3 * torch.tanh(act[:, 18:])), -1)

    @torch.no_grad()
    def evaluate(module):
        se, ref, act_err, n = 0., 0., 0., 0
        per_dim_se, per_dim_var = 0., 0.
        zs = []
        for s in range(0, len(test_ids), a.batch):
            ids = test_ids[s:s + a.batch]
            h, pv, _ = batch(ids)
            zt, zp = z_mu(pv), module(h)[1]
            se += (zp - zt).pow(2).sum().item()
            ref += zt.pow(2).sum().item()
            zs.append(zt.cpu())
            per_dim_se = per_dim_se + (zp - zt).pow(2).sum(0).cpu()
            x = teacher.obs_normalizer(policy_input(ids))
            a_t = applied(teacher.deterministic(teacher.mlp(torch.cat((x, zt), -1))))
            a_s = applied(teacher.deterministic(teacher.mlp(torch.cat((x, zp), -1))))
            act_err += (a_s - a_t).pow(2).mean(1).sum().item()
            n += len(ids)
        zs = torch.cat(zs)
        r2 = 1 - per_dim_se / ((zs - zs.mean(0)).pow(2).sum(0) + 1e-9)
        return {"latent_rel_mse": se / max(ref, 1e-9), "action_mse": act_err / n,
                "r2_per_dim": [round(float(v), 3) for v in r2], "r2_mean": float(r2.mean())}

    def fit(module, label):
        module = module.to(dev).train()
        opt = torch.optim.Adam([q for nme, q in module.named_parameters() if not nme.startswith("normalizer")], lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        for ep in range(a.epochs):
            perm = train_ids[torch.randperm(len(train_ids))]
            tot, nb = 0., 0
            for s in range(0, len(perm), a.batch):
                h, pv, _ = batch(perm[s:s + a.batch])
                with torch.no_grad():
                    zt = z_mu(pv)
                loss = (module(h)[1] - zt).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            if ep % 5 == 4 or ep == a.epochs - 1:
                module.eval()
                r = evaluate(module)
                module.train()
                print(f"[check] {label} epoch {ep + 1}: train mse {tot / nb:.4f} | held-out latent rel {r['latent_rel_mse']:.3f} "
                      f"R2 {r['r2_mean']:.3f} action mse {r['action_mse']:.4f}", flush=True)
        return module.eval()

    out = {"run_module": evaluate(student.adaptation)}
    print(f"[check] run's adaptation module: {json.dumps({k: v for k, v in out['run_module'].items() if k != 'r2_per_dim'})}", flush=True)
    tuned = fit(copy.deepcopy(student.adaptation), "fine-tuned")
    out["fine_tuned"] = evaluate(tuned)
    fresh = AdaptationModule(student.adaptation.head[-1].out_features, student.adaptation.conv[0].out_channels,
                             explicit_dim=0)
    fresh.normalizer.load_state_dict(student.adaptation.normalizer.state_dict())
    fresh = fit(fresh, "fresh")
    out["fresh"] = evaluate(fresh)
    act_err, n = 0., 0                                      # reference: z replaced by its held-out mean
    with torch.no_grad():
        zs = torch.cat([z_mu(batch(test_ids[s:s + a.batch])[1]).cpu() for s in range(0, len(test_ids), a.batch)])
        zbar = zs.mean(0).to(dev)
        for s in range(0, len(test_ids), a.batch):
            ids = test_ids[s:s + a.batch]
            _, pv, _ = batch(ids)
            zt = z_mu(pv)
            x = teacher.obs_normalizer(policy_input(ids))
            a_t = applied(teacher.deterministic(teacher.mlp(torch.cat((x, zt), -1))))
            a_m = applied(teacher.deterministic(teacher.mlp(torch.cat((x, zbar.expand_as(zt)), -1))))
            act_err += (a_m - a_t).pow(2).mean(1).sum().item(); n += len(ids)
    out["z_mean_action_mse"] = act_err / n
    for k in ("run_module", "fine_tuned", "fresh"):
        r = out[k]
        print(f"[check] {k:11s} latent rel {r['latent_rel_mse']:.3f}  R2 mean {r['r2_mean']:.3f}  action mse {r['action_mse']:.4f}")
    print(f"[check] z = mean      action mse {out['z_mean_action_mse']:.4f}")
    print("[check] fresh R2 per dim", out["fresh"]["r2_per_dim"])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
