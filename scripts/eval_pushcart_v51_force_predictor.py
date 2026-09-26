"""Offline accuracy test of the V5.1 interaction (load) predictor, plus proprioceptive estimator baselines.

Answers, for one trained checkpoint:
  1. How accurate is the in-policy head (R2 / RMSE per axis, by condition, frequency band, lag)?
  2. How much of the label is sub-control-rate content (200 Hz substeps vs the 50 Hz sampled label)?
  3. With the SAME actor inputs, how well do dedicated supervised estimators (own optimizer) do, for the
     resultant [Fx, Fy, Mz] and for per-hand forces (left/right)? Includes a 5-frame FALCON-style history.
  4. Does the policy actually use the predicted-load channel (pred / truth / zero swap, closed loop)?
  5. Response to known cart force pulses and a sustained resistance (build-up / hold / release).

Labels are the training labels (subtree-compensated hand loads, not a calibrated sensor).
"""
import argparse
import dataclasses
import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500, help="control steps per natural-distribution run")
parser.add_argument("--level", type=int, default=None, help="default: the level saved in the checkpoint")
parser.add_argument("--stride", type=int, default=5, help="history subsampling for the offline estimators")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--swap_steps", type=int, default=1000)
parser.add_argument("--pulse_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from tensordict import TensorDict

    from homie_isaaclab.math_utils import euler_from_quaternion
    from homie_pushcart_v51.cart_quality import resultant_wrench, rotate_planar
    from homie_pushcart_v51.curriculum import CurriculumTracker
    from homie_pushcart_v51.env_smoke import make_configs
    from homie_pushcart_v51.observation import WRENCH_SCALE
    from homie_pushcart_v51.task_env import PushCartEnv

    dev = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = args.output or ROOT / f"logs/bench/v51_force_predictor_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_np = np.array(WRENCH_SCALE, dtype=np.float32)
    AXES = ("Fx_N", "Fy_N", "Mz_Nm")
    HAND_AXES = ("L_fx", "L_fy", "L_fz", "R_fx", "R_fy", "R_fz")

    saved = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    infos = saved.get("infos") or {}
    trained_level = (infos.get("homie_pushcart_v51_curriculum") or {}).get("tracker", {}).get("level")
    level = args.level if args.level is not None else (trained_level if trained_level is not None else 17)
    iteration = saved.get("iter")
    del saved
    print(f"[force-eval] checkpoint iter {iteration}, saved curriculum level {trained_level}, evaluating level {level}",
          flush=True)

    cfg, task = make_configs(args.num_envs, True, level, None, checkpoint=args.checkpoint)
    task.curriculum.enabled = False
    homie_ckpt = getattr(cfg, "pushcart_saved_checkpoint", None)
    env = PushCartEnv(cfg, task, ROOT, dev, True, level=level, checkpoint=Path(homie_ckpt) if homie_ckpt else None)
    N, dec = env.num_envs, env.cfg.control.decimation

    from homie_pushcart_v51.hl_policy import HighLevelPolicy
    hl = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, env.num_hl_actions, dev)
    actor = hl.actor
    FD = actor.frame_dim
    scale_t = torch.tensor(WRENCH_SCALE, device=dev)

    # ------------------------------------------------------------ 200 Hz substep label capture (read-only)
    substeps = []
    pending = {"sim": False}
    orig_sim_step, orig_update = env.sim.step, env.scene.update

    def hooked_sim_step(*a, **k):
        pending["sim"] = True
        return orig_sim_step(*a, **k)

    def hooked_update(dt):
        orig_update(dt)
        if not pending["sim"]:
            return
        pending["sim"] = False
        mass, pos, vel, inertia = env._body_dynamics(env.robot, env._hand_body_ids)
        acc = env.robot.data.body_com_acc_w.torch[:, env._hand_body_ids]
        measured, palm_pose, _ = env._measure_hand_wrench(mass, pos, vel, inertia, acc, env._hand_local_ids)
        measured = torch.nan_to_num(measured)
        force, moment = resultant_wrench(measured[..., :3], measured[..., 3:], palm_pose[..., :3],
                                         env.handle_center_world())
        _, cart_quat, _, _ = env.cart_state()
        yaw = euler_from_quaternion(cart_quat)[2].reshape(-1)
        label = rotate_planar(torch.stack((force[:, 0], force[:, 1], moment[:, 2]), -1), yaw, inverse=True)
        hands = rotate_planar(measured[..., :3], yaw[:, None], inverse=True).reshape(N, 6)
        substeps.append(torch.cat((label, hands), -1))

    env.sim.step, env.scene.update = hooked_sim_step, hooked_update

    def env_step(action):
        substeps.clear()
        pending["sim"] = False
        obs, critic, rew, done, extras = env.step(action)
        if len(substeps) < dec:
            raise RuntimeError(f"expected {dec} physics substeps, captured {len(substeps)}")
        return obs, rew, done, extras, torch.stack(substeps[:dec], 1)  # [N, dec, 9]

    def heads(obs):
        latent = actor.history_encoder(obs)
        return latent, actor.interaction_head(latent)

    def act(obs, mode="pred", truth=None):
        latent, pred = heads(obs)
        if mode == "stoch":
            td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
            return actor(td, stochastic_output=True), pred, latent
        x = {"pred": pred, "det": pred, "truth": truth, "zero": torch.zeros_like(pred)}[mode]
        out = actor.mlp(torch.cat((obs[:, -FD:], latent, x), -1))
        return actor.distribution.deterministic_output(out), pred, latent

    def set_level(disturbance=None, push=None):
        env.curriculum = CurriculumTracker(env.task_cfg, level)
        p = env.curriculum.params
        if disturbance is not None:
            p = dataclasses.replace(p, disturbance_frac=disturbance, robot_push=push)
        env._apply_level(p)

    # ------------------------------------------------------------ natural-distribution rollouts
    def rollout(mode, steps, keep_obs):
        set_level()
        env.fixed_command = env.fixed_payload_kg = env.fixed_caster_angle = None
        obs, _ = env.reset()
        rec = {k: [] for k in ("pred", "label", "sub", "hand_m", "latent", "ep", "cmd", "cartv", "mass",
                               "pulse", "push", "stall", "gate", "rew", "term")}
        X, rows = [], []
        with torch.no_grad():
            for t in range(steps):
                action, _, _ = act(obs, mode)
                obs, rew, done, extras, sub = env_step(action)
                latent, pred = heads(obs)
                rec["pred"].append(pred.cpu())
                rec["label"].append(env.interaction_target.clone().cpu())
                rec["sub"].append(sub.cpu())
                moments = rotate_planar(env.ts.torque_w_raw, env.ts.cart_yaw[:, None], inverse=True).reshape(N, 6)
                rec["hand_m"].append(moments.cpu())
                rec["latent"].append(latent.cpu())
                rec["ep"].append(env.episode_length_buf.clone().cpu())
                rec["cmd"].append(env.cmd_ref.clone().cpu())
                rec["cartv"].append(torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1).cpu())
                rec["mass"].append(env.cart_mass_total.clone().cpu())
                rec["pulse"].append((env.cart_applied_wrench.abs().sum(-1) > 0).cpu())
                rec["push"].append((env.applied_robot_wrench.abs().sum(-1) > 0).cpu())
                rec["stall"].append((env.stall_timer > 0).cpu())
                rec["gate"].append(env.recovery_authority.gate[:, 0].clone().cpu())
                rec["rew"].append(rew.cpu())
                rec["term"].append(extras["terminated"].cpu())
                if keep_obs and t % args.stride == 0:
                    X.append(obs.half().cpu())
                    rows.append(t)
                if t % 250 == 0:
                    print(f"[force-eval] {mode} rollout step {t}/{steps}", flush=True)
        rec = {k: torch.stack(v).numpy() for k, v in rec.items()}
        rec["X"], rec["rows"] = (torch.stack(X) if X else None), np.array(rows)
        return rec

    # ------------------------------------------------------------ metrics
    def r2_rmse(p, y, m):
        """p, y: [..., D]; m: [...] boolean. Returns dict of per-dim r2, rmse, label std."""
        p, y = p[m].astype(np.float64), y[m].astype(np.float64)
        err = ((p - y) ** 2).mean(0)
        var = y.var(0)
        return {"n": int(m.sum()), "r2": (1 - err / np.maximum(var, 1e-12)).tolist(),
                "rmse": np.sqrt(err).tolist(), "label_std": np.sqrt(var).tolist(),
                "label_mean": y.mean(0).tolist(), "pred_std": p.std(0).tolist()}

    def huber(p, y, m, delta=1.0):
        d = np.abs(p[m] - y[m])
        return float(np.where(d <= delta, 0.5 * d * d, delta * (d - 0.5 * delta)).mean(-1).mean())

    def contiguous(ep, k):
        """[T-k, N] mask: sample t+k continues the same episode as sample t."""
        return ep[k:] == ep[:len(ep) - k] + k

    def lag_scan(p, y, valid, ep, max_lag=10):
        """Positive lag: prediction follows the label by k control steps."""
        out = []
        for d in range(p.shape[-1]):
            best, corr = None, {}
            for k in range(-max_lag, max_lag + 1):
                a = abs(k)
                if k >= 0:
                    pp, yy = p[a:, :, d], y[:len(y) - a, :, d]
                    mm = valid[a:] & valid[:len(y) - a] & contiguous(ep, a)
                else:
                    pp, yy = p[:len(p) - a, :, d], y[a:, :, d]
                    mm = valid[a:] & valid[:len(y) - a] & contiguous(ep, a)
                c = float(np.corrcoef(pp[mm], yy[mm])[0, 1]) if mm.sum() > 10 else float("nan")
                corr[k] = c
                if best is None or c > corr[best]:
                    best = k
            out.append({"best_lag_steps": best, "best_lag_ms": best * env.dt * 1000, "corr_at_best": corr[best],
                        "corr_at_0": corr[0]})
        return out

    def segments(valid, ep, length):
        """Yield (n, t0) for non-overlapping windows fully valid and within one episode."""
        T, Nn = valid.shape
        for n in range(Nn):
            t = 0
            while t + length <= T:
                ok = valid[t:t + length, n].all() and (np.diff(ep[t:t + length, n]) == 1).all()
                if ok:
                    yield n, t
                    t += length
                else:
                    t += 1

    def band_analysis(p, y, valid, ep, rate, length, bands, sub=1):
        """Welch-style (Hann, no detrend) label and error energy per band; band R2 = 1 - E_err / E_label."""
        win = np.hanning(length)[:, None]
        freqs = np.fft.rfftfreq(length, 1.0 / rate)
        Ey, Ee, count = np.zeros((len(freqs), y.shape[-1])), np.zeros((len(freqs), y.shape[-1])), 0
        for n, t in segments(valid, ep, length // sub):
            yy = y[t * sub:(t * sub + length), n] if sub > 1 else y[t:t + length, n]
            Ey += np.abs(np.fft.rfft(yy * win, axis=0)) ** 2
            if p is not None:
                Ee += np.abs(np.fft.rfft((p[t:t + length, n] - yy) * win, axis=0)) ** 2
            count += 1
        res = {"segments": count}
        total = Ey.sum(0)
        for lo, hi in bands:
            sel = (freqs >= lo) & (freqs < hi)
            entry = {"label_energy_frac": (Ey[sel].sum(0) / np.maximum(total, 1e-12)).tolist()}
            if p is not None:
                entry["band_r2"] = (1 - Ee[sel].sum(0) / np.maximum(Ey[sel].sum(0), 1e-12)).tolist()
            res[f"{lo}-{hi}Hz"] = entry
        return res, freqs, Ey / max(count, 1), Ee / max(count, 1)

    def analyse(rec, name):
        pred = rec["pred"] * scale_np
        label = rec["label"][..., :3] * scale_np
        # ep >= 2: the first control step after reset contains the 2 warm-up physics substeps
        valid = (rec["label"][..., 3] > 0.5) & (rec["ep"] >= 2)
        sub = rec["sub"]                                     # [T, N, dec, 9] physical units
        avg = sub[..., :3].mean(2)
        last = sub[:, :, -1, :3]
        consistency = float(np.abs(last[valid] - label[valid]).max()) if valid.any() else float("nan")
        res = {"valid_frac": float(valid.mean()), "substep_label_consistency_max_abs": consistency,
               "label_spike_frac(|norm|>3)": float((np.abs(rec["label"][..., :3])[valid] > 3).any(-1).mean()),
               "label_abs_p99_phys": np.percentile(np.abs(label[valid]), 99, axis=0).tolist(),
               "label_abs_max_phys": np.abs(label[valid]).max(0).tolist(),
               "training_huber_loss": huber(rec["pred"], rec["label"][..., :3], valid),
               "overall_vs_label": r2_rmse(pred, label, valid),
               "overall_vs_substep_mean_label": r2_rmse(pred, avg, valid),
               "substep_mean_label_vs_label": r2_rmse(avg, label, valid),
               "lag_vs_label": lag_scan(pred, label, valid, rec["ep"]),
               "lag_vs_substep_mean_label": lag_scan(pred, avg, valid, rec["ep"])}
        cmd_mag = np.linalg.norm(rec["cmd"], axis=-1)
        recent = lambda flag: np.maximum.accumulate(np.where(flag, np.arange(len(flag))[:, None], -10 ** 6), 0)
        steps_since = lambda flag: np.arange(len(flag))[:, None] - recent(flag)
        within = lambda flag, s: steps_since(flag) <= int(s / env.dt)
        mass = rec["mass"]
        conditions = {
            "standing(|cmd_ref|<0.05)": cmd_mag < 0.05,
            "moving(|cmd_ref|>=0.05)": cmd_mag >= 0.05,
            "cart_pulse(<=0.5s since)": within(rec["pulse"], 0.5),
            "robot_push(<=0.5s since)": within(rec["push"], 0.5),
            "wheel_stall_active": rec["stall"],
            "no_disturbance(>1s since any)": ~within(rec["pulse"] | rec["push"] | rec["stall"], 1.0),
            "recovery_gate>0.05": rec["gate"] > 0.05,
            "first_0.5s_of_episode": rec["ep"] <= int(0.5 / env.dt),
        }
        for lo, hi in ((0, 20), (20, 30), (30, 60)):
            conditions[f"cart_mass_{lo}-{hi}kg"] = (mass >= lo) & (mass < hi)
        res["by_condition"] = {k: r2_rmse(pred, label, valid & m) for k, m in conditions.items() if (valid & m).sum() > 100}
        b50 = [(0, 0.5), (0.5, 2), (2, 5), (5, 10), (10, 25.01)]
        res["bands_50Hz"], freqs, Ey, Ee = band_analysis(pred, label, valid, rec["ep"], 1 / env.dt, 128, b50)
        # substep label at 200 Hz: how much energy lies above the 25 Hz control Nyquist (aliased into the label)
        T = sub.shape[0]
        sub_series = sub[..., :3].transpose(0, 2, 1, 3).reshape(T * dec, N, 3)
        b200 = [(0, 0.5), (0.5, 5), (5, 25), (25, 100.01)]
        res["bands_200Hz_label_only"], f200, Ey200, _ = band_analysis(None, sub_series, valid, rec["ep"], 1 / env.cfg.sim.dt,
                                                                      512, b200, sub=dec)
        hands = sub[:, :, -1, 3:]
        hv = hands[valid]
        res["per_hand"] = {
            "hand_force_std_N": hv.std(0).tolist(), "hand_force_mean_N": hv.mean(0).tolist(),
            "left_right_corr_fx_fy_fz": [float(np.corrcoef(hv[:, i], hv[:, 3 + i])[0, 1]) for i in range(3)],
            "sum_std_N_fx_fy_fz": (hv[:, :3] + hv[:, 3:]).std(0).tolist(),
            "half_diff_std_N_fx_fy_fz": (0.5 * (hv[:, :3] - hv[:, 3:])).std(0).tolist(),
            "vertical_sum_mean_N": float((hv[:, 2] + hv[:, 5]).mean()),
        }
        res["closed_loop"] = {"reward_per_step": float(rec["rew"].mean()),
                              "terminations": int(rec["term"].sum()),
                              "track_rmse_vx_vy_wz": np.sqrt(((rec["cartv"] - rec["cmd"]) ** 2).mean((0, 1))).tolist()}
        np.savez_compressed(out_dir / f"spectra_{name}.npz", freqs=freqs, label_psd=Ey, error_psd=Ee, f200=f200,
                            label_psd_200=Ey200)
        return res

    # ------------------------------------------------------------ offline estimators (own optimizer)
    class Est(nn.Module):
        def __init__(self, din, dout, hidden, bottleneck=None):
            super().__init__()
            layers, d = [], din
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            if bottleneck:
                layers += [nn.Linear(d, bottleneck), nn.ELU(), nn.Linear(bottleneck, 64), nn.ELU()]
                d = 64
            layers.append(nn.Linear(d, dout))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def offline(datasets):
        """Train on even env ids, test on odd; standardized targets, AdamW + cosine."""
        Xs, Ys, P, envs = [], [], [], []
        for rec in datasets:
            rows = rec["rows"]
            sub = rec["sub"][rows]                                   # [R, N, dec, 9]
            valid = (rec["label"][rows, :, 3] > 0.5) & (rec["ep"][rows] >= 2)
            label = rec["label"][rows, :, :3] * scale_np
            Y = np.concatenate((label, sub[..., :3].mean(2), sub[:, :, -1, 3:], sub[..., 3:].mean(2)), -1)
            Xs.append(rec["X"][torch.from_numpy(valid)])
            Ys.append(Y[valid])
            P.append((rec["pred"][rows] * scale_np)[valid])
            envs.append(np.broadcast_to(np.arange(N), valid.shape)[valid])
        X = torch.cat(Xs).to(dev)
        Y = torch.from_numpy(np.concatenate(Ys)).float().to(dev)
        P = np.concatenate(P)
        env_id = np.concatenate(envs)
        tr = torch.from_numpy(env_id % 2 == 0).to(dev)
        te = ~tr
        names = [f"label_last_{a}" for a in AXES] + [f"label_avg4_{a}" for a in AXES] + \
                [f"hand_last_{a}" for a in HAND_AXES] + [f"hand_avg4_{a}" for a in HAND_AXES]
        y_mu, y_sd = Y[tr].mean(0), Y[tr].std(0).clamp_min(1e-6)

        def score(pred_phys, target):
            err = ((pred_phys - target) ** 2).mean(0)
            var = target.var(0, unbiased=False)
            return {n: {"r2": float(1 - e / v), "rmse": float(math.sqrt(e))} for n, e, v in zip(names, err.tolist(), var.tolist())}

        results = {"train_rows": int(tr.sum()), "test_rows": int(te.sum())}
        # in-policy head on the same test rows (for the resultant targets)
        te_np = te.cpu().numpy()
        head = torch.from_numpy(np.concatenate((P, P), -1)[te_np]).to(dev)
        err = ((head - Y[te][:, :6]) ** 2).mean(0)
        var = Y[te][:, :6].var(0, unbiased=False)
        results["in_policy_head"] = {n: {"r2": float(1 - e / v), "rmse": float(math.sqrt(e))}
                                     for n, e, v in zip(names[:6], err.tolist(), var.tolist())}

        def run(tag, cols, hidden, bottleneck=None, linear=False):
            Xin = X if cols is None else X[:, cols]
            idx = tr.nonzero().flatten()
            s1 = torch.zeros(Xin.shape[1], device=dev, dtype=torch.float64)
            s2 = torch.zeros_like(s1)
            for j in idx.split(16384):
                x = Xin[j].double()
                s1 += x.sum(0)
                s2 += x.square().sum(0)
            mu = (s1 / len(idx)).float()
            sd = (s2 / len(idx) - (s1 / len(idx)) ** 2).clamp_min(0).sqrt().float().clamp_min(1e-2)
            model = (nn.Linear(Xin.shape[1], Y.shape[1]) if linear else Est(Xin.shape[1], Y.shape[1], hidden, bottleneck)).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            idx_tr = tr.nonzero().flatten()
            steps_per_epoch = max(1, len(idx_tr) // 4096)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * steps_per_epoch)
            for epoch in range(args.epochs):
                perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
                for b in range(steps_per_epoch):
                    i = perm[b * 4096:(b + 1) * 4096]
                    x = (Xin[i].float() - mu) / sd
                    loss = F.mse_loss(model(x), (Y[i] - y_mu) / y_sd)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()
            with torch.no_grad():
                idx_te = te.nonzero().flatten()
                out = torch.cat([model((Xin[j].float() - mu) / sd) for j in idx_te.split(16384)]) * y_sd + y_mu
            results[tag] = score(out, Y[idx_te])
            print(f"[force-eval] offline {tag}: " + ", ".join(
                f"{n}={results[tag][n]['r2']:.2f}" for n in names[:3] + names[6:9]), flush=True)

        H = X.shape[1] // FD
        full = None
        last5 = torch.arange((H - 5) * FD, H * FD, device=dev)
        last1 = torch.arange((H - 1) * FD, H * FD, device=dev)
        run("linear_20frames", full, None, linear=True)
        run("mlp_same_bottleneck_20frames", full, (256, 128), bottleneck=16)
        run("mlp_wide_20frames", full, (512, 256, 128))
        run("mlp_wide_5frames_falcon_like", last5, (512, 256, 128))
        run("mlp_wide_1frame", last1, (512, 256, 128))
        return results

    # ------------------------------------------------------------ policy-uses-channel swap test
    def swap_test():
        out = {}
        for mode in ("pred", "truth", "zero"):
            torch.manual_seed(args.seed)
            set_level()
            env.fixed_command = env.fixed_payload_kg = env.fixed_caster_angle = None
            obs, _ = env.reset()
            rew_sum, term, err2, steps = 0.0, 0, torch.zeros(3, device=dev), 0
            with torch.no_grad():
                for t in range(args.swap_steps):
                    truth = env.interaction_target[:, :3].clone()
                    action, _, _ = act(obs, mode, truth)
                    obs, rew, done, extras, _ = env_step(action)
                    rew_sum += float(rew.mean())
                    term += int(extras["terminated"].sum())
                    err2 += ((torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1) - env.cmd_ref) ** 2).mean(0)
                    steps += 1
            out[mode] = {"reward_per_step": rew_sum / steps, "terminations": term,
                         "track_rmse_vx_vy_wz": (err2 / steps).sqrt().tolist()}
            print(f"[force-eval] swap {mode}: {out[mode]}", flush=True)
        return out

    # ------------------------------------------------------------ known pulses / sustained resistance
    def pulse_test(command):
        set_level(disturbance=0.0, push=False)
        env.fixed_command, env.fixed_payload_kg, env.fixed_caster_angle = list(command), 12.5, 0.0
        obs, _ = env.reset()
        groups = {0: ("none", None), 1: ("+Fx 25N 0.2s", (25, 0, 0, .2)), 2: ("-Fx 25N 0.2s", (-25, 0, 0, .2)),
                  3: ("+Fy 25N 0.2s", (0, 25, 0, .2)), 4: ("-Fy 25N 0.2s", (0, -25, 0, .2)),
                  5: ("+Mz 3Nm 0.2s", (0, 0, 3, .2)), 6: ("-Mz 3Nm 0.2s", (0, 0, -3, .2)),
                  7: ("-Fx 20N sustained 3s", (-20, 0, 0, 3.0))}
        gid = torch.arange(N, device=dev) % len(groups)
        t0 = int(5.0 / env.dt)
        P, L, E = [], [], []
        with torch.no_grad():
            for t in range(args.pulse_steps):
                if t == t0:
                    yaw = env.ts.cart_yaw
                    for g, (_, spec) in groups.items():
                        if spec is None:
                            continue
                        ids = (gid == g).nonzero().flatten()
                        fx, fy, mz, dur = spec
                        c, s = torch.cos(yaw[ids]), torch.sin(yaw[ids])
                        force = torch.stack((c * fx - s * fy, s * fx + c * fy, torch.zeros_like(c)), -1)
                        torque = torch.zeros_like(force)
                        torque[:, 2] = mz
                        env.inject_cart_wrench(force, torque, dur, env_ids=ids)
                action, _, _ = act(obs, "det")
                obs, _, _, _, _ = env_step(action)
                _, pred = heads(obs)
                P.append((pred * scale_t).cpu())
                L.append((env.interaction_target[:, :3] * scale_t).cpu())
                E.append(env.episode_length_buf.clone().cpu())
        P, L, E = torch.stack(P).numpy(), torch.stack(L).numpy(), torch.stack(E).numpy()
        alive = (np.diff(E, axis=0) == 1).all(0)
        g_np = gid.cpu().numpy()
        res, traces = {}, {}
        win = slice(t0 - int(0.5 / env.dt), min(args.pulse_steps, t0 + int(4.0 / env.dt)))
        pre = t0 - win.start                              # window index of the onset
        for g, (name, spec) in groups.items():
            m = alive & (g_np == g)
            p, l = P[:, m].mean(1), L[:, m].mean(1)
            traces[name] = {"pred": p, "label": l}
            if spec is None:
                res[name] = {"envs": int(m.sum())}
                continue
            # each group is referenced to its own 0.5 s pre-onset mean (same envs, same payload/friction)
            dp, dl = p[win] - p[win][:pre].mean(0), l[win] - l[win][:pre].mean(0)
            ax = 2 if spec[2] else (1 if spec[1] else 0)
            post_l, post_p = dl[pre:, ax], dp[pre:, ax]
            peak_i = int(np.argmax(np.abs(post_l)))
            ppeak_i = int(np.argmax(np.abs(post_p)))
            thr = 0.5 * post_l[peak_i]
            cross = lambda s: next((i for i in range(len(s)) if abs(s[i]) >= abs(thr) and np.sign(s[i]) == np.sign(thr)), None)
            il, ip = cross(post_l), cross(post_p)
            per_env_err = (P[win][:, m, ax] - L[win][:, m, ax])
            res[name] = {"envs": int(m.sum()), "axis": AXES[ax],
                         "label_peak_delta": float(post_l[peak_i]), "label_peak_t_ms": peak_i * env.dt * 1000,
                         "pred_peak_delta": float(post_p[ppeak_i]), "pred_peak_t_ms": ppeak_i * env.dt * 1000,
                         "pred_to_label_peak_ratio": float(post_p[ppeak_i] / post_l[peak_i]),
                         "sign_correct": bool(np.sign(post_p[ppeak_i]) == np.sign(post_l[peak_i])),
                         "half_rise_lag_ms": None if il is None or ip is None else (ip - il) * env.dt * 1000,
                         "trace_corr": float(np.corrcoef(dp[:, ax], dl[:, ax])[0, 1]),
                         "mean_trace_rmse": float(np.sqrt(((dp - dl)[:, ax] ** 2).mean())),
                         "per_env_rmse_in_window": float(np.sqrt((per_env_err ** 2).mean()))}
            if spec[3] > 1.0:
                hold = slice(pre + int(1.0 / env.dt), pre + int(3.0 / env.dt))
                rel = slice(pre + int(3.5 / env.dt), pre + int(4.0 / env.dt))
                res[name]["hold_label_mean"] = float(dl[hold, ax].mean() if dl[hold].size else float("nan"))
                res[name]["hold_pred_mean"] = float(dp[hold, ax].mean() if dp[hold].size else float("nan"))
                res[name]["after_release_label_mean"] = float(dl[rel, ax].mean() if dl[rel].size else float("nan"))
                res[name]["after_release_pred_mean"] = float(dp[rel, ax].mean() if dp[rel].size else float("nan"))
        res["_alive_envs"] = int(alive.sum())
        return res, traces, t0

    # ------------------------------------------------------------ run everything
    report = {"checkpoint": str(args.checkpoint), "iteration": iteration, "saved_level": trained_level,
              "eval_level": level, "level_params": env.curriculum.params.as_dict(), "num_envs": N,
              "steps": args.steps, "control_dt_s": env.dt, "physics_dt_s": env.cfg.sim.dt,
              "wrench_scale": list(WRENCH_SCALE),
              "note": "Labels are subtree-compensated simulation loads (dynamic_contact_wrench_calibrated=False)."}
    det = rollout("det", args.steps, True)
    report["natural_deterministic"] = analyse(det, "det")
    stoch = rollout("stoch", args.steps, True)
    report["natural_stochastic"] = analyse(stoch, "stoch")
    np.savez_compressed(out_dir / "traces_det_first32.npz", pred=det["pred"][:, :32] * scale_np,
                        label=det["label"][:, :32], sub=det["sub"][:, :32], ep=det["ep"][:, :32],
                        cmd=det["cmd"][:, :32], latent=det["latent"][:, :32])
    report["offline_estimators"] = offline([det, stoch])
    del det["X"], stoch["X"]
    torch.cuda.empty_cache()
    report["channel_swap_closed_loop"] = swap_test()
    pulses = {}
    for label, command in (("moving_0.3", (0.3, 0.0, 0.0)), ("standing", (0.0, 0.0, 0.0))):
        res, traces, t0 = pulse_test(command)
        pulses[label] = res
        names = list(traces)
        np.savez_compressed(out_dir / f"pulse_{label}.npz", t0=t0, names=np.array(names),
                            **{f"g{i}_{kk}": vv for i, k in enumerate(names) for kk, vv in traces[k].items()})
    report["known_pulses"] = pulses
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred = det["pred"] * scale_np
        label = det["label"][..., :3] * scale_np
        valid = det["label"][..., 3] > 0.5
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for i, a in enumerate(AXES):
            sel = np.random.choice(valid.sum(), min(20000, valid.sum()), replace=False)
            axs[0, i].scatter(label[valid][sel, i], pred[valid][sel, i], s=1, alpha=0.2)
            lim = np.percentile(np.abs(label[valid][:, i]), 99.5)
            axs[0, i].plot([-lim, lim], [-lim, lim], "r--")
            axs[0, i].set_xlim(-lim, lim); axs[0, i].set_ylim(-lim, lim)
            axs[0, i].set_xlabel(f"label {a}"); axs[0, i].set_ylabel(f"pred {a}")
            t = np.arange(min(300, len(pred) // 3), min(550, len(pred)))
            axs[1, i].plot(t * env.dt, det["sub"][t, 0, -1, i], lw=.8, label="label (last substep)")
            axs[1, i].plot(t * env.dt, det["sub"][t, 0, :, i].mean(-1), lw=.8, label="label (4-substep mean)")
            axs[1, i].plot(t * env.dt, pred[t, 0, i], lw=1.2, label="pred")
            axs[1, i].set_xlabel("s"); axs[1, i].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out_dir / "scatter_and_trace.png", dpi=120); plt.close(fig)
        spec = np.load(out_dir / "spectra_det.npz")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for i, a in enumerate(AXES):
            axs[i].semilogy(spec["freqs"], spec["label_psd"][:, i], label="label PSD")
            axs[i].semilogy(spec["freqs"], spec["error_psd"][:, i], label="error PSD")
            axs[i].set_title(a); axs[i].set_xlabel("Hz"); axs[i].legend()
        fig.tight_layout(); fig.savefig(out_dir / "psd.png", dpi=120); plt.close(fig)
        for label_name in pulses:
            data = np.load(out_dir / f"pulse_{label_name}.npz")
            t0 = int(data["t0"])
            names = [str(n) for n in data["names"]]
            fig, axs = plt.subplots(2, 4, figsize=(18, 7))
            for j, n in enumerate(names):
                ax = axs.flat[j]
                t = (np.arange(len(data[f"g{j}_pred"])) - t0) * env.dt
                for i, c in enumerate("brg"):
                    ax.plot(t, data[f"g{j}_label"][:, i], c, lw=.7, alpha=.6)
                    ax.plot(t, data[f"g{j}_pred"][:, i], c + "--", lw=1.2)
                ax.set_xlim(-1, 4.5); ax.set_title(n, fontsize=9)
            fig.suptitle(f"{label_name}: solid=label, dashed=pred; blue Fx, red Fy, green Mz")
            fig.tight_layout(); fig.savefig(out_dir / f"pulse_{label_name}.png", dpi=110); plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"[force-eval] plotting skipped: {exc}", flush=True)
    print(f"[force-eval] wrote {out_dir}", flush=True)
finally:
    launcher.app.close()
