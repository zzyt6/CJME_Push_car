"""New Isaac closed-loop force evaluation, six axes scored separately.

Twelve new command/payload conditions run in isolated parallel environments.
All physics and actuation follow the existing nominal evaluation contract.
No optimizer, fitting, model selection, or old evaluation samples are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
CHECKPOINT = ROOT / "logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt"
HEADS = {
    "full": ("explicit_pd", ROOT / "logs/bench/v52_falcon_explicit/model_explicit_pd_full_22/estimator.pt"),
    "fullbody": ("explicit_pd", ROOT / "logs/bench/v52_falcon_explicit/falcon_a3/model_explicit_pd_falcon_fullbody_11/estimator.pt"),
    "original": ("training", None),
}
COMMANDS = {"forward025": [.25, 0., 0.], "forward050": [.50, 0., 0.],
            "left020": [.35, 0., .20], "right020": [.35, 0., -.20]}
PAYLOADS = (5., 15., 22.5)
CONDITIONS = [dict(condition_id=i, payload_kg=payload, command_name=name, command=command)
              for i, (payload, name, command) in enumerate((p, n, c) for p in PAYLOADS for n, c in COMMANDS.items())]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def evaluate(args):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from homie_pushcart_v52.control_contract import nominal_configs, control_contract
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.hl_policy import HighLevelPolicy
    from homie_pushcart_v52.transfer_estimator import install_estimator
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE
    from summarize_v52_force_axis_accuracy import component_metrics

    class FixedGridEnv(PushCartEnv):
        """Assign conditions through the production fixed-payload/command hooks.

        Grouped resets only choose scalar parameters before calling the original
        reset implementations; they do not reimplement mass/inertia or controls.
        """
        def _randomize_cart(self, env_ids):
            if not hasattr(self, "test_payloads"):
                return super()._randomize_cart(env_ids)
            previous = self.fixed_payload_kg
            try:
                for payload in PAYLOADS:
                    selected = env_ids[self.test_payloads[env_ids.long()] == payload]
                    if len(selected):
                        self.fixed_payload_kg = payload
                        super()._randomize_cart(selected)
            finally:
                self.fixed_payload_kg = previous

        def _sample_task_command(self, env_ids):
            if not hasattr(self, "test_commands"):
                return super()._sample_task_command(env_ids)
            previous = self.fixed_command
            try:
                for command in COMMANDS.values():
                    choice = torch.tensor(command, device=self.device)
                    selected = env_ids[(self.test_commands[env_ids.long()] == choice).all(-1)]
                    if len(selected):
                        self.fixed_command = command
                        super()._sample_task_command(selected)
            finally:
                self.fixed_command = previous

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    drive, estimator = HEADS[args.head]
    files = [CHECKPOINT, ROOT / "assets/g1_pushcart_v5/g1_pushcart.usd", Path(__file__)]
    if estimator is not None:
        files.append(estimator)
    before = [fingerprint(p) for p in files]
    cfg, task = nominal_configs(CHECKPOINT, drive, args.steps * .02 + 5.)
    cfg.env.num_envs = len(CONDITIONS)
    low = Path(cfg.pushcart_saved_checkpoint)
    if not low.is_absolute():
        low = ROOT / low
    before.append(fingerprint(low))
    files.append(low)
    env = FixedGridEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=low)
    env.test_payloads = torch.tensor([c["payload_kg"] for c in CONDITIONS], device=args.device)
    env.test_commands = torch.tensor([c["command"] for c in CONDITIONS], device=args.device)
    env.fixed_payload_kg = 0.
    env.fixed_command = [0., 0., 0.]  # fixed-command semantics, including subset resets
    env.fixed_caster_angle = 0.
    policy = HighLevelPolicy(CHECKPOINT, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    if estimator is not None:
        install_estimator(policy.actor, estimator)
    scale = torch.tensor(HAND_WRENCH_SCALE, device=args.device).repeat(2)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
    failures = [None] * env.num_envs
    invalid = np.zeros(env.num_envs, dtype=int)
    warmup = round(args.warmup_s / env.dt)
    assert abs(env.dt - .02) < 1e-8 and warmup < args.steps
    records = {key: [] for key in ("x", "falcon_x", "prediction", "target", "condition_id", "time_s",
                                   "cart_mass_kg", "command_task", "hand_substep_count")}
    started = time.time()
    with torch.inference_mode():
        obs, _ = env.reset()
        np.testing.assert_allclose(env.cart_mass_total.cpu(), env.test_payloads.cpu() + 14.4, atol=1e-4, rtol=0)
        np.testing.assert_allclose(env.cmd_task.cpu(), env.test_commands.cpu(), atol=1e-7, rtol=0)
        reset_mass = env.cart_mass_total.cpu().tolist()
        for step in range(args.steps):
            obs, _, _, done, _ = env.step(policy(obs, estimator_obs=env.falcon_obs))
            newly_failed = active & done.bool()
            for index in newly_failed.nonzero().flatten().cpu().tolist():
                failures[index] = dict(step=step + 1, time_s=(step + 1)*env.dt, reason="Isaac termination/reset")
            active &= ~done.bool()
            if step >= warmup:
                td = TensorDict({"policy": obs, "falcon": env.falcon_obs}, batch_size=[env.num_envs])
                prediction = policy.actor.predict_hand_wrench(td) * scale
                target = env.interaction_target[:, :12] * scale
                finite = torch.isfinite(prediction).all(-1) & torch.isfinite(target).all(-1) & torch.isfinite(obs[:, -680:]).all(-1)
                label_valid = env.interaction_target[:, 12] > .5
                valid = active & label_valid & finite
                invalid += (active & ~valid).cpu().numpy().astype(int)
                values = dict(x=obs[:, -680:], falcon_x=env.falcon_obs, prediction=prediction, target=target,
                              condition_id=torch.arange(env.num_envs, device=args.device),
                              time_s=torch.full((env.num_envs,), (step+1)*env.dt, device=args.device, dtype=torch.float64),
                              cart_mass_kg=env.cart_mass_total, command_task=env.cmd_task,
                              hand_substep_count=env.hand_wrench_count)
                for key, value in values.items():
                    records[key].append(value[valid].detach().cpu().clone())
            if (step + 1) % 50 == 0:
                progress = dict(head=args.head, step=step+1, requested_steps=args.steps,
                                active_conditions=int(active.sum()), scored_samples=sum(len(x) for x in records["x"]),
                                elapsed_s=time.time()-started)
                write(args.output / "progress.json", progress)
                print("PROGRESS", json.dumps(progress), flush=True)
    data = {key: torch.cat(value) for key, value in records.items()}
    assert len(data["x"]), "No valid newly collected samples"
    prediction = data["prediction"].double().numpy().reshape(-1, 2, 6)[:, :, :3]
    target = data["target"].double().numpy().reshape(-1, 2, 6)[:, :, :3]
    ids = data["condition_id"].numpy()
    rows = []
    for c in CONDITIONS:
        i = c["condition_id"]
        mask = ids == i
        rows.append(dict(**c, samples=int(mask.sum()), failure=failures[i], invalid_scored_steps=int(invalid[i]),
                         actual_reset_cart_mass_kg=reset_mass[i],
                         axes=component_metrics(prediction[mask], target[mask]) if mask.any() else {}))
    after = [fingerprint(p) for p in files]
    assert before == after, "A source asset or model changed during evaluation"
    report = dict(schema="v52_fresh_isaac_axes_v1", engine="isaac_physx", head=args.head,
                  new_collection=True, started_utc=datetime.fromtimestamp(started, timezone.utc).isoformat(),
                  seed=args.seed, upper_drive=drive, num_envs=env.num_envs, parallel_conditions=True,
                  steps=args.steps, control_dt=env.dt, warmup_s=args.warmup_s,
                  samples=len(ids), invalid_scored_steps=int(invalid.sum()),
                  completed_conditions=sum(f is None for f in failures), total_conditions=len(CONDITIONS),
                  observation_noise=False, observation_delay=False,
                  frame="Hand-on-cart force, cart-heading axes, physical N; interval-mean reference.",
                  source_hashes=before, unchanged_sources=True,
                  elapsed_s=time.time()-started,
                  control=control_contract(cfg, env.dof_names, env.torque_limits.cpu().tolist(), drive),
                  physics=env.physics_report, conditions=rows, axes=component_metrics(prediction, target),
                  limits=["New evaluation grid, not a new training dataset or retraining.",
                          "Nominal/noiseless simulation and a single seed; adjacent samples correlated.",
                          "Each head controls its own trajectories; different heads are not a one-factor architecture comparison.",
                          "No same-new-grid MuJoCo data was collected in this run."])
    torch.save(dict(schema="v52_fresh_isaac_samples_v1", report=report, **data), args.output / "samples.pt")
    np.savez_compressed(args.output / "forces.npz", prediction=prediction, target=target, condition_id=ids, time_s=data["time_s"].numpy())
    write(args.output / "report.json", report)
    print("COMPLETE", args.head, report["samples"], report["completed_conditions"], flush=True)


def driver(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", dict(schema="v52_fresh_isaac_axes_manifest_v1",
          created_utc=datetime.now(timezone.utc).isoformat(), conditions=CONDITIONS, seed=args.seed,
          steps=args.steps, warmup_s=args.warmup_s, heads=list(HEADS),
          note="New simulator trajectories; new payloads and commands relative to the previous 12-condition evaluation. Frozen existing weights."))
    for head in HEADS:
        destination = args.output / head
        destination.mkdir()
        command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                   "--head", head, "--output", str(destination), "--seed", str(args.seed),
                   "--steps", str(args.steps), "--warmup_s", str(args.warmup_s), "--viz", "none"]
        with (destination / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            state = dict(head=head, pid=process.pid, command=command, started_unix=time.time(), status="running")
            write(args.output / "job.json", state)
            print("START", head, "PID", process.pid, flush=True)
            code = process.wait()
        state.update(exit_code=code, status="complete" if code == 0 else "failed", ended_unix=time.time())
        write(args.output / "job.json", state)
        print("END", head, code, flush=True)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=tuple(HEADS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup_s", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/bench/v52_isaac_fresh_axes_20260927")
    preliminary, _ = parser.parse_known_args()
    if preliminary.head:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        launcher = AppLauncher(args)
        try:
            evaluate(args)
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            launcher.app.close()
    else:
        driver(parser.parse_args())
