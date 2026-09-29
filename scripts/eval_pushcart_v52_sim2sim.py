"""Frozen-checkpoint force-estimator evaluation in legacy/aligned MuJoCo or Isaac.

No training or estimator-dependent scene tuning. The default retains the original
nominal grasp and URDF inertia. Proposed corrections are explicit experiments,
pending joint-mapping review. Labels still come from actual cart interactions.
Example: python scripts/eval_pushcart_v52_sim2sim.py --engine mujoco --steps 500
Isaac: add --engine isaac --viz none (run separately, with the same conditions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
DEFAULT_CKPT = ROOT / "logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt"
COMMANDS = {"forward03": [.3, 0., 0.], "forward06": [.6, 0., 0.],
            "left": [.3, 0., .3], "right": [.3, 0., -.3]}


def metrics(pred, target):
    import numpy as np
    p, y = np.asarray(pred), np.asarray(target)
    if not p.size or not y.size:
        return {"n": 0}
    valid = np.isfinite(p).all(-1) & np.isfinite(y).all(-1)
    p, y = p[valid], y[valid]
    if not len(y):
        return {"n": 0}
    mse, var = ((p - y)**2).mean(0), y.var(0)
    return dict(n=len(y), rmse=np.sqrt(mse).tolist(),
                r2=[float(1 - e / v) if v > 1e-10 else None for e, v in zip(mse, var)],
                bias=(p-y).mean(0).tolist(), label_std=np.sqrt(var).tolist())


def summarize(profile, rows, metadata):
    import numpy as np
    valid_rows = [r for r in rows if len(r["prediction"])]
    output = dict(profile=profile, metadata=metadata, conditions=[])
    for row in rows:
        result = {k: v for k, v in row.items() if k not in ("prediction", "target", "pred_handle", "true_handle")}
        result["hands"] = metrics(row["prediction"], row["target"])
        result["handle"] = metrics(row["pred_handle"], row["true_handle"])
        output["conditions"].append(result)
    if valid_rows:
        for key, p, y in (("hands", "prediction", "target"), ("handle", "pred_handle", "true_handle")):
            output[key] = metrics(np.concatenate([r[p] for r in valid_rows]), np.concatenate([r[y] for r in valid_rows]))
    output["completed_conditions"] = sum(not r["failure"] for r in rows)
    output["total_conditions"] = len(rows)
    return output


def mujoco_run(args):
    import numpy as np
    import torch
    from pushcart_bench.policies_v5x import V5xPolicy
    from pushcart_bench.extra_metrics import Recorder
    from pushcart_bench.sim2sim_scene import model_report
    torch.set_num_threads(1)
    for profile in args.profiles:
        policy = V5xPolicy("homie_pushcart_v52", args.checkpoint, scene_profile=profile, upper_drive=args.upper_drive,
                          grasp_profile=getattr(args, 'grasp_profile', None),
                          inertia_profile=getattr(args, 'inertia_profile', None))
        if args.estimator:
            from homie_pushcart_v52.transfer_estimator import install_estimator
            install_estimator(policy.runner.actor, args.estimator)
        policy.env.audit_newton_euler = args.audit_labels and profile == "isaac"
        recorder = Recorder(policy)
        rows = []
        label_differences = []
        captures = []
        for payload in args.payloads:
            for name in args.commands:
                cmd = COMMANDS[name]
                policy.reset(payload, seed=args.seed)
                failure = None
                observations, falcon_observations, tracking = [], [], []
                for step in range(args.steps):
                    policy.step(cmd)
                    if args.capture:
                        observations.append(policy.runner.hl_obs[-680:].copy())
                        falcon_observations.append(policy.runner.falcon_obs[0].numpy().copy())
                    state = policy.runner.ts
                    tracking.append(np.r_[state['cart_v'], state['cart_wz']] - policy.runner.cmd_ref)
                    if policy.env.sub_newton_euler:
                        label_differences.extend((np.asarray(policy.env.sub_newton_euler) - policy.env.sub_hands).reshape(-1, 6))
                    failure = policy.failure()
                    if failure:
                        break
                a = recorder.arrays()
                start = min(round(args.warmup_s / policy.dt), len(a["yaw"]))
                target = a["hands_c"][start:].reshape(-1, 12)
                # Match training: mean per-hand labels, then fixed grasp geometry.
                actor = policy.runner.actor
                with torch.no_grad():
                    true_handle = actor.compose_resultant(torch.tensor(target / policy.runner.hand_scale, dtype=torch.float32)).numpy() * policy.runner.wrench_scale
                rows.append(dict(condition=name, payload_kg=payload, steps=step+1, failure=failure,
                                 prediction=a["pred_hands"][start:], target=target,
                                 pred_handle=a["pred_resultant"][start:], true_handle=true_handle,
                                 tracking_rmse=np.sqrt(np.mean(np.asarray(tracking)[start:]**2, axis=0)).tolist()
                                 if len(tracking)>start else None))
                if args.capture:
                    captures.append((np.asarray(observations)[start:], target / policy.runner.hand_scale,
                                     np.asarray(falcon_observations)[start:]))
                print(f"[{profile}] {payload:g} kg {name}: {step+1} steps, failure={failure}", flush=True)
        report = summarize(profile, rows, model_report(policy.env))
        report.update(engine="mujoco", checkpoint=str(args.checkpoint), homie_checkpoint=str(policy.homie_checkpoint),
                      upper_drive=policy.env.upper_drive, seed=args.seed, steps=args.steps, warmup_s=args.warmup_s,
                      observation_noise=False, observation_delay=False,
                      estimator_override=str(args.estimator) if args.estimator else None,
                      hand_axes=[f"{s}_{a}" for s in "LR" for a in ("fx", "fy", "fz", "mx", "my", "mz")],
                      note="Closed-loop trajectories differ by scene. Failed conditions are retained; no retraining.")
        if label_differences:
            difference = np.array(label_differences)
            report["newton_euler_minus_weld"] = dict(samples=len(difference),
                rmse=np.sqrt((difference**2).mean(0)).tolist(), max_abs=np.max(abs(difference), axis=0).tolist(),
                note="Independent arm RNE balance at the same solver sample; moment differences include soft-weld anchor separation.")
        name = profile if policy.env.upper_drive == "training" else f"{profile}_{policy.env.upper_drive}"
        if policy.env.grasp_profile != 'nominal':
            name += '_' + policy.env.grasp_profile
        if policy.env.inertia_profile != 'urdf':
            name += '_' + policy.env.inertia_profile + '_inertia'
        write(args, name, report)
        if args.capture:
            save_capture(args, name, captures, report)


def write(args, name, report):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(name, "handle", report.get("handle"), flush=True)


def save_capture(args, name, captures, report):
    import numpy as np
    import torch
    valid = [(i, x, y, f) for i, (x, y, f) in enumerate(captures) if len(x)]
    if not valid:
        raise ValueError('No evaluation samples to capture')
    torch.save(dict(schema='v52_transfer_eval_v1',
                    x=torch.from_numpy(np.concatenate([x for _, x, _, _ in valid]).astype(np.float32)),
                    falcon_x=torch.from_numpy(np.concatenate([f for _, _, _, f in valid]).astype(np.float32)),
                    y=torch.from_numpy(np.concatenate([y for _, _, y, _ in valid]).astype(np.float32)),
                    condition_id=torch.from_numpy(np.concatenate([np.full(len(x), i) for i, x, _, _ in valid])),
                    report=report), args.output / f'{name}_samples.pt')


def isaac_run(args, launcher):
    import numpy as np
    import torch
    from tensordict import TensorDict
    from homie_pushcart_v52.control_contract import nominal_configs, control_contract
    from homie_pushcart_v52.task_env import PushCartEnv
    from homie_pushcart_v52.hl_policy import HighLevelPolicy
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE, WRENCH_SCALE
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    cfg, task = nominal_configs(args.checkpoint, args.upper_drive, args.steps * .02 + 5.)
    low_checkpoint = getattr(cfg, "pushcart_saved_checkpoint", None)
    if low_checkpoint:
        low_checkpoint = Path(low_checkpoint)
        if not low_checkpoint.is_absolute():
            low_checkpoint = ROOT / low_checkpoint
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=low_checkpoint)
    env.physics_report['control'] = control_contract(cfg, env.dof_names, env.torque_limits.cpu().tolist(), env.upper_drive)
    report_name = "physx" if env.upper_drive == "training" else f"physx_{env.upper_drive}"
    env.fixed_caster_angle = 0.
    import warp as wp
    env.physics_report['robot_materials'] = wp.to_torch(env.robot.root_view.get_material_properties())[0].cpu().tolist()
    env.physics_report['robot_joint_friction'] = env.robot.data.joint_friction_coeff.torch[0, env.joint_ids].cpu().tolist()
    env.physics_report['cart_armature'] = env.cart.data.joint_armature.torch[0].cpu().tolist()
    env.physics_report['cart_joint_friction'] = env.cart.data.joint_friction_coeff.torch[0].cpu().tolist()
    hl = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
    if args.estimator:
        from homie_pushcart_v52.transfer_estimator import install_estimator
        install_estimator(hl.actor, args.estimator)
    scale = torch.tensor(HAND_WRENCH_SCALE, device=args.device).repeat(2)
    wrench_scale = torch.tensor(WRENCH_SCALE, device=args.device)
    rows = []
    captures = []
    for payload in args.payloads:
        for name in args.commands:
            cmd = COMMANDS[name]
            env.fixed_payload_kg, env.fixed_command = payload, cmd
            with torch.inference_mode():
                obs, _ = env.reset()
            p, y, ph, yh = [], [], [], []
            observations, falcon_observations, tracking = [], [], []
            failure = None
            for step in range(args.steps):
                with torch.inference_mode():
                    action = hl(obs, estimator_obs=env.falcon_obs)
                    obs, _, _, done, _ = env.step(action)
                    if bool(done[0]):
                        failure = "Isaac termination/reset"
                        break
                    if step * env.dt < args.warmup_s or env.interaction_target[0, 12] < .5:
                        continue
                    td = TensorDict({"policy": obs, "falcon": env.falcon_obs}, batch_size=[1])
                    pred = hl.actor.predict_hand_wrench(td)
                    target = env.interaction_target[:, :12]
                    p.append((pred[0] * scale).cpu().numpy())
                    y.append((target[0] * scale).cpu().numpy())
                    ph.append((hl.actor.compose_resultant(pred)[0] * wrench_scale).cpu().numpy())
                    yh.append((hl.actor.compose_resultant(target)[0] * wrench_scale).cpu().numpy())
                    tracking.append((torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1)[0] - env.cmd_ref[0]).cpu().numpy())
                    if args.capture:
                        observations.append(obs[0, -680:].cpu().numpy().copy())
                        falcon_observations.append(env.falcon_obs[0].cpu().numpy().copy())
            rows.append(dict(condition=name, payload_kg=payload, steps=step+1, failure=failure,
                             prediction=p, target=y, pred_handle=ph, true_handle=yh,
                             tracking_rmse=np.sqrt(np.mean(np.asarray(tracking)**2, axis=0)).tolist() if tracking else None))
            if args.capture:
                captures.append((np.asarray(observations), np.asarray(y).reshape(-1, 12) / scale.cpu().numpy(),
                                 np.asarray(falcon_observations)))
            print(f"[PhysX] {payload:g} kg {name}: {step+1} steps, failure={failure}", flush=True)
            # Preserve completed conditions even if the native simulator fails
            # during a later reset; a partial report is explicitly marked.
            partial = summarize("physx", rows, env.physics_report)
            partial.update(checkpoint=str(args.checkpoint), partial=True,
                           requested_conditions=len(args.payloads)*len(args.commands))
            write(args, report_name + "_partial", partial)
    report = summarize("physx", rows, env.physics_report)
    report.update(engine="isaac_physx", checkpoint=str(args.checkpoint), homie_checkpoint=str(env.policy.checkpoint),
                  upper_drive=env.upper_drive, seed=args.seed, steps=args.steps,
                  warmup_s=args.warmup_s, observation_noise=False,
                  observation_delay=False, note="Newton-Euler subtree proxy labels; deterministic nominal model.")
    report['estimator_override'] = str(args.estimator) if args.estimator else None
    write(args, report_name, report)
    if args.capture:
        save_capture(args, report_name, captures, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--upper_drive", choices=("training", "explicit_pd"), default=None,
                        help="default restores the checkpoint; explicit_pd is a separate drive-transfer experiment")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--estimator", type=Path, help="attach a trained standalone force estimator to the actual actor")
    parser.add_argument("--capture", action="store_true", help="save held-out raw estimator samples for same-trajectory comparison")
    parser.add_argument("--profiles", nargs="+", choices=("legacy", "isaac"), default=["legacy", "isaac"])
    parser.add_argument("--grasp_profile", choices=("nominal", "rigid_limit"), default=None,
                        help="default: nominal; rigid_limit is an explicit grasp-impedance experiment")
    parser.add_argument("--inertia_profile", choices=("urdf", "physx"), default=None,
                        help="default: urdf; physx is an explicit runtime-inertia compatibility experiment")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--payloads", type=float, nargs="+", default=[0., 12.5, 25.])
    parser.add_argument("--commands", nargs="+", choices=tuple(COMMANDS), default=list(COMMANDS))
    parser.add_argument("--audit_labels", action="store_true", help="independent RNE balance vs weld labels on aligned MuJoCo")
    parser.add_argument("--warmup_s", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/bench/v52_sim2sim_alignment")
    preliminary, _ = parser.parse_known_args()
    if preliminary.inertia_profile == 'physx' and (preliminary.engine != 'mujoco' or preliminary.profiles != ['isaac']):
        parser.error('Use --engine mujoco --profiles isaac for PhysX inertia compatibility')
    if preliminary.grasp_profile == 'rigid_limit' and (preliminary.engine != 'mujoco' or preliminary.profiles != ['isaac']):
        parser.error('Use --engine mujoco --profiles isaac for the rigid_limit grasp diagnostic')
    if preliminary.engine == "mujoco" and preliminary.upper_drive == "explicit_pd" and "legacy" in preliminary.profiles:
        parser.error("Use --profiles isaac with --upper_drive explicit_pd; legacy is the frozen historical scene")
    if preliminary.engine == "isaac":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        launcher = AppLauncher(args)
        try:
            isaac_run(args, launcher)
        except BaseException:
            # Kit shutdown can terminate the process before Python prints an
            # outstanding exception; preserve the original failure first.
            import traceback
            traceback.print_exc()
            raise
        finally:
            launcher.app.close()
    else:
        mujoco_run(parser.parse_args())


if __name__ == "__main__":
    main()
