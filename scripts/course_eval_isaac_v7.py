"""Isaac / PhysX twin of the MuJoCo bench (2026-10-08): the same 20 m composite course and follower in the V7 training
plant, to tell a sim2sim gap from a policy weakness. Usage:
    E:\envs\RLisaac601\python.exe scripts/course_eval_isaac_v7.py --policy V7 --checkpoint <model.pt> --payload 25 --headless

20 m composite course in Isaac on the V7 plant (realistic grasp, Coulomb cart, d = 1 physics ranges), driven by
the official bench path follower (pushcart_bench, benchmark.yaml gains). A run fails the moment the episode
terminates (grip lost: over capacity 0.4 s / levered past the break angle / slid 8 cm; fall; cart tilt...)."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
CKPT = {
    "V3.2": ROOT / "logs/rsl_rl/homie_pushcart/2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt",
    "V6": ROOT / "logs/rsl_rl/homie_pushcart_v6/2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt",
}


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("HOMIE", "V3.2", "V6", "V7", "V8"))
    p.add_argument("--mix", nargs="+", default=None,
                   help="POLICY=CHECKPOINT ... : one Isaac launch, environment i runs entry i %% len (V6 / V7 only)")
    p.add_argument("--checkpoint", default="", help="V6 / V7 checkpoint (default V6: model_19999)")
    p.add_argument("--tag", default="")
    p.add_argument("--difficulty", type=float, default=1.0,
                   help="1: full physics randomization and observation noise; 0: physics within 25 %% of nominal, no noise")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--payload", type=float, default=15.0)
    p.add_argument("--disturbances", choices=("off", "on"), default="off")
    p.add_argument("--zero_leg_residual", action="store_true", help="V8: force the 12 leg-residual actions to zero")
    p.add_argument("--stop_before_goal", type=float, default=0.5,
                   help="end a run this arc length before the goal (m; 0 = drive to the goal and settle)")
    p.add_argument("--upper_gain_scale", type=float, default=1.0, help="scale upper-body drive stiffness / damping")
    p.add_argument("--task_set", nargs="*", default=[], help="task overrides section.key=<json>, e.g. "
                                                              "grasp.friction_coeff_range=[0.8,0.8]")
    p.add_argument("--save_frames", type=int, default=0, help="V8: save the observation frames + explicit truth of "
                                                                "the first N envs (input-shift analysis)")
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import dataclasses
        import numpy as np
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv, OVERLOAD_KINDS
        from pushcart_bench.config_io import load_config
        from pushcart_bench.follower import FollowerConfig, PathFollower
        from pushcart_bench.paths import build_path
        dev, N = a.device, a.num_envs
        bench = load_config()
        fcfg = FollowerConfig.from_dict({**bench["follower"], "cruise": True, "timeout_s": 90.0})   # as the documented course check
        course = build_path("composite_course", bench["paths"]["composite_course"])
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = N
        if a.upper_gain_scale != 1.0:                             # diagnostic: arm / waist drive compliance
            for k in ("waist", "shoulder", "elbow", "wrist", "hand"):
                cfg.control.stiffness[k] = cfg.control.stiffness[k] * a.upper_gain_scale
                cfg.control.damping[k] = cfg.control.damping[k] * a.upper_gain_scale
            print(f"[course] upper drive gains x{a.upper_gain_scale}", flush=True)
        if a.policy == "V8":
            # V8 teacher: the run's saved task (V7.1 plant), its paired fine-tuned HOMIE velocity estimator (frozen)
            from homie_pushcart_v8.task_config import task_cfg_from_run
            from homie_pushcart_v8.task_env import PushCartEnv as V8Env
            from homie_pushcart_v8.training import homie_velocity_file
            from homie_pushcart.hl_policy import import_rsl_rl5
            import_rsl_rl5()
            from homie_pushcart_v8.student_policy import is_student_run
            v8_student = is_student_run(a.checkpoint)
            task = task_cfg_from_run(a.checkpoint)
            est = homie_velocity_file(a.checkpoint)
            if est is None and task.v8.homie_velocity.init_checkpoint:     # student run: the teacher's estimator
                est = Path(task.v8.homie_velocity.init_checkpoint)
            task.v8.emit_history = v8_student
            task.v8.homie_velocity.init_checkpoint = str(est) if est is not None else ""
            task.v8.homie_velocity.train = False
            print(f"[course] V8 HOMIE velocity estimator: {est}", flush=True)
            EnvClass = V8Env
        else:
            task = PushCartTaskCfg()
            EnvClass = PushCartEnv
        task.curriculum_v7.enabled = False
        for item in a.task_set:                                   # diagnostic physics overrides
            key, val = item.split("=", 1)
            obj = task
            *path, leaf = key.split(".")
            for part in path:
                obj = getattr(obj, part)
            if not hasattr(obj, leaf):
                raise ValueError(f"unknown task field {key}")
            setattr(obj, leaf, json.loads(val))
            print(f"[course] task.{key} = {val}", flush=True)
        task.env.episode_length_s = fcfg.timeout_s + 10.0
        env = EnvClass(cfg, task, ROOT, dev, True, level=0)
        env.cur.d[:] = a.difficulty
        env._refresh_global_level()
        env.fixed_payload_kg = a.payload
        env.fixed_command = [0., 0., 0.]
        env.reset_idx(torch.arange(N, device=dev)); env.reset()

        def quiet():
            env.disturbance_frac[:] = 0.; env.bias_force_w[:] = 0.; env.bump_force_w[:] = 0.; env.bump_timer[:] = 0
            env.push_scale[:] = 0.
            if dataclasses.is_dataclass(env.level_params):
                env.level_params = dataclasses.replace(env.level_params, disturbance_frac=0.0)
        if a.disturbances == "off":
            quiet()
        snap = {}
        orig_ct = env.check_termination

        def check_termination():
            orig_ct()
            snap["kinds"] = env.grasp_overload_kind.clone()
            snap["reasons"] = {k: v.clone() for k, v in env.term_reason_buf.items()}
        env.check_termination = check_termination
        if a.mix:
            from homie_pushcart_v6.hl_policy import HighLevelPolicy as P6
            from homie_pushcart_v7.hl_policy import HighLevelPolicy as P7
            entries = [m.split("=", 1) for m in a.mix]
            pols = [(P6 if name == "V6" else P7)(Path(ck), device=dev) for name, ck in entries]
            owner = torch.arange(N, device=dev) % len(pols)

            def act_fn(o):
                out = torch.zeros(N, 18, device=dev)
                for k, pol in enumerate(pols):
                    out[owner == k] = pol(o["policy"], o["estimator"])[owner == k]
                return out
            a.policy = "+".join(name for name, _ in entries)
        elif a.policy == "V3.2":
            from homie_pushcart.hl_policy import HighLevelPolicy as P32
            pol = P32(CKPT["V3.2"], 620, 652, 18, device=dev)
            act_fn = lambda o: pol(env.legacy_actor_obs)                                   # noqa: E731
        elif a.policy == "V6":
            from homie_pushcart_v6.hl_policy import HighLevelPolicy as P6
            pol = P6(Path(a.checkpoint) if a.checkpoint else CKPT["V6"], device=dev)
            act_fn = lambda o: pol(o["policy"], o["estimator"])                            # noqa: E731
        elif a.policy == "V7":
            from homie_pushcart_v7.hl_policy import HighLevelPolicy as P7
            pol = P7(Path(a.checkpoint), device=dev)
            act_fn = lambda o: pol(o["policy"], o["estimator"])                            # noqa: E731
        elif a.policy == "V8":
            from homie_pushcart.hl_policy import import_rsl_rl5
            import_rsl_rl5()
            if v8_student:                       # deployable student: (policy, 32-frame history)
                from homie_pushcart_v8.student_policy import load_student
                spol = load_student(a.checkpoint, dev)
                a.tag += "_student"
                pol = lambda policy, privileged: spol(policy, env.observation_dict()["history"])   # noqa: E731
            else:                                # ROA teacher (simulation only): (policy, privileged)
                from homie_pushcart_v8.teacher_policy import load_teacher
                pol = load_teacher(a.checkpoint, dev)
            if a.zero_leg_residual:
                def act_fn(o):
                    out = pol(o["policy"], o["privileged"]).clone()
                    out[:, 18:] = 0.0
                    return out
                a.tag += "_legres0"
            else:
                act_fn = lambda o: pol(o["policy"], o["privileged"])                        # noqa: E731
        else:
            act_fn = lambda o: torch.zeros(N, 18, device=dev)                              # noqa: E731
        # settle 1 s standing, then start every env's course at its own cart pose
        for _ in range(50):
            with torch.inference_mode():
                act = act_fn(env.observation_dict())
            env.step(torch.as_tensor(act, device=dev).clone())
        pose = env.cart.data.root_link_pose_w.torch[:, :3].cpu().numpy()
        yaw0 = env.ts.cart_yaw.cpu().numpy()
        followers = [PathFollower(course.transformed(pose[i, :2], float(yaw0[i])), fcfg, env.dt) for i in range(N)]
        env.fixed_command = None
        active = np.ones(N, bool)
        result = [{"status": "running"} for _ in range(N)]
        xt = [[] for _ in range(N)]
        xt19 = [[] for _ in range(N)]                       # bench statistic: progress s < 19 m
        p0, c0, s0 = pose[:, :2].copy(), np.cos(yaw0), np.sin(yaw0)
        traj = [[] for _ in range(N)]                       # (t, x, y, s, cross_track) in the course's local frame
        ratios, short, run = [], np.zeros(N), torch.zeros(N, device=dev)
        frame_log = []
        gait = {k: [] for k in ("slip", "fpeak", "f_over", "stumble", "roll", "pitch", "angvel_xy", "height",
                                "torque_rms", "power", "track_err", "res_abs", "res_clip", "air_time", "flight",
                                "double")}
        fmax = float(env.cfg.rewards.max_contact_force)
        nl = int(env.num_lower_dof)
        gait_air = torch.zeros(N, 2, device=dev)
        gait_prev = torch.ones(N, 2, dtype=torch.bool, device=dev)
        print(f"[gait] max_contact_force {fmax} N, lower dofs {nl}", flush=True)
        steps = int((fcfg.timeout_s + fcfg.post_reach_s + 2) / env.dt)
        for t in range(steps):
            pose = env.cart.data.root_link_pose_w.torch[:, :3].cpu().numpy()
            yaw = env.ts.cart_yaw.cpu().numpy()
            vel = torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1).cpu().numpy()
            cmd = np.zeros((N, 3), np.float32)
            for i in np.nonzero(active)[0]:
                f = followers[i]
                cmd[i] = f.update((pose[i, 0], pose[i, 1], yaw[i]), vel[i])
                dx, dy = pose[i, 0] - p0[i, 0], pose[i, 1] - p0[i, 1]
                traj[i].append((f.state.t, c0[i] * dx + s0[i] * dy, -s0[i] * dx + c0[i] * dy, f.state.s, f.state.cross_track,
                                vel[i, 0], vel[i, 1], vel[i, 2]))         # + cart-frame vx, vy, wz
                if not f.state.reached:
                    xt[i].append(abs(f.state.cross_track))
                if f.state.s < 19.0:
                    xt19[i].append(abs(f.state.cross_track))
                if a.stop_before_goal > 0 and f.state.s >= f.path.length - a.stop_before_goal:
                    # the pooled statistic only covers s < 19 m: skip the goal homing / settling phase
                    result[i] = {"status": "reached", "t": f.state.t, "goal_err_m": f.state.dist_goal}
                    active[i] = False
                elif f.state.reached and f.state.t - f.state.t_reached >= fcfg.post_reach_s:
                    result[i] = {"status": "reached", "t": f.state.t_reached, "goal_err_m": f.state.dist_goal}
                    active[i] = False
                elif f.state.timed_out:
                    result[i] = {"status": "timeout", "s": f.state.s}
                    active[i] = False
            if a.save_frames and hasattr(env, "frames_short"):
                k = a.save_frames
                frame_log.append(np.concatenate((env.frames_short[:k, -1].cpu().numpy(), env.privileged[:k].cpu().numpy(),
                                                 active[:k, None].astype(np.float32)), -1))
            env.cmd_task[:] = torch.as_tensor(cmd, device=dev)
            env.cmd_timer[:] = 10 ** 9
            with torch.inference_mode():
                act = act_fn(env.observation_dict())
            env.step(torch.as_tensor(act, device=dev).clone())
            if a.disturbances == "off":
                quiet()
            done = env.reset_buf.cpu().numpy()
            # gait bookkeeping on active, not-just-reset envs
            gm = torch.as_tensor(active, device=dev) & ~env.reset_buf
            if gm.any():
                F = env.contact_forces[:, env.feet_indices, :][gm]                 # [n, 2, 3]
                c = F[..., 2] > 1.0
                vxy = env.feet_vel[gm][..., :2].norm(dim=-1)
                if c.any():
                    gait["slip"].append(vxy[c])
                fn = F.norm(dim=-1)
                gait["fpeak"].append(fn.max(-1).values)
                gait["f_over"].append((fn - fmax).clamp(min=0).sum(-1))
                gait["stumble"].append((F[..., :2].norm(dim=-1) > 3 * F[..., 2].abs()).any(-1).float())
                g = env.projected_gravity[gm]
                gait["roll"].append(g[:, 1]); gait["pitch"].append(g[:, 0])
                gait["angvel_xy"].append(env.base_ang_vel[gm][:, :2].norm(dim=-1))
                gait["height"].append(env.root_states[gm][:, 2])
                tq = env.torques[gm][:, :nl]
                gait["torque_rms"].append(tq.square().mean(-1).sqrt())
                gait["power"].append((tq * env.dof_vel[gm][:, :nl]).abs().sum(-1))
                gait["track_err"].append((env.joint_pos_target[gm][:, :nl] - env.dof_pos[gm][:, :nl]).abs().mean(-1))
                nc = c.sum(-1)
                gait["flight"].append((nc == 0).float()); gait["double"].append((nc == 2).float())
                # own swing timer (HOMIE zeroes feet_air_time at the end of its step)
                land = c & ~gait_prev[gm]
                if land.any():
                    gait["air_time"].append(gait_air[gm][land])
                if hasattr(env, "leg_residual"):
                    r_ = env.leg_residual[gm]
                    gait["res_abs"].append(r_.abs().mean(-1))
                    from homie_pushcart_v8.task_config import leg_residual_bound
                    gait["res_clip"].append((r_.abs() >= 0.95 * leg_residual_bound(task)).float().mean(-1))
                    gait.setdefault("res_vec", []).append(r_.detach().cpu())
                    if "res_prev" in gait:
                        pm = gm.cpu() & gait["res_prev_mask"]
                        gait.setdefault("res_flip", []).append(((env.leg_residual.cpu()[pm] * gait["res_prev"][pm]) < 0).float())
                    gait["res_prev"] = env.leg_residual.detach().cpu().clone(); gait["res_prev_mask"] = gm.cpu().clone()
            c_all = env.contact_forces[:, env.feet_indices, 2] > 1.0
            gait_air = torch.where(c_all, torch.zeros_like(gait_air), gait_air + env.dt)
            gait_prev = c_all
            # grip load bookkeeping on active envs
            act_t = torch.as_tensor(active, device=dev) & env._hand_valid & ~env.reset_buf
            r = torch.cat((env.ts.grasp_ratio, env.ts.grasp_push_ratio.unsqueeze(-1)), -1)
            if act_t.any():
                ratios.append(r[act_t].reshape(-1, 6)[::2].half())
            over = (r > 1).any(-1).any(-1) & act_t
            ended = (~over & (run >= 5) & (run < task.termination.persist_steps)).cpu().numpy()
            short += ended
            run = torch.where(over, run + 1, torch.zeros_like(run))
            for i in np.nonzero(active & done)[0]:
                reasons = [k for k, v in snap["reasons"].items() if bool(v[i])]
                kinds = [k for j, k in enumerate(OVERLOAD_KINDS) if bool(snap["kinds"][i, j])]
                result[i] = {"status": "failed", "reasons": reasons, "kinds": kinds, "s": followers[i].state.s,
                             "t": followers[i].state.t}
                active[i] = False
            if not active.any():
                break
        st = [r["status"] for r in result]
        fails = [r for r in result if r["status"] == "failed"]
        reached = [r for r in result if r["status"] == "reached"]
        why = {}
        for r in fails:
            key = "+".join(r["kinds"]) if r["kinds"] else "+".join(r["reasons"]) or "?"
            why[key] = why.get(key, 0) + 1
        R = torch.cat(ratios).float()
        names = ["pull", "lift", "roll", "tilt", "yaw", "push"]
        xt_ok = np.concatenate([np.array(xt[i]) for i in range(N) if result[i]["status"] == "reached" and xt[i]]) if reached else np.zeros(1)
        xt_all = np.concatenate([np.array(v) for v in xt if v])
        out = {"policy": a.policy, "disturbances": a.disturbances, "payload_kg": a.payload, "runs": N,
               "success_rate": st.count("reached") / N, "failed_rate": st.count("failed") / N,
               "timeout_rate": st.count("timeout") / N, "failure_causes": why,
               "failed_progress_m_p10_p50_p90": np.percentile([r["s"] for r in fails], [10, 50, 90]).round(2).tolist() if fails else None,
               "finish_time_s_p50_p90": np.percentile([r["t"] for r in reached], [50, 90]).round(2).tolist() if reached else None,
               "goal_err_m_p50": float(np.median([r["goal_err_m"] for r in reached])) if reached else None,
               "cross_track_rms_m_successful_runs": float(np.sqrt(np.mean(xt_ok ** 2))),
               "cross_track_max_m_p90_successful_runs": float(np.percentile([max(xt[i]) for i in range(N) if result[i]["status"] == "reached" and xt[i]] or [0], 90)),
               "cross_track_rms_m_all_runs_until_end": float(np.sqrt(np.mean(xt_all ** 2))),
               "load_ratio_p99": dict(zip(names, [round(v, 2) for v in torch.quantile(R[:400000], 0.99, dim=0).tolist()])),
               "short_overloads_per_run": float(short.mean()),
               "cross_track_rms_m_first19m_per_run_p50_p90": np.percentile(
                   [float(np.sqrt(np.mean(np.square(v)))) for v in xt19 if v], [50, 90]).round(4).tolist(),
               "cross_track_rms_m_first19m_pooled": float(np.sqrt(np.mean(np.square(np.concatenate([np.array(v) for v in xt19 if v])))))}
        def q(k, f):
            if not gait[k]:
                return None
            v = torch.cat([x.reshape(-1).float() for x in gait[k]])
            return round(float(f(v)), 4)
        out["gait"] = {
            "foot_slip_speed_in_contact_mean_m_s": q("slip", torch.mean),
            "foot_slip_speed_in_contact_p95_m_s": q("slip", lambda v: torch.quantile(v[:1000000], 0.95)),
            "foot_force_peak_p50_N": q("fpeak", torch.median),
            "foot_force_peak_p99_N": q("fpeak", lambda v: torch.quantile(v[:1000000], 0.99)),
            "foot_force_over_limit_mean_N": q("f_over", torch.mean),
            "stumble_rate": q("stumble", torch.mean),
            "base_roll_rms_rad": q("roll", lambda v: v.square().mean().sqrt()),
            "base_pitch_mean_rad": q("pitch", torch.mean),
            "base_pitch_rms_rad": q("pitch", lambda v: v.square().mean().sqrt()),
            "base_angvel_xy_rms_rad_s": q("angvel_xy", lambda v: v.square().mean().sqrt()),
            "base_height_mean_m": q("height", torch.mean), "base_height_std_m": q("height", torch.std),
            "leg_torque_rms_Nm": q("torque_rms", torch.mean), "leg_power_W": q("power", torch.mean),
            "leg_joint_tracking_err_rad": q("track_err", torch.mean),
            "swing_air_time_mean_s": q("air_time", torch.mean),
            "flight_phase_rate": q("flight", torch.mean), "double_support_rate": q("double", torch.mean),
            "leg_residual_abs_mean": q("res_abs", torch.mean), "leg_residual_at_clip_rate": q("res_clip", torch.mean)}
        if gait.get("res_vec"):
            RV = torch.cat(gait["res_vec"]).float()
            out["gait"]["leg_residual_per_joint_mean"] = [round(x, 3) for x in RV.mean(0).tolist()]
            out["gait"]["leg_residual_per_joint_positive_frac"] = [round(x, 3) for x in (RV > 0).float().mean(0).tolist()]
            if gait.get("res_flip"):
                FL = torch.cat(gait["res_flip"])
                out["gait"]["leg_residual_sign_flip_per_step_per_joint"] = [round(x, 4) for x in FL.mean(0).tolist()]
        for k in ("res_vec", "res_flip", "res_prev", "res_prev_mask"):
            gait.pop(k, None)
        print("[gait]", json.dumps(out["gait"]), flush=True)
        print("[course]", json.dumps(out), flush=True)
        o = ROOT / "logs/validation/v7_course_isaac"; o.mkdir(parents=True, exist_ok=True)
        names = [m.split("=", 1)[0] for m in a.mix] if a.mix else [a.policy]
        for k, name in enumerate(names):
            ids = [i for i in range(N) if i % len(names) == k]
            np.savez_compressed(o / f"traj_{name}_{a.payload:g}kg_d{a.difficulty:g}{a.tag}.npz",
                                course_xy=course.xy, status=np.array([result[i]["status"] for i in ids]),
                                **{f"env{j}": np.array(traj[i]) for j, i in enumerate(ids)})
        if frame_log:
            np.save(o / f"frames_{a.policy}_{a.payload:g}kg{a.tag}.npy", np.array(frame_log))
        (o / f"{a.policy}_{a.payload:g}kg_d{a.difficulty:g}_dist{a.disturbances}{a.tag}.json").write_text(json.dumps({**out, "runs_detail": result}, indent=1), encoding="utf-8")
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush()
        raise
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
