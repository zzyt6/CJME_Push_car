"""Push-cart project entry point.

Sub-commands that only need yourdfpy / trimesh / viser run without Isaac Sim.
Sub-commands that need Kit (pack, smoke, train, play) start it themselves.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
sys.path.insert(0, str(ROOT / "HomieRL" / "rsl_rl"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="HOMIE G1 push-cart tools")
    sub = parser.add_subparsers(dest="command", required=True)

    pose = sub.add_parser("pose", help="initial grasp pose")
    pose_sub = pose.add_subparsers(dest="action", required=True)
    imp = pose_sub.add_parser("import", help="trajectory JSON last frame -> initial_pose.yaml (+ g1_grasp.urdf)")
    imp.add_argument("json", type=Path)
    view = pose_sub.add_parser("view", help="static viser check of robot + cart from YAML")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--no_obj", action="store_true", help="hide the xiaotuiche.obj visual mesh")

    cart = sub.add_parser("cart", help="cart asset")
    cart_sub = cart.add_subparsers(dest="action", required=True)
    cart_sub.add_parser("build", help="cart_params.yaml + initial_pose.yaml -> cart.urdf (writes derived section)")

    usd = sub.add_parser("usd", help="packed USD scene")
    usd_sub = usd.add_parser if False else usd.add_subparsers(dest="action", required=True)
    usd_view = usd_sub.add_parser("view", help="render the packed USD with viser (usd-core, no Kit)")
    usd_view.add_argument("--port", type=int, default=8081)
    usd_view.add_argument("--path", type=Path)

    sub.add_parser("pack", help="build g1_pushcart.usd inside Kit (headless); extra args go to AppLauncher")

    teleop = sub.add_parser("teleop", help="Xbox gamepad teleop of the frozen HOMIE policy (needs --viz kit)")
    teleop.add_argument("--mode", choices=["robot", "cart"], default="robot",
                        help="robot: G1 only at the preset grasp pose; cart: packed scene with the cart attached")
    teleop.add_argument("--checkpoint", type=Path, help="HOMIE model_*.pt (default: latest in the training run)")
    teleop.add_argument("--input", choices=["auto", "gamepad", "keyboard"], default="auto")
    teleop.add_argument("--episode_s", type=float, default=3600.0, help="time-out before an automatic reset")
    teleop.add_argument("--smoothing", type=float, default=0.1, help="command low-pass time constant [s], 0 = off")
    teleop.add_argument("--no_invert", action="store_true",
                        help="keep Isaac Lab's raw stick signs (default: stick right = move right / turn right)")

    mj = sub.add_parser("mj", help="MuJoCo sim2sim (CPU physics + viser rendering + Xbox gamepad)")
    mj_sub = mj.add_subparsers(dest="action", required=True)
    mj_sub.add_parser("build", help="generate g1_grasp.xml / g1_pushcart.xml + mjcf_report.json")
    mj_teleop = mj_sub.add_parser("teleop", help="gamepad teleop of the frozen HOMIE policy in MuJoCo")
    mj_teleop.add_argument("--mode", choices=["robot", "cart"], default="cart")
    mj_teleop.add_argument("--checkpoint", type=Path)
    mj_teleop.add_argument("--port", type=int, default=8080)
    mj_teleop.add_argument("--no_invert", action="store_true")
    mj_teleop.add_argument("--smoothing", type=float, default=0.1)
    mj_teleop.add_argument("--render_hz", type=float, default=25.0)
    mj_teleop.add_argument("--hl", type=Path, help="high-level checkpoint: gamepad gives the task command, the policy corrects HOMIE")
    mj_teleop.add_argument("--level", type=int, default=6, help="curriculum level defining the correction scale (with --hl)")
    mj_view = mj_sub.add_parser("view", help="legacy/V3 and V6 two-stage viewer: choose a training run and checkpoint; "
                                             "eval scenarios / gamepad / sliders, live plots; cart visual mesh only")
    mj_view.add_argument("--checkpoint", type=Path, help="legacy/V3 or V6 model_*.pt (default: newest checkpoint across both log roots)")
    mj_view.add_argument("--run", type=Path, help="training run directory or name under homie_pushcart or homie_pushcart_v6 logs")
    mj_view.add_argument("--homie_checkpoint", type=Path)
    mj_view.add_argument("--level", type=int, help="curriculum level for the correction scale (default: top level)")
    mj_view.add_argument("--scenario", help="start in this eval scenario (default: manual command)")
    mj_view.add_argument("--auto", action="store_true", help="cycle through all eval scenarios")
    mj_view.add_argument("--payload", type=float, default=0.0, help="payload [kg] in manual mode")
    mj_view.add_argument("--transient_s", type=float, default=1.5, help="seconds excluded from the scenario statistics after a command change")
    mj_view.add_argument("--port", type=int, default=8080)
    mj_view.add_argument("--render_hz", type=float, default=25.0)
    mj_view.add_argument("--no_invert", action="store_true")
    mj_view.add_argument("--smoothing", type=float, default=0.1)
    mj_eval = mj_sub.add_parser("eval", help="MuJoCo sim2sim: same scenario suite as `eval` for a high-level checkpoint (or passthrough)")
    mj_eval.add_argument("--checkpoint", type=Path, help="high-level model_*.pt; omit for the passthrough baseline")
    mj_eval.add_argument("--homie_checkpoint", type=Path)
    mj_eval.add_argument("--level", type=int, default=6)
    mj_eval.add_argument("--scenarios", nargs="*")
    mj_eval.add_argument("--transient_s", type=float, default=1.5)
    mj_smoke = mj_sub.add_parser("smoke", help="headless MuJoCo check (no viser)")
    mj_smoke.add_argument("--mode", choices=["robot", "cart"], default="cart")
    mj_smoke.add_argument("--checkpoint", type=Path)
    mj_smoke.add_argument("--steps", type=int, default=500)
    mj_smoke.add_argument("--settle_steps", type=int, default=100)
    mj_smoke.add_argument("--vx", type=float, default=0.3)
    mj_smoke.add_argument("--vy", type=float, default=0.0)
    mj_smoke.add_argument("--wz", type=float, default=0.0)

    cartdiag = sub.add_parser("cartdiag", help="Isaac Lab: cart alone, rolling test for cylinder vs sphere wheel colliders")
    cartdiag.add_argument("--colliders", nargs="+", default=["cylinder", "sphere"])
    cartdiag.add_argument("--v0", type=float, default=0.5)
    cartdiag.add_argument("--steps", type=int, default=400)
    cartdiag.add_argument("--pos_iters", type=int, default=4)
    cartdiag.add_argument("--vel_iters", type=int, default=0)

    stance = sub.add_parser("stance", help="Isaac Lab: measure HOMIE's zero-command standing pose; --apply bakes it into initial_pose.yaml")
    stance.add_argument("--checkpoint", type=Path)
    stance.add_argument("--num_envs", type=int, default=8)
    stance.add_argument("--settle_s", type=float, default=3.0)
    stance.add_argument("--average_s", type=float, default=3.0)
    stance.add_argument("--apply", action="store_true", help="write legs + pelvis height into initial_pose.yaml and rebuild cart.urdf")
    stance.add_argument("--upright_base", action="store_true",
                        help="with --apply: keep the pelvis upright at reset instead of baking HOMIE's measured standing pitch")

    export = sub.add_parser("export", help="TorchScript / ONNX export of a high-level checkpoint + hl_interface.json (CPU, no Kit)")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, help="default: <checkpoint dir>/exported")
    export.add_argument("--level", type=int, default=17, help="curriculum level whose correction scale is documented")

    train_parser = sub.add_parser("train", help="phase B training via NVIDIA's rsl_rl train.py (extra args forwarded, e.g. --num_envs 4096 --headless --level 0)")
    train_parser.add_argument("--quality_profile", choices=["diagnostic", "ripple", "wrench", "combined", "v4"],
                              help="v4: combined speed quality, stop/go/turn sequences and observable idle correction")
    sub.add_parser("play", help="play a high-level checkpoint via NVIDIA's rsl_rl play.py (--checkpoint <run>/model_N.pt)")

    ev = sub.add_parser("eval", help="Isaac Lab: fixed scenario suite for a high-level checkpoint (or passthrough baseline without --checkpoint)")
    ev.add_argument("--checkpoint", type=Path, help="high-level model_*.pt; omit for the passthrough (zero-action) baseline")
    ev.add_argument("--homie_checkpoint", type=Path)
    ev.add_argument("--num_envs", type=int, default=16)
    ev.add_argument("--level", type=int, default=6, help="curriculum level defining action scales / noise (frozen)")
    ev.add_argument("--scenarios", nargs="*", help="subset of scenario names (default: all)")
    ev.add_argument("--transient_s", type=float, default=1.5)
    ev.add_argument("--no_randomization", action="store_true")

    envsmoke = sub.add_parser("envsmoke", help="Isaac Lab: PushCartEnv (phase B) with zero/random high-level actions; reset invariants + termination stats")
    envsmoke.add_argument("--num_envs", type=int, default=16)
    envsmoke.add_argument("--steps", type=int, default=300)
    envsmoke.add_argument("--policy", choices=["zero", "random", "sine"], default="random")
    envsmoke.add_argument("--action_std", type=float, default=0.3)
    envsmoke.add_argument("--level", type=int, help="curriculum level to start at (default: task cfg init_level)")
    envsmoke.add_argument("--episode_s", type=float, help="override episode length")
    envsmoke.add_argument("--no_randomization", action="store_true", help="disable HOMIE's robot randomisation and noise")
    envsmoke.add_argument("--checkpoint", type=Path)
    envsmoke.add_argument("--seed", type=int, default=0)
    envsmoke.add_argument("--force_trace", type=float, default=0.0, help="print every step whose max grasp force exceeds this [N]")
    envsmoke.add_argument("--hl_checkpoint", type=Path, help="drive the env with a trained high-level checkpoint instead of --policy")
    envsmoke.add_argument("--hl_noise_std", type=float, default=0.0, help="Gaussian exploration noise added to the checkpoint's actions")
    envsmoke.add_argument("--allow_contact", action="store_true", help="diagnostic: re-enable robot <-> cart collisions")
    envsmoke.add_argument("--cart_iters", nargs=2, type=int, metavar=("POS", "VEL"), help="diagnostic: cart solver iterations (default from cart_params)")
    envsmoke.add_argument("--legacy_push", action="store_true", help="diagnostic: HOMIE's root-velocity-jump push instead of the force pulse")
    envsmoke.add_argument("--no_force_clip", action="store_true", help="diagnostic: no clipping of grasp forces in rewards")
    envsmoke.add_argument("--no_stall", action="store_true", help="diagnostic: disable the single-wheel stall disturbance")
    envsmoke.add_argument("--no_kick", action="store_true", help="diagnostic: disable the cart velocity kick disturbance")
    envsmoke.add_argument("--no_robot_push", action="store_true", help="diagnostic: disable robot pushes, retaining observation noise")
    envsmoke.add_argument("--no_idle_correction", action="store_true", help="diagnostic: V4 command distribution with legacy idle gate")
    envsmoke.add_argument("--fixed_command", nargs=3, type=float, metavar=("VX", "VY", "WZ"), help="diagnostic: fixed task command")
    envsmoke.add_argument("--payload", type=float, help="diagnostic: fixed payload kg")
    envsmoke.add_argument("--quality_profile", choices=["diagnostic", "ripple", "wrench", "combined", "v4"],
                          help="exercise V3.1 cart-quality filters and rewards")

    calibration = sub.add_parser("wrenchcal", help="Isaac no-cart static/dynamic and known-load palm wrench validation")
    calibration.add_argument("--num_envs", type=int, default=4)
    calibration.add_argument("--steps", type=int, default=180, help="control steps per calibration phase")
    calibration.add_argument("--checkpoint", type=Path)
    calibration.add_argument("--output", type=Path, help="new output directory; default: timestamped logs/bench directory")

    smoke = sub.add_parser("smoke", help="headless HOMIE-driven check of the robot / cart scene (Isaac Lab)")
    smoke.add_argument("--mode", choices=["robot", "cart"], default="cart")
    smoke.add_argument("--checkpoint", type=Path)
    smoke.add_argument("--num_envs", type=int, default=2)
    smoke.add_argument("--steps", type=int, default=300)
    smoke.add_argument("--settle_steps", type=int, default=50, help="steps with zero command before applying --vx/--vy/--wz")
    smoke.add_argument("--vx", type=float, default=0.3)
    smoke.add_argument("--vy", type=float, default=0.0)
    smoke.add_argument("--wz", type=float, default=0.0)
    smoke.add_argument("--arms", choices=["grasp", "default"], default="grasp",
                       help="hold the upper body at the grasp pose (default) or at HOMIE's zero pose (comparison)")
    smoke.add_argument("--wheel_damping", type=float, help="override wheels.wheel_damping for this run")
    smoke.add_argument("--cart_mass", type=float, help="override the cart base_link mass [kg] at runtime")

    args, extra = parser.parse_known_args(argv)
    if extra and args.command not in ("pack", "teleop", "smoke", "cartdiag", "stance", "envsmoke", "eval", "wrenchcal", "train", "play"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "export":
        from homie_pushcart.export import export_checkpoint
        export_checkpoint(args.checkpoint, args.output, args.level)
        return 0
    if args.command in ("train", "play"):
        # NVIDIA's entrypoints run in a clean child interpreter; nothing from Kit may be imported here.
        from homie_pushcart.training import play as play_main, train as train_main
        if args.command == "train" and args.quality_profile:
            extra = ["--quality_profile", args.quality_profile, *extra]
        return train_main(extra) if args.command == "train" else play_main(extra)

    if args.command == "pose" and args.action == "import":
        from homie_pushcart.pose import import_initial_pose
        from homie_pushcart.paths import INITIAL_POSE_YAML
        pose_data = import_initial_pose(args.json)
        print(f"[pushcart] wrote {INITIAL_POSE_YAML}")
        print(f"[pushcart] pelvis z on ground: {pose_data['robot']['base_pos'][2]:.4f} m")
        for side in ("left", "right"):
            print(f"[pushcart] {side} palm world: {pose_data['reference']['palm_world'][side]}")
            print(f"[pushcart] {side} grasp point (palm frame, initial guess): {pose_data['grasp']['point_in_palm'][side]}")
        print("[pushcart] planner TCP expressed in palm frame under two pelvis-height hypotheses:")
        for label, h in pose_data["reference"]["planner_tcp_in_palm_hypotheses"].items():
            print(f"    {label}: {h}")
        return 0
    if args.command == "cart" and args.action == "build":
        from homie_pushcart.assets.cart_urdf import build_cart
        report = build_cart()
        print(f"[pushcart] cart.urdf written; derived: {report['derived']}")
        return 0
    if args.command == "pose" and args.action == "view":
        from homie_pushcart.viser_check import run_pose_view
        run_pose_view(port=args.port, show_obj=not args.no_obj)
        return 0
    if args.command == "usd" and args.action == "view":
        from homie_pushcart.viser_check import run_usd_view
        run_usd_view(path=args.path, port=args.port)
        return 0
    if args.command == "pack":
        from homie_pushcart.assets.pack_usd import main as pack_main
        return pack_main(extra)
    if args.command == "mj":
        if args.action == "build":
            from homie_pushcart.assets.mjcf_scene import build_all
            import json as _json
            reports = build_all()
            for name, rep in reports.items():
                print(f"[mjcf] {name}: " + _json.dumps({k: v for k, v in rep.items() if k != "contacts_at_preset"}))
                print(f"[mjcf] {name}: contacts at preset = {rep['contacts_at_preset']}")
            ok = all(r["robot_hinge_order_matches_homie"] for r in reports.values()) and \
                (reports["cart"]["equality_max_violation_at_preset"] or 0.0) < 1e-3
            print("MJCF BUILD PASS" if ok else "MJCF BUILD CHECK FAILED")
            return 0 if ok else 1
        from homie_pushcart.mujoco_teleop import run_mujoco_smoke, run_mujoco_teleop
        if args.action == "smoke":
            result = run_mujoco_smoke(args)
            return 0 if result["result"] == "PASS" else 2
        if args.action == "eval":
            from homie_pushcart.mujoco_hl import run_mujoco_eval
            run_mujoco_eval(args)
            return 0
        if args.action == "view":
            from homie_pushcart.mujoco_viewer import run_two_stage_viewer
            run_two_stage_viewer(args)
            return 0
        run_mujoco_teleop(args)
        return 0
    if args.command in ("teleop", "smoke", "cartdiag", "stance", "envsmoke", "eval", "wrenchcal"):
        from isaaclab.app import AppLauncher
        app_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(app_parser)
        app_args = app_parser.parse_args(extra)
        for key, value in vars(args).items():
            setattr(app_args, key, value)
        launcher = AppLauncher(app_args)
        code = 0
        try:
            if args.command == "teleop":
                from homie_pushcart.teleop import run_teleop
                run_teleop(app_args, launcher, ROOT)
            elif args.command == "cartdiag":
                from homie_pushcart.cart_diag import run_cart_diag
                run_cart_diag(app_args, launcher)
            elif args.command == "stance":
                from homie_pushcart.stance import run_stance
                run_stance(app_args, launcher, ROOT)
            elif args.command == "wrenchcal":
                from homie_pushcart.wrench_calibration import run_wrench_calibration
                result = run_wrench_calibration(app_args, launcher, ROOT)
                code = 0 if result["result"] == "PASS" else 2
            elif args.command == "envsmoke":
                from homie_pushcart.env_smoke import run_env_smoke
                result = run_env_smoke(app_args, launcher, ROOT)
                code = 0 if result["result"] == "PASS" else 2
            elif args.command == "eval":
                from homie_pushcart.evaluate import run_eval
                run_eval(app_args, launcher, ROOT)
            else:
                from homie_pushcart.smoke import run_smoke
                result = run_smoke(app_args, launcher, ROOT)
                code = 0 if result["result"] == "PASS" else 2
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            launcher.app.close(exit_code=1)
            raise
        launcher.app.close()
        return code
    parser.error("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
