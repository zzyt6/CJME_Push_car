"""V7: the V6 interface on a slip-capable Dex3 grasp and Coulomb cart resistance.

    pushcart_v7.py build grasp                  CPU: fit Dex3 fingers to the 22 mm tube, bake g1_grasp_v7.urdf
    pushcart_v7.py build urdf|mjcf|manifest     CPU: cart URDF with the grasp chain, MuJoCo scene, asset hashes
    pushcart_v7.py build usd [--headless]       inside Kit: packed USD (then rewrites the manifest)
    pushcart_v7.py build all                    grasp + urdf + mjcf + manifest (then `build usd` in Kit)
    pushcart_v7.py train|play [...]             NVIDIA RSL-RL entrypoints, as V6
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))


def build(argv) -> int:
    parser = argparse.ArgumentParser(prog="pushcart_v7.py build", allow_abbrev=False)
    parser.add_argument("target", choices=("grasp", "urdf", "mjcf", "manifest", "usd", "all"))
    args, extra = parser.parse_known_args(argv)
    from homie_pushcart_v7.assets import build as b
    if args.target in ("grasp", "all"):
        record = b.fit_and_write_grasp()
        print(json.dumps({s: {k: v for k, v in d.items() if k in ("axis_shift_m", "palm_clearance_m", "surface_gap_after_m")}
                          for s, d in record["diagnostics"].items()}, indent=1))
    params = b.load_v7_params()
    if args.target in ("urdf", "all"):
        print(json.dumps(b.write_cart_urdf_v7(params), indent=1))
    if args.target in ("mjcf", "all"):
        report = b.build_mjcf_v7(params)
        print(json.dumps({k: report[k] for k in ("output", "equality_max_violation_at_preset", "grasp_chain",
                                                 "caster_frictionloss", "robot_hinge_order_matches_homie")}, indent=1))
    if args.target == "usd":
        from isaaclab.app import AppLauncher
        p = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(p)
        launcher = AppLauncher(p.parse_args(extra))
        try:
            report = b.pack_usd_v7(params)
            ok = (report["robot_joint_set_matches_homie"] and report["robot_cart_collisions_filtered"]
                  and all(v["grasp_frame_mismatch_m"] < 1e-4 for v in report["static_checks"].values())
                  and all(n in report["cart_joints"] for n in ("grasp_left_slide_joint", "grasp_left_roll_joint",
                                                               "grasp_right_slide_joint", "grasp_right_roll_joint")))
            print(json.dumps({k: report[k] for k in ("output", "cart_joints", "grasp_joints", "static_checks")}, indent=1))
            print("V7 PACK PASS" if ok else "V7 PACK CHECK FAILED", flush=True)
            b.write_build_manifest()
            return 0 if ok else 1
        finally:
            launcher.app.close()
    print(json.dumps(b.write_build_manifest(), indent=1))
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        return build(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "play", "build"))
    args, extra = parser.parse_known_args()
    from homie_pushcart_v7 import training
    return getattr(training, args.command)(extra)


if __name__ == "__main__":
    raise SystemExit(main())
