"""Read-only USD FixedJoint / MuJoCo weld frame audit; no simulation steps.

Also compiles a disposable in-memory site-weld representation to check the
mapping. It never saves that model or changes production assets/defaults.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def pose(position, quaternion_wxyz):
    transform = np.eye(4)
    transform[:3, 3] = position
    transform[:3, :3] = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    return transform


def usd_frame(joint, index):
    position = getattr(joint, f"GetLocalPos{index}Attr")().Get()
    q = getattr(joint, f"GetLocalRot{index}Attr")().Get()
    return pose(position, [q.GetReal(), *q.GetImaginary()])


def frame_values(transform):
    return dict(position=transform[:3, 3].tolist(),
                quaternion_wxyz=np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1).tolist())


def errors(first, second):
    return dict(position_m=float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
                rotation_rad=float(Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()))


def world_body(model, data, name):
    body = model.body(name).id
    return pose(data.xpos[body], data.xquat[body])


def audit(output, checkpoint):
    from homie_pushcart_v52.control_contract import nominal_configs
    from homie_pushcart_v52.paths import PACKED_USD, G1_GRASP_URDF, CART_URDF
    from pushcart_bench.policies_v5x import InstrumentedEnv
    from pushcart_bench.sim2sim_scene import LEGACY_SCENE

    cfg, _ = nominal_configs(checkpoint)
    env = InstrumentedEnv(True, scene_profile="isaac", cfg=cfg,
                          upper_drive="training", inertia_profile="urdf")
    env.reset()
    model, data = env.model, env.data
    stage = Usd.Stage.Open(str(PACKED_USD))
    cache = UsdGeom.XformCache()
    rows, mapped_frames = {}, {}
    for side in ("left", "right"):
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f"/Scene/GraspJoints/{side}"))
        if not joint:
            raise ValueError(f"Missing fixed joint: {side}")
        palm_path, = joint.GetBody0Rel().GetTargets()
        cart_path, = joint.GetBody1Rel().GetTargets()
        palm = stage.GetPrimAtPath(palm_path)
        wrist_name = f"{side}_wrist_yaw_link"
        wrist = palm
        while wrist.IsValid() and wrist.GetName() != wrist_name:
            wrist = wrist.GetParent()
        if not wrist.IsValid():
            raise ValueError(f"Cannot fold {palm_path} into {wrist_name}")
        if stage.GetPrimAtPath(cart_path).GetName() != "base_link":
            raise ValueError(f"Unexpected cart frame: {cart_path}")
        # Gf matrices use row-vector convention; NumPy below uses column vectors.
        palm_world = np.asarray(cache.GetLocalToWorldTransform(palm)).T
        wrist_world = np.asarray(cache.GetLocalToWorldTransform(wrist)).T
        wrist_palm = np.linalg.inv(wrist_world) @ palm_world
        np.testing.assert_allclose(wrist_palm[:3, :3].T @ wrist_palm[:3, :3], np.eye(3), atol=1e-7)
        palm_frame, cart_frame = usd_frame(joint, 0), usd_frame(joint, 1)
        wrist_frame = wrist_palm @ palm_frame
        desired_relative = wrist_frame @ np.linalg.inv(cart_frame)
        eq_id = model.eq(f"grasp_{side}").id
        if model.eq_objtype[eq_id] != mujoco.mjtObj.mjOBJ_BODY:
            raise ValueError("This audit expects the historical body-based weld")
        eq_data = model.eq_data[eq_id]
        existing_relative = pose(np.zeros(3), eq_data[6:10])
        existing_relative[:3, 3] = eq_data[3:6] - existing_relative[:3, :3] @ eq_data[:3]
        relative_error = errors(desired_relative, existing_relative)
        wrist_reset, cart_reset = (world_body(model, data, name) for name in (wrist_name, "cart_base"))
        old_first = wrist_reset @ np.r_[eq_data[3:6], 1.]
        old_second = cart_reset @ np.r_[eq_data[:3], 1.]
        break_values = {}
        for label, getter in (("break_force", joint.GetBreakForceAttr), ("break_torque", joint.GetBreakTorqueAttr)):
            value = float(getter().Get())
            break_values[label] = value if np.isfinite(value) else "infinity"
        rows[side] = dict(
            isaac_joint_type=joint.GetPrim().GetTypeName(),
            isaac_body0=str(palm_path), isaac_body1=str(cart_path),
            isaac_flags=dict(enabled=joint.GetJointEnabledAttr().Get(),
                             excluded_from_articulation=joint.GetExcludeFromArticulationAttr().Get(),
                             collision_enabled=joint.GetCollisionEnabledAttr().Get(), **break_values),
            isaac_local_frame0_palm=frame_values(palm_frame),
            isaac_local_frame1_cart=frame_values(cart_frame),
            wrist_to_palm=wrist_palm.tolist(),
            isaac_joint_frame_in_wrist=wrist_frame.tolist(),
            isaac_joint_frame_in_cart=cart_frame.tolist(),
            proposed_site_in_wrist=frame_values(wrist_frame),
            proposed_site_in_cart=frame_values(cart_frame),
            mujoco_wrist_anchor=eq_data[3:6].tolist(), mujoco_cart_anchor=eq_data[:3].tolist(),
            desired_vs_existing_relative_translation_m=relative_error["position_m"],
            desired_vs_existing_relative_rotation_rad=relative_error["rotation_rad"],
            anchor_separation_in_cart_m=float(np.linalg.norm(cart_frame[:3, 3] - eq_data[:3])),
            isaac_frames_at_mujoco_reset=errors(wrist_reset @ wrist_frame, cart_reset @ cart_frame),
            old_anchor_pair_at_reset_error_m=float(np.linalg.norm(old_first - old_second)),
            current_solimp=model.eq_solimp[eq_id].tolist(), current_solref=model.eq_solref[eq_id].tolist(),
            torquescale=float(eq_data[10]))
        mapped_frames[side] = (wrist_frame, cart_frame)

    # Verify that the installed engine accepts the explicit frames. This model
    # exists only in memory; no mj_step, policy action, fitting, or XML save.
    root = ET.parse(env.xml_path).getroot()
    for side, frames in mapped_frames.items():
        site_names = []
        for part, body_name, frame in zip(("wrist", "cart"), (f"{side}_wrist_yaw_link", "cart_base"), frames):
            name = f"review_{side}_{part}"
            values = frame_values(frame)
            body = root.find(f".//body[@name='{body_name}']")
            ET.SubElement(body, "site", name=name, size="0.001",
                          pos=" ".join(map(str, values["position"])),
                          quat=" ".join(map(str, values["quaternion_wxyz"])))
            site_names.append(name)
        weld = root.find(f"equality/weld[@name='grasp_{side}']")
        for attr in ("body1", "body2", "relpose", "anchor"):
            weld.attrib.pop(attr, None)
        weld.set("site1", site_names[0])
        weld.set("site2", site_names[1])
    candidate = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (candidate.nq, candidate.nv, candidate.nbody) != (model.nq, model.nv, model.nbody):
        raise ValueError("Frame-only representation changed model dimensions")
    probe = mujoco.MjData(candidate)
    probe.qpos[:] = data.qpos
    mujoco.mj_forward(candidate, probe)
    for side in rows:
        eq_id = candidate.eq(f"grasp_{side}").id
        mask = ((probe.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (probe.efc_id == eq_id))
        site_ids = (int(candidate.eq_obj1id[eq_id]), int(candidate.eq_obj2id[eq_id]))
        rows[side]["in_memory_site_check"] = dict(
            object_type=int(candidate.eq_objtype[eq_id]),
            resolved_body_names=[candidate.body(int(candidate.site_bodyid[s])).name for s in site_ids],
            constraint_rows=int(mask.sum()), constraint_residual=probe.efc_pos[mask].tolist(),
            note="Compile/forward check at reset only; no dynamic equivalence claim.")

    sources = {Path(layer.realPath).resolve() for layer in stage.GetUsedLayers() if layer.realPath}
    sources.update((G1_GRASP_URDF, CART_URDF, LEGACY_SCENE, env.xml_path))
    report = dict(
        schema="v52_grasp_frames_v1", mujoco_version=mujoco.__version__,
        source=str(PACKED_USD), grasp="nominal", inertia="urdf", model_asset=str(env.xml_path),
        usd_units=dict(meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), up_axis=str(UsdGeom.GetStageUpAxis(stage))),
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
        rows=rows, note="Read-only frame audit; no stepping, parameter fitting or production grasp changes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({side: {key: value for key, value in row.items()
                            if key.startswith(("desired_vs", "anchor_sep", "isaac_frames", "in_memory"))}
                      for side, row in rows.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/bench/v52_grasp_mapping_review/joint_frames.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "logs/rsl_rl/homie_pushcart_v52/2026-09-26_01-56-48_v52_hand_estimator_20k/model_19999.pt")
    args = parser.parse_args()
    audit(args.output, args.checkpoint)
