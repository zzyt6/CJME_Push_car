"""Pack robot + cart + grasp joints into one USD scene (runs inside Kit, headless).

Output layout (self-contained, all relative references):

    assets/g1_pushcart/
        g1_pushcart.usd          /Scene (default prim)
                                   /Scene/Robot       -> reference g1_grasp/g1_grasp.usda
                                   /Scene/Cart        -> reference cart/cart.usda
                                   /Scene/GraspJoints/{left,right}  PhysicsFixedJoint palm <-> cart base_link
        g1_grasp/                URDF importer output of g1_grasp.urdf (same converter settings as HOMIE's env)
        cart/                    URDF importer output of cart.urdf
        asset_report.json        joint order, masses, joint frames, static consistency errors

The robot's 27 joint states and the link transforms are authored at the preset
pose so the file is statically self-consistent (viser `usd view`), and PhysX
starts from a satisfied grasp constraint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..paths import (ASSET_DIR, ASSET_REPORT, CART_URDF, G1_GRASP_URDF, G1_URDF, HOMIE_JOINTS, PACKED_USD,
                     PALM_LINKS, ROOT)
from ..pose import Kinematics, load_initial_pose
from .cart_urdf import derive_cart_pose, load_cart_params


def _quatf(matrix: np.ndarray):
    from pxr import Gf
    q = R.from_matrix(matrix).as_quat()  # xyzw
    return Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _quatd(matrix: np.ndarray):
    from pxr import Gf
    q = R.from_matrix(matrix).as_quat()
    return Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def convert_urdf(urdf_path: Path, usd_dir: Path, self_collision: bool = False) -> Path:
    """Same converter settings as HOMIE's env.py UrdfFileCfg (fix_base False, keep fixed joints, zero drives)."""
    import shutil
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # The converter appends _1, _2 ... when the directory already holds an output; keep a single copy.
    if usd_dir.exists():
        shutil.rmtree(usd_dir)
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path), usd_dir=str(usd_dir), force_usd_conversion=True,
        fix_base=False, merge_fixed_joints=False, self_collision=self_collision,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
    )
    converter = UrdfConverter(cfg)
    return Path(converter.usd_path)


def link_chain(urdf_root: ET.Element, link: str) -> list[str]:
    parents = {j.find("child").get("link"): j.find("parent").get("link") for j in urdf_root.findall("joint")}
    chain = [link]
    while chain[0] in parents:
        chain.insert(0, parents[chain[0]])
    return chain


def robot_link_path(urdf_root: ET.Element, link: str, robot_prim: str = "/Scene/Robot") -> str:
    return robot_prim + "/Geometry/" + "/".join(link_chain(urdf_root, link))


def find_prim_named(stage, root_path: str, name: str):
    from pxr import Usd
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    raise KeyError(f"{name} not found under {root_path}")


def pack(output: Path = PACKED_USD, *, asset_dir: Path = ASSET_DIR, cart_urdf: Path = CART_URDF,
         report_path: Path = ASSET_REPORT, params: dict | None = None, grasp_target=None,
         robot_urdf: Path = G1_GRASP_URDF, pose: dict | None = None) -> dict:
    """Pack robot + cart + grasp joints. The keyword arguments let a later version pack its own cart.

    ``grasp_target(stage, side, cart_T, gp_world, palm_T)`` may return ``(prim, local_pos1, local_rot1)`` to attach
    the palm joint to a cart link other than ``base_link``; by default the V3.2 palm <-> base_link joint is built.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    asset_dir.mkdir(parents=True, exist_ok=True)
    pose = pose if pose is not None else load_initial_pose()
    params = params if params is not None else load_cart_params()
    kin = Kinematics(load_meshes=False)
    derived = derive_cart_pose(params, pose, kin)
    kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])

    robot_usd = convert_urdf(robot_urdf, asset_dir / "g1_grasp")
    cart_usd = convert_urdf(cart_urdf, asset_dir / "cart")
    urdf_root = ET.parse(robot_urdf).getroot()
    cart_root = ET.parse(cart_urdf).getroot()

    if output.exists():
        output.unlink()
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    scene = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(scene.GetPrim())

    def add_reference(path: str, usd_file: Path, translation, rotation_matrix):
        xform = UsdGeom.Xform.Define(stage, path)
        rel = Path(usd_file).resolve().relative_to(output.parent.resolve()).as_posix()
        xform.GetPrim().GetReferences().AddReference("./" + rel)
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_quatd(rotation_matrix))
        return xform

    base_rot = R.from_quat(pose["robot"]["base_quat_xyzw"]).as_matrix()
    add_reference("/Scene/Robot", robot_usd, pose["robot"]["base_pos"], base_rot)
    cart_rot = R.from_euler("z", derived["base_yaw"]).as_matrix()
    add_reference("/Scene/Cart", cart_usd, derived["base_pos_world"], cart_rot)

    # --- joint states (degrees) and link transforms at the preset pose -------------------------------
    joint_paths = {}
    for joint in urdf_root.findall("joint"):
        name = joint.get("name")
        if joint.get("type") == "fixed":
            continue
        path = f"/Scene/Robot/Physics/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise KeyError(f"joint prim missing in converted robot: {path}")
        joint_paths[name] = path
        # PhysicsJointStateAPI:angular is already applied by the importer; angular state is in degrees.
        over_joint = stage.OverridePrim(path)
        for attr_name, value in (("state:angular:physics:position", math.degrees(pose["robot"]["joint_pos"][name])),
                                 ("state:angular:physics:velocity", 0.0)):
            attr = over_joint.GetAttribute(attr_name)
            if not attr.IsValid():
                attr = over_joint.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            attr.Set(float(value))
        child, parent = joint.find("child").get("link"), joint.find("parent").get("link")
        local = np.linalg.inv(kin.base_T(parent)) @ kin.base_T(child)
        link_prim = stage.GetPrimAtPath(robot_link_path(urdf_root, child))
        if not link_prim.IsValid():
            raise KeyError(f"link prim missing: {robot_link_path(urdf_root, child)}")
        translate = link_prim.GetAttribute("xformOp:translate")
        orient = link_prim.GetAttribute("xformOp:orient")
        if not translate.IsValid() or not orient.IsValid():
            raise ValueError(f"{child}: converted link lacks translate/orient xform ops")
        translate.Set(Gf.Vec3d(*local[:3, 3].tolist()))
        orient.Set(_quatd(local[:3, :3]) if orient.GetTypeName() == Sdf.ValueTypeNames.Quatd else _quatf(local[:3, :3]))

    # --- grasp joints ----------------------------------------------------------------------------------
    cart_base_prim = find_prim_named(stage, "/Scene/Cart", "base_link")
    cart_T = np.eye(4)
    cart_T[:3, :3] = cart_rot
    cart_T[:3, 3] = derived["base_pos_world"]
    joints_scope = UsdGeom.Scope.Define(stage, "/Scene/GraspJoints")
    grasp_report = {}
    for side, palm in PALM_LINKS.items():
        palm_path = robot_link_path(urdf_root, palm)
        if not stage.GetPrimAtPath(palm_path).IsValid():
            raise KeyError(f"palm prim missing: {palm_path}")
        palm_T = kin.world_T(palm)
        gp_palm = np.asarray(pose["grasp"]["point_in_palm"][side], dtype=float)
        gp_world = (palm_T @ np.append(gp_palm, 1.0))[:3]
        cart_inv = np.linalg.inv(cart_T)
        gp_cart = (cart_inv @ np.append(gp_world, 1.0))[:3]
        rot1 = cart_inv[:3, :3] @ palm_T[:3, :3]
        body1_prim = cart_base_prim
        if grasp_target is not None:
            body1_prim, gp_cart, rot1 = grasp_target(stage, side, cart_T, gp_world, palm_T)
        joint = UsdPhysics.FixedJoint.Define(stage, f"/Scene/GraspJoints/{side}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
        joint.CreateBody1Rel().SetTargets([body1_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*gp_palm.tolist()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1., 0., 0., 0.))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in gp_cart]))
        joint.CreateLocalRot1Attr().Set(_quatf(rot1))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        grasp_report[side] = {"palm_prim": palm_path, "cart_prim": str(body1_prim.GetPath()),
                              "local_pos0_palm": gp_palm.tolist(), "local_pos1_cart": gp_cart.tolist(),
                              "grasp_point_world": gp_world.tolist()}
    del joints_scope

    # --- no robot <-> cart contacts ------------------------------------------------------------------
    # The only robot colliders are the two feet. A foot touching a caster would close a rigid loop
    # (foot -> legs -> arms -> grasp joints -> cart -> wheel); the pair is filtered at the asset level and
    # the task penalises foot-caster proximity instead (modelling choice; feet stay > 0.12 m away under the
    # penalty). Note: the kN palm forces seen in the first training run were NOT caused by this loop - the
    # attribution runs showed the filter changed nothing; the cause was the wheel-stall brake (task_env.py).
    # The grasp joints are unaffected by filtering.
    filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/Scene/Cart"))
    filtered.CreateFilteredPairsRel().SetTargets([Sdf.Path("/Scene/Robot")])

    # --- physics materials for the cart (friction is not part of URDF; bind it here) -----------------
    from pxr import UsdShade

    def define_material(path: str, static: float, dynamic: float, restitution: float):
        material = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(static))
        api.CreateDynamicFrictionAttr().Set(float(dynamic))
        api.CreateRestitutionAttr().Set(float(restitution))
        return material

    wheel_mat = define_material("/Scene/Cart/PhysicsMaterials/wheel", params["wheels"]["friction"],
                                params["wheels"]["friction"], params["wheels"]["restitution"])
    body_mat = define_material("/Scene/Cart/PhysicsMaterials/body", params["body"]["friction"],
                               params["body"]["friction"], params["body"]["restitution"])
    material_report = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Scene/Cart"), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        material = wheel_mat if prim.GetName().startswith("wheel_") else body_mat
        target = stage.OverridePrim(prim.GetPath()) if prim.IsInstanceProxy() else prim
        UsdShade.MaterialBindingAPI.Apply(target).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        material_report[prim.GetName()] = str(material.GetPath())

    stage.GetRootLayer().Save()

    # --- verification on the composed stage ---------------------------------------------------------
    stage = Usd.Stage.Open(str(output))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    palm_errors = {}
    for side, palm in PALM_LINKS.items():
        prim = stage.GetPrimAtPath(robot_link_path(urdf_root, palm))
        usd_T = np.array(cache.GetLocalToWorldTransform(prim)).T  # column-vector convention
        palm_errors[side] = {
            "position_error_m": float(np.linalg.norm(usd_T[:3, 3] - kin.world_T(palm)[:3, 3])),
            "rotation_error_deg": float(np.degrees(np.linalg.norm(
                R.from_matrix(usd_T[:3, :3] @ kin.world_T(palm)[:3, :3].T).as_rotvec()))),
        }
        j = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f"/Scene/GraspJoints/{side}"))
        body0 = stage.GetPrimAtPath(j.GetBody0Rel().GetTargets()[0])
        body1 = stage.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
        T0 = np.array(cache.GetLocalToWorldTransform(body0)).T
        T1 = np.array(cache.GetLocalToWorldTransform(body1)).T
        p0 = (T0 @ np.append(np.array(j.GetLocalPos0Attr().Get()), 1.0))[:3]
        p1 = (T1 @ np.append(np.array(j.GetLocalPos1Attr().Get()), 1.0))[:3]
        palm_errors[side]["grasp_frame_mismatch_m"] = float(np.linalg.norm(p0 - p1))

    def masses(root_path):
        out = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.HasAPI(UsdPhysics.MassAPI):
                attr = prim.GetAttribute("physics:mass")
                if attr and attr.HasAuthoredValue():
                    out[prim.GetName()] = float(attr.Get())
        return out

    robot_joint_prims = [p for p in stage.GetPrimAtPath("/Scene/Robot/Physics").GetChildren()
                         if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    usd_joint_order = [p.GetName() for p in robot_joint_prims]
    cart_joint_prims = [p for p in stage.GetPrimAtPath("/Scene/Cart/Physics").GetChildren()
                        if p.IsA(UsdPhysics.Joint)] if stage.GetPrimAtPath("/Scene/Cart/Physics").IsValid() else []
    limits = {}
    for prim in robot_joint_prims:
        rj = UsdPhysics.RevoluteJoint(prim)
        limits[prim.GetName()] = [math.radians(rj.GetLowerLimitAttr().Get()), math.radians(rj.GetUpperLimitAttr().Get())]
    urdf_limits = {j.get("name"): [float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))]
                   for j in ET.parse(G1_URDF).getroot().findall("joint") if j.get("type") != "fixed"}
    limit_error = max(abs(limits[n][i] - urdf_limits[n][i]) for n in HOMIE_JOINTS for i in (0, 1))
    robot_masses = masses("/Scene/Robot")
    bound = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Scene/Cart"), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding(materialPurpose="physics")
            mat_prim = stage.GetPrimAtPath(binding.GetMaterialPath())
            api = UsdPhysics.MaterialAPI(mat_prim)
            bound[prim.GetName()] = {
                "material": str(binding.GetMaterialPath()),
                "static_friction": api.GetStaticFrictionAttr().Get() if mat_prim.IsValid() else None,
                "dynamic_friction": api.GetDynamicFrictionAttr().Get() if mat_prim.IsValid() else None,
                "restitution": api.GetRestitutionAttr().Get() if mat_prim.IsValid() else None,
            }
    cart_joint_damping = {}
    for prim in cart_joint_prims:
        attr = prim.GetAttribute("drive:angular:physics:damping")
        cart_joint_damping[prim.GetName()] = attr.Get() if attr else None

    report = {
        "output": str(output), "robot_usd": str(robot_usd), "cart_usd": str(cart_usd),
        # USD child order is the importer's own; Isaac Lab resolves joints by name (find_joints preserve_order).
        "robot_joint_order_usd": usd_joint_order,
        "robot_joint_set_matches_homie": sorted(usd_joint_order) == sorted(HOMIE_JOINTS),
        "robot_joint_count": len(usd_joint_order),
        "robot_joint_limits_max_abs_error_rad": limit_error,
        "robot_link_masses_kg": robot_masses,
        "robot_total_mass_kg": float(sum(robot_masses.values())),
        "cart_model": params["model"],
        "cart_joints": [p.GetName() for p in cart_joint_prims],
        "cart_joint_damping": cart_joint_damping,
        "cart_collision_materials": bound,
        "cart_link_masses_kg": masses("/Scene/Cart"),
        "pelvis_world": pose["robot"]["base_pos"],
        "cart_base_world": derived["base_pos_world"], "cart_base_yaw": derived["base_yaw"],
        "handle_height": derived["handle_height"],
        "grasp_joints": grasp_report,
        "static_checks": palm_errors,
        "robot_cart_collisions_filtered": [str(p) for p in UsdPhysics.FilteredPairsAPI(
            stage.GetPrimAtPath("/Scene/Cart")).GetFilteredPairsRel().GetTargets()] == ["/Scene/Robot"],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Pack g1_pushcart.usd inside Kit")
    parser.add_argument("--output", type=Path, default=PACKED_USD)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv or [])
    launcher = AppLauncher(args)
    code = 0
    try:
        report = pack(args.output)
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("robot_link_masses_kg", "cart_link_masses_kg", "grasp_joints")},
                         indent=2, ensure_ascii=False))
        all_bound = all(v["material"] and v["static_friction"] is not None for v in report["cart_collision_materials"].values())
        ok = report["robot_joint_set_matches_homie"] and report["robot_joint_count"] == 27 \
            and report["robot_joint_limits_max_abs_error_rad"] < 1e-4 and all_bound and report["robot_cart_collisions_filtered"] \
            and all(v["position_error_m"] < 1e-4 and v["grasp_frame_mismatch_m"] < 1e-4
                    for v in report["static_checks"].values())
        print("PACK PASS" if ok else "PACK CHECK FAILED", flush=True)
        code = 0 if ok else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    launcher.app.close()
    return code
