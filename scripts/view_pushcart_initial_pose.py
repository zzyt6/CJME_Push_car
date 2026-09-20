"""Paper-figure viewer for the packed G1 + cart initial pose.

Only render the authored USD transforms: no policy, reset, or physics step.
Start at the paper camera angle, with free navigation, metal shading and a green backdrop.
All presentation overrides live in a temporary stage; source assets are never saved.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

VIEWS = {
    # Matched to the user's reference screenshot (front + left side of the cart).
    "paper": ((3.3570, 1.8464, 1.8210), (1.07344, 0.26869, 0.68255)),
    "overview": ((2.8, -2.6, 1.9), (0.5, 0.0, 0.7)),
    "side": ((0.4, -3.2, 1.2), (0.4, 0.0, 0.75)),
    "front": ((3.4, 0.0, 1.5), (0.35, 0.0, 0.8)),
    "hands": ((1.0, -0.75, 1.25), (0.32, 0.0, 0.92)),
}
PAPER_CAMERA = "/World/PaperCamera"
GREEN = (0.0, 1.0, 0.0)


def create_scene():
    """Reference the training asset in a temporary, static viewing stage."""
    # Kit-dependent imports must follow AppLauncher construction.
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    from homie_pushcart.paths import PACKED_USD
    from homie_pushcart.pose import load_initial_pose

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdGeom.Xform.Define(stage, "/World/Scene")
    if not scene.GetPrim().GetReferences().AddReference(str(PACKED_USD)):
        raise RuntimeError(f"Could not reference {PACKED_USD}")

    # Detect a stale packed pose instead of silently displaying the wrong reset.
    pose = load_initial_pose()["robot"]
    robot = stage.GetPrimAtPath("/World/Scene/Robot")
    cart = stage.GetPrimAtPath("/World/Scene/Cart/Geometry/base_link")
    if not robot.IsValid() or not cart.IsValid():
        raise RuntimeError("Packed USD is incomplete. Run .\\run_one_stage_pushcart.ps1 pack --viz none")
    translation = robot.GetAttribute("xformOp:translate").Get()
    orientation = robot.GetAttribute("xformOp:orient").Get()
    quat = (*orientation.GetImaginary(), orientation.GetReal())
    matches = all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(translation, pose["base_pos"]))
    matches &= all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(quat, pose["base_quat_xyzw"]))
    for name, angle in pose["joint_pos"].items():
        attr = stage.GetAttributeAtPath(f"/World/Scene/Robot/Physics/{name}.state:angular:physics:position")
        value = attr.Get() if attr.IsValid() else None
        matches &= value is not None and math.isclose(math.radians(value), angle, abs_tol=1e-6)
    if not matches:
        raise RuntimeError("Packed USD differs from initial_pose.yaml. Rebuild with "
                           ".\\run_one_stage_pushcart.ps1 cart build, then "
                           ".\\run_one_stage_pushcart.ps1 pack --viz none")

    # Make the temporary reference visual-only. Disabling dynamics alone does
    # not hide joint rings, mass gizmos or the importer's guide-purpose colliders.
    # Include instance proxies: the foot collision meshes are inside instances.
    physics_prefixes = ("Physics", "Physx", "Newton")
    edits = []
    body_count = helper_count = joint_count = 0
    for prim in Usd.PrimRange(scene.GetPrim(), Usd.TraverseInstanceProxies()):
        imageable = UsdGeom.Imageable(prim)
        # The full cart OBJ already includes its wheels. The URDF adds four
        # simplified cylinder visuals on top of them for the articulated wheels.
        wheel_visual = (prim.IsA(UsdGeom.Cylinder)
                        and str(prim.GetPath()).startswith("/World/Scene/Cart/")
                        and prim.GetName() in {f"wheel_{i}" for i in range(4)})
        # Added grasp bars are separate from the OBJ's original curved handles.
        # Hide only these two prims in the temporary viewer stage.
        bar_visual = str(prim.GetPath()) in {
            "/World/Scene/Cart/Geometry/base_link/bar_0",
            "/World/Scene/Cart/Geometry/base_link/bar_1",
        }
        hide = bool(imageable) and (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or imageable.ComputePurpose() == UsdGeom.Tokens.guide or wheel_visual or bar_visual)
        schemas = [name for name in prim.GetAppliedSchemas() if name.startswith(physics_prefixes)]
        is_joint = prim.IsA(UsdPhysics.Joint)
        body_count += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if hide or schemas or is_joint:
            edits.append((prim.GetPath(), hide, schemas, is_joint))
    for path, hide, schemas, is_joint in edits:
        prim = stage.GetPrimAtPath(path)
        while prim.IsInstanceProxy():
            parent = prim.GetParent()
            while parent.IsInstanceProxy():
                parent = parent.GetParent()
            parent.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        if hide:
            UsdGeom.Imageable(prim).MakeInvisible()
            helper_count += 1
        for schema in schemas:
            prim.RemoveAppliedSchema(schema)
        if is_joint:
            prim.SetActive(False)
            joint_count += 1

    # A stainless-steel finish for this figure only. Bind on the instance root
    # with stronger inheritance so the OBJ's original painted material is overridden.
    steel = UsdShade.Material.Define(stage, "/World/Looks/CartStainlessSteel")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/CartStainlessSteel/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.62, 0.65, 0.69))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.24)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    steel.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    cart_visual = stage.GetPrimAtPath("/World/Scene/Cart/Geometry/base_link/obj_visual")
    UsdShade.MaterialBindingAPI.Apply(cart_visual).Bind(steel, UsdShade.Tokens.strongerThanDescendants)

    # No visible ground: the backdrop must stay uniform, including below wheels
    # and through the shelves. A renderer background adds no green bounce light.
    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(350.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(15.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    # Original light: (315, 0, 135) degrees. Turn its azimuth by 180 degrees,
    # retaining the elevation. This stays fixed in world space as the camera moves.
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 315.0))
    # A broad softbox on the same side produces readable reflections on the steel.
    softbox = UsdLux.RectLight.Define(stage, "/World/SteelSoftbox")
    softbox.CreateWidthAttr(3.0)
    softbox.CreateHeightAttr(2.0)
    softbox.CreateIntensityAttr(1200.0)
    softbox.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    softbox_xform = UsdGeom.Xformable(softbox)
    softbox_xform.AddTranslateOp().Set(Gf.Vec3d(3.7, 3.0, 4.89264))
    softbox_xform.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 315.0))
    print(f"[initial-pose] Asset: {PACKED_USD}", flush=True)
    print(f"[initial-pose] YAML matches: {len(pose['joint_pos'])} joints; {body_count} rigid bodies frozen.", flush=True)
    print(f"[initial-pose] Visual only: {helper_count} collision/guide/helper prims hidden; "
          f"{joint_count} joint gizmos removed.", flush=True)
    return stage


def configure_clean_viewport():
    """Show rendered meshes without lights/camera icons, grids or guide overlays."""
    import carb.settings
    import omni.usd

    settings = carb.settings.get_settings()
    # Kit's displayOptions bit 10 is scene meshes; other bits are editor overlays.
    settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
    settings.set_bool("/persistent/app/hydra/displayPurpose/guide", False)
    settings.set_int("/rtx/background/source/type", 2)
    settings.set("/rtx/background/source/color", GREEN)
    # Keep the chroma-key green unaltered by exposure/tone mapping.
    settings.set_bool("/rtx/post/backgroundZeroAlpha/postTonemapBackgroundComposite", True)
    settings.set_bool("/rtx/post/histogram/enabled", False)
    omni.usd.get_context().get_selection().clear_selected_prim_paths()


def configure_camera(stage, view, resolution):
    """Set the initial view once; let Kit handle orbit, pan and zoom afterwards."""
    import omni.usd
    from isaacsim.core.rendering_manager import ViewportManager
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, PAPER_CAMERA)
    camera.CreateFocalLengthAttr(18.147562)
    # Match the vertical field of view of the supplied screenshot. The paper
    # view uses its cropped aspect ratio instead of the editor's wide viewport.
    camera.CreateVerticalApertureAttr(11.75)
    camera.CreateHorizontalApertureAttr(11.75 * resolution[0] / resolution[1])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.CreateFStopAttr(0.0)
    eye, target = (Gf.Vec3d(*p) for p in VIEWS[view])
    transform = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    op = xform.AddTransformOp()
    op.Set(transform)
    camera.GetPrim().CreateAttribute("omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d).Set(
        Gf.Vec3d(0.0, 0.0, -(target - eye).GetLength()))
    ViewportManager.set_camera(PAPER_CAMERA)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No render viewport. Use --viz kit or --enable_cameras for headless capture")
    viewport.resolution = tuple(resolution)
    selection = omni.usd.get_context().get_selection()

    def clear_selection():
        # Hide object handles without resetting the user's camera or camera choice.
        if selection.get_selected_prim_paths():
            selection.clear_selected_prim_paths()

    return viewport, clear_selection


def main(argv=None):
    from isaaclab.app import AppLauncher
    from homie_pushcart.paths import PACKED_USD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=VIEWS, default="paper", help="initial camera angle; navigation stays enabled")
    parser.add_argument("--fps", type=float, default=30.0, help="maximum viewer refresh rate")
    parser.add_argument("--frames", type=int, default=0, help="close after N rendered frames; 0 keeps the window open")
    parser.add_argument("--resolution", nargs=2, type=int, default=(1920, 1600), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, help="save a clean PNG and exit (no editor UI in the image)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.frames < 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--frames must be >= 0 and --fps must be a finite positive number")
    if min(args.resolution) < 64:
        parser.error("--resolution dimensions must be at least 64 pixels")
    if args.output:
        if args.output.suffix.lower() != ".png":
            parser.error("--output must be a .png file")
        args.frames = args.frames or 120
        args.enable_cameras = True
    if not PACKED_USD.is_file():
        parser.error("Missing packed USD. Run .\\run_one_stage_pushcart.ps1 pack --viz none first")
    # Lab 3 defaults to headless; this script is a GUI viewer by default.
    if args.visualizer is None and not args.headless:
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
    if args.visualizer not in (None, ["kit"], ["none"]):
        parser.error("Use --viz kit for the window or --viz none --frames N for a smoke check")
    if (args.headless or args.visualizer == ["none"]) and args.frames == 0:
        parser.error("Headless checks require --frames N to finish automatically")

    launcher = AppLauncher(args)
    app = launcher.app
    exit_code = 1
    try:
        import omni.timeline
        from pxr import Usd, UsdGeom

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_auto_update(False)
        stage = create_scene()
        configure_clean_viewport()
        app.update()
        viewport = clear_selection = None
        if args.visualizer == ["kit"] or args.enable_cameras:
            viewport, clear_selection = configure_camera(stage, args.view, args.resolution)

        # A finite run also verifies that all authored link poses stay unchanged.
        def transforms():
            cache = UsdGeom.XformCache()
            return {str(p.GetPath()): cache.GetLocalToWorldTransform(p)
                    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Scene"))
                    if p.IsA(UsdGeom.Xformable)}

        initial = transforms() if args.frames else None
        print(f"[initial-pose] Paper view ready: initial {args.view} camera, free navigation, stainless steel, "
              "opposite-side world lighting, green background. Close the window to exit.", flush=True)
        frame = 0
        while app.is_running() and (args.frames == 0 or frame < args.frames):
            start = time.perf_counter()
            if timeline.is_playing():
                timeline.stop()
            if clear_selection:
                clear_selection()
            app.update()
            frame += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - start)))
        if initial is not None and app.is_running():
            if transforms() != initial or timeline.get_current_time() != 0.0:
                raise RuntimeError("Static-view check failed: the scene or timeline moved")
            print(f"[initial-pose] STATIC VIEW PASS: {frame} frames, no pose drift, timeline = 0 s.", flush=True)
        if args.output and app.is_running():
            import asyncio
            from omni.kit.viewport.utility import capture_viewport_to_file

            args.output.parent.mkdir(parents=True, exist_ok=True)
            capture = capture_viewport_to_file(viewport, str(args.output.resolve()))
            pending = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
            deadline = time.monotonic() + 30.0
            while app.is_running() and not pending.done() and time.monotonic() < deadline:
                clear_selection()
                app.update()
            if not pending.done() or not pending.result():
                raise RuntimeError("PNG capture did not complete")
            print(f"[initial-pose] Saved: {args.output.resolve()}", flush=True)
        exit_code = 0
    finally:
        app.close(exit_code=exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
