"""Isaac Sim paper-photo viewer for the two-stage task's actual robot reset pose.

Only the robot is referenced from the packed training scene. No policy or physics
is run. All visibility, camera and lighting edits stay in an anonymous stage.
Default: a freely movable front orthographic camera and pure green background.

Run with run_two_stage_robot_initial_pose.ps1. Use --output FILE.png for a clean
image and automatic exit, or the Photo controls window to capture a chosen view.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
ROBOT_PATH = "/World/Robot"
CAMERA_PATH = "/World/FrontCamera"


def create_scene():
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    from homie_pushcart.paths import PACKED_USD
    from homie_pushcart.pose import load_initial_pose

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    robot = UsdGeom.Xform.Define(stage, ROBOT_PATH).GetPrim()
    if not robot.GetReferences().AddReference(str(PACKED_USD), "/Scene/Robot"):
        raise RuntimeError(f"Could not reference the robot in {PACKED_USD}")

    # Fail on a stale packed asset instead of depicting a different reset pose.
    pose = load_initial_pose()["robot"]
    translation = robot.GetAttribute("xformOp:translate").Get()
    orientation = robot.GetAttribute("xformOp:orient").Get()
    if translation is None or orientation is None:
        raise RuntimeError("Packed robot has no authored initial root pose")
    quat = (*orientation.GetImaginary(), orientation.GetReal())
    matches = all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(translation, pose["base_pos"]))
    matches &= all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(quat, pose["base_quat_xyzw"]))
    for name, angle in pose["joint_pos"].items():
        attr = stage.GetAttributeAtPath(f"{ROBOT_PATH}/Physics/{name}.state:angular:physics:position")
        value = attr.Get() if attr.IsValid() else None
        matches &= value is not None and math.isclose(math.radians(value), angle, abs_tol=1e-6)
    if not matches:
        raise RuntimeError("Packed robot differs from initial_pose.yaml; update the training asset before photographing it")

    edits = []
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        imageable = UsdGeom.Imageable(prim)
        hide = bool(imageable) and (prim.HasAPI(UsdPhysics.CollisionAPI)
                                    or imageable.ComputePurpose() == UsdGeom.Tokens.guide)
        schemas = [name for name in prim.GetAppliedSchemas()
                   if name.startswith(("Physics", "Physx", "Newton"))]
        joint = prim.IsA(UsdPhysics.Joint)
        if hide or schemas or joint:
            edits.append((prim.GetPath(), hide, schemas, joint))
    for path, hide, schemas, joint in edits:
        prim = stage.GetPrimAtPath(path)
        while prim.IsInstanceProxy():
            parent = prim.GetParent()
            while parent.IsInstanceProxy():
                parent = parent.GetParent()
            parent.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        if hide:
            UsdGeom.Imageable(prim).MakeInvisible()
        for schema in schemas:
            prim.RemoveAppliedSchema(schema)
        if joint:
            prim.SetActive(False)

    # White fill plus a fixed front-side key. Background green never illuminates
    # the robot; no floor, grid, cart or editor gizmos appear in the photo.
    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(350.0)
    fill.CreateColorAttr(Gf.Vec3f(1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(1800.0)
    key.CreateAngleAttr(15.0)
    key.CreateColorAttr(Gf.Vec3f(1.0))
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 315.0))
    print(f"[robot-photo] Pose: initial_pose.yaml; root {tuple(translation)}; "
          f"{len(pose['joint_pos'])} joint angles verified. Robot only, no physics.", flush=True)
    return stage


def configure_viewport():
    import carb.settings
    import omni.usd

    settings = carb.settings.get_settings()
    settings.set_int("/persistent/app/viewport/displayOptions", 1 << 10)
    settings.set_bool("/persistent/app/hydra/displayPurpose/guide", False)
    settings.set_int("/rtx/background/source/type", 2)
    settings.set("/rtx/background/source/color", (0.0, 1.0, 0.0))
    settings.set_bool("/rtx/post/backgroundZeroAlpha/postTonemapBackgroundComposite", True)
    settings.set_bool("/rtx/post/histogram/enabled", False)
    omni.usd.get_context().get_selection().clear_selected_prim_paths()


def front_camera(stage, resolution, projection):
    """Fit the full robot at eye level, looking along its horizontal forward axis."""
    from isaacsim.core.rendering_manager import ViewportManager
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Sdf, Usd, UsdGeom

    robot = stage.GetPrimAtPath(ROBOT_PATH)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"],
                              useExtentsHint=False).ComputeWorldBound(robot).ComputeAlignedRange()
    size, center = bounds.GetSize(), bounds.GetMidpoint()
    if bounds.IsEmpty() or not all(math.isfinite(v) for v in size) or size[2] <= 0:
        raise RuntimeError("Cannot frame the robot: invalid visible mesh bounds")
    orientation = robot.GetAttribute("xformOp:orient").Get()
    forward = Gf.Rotation(orientation).TransformDir(Gf.Vec3d(1, 0, 0))
    forward[2] = 0.0
    forward.Normalize()
    right = Gf.Cross(forward, Gf.Vec3d(0, 0, 1))
    width = abs(right[0]) * size[0] + abs(right[1]) * size[1]
    aspect = resolution[0] / resolution[1]
    height = max(size[2], width / aspect) * 1.16
    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.CreateProjectionAttr(projection)
    if projection == "orthographic":
        # USD orthographic apertures are in tenths of a scene unit (meters here).
        camera.CreateVerticalApertureAttr(height * 10.0)
        camera.CreateHorizontalApertureAttr(height * aspect * 10.0)
        distance = 4.0
    else:
        camera.CreateVerticalApertureAttr(24.0)
        camera.CreateHorizontalApertureAttr(24.0 * aspect)
        camera.CreateFocalLengthAttr(85.0)
        distance = height * 85.0 / 24.0 + size[0] / 2
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.CreateFStopAttr(0.0)
    eye = center + forward * distance
    transform = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse()
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    # Get an existing transform op when returning to front view.
    attr = camera.GetPrim().GetAttribute("xformOp:transform")
    op = UsdGeom.XformOp(attr) if attr else xform.AddTransformOp()
    xform.SetXformOpOrder([op])
    op.Set(transform)
    camera.GetPrim().CreateAttribute("omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d).Set(
        Gf.Vec3d(0, 0, -distance))
    ViewportManager.set_camera(CAMERA_PATH)
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport for rendering")
    viewport.resolution = tuple(resolution)
    return viewport


def capture_png(app, viewport, path, tick):
    import asyncio
    from omni.kit.viewport.utility import capture_viewport_to_file

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    capture = capture_viewport_to_file(viewport, str(path))
    pending = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
    deadline = time.monotonic() + 30.0
    while app.is_running() and not pending.done() and time.monotonic() < deadline:
        tick()
    if not pending.done() or not pending.result() or not path.is_file():
        raise RuntimeError(f"PNG capture did not complete: {path}")
    print(f"[robot-photo] Saved: {path}", flush=True)
    return path


def photo_controls(actions, resolution):
    import omni.ui as ui

    window = ui.Window("Robot photo", width=310, height=145)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(f"Green screen | {resolution[0]} x {resolution[1]}", height=20)
            ui.Button("Front view", clicked_fn=lambda: actions.update(front=True), height=30)
            ui.Button("Save current view (PNG)", clicked_fn=lambda: actions.update(capture=True), height=30)
            ui.Label("Saved under logs/robot_initial_pose", height=20)
    return window


def main(argv=None):
    from isaaclab.app import AppLauncher
    from homie_pushcart.paths import PACKED_USD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=("orthographic", "perspective"), default="orthographic")
    parser.add_argument("--resolution", nargs=2, type=int, default=(1200, 2000), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, help="save a clean front PNG and exit")
    parser.add_argument("--frames", type=int, default=0, help="exit after N frames; 0 keeps the viewer open")
    parser.add_argument("--fps", type=float, default=30.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.frames < 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--frames must be >= 0 and --fps must be positive and finite")
    if min(args.resolution) < 64:
        parser.error("--resolution dimensions must be at least 64")
    if args.output:
        if args.output.suffix.lower() != ".png":
            parser.error("--output must be a PNG path")
        args.frames = max(args.frames, 90)
    if not PACKED_USD.is_file():
        parser.error(f"Packed training asset is missing: {PACKED_USD}")
    if args.visualizer is None and not args.headless:
        args.visualizer, args.visualizer_explicit = ["kit"], True
    if args.visualizer not in (None, ["kit"], ["none"]):
        parser.error("Use --viz kit for the window or --viz none for a headless photo")
    if (args.headless or args.visualizer == ["none"]) and args.frames == 0:
        parser.error("Headless mode requires --output FILE.png or --frames N")
    args.enable_cameras = True
    app = AppLauncher(args).app
    exit_code = 1
    try:
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.set_auto_update(False)
        stage = create_scene()
        configure_viewport()
        app.update()
        viewport = front_camera(stage, args.resolution, args.projection)
        selection = omni.usd.get_context().get_selection()

        def tick():
            if timeline.is_playing():
                timeline.stop()
            if selection.get_selected_prim_paths():
                selection.clear_selected_prim_paths()
            app.update()  # Never overwrite camera transforms during navigation.

        def transforms():
            cache = UsdGeom.XformCache()
            return {str(p.GetPath()): cache.GetLocalToWorldTransform(p)
                    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if p.IsA(UsdGeom.Xformable)}

        initial = transforms()
        actions = {"front": False, "capture": False}
        controls = photo_controls(actions, args.resolution) if args.visualizer == ["kit"] else None
        print("[robot-photo] Ready: front view, free navigation, green background. "
              "Use Front view / Save current view in the Robot photo window.", flush=True)
        frame = 0
        while app.is_running() and (args.frames == 0 or frame < args.frames):
            started = time.perf_counter()
            if actions["front"]:
                viewport = front_camera(stage, args.resolution, args.projection)
                actions["front"] = False
                # Let denoising converge after changing cameras before a photo.
                for _ in range(30):
                    tick()
            tick()
            frame += 1
            if actions["capture"]:
                actions["capture"] = False
                path = ROOT / "logs/robot_initial_pose" / f"robot_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                capture_png(app, viewport, path, tick)
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - started)))
        if app.is_running():
            if transforms() != initial or timeline.get_current_time() != 0.0:
                raise RuntimeError("Static pose check failed: robot or timeline changed")
            print(f"[robot-photo] STATIC VIEW PASS: {frame} frames; no pose drift.", flush=True)
            if args.output:
                capture_png(app, viewport, args.output, tick)
        if controls:
            controls.destroy()
        exit_code = 0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
