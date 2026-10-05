"""MuJoCo sim2sim teleop: CPU physics, viser rendering, Xbox gamepad (XInput) or GUI sliders."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from .gamepad import BUTTON_A, BUTTON_B, XInputGamepad
from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
from .mujoco_env import MujocoHomieEnv
from .paths import HOMIE_RUN_DIR, ROOT
from .teleop import CommandMapper


# ---------------------------------------------------------------------------- viser renderer

class MujocoViserRenderer:
    """Render asset visuals, with an optional robot-only collision overlay.

    The cart OBJ already includes its shelves, handles and wheels. Its auxiliary
    primitive geoms are physics/debug representations, not additional visuals.
    Rendering choices stay local to Viser: the MuJoCo model is never modified.
    """

    def __init__(self, server, model: mujoco.MjModel, show_robot_collision: bool = False):
        import trimesh
        from scipy.spatial.transform import Rotation as R
        self._R = R
        self.server, self.model = server, model
        self.handles = {}
        self.collision_handles = {}
        self._scene_handles = [
            server.scene.add_grid("/ground", width=80, height=80, plane="xy", cell_size=0.5, section_size=2.0),
            server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.006),
        ]
        cart_bodies = set()
        for body_id in range(1, model.nbody):
            body_name = model.body(body_id).name
            if (body_name.startswith(("cart_", "wheel_", "caster_"))
                    or int(model.body_parentid[body_id]) in cart_bodies):
                cart_bodies.add(body_id)
        for g in range(model.ngeom):
            gtype = model.geom_type[g]
            if gtype in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            is_visual = model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
            is_cart = int(model.geom_bodyid[g]) in cart_bodies
            if is_cart and not (is_visual and gtype == mujoco.mjtGeom.mjGEOM_MESH):
                continue
            size = model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                verts, faces = model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn]
            else:
                if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    mesh = trimesh.creation.box(extents=2 * size)
                elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
                elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
                elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0])
                else:
                    continue
                verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            body = model.body(model.geom_bodyid[g]).name
            name = model.geom(g).name or f"{body}_g{g}"
            rgba = model.geom_rgba[g]
            color = tuple(int(255 * c) for c in rgba[:3])
            opacity = None if is_cart or rgba[3] >= 0.999 else float(rgba[3])
            if is_visual:
                self.handles[g] = server.scene.add_mesh_simple(f"/mj/{name}", verts, faces, color=color, opacity=opacity)
            else:
                self.collision_handles[g] = server.scene.add_mesh_simple(
                    f"/mj_collision/{name}", verts, faces, color=(80, 140, 255), opacity=0.35, visible=show_robot_collision)

    def set_collision_visible(self, visible: bool):
        for h in self.collision_handles.values():
            h.visible = visible

    def remove(self):
        """Remove this renderer's nodes before replacing its model."""
        for handle in list(self.handles.values()) + list(self.collision_handles.values()) + self._scene_handles:
            handle.remove()
        self.handles.clear()
        self.collision_handles.clear()
        self._scene_handles.clear()

    def update(self, data: mujoco.MjData):
        with self.server.atomic():
            for g, h in list(self.handles.items()) + list(self.collision_handles.items()):
                q = self._R.from_matrix(data.geom_xmat[g].reshape(3, 3)).as_quat()
                h.wxyz = np.array([q[3], q[0], q[1], q[2]])
                h.position = data.geom_xpos[g].copy()


# ---------------------------------------------------------------------------- helpers

def _yaw_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _make(args):
    with_cart = args.mode == "cart"
    env = MujocoHomieEnv(with_cart=with_cart)
    checkpoint = Path(args.checkpoint) if getattr(args, "checkpoint", None) else frozen_checkpoint()
    policy = HimLowLevelPolicy(checkpoint, device="cpu")
    return env, policy, checkpoint


# ---------------------------------------------------------------------------- teleop

def run_mujoco_teleop(args):
    import torch
    import viser

    hl_runner = None
    if getattr(args, "hl", None):
        # high-level policy in the loop: the gamepad gives the *task* command (cart frame, 75 % of HOMIE's ranges)
        from types import SimpleNamespace
        from .mujoco_hl import make_runner
        if args.mode != "cart":
            raise ValueError("--hl needs --mode cart")
        hl_args = SimpleNamespace(checkpoint=args.hl, homie_checkpoint=getattr(args, "checkpoint", None), level=args.level)
        hl_runner, checkpoint = make_runner(hl_args)
        env, policy = hl_runner.env, hl_runner.lowlevel
        tc = hl_runner.task_cfg.commands
        ranges = SimpleNamespace(lin_vel_x=list(tc.vx_range_max), lin_vel_y=[-tc.vy_max, tc.vy_max], ang_vel_yaw=[-tc.wz_max, tc.wz_max])
    else:
        env, policy, checkpoint = _make(args)
        ranges = env.cfg.commands.ranges
    mapper = CommandMapper(ranges, invert_lateral=not args.no_invert, invert_yaw=not args.no_invert,
                           smoothing_tau_s=args.smoothing, dt=env.dt)
    pad = XInputGamepad()
    server = viser.ViserServer(port=args.port, label=f"HOMIE sim2sim teleop ({args.mode})")
    renderer = MujocoViserRenderer(server, env.model)

    gui = server.gui
    gui.add_markdown(f"### MuJoCo sim2sim · {'G1 + 推车' if env.with_cart else '只有 G1'}\n"
                     f"HOMIE `{checkpoint.name}`" + (f" · 上层 `{Path(args.hl).name}` (level {args.level})" if hl_runner else "") +
                     f"  ·  控制 {1 / env.dt:.0f} Hz / 物理 {1 / env.model.opt.timestep:.0f} Hz")
    status = gui.add_markdown("")
    with gui.add_folder("手柄"):
        pad_state = gui.add_markdown("XInput: 未连接" if not pad.available else "XInput: 等待手柄…")
        use_sliders = gui.add_checkbox("用滑条代替手柄", False)
    with gui.add_folder("滑条命令 (无手柄时)"):
        s_vx = gui.add_slider("vx [m/s]", float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1]), 0.05, 0.0)
        s_vy = gui.add_slider("vy [m/s]", float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1]), 0.05, 0.0)
        s_wz = gui.add_slider("wz [rad/s]", float(ranges.ang_vel_yaw[0]), float(ranges.ang_vel_yaw[1]), 0.05, 0.0)
        zero_btn = gui.add_button("命令归零")
    with gui.add_folder("控制"):
        reset_btn = gui.add_button("复位到初始位姿", color="green")
        follow = gui.add_checkbox("镜头跟随机器人 (固定偏移)", True)
        show_col = gui.add_checkbox("显示机器人碰撞体", False)
        rt = gui.add_slider("速度倍率 (1 = 实时)", 0.1, 2.0, 0.1, 1.0)
    flags = {"reset": False, "zero": False}
    reset_btn.on_click(lambda _: flags.__setitem__("reset", True))
    zero_btn.on_click(lambda _: flags.__setitem__("zero", True))
    show_col.on_update(lambda _: renderer.set_collision_visible(show_col.value))

    camera_offset = np.array([-2.0, -2.6, 1.6])

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # The client's camera is not initialised yet at connect time; apply the view a moment later.
        import threading

        def later():
            time.sleep(1.0)
            with client.atomic():
                client.camera.position = np.array([0.5, 0., 0.7]) + camera_offset
                client.camera.look_at = np.array([0.5, 0., 0.7])

        threading.Thread(target=later, daemon=True).start()

    print("=" * 72)
    print(f"[mj-teleop] mode={args.mode}  xml={env.xml_path.name}  checkpoint={checkpoint}  iter={policy.iteration}"
          + (f"  high-level={args.hl} (iter {hl_runner.hl.iteration}, level {args.level})" if hl_runner else ""))
    print(f"[mj-teleop] viser: http://localhost:{server.get_port()}   XInput available: {pad.available}")
    print(f"[mj-teleop] left stick: forward {ranges.lin_vel_x[1]:+.1f} / back {ranges.lin_vel_x[0]:+.1f} m/s, lateral ±{ranges.lin_vel_y[1]:.1f}; "
          f"right stick: yaw ±{ranges.ang_vel_yaw[1]:.1f} rad/s; height fixed {env.height_target:.2f} m; A reset, B zero")
    print("=" * 72, flush=True)

    obs = hl_runner.reset() if hl_runner else env.reset()
    renderer.update(env.data)
    render_every = max(1, int(round((1 / env.dt) / args.render_hz)))
    falls, steps = 0, 0
    next_tick = time.perf_counter()
    last_status = 0.0
    last_yaw_pos = None
    with torch.inference_mode():
        while True:
            state = pad.poll()
            if state["connected"] and not use_sliders.value:
                raw = (state["ly"], state["lx"], state["rx"])
                if state["pressed"] & BUTTON_A:
                    flags["reset"] = True
                if state["pressed"] & BUTTON_B:
                    flags["zero"] = True
                source = "gamepad"
            else:
                raw = (s_vx.value / (ranges.lin_vel_x[1] if s_vx.value >= 0 else -ranges.lin_vel_x[0]),
                       -s_vy.value / ranges.lin_vel_y[1], -s_wz.value / ranges.ang_vel_yaw[1])
                source = "sliders" if use_sliders.value or not state["connected"] else "gamepad"
            if flags["zero"]:
                flags["zero"] = False
                mapper.state = [0., 0., 0.]
                s_vx.value, s_vy.value, s_wz.value = 0., 0., 0.
                raw = (0., 0., 0.)
            vx, vy, wz = mapper(raw)
            if flags["reset"]:
                flags["reset"] = False
                obs = hl_runner.reset() if hl_runner else env.reset()
                print("[mj-teleop] manual reset", flush=True)
            if hl_runner:
                obs, fallen = hl_runner.step((vx, vy, wz))
                failure = "fall" if fallen else hl_runner.geometry_failure()
                fallen = failure is not None
            else:
                env.set_command(vx, vy, wz)
                actions = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                obs, fallen = env.step(actions)
                failure = "fall" if fallen else None
            steps += 1
            if fallen:
                falls += 1
                print(f"[mj-teleop] {failure} -> reset (total {falls})", flush=True)
                obs = hl_runner.reset() if hl_runner else env.reset()
            if steps % render_every == 0:
                renderer.update(env.data)
                if follow.value:
                    pelvis = env.data.xpos[env.pelvis_id]
                    target = np.array([pelvis[0], pelvis[1], 0.7])
                    for client in server.get_clients().values():
                        with client.atomic():
                            client.camera.look_at = target
                            client.camera.position = target + camera_offset
            now = time.perf_counter()
            if now - last_status >= 0.5:
                bv = env.base_lin_vel
                line = (f"[{source}] cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | base vx={bv[0]:+.2f} vy={bv[1]:+.2f} "
                        f"wz={env.base_ang_vel[2]:+.2f} h={env.root_height:.3f}")
                md = [f"- 输入源: **{source}**", f"- 命令: vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}",
                      f"- 基座: vx {bv[0]:+.2f}  vy {bv[1]:+.2f}  wz {env.base_ang_vel[2]:+.2f}  高度 {env.root_height:.3f} m"]
                if hl_runner:
                    ch = hl_runner.cmd_homie
                    line += f" | homie cmd {ch[0]:+.2f} {ch[1]:+.2f} {ch[2]:+.2f} | upper |dq| {np.abs(hl_runner.upper_delta_q).max():.2f}"
                    md += [f"- 上层 → HOMIE 命令: vx {ch[0]:+.2f}  vy {ch[1]:+.2f}  wz {ch[2]:+.2f}；上肢修正最大 {np.abs(hl_runner.upper_delta_q).max():.2f} rad"]
                cart = env.cart_state()
                if cart is not None:
                    pos, quat, lin, ang = cart
                    forces = env.grasp_forces()
                    line += (f" | cart v={np.hypot(lin[0], lin[1]):.2f} yaw={_yaw_deg(quat):+.1f}° "
                             f"| grasp F L={forces.get('left', 0):.0f}N R={forces.get('right', 0):.0f}N")
                    md += [f"- 推车: 速度 {np.hypot(lin[0], lin[1]):.2f} m/s  偏航 {_yaw_deg(quat):+.1f}°  偏航率 {ang[2]:+.2f}",
                           f"- 抓握约束力: 左 {forces.get('left', 0):.0f} N  右 {forces.get('right', 0):.0f} N"]
                line += f" | falls={falls} t={steps * env.dt:.0f}s"
                md += [f"- 摔倒 {falls} 次 · 仿真时间 {steps * env.dt:.0f} s"]
                status.content = "\n".join(md)
                pad_state.content = ("XInput: 已连接" if state["connected"] else
                                     ("XInput: 未检测到手柄（用滑条）" if pad.available else "XInput: 不可用（非 Windows）"))
                print(line, flush=True)
                last_status = now
            next_tick += env.dt / max(rt.value, 0.05)
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:
                next_tick = time.perf_counter()


# ---------------------------------------------------------------------------- headless smoke

def run_mujoco_smoke(args) -> dict:
    import torch

    env, policy, checkpoint = _make(args)
    obs = env.reset()
    env.set_command(0., 0., 0.)
    falls, windows, base_vel, yaw_rate, cart_speed, grasp = 0, [], [], [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(args.steps):
            on = step >= args.settle_steps
            env.set_command(args.vx if on else 0., args.vy if on else 0., args.wz if on else 0.)
            actions = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, fallen = env.step(actions)
            if not np.isfinite(obs).all():
                raise FloatingPointError(f"non-finite observation at step {step}")
            if fallen:
                falls += 1
                obs = env.reset()
            base_vel.append(env.base_lin_vel[:2].copy())
            yaw_rate.append(float(env.base_ang_vel[2]))
            cart = env.cart_state()
            if cart is not None:
                cart_speed.append(float(np.hypot(cart[2][0], cart[2][1])))
                grasp.append(max(env.grasp_forces().values(), default=0.0))
            if (step + 1) % 100 == 0:
                seg = slice(step - 99, step + 1)
                speeds = np.linalg.norm(np.array(base_vel)[seg], axis=1)
                windows.append({"t_s": round((step + 1) * env.dt, 2), "cmd_vx": args.vx if on else 0.,
                                "base_speed_mean": float(speeds.mean()), "yaw_rate_abs_mean": float(np.abs(yaw_rate[seg]).mean()),
                                "cart_speed_mean": float(np.mean(cart_speed[seg])) if cart_speed else None,
                                "grasp_force_max_N": float(np.max(grasp[seg])) if grasp else None,
                                "root_height": env.root_height})
                print("[mj-smoke] window", json.dumps(windows[-1]), flush=True)
    elapsed = time.perf_counter() - start
    result = {"result": "PASS" if falls == 0 else "FALLS", "mode": args.mode, "steps": args.steps, "falls": falls,
              "checkpoint": str(checkpoint), "command": [args.vx, args.vy, args.wz],
              "realtime_factor": args.steps * env.dt / elapsed, "env": env.info(), "windows": windows}
    if env.with_cart:
        cart = env.cart_state()
        result["cart_travel_x_m"] = float(cart[0][0] - env.derived["base_pos_world"][0])
    path = ROOT / "logs/pushcart_mj_smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("[mj-smoke] " + json.dumps({k: v for k, v in result.items() if k != "windows"}), flush=True)
    return result
