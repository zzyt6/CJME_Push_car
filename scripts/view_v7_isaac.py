"""Live view of the V7 push-cart task on the Isaac backend (Viser in the browser).

The simulation is the V7 Isaac task itself (packed V7 USD, realistic grasp, Coulomb cart); the browser shows the
USD's own visual geometry, each piece moved with its rigid body's pose read from Isaac every frame. A course from
pushcart_bench/config/benchmark.yaml is driven by the bench path follower; a run succeeds 0.5 m before the end of
the course; a termination (grip lost, fall, ...) is only recorded as an event -- the run continues, nothing is reset.

Run it on the CPU (--device cpu): one environment steps in ~16 ms there (real time: 20 ms) vs ~47 ms on a GPU shared
with training, where thousands of tiny kernels per step are launch-bound. A plotly figure costs ~45 ms, so one
figure is redrawn per 0.25 s.

    E:\\envs\\RLisaac601\\python.exe scripts/view_v7_isaac.py --headless --port 8091
    .\\run_v7_isaac_viewer.ps1 -Port 8091
"""
from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
V7_RUNS = ROOT / "logs/rsl_rl/homie_pushcart_v7"
OTHER_CKPT = {
    "V6": ROOT / "logs/rsl_rl/homie_pushcart_v6/2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt",
    "V3.2": ROOT / "logs/rsl_rl/homie_pushcart/2026-09-21_16-56-46_v3_1_speed_quality_scratch/model_19999.pt",
}
POLICIES = ("V7", "V6", "V3.2", "HOMIE")


# ----------------------------------------------------------------------------------------------- USD visuals
def usd_visuals(usd_path: Path):
    """Every renderable (purpose 'default') gprim of the packed USD, in its rigid body's frame.
    Returns [(body_name, prim_name, vertices (V,3), faces (F,3), rgb)]."""
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world = lambda prim: np.array(cache.GetLocalToWorldTransform(prim)).T       # noqa: E731  column convention
    out = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue
        body = prim.GetParent()
        while body and body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or not body.IsValid():
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            faces, k = [], 0
            for c in counts:                                    # fan triangulation
                faces.extend([idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1))
                k += c
            verts, faces = pts, np.asarray(faces, dtype=np.int64)
        elif prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0
            m = trimesh.creation.box(extents=[size] * 3)
            verts, faces = np.asarray(m.vertices), np.asarray(m.faces)
        elif prim.IsA(UsdGeom.Cylinder):
            g = UsdGeom.Cylinder(prim)
            m = trimesh.creation.cylinder(radius=g.GetRadiusAttr().Get(), height=g.GetHeightAttr().Get(), sections=32)
            axis = g.GetAxisAttr().Get()
            if axis == "X":
                m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
            elif axis == "Y":
                m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            verts, faces = np.asarray(m.vertices), np.asarray(m.faces)
        elif prim.IsA(UsdGeom.Sphere):
            m = trimesh.creation.icosphere(subdivisions=2, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            verts, faces = np.asarray(m.vertices), np.asarray(m.faces)
        else:
            continue
        rel = np.linalg.inv(world(body)) @ world(prim)
        verts = verts @ rel[:3, :3].T + rel[:3, 3]
        color = (150, 150, 155)
        dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if dc is not None and len(dc):
            color = tuple(int(255 * min(max(float(c), 0.), 1.)) for c in dc[0])
        name = body.GetName()
        if name == "base_link" and prim.GetName().startswith("bar_"):
            color = (60, 60, 70)                                # handle tube
        out.append((name, prim.GetName(), verts.astype(np.float32), faces.astype(np.int32), color))
    return out


def list_v7_runs():
    runs = sorted((p for p in V7_RUNS.iterdir() if p.is_dir() and any(p.glob("model_*.pt"))),
                  key=lambda p: p.stat().st_mtime, reverse=True) if V7_RUNS.is_dir() else []
    return [p.name for p in runs]


def list_checkpoints(run: str):
    files = sorted((V7_RUNS / run).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    return [p.name for p in files]


# ----------------------------------------------------------------------------------------------- viewer
class IsaacV7Viewer:
    def __init__(self, port: int, args):
        import viser
        from pushcart_bench.config_io import load_config
        self.bench = load_config()
        self.server = viser.ViserServer(port=port, label="V7 Isaac push-cart")
        gui = self.server.gui
        self.title = gui.add_markdown("### V7 推车 · Isaac 后端")
        self.status = gui.add_markdown("_启动中…_")
        runs = list_v7_runs() or ["(无)"]
        with gui.add_folder("运行"):
            paths = list(self.bench["paths"].keys())
            self.dd_path = gui.add_dropdown("路径", paths, initial_value=args.path if args.path in paths else paths[0])
            self.dd_pol = gui.add_dropdown("策略", list(POLICIES), initial_value="V7")
            self.dd_run = gui.add_dropdown("V7 训练运行", runs, initial_value=args.run if args.run in runs else runs[0])
            ck = list_checkpoints(self.dd_run.value) if runs[0] != "(无)" else ["(无)"]
            self.dd_ckpt = gui.add_dropdown("V7 检查点", ck, initial_value=args.checkpoint if args.checkpoint in ck else ck[0])
            self.dd_run.on_update(lambda _: self._refresh_ckpts())
            self.sl_vnom = gui.add_slider("巡线速度 vx [m/s]", 0.15, 0.9, 0.05, args.v_nom)
            self.sl_payload = gui.add_slider("载荷 [kg]", 0.0, 25.0, 0.5, args.payload)
            self.sl_diff = gui.add_slider("难度 d（握持能力/物理随机范围）", 0.0, 1.0, 0.05, args.difficulty)
            self.cb_dist = gui.add_checkbox("扰动（碰撞力 / 坡度 / 推机器人）", False)
            self.cb_cruise = gui.add_checkbox("保持指定速度 (弯道不减速)", True)
            self.btn = gui.add_button("开始", color="green")
            self.btn.on_click(lambda _: self._request())
        with gui.add_folder("显示"):
            self.cb_follow = gui.add_checkbox("镜头跟随", True)
            self.cb_force = gui.add_checkbox("手部受力箭头 (1 m = 250 N)", True)
            self.sl_speed = gui.add_slider("播放倍率 (1 = 实时, 0 = 不限速)", 0.0, 4.0, 0.1, 1.0)
        with gui.add_folder("曲线 (最近 10 s)"):
            self.plot_v = gui.add_plotly(self._fig("vx", []), aspect=2.2)
            self.plot_w = gui.add_plotly(self._fig("wz", []), aspect=2.2)
            self.plot_y = gui.add_plotly(self._fig("vy", []), aspect=2.2)
            self.plot_g = gui.add_plotly(self._fig("握持载荷 / 能力", []), aspect=2.2)
        with gui.add_folder("已完成的运行", expand_by_default=False):
            self.results_md = gui.add_markdown("_尚无_")
        self.rows = []
        self.hist = {k: deque(maxlen=500) for k in ("t", "cmd_vx", "vx", "cmd_vy", "vy", "cmd_wz", "wz", "ct",
                                                   "pull_l", "pull_r", "roll", "tilt", "yaw")}
        self.request = threading.Event()
        self.request.set()                                  # start one run immediately
        self.trail = deque(maxlen=4000)
        self.path_handles, self.trail_handle, self.arrow_handles = [], None, []
        self.camera_offset = __import__("numpy").array([-3.5, -4.5, 2.6])

    def _refresh_ckpts(self):
        ck = list_checkpoints(self.dd_run.value)
        self.dd_ckpt.options = ck
        self.dd_ckpt.value = ck[0]

    def _request(self):
        self.request.set()

    @staticmethod
    def _fig(title, traces):
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, x, y, dash in traces:
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=name, line=dict(dash=dash, width=1.6)))
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.97), margin=dict(l=40, r=10, t=28, b=28),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)), template="plotly_white", height=230)
        return fig

    def add_meshes(self, visuals):
        self.meshes = []
        for body, prim, v, f, color in visuals:
            h = self.server.scene.add_mesh_simple(f"/isaac/{body}/{prim}", v, f, color=color)
            self.meshes.append((body, h))
        self.server.scene.add_grid("/ground", width=80, height=80, plane="xy", cell_size=0.5, section_size=2.0)

    def draw_path(self, path):
        from pushcart_bench.view import BenchViewer
        for h in self.path_handles:
            h.remove()
        self.path_handles = BenchViewer._draw_dashed_path(self.server.scene, path.xy)
        gx, gy, _ = path.goal
        self.path_handles.append(self.server.scene.add_icosphere("/bench/goal", radius=0.12, color=(230, 60, 60), position=(gx, gy, 0.12)))
        self.la = self.server.scene.add_icosphere("/bench/lookahead", radius=0.06, color=(60, 120, 255), position=(gx, gy, 0.06))
        self.path_handles.append(self.la)
        self.trail.clear()
        if self.trail_handle is not None:
            self.trail_handle.remove()
            self.trail_handle = None
        for h in self.hist.values():
            h.clear()


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--path", default="composite_course")
    p.add_argument("--run", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--payload", type=float, default=15.0)
    p.add_argument("--v_nom", type=float, default=0.5)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--finish_margin_m", type=float, default=0.5)
    p.add_argument("--timeout_s", type=float, default=90.0)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import dataclasses
        import numpy as np
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v7.paths import PACKED_USD
        from homie_pushcart_v7.task_config import PushCartTaskCfg
        from homie_pushcart_v7.task_env import PushCartEnv, OVERLOAD_KINDS
        from pushcart_bench.follower import FollowerConfig, PathFollower
        from pushcart_bench.paths import build_path

        view = IsaacV7Viewer(a.port, a)
        view.add_meshes(usd_visuals(PACKED_USD))
        dev = a.device
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = 1
        task = PushCartTaskCfg(); task.curriculum_v7.enabled = False
        task.env.episode_length_s = a.timeout_s + 10.0
        env = PushCartEnv(cfg, task, ROOT, dev, True, level=0)
        origin = env.env_origins[0].cpu().numpy()
        bodies = {}
        for asset in (env.robot, env.cart):
            for i, n in enumerate(asset.body_names):
                bodies.setdefault(n, (asset, i))
        view.meshes = [(b, h) for b, h in view.meshes if b in bodies]
        gl = [env.cart.body_names.index(f"grasp_{s}_link") for s in ("left", "right")]
        snap = {}
        orig_ct = env.check_termination

        def check_termination():
            """Record terminations as events, never reset: this viewer is for watching a run end to end."""
            orig_ct()
            snap["kinds"] = env.grasp_overload_kind.clone()
            snap["reasons"] = {k: v.clone() for k, v in env.term_reason_buf.items()}
            snap["event"] = env.terminated_buf.clone()
            env.overload_counter[:] = 0                  # a new overload event needs another full persist window
            env.terminated_buf[:] = False
            env.reset_buf[:] = env.time_out_buf
        env.check_termination = check_termination
        loaded = {}

        def policy(kind):
            key = (kind, view.dd_run.value, view.dd_ckpt.value) if kind == "V7" else (kind,)
            if key in loaded:
                return loaded[key]
            if kind == "V7":
                import homie_pushcart_v7.task_config as tcm
                from homie_pushcart_v7.hl_policy import HighLevelPolicy as P7
                orig = tcm.task_cfg_from_run
                tcm.task_cfg_from_run = lambda *x, **k: None   # older V7 runs predate newer config keys; actor only
                try:
                    pol = P7(V7_RUNS / view.dd_run.value / view.dd_ckpt.value, device=dev)
                finally:
                    tcm.task_cfg_from_run = orig
                fn = lambda o: pol(o["policy"], o["estimator"])                            # noqa: E731
            elif kind == "V6":
                from homie_pushcart_v6.hl_policy import HighLevelPolicy as P6
                pol = P6(OTHER_CKPT["V6"], device=dev)
                fn = lambda o: pol(o["policy"], o["estimator"])                            # noqa: E731
            elif kind == "V3.2":
                from homie_pushcart.hl_policy import HighLevelPolicy as P32
                pol = P32(OTHER_CKPT["V3.2"], 620, 652, 18, device=dev)
                fn = lambda o: pol(env.legacy_actor_obs)                                   # noqa: E731
            else:
                fn = lambda o: torch.zeros(1, 18, device=dev)                              # noqa: E731
            loaded.clear()
            loaded[key] = fn
            return fn

        def act(fn):
            with torch.inference_mode():
                return torch.as_tensor(fn(env.observation_dict()), device=dev).clone()

        def quiet():
            env.disturbance_frac[:] = 0.; env.bias_force_w[:] = 0.; env.bump_force_w[:] = 0.; env.bump_timer[:] = 0
            env.push_scale[:] = 0.

        def render(follower):
            with view.server.atomic():
                cache, all_pose = {}, {id(env.robot): env.robot.data.body_link_pose_w.torch[0].cpu().numpy(),
                                       id(env.cart): env.cart.data.body_link_pose_w.torch[0].cpu().numpy()}
                for body, h in view.meshes:
                    if body not in cache:
                        asset, i = bodies[body]
                        pose = all_pose[id(asset)][i]
                        q = pose[3:7]                                             # x y z w
                        cache[body] = (pose[:3] - origin, np.array([q[3], q[0], q[1], q[2]]))
                    h.position, h.wxyz = cache[body]
                x, y = cache["base_link"][0][:2] if "base_link" in cache else (0., 0.)
                lx, ly = follower.state.lookahead_xy if follower else (x, y)
                view.la.position = (lx, ly, 0.06)
                # hand force arrows from each grasp point
                for hnd in view.arrow_handles:
                    hnd.remove()
                view.arrow_handles = []
                if view.cb_force.value and bool(env._hand_valid[0]):
                    yaw = float(env.ts.cart_yaw[0]); c, s = math.cos(yaw), math.sin(yaw)
                    F = env.ts.grasp_force_c[0].cpu().numpy()
                    ratio = env.ts.grasp_ratio[0].cpu().numpy().max(-1)
                    for k in range(2):
                        p0 = env.cart.data.body_link_pose_w.torch[0, gl[k], :3].cpu().numpy() - origin
                        fw = np.array([c * F[k, 0] - s * F[k, 1], s * F[k, 0] + c * F[k, 1], F[k, 2]])
                        r = float(min(ratio[k], 1.5) / 1.5)
                        col = (int(60 + 195 * r), int(200 - 160 * r), 60)
                        view.arrow_handles.append(view.server.scene.add_spline_catmull_rom(
                            f"/force/{k}", np.stack((p0, p0 + fw / 250.)), color=col, line_width=5.0))
                pt = np.array([x, y, 0.03])
                view.frame = getattr(view, "frame", 0) + 1
                if not view.trail or np.linalg.norm(pt - view.trail[-1]) > 1e-3:
                    view.trail.append(tuple(pt))
                    if len(view.trail) > 2 and view.frame % 5 == 0:
                        view.trail_handle = view.server.scene.add_point_cloud(
                            "/bench/trail", np.asarray(view.trail, dtype=np.float32)[-1500:], colors=(40, 200, 90), point_size=.018)
                if view.cb_follow.value:
                    target = np.array([x, y, 0.6])
                    for client in view.server.get_clients().values():
                        with client.atomic():
                            client.camera.look_at = target
                            client.camera.position = target + view.camera_offset

        def refresh_text(t, follower, cmd, vel, label, extra=""):
            st = follower.state
            g = env
            mult = float(g.capacity_mult[0])
            lines = [f"- {label}",
                     f"- t = {t:.1f} s · 进度 **{st.s:.2f} / {follower.path.length:.2f} m** · 横向误差 {st.cross_track:+.3f} m · 航向误差 {math.degrees(st.heading_err):+.1f}°",
                     f"- 命令 vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f} · 实际 vx {vel[0]:+.2f} vy {vel[1]:+.2f} wz {vel[2]:+.2f}",
                     f"- 本次握持：μ {float(g.grip_mu[0, 0]):.2f}/{float(g.grip_mu[0, 1]):.2f} · N {float(g.grip_normal[0, 0]):.0f}/{float(g.grip_normal[0, 1]):.0f} N · "
                     f"回拉能力 {float(g.pull_capacity[0, 0]) * mult:.0f}/{float(g.pull_capacity[0, 1]) * mult:.0f} N · 绕管 {float(g.roll_capacity[0, 0]) * mult:.1f}/{float(g.roll_capacity[0, 1]) * mult:.1f} N·m · 倍率 {mult:.2f}",
                     f"- 沿杆滑移 左 {1e3 * float(g.slide_anchor[0, 0] - g.slide_init[0, 0]):+.1f} / 右 {1e3 * float(g.slide_anchor[0, 1] - g.slide_init[0, 1]):+.1f} mm · "
                     f"转角/断开角 {float(g.ts.pivot[0].max()):.2f} · 内力占比 {float(g.ts.internal_frac[0]):.2f}"]
            if extra:
                lines.append(extra)
            view.status.content = "\n".join(lines)
            h = view.hist
            tt = list(h["t"])[::2]                              # <= 250 points per trace
            if len(tt) > 2:
                d = lambda k: list(h[k])[::2]                   # noqa: E731
                view.plot_turn = (getattr(view, "plot_turn", -1) + 1) % 4   # one figure per refresh (~45 ms each)
                k = view.plot_turn
                if k < 3:
                    axis, plot, unit = (("vx", view.plot_v, "m/s"), ("wz", view.plot_w, "rad/s"), ("vy", view.plot_y, "m/s"))[k]
                    plot.figure = view._fig(f"{axis} [{unit}]: 命令 / 实际",
                                            [("命令", tt, d("cmd_" + axis), "dash"), ("实际", tt, d(axis), "solid")])
                else:
                    view.plot_g.figure = view._fig("握持载荷 / 能力（>1 超限）", [
                        ("回拉 左", tt, d("pull_l"), "solid"), ("回拉 右", tt, d("pull_r"), "solid"),
                        ("绕管 max", tt, d("roll"), "dot"), ("前后倾 max", tt, d("tilt"), "dot"), ("偏转 max", tt, d("yaw"), "dot"),
                        ("横向误差 m", tt, d("ct"), "dash")])

        follower, running, label = None, False, ""
        t0, next_tick, last_text, steps = 0., time.perf_counter(), 0., 0
        while app.app.is_running():
            if view.request.is_set():
                view.request.clear()
                kind = view.dd_pol.value
                view.status.content = f"_加载 {kind} 并复位…_"
                fn = policy(kind)
                env.cur.d[:] = float(view.sl_diff.value)
                env._refresh_global_level()
                if not view.cb_dist.value and dataclasses.is_dataclass(env.level_params):
                    env.level_params = dataclasses.replace(env.level_params, disturbance_frac=0.0)
                env.fixed_payload_kg = float(view.sl_payload.value)
                env.fixed_command = [0., 0., 0.]
                env.reset_idx(torch.arange(1, device=dev)); env.reset()
                for _ in range(50):
                    env.step(act(fn))
                    if not view.cb_dist.value:
                        quiet()
                fcfg = FollowerConfig.from_dict({**view.bench["follower"], "cruise": bool(view.cb_cruise.value),
                                                 "v_nom": float(view.sl_vnom.value), "timeout_s": a.timeout_s, "near_dist": 0.0})
                course = build_path(view.dd_path.value, view.bench["paths"][view.dd_path.value])
                pose = env.cart.data.root_link_pose_w.torch[0, :3].cpu().numpy() - origin
                path = course.transformed(pose[:2], float(env.ts.cart_yaw[0]))
                follower = PathFollower(path, fcfg, env.dt)
                view.draw_path(path)
                env.fixed_command = None
                ckpt = f" {view.dd_run.value}/{view.dd_ckpt.value}" if kind == "V7" else ""
                label = (f"策略 **{kind}**{ckpt} · 路径 {view.dd_path.value} · {view.sl_payload.value:.1f} kg · d {view.sl_diff.value:.2f} · "
                         f"扰动 {'开' if view.cb_dist.value else '关'}")
                running, steps, next_tick = True, 0, time.perf_counter()
                events, events_active = [], False
                wall0, last_log = time.perf_counter(), time.perf_counter()
            if not running:
                render(follower) if follower else None
                time.sleep(0.05)
                continue
            pose = env.cart.data.root_link_pose_w.torch[0, :3].cpu().numpy() - origin
            yaw = float(env.ts.cart_yaw[0])
            vel = torch.cat((env.ts.cart_v[0], env.ts.cart_wz[0:1])).cpu().numpy()
            cmd = follower.update((pose[0], pose[1], yaw), vel)
            env.cmd_task[:] = torch.as_tensor(cmd, device=dev, dtype=torch.float32)
            env.cmd_timer[:] = 10 ** 9
            env.step(act(fn))
            if not view.cb_dist.value:
                quiet()
            steps += 1
            t = follower.state.t
            h = view.hist
            r = env.ts.grasp_ratio[0].cpu().numpy()
            h["t"].append(t); h["cmd_vx"].append(cmd[0]); h["vx"].append(vel[0]); h["cmd_vy"].append(cmd[1]); h["vy"].append(vel[1])
            h["cmd_wz"].append(cmd[2]); h["wz"].append(vel[2]); h["ct"].append(follower.state.cross_track)
            h["pull_l"].append(float(r[0, 0])); h["pull_r"].append(float(r[1, 0]))
            h["roll"].append(float(r[:, 2].max())); h["tilt"].append(float(r[:, 3].max())); h["yaw"].append(float(r[:, 4].max()))
            result = None
            ev = bool(snap.get("event", torch.zeros(1, dtype=torch.bool))[0])
            if ev and not events_active:                 # rising edge: one event per episode of grip loss
                kinds = [k for j, k in enumerate(OVERLOAD_KINDS) if bool(snap["kinds"][0, j])]
                reasons = [k for k, v in snap["reasons"].items() if bool(v[0])]
                events.append(f"{t:.1f} s / {follower.state.s:.1f} m：{'+'.join(kinds) or '+'.join(reasons)}")
            events_active = ev
            if follower.state.s >= follower.path.length - a.finish_margin_m:
                result = f"到达终点前 {a.finish_margin_m} m，用时 {t:.1f} s，脱手/失败事件 {len(events)} 次"
            elif t >= a.timeout_s:
                result = f"超时：{a.timeout_s:.0f} s 只走到 {follower.state.s:.1f} m，脱手/失败事件 {len(events)} 次"
            if steps % 2 == 0 or result:
                render(follower)
            now = time.perf_counter()
            if now - last_text > 0.25 or result:
                last_text = now
                rt = t / max(now - wall0, 1e-6)
                ev_text = (f"- 脱手/失败事件 {len(events)} 次：" + "；".join(events[-4:])) if events else "- 脱手/失败事件 0 次"
                refresh_text(t, follower, cmd, vel, label + f" · 实时倍率 {rt:.2f}",
                             ev_text + (("\n- **" + result + "**") if result else ""))
                if now - last_log > 5.0:
                    last_log = now
                    print(f"[viewer] t {t:.1f} s  s {follower.state.s:.1f} m  realtime x{rt:.2f}", flush=True)
            if result:
                running = False
                xt = np.abs(np.asarray(h["ct"]))
                view.rows.append(f"| {label.split(' · ')[0]} | {view.dd_path.value} | {view.sl_payload.value:.0f} | {result} | {np.sqrt((xt ** 2).mean()):.3f} |")
                view.results_md.content = "\n".join(["| 策略 | 路径 | kg | 结果 | 横向 RMS (最近10 s) |", "|---|---|---|---|---|"] + view.rows[-20:])
                continue
            if view.sl_speed.value > 0.05:
                next_tick += env.dt / view.sl_speed.value
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -0.5:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush()
        raise
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
