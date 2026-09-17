"""Static viser checks of the initial grasp configuration.

``pose`` mode renders g1_grasp.urdf at the preset pose together with the cart
built from cart_params.yaml, with sliders to tune the grasp centre and bar
radius and a button to save both YAML files (+ regenerate cart.urdf).

``usd`` mode renders the packed g1_pushcart.usd with usd-core, without Kit.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from .assets.cart_urdf import (Primitive, build_cart, cart_primitives, derive_cart_pose, load_cart_params,
                               save_cart_params, write_cart_urdf)
from .paths import ARM_JOINTS, CART_OBJ, PACKED_USD, PALM_LINKS, WAIST_JOINTS
from .pose import Kinematics, load_initial_pose, save_initial_pose

HAND_COLOR = {"palm": (255, 140, 0), "finger": (255, 215, 70)}
BODY_COLOR = (178, 184, 190)
COLLISION_COLOR = (80, 140, 255)
BAR_COLOR = (220, 60, 60)
MIRROR = np.array([1., -1., 1.])


def _wxyz_pos(T: np.ndarray):
    q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]]), T[:3, 3].copy()


def _rpy_matrix(rpy):
    return R.from_euler("xyz", rpy).as_matrix()


def _primitive_mesh(prim: Primitive) -> trimesh.Trimesh:
    if prim.kind == "box":
        mesh = trimesh.creation.box(extents=prim.size)
    elif prim.kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=prim.size[0], height=prim.size[1], sections=32)
    else:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=prim.size[0])
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(prim.rpy)
    T[:3, 3] = prim.xyz
    mesh.apply_transform(T)
    return mesh


def wheel_primitives(params: dict) -> list[Primitive]:
    w = params["wheels"]
    prims = []
    for i, (x, y) in enumerate(w["positions_xy"]):
        wx = x - w["caster_trail"] if params["model"] == "caster" else x
        prims.append(Primitive(f"wheel_{i}", "cylinder", (w["radius"], w["width"]), (wx, y, w["radius"]),
                               rpy=(math.pi / 2, 0., 0.), collision=True))
    return prims


def load_cart_obj(params: dict) -> trimesh.Trimesh:
    mesh = trimesh.load(str(CART_OBJ), force="mesh")
    mesh.apply_scale(params["visual"]["scale"])
    mesh.apply_translation(params["visual"]["offset"])
    return mesh


def bar_surface_distance(points: np.ndarray, center, axis, radius: float, half_length: float) -> np.ndarray:
    """Signed distance from points to a finite cylinder surface (negative = inside the bar)."""
    d = points - np.asarray(center)
    axis = np.asarray(axis) / np.linalg.norm(axis)
    along = d @ axis
    radial = np.linalg.norm(d - along[:, None] * axis, axis=1)
    inside_span = np.abs(along) <= half_length
    beyond = np.abs(along) - half_length
    dist = np.where(inside_span, radial - radius, np.hypot(np.maximum(radial - radius, 0.), np.maximum(beyond, 0.)))
    return dist


def hand_clearance(kin: Kinematics, side: str, center, axis, radius: float, half_length: float) -> float:
    """Minimum distance from the hand's visual meshes (palm + fingers) to the bar surface."""
    graph = kin.urdf.scene.graph
    points = []
    for node in graph.nodes_geometry:
        link = graph.transforms.parents[node]
        if not link.startswith(f"{side}_hand"):
            continue
        T, geom = graph[node]
        verts = kin.urdf.scene.geometry[geom].vertices
        points.append((kin.world_T_base @ T @ np.c_[verts, np.ones(len(verts))].T).T[:, :3])
    return float(bar_surface_distance(np.concatenate(points), center, axis, radius, half_length).min())


class PoseView:
    def __init__(self, server: viser.ViserServer, show_obj: bool = True):
        self.server = server
        self.pose = load_initial_pose()
        self.params = load_cart_params()
        self.kin = Kinematics(load_meshes=True)
        self.robot_nodes = {}
        self.cart_nodes = {}
        self.cart_obj_handle = None
        self._obj_opacity = None
        self.derived = None
        self._build_static()
        self._build_gui(show_obj)
        self.refresh()

    # ------------------------------------------------------------------ scene
    def _build_static(self):
        s = self.server.scene
        s.add_grid("/ground", width=8, height=8, plane="xy", cell_size=0.25, section_size=1.0)
        s.add_frame("/world", axes_length=0.3, axes_radius=0.006)
        graph = self.kin.urdf.scene.graph
        for node in graph.nodes_geometry:
            _, geom = graph[node]
            link = graph.transforms.parents[node]
            mesh = self.kin.urdf.scene.geometry[geom]
            if "hand_palm" in link:
                color = HAND_COLOR["palm"]
            elif "_hand_" in link:
                color = HAND_COLOR["finger"]
            else:
                color = BODY_COLOR
            self.robot_nodes[node] = s.add_mesh_simple(f"/robot/{node}", mesh.vertices, mesh.faces, color=color)
        for side in ("left", "right"):
            s.add_frame(f"/palm_{side}", axes_length=0.06, axes_radius=0.003)
            s.add_icosphere(f"/grasp_{side}", radius=0.008, color=(40, 200, 80))
            s.add_icosphere(f"/planner_tcp_{side}", radius=0.006, color=(250, 250, 250), visible=False)
        s.add_frame("/cart", show_axes=True, axes_length=0.2, axes_radius=0.004)
        self.cart_obj_mesh = load_cart_obj(self.params)

    def _build_gui(self, show_obj: bool):
        g = self.server.gui
        g.add_markdown("### 推车抓握初始位姿 · 静态检查\n绿球 = 抓握中心；红杆 = 把手横杆；蓝色半透明 = 推车碰撞体；橙/黄 = Dex3 掌与手指。")
        with g.add_folder("抓握中心 (左掌坐标系; 右手默认镜像)"):
            p = self.pose["grasp"]["point_in_palm"]["left"]
            self.gp_x = g.add_slider("x (沿手指方向)", 0.0, 0.15, 0.001, float(p[0]))
            self.gp_y = g.add_slider("y (掌面法向)", -0.10, 0.10, 0.001, float(p[1]))
            self.gp_z = g.add_slider("z (指间方向)", -0.06, 0.06, 0.001, float(p[2]))
            self.mirror = g.add_checkbox("右手镜像左手 (x, -y, z)", True)
            pr = self.pose["grasp"]["point_in_palm"]["right"]
            self.gp_rx = g.add_slider("右 x", 0.0, 0.15, 0.001, float(pr[0]), visible=False)
            self.gp_ry = g.add_slider("右 y", -0.10, 0.10, 0.001, float(pr[1]), visible=False)
            self.gp_rz = g.add_slider("右 z", -0.06, 0.06, 0.001, float(pr[2]), visible=False)
        with g.add_folder("把手与推车"):
            self.bar_radius = g.add_slider("横杆半径 [m] (仅视觉)", 0.004, 0.03, 0.0005, float(self.params["handle"]["radius"]))
            self.show_obj = g.add_checkbox("显示 xiaotuiche.obj 视觉网格", show_obj)
            self.show_collision = g.add_checkbox("显示碰撞基本体", True)
            self.obj_opacity = g.add_slider("OBJ 透明度", 0.05, 1.0, 0.05, float(self.params["visual"]["opacity"]))
        with g.add_folder("显示"):
            self.show_frames = g.add_checkbox("显示掌坐标系", True)
            self.show_planner = g.add_checkbox("显示规划器 TCP 参考点 (白)", False)
        with g.add_folder("上肢关节 (预设值, 可微调)", expand_by_default=False):
            self.joint_sliders = {}
            for name in WAIST_JOINTS + ARM_JOINTS:
                self.joint_sliders[name] = g.add_slider(name.replace("_joint", ""), -3.2, 3.2, 0.005,
                                                        float(self.pose["robot"]["joint_pos"][name]))
        self.readout = g.add_markdown("")
        self.save_button = g.add_button("保存 YAML 并重建 cart.urdf", color="green")
        self.reset_button = g.add_button("恢复为文件中的值")

        for h in [self.gp_x, self.gp_y, self.gp_z, self.gp_rx, self.gp_ry, self.gp_rz, self.bar_radius,
                  self.show_obj, self.show_collision, self.obj_opacity, self.show_frames, self.show_planner,
                  *self.joint_sliders.values()]:
            h.on_update(lambda _: self.refresh())

        @self.mirror.on_update
        def _(_):
            for h in (self.gp_rx, self.gp_ry, self.gp_rz):
                h.visible = not self.mirror.value
            self.refresh()

        @self.save_button.on_click
        def _(_):
            self.save()

        @self.reset_button.on_click
        def _(_):
            self.pose = load_initial_pose()
            self.params = load_cart_params()
            p = self.pose["grasp"]["point_in_palm"]["left"]
            self.gp_x.value, self.gp_y.value, self.gp_z.value = map(float, p)
            pr = self.pose["grasp"]["point_in_palm"]["right"]
            self.gp_rx.value, self.gp_ry.value, self.gp_rz.value = map(float, pr)
            self.bar_radius.value = float(self.params["handle"]["radius"])
            for name, h in self.joint_sliders.items():
                h.value = float(self.pose["robot"]["joint_pos"][name])
            self.refresh()

    # ------------------------------------------------------------------ state
    def current_pose(self) -> dict:
        pose = {**self.pose, "robot": {**self.pose["robot"], "joint_pos": dict(self.pose["robot"]["joint_pos"])},
                "grasp": dict(self.pose["grasp"])}
        for name, h in self.joint_sliders.items():
            pose["robot"]["joint_pos"][name] = float(h.value)
        left = np.array([self.gp_x.value, self.gp_y.value, self.gp_z.value], dtype=float)
        right = left * MIRROR if self.mirror.value else np.array([self.gp_rx.value, self.gp_ry.value, self.gp_rz.value])
        pose["grasp"] = {**pose["grasp"], "point_in_palm": {"left": left.tolist(), "right": right.tolist()}}
        return pose

    def current_params(self) -> dict:
        params = {**self.params, "handle": {**self.params["handle"], "radius": float(self.bar_radius.value)}}
        return params

    def refresh(self):
        pose, params = self.current_pose(), self.current_params()
        kin = self.kin
        # legs are part of joint_pos; pelvis height comes from the stored value (feet on ground for the preset legs)
        kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])
        graph = kin.urdf.scene.graph
        for node, handle in self.robot_nodes.items():
            T, _ = graph[node]
            handle.wxyz, handle.position = _wxyz_pos(kin.world_T_base @ T)
        derived = derive_cart_pose(params, pose, kin)
        self.derived = derived
        s = self.server.scene
        for side in ("left", "right"):
            wxyz, pos = _wxyz_pos(kin.world_T(PALM_LINKS[side]))
            frame = s.add_frame(f"/palm_{side}", axes_length=0.06, axes_radius=0.003, wxyz=wxyz, position=pos,
                                visible=self.show_frames.value)
            s.add_icosphere(f"/grasp_{side}", radius=0.008, color=(40, 200, 80),
                            position=np.array(derived["grasp_points_world"][side]))
            hyp = pose["reference"]["planner_tcp_in_palm_hypotheses"]["homie_stance"][side]
            s.add_icosphere(f"/planner_tcp_{side}", radius=0.006, color=(250, 250, 250),
                            position=kin.grasp_point_world(side, hyp), visible=self.show_planner.value)
        # cart
        cart_T = np.eye(4)
        cart_T[:3, :3] = R.from_euler("z", derived["base_yaw"]).as_matrix()
        cart_T[:3, 3] = derived["base_pos_world"]
        wxyz, pos = _wxyz_pos(cart_T)
        s.add_frame("/cart", show_axes=True, axes_length=0.2, axes_radius=0.004, wxyz=wxyz, position=pos)
        prims = cart_primitives(params, derived["handle_height"]) + wheel_primitives(params)
        seen = set()
        for prim in prims:
            mesh = _primitive_mesh(prim)
            is_bar = prim.name.startswith("bar_")
            color = BAR_COLOR if is_bar else COLLISION_COLOR
            visible = True if is_bar else self.show_collision.value
            self.cart_nodes[prim.name] = s.add_mesh_simple(
                f"/cart/{prim.name}", mesh.vertices, mesh.faces, color=color,
                opacity=None if is_bar else 0.45, visible=visible)
            seen.add(prim.name)
        for name in list(self.cart_nodes):
            if name not in seen:
                self.cart_nodes.pop(name).remove()
        opacity = float(self.obj_opacity.value)
        if self.cart_obj_handle is None or self._obj_opacity != opacity:
            self.cart_obj_handle = s.add_mesh_simple("/cart/obj_visual", self.cart_obj_mesh.vertices,
                                                      self.cart_obj_mesh.faces, color=(150, 160, 170), opacity=opacity)
            self._obj_opacity = opacity
        self.cart_obj_handle.visible = bool(self.show_obj.value)
        # clearance between each hand's meshes and the near-end bar surface
        clearance = {side: hand_clearance(kin, side, derived["bar_center_world"], derived["bar_axis_world"],
                                          params["handle"]["radius"], params["handle"]["length"] / 2)
                     for side in ("left", "right")}
        gp = derived["grasp_points_world"]
        spacing = float(np.linalg.norm(np.array(gp["left"]) - np.array(gp["right"])))
        self.readout.content = "\n".join([
            "#### 数值",
            f"- pelvis 高度: **{pose['robot']['base_pos'][2]:.4f} m**（预设腿部角度、脚底贴地）",
            f"- 左掌 (世界): {np.round(kin.world_T(PALM_LINKS['left'])[:3, 3], 4).tolist()}",
            f"- 右掌 (世界): {np.round(kin.world_T(PALM_LINKS['right'])[:3, 3], 4).tolist()}",
            f"- 抓握点 左: {np.round(gp['left'], 4).tolist()} 右: {np.round(gp['right'], 4).tolist()}",
            f"- 双手间距: **{spacing:.3f} m**",
            f"- 把手高度 (杆轴离地): **{derived['handle_height']:.4f} m**；杆 x = {derived['bar_center_world'][0]:.4f} m",
            f"- 杆倾斜 (相对世界 y 轴): {derived['bar_tilt_deg']:.2f}°；推车 yaw {math.degrees(derived['base_yaw']):.2f}°",
            f"- 推车底盘中心 (世界): {np.round(derived['base_pos_world'], 4).tolist()}",
            f"- 手部网格到杆面最小间隙 左 **{clearance['left']*1000:.1f} mm** / 右 **{clearance['right']*1000:.1f} mm**（负值 = 穿透）",
            f"- 横杆半径 {params['handle']['radius']*1000:.1f} mm；推车模型 `{params['model']}`",
        ])

    def save(self):
        pose, params = self.current_pose(), self.current_params()
        pose["reference"] = {**pose.get("reference", {}),
                             "palm_world": {side: self.kin.world_T(PALM_LINKS[side])[:3, 3].tolist() for side in PALM_LINKS},
                             "grasp_point_world": self.derived["grasp_points_world"]}
        save_initial_pose(pose)
        self.pose = load_initial_pose()
        params["derived"] = {k: self.derived[k] for k in ("handle_height", "bar_center_world", "bar_axis_world", "base_pos_world", "base_yaw")}
        save_cart_params(params)
        self.params = load_cart_params()
        write_cart_urdf(self.params, self.derived["handle_height"])
        self.server.gui.add_markdown(f"已保存 {time.strftime('%H:%M:%S')}: initial_pose.yaml, cart_params.yaml, cart.urdf")


def run_pose_view(port: int = 8080, show_obj: bool = True):
    server = viser.ViserServer(port=port, label="HOMIE push-cart pose check")
    view = PoseView(server, show_obj=show_obj)

    def set_camera(client: viser.ClientHandle, position, look_at):
        with client.atomic():
            client.camera.look_at = np.asarray(look_at, dtype=float)
            client.camera.position = np.asarray(position, dtype=float)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # The client's camera is not initialised at connect time; apply the initial view a moment later.
        import threading

        def later():
            time.sleep(1.0)
            target = np.array(view.derived["bar_center_world"])
            set_camera(client, target + np.array([-0.55, 0.9, 0.35]), target)

        threading.Thread(target=later, daemon=True).start()

    with server.gui.add_folder("视角"):
        overview = server.gui.add_button("整体视角")
        hands = server.gui.add_button("手部特写")
        left_hand = server.gui.add_button("左手掌心方向 (沿杆轴看)")

        @overview.on_click
        def _(event: viser.GuiEvent):
            set_camera(event.client, [2.6, -2.4, 1.8], [0.5, 0., 0.6])

        @hands.on_click
        def _(event: viser.GuiEvent):
            target = np.array(view.derived["bar_center_world"])
            set_camera(event.client, target + np.array([-0.35, 0.5, 0.25]), target)

        @left_hand.on_click
        def _(event: viser.GuiEvent):
            target = np.array(view.derived["grasp_points_world"]["left"])
            set_camera(event.client, target + np.array([0.0, 0.45, 0.0]), target)
    print(f"[pushcart] viser pose view: http://localhost:{server.get_port()}  (Ctrl+C to stop)", flush=True)
    while True:
        time.sleep(1.0)


# ---------------------------------------------------------------------------- packed USD view

def run_usd_view(path: Path | None = None, port: int = 8081):
    """Render every Mesh prim of the packed stage at its authored world transform."""
    from pxr import Usd, UsdGeom

    path = Path(path) if path else PACKED_USD
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(path)
    server = viser.ViserServer(port=port, label="HOMIE push-cart USD check")
    server.scene.add_grid("/ground", width=8, height=8, plane="xy", cell_size=0.25, section_size=1.0)
    server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.006)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    count = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        if points.size == 0 or indices.size == 0:
            continue
        faces, k = [], 0
        for c in counts:
            poly = indices[k:k + c]
            for j in range(1, c - 1):
                faces.append((poly[0], poly[j], poly[j + 1]))
            k += c
        world = np.array(cache.GetLocalToWorldTransform(prim))  # row-major, row vectors
        verts = np.c_[points, np.ones(len(points))] @ world
        path_str = str(prim.GetPath())
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        is_collision = purpose == UsdGeom.Tokens.guide or "collision" in path_str.lower()
        color = COLLISION_COLOR if is_collision else (HAND_COLOR["finger"] if "_hand_" in path_str else BODY_COLOR)
        if "Cart" in path_str and not is_collision:
            color = (150, 160, 170)
        server.scene.add_mesh_simple(f"/usd{path_str}", verts[:, :3], np.array(faces), color=color,
                                     opacity=0.45 if is_collision else None)
        count += 1
    server.gui.add_markdown(f"### 打包 USD 静态检查\n文件: `{path}`\n网格数: {count}")
    print(f"[pushcart] viser usd view: http://localhost:{server.get_port()}  ({count} meshes; Ctrl+C to stop)", flush=True)
    while True:
        time.sleep(1.0)
