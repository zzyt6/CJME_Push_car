"""Optional live view of a benchmark run: MuJoCo scene, reference path, goal, lookahead point, trail, telemetry."""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from threading import RLock
import time

import numpy as np


def checkpoint_policy_spec(checkpoint) -> str:
    """Route V3/V6 checkpoints or None (passthrough) to the matching bench adapter."""
    if checkpoint is None:
        return "passthrough"
    from homie_pushcart.mujoco_viewer import checkpoint_family
    path = Path(checkpoint).resolve()
    kind = "v6" if checkpoint_family(path) == "v6" else "two_stage"
    return f"{kind}:{path}"


class BenchViewer:
    def __init__(self, port: int = 8080, speed: float = 1.0, render_hz: float = 25.0):
        import viser
        from homie_pushcart.mujoco_teleop import MujocoViserRenderer
        self._Renderer = MujocoViserRenderer
        self.server = viser.ViserServer(port=port, label="push-cart benchmark")
        self.renderer, self._model_id = None, None
        self.render_hz = render_hz
        self.trail = deque(maxlen=8000)
        self.hist = {k: deque(maxlen=500) for k in ("t", "cmd_vx", "vx", "cmd_vy", "vy", "cmd_wz", "wz",
                                                   "est_vx", "est_vy", "est_wz", "ct", "fl", "fr")}
        gui = self.server.gui
        self.title = gui.add_markdown("### 推车控制对比实验")
        self.status = gui.add_markdown("")
        with gui.add_folder("运行"):
            self.dd_path = gui.add_dropdown("路径", ["s_large"], initial_value="s_large")
            self.dd_pol = gui.add_dropdown("策略", ["passthrough"], initial_value="passthrough")
            self.sl_vnom = gui.add_slider("指定巡线速度 vx [m/s]", 0.15, 0.90, 0.05, 0.50)
            self.sl_payload = gui.add_slider("载荷 [kg]", 0.0, 50.0, 0.5, 5.0)
            self.cb_cruise = gui.add_checkbox("保持指定速度 (弯道不减速)", True)
            self.btn_start = gui.add_button("开始", color="green")
            self.btn_start.on_click(lambda _: self.request_start())
        with gui.add_folder("显示"):
            self.follow = gui.add_checkbox("镜头跟随", True)
            self.speed = gui.add_slider("播放倍率 (1 = 实时, 0 = 不限速)", 0.0, 4.0, 0.1, speed)
        with gui.add_folder("曲线 (最近 10 s)"):
            self.plot_v = gui.add_plotly(self._figure("vx", []), aspect=2.2)
            self.plot_y = gui.add_plotly(self._figure("vy", []), aspect=2.2)
            self.plot_w = gui.add_plotly(self._figure("wz", []), aspect=2.2)
            self.plot_e = gui.add_plotly(self._figure("横向误差 / 抓握力", []), aspect=2.2)
        with gui.add_folder("已完成的运行", expand_by_default=False):
            self.results_md = gui.add_markdown("_尚无_")
        self.results_rows = []
        self.camera_offset = np.array([-4.0, -5.0, 2.8])
        self.want_rerun = False
        self.live_policy = None
        self._selection_lock = RLock()
        self._k_curv0 = 0.6
        self._vmin0 = 0.35
        self._la_max0 = 1.0
        self._steps = 0
        self._last_plot = 0.0
        self._next_tick = time.perf_counter()
        self._goal = None
        self._path_handle = self._goal_handle = self._la_handle = self._trail_handle = None
        self._path_handles = []
        print(f"[bench-view] viser: http://localhost:{self.server.get_port()}", flush=True)

    def request_start(self):
        """Snapshot the checkpoint at the button click; load only in the simulation thread."""
        with self._selection_lock:
            if self.live_policy is not None:
                try:
                    self.requested_checkpoint = self._checkpoint_from_gui()
                except (ValueError, FileNotFoundError) as exc:
                    self.md_load.content = f"**无法开始：** `{exc}`。当前模型保持不变。"
                    return
            self.want_rerun = True

    def setup_checkpoint_selection(self, policy, *, level=17, homie_checkpoint=None):
        """Optional V6 entry-point controls. Merely listing runs never loads their tensors."""
        from homie_pushcart.mujoco_viewer import checkpoint_family, list_checkpoints, list_viewer_runs
        self.live_policy = policy
        self._policy_level, self._homie_checkpoint = level, homie_checkpoint
        self.loaded_checkpoint = Path(policy.hl_checkpoint).resolve()
        self.selected_checkpoint = self.requested_checkpoint = self.loaded_checkpoint
        self._syncing_checkpoints = False
        runs = set(list_viewer_runs()) | {self.loaded_checkpoint.parent}
        self.run_choices = {"passthrough · 直通 HOMIE": None}
        for run in sorted(runs, key=lambda p: (p.name, str(p.parent)), reverse=True):
            try:
                family = "V6" if checkpoint_family(run) == "v6" else "V3 / 更早版本"
            except Exception:
                family = "配置待检查"
            label = f"{family} · {run.name}"
            if label in self.run_choices:
                label += f" ({run.parent})"
            self.run_choices[label] = run
        initial_run = next(label for label, path in self.run_choices.items() if path == self.loaded_checkpoint.parent)
        self.dd_pol.visible = False
        with self.server.gui.add_folder("模型选择"):
            self.dd_run = self.server.gui.add_dropdown("模型 / 训练运行", tuple(self.run_choices), initial_value=initial_run)
            names = [p.name for p in list_checkpoints(self.loaded_checkpoint.parent)]
            if self.loaded_checkpoint.name not in names:
                names.append(self.loaded_checkpoint.name)
            self.dd_ckpt = self.server.gui.add_dropdown("检查点", tuple(names), initial_value=self.loaded_checkpoint.name)
            self.md_model = self.server.gui.add_markdown(f"当前模型：**{policy.name}**")
            self.md_load = self.server.gui.add_markdown("选择模型和检查点后，点击 **开始** 加载并从起点重新运行。passthrough 无需检查点。")
        self._checkpoint_run = self.loaded_checkpoint.parent
        self._checkpoint_names = tuple(names)
        self.dd_run.on_update(lambda _: self._select_run())
        self.dd_ckpt.on_update(lambda _: self._select_checkpoint())

    def _select_run(self):
        self._select_checkpoint()

    def _select_checkpoint(self):
        with self._selection_lock:
            if self._syncing_checkpoints:
                return
            try:
                checkpoint = self._checkpoint_from_gui()
            except (ValueError, FileNotFoundError) as exc:
                self.md_load.content = f"**选择无效：** `{exc}`。当前模型保持不变。"
                return
            selected = "passthrough · 直通 HOMIE（无需检查点）" if checkpoint is None else f"{checkpoint.parent.name}/{checkpoint.name}"
            self.md_load.content = f"待加载：`{selected}`。点击 **开始** 生效。"

    def _checkpoint_from_gui(self):
        """Resolve a consistent current GUI pair; browser callbacks can arrive out of order.

        The lock serializes our callbacks. Viser updates handle values before scheduling
        them, so also recheck the run after disk discovery and after reading the filename.
        A late run callback must not replace a checkpoint the user already selected.
        """
        from homie_pushcart.mujoco_viewer import list_checkpoints
        while True:
            run_label = self.dd_run.value
            run = self.run_choices[run_label]
            if run is None:
                self._syncing_checkpoints = True
                try:
                    self.dd_ckpt.disabled = True
                    self._checkpoint_names = ("无需检查点",)
                    self.dd_ckpt.options = self._checkpoint_names
                    self.dd_ckpt.value = self._checkpoint_names[0]
                    self._checkpoint_run = None
                finally:
                    self._syncing_checkpoints = False
                if self.dd_run.value != run_label:
                    continue
                self.selected_checkpoint = None
                return None
            if self._checkpoint_run != run:
                checkpoints = list_checkpoints(run)
                if self.dd_run.value != run_label:
                    continue
                if not checkpoints:
                    raise FileNotFoundError(f"该运行没有可用检查点：{run}")
                self._syncing_checkpoints = True
                try:
                    self._checkpoint_names = tuple(p.name for p in checkpoints)
                    self.dd_ckpt.options = self._checkpoint_names
                    self.dd_ckpt.value = checkpoints[-1].name
                    self.dd_ckpt.disabled = False
                    self._checkpoint_run = run
                finally:
                    self._syncing_checkpoints = False
            filename = self.dd_ckpt.value
            if self.dd_run.value != run_label:
                continue
            if filename not in self._checkpoint_names:
                raise ValueError(f"检查点 {filename!r} 不属于当前训练运行 {run.name}")
            checkpoint = (run / filename).resolve()
            if checkpoint.parent != run or not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            self.selected_checkpoint = checkpoint
            return checkpoint

    def prepare_selected_policy(self, condition):
        """Build and reset a candidate before committing; failed loads leave the live plant intact."""
        from .policies import make_policy
        checkpoint = self.requested_checkpoint
        if checkpoint == self.loaded_checkpoint:
            self.md_load.content = "使用当前模型，从起点重新运行。"
            return self.live_policy
        selected = "passthrough" if checkpoint is None else f"{checkpoint.parent.name}/{checkpoint.name}"
        self.md_load.content = f"正在加载 `{selected}`…"
        try:
            if checkpoint is not None and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            candidate = make_policy(checkpoint_policy_spec(checkpoint), level=self._policy_level,
                                    homie_checkpoint=self._homie_checkpoint, scene_version="original")
            condition.apply(candidate)
        except Exception as exc:
            self.md_load.content = f"**加载失败：** `{type(exc).__name__}: {exc}`\n\n当前模型 **{self.live_policy.name}** 保持不变。"
            print(f"[bench-view] checkpoint load failed: {checkpoint}: {exc}", flush=True)
            return None
        self.live_policy = candidate
        self.loaded_checkpoint = checkpoint
        self.md_model.content = f"当前模型：**{candidate.name}**"
        self.md_load.content = "模型已加载；从起点重新运行，观测历史、估计器历史和轨迹已重置。"
        return candidate

    def setup_live(self, path_names, policy_names, payload: float = 5.0, v_nom: float = 0.5, cruise: bool = True,
                   follower_defaults=None):
        """Fill the live-run dropdowns / sliders used by ``--view`` path following."""
        self.dd_path.options = list(path_names)
        self.dd_path.value = path_names[0]
        self.dd_pol.options = list(policy_names)
        self.dd_pol.value = policy_names[0]
        self.sl_payload.value = float(payload)
        self.sl_vnom.value = float(v_nom)
        self.cb_cruise.value = bool(cruise)
        if follower_defaults:
            self._k_curv0 = float(follower_defaults.get("k_curv", 0.6))
            self._vmin0 = float(follower_defaults.get("v_min_frac", 0.35))
            self._la_max0 = float(follower_defaults.get("lookahead_max", 1.0))

    def sync_follower(self, fcfg, path_length: float | None = None):
        """Push the current GUI speed / cruise flags onto the live follower config (each sim step)."""
        fcfg.v_nom = float(self.sl_vnom.value)
        fcfg.cruise = bool(self.cb_cruise.value)
        if fcfg.cruise:
            fcfg.k_curv = 0.0
            fcfg.v_min_frac = 1.0
            fcfg.lookahead_max = max(self._la_max0, 1.8)
        else:
            fcfg.k_curv = self._k_curv0
            fcfg.v_min_frac = self._vmin0
            fcfg.lookahead_max = self._la_max0
        if path_length is not None:
            fcfg.timeout_s = max(60.0, 2.5 * float(path_length) / max(fcfg.v_nom, 0.15) + 20.0)

    @staticmethod
    def _figure(title, traces):
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, x, y, dash in traces:
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=name, line=dict(dash=dash, width=1.6)))
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.97), margin=dict(l=40, r=10, t=28, b=28),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)), template="plotly_white", height=230)
        return fig

    def _clear_path(self):
        for h in list(self._path_handles) + ([self._path_handle] if self._path_handle is not None else []):
            try:
                h.remove()
            except Exception:
                h.visible = False
        self._path_handles = []
        self._path_handle = None

    @staticmethod
    def _draw_dashed_path(scene, xy, z: float = 0.03, dash: float = 0.28, gap: float = 0.18):
        """Black dashed ground polyline (viser has no native dash style)."""
        pts2 = np.asarray(xy, dtype=float)
        if len(pts2) < 2:
            return []
        seg = np.linalg.norm(np.diff(pts2, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(cum[-1])
        handles = []
        t, i = 0.0, 0
        while t < length - 1e-4:
            t1 = min(t + dash, length)
            ss = np.linspace(t, t1, max(3, int(round((t1 - t) / 0.06)) + 1))
            xs = np.interp(ss, cum, pts2[:, 0])
            ys = np.interp(ss, cum, pts2[:, 1])
            pts = np.stack([xs, ys, np.full_like(xs, z)], axis=1)
            handles.append(scene.add_spline_catmull_rom(
                f"/bench/path/dash_{i}", pts, color=(8, 8, 8), line_width=4.5))
            i += 1
            t = t1 + gap
        return handles

    # ------------------------------------------------------------------ run lifecycle
    def begin_run(self, path, policy, label: str):
        model = policy.env.model
        if self.renderer is None or self._model_id != id(model):
            if self.renderer is not None:
                self.renderer.remove()
            self.renderer = self._Renderer(self.server, model)
            self._model_id = id(model)
        self.renderer.update(policy.env.data)
        self.trail.clear()
        if self._trail_handle is not None:
            self._trail_handle.remove()
            self._trail_handle = None
        for h in self.hist.values():
            h.clear()
        for axis, plot in (("vx", self.plot_v), ("vy", self.plot_y), ("wz", self.plot_w), ("横向误差 / 抓握力", self.plot_e)):
            plot.figure = self._figure(axis, [])
        self._steps = 0
        self._last_plot = 0.0
        self._next_tick = time.perf_counter()
        self.title.content = f"### 推车控制对比实验\n{label} · 策略 **{policy.name}**"
        scene = self.server.scene
        self._clear_path()
        if path is not None:
            self._path_handles = self._draw_dashed_path(scene, path.xy)
            gx, gy, _ = path.goal
            self._goal_handle = scene.add_icosphere("/bench/goal", radius=0.12, color=(230, 60, 60), position=(gx, gy, 0.12))
            self._la_handle = scene.add_icosphere("/bench/lookahead", radius=0.06, color=(60, 120, 255), position=(gx, gy, 0.06))
            self._goal = (gx, gy)
        else:
            for h in (self._goal_handle, self._la_handle):
                if h is not None:
                    h.visible = False
            self._goal = None

    def on_step(self, policy, follower, cmd, trace):
        self._steps += 1
        x, y, yaw = policy.pose()
        vel = policy.velocity()
        tel = trace.telemetry[-1] if trace.telemetry else {}
        h = self.hist
        h["t"].append(trace.t[-1]); h["cmd_vx"].append(cmd[0]); h["vx"].append(vel[0]); h["cmd_wz"].append(cmd[2]); h["wz"].append(vel[2])
        h["cmd_vy"].append(cmd[1]); h["vy"].append(vel[1])
        for axis in ("vx", "vy", "wz"):
            h["est_" + axis].append(tel.get("cart_est_" + axis, float("nan")))
        h["ct"].append(follower.state.cross_track if follower is not None else 0.0)
        h["fl"].append(tel.get("grasp_left", float("nan"))); h["fr"].append(tel.get("grasp_right", float("nan")))
        every = max(1, int(round((1 / policy.dt) / self.render_hz)))
        if self._steps % every == 0:
            self.renderer.update(policy.env.data)
            point = np.array([x, y, 0.03])
            # Repeated near-identical positions can produce degenerate browser
            # spline geometry while the robot is stationary. This affects only
            # the displayed trail, never recorded poses or benchmark metrics.
            new_point = not self.trail or np.linalg.norm(point - self.trail[-1]) >= 1e-4
            if new_point:
                self.trail.append(tuple(point))
            if new_point and len(self.trail) >= 2:
                self._trail_handle = self.server.scene.add_point_cloud(
                    "/bench/trail", np.asarray(self.trail, dtype=np.float32)[-1500:],
                    colors=(40, 200, 90), point_size=.018)
            if follower is not None and self._la_handle is not None:
                lx, ly = follower.state.lookahead_xy
                self._la_handle.position = (lx, ly, 0.06)
            if self.follow.value:
                target = np.array([x, y, 0.6])
                for client in self.server.get_clients().values():
                    with client.atomic():
                        client.camera.look_at = target
                        client.camera.position = target + self.camera_offset
        now = time.perf_counter()
        if now - self._last_plot >= 0.3:
            self._last_plot = now
            self._refresh(policy, follower, cmd, vel, tel, trace)
        # pacing
        if self.speed.value > 0.05:
            self._next_tick += policy.dt / self.speed.value
            sleep = self._next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:
                self._next_tick = time.perf_counter()

    def _refresh(self, policy, follower, cmd, vel, tel, trace):
        lines = [f"- t = {trace.t[-1]:.1f} s · 命令 vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f} · 实际 vx {vel[0]:+.2f} vy {vel[1]:+.2f} wz {vel[2]:+.2f}"]
        if follower is not None:
            st = follower.state
            lines.append(f"- 进度 {st.s:.2f} / {follower.path.length:.2f} m · 横向误差 {st.cross_track:+.3f} m · 航向误差 {math.degrees(st.heading_err):+.1f}° · 距目标 {st.dist_goal:.2f} m"
                         + (" · **已到达**" if st.reached else ""))
        if "cart_est_vx" in tel:
            horizon = float(tel.get("cart_est_horizon_s", 0.))
            estimate_label = f"下一步车速预测 (+{horizon*1000:.0f} ms)" if horizon else "学习式车速估计"
            lines.append(f"- {estimate_label} vx {tel['cart_est_vx']:+.2f} vy {tel['cart_est_vy']:+.2f} wz {tel['cart_est_wz']:+.2f}")
        if "grasp_left" in tel:
            lines.append(f"- 抓握力 左 {tel['grasp_left']:.0f} / 右 {tel['grasp_right']:.0f} N · 推 {tel['push_left']:+.0f}/{tel['push_right']:+.0f} · "
                         f"侧 {tel['lat_left']:+.0f}/{tel['lat_right']:+.0f} · 竖 {tel['up_left']:+.0f}/{tel['up_right']:+.0f} N")
            hc = tel.get("homie_cmd")
            if hc is not None:
                lines.append(f"- 上层→HOMIE 命令 {hc[0]:+.2f} {hc[1]:+.2f} {hc[2]:+.2f} · 上肢修正最大 {tel.get('upper_dq_max', 0):.2f} rad · 载荷 {tel.get('payload_kg', 0):.1f} kg")
        if hasattr(self, "sl_vnom"):
            live = "保持指定速度" if self.cb_cruise.value else "弯道可减速贴线"
            lines.append(f"- 指定 vx {self.sl_vnom.value:.2f} m/s · {live} · 滑条载荷 {self.sl_payload.value:.1f} kg（改路径/策略/载荷后点「开始」）")
        if trace.failure:
            lines.append(f"- **失败: {trace.failure}**")
        self.status.content = "\n".join(lines)
        h = self.hist
        t = list(h["t"])
        if len(t) > 2:
            for axis, plot, unit in (("vx", self.plot_v, "m/s"), ("vy", self.plot_y, "m/s"), ("wz", self.plot_w, "rad/s")):
                traces = [("命令", t, h["cmd_" + axis], "dash"), ("实际", t, h[axis], "solid")]
                if np.isfinite(list(h["est_" + axis])).any():
                    horizon = float(tel.get("cart_est_horizon_s", 0.))
                    label = f"预测 (+{horizon*1000:.0f} ms)" if horizon else "估计"
                    traces.append((label, [v+horizon for v in t], h["est_" + axis], "dot"))
                plot.figure = self._figure(f"{axis} [{unit}]: 命令 / 实际 / 估计", traces)
            traces = [("横向误差 [m]", t, h["ct"], "solid")]
            if np.isfinite(list(h["fl"])).any():
                traces += [("|F| 左 /100 N", t, [v / 100 for v in h["fl"]], "dot"), ("|F| 右 /100 N", t, [v / 100 for v in h["fr"]], "dot")]
            self.plot_e.figure = self._figure("横向误差 / 抓握力", traces)

    def end_run(self, result: dict):
        if self.want_rerun:
            return  # User restart is not a completed run or a timeout.
        m = result["metrics"]
        if result["kind"] == "path":
            row = (f"| {result['label']} | {result['policy']['name']} | {'✓' if m.get('success') else (m.get('failure') or 'timeout')} | "
                   f"{m.get('time_to_goal') if m.get('time_to_goal') is not None else '-'} | {m.get('final_pos_err', float('nan')):.3f} | "
                   f"{m.get('cross_track_rms', float('nan')):.3f} | {m.get('cross_track_max', float('nan')):.3f} |")
            header = ["| 运行 | 策略 | 结果 | 到点 s | 终点误差 m | 横向 RMS m | 横向 max m |", "|---|---|---|---|---|---|---|"]
        else:
            if result.get("step", {}).get("stopping"):
                stop = "已确认" if m.get("stopping_confirmed") else "未确认"
            else:
                stop = "-"
            row = (f"| {result['label']} | {result['policy']['name']} | {m.get('signal')} | {m.get('rise_time')} | {m.get('settling_time')} | "
                   f"{m.get('overshoot')} | {m.get('ss_error')} | {stop} | {m.get('failure') or '-'} |")
            header = ["| 运行 | 策略 | 信号 | 上升 s | 整定 s | 超调 | 稳态误差 | 最终停稳 | 失败 |", "|---|---|---|---|---|---|---|---|---|"]
        self.results_rows.append(row)
        self.results_md.content = "\n".join(header + self.results_rows[-30:])
