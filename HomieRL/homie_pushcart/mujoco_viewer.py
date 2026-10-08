"""Two-stage viewer: a trained high-level checkpoint on top of the frozen HOMIE, MuJoCo physics, viser rendering.

Made for looking at *training results* rather than for teleoperation (that is ``mj teleop``):

* picks the newest ``model_*.pt`` across the legacy and V6 training runs unless told otherwise,
  and lets you switch runs and checkpoints from the browser;
* runs the fixed evaluation scenarios of ``evaluate.SCENARIOS`` (hold, push, turn, stop, 25 kg, reversed
  casters, ...) one by one or in a loop, and reports the same RMSE / grasp-force numbers as ``mj eval``;
* a switch turns the high-level policy off (pure HOMIE passthrough) so the correction the policy learned is
  visible by direct comparison;
* live plots of task command vs. HOMIE command vs. cart velocity, grasp forces split into push / lateral /
  vertical components in the cart frame, and the upper-body correction per joint;
* gamepad or slider task commands for free driving.

CPU only (MuJoCo + torch inference), no Isaac Sim. Open http://localhost:<port> in a browser.
"""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from .evaluate import SCENARIOS
from .gamepad import BUTTON_A, BUTTON_B, XInputGamepad
from .mujoco_teleop import MujocoViserRenderer, _yaw_deg
from .paths import ROOT
from .task_config import run_config_path, task_cfg_from_run
from .teleop import CommandMapper

HL_LOG_DIR = ROOT / "logs" / "rsl_rl" / "homie_pushcart"
VIEWER_LOG_DIRS = (HL_LOG_DIR, ROOT / "logs" / "rsl_rl" / "homie_pushcart_v6", ROOT / "logs" / "rsl_rl" / "homie_pushcart_v7",
                   ROOT / "logs" / "rsl_rl" / "homie_pushcart_v8")
UPPER_NAMES = ["waist_yaw", "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wr_roll", "L_wr_pitch", "L_wr_yaw",
               "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wr_roll", "R_wr_pitch", "R_wr_yaw"]
MANUAL = "手动 (手柄 / 滑条)"


# ---------------------------------------------------------------------------- checkpoint discovery

def _iteration(path: Path) -> int:
    m = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(m.group(1)) if m else -1


def list_checkpoints(run_dir: Path) -> list[Path]:
    """``model_*.pt`` of one run, oldest first."""
    return sorted((p for p in Path(run_dir).glob("model_*.pt") if p.is_file() and _iteration(p) >= 0), key=_iteration)


def find_hl_checkpoint(checkpoint: Path | None = None, run: Path | None = None, log_dir: Path = HL_LOG_DIR) -> Path:
    """Legacy discovery, also used by benchmarks; its default must stay in the legacy root."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"high-level checkpoint not found: {path}")
        return path
    if run is not None:
        run_dir = Path(run)
        if not run_dir.is_dir():
            run_dir = Path(log_dir) / run
        ckpts = list_checkpoints(run_dir)
        if not ckpts:
            raise FileNotFoundError(f"no model_*.pt in {run_dir}")
        return ckpts[-1]
    runs = sorted((d for d in Path(log_dir).iterdir() if d.is_dir() and list_checkpoints(d)), key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"no training run with checkpoints under {log_dir}")
    return list_checkpoints(runs[-1])[-1]


def list_viewer_runs(log_dirs=VIEWER_LOG_DIRS) -> list[Path]:
    """All saved training runs visible to the viewer, sorted by run timestamp/name."""
    runs = {p.resolve() for root in log_dirs if Path(root).is_dir()
            for p in Path(root).iterdir() if p.is_dir() and list_checkpoints(p)}
    return sorted(runs, key=lambda p: (p.name, str(p.parent)))


def find_viewer_checkpoint(checkpoint: Path | None = None, run: Path | None = None,
                           log_dirs=VIEWER_LOG_DIRS) -> Path:
    """Viewer discovery spans legacy and V6 without changing legacy benchmark defaults."""
    if checkpoint is not None:
        return find_hl_checkpoint(checkpoint=checkpoint).resolve()
    if run is not None:
        path = Path(run)
        candidates = [path.resolve()] if path.is_dir() else [
            (Path(root) / path).resolve() for root in log_dirs if (Path(root) / path).is_dir()]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous training run {run}; use its full directory path")
        if not candidates or not (checkpoints := list_checkpoints(candidates[0])):
            raise FileNotFoundError(f"no model_*.pt for training run {run}")
        return checkpoints[-1]
    runs = list_viewer_runs(log_dirs)
    if not runs:
        raise FileNotFoundError(f"no training checkpoints under {', '.join(map(str, log_dirs))}")
    return list_checkpoints(runs[-1])[-1]


def checkpoint_family(checkpoint_or_run: Path) -> str:
    """Read only task.version, not V6's nested baseline_source.version (V3.2).

    Composing YAML avoids constructing Python objects embedded in historical
    Isaac configuration files. Actual policy loaders retain their own checks.
    """
    import yaml
    path = Path(checkpoint_or_run)
    source = run_config_path(path)
    if source is None:
        if "homie_pushcart_v8" in path.parts:
            return "v8"
        return "v7" if "homie_pushcart_v7" in path.parts else ("v6" if "homie_pushcart_v6" in path.parts else "legacy")
    root = yaml.compose(source.read_text(encoding="utf-8"))

    def field(node, key):
        if isinstance(node, yaml.MappingNode):
            return next((value for name, value in node.value if name.value == key), None)
        return None

    release_node = field(field(root, "task"), "release")             # V8 subclasses the V7.1 task (version 7)
    if isinstance(release_node, yaml.ScalarNode) and str(release_node.value).startswith("8"):
        return "v8"
    version_node = field(field(root, "task"), "version")
    version = version_node.value.lower() if isinstance(version_node, yaml.ScalarNode) else None
    if version in ("6", "v6"):
        return "v6"
    if version in ("7", "v7"):
        return "v7"
    if version is None or version in ("1", "2", "3") or version.startswith(("v1", "v2", "v3")):
        return "legacy"
    raise ValueError(f"unsupported saved task.version={version!r}: {source}")


def make_viewer_runner(checkpoint: Path, level: int | None = None, homie_checkpoint: Path | None = None):
    """Use each checkpoint family's existing control and observation implementation."""
    checkpoint = Path(checkpoint)
    family = checkpoint_family(checkpoint)
    if family == "v6":
        from homie_pushcart_v6.task_config import task_cfg_from_run as v6_task_cfg
        from homie_pushcart_v6.mujoco_runner import MujocoV6Runner
        tc = v6_task_cfg(checkpoint, verbose=False)
    elif family == "v8":                         # V8 teacher on the V7 plant (fine-tuned HOMIE velocity, leg residual)
        from homie_pushcart_v8.task_config import task_cfg_from_run as v8_task_cfg
        from homie_pushcart_v8.mujoco_runner import MujocoV8Runner
        tc = v8_task_cfg(checkpoint, verbose=False)
    elif family == "v7":                         # V7 plant (its own MJCF and grasp chain)
        from homie_pushcart_v7.task_config import task_cfg_from_run as v7_task_cfg
        from homie_pushcart_v7.mujoco_runner import MujocoV7Runner
        tc = v7_task_cfg(checkpoint, verbose=False)
    else:
        from .mujoco_hl import make_runner
        tc = task_cfg_from_run(checkpoint, verbose=False)
    level = tc.curriculum.num_levels - 1 if level is None else int(level)
    if not 0 <= level < tc.curriculum.num_levels:
        raise ValueError(f"level {level} is outside 0..{tc.curriculum.num_levels - 1}")
    if family == "v6":
        runner = MujocoV6Runner(checkpoint, level=level, homie_checkpoint=homie_checkpoint)
        return runner, runner.homie_checkpoint, family
    if family == "v7":
        runner = MujocoV7Runner(checkpoint, level=level, homie_checkpoint=homie_checkpoint)
        return runner, runner.homie_checkpoint, family
    if family == "v8":
        runner = MujocoV8Runner(checkpoint, level=level, homie_checkpoint=homie_checkpoint)
        return runner, runner.homie_checkpoint, family
    args = SimpleNamespace(checkpoint=checkpoint, level=level, homie_checkpoint=homie_checkpoint)
    runner, homie = make_runner(args)
    return runner, homie, family


# ---------------------------------------------------------------------------- viewer

class TwoStageViewer:
    def __init__(self, args):
        import torch  # noqa: F401  (inference below)
        import viser

        self.args = args
        self._requests = SimpleQueue()
        self._syncing_gui = False
        self.hl_checkpoint = find_viewer_checkpoint(getattr(args, "checkpoint", None), getattr(args, "run", None))
        self.run_dir = self.hl_checkpoint.parent
        self.runner, self.homie_checkpoint, self.family = make_viewer_runner(
            self.hl_checkpoint, getattr(args, "level", None), getattr(args, "homie_checkpoint", None))
        self.tc, self.level = self.runner.task_cfg, self.runner.level
        self.env = self.runner.env
        self.policy = self.runner.hl                       # includes V6's independent-history bridge
        self.dt = self.env.dt

        # task command ranges (75 % of HOMIE's) for the gamepad / sliders
        self._reset_command_mapper()
        self.pad = XInputGamepad()

        self.server = viser.ViserServer(port=args.port, label="HOMIE push-cart: two-stage viewer")
        self.renderer = MujocoViserRenderer(self.server, self.env.model)
        m = self.env.model
        self.wrist_ids = {side: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link") for side in ("left", "right")}
        # viser mesh handles cannot be recoloured in place: one sphere per colour bin, toggled by visibility
        self.force_bins = [(int(60 + 195 * k), int(200 - 170 * k), 60) for k in np.linspace(0., 1., 6)]
        self.force_markers = {side: [self.server.scene.add_icosphere(f"/force/{side}/{i}", radius=0.035, color=rgb, visible=(i == 0))
                                     for i, rgb in enumerate(self.force_bins)] for side in self.wrist_ids}

        # scenario state
        self.scenario = None                 # name or None (manual)
        self.scenario_t = 0.0
        self.scenario_cmd = np.zeros(3)
        self.scenario_stats = {}             # name -> dict (last completed run)
        self._acc = None                     # accumulators of the running scenario
        self.auto_queue: list[str] = []

        # telemetry
        self.hist_len = int(10.0 / self.dt)
        self.hist = {k: deque(maxlen=self.hist_len) for k in
                     ("t", "task_vx", "homie_vx", "cart_vx", "task_wz", "homie_wz", "cart_wz", "fL", "fR", "pushL", "pushR", "latL", "latR", "upL", "upR")}
        self.counts = {"fall": 0, "cart_yaw": 0, "cart_distance": 0, "cart_tilt": 0, "resets": 0}
        self.sim_time = 0.0
        self.flags = {"reset": False, "zero": False, "run_scenario": False}

        self._build_gui()
        self.runner.set_payload(float(getattr(args, "payload", 0.0) or 0.0))
        self.obs = self.runner.reset()
        self.renderer.update(self.env.data)

    def _reset_command_mapper(self):
        cc = self.tc.commands
        self.ranges = SimpleNamespace(lin_vel_x=list(cc.vx_range_max), lin_vel_y=[-cc.vy_max, cc.vy_max],
                                      ang_vel_yaw=[-cc.wz_max, cc.wz_max])
        self.mapper = CommandMapper(self.ranges, invert_lateral=not self.args.no_invert,
                                    invert_yaw=not self.args.no_invert,
                                    smoothing_tau_s=self.args.smoothing, dt=self.dt)

    @staticmethod
    def _family_label(family):
        return {"v6": "V6", "v7": "V7", "v8": "V8 教师"}.get(family, "V3 / 更早版本")

    def _run_choices(self):
        runs = set(list_viewer_runs()) | {self.run_dir.resolve()}
        choices = {}
        for run in sorted(runs, key=lambda p: (p.name, str(p.parent)), reverse=True):
            try:
                family = self._family_label(checkpoint_family(run))
            except Exception:  # An invalid saved run must not hide the other choices.
                family = "配置待检查"
            label = f"{family} · {run.name}"
            if label in choices:
                label += f" ({run.parent})"
            choices[label] = run
        return choices

    def _run_label(self, run):
        return next(label for label, path in self.run_choices.items() if path == Path(run).resolve())

    def _refresh_header(self):
        self.md_header.content = (
            f"### 两级策略查看器 · MuJoCo\n"
            f"**{self._family_label(self.family)}** · 运行 `{self.run_dir.name}`\n\n"
            f"HOMIE `{self.homie_checkpoint.name}` (iter {self.runner.lowlevel.iteration}) · 控制 {1 / self.dt:.0f} Hz")

    # ------------------------------------------------------------------ GUI
    def _build_gui(self):
        gui = self.server.gui
        args = self.args
        self.md_header = gui.add_markdown("")
        self._refresh_header()
        self.status = gui.add_markdown("")

        with gui.add_folder("策略"):
            self.run_choices = self._run_choices()
            self.dd_run = gui.add_dropdown("训练运行", tuple(self.run_choices), initial_value=self._run_label(self.run_dir))
            ckpts = list_checkpoints(self.run_dir)
            names = [p.name for p in ckpts] or [self.hl_checkpoint.name]
            self.dd_ckpt = gui.add_dropdown("检查点", tuple(names), initial_value=self.hl_checkpoint.name)
            self.cb_hl = gui.add_checkbox("上层策略开启 (关 = 纯 HOMIE 直通)", True)
            self.sl_level = gui.add_slider("课程级别 (臂/腰修正尺度)", 0, self.tc.curriculum.num_levels - 1, 1, self.level)
            self.md_policy = gui.add_markdown("")
            self.md_load = gui.add_markdown("切换训练运行或检查点后自动加载并复位。")
        with gui.add_folder("场景"):
            self.dd_scn = gui.add_dropdown("场景", (MANUAL, *SCENARIOS.keys()),
                                           initial_value=args.scenario if getattr(args, "scenario", None) in SCENARIOS else MANUAL)
            self.btn_run = gui.add_button("运行所选场景", color="blue")
            self.cb_auto = gui.add_checkbox("自动巡演全部场景", bool(getattr(args, "auto", False)))
            self.sl_payload = gui.add_slider("手动模式载荷 [kg]", 0.0, 50.0, 0.5, float(getattr(args, "payload", 0.0) or 0.0))
            self.sl_caster = gui.add_slider("手动模式万向轮初始角 [deg]", -180.0, 180.0, 5.0, 0.0)
            self.btn_reset = gui.add_button("复位", color="green")
            self.md_scn = gui.add_markdown("")
        with gui.add_folder("手动命令 (车坐标系任务命令)"):
            self.md_pad = gui.add_markdown("XInput: 未连接" if not self.pad.available else "XInput: 等待手柄…")
            self.cb_sliders = gui.add_checkbox("用滑条代替手柄", not self.pad.available)
            r = self.ranges
            self.s_vx = gui.add_slider("vx [m/s]", float(r.lin_vel_x[0]), float(r.lin_vel_x[1]), 0.05, 0.0)
            self.s_vy = gui.add_slider("vy [m/s]", float(r.lin_vel_y[0]), float(r.lin_vel_y[1]), 0.05, 0.0)
            self.s_wz = gui.add_slider("wz [rad/s]", float(r.ang_vel_yaw[0]), float(r.ang_vel_yaw[1]), 0.05, 0.0)
            self.btn_zero = gui.add_button("命令归零")
        with gui.add_folder("显示"):
            self.cb_follow = gui.add_checkbox("镜头跟随", True)
            self.cb_col = gui.add_checkbox("显示机器人碰撞体", False)
            self.cb_markers = gui.add_checkbox("显示抓握力标记 (绿→红 = 0→100 N)", True)
            self.sl_rt = gui.add_slider("速度倍率 (1 = 实时)", 0.1, 3.0, 0.1, 1.0)
        with gui.add_folder("曲线 (最近 10 s)", expand_by_default=True):
            self.plot_v = gui.add_plotly(self._figure("vx [m/s]", []), aspect=2.2)
            self.plot_w = gui.add_plotly(self._figure("wz [rad/s]", []), aspect=2.2)
            self.plot_f = gui.add_plotly(self._figure("抓握力 [N]", []), aspect=2.2)
        with gui.add_folder("上肢修正 [rad]", expand_by_default=False):
            self.md_upper = gui.add_markdown("")
        with gui.add_folder("场景统计 (与 mj eval 同定义)", expand_by_default=False):
            self.md_stats = gui.add_markdown("尚无完成的场景。")

        self.btn_reset.on_click(lambda _: self.flags.__setitem__("reset", True))
        self.btn_zero.on_click(lambda _: self.flags.__setitem__("zero", True))
        self.btn_run.on_click(lambda _: self.flags.__setitem__("run_scenario", True))
        self.dd_run.on_update(lambda _: self._queue_request("run", self.run_choices[self.dd_run.value]))
        self.dd_ckpt.on_update(lambda _: self._queue_request(
            "checkpoint", self.run_choices[self.dd_run.value] / self.dd_ckpt.value))
        self.cb_col.on_update(lambda _: self._queue_request("collision", self.cb_col.value))
        self.cb_hl.on_update(lambda _: self._queue_request("hl", self.cb_hl.value))
        self.sl_level.on_update(lambda _: self._queue_request("level", int(self.sl_level.value)))
        self.cb_markers.on_update(lambda _: self._queue_request("markers", self.cb_markers.value))
        self._refresh_policy_md()

        camera_offset = np.array([-2.0, -2.6, 1.6])
        self.camera_offset = camera_offset

        @self.server.on_client_connect
        def _(client):
            import threading

            def later():
                time.sleep(1.0)
                with client.atomic():
                    client.camera.position = np.array([0.5, 0., 0.7]) + camera_offset
                    client.camera.look_at = np.array([0.5, 0., 0.7])

            threading.Thread(target=later, daemon=True).start()

    @staticmethod
    def _figure(title: str, traces):
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, x, y, dash in traces:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(dash=dash, width=1.6)))
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.97), margin=dict(l=40, r=10, t=28, b=28),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)), xaxis=dict(title=None), yaxis=dict(title=None),
                          template="plotly_white", height=230)
        return fig

    def _set_markers(self, forces: dict | None):
        """Move the palm force markers and show the colour bin of the current magnitude (None hides all)."""
        with self.server.atomic():
            for side, spheres in self.force_markers.items():
                if forces is None:
                    for h in spheres:
                        h.visible = False
                    continue
                pos = self.env.data.xpos[self.wrist_ids[side]].copy()
                k = min(len(spheres) - 1, int(min(1.0, forces.get(side, 0.) / 100.0) * (len(spheres) - 1) + 0.5))
                for i, h in enumerate(spheres):
                    h.visible = (i == k)
                    if i == k:
                        h.position = pos

    def _refresh_policy_md(self):
        it = self.policy.iteration if self.policy is not None else "-"
        self.md_policy.content = (f"**{self._family_label(self.family)}** · 检查点 `{self.hl_checkpoint.name}` · 训练迭代 **{it}** · 上层 **{'开' if self.runner.hl is not None else '关 (直通)'}** · "
                                  f"level {self.runner.level}: 臂 {self.runner.lp.arm_scale:.2f} / 腰 {self.runner.lp.waist_scale:.2f} rad")

    def _apply_hl_switch(self):
        self.runner.hl = self.policy if self.cb_hl.value else None
        self.runner.hl_actions[:] = 0.
        self._refresh_policy_md()

    def _queue_request(self, kind, value):
        # Viser callbacks may run concurrently with simulation. They never touch
        # runners, policies, or MuJoCo state; the simulation thread consumes this.
        if not self._syncing_gui:
            self._requests.put((kind, value))

    def _sync_model_selectors(self):
        self._syncing_gui = True
        try:
            with self.server.atomic():
                self.run_choices = self._run_choices()
                self.dd_run.options = tuple(self.run_choices)
                self.dd_run.value = self._run_label(self.run_dir)
                names = [p.name for p in list_checkpoints(self.run_dir)]
                if self.hl_checkpoint.name not in names:
                    names.append(self.hl_checkpoint.name)
                self.dd_ckpt.options = tuple(names)
                self.dd_ckpt.value = self.hl_checkpoint.name
                self.sl_level.max = self.tc.curriculum.num_levels - 1
                self.sl_level.value = self.runner.level
        finally:
            self._syncing_gui = False

    def _show_load_error(self, path, exc):
        self.md_load.content = f"**加载失败，继续运行原模型。** `{Path(path).name}`\n\n{type(exc).__name__}: {exc}"
        self._sync_model_selectors()
        print(f"[mj-view] could not load {path}: {type(exc).__name__}: {exc}", flush=True)

    def _process_requests(self) -> bool:
        """Apply queued GUI changes between physics steps; return whether the model changed."""
        checkpoint = None
        updates = {}
        while True:
            try:
                kind, value = self._requests.get_nowait()
            except Empty:
                break
            if kind == "run":
                try:
                    checkpoint = find_viewer_checkpoint(run=value)
                except Exception as exc:
                    self._show_load_error(value, exc)
            elif kind == "checkpoint":
                checkpoint = value
            else:
                updates[kind] = value
        changed = self._load_checkpoint(checkpoint) if checkpoint is not None else False
        if "level" in updates:
            self.level = max(0, min(updates["level"], self.tc.curriculum.num_levels - 1))
            self.runner.set_level(self.level)
            self._syncing_gui = True
            try:
                self.sl_level.value = self.level
            finally:
                self._syncing_gui = False
            self._refresh_policy_md()
        if "hl" in updates:
            self._apply_hl_switch()
        if "collision" in updates:
            self.renderer.set_collision_visible(updates["collision"])
        if "markers" in updates and not updates["markers"]:
            self._set_markers(None)
        return changed

    def _load_checkpoint(self, checkpoint: Path | str) -> bool:
        """Build and reset a complete candidate before replacing the running model.

        Even two checkpoints in one run reload saved configuration and all
        observation histories. The V6 policy bridge stays bound to its own runner.
        """
        path = Path(checkpoint)
        path = path if path.is_absolute() else self.run_dir / path
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            runner, homie, family = make_viewer_runner(
                path, int(self.sl_level.value), getattr(self.args, "homie_checkpoint", None))
            policy = runner.hl
            runner.hl = policy if self.cb_hl.value else None
            runner.set_payload(float(self.sl_payload.value))
            obs = runner.reset(caster_angle=math.radians(float(self.sl_caster.value)))
        except Exception as exc:
            self._show_load_error(path, exc)
            return False

        # Model/configuration errors above leave the old runner and scene intact.
        # Renderer replacement changes only visualization, never the model geoms.
        self.renderer.remove()
        try:
            renderer = MujocoViserRenderer(self.server, runner.env.model)
            renderer.set_collision_visible(self.cb_col.value)
            renderer.update(runner.env.data)
        except Exception as exc:
            self.renderer = MujocoViserRenderer(self.server, self.env.model)
            self.renderer.set_collision_visible(self.cb_col.value)
            self.renderer.update(self.env.data)
            self._show_load_error(path, exc)
            return False
        self.runner, self.env, self.policy, self.obs = runner, runner.env, policy, obs
        self.homie_checkpoint, self.family, self.renderer = homie, family, renderer
        self.hl_checkpoint, self.run_dir = path.resolve(), path.resolve().parent
        self.tc, self.level, self.dt = runner.task_cfg, runner.level, runner.env.dt
        self.wrist_ids = {side: mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
                          for side in ("left", "right")}
        self._reset_command_mapper()
        self._sync_model_selectors()
        for slider, bounds in ((self.s_vx, self.ranges.lin_vel_x), (self.s_vy, self.ranges.lin_vel_y),
                               (self.s_wz, self.ranges.ang_vel_yaw)):
            slider.min, slider.max, slider.value = float(bounds[0]), float(bounds[1]), 0.0
        self.scenario, self._acc, self.scenario_t = None, None, 0.0
        self.scenario_cmd[:] = 0.0
        self.auto_queue.clear()
        self.scenario_stats.clear()
        self.dd_scn.value, self.cb_auto.value = MANUAL, False
        self.flags.update(reset=False, zero=False, run_scenario=False)
        self.hist_len = int(10.0 / self.dt)
        self.hist = {key: deque(maxlen=self.hist_len) for key in self.hist}
        self.counts = {key: 0 for key in self.counts}
        self.counts["resets"] = 1
        self.sim_time = 0.0
        self.status.content = "模型已切换，命令归零。"
        self.md_scn.content = "已复位到手动模式。"
        self.md_stats.content = "尚无完成的场景。"
        self.md_upper.content = ""
        self.plot_v.figure = self._figure("vx [m/s]", [])
        self.plot_w.figure = self._figure("wz [rad/s]", [])
        self.plot_f.figure = self._figure("抓握力 [N]", [])
        self._set_markers(runner.ts["grasp"] if self.cb_markers.value else None)
        self._refresh_header()
        self._refresh_policy_md()
        self.md_load.content = f"已加载 `{self.run_dir.name}/{path.name}`，已复位。"
        print(f"[mj-view] loaded {path} ({family}, iter {self.policy.iteration})", flush=True)
        return True

    # ------------------------------------------------------------------ scenarios
    def _start_scenario(self, name: str):
        cmd, payload, caster, seconds, switch = SCENARIOS[name]
        self.scenario = name
        self.scenario_t = 0.0
        self.scenario_cmd = np.array(cmd, dtype=float)
        self.runner.set_payload(payload)
        self.obs = self.runner.reset(caster_angle=caster)
        self.counts["resets"] += 1
        self._acc = {"vx_err": [], "wz_err": [], "dist": [], "grasp": [], "failures": {}, "switched": False}
        caster_deg = math.degrees(caster or 0.0)
        self.md_scn.content = f"运行中: **{name}** · 命令 {cmd} · 载荷 {payload} kg · 万向轮 {caster_deg:.0f}° · {seconds:.0f} s"
        print(f"[mj-view] scenario {name}: cmd={cmd} payload={payload} kg caster={caster_deg:.0f} deg", flush=True)

    def _finish_scenario(self):
        name, acc = self.scenario, self._acc
        vx_err, wz_err = np.array(acc["vx_err"]), np.array(acc["wz_err"])
        self.scenario_stats[name] = {
            "vx_rmse": float(np.sqrt(np.mean(vx_err ** 2))) if len(vx_err) else float("nan"),
            "vx_bias": float(np.mean(vx_err)) if len(vx_err) else float("nan"),
            "wz_rmse": float(np.sqrt(np.mean(wz_err ** 2))) if len(wz_err) else float("nan"),
            "dist": float(np.mean(acc["dist"])) if acc["dist"] else float("nan"),
            "grasp_p95": float(np.percentile(acc["grasp"], 95)) if acc["grasp"] else float("nan"),
            "failures": dict(acc["failures"]), "hl": self.runner.hl is not None,
            "ckpt": self.hl_checkpoint.name,
        }
        s = self.scenario_stats[name]
        print(f"[mj-view] {name:15s} vx rmse {s['vx_rmse']:.3f} (bias {s['vx_bias']:+.3f}) wz rmse {s['wz_rmse']:.3f} "
              f"| dist {s['dist']:.2f} | grasp p95 {s['grasp_p95']:.0f} N | failures {s['failures']} | hl={'on' if s['hl'] else 'off'}", flush=True)
        rows = ["| 场景 | 上层 | 检查点 | vx RMSE | vx 偏差 | wz RMSE | 距离 | 力 p95 | 失败 |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in self.scenario_stats.items():
            rows.append(f"| {k} | {'开' if v['hl'] else '关'} | {v['ckpt']} | {v['vx_rmse']:.3f} | {v['vx_bias']:+.3f} | {v['wz_rmse']:.3f} | "
                        f"{v['dist']:.2f} | {v['grasp_p95']:.0f} N | {v['failures'] or '-'} |")
        self.md_stats.content = "\n".join(rows)
        self.scenario, self._acc = None, None
        self.scenario_cmd[:] = 0.
        self.md_scn.content = f"完成: **{name}**"
        if self.auto_queue:
            self._start_scenario(self.auto_queue.pop(0))
        elif self.cb_auto.value:
            self.auto_queue = list(SCENARIOS.keys())
            self._start_scenario(self.auto_queue.pop(0))

    def _scenario_command(self) -> np.ndarray:
        cmd, payload, caster, seconds, switch = SCENARIOS[self.scenario]
        if switch is not None and self.scenario_t >= switch[0] and not self._acc["switched"]:
            self.scenario_cmd = np.array(switch[1], dtype=float)
            self._acc["switched"] = True
        return self.scenario_cmd

    def _record_scenario(self, failure: str | None):
        cmd, payload, caster, seconds, switch = SCENARIOS[self.scenario]
        acc, ts = self._acc, self.runner.ts
        if failure:
            acc["failures"][failure] = acc["failures"].get(failure, 0) + 1
            return
        transient = float(self.args.transient_s)
        in_transient = self.scenario_t < transient or (switch is not None and abs(self.scenario_t - switch[0]) < transient)
        if in_transient:
            return
        acc["vx_err"].append(self.scenario_cmd[0] - ts["cart_v"][0])
        acc["wz_err"].append(self.scenario_cmd[2] - ts["cart_wz"])
        acc["dist"].append(ts["dist"])
        acc["grasp"].append(max(ts["grasp"].values(), default=0.0))

    # ------------------------------------------------------------------ main loop
    def run(self):
        import torch
        args = self.args
        print("=" * 78)
        print(f"[mj-view] run {self.run_dir}")
        print(f"[mj-view] high-level {self.hl_checkpoint.name} (iter {self.policy.iteration})  HOMIE {self.homie_checkpoint.name}  level {self.level}")
        print(f"[mj-view] viser: http://localhost:{self.server.get_port()}   XInput available: {self.pad.available}")
        print("[mj-view] scenarios: " + ", ".join(SCENARIOS.keys()))
        print("=" * 78, flush=True)
        if self.dd_scn.value != MANUAL:
            self._start_scenario(self.dd_scn.value)
        elif self.cb_auto.value:
            self.auto_queue = list(SCENARIOS.keys())
            self._start_scenario(self.auto_queue.pop(0))

        render_every = max(1, int(round((1 / self.dt) / args.render_hz)))
        steps, next_tick, last_status, wall_start, sim_start = 0, time.perf_counter(), 0.0, time.perf_counter(), 0.0
        with torch.inference_mode():
            while True:
                # ---- input
                if self._process_requests():
                    render_every = max(1, int(round((1 / self.dt) / args.render_hz)))
                    steps, next_tick, last_status, wall_start, sim_start = (
                        0, time.perf_counter(), 0.0, time.perf_counter(), self.sim_time)
                state = self.pad.poll()
                gamepad_active = state["connected"] and not self.cb_sliders.value
                if gamepad_active:
                    if state["pressed"] & BUTTON_A:
                        self.flags["reset"] = True
                    if state["pressed"] & BUTTON_B:
                        self.flags["zero"] = True
                if self.flags["run_scenario"]:
                    self.flags["run_scenario"] = False
                    if self.dd_scn.value != MANUAL:
                        self.auto_queue = []
                        self._start_scenario(self.dd_scn.value)
                    elif self.cb_auto.value:
                        self.auto_queue = list(SCENARIOS.keys())
                        self._start_scenario(self.auto_queue.pop(0))
                if self.flags["zero"]:
                    self.flags["zero"] = False
                    self.mapper.state = [0., 0., 0.]
                    self.s_vx.value, self.s_vy.value, self.s_wz.value = 0., 0., 0.
                if self.flags["reset"]:
                    self.flags["reset"] = False
                    if self.scenario is not None:
                        self._start_scenario(self.scenario)
                    else:
                        self.runner.set_payload(float(self.sl_payload.value))
                        self.obs = self.runner.reset(caster_angle=math.radians(float(self.sl_caster.value)))
                        self.counts["resets"] += 1

                if self.scenario is not None:
                    cmd = self._scenario_command()
                    source = f"场景 {self.scenario}"
                else:
                    if gamepad_active:
                        raw = (state["ly"], state["lx"], state["rx"])
                        source = "手柄"
                    else:
                        r = self.ranges
                        raw = (self.s_vx.value / (r.lin_vel_x[1] if self.s_vx.value >= 0 else -r.lin_vel_x[0]),
                               -self.s_vy.value / r.lin_vel_y[1], -self.s_wz.value / r.ang_vel_yaw[1])
                        source = "滑条"
                    cmd = np.array(self.mapper(raw))

                # ---- step
                self.obs, fallen = self.runner.step(cmd)
                failure = "fall" if fallen else self.runner.geometry_failure()
                steps += 1
                self.sim_time += self.dt
                if self.scenario is not None:
                    self.scenario_t += self.dt
                    self._record_scenario(failure)
                if failure:
                    self.counts[failure] = self.counts.get(failure, 0) + 1
                    print(f"[mj-view] {failure} -> reset", flush=True)
                    if self.scenario is not None:
                        cmd_s, payload, caster, _, _ = SCENARIOS[self.scenario]
                        self.obs = self.runner.reset(caster_angle=caster)
                    else:
                        self.obs = self.runner.reset(caster_angle=math.radians(float(self.sl_caster.value)))
                    self.counts["resets"] += 1
                if self.scenario is not None and self.scenario_t >= SCENARIOS[self.scenario][3]:
                    self._finish_scenario()

                # ---- telemetry buffers
                ts = self.runner.ts
                fvec = self.env.grasp_force_vectors()
                yaw = math.radians(_yaw_deg(self.env.cart_state()[1]))
                c, s = math.cos(yaw), math.sin(yaw)
                comp = {}
                for side in ("left", "right"):
                    f = fvec.get(side, np.zeros(3))
                    comp[side] = (c * f[0] + s * f[1], -s * f[0] + c * f[1], f[2])        # push, lateral, up (cart frame)
                h = self.hist
                h["t"].append(self.sim_time)
                h["task_vx"].append(cmd[0]); h["homie_vx"].append(self.runner.cmd_homie[0]); h["cart_vx"].append(ts["cart_v"][0])
                h["task_wz"].append(cmd[2]); h["homie_wz"].append(self.runner.cmd_homie[2]); h["cart_wz"].append(ts["cart_wz"])
                h["fL"].append(ts["grasp"].get("left", 0.)); h["fR"].append(ts["grasp"].get("right", 0.))
                h["pushL"].append(comp["left"][0]); h["pushR"].append(comp["right"][0])
                h["latL"].append(comp["left"][1]); h["latR"].append(comp["right"][1])
                h["upL"].append(comp["left"][2]); h["upR"].append(comp["right"][2])

                # ---- render
                if steps % render_every == 0:
                    self.renderer.update(self.env.data)
                    if self.cb_markers.value:
                        self._set_markers(ts["grasp"])
                    if self.cb_follow.value:
                        pelvis = self.env.data.xpos[self.env.pelvis_id]
                        target = np.array([pelvis[0], pelvis[1], 0.7])
                        for client in self.server.get_clients().values():
                            with client.atomic():
                                client.camera.look_at = target
                                client.camera.position = target + self.camera_offset

                now = time.perf_counter()
                if now - last_status >= 0.25:
                    last_status = now
                    self._refresh_status(cmd, source, comp, steps, wall_start, sim_start, state)
                    self._refresh_plots()

                next_tick += self.dt / max(self.sl_rt.value, 0.05)
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -0.5:
                    next_tick = time.perf_counter()

    # ------------------------------------------------------------------ status / plots
    def _refresh_status(self, cmd, source, comp, steps, wall_start, sim_start, pad_state):
        r, ts, tc = self.runner, self.runner.ts, self.tc
        ch = r.cmd_homie
        res = ch - cmd
        bv = self.env.base_lin_vel
        lp = r.lp
        thr = np.array([tc.termination.pull_n, tc.termination.up_n, tc.termination.tangential_n]) * lp.grasp_threshold_mult \
            + tc.termination.per_kg_n * max(0., r.cart_mass_total - float(self.env.cart_params["mass"]["base_kg"]))
        pull = max(-comp["left"][0], -comp["right"][0], 0.)
        up = max(comp["left"][2], comp["right"][2], 0.)
        tang = max(abs(comp["left"][1]), abs(comp["right"][1]))
        flags = []
        if pull > thr[0]:
            flags.append(f"拉 {pull:.0f} > {thr[0]:.0f} N")
        if up > thr[1]:
            flags.append(f"提 {up:.0f} > {thr[1]:.0f} N")
        if tang > thr[2]:
            flags.append(f"切向 {tang:.0f} > {thr[2]:.0f} N")
        rtf = (self.sim_time - sim_start) / max(time.perf_counter() - wall_start, 1e-6)
        md = [
            f"- 输入: **{source}** · 上层 **{'开' if r.hl is not None else '关 (直通)'}** · 载荷 **{r.cart_mass_total - float(self.env.cart_params['mass']['base_kg']):.1f} kg** (车总重 {r.cart_mass_total:.1f})",
            f"- 任务命令 vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
            f"- 上层→HOMIE 命令 vx {ch[0]:+.2f}  vy {ch[1]:+.2f}  wz {ch[2]:+.2f}  （残差 {res[0]:+.2f} / {res[1]:+.2f} / {res[2]:+.2f}）",
            f"- 推车 (车坐标系) vx **{ts['cart_v'][0]:+.2f}**  vy {ts['cart_v'][1]:+.2f}  wz **{ts['cart_wz']:+.2f}**  → 误差 vx {cmd[0] - ts['cart_v'][0]:+.2f}  wz {cmd[2] - ts['cart_wz']:+.2f}",
            f"- 基座 vx {bv[0]:+.2f}  vy {bv[1]:+.2f}  wz {self.env.base_ang_vel[2]:+.2f}  高度 {self.env.root_height:.3f} m",
            f"- 抓握力 |F| 左 **{ts['grasp'].get('left', 0.):.0f}** / 右 **{ts['grasp'].get('right', 0.):.0f}** N · 推(x) {comp['left'][0]:+.0f}/{comp['right'][0]:+.0f} · 侧(y) {comp['left'][1]:+.0f}/{comp['right'][1]:+.0f} · 竖(z) {comp['left'][2]:+.0f}/{comp['right'][2]:+.0f}",
            f"- 过载阈值 (level {r.level}, ×{lp.grasp_threshold_mult:.1f}): 拉 {thr[0]:.0f} / 提 {thr[1]:.0f} / 切向 {thr[2]:.0f} N · " + ("**超限: " + "，".join(flags) + "**" if flags else "未超限"),
            f"- 机器人–车: 距离 {ts['dist']:.2f} m (终止 < {tc.termination.distance_m[0]} / > {tc.termination.distance_m[1]}) · 偏航差 {math.degrees(ts['rel_yaw']):+.1f}° · 车倾斜 {math.degrees(math.asin(min(1., ts['cart_tilt']))):.1f}°",
            f"- 上肢修正最大 |Δq| {np.abs(r.upper_delta_q).max():.2f} rad · 摔倒 {self.counts['fall']} · 几何终止 yaw/dist/tilt {self.counts['cart_yaw']}/{self.counts['cart_distance']}/{self.counts['cart_tilt']} · 复位 {self.counts['resets']}",
            f"- 仿真 {self.sim_time:.1f} s · 实时倍率 {rtf:.2f}×",
        ]
        self.status.content = "\n".join(md)
        self.md_pad.content = ("XInput: 已连接" if pad_state["connected"] else
                               ("XInput: 未检测到手柄 (用滑条)" if self.pad.available else "XInput: 不可用 (非 Windows)"))
        dq = r.upper_delta_q
        a = r.hl_actions
        rows = ["| 关节 | Δq [rad] | 动作 | 力矩 [N·m] |", "|---|---|---|---|"]
        for i, name in enumerate(UPPER_NAMES):
            rows.append(f"| {name} | {dq[i]:+.3f} | {a[3 + i]:+.2f} | {ts['upper_tau'][i]:+.1f} |")
        self.md_upper.content = "\n".join(rows)

    def _refresh_plots(self):
        h = self.hist
        t = list(h["t"])
        if len(t) < 2:
            return
        self.plot_v.figure = self._figure("vx [m/s]: 任务命令 / HOMIE 命令 / 推车", [
            ("任务命令", t, list(h["task_vx"]), "dash"), ("HOMIE 命令", t, list(h["homie_vx"]), "dot"), ("推车", t, list(h["cart_vx"]), "solid")])
        self.plot_w.figure = self._figure("wz [rad/s]: 任务命令 / HOMIE 命令 / 推车", [
            ("任务命令", t, list(h["task_wz"]), "dash"), ("HOMIE 命令", t, list(h["homie_wz"]), "dot"), ("推车", t, list(h["cart_wz"]), "solid")])
        self.plot_f.figure = self._figure("抓握力 [N]: |F| 与车坐标系分量", [
            ("|F| 左", t, list(h["fL"]), "solid"), ("|F| 右", t, list(h["fR"]), "solid"),
            ("推 左", t, list(h["pushL"]), "dash"), ("推 右", t, list(h["pushR"]), "dash"),
            ("侧 左", t, list(h["latL"]), "dot"), ("侧 右", t, list(h["latR"]), "dot"),
            ("竖 左", t, list(h["upL"]), "dashdot"), ("竖 右", t, list(h["upR"]), "dashdot")])


def run_two_stage_viewer(args):
    TwoStageViewer(args).run()
