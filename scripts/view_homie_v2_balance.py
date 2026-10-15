"""Teleop of HOMIE walking + the v2 balance (stand-still) model on the Isaac backend (Viser in the browser).

Same controls as the one-stage MuJoCo teleop: an Xbox gamepad through XInput (left stick up/down -> vx,
left/right -> vy, right stick left/right -> wz; A resets, B zeroes the command) or the GUI sliders. The
command is smoothed with the 0.4 s filter the switch states were recorded with. While the smoothed
|command| >= 0.05 the walking model (model_60000) walks it; below 0.05 the balance model takes over with a
zero command (GR00T's Balance/Walk switch). Untick the switch to watch the walking model's zero command.
The simulation is the v2 balance task (one environment, no randomization); the robot starts in the stage-2
grasp stance with the arms holding the grasp. Extra: random arm motion (the training curriculum), a push
button. Readouts: active policy, pelvis height against the 0.7592 m push-cart start, pitch, steps and foot
drift while balancing.

    .\\run_homie_v2_balance_viewer.ps1 -Port 8092
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
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL"), str(ROOT / "scripts")]
BALANCE_RUNS = ROOT / "logs/rsl_rl/homie_g1_v2_balance"
WALK_CKPT = ROOT / "logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt"
ROBOT_USD = ROOT / ".cache/g1_isaaclab_v2/g1/g1.usda"
DEFAULT_RUN, DEFAULT_CKPT = "2026-10-14_01-17-28_balance_pelvis_4096_10k", "model_68600.pt"
STANCE_Z = 0.759214
SWITCH_NORM, FILTER_TAU_S, LIFT_N, LIFT_STEPS = .05, .4, 1., 3


def list_runs():
    runs = sorted((p for p in BALANCE_RUNS.iterdir() if p.is_dir() and not p.name.startswith("0000_")
                   and any(p.glob("model_*.pt"))), key=lambda p: p.stat().st_mtime, reverse=True) \
        if BALANCE_RUNS.is_dir() else []
    return [p.name for p in runs]


def list_checkpoints(run):
    return [p.name for p in sorted((BALANCE_RUNS / run).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]), reverse=True)]


class BalanceViewer:
    def __init__(self, port, args, ranges, pad):
        import numpy as np
        import viser
        self.server = viser.ViserServer(port=port, label="HOMIE v2 walk + balance")
        gui = self.server.gui
        gui.add_markdown("### HOMIE v2 行走 + 平衡模型 · Isaac 后端\n"
                         "手柄：左摇杆上下 vx、左右 vy，右摇杆左右 wz；A 复位，B 命令归零")
        self.status = gui.add_markdown("_启动中…_")
        with gui.add_folder("手柄"):
            self.pad_state = gui.add_markdown("XInput: 未连接" if not pad.available else "XInput: 等待手柄…")
            self.use_sliders = gui.add_checkbox("用滑条代替手柄", False)
        with gui.add_folder("滑条命令 (无手柄时)"):
            self.s_vx = gui.add_slider("vx [m/s]", float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1]), 0.05, 0.0)
            self.s_vy = gui.add_slider("vy [m/s]", float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1]), 0.05, 0.0)
            self.s_wz = gui.add_slider("wz [rad/s]", float(ranges.ang_vel_yaw[0]), float(ranges.ang_vel_yaw[1]), 0.05, 0.0)
            zero = gui.add_button("命令归零")
            zero.on_click(lambda _: self.zero.set())
        runs = list_runs() or ["(无)"]
        with gui.add_folder("平衡模型"):
            self.cb_switch = gui.add_checkbox("|命令| < 0.05 时自动切换到平衡模型", True)
            self.dd_run = gui.add_dropdown("训练运行", runs, initial_value=args.run if args.run in runs else
                                           (DEFAULT_RUN if DEFAULT_RUN in runs else runs[0]))
            ck = list_checkpoints(self.dd_run.value) if runs[0] != "(无)" else ["(无)"]
            self.dd_ckpt = gui.add_dropdown("检查点", ck, initial_value=args.checkpoint if args.checkpoint in ck else
                                            (DEFAULT_CKPT if DEFAULT_CKPT in ck else ck[0]))
            self.dd_run.on_update(lambda _: self._refresh_ckpts())
        with gui.add_folder("扰动"):
            self.cb_arms = gui.add_checkbox("手臂随机动作（训练课程，含 30% 抓握保持）", False)
            self.sl_push = gui.add_slider("推一下的速度 [m/s]", 0.05, 0.5, 0.05, 0.2)
            push = gui.add_button("推一下（随机水平方向）")
            push.on_click(lambda _: self.push.set())
        with gui.add_folder("控制"):
            reset = gui.add_button("复位到推车站姿", color="green")
            reset.on_click(lambda _: self.reset.set())
            self.cb_follow = gui.add_checkbox("镜头跟随", True)
            self.sl_speed = gui.add_slider("播放倍率 (1 = 实时, 0 = 不限速)", 0.0, 2.0, 0.1, 1.0)
        with gui.add_folder("曲线 (最近 10 s)"):
            self.plot_z = gui.add_plotly(self._fig("骨盆高度 [m]", []), aspect=2.2)
            self.plot_f = gui.add_plotly(self._fig("脚速 [m/s]", []), aspect=2.2)
        self.hist = {k: deque(maxlen=500) for k in ("t", "z", "ref", "fl", "fr", "cmd")}
        self.reset, self.zero, self.push = threading.Event(), threading.Event(), threading.Event()
        self.camera_offset = np.array([1.8, -2.6, 0.9])
        self.plot_turn = 0

    def _refresh_ckpts(self):
        ck = list_checkpoints(self.dd_run.value)
        self.dd_ckpt.options = ck
        self.dd_ckpt.value = ck[0]

    @staticmethod
    def _fig(title, traces):
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, x, y, dash in traces:
            fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=name, line=dict(dash=dash, width=1.6)))
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.97), margin=dict(l=50, r=10, t=28, b=28),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)), template="plotly_white", height=230)
        return fig

    def add_meshes(self, visuals, names):
        self.meshes = [(b, self.server.scene.add_mesh_simple(f"/robot/{b}/{p}", v, f, color=c))
                       for b, p, v, f, c in visuals if b in names]
        self.server.scene.add_grid("/ground", width=40, height=40, plane="xy", cell_size=0.25, section_size=1.0)


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--run", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--demo", action="store_true", help="drive the command by itself: 4 s at vx 0.6 m/s, 6 s at zero, repeated")
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import numpy as np
        import torch
        from homie_isaaclab_v2.config import G1RoughCfg
        from homie_isaaclab_v2.env import HomieIsaacLabEnv
        from homie_pushcart.gamepad import BUTTON_A, BUTTON_B, XInputGamepad
        from homie_pushcart.lowlevel import HimLowLevelPolicy
        from homie_pushcart.teleop import CommandMapper
        from view_v7_isaac import usd_visuals

        dev = a.device
        cfg = G1RoughCfg()
        cfg.env.num_envs = 1
        cfg.env.episode_length_s = 1e5
        cfg.commands.resampling_time = 1e6
        cfg.balance.enabled = True
        cfg.balance.bank_ratio = 0.               # every (re)start is the grasp stance
        for key in dir(cfg.domain_rand):
            if key.startswith("randomize_") or key in ("use_random", "push_robots", "delay"):
                setattr(cfg.domain_rand, key, False)
        cfg.noise.add_noise = False
        ranges = cfg.commands.ranges
        pad = XInputGamepad()
        view = BalanceViewer(a.port, a, ranges, pad)
        env = HomieIsaacLabEnv(cfg, ROOT, dev, True)
        origin = env.env_origins[0].cpu().numpy()
        view.add_meshes(usd_visuals(ROBOT_USD), set(env.robot.body_names))
        walk = HimLowLevelPolicy(WALK_CKPT, dev)
        mapper = CommandMapper(ranges, smoothing_tau_s=FILTER_TAU_S, dt=env.dt)
        cmd_slice = slice(env.num_obs - env.num_one_step_obs, env.num_obs - env.num_one_step_obs + 3)
        loaded = {}

        def balance_policy():
            key = (view.dd_run.value, view.dd_ckpt.value)
            if key not in loaded:
                loaded.clear()
                loaded[key] = HimLowLevelPolicy(BALANCE_RUNS / key[0] / key[1], dev)
            return loaded[key]

        def render():
            poses = env.robot.data.body_link_pose_w.torch[0].cpu().numpy()
            index = {n: i for i, n in enumerate(env.robot.body_names)}
            with view.server.atomic():
                for body, h in view.meshes:
                    pose = poses[index[body]]
                    h.position, h.wxyz = pose[:3] - origin, np.array([pose[6], pose[3], pose[4], pose[5]])
                if view.cb_follow.value:
                    target = poses[index["pelvis"], :3] - origin
                    target[2] = 0.6
                    for client in view.server.get_clients().values():
                        with client.atomic():
                            client.camera.look_at = target
                            client.camera.position = target + view.camera_offset

        def plots():
            h = view.hist
            tt = list(h["t"])[::2]
            if len(tt) < 3:
                return
            d = lambda k: list(h[k])[::2]                                                     # noqa: E731
            view.plot_turn ^= 1
            if view.plot_turn:
                view.plot_z.figure = view._fig("骨盆高度 [m]（虚线 = 推车起始 0.7592）",
                                               [("骨盆", tt, d("z"), "solid"), ("推车起始", tt, d("ref"), "dash")])
            else:
                view.plot_f.figure = view._fig("脚速 [m/s] / |速度命令|",
                                               [("左脚", tt, d("fl"), "solid"), ("右脚", tt, d("fr"), "solid"),
                                                ("|命令|", tt, d("cmd"), "dash")])

        obs, _ = env.reset()
        unloaded = torch.zeros(1, 2, dtype=torch.long, device=dev)
        t, falls, steps_bal, feet0, bal_since, was_balance = 0., 0, 0, None, None, False
        next_tick, last_text, last_log = time.perf_counter(), 0., 0.
        print(f"[balance-viewer] viser: http://localhost:{a.port}   XInput available: {pad.available}", flush=True)
        while app.app.is_running():
            # ---- input (gamepad first, sliders otherwise), smoothed like the recorded switch states
            state = pad.poll()
            if state["connected"] and not view.use_sliders.value:
                raw, source = (state["ly"], state["lx"], state["rx"]), "手柄"
                if state["pressed"] & BUTTON_A:
                    view.reset.set()
                if state["pressed"] & BUTTON_B:
                    view.zero.set()
            else:
                raw = (view.s_vx.value / (ranges.lin_vel_x[1] if view.s_vx.value >= 0 else -ranges.lin_vel_x[0]),
                       -view.s_vy.value / ranges.lin_vel_y[1], -view.s_wz.value / ranges.ang_vel_yaw[1])
                source = "滑条"
            if a.demo:                                         # self-test: 4 s at vx 0.6 m/s, 6 s at zero
                raw, source = ((0.6 / ranges.lin_vel_x[1], 0., 0.) if t % 10. < 4. else (0., 0., 0.)), "演示"
            view.pad_state.content = f"XInput: {'已连接' if state['connected'] else '未连接'}" if pad.available else "XInput: 不可用"
            if view.zero.is_set():
                view.zero.clear()
                mapper.state = [0., 0., 0.]
                view.s_vx.value = view.s_vy.value = view.s_wz.value = 0.
                raw = (0., 0., 0.)
            cmd = torch.tensor([mapper(raw)], device=dev, dtype=torch.float32)
            if view.reset.is_set():
                view.reset.clear()
                obs, _ = env.reset()
                mapper.state = [0., 0., 0.]
                cmd.zero_(); unloaded.zero_()
                was_balance, feet0, bal_since, steps_bal = False, None, None, 0
                for k in view.hist.values():
                    k.clear()
            # ---- policy selection: GR00T's switch on the smoothed command
            balance = view.cb_switch.value and float(cmd.norm()) < SWITCH_NORM
            if balance:
                cmd.zero_()                                    # trained with an exact zero command
            if balance and not was_balance:
                feet0, bal_since, steps_bal = env.feet_pos[0, :, :2].clone(), t, 0
            was_balance = balance
            env.commands[:, :3] = cmd
            env.commands[:, 4] = cfg.rewards.base_height_target
            env.cfg.domain_rand.upper_body_randomization = bool(view.cb_arms.value)
            obs = obs.clone()
            obs[:, cmd_slice] = cmd * env.commands_scale
            with torch.inference_mode():
                action = (balance_policy() if balance else walk)(obs).clone()
            if view.push.is_set():
                view.push.clear()
                theta = 2 * math.pi * float(torch.rand(1))
                vel = env.robot.data.root_com_vel_w.torch.clone()
                vel[:, :2] = view.sl_push.value * torch.tensor([math.cos(theta), math.sin(theta)], device=dev)
                env.robot.write_root_velocity_to_sim_index(root_velocity=vel)
            obs, _, _, _, _, reset_ids, _ = env.step(action)
            t += env.dt
            if len(reset_ids):
                falls += 1
                mapper.state = [0., 0., 0.]
                was_balance, feet0, bal_since = False, None, None
            off = env.contact_forces[:, env.feet_indices, 2] < LIFT_N
            unloaded = torch.where(off, unloaded + 1, 0)
            if balance:
                steps_bal += int((unloaded == LIFT_STEPS).sum())
            z = float(env.root_states[0, 2])
            fv = env.feet_vel[0].norm(dim=-1).cpu().numpy()
            h = view.hist
            h["t"].append(t); h["z"].append(z); h["ref"].append(STANCE_Z)
            h["fl"].append(float(fv[0])); h["fr"].append(float(fv[1])); h["cmd"].append(float(torch.tensor(mapper.state).norm()))
            render()
            now = time.perf_counter()
            if now - last_text > 0.25:
                last_text = now
                bv = env.base_lin_vel[0].cpu().numpy()
                c = mapper.state
                lines = [f"- 输入源：**{source}** · 命令 vx {c[0]:+.2f}  vy {c[1]:+.2f}  wz {c[2]:+.2f}",
                         f"- 当前策略：{'**平衡模型** ' + view.dd_ckpt.value if balance else '行走模型 model_60000'}",
                         f"- 骨盆速度 vx {bv[0]:+.2f}  vy {bv[1]:+.2f}  wz {float(env.base_ang_vel[0, 2]):+.2f}",
                         f"- 骨盆高度 **{z:.4f} m**（推车起始 {STANCE_Z:.4f}，差 {1e3 * (z - STANCE_Z):+.1f} mm）· "
                         f"前倾 {float(env.pitch[0, 0]):+.3f} rad · 摔倒 {falls} 次"]
                if balance and feet0 is not None:
                    drift = (env.feet_pos[0, :, :2] - feet0).norm(dim=-1).max().item()
                    lines.append(f"- 进入平衡 {t - bal_since:.1f} s：迈步 **{steps_bal}** 次 · 脚掌漂移 **{1e3 * drift:.0f} mm**")
                view.status.content = "\n".join(lines)
                plots()
            if now - last_log > (1. if a.demo else 5.):
                last_log = now
                print(f"[balance-viewer] t {t:.1f} s  {'balance' if balance else 'walk'}  cmd {mapper.state[0]:+.2f} "
                      f"{mapper.state[1]:+.2f} {mapper.state[2]:+.2f}  base vx {float(env.base_lin_vel[0, 0]):+.2f}  "
                      f"pelvis {z:.4f}  balance steps {steps_bal if balance else '-'}  falls {falls}", flush=True)
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
