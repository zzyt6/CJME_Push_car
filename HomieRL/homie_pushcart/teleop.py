"""Xbox gamepad teleoperation of the frozen HOMIE policy, with or without the cart.

Stick mapping follows the Unitree remote: left stick up/down -> forward speed,
left stick left/right -> lateral speed, right stick left/right -> yaw rate.
Height is not controlled; HOMIE keeps its trained default (0.74 m).
Runs inside Kit with the GUI (``--viz kit``); the gamepad is read through carb.input.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import torch

from .paths import HOMIE_RUN_DIR, ROOT


class CommandMapper:
    """Scale raw [-1, 1] stick values to HOMIE's trained command ranges."""

    def __init__(self, ranges, invert_lateral=True, invert_yaw=True, smoothing_tau_s=0.1, dt=0.02):
        self.vx_range = tuple(ranges.lin_vel_x)       # [-0.8, 1.2]
        self.vy_range = tuple(ranges.lin_vel_y)       # [-0.5, 0.5]
        self.wz_range = tuple(ranges.ang_vel_yaw)     # [-0.8, 0.8]
        self.sign_y = -1.0 if invert_lateral else 1.0
        self.sign_yaw = -1.0 if invert_yaw else 1.0
        self.alpha = 1.0 if smoothing_tau_s <= 0 else min(1.0, dt / smoothing_tau_s)
        self.state = [0.0, 0.0, 0.0]

    def __call__(self, raw):
        vx_raw, vy_raw, wz_raw = (float(v) for v in raw)
        vx = vx_raw * (self.vx_range[1] if vx_raw >= 0 else -self.vx_range[0])
        vy = self.sign_y * vy_raw * self.vy_range[1]
        wz = self.sign_yaw * wz_raw * self.wz_range[1]
        target = [max(self.vx_range[0], min(self.vx_range[1], vx)),
                  max(self.vy_range[0], min(self.vy_range[1], vy)),
                  max(self.wz_range[0], min(self.wz_range[1], wz))]
        self.state = [s + self.alpha * (t - s) for s, t in zip(self.state, target)]
        return tuple(self.state)


class CompositeInput:
    """Gamepad and keyboard listened to at the same time; the one with the larger stick magnitude wins."""

    def __init__(self, gamepad=None, keyboard=None):
        self.gamepad, self.keyboard = gamepad, keyboard
        self.active = "none"

    def advance(self):
        best, best_norm = torch.zeros(3), -1.0
        for name, dev in (("gamepad", self.gamepad), ("keyboard", self.keyboard)):
            if dev is None:
                continue
            cmd = dev.advance().detach().cpu().float()
            norm = float(cmd.abs().max())
            if norm > best_norm:
                best, best_norm, self.active = cmd, norm, name
        return best

    def add_callbacks(self, reset_fn, zero_fn):
        import carb
        if self.gamepad is not None:
            self.gamepad.add_callback(carb.input.GamepadInput.A, reset_fn)
            self.gamepad.add_callback(carb.input.GamepadInput.B, zero_fn)
        if self.keyboard is not None:
            self.keyboard.add_callback("R", reset_fn)


def make_input_device(kind: str, device: str):
    """Build the input stack. Raw stick values are in [-1, 1]; CommandMapper applies HOMIE's ranges."""
    gamepad = keyboard = None
    names = []
    if kind in ("gamepad", "auto"):
        from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
        gamepad = Se2Gamepad(Se2GamepadCfg(v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.0,
                                           dead_zone=0.08, sim_device=device))
        name = gamepad._input.get_gamepad_name(gamepad._gamepad)
        names.append(f"gamepad[{name or 'not enumerated by Kit yet; connect it before launch'}]")
    if kind in ("keyboard", "auto"):
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
        keyboard = Se2Keyboard(Se2KeyboardCfg(v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.0,
                                              sim_device=device))
        names.append("keyboard[arrows/numpad move, Z/X yaw, R reset]")
    return CompositeInput(gamepad, keyboard), " + ".join(names)


def play_config(num_envs: int, episode_length_s: float):
    from homie_isaaclab.config import G1RoughCfg
    cfg = G1RoughCfg()
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = episode_length_s
    for key in dir(cfg.domain_rand):
        if key.startswith('randomize_') or key in ('use_random', 'push_robots', 'delay'):
            setattr(cfg.domain_rand, key, False)
    cfg.noise.add_noise = False
    return cfg


def run_teleop(args, launcher, root: Path = ROOT):
    from .lowlevel import HimLowLevelPolicy, frozen_checkpoint
    from .sim_env import PushCartSimEnv

    headless = bool(getattr(launcher, "_headless", True))
    if headless:
        raise RuntimeError("teleop needs the Kit window for gamepad/keyboard input: add --viz kit")
    device = args.device
    with_cart = args.mode == "cart"
    checkpoint = Path(args.checkpoint) if args.checkpoint else frozen_checkpoint()
    cfg = play_config(num_envs=1, episode_length_s=args.episode_s)
    env = PushCartSimEnv(cfg, root, device, headless, with_cart=with_cart)
    policy = HimLowLevelPolicy(checkpoint, device)
    mapper = CommandMapper(cfg.commands.ranges, invert_lateral=not args.no_invert, invert_yaw=not args.no_invert,
                           smoothing_tau_s=args.smoothing, dt=env.dt)
    input_device, input_name = make_input_device(args.input, device)

    request_reset = {"flag": False}

    def _flag_reset():
        request_reset["flag"] = True

    zero_cmd = {"flag": False}

    def _flag_zero():
        zero_cmd["flag"] = True

    try:
        input_device.add_callbacks(_flag_reset, _flag_zero)
    except Exception as exc:  # noqa: BLE001
        print(f"[teleop] button callbacks unavailable: {exc}", flush=True)

    print("=" * 72)
    print(f"[teleop] mode={args.mode}  cart={'yes' if with_cart else 'no'}  input={input_name}")
    print(f"[teleop] checkpoint={checkpoint}  iter={policy.iteration}")
    print(f"[teleop] left stick: forward {cfg.commands.ranges.lin_vel_x[1]:+.1f} / back {cfg.commands.ranges.lin_vel_x[0]:+.1f} m/s, "
          f"lateral ±{cfg.commands.ranges.lin_vel_y[1]:.1f} m/s;  right stick: yaw ±{cfg.commands.ranges.ang_vel_yaw[1]:.1f} rad/s;  "
          f"height fixed {cfg.rewards.base_height_target:.2f} m")
    print("[teleop] A / R = reset,  B = zero command.  Close the window or Ctrl+C to stop.")
    print("=" * 72, flush=True)

    obs, _ = env.reset()
    env.set_command(0., 0., 0.)
    falls, steps = 0, 0
    period = env.dt
    next_tick = time.perf_counter()
    last_status = time.perf_counter()
    app = launcher.app
    with torch.inference_mode():
        while app.is_running():
            raw = input_device.advance()
            if zero_cmd["flag"]:
                zero_cmd["flag"] = False
                mapper.state = [0., 0., 0.]
                raw = torch.zeros(3)
            vx, vy, wz = mapper(raw)
            env.set_command(vx, vy, wz)
            if request_reset["flag"]:
                request_reset["flag"] = False
                obs, _ = env.reset()
                env.set_command(vx, vy, wz)
                print("[teleop] manual reset", flush=True)
            actions = policy(obs)
            obs, _, _, done, info, ids, _ = env.step(actions)
            steps += 1
            terminated = int(info["terminated"].sum().item()) if "terminated" in info else 0
            falls += terminated
            if terminated:
                print(f"[teleop] fall -> reset (total falls {falls})", flush=True)
            now = time.perf_counter()
            if now - last_status >= 0.5:
                base_v = env.base_lin_vel[0]
                line = (f"[{input_device.active:8s}] cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} | "
                        f"base vx={base_v[0]:+.2f} vy={base_v[1]:+.2f} wz={env.base_ang_vel[0, 2]:+.2f} | "
                        f"h={env.root_states[0, 2]:.3f}")
                cart = env.cart_state()
                if cart is not None:
                    pos, quat, lin, ang = cart
                    yaw = math.atan2(2 * (quat[0, 3] * quat[0, 2] + quat[0, 0] * quat[0, 1]),
                                     1 - 2 * (quat[0, 1] ** 2 + quat[0, 2] ** 2))
                    speed = math.hypot(lin[0, 0].item(), lin[0, 1].item())
                    line += f" | cart v={speed:.2f} m/s yaw={math.degrees(yaw):+.1f}° wz={ang[0, 2]:+.2f}"
                line += f" | falls={falls} t={steps * period:.0f}s"
                print(line, flush=True)
                last_status = now
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -1.0:
                next_tick = time.perf_counter()
    print(f"[teleop] stopped after {steps} steps, {falls} falls", flush=True)
