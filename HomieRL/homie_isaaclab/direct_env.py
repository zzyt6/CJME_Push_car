"""Registered DirectRLEnv exposing HOMIE through the standard Gymnasium API.

The base class owns context creation, config resolution and seeding. HOMIE's
existing scene and step implementation retain the mixed upper/lower control
and pre-reset estimator targets. No runner-specific tuple crosses Gymnasium.
"""
from pathlib import Path
import json

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_physx.physics import PhysxCfg

from .config import G1RoughCfg
from .env import HomieIsaacLabEnv


def _apply_dict(target, values):
    for name, value in values.items():
        current = getattr(target, name)
        if isinstance(value, dict) and not isinstance(current, dict):
            _apply_dict(current, value)
        else:
            setattr(target, name, value)


class HomieDirectEnv(DirectRLEnv):
    def _init_sim(self, render_mode=None, **kwargs):
        if not isinstance(self.cfg.sim.physics, PhysxCfg):
            raise ValueError("HOMIE currently requires physics=physx.")
        if render_mode == "rgb_array":
            raise ValueError("HOMIE video recording is not implemented; omit --video.")
        cfg = G1RoughCfg()
        _apply_dict(cfg, self.cfg.homie)
        cfg.env.num_envs = self.cfg.scene.num_envs
        cfg.env.env_spacing = self.cfg.scene.env_spacing
        cfg.env.episode_length_s = self.cfg.episode_length_s
        cfg.control.decimation = self.cfg.decimation
        cfg.sim.dt = self.cfg.sim.dt
        cfg.sim.gravity = list(self.cfg.sim.gravity)
        if not self.cfg.enable_randomization:
            for key in dir(cfg.domain_rand):
                if key.startswith("randomize_") or key in ("use_random", "push_robots", "delay"):
                    setattr(cfg.domain_rand, key, False)
            cfg.noise.add_noise = False
        root = Path(__file__).resolve().parents[2]
        self.homie = HomieIsaacLabEnv(
            cfg, root, self.device, not self.sim.has_gui, sim_context=self.sim,
        )
        self.scene = self.homie.scene
        self.sim.register_interactive_scene(self.scene)
        self.scene.initialize_renderers()
        self.viewport_camera_controller = None
        self.video_recorder = None
        self._window = None
        self._debug_vis_handle = None
        self.has_debug_vis_implementation = False
        self.extras = {}
        self._configure_gym_env_spaces()
        self.metadata = {**self.metadata, "render_fps": round(1 / self.step_dt)}
        if self.cfg.log_dir:
            log_dir = Path(self.cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "physics_config.json").write_text(
                json.dumps(self.homie.physics_report, indent=2), encoding="utf-8",
            )

    @property
    def episode_length_buf(self):
        return self.homie.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.homie.episode_length_buf = value

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.homie.reset()
        return self._get_observations(), {}

    def _get_observations(self):
        # Do not recompute here: HOMIE observation computation shifts history.
        # The official wrapper may query observations repeatedly without a step.
        return {"policy": self.homie.obs_buf, "critic": self.homie.privileged_obs_buf}

    def step(self, actions):
        obs, critic, rewards, _, extras, ids, terminal = self.homie.step(actions)
        extras["homie_terminal_ids"] = ids
        extras["homie_terminal_critic"] = terminal
        self.extras = extras
        return {"policy": obs, "critic": critic}, rewards, extras["terminated"], extras["time_outs"], extras

    def render(self, recompute=False):
        if recompute:
            self.sim.render()
        return None

    def close(self):
        if not self._is_closed:
            super().close()
            del self.homie
