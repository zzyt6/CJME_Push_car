"""Registered DirectRLEnv exposing ``PushCartEnv`` through the standard Gymnasium / RSL-RL API.

Mirrors ``homie_isaaclab.direct_env.HomieDirectEnv``: the base class owns the simulation context, config
resolution and seeding; the task env keeps HOMIE's mixed stepping and its own observation groups.
"""
from pathlib import Path
import json

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_physx.physics import PhysxCfg

from homie_isaaclab.config import G1RoughCfg
from homie_isaaclab.direct_env import _apply_dict

from .paths import ROOT
from .task_config import PushCartTaskCfg
from .task_env import PushCartEnv


class PushCartDirectEnv(DirectRLEnv):
    def v5_curriculum_state_dict(self):
        from .checkpointing import curriculum_state
        return curriculum_state(self.homie)

    def v5_load_curriculum_state_dict(self, state, explicit_level=None):
        from .checkpointing import restore_curriculum
        return restore_curriculum(self.homie, state, explicit_level)

    def _init_sim(self, render_mode=None, **kwargs):
        if not isinstance(self.cfg.sim.physics, PhysxCfg):
            raise ValueError("The push-cart task currently requires physics=physx.")
        if render_mode == "rgb_array":
            raise ValueError("Video recording is not implemented; omit --video.")
        cfg = G1RoughCfg()
        _apply_dict(cfg, self.cfg.homie)
        cfg.env.num_envs = self.cfg.scene.num_envs
        cfg.env.env_spacing = self.cfg.scene.env_spacing
        cfg.control.decimation = self.cfg.decimation
        cfg.sim.dt = self.cfg.sim.dt
        cfg.sim.gravity = list(self.cfg.sim.gravity)
        if not self.cfg.enable_randomization:
            for key in dir(cfg.domain_rand):
                if key.startswith("randomize_") or key in ("use_random", "push_robots", "delay"):
                    setattr(cfg.domain_rand, key, False)
            cfg.noise.add_noise = False
        task_cfg = PushCartTaskCfg()
        _apply_dict(task_cfg, self.cfg.task)
        task_cfg.env.episode_length_s = self.cfg.episode_length_s
        cart_params = None
        if self.cfg.cart_solver_iterations:
            from .assets.cart_urdf import load_cart_params
            cart_params = load_cart_params()
            pos_it, vel_it = self.cfg.cart_solver_iterations
            cart_params["solver"] = {"position_iterations": int(pos_it), "velocity_iterations": int(vel_it)}
        self.homie = PushCartEnv(
            cfg, task_cfg, ROOT, self.device, not self.sim.has_gui, sim_context=self.sim,
            checkpoint=Path(self.cfg.checkpoint) if self.cfg.checkpoint else None, level=self.cfg.curriculum_level,
            cart_params=cart_params, allow_robot_cart_contact=bool(self.cfg.allow_robot_cart_contact),
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
            (log_dir / "physics_config.json").write_text(json.dumps(self.homie.physics_report, indent=2), encoding="utf-8")
            (log_dir / "pushcart_task.json").write_text(json.dumps({
                "version": 5, "implementation_revision": PushCartTaskCfg.implementation_revision,
                "task_id": "Homie-G1-PushCart-V5-Direct-v0",
                "homie_checkpoint": str(self.homie.policy.checkpoint), "homie_iteration": self.homie.policy.iteration,
                "actor_obs": self.homie.num_hl_obs, "critic_obs": self.homie.num_hl_critic_obs,
                "actions": self.homie.num_hl_actions, "initial_level": self.homie.curriculum.level,
                "level_params": self.homie.level_params.as_dict(),
                "dynamics": getattr(self.homie, "dynamics_metadata", None),
                "reaction": self.cfg.task.get("reaction"),
                "cart_quality": ({**self.homie.cart_quality.metadata(),
                                  "wrench_source": "physx_incoming_joint_hand_subtree_newton_euler_proxy",
                                  "moment_origin": "cart_assembly_com", "sampling": "last_physics_substep_at_control_rate"}
                                 if self.homie.cart_quality is not None else None),
            }, indent=2), encoding="utf-8")

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
        return {"policy": self.homie.hl_obs, "critic": self.homie.hl_critic_obs}

    def step(self, actions):
        obs, critic, rewards, _, extras = self.homie.step(actions)
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
