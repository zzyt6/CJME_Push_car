"""V6's Gymnasium wrapper; the task owns V3.2 physics and separate estimator labels."""
import json
from pathlib import Path

from isaaclab.envs import DirectRLEnv
from isaaclab_physx.physics import PhysxCfg

from homie_isaaclab.config import G1RoughCfg
from homie_isaaclab.direct_env import _apply_dict
from homie_pushcart.direct_env import PushCartDirectEnv as V3DirectEnv

from .baseline import apply_config, baseline_source
from .paths import ROOT
from .physics import physics_manifest
from .task_config import PushCartTaskCfg, validate_task_schema
from .task_env import PushCartEnv


class PushCartDirectEnv(V3DirectEnv):
    def _init_sim(self, render_mode=None, **kwargs):
        if not isinstance(self.cfg.sim.physics, PhysxCfg):
            raise ValueError("V6 uses the pinned V3.2 PhysX training backend")
        if render_mode == "rgb_array":
            raise ValueError("Video recording is not implemented; omit --video")
        if self.cfg.allow_robot_cart_contact or self.cfg.cart_solver_iterations is not None:
            raise ValueError("V6 retains the V3.2 collision filter and cart solver settings")
        manifest = physics_manifest()
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
        apply_config(task_cfg, {key: value for key, value in self.cfg.task.items() if key != "source_yaml"})
        task_cfg.env.episode_length_s = self.cfg.episode_length_s
        validate_task_schema(task_cfg)
        self.homie = PushCartEnv(
            cfg, task_cfg, ROOT, self.device, not self.sim.has_gui, sim_context=self.sim,
            checkpoint=Path(self.cfg.checkpoint) if self.cfg.checkpoint else None,
            level=self.cfg.curriculum_level,
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
                json.dumps(self.homie.physics_report, indent=2), encoding="utf-8")
            (log_dir / "physics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            metadata = {
                "version": 6, "implementation_revision": task_cfg.implementation_revision,
                "task_id": "Homie-G1-PushCart-V6-Direct-v0",
                "observation_schema": task_cfg.observation_schema,
                "baseline_source": baseline_source(), "physics_manifest": manifest,
                "homie_checkpoint": str(self.homie.policy.checkpoint),
                "homie_iteration": self.homie.policy.iteration,
                "actor_observation": 605, "estimator_observation": 462,
                "actor_mlp_input": 652, "critic_observation": 665, "actions": 18,
                "actor_history": 5, "cart_estimator_history": 6, "force_estimator_history": 5,
                "initial_level": self.homie.curriculum.level,
                "level_params": self.homie.level_params.as_dict(),
                "labels": {
                    "hand_target": "12 palm-origin Newton-Euler hand-subtree labels, cart-heading axes; "
                                   "per-hand [Fx,Fy,Fz,Mx,My,Mz]/[50,50,50,5,5,5], then valid",
                    "him_next_target": "pre-reset next clean HOMIE76+relative yaw; "
                                       "cart COM [vx,vy,wz]*[2,2,.5]; then valid (81 dimensions)",
                    "reward_source": "unchanged V3.2 COM velocity and gravity-compensated hand proxy",
                    "privilege_use": "labels and Critic only; Actor receives predictions",
                },
                "cart_quality": ({**self.homie.cart_quality.metadata(),
                                  "wrench_source": "physx_incoming_joint_gravity_compensated_proxy",
                                  "moment_origin": "cart_assembly_com",
                                  "sampling": "last_physics_substep_at_control_rate"}
                                 if self.homie.cart_quality is not None else None),
            }
            (log_dir / "pushcart_task.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _get_observations(self):
        return self.homie.observation_dict()

    def step(self, actions):
        _, _, rewards, _, extras = self.homie.step(actions)
        self.extras = extras
        return self._get_observations(), rewards, extras["terminated"], extras["time_outs"], extras
