"""V8's Gymnasium wrapper (teacher phase): the V7.1 plant through the V8 observation / action interface."""
import json
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

from homie_isaaclab.config import G1RoughCfg
from homie_isaaclab.direct_env import _apply_dict
from homie_pushcart.direct_env import PushCartDirectEnv as V3DirectEnv
from homie_pushcart_v7.baseline import apply_config, baseline_source
from homie_pushcart_v7.paths import ROOT
from homie_pushcart_v7.physics import physics_manifest

from .task_config import (ACTION_DIM, EXPLICIT_DIM, FRAME_DIM, LONG_HISTORY, POLICY_HISTORY, PRIVILEGED_DIM,
                          PushCartTaskCfg, validate_v8)
from .task_env import PushCartEnv


class PushCartDirectEnv(V3DirectEnv):
    def _init_sim(self, render_mode=None, **kwargs):
        if not isinstance(self.cfg.sim.physics, PhysxCfg):
            raise ValueError("V8 uses the pinned V3.2 PhysX training backend")
        if render_mode == "rgb_array":
            raise ValueError("Video recording is not implemented; omit --video")
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
        validate_v8(task_cfg)
        self.homie = PushCartEnv(
            cfg, task_cfg, ROOT, self.device, not self.sim.has_gui, sim_context=self.sim,
            checkpoint=Path(self.cfg.checkpoint) if self.cfg.checkpoint else None,
            level=self.cfg.curriculum_level,
        )
        if self.cfg.log_dir:                            # fine-tuned HOMIE velocity estimator next to the checkpoints
            self.homie.homie_velocity_save_path = Path(self.cfg.log_dir) / "homie_velocity_estimator.pt"
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
            (log_dir / "physics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            metadata = {
                "release": task_cfg.release, "phase": "teacher", "task_id": "Homie-G1-PushCart-V8-Teacher-Direct-v0",
                "observation_schema": task_cfg.v8.observation_schema, "plant_base": "V7.1 (V7 revision 10)",
                "baseline_source": baseline_source(), "physics_manifest": manifest,
                "homie_checkpoint": str(self.homie.policy.checkpoint), "homie_iteration": self.homie.policy.iteration,
                "frame": {"dim": FRAME_DIM, "layout": "HOMIE76(noisy) | task command 3 | upper torque 15 | camera "
                                                     "relative yaw 1 | applied actions 30 | residual scale 1"},
                "policy_history": POLICY_HISTORY, "long_history": LONG_HISTORY,
                "privileged": {"dim": PRIVILEGED_DIM, "explicit_first": EXPLICIT_DIM,
                               "layout": "cart v (x2, x2, x0.5) 3 | cart true mass (norm) 1 | robot v x2 3 | hand "
                                         "wrenches 12 | hand valid 1 | payload COM 2 | V7 grasp + cart privileges 30 | "
                                         "yaw damping /10 1 | wheel friction 1 | caster sin/cos 8"},
                "actions": {"dim": ACTION_DIM, "layout": "V7 high-level 18 | HOMIE leg residual 12 (clip "
                                                          f"{task_cfg.v8.leg_residual.clip}, enabled "
                                                          f"{task_cfg.v8.leg_residual.enabled})"},
                "camera": {k: getattr(task_cfg.v8.camera, k) for k in dir(task_cfg.v8.camera) if not k.startswith("_")},
            }
            (log_dir / "pushcart_task.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    def _get_observations(self):
        return self.homie.observation_dict()

    def step(self, actions):
        _, _, rewards, _, extras = self.homie.step(actions)
        self.extras = extras
        return self._get_observations(), rewards, extras["terminated"], extras["time_outs"], extras
