"""Export a high-level checkpoint to TorchScript / ONNX with a plain-tensor interface, plus an interface JSON.

Deployment runs two networks side by side at 50 Hz:
    HOMIE  : 456 -> 12 leg actions (existing ONNX)                  actor input = HOMIE's 6 x 76 history
    upper  : V5 stacked frames -> 18 (this export; normaliser folded in)
The interface file documents every slice of the V5 frame, the units/scales, and how the 18 outputs
are turned into a HOMIE command and 15 upper-body PD targets (identical to ``PushCartEnv.step``).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .paths import ROOT
from .task_config import HL_FRAME, HOMIE_FRAME, PushCartTaskCfg, num_actor_obs, num_critic_obs, task_cfg_from_run


class _PlainActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        import copy
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        # A heteroscedastic actor has an additional standard-deviation head.
        # Export its deterministic mean through RSL-RL's own contract.
        self.output_module = (actor.distribution.as_deterministic_output_module()
                              if actor.distribution is not None else torch.nn.Identity())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        output = self.mlp(self.obs_normalizer(obs))
        return self.output_module(output)


def frame_layout(task_cfg: PushCartTaskCfg, homie_cfg=None) -> list[dict]:
    """Slices of one V5 actor frame, including the causal reaction measurement."""
    s = task_cfg.observations
    if homie_cfg is None:
        from homie_isaaclab.config import G1RoughCfg
        homie_cfg = G1RoughCfg()
    hs = homie_cfg.normalization.obs_scales
    commands_scale = [hs.lin_vel, hs.lin_vel, hs.ang_vel]
    items = [
        ("homie_cmd", 3, f"current HOMIE command (vx, vy, wz) x {commands_scale}"),
        ("homie_height_cmd", 1, f"HOMIE height command (constant {homie_cfg.rewards.base_height_target})"),
        ("imu_ang_vel", 3, f"pelvis IMU angular velocity x {hs.ang_vel}"),
        ("imu_gravity", 3, "projected gravity in the pelvis frame"),
        ("dof_pos", 27, f"(joint position - HOMIE default) x {hs.dof_pos}, HOMIE order (12 legs, waist_yaw, 7 + 7 arm)"),
        ("dof_vel", 27, f"joint velocity x {hs.dof_vel}"),
        ("leg_actions", 12, "previous HOMIE leg actions"),
        ("task_cmd", 3, f"task command reference (vx*, vy*, wz*) in the cart frame x {commands_scale}; first-order response of the "
                        f"commanded step, tau {task_cfg.commands.ref_tau_s} s (0 = the raw command)"),
        ("cart_mass", 1, f"(estimated total cart mass kg - {s.cart_mass_norm[0]}) / {s.cart_mass_norm[1]}; includes configured mass-estimate uncertainty"),
        ("upper_torque", 15, f"measured upper-body joint torques x {s.torque_scale} (waist_yaw, 7 + 7 arm)"),
        ("est_base_lin_vel", 3, f"HIM estimator base linear velocity x {hs.lin_vel}"),
        ("cmd_minus_est", 3, f"(homie vx - est vx, homie vy - est vy, homie wz - imu wz) x {commands_scale}"),
        ("handle_pose_in_base", 4, "handle bar centre in the pelvis frame (x, y, z) [m] and cart yaw - robot yaw [rad]"),
        ("last_hl_action", 18, "previous clipped high-level action"),
        ("correction_scale", 1, f"arm correction scale / {task_cfg.actions.arm_scale_range[1]} (1.0 at the final level)"),
    ]
    from .reaction import DEFAULT_REACTION_CFG
    reaction = task_cfg.reaction
    value = lambda name: getattr(reaction, name, DEFAULT_REACTION_CFG[name])
    items.extend([
        ("reaction_wrench", 6, f"causal virtual wrist robot-on-cart [Fx,Fy,Fz,Mx,My,Mz] about calibrated estimated whole-cart COM, observed cart-heading axes; divide by {value('sensor_scale')}"),
        ("reaction_required_wrench", 3, f"task-model [Fx,Fy,Mz], delayed and filtered identically to sensor; divide by {value('wrench_scale')}"),
        ("reaction_excess_wrench", 3, f"measured minus aligned task-model wrench [Fx,Fy,Mz]; divide by {value('wrench_scale')}"),
        ("reaction_command_correction", 3, f"bounded compensator [vx,vy,wz] in cart-heading axes; divide by {value('correction_limit')}"),
        ("reaction_recovery_gate", 1, f"finite-duration static recovery gate divided by {value('gate_max')}"),
        ("reaction_sensor_valid", 1, f"1 only when causal sample is valid; all 17 reaction features clip to +/-{value('feature_clip')}"),
    ])
    out, start = [], 0
    for name, dim, note in items:
        out.append({"name": name, "start": start, "dim": dim, "note": note})
        start += dim
    assert start == HL_FRAME and items[0:7] and sum(d for _, d, _ in items[:7]) == HOMIE_FRAME
    return out


def export_checkpoint(checkpoint: Path, output_dir: Path | None = None, level: int = 17) -> dict:
    from .curriculum import level_params
    from .hl_policy import HighLevelPolicy
    task_cfg = task_cfg_from_run(checkpoint)                       # the run's own residual scale / clip / reference
    from .env_smoke import make_configs
    homie_cfg, _ = make_configs(1, False, checkpoint=checkpoint)
    hl = HighLevelPolicy(checkpoint, num_actor_obs(task_cfg), num_critic_obs(task_cfg), task_cfg.env.num_hl_actions, "cpu")
    # The V5.2 actor contains the history encoder and the per-hand force estimator; export all of it.
    plain = hl.actor.as_jit().eval()
    output_dir = output_dir or (Path(checkpoint).parent / "exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, hl.num_actor_obs)
    with torch.inference_mode():
        reference = hl(example)
        if not torch.allclose(reference, plain(example), atol=1e-6, rtol=1e-5):
            raise ValueError("Exported deterministic actor disagrees with checkpoint inference")
        scripted = torch.jit.script(plain)
        jit_path = output_dir / "hl_policy.pt"
        scripted.save(str(jit_path))
        onnx_path = output_dir / "hl_policy.onnx"
        torch.onnx.export(plain, example, str(onnx_path), input_names=["actor_obs"], output_names=["hl_action"],
                          dynamic_axes={"actor_obs": {0: "batch"}, "hl_action": {0: "batch"}}, opset_version=17, dynamo=False)
        check = torch.jit.load(str(jit_path))(example)
    lp = level_params(level, task_cfg)
    from homie_isaaclab.math_utils import class_to_dict
    interface = {
        "version": 5, "implementation_revision": getattr(task_cfg, "implementation_revision", 1),
        "runtime_implementation_revision": PushCartTaskCfg.implementation_revision,
        "task_id": "Homie-G1-PushCart-V52-Direct-v0",
        "checkpoint": str(checkpoint), "iteration": hl.iteration,
        "rate_hz": 1. / (homie_cfg.sim.dt * homie_cfg.control.decimation),
        "actor_obs_dim": hl.num_actor_obs, "frames": task_cfg.env.actor_history, "frame_dim": HL_FRAME,
        "frame_order": f"oldest first; the newest frame occupies the last {HL_FRAME} entries",
        "frame_layout": frame_layout(task_cfg, homie_cfg),
        "obs_clip": homie_cfg.normalization.clip_observations,
        "action_dim": hl.num_actions, "action_clip": task_cfg.actions.clip,
        "task_config_source": getattr(task_cfg, "source_yaml", None),
        "action_to_command": {
            "reference": f"cmd_ref += min(1, dt / {task_cfg.commands.ref_tau_s}) * (task_cmd - cmd_ref)  (tau 0: cmd_ref = task_cmd)",
            "idle_gate": f"gate = max(clip(|cmd_ref| / {task_cfg.actions.idle_gate_speed}, 0, 1), idle_correction_gate, reaction_recovery_gate) (zero speed threshold: command gate = 1); scales a[0:3] and a[3:18]",
            "idle_correction": {
                "config": class_to_dict(task_cfg.idle_correction),
                "runtime": "homie_pushcart_v52.idle_correction.IdleCorrectionTracker",
                "inputs": "actor-visible handle pose, HIM estimated base velocity, IMU angular velocity and projected gravity; unscaled; known operator task command",
                "timing": "reset history on episode reset; update once per observed control step; next action uses max(command gate, tracker.gate(cmd_ref, task_cmd))",
                "network_contains_gate": False,
            },
            "reaction_recovery": {
                "config": class_to_dict(task_cfg.reaction),
                "runtime": "homie_pushcart_v52.reaction_observation.ReactionObservation (owns ReactionController)",
                "network_contains_controller": False,
                "inputs": "observed delayed/noisy relative handle pose and yaw, HOMIE base velocity estimate, IMU gravity/angular velocity, virtual robot-on-cart wrist wrench shifted to calibrated cart COM, task reference, estimated cart mass and configured robot mass",
                "mass_contract": "cart mass is an estimate with configured uncertainty; simulator supplies actual robot assembly mass as a known model parameter, which hardware must replace with identified total robot mass",
                "kinematic_model": "windowed observed handle motion estimates cart-COM twist using nominal handle-COM lever arm and inertia m*r_gyration^2; yaw-only transforms assume near-upright motion",
                "motion_confirmation": "force-based correction requires matching measured velocity-error direction; its magnitude is capped by force_motion_cap_gain * abs(velocity_error) before command limits/filter",
                "measurement_contract": "common cart-heading axes; causal delay/filter must also be applied to required wrench before residual; no simulated disturbance/contact truth in actor inputs",
                "timing": "reset at episode start; update exactly once per control observation; next action consumes stored correction and recovery gate",
                "scope": "bounded velocity-command compensation through frozen HOMIE; not torque or contact-force tracking",
            },
            "homie_cmd_target": "clip(ReactionObservation.reference_base(cmd_ref) + gate * a[0:3] * cmd_residual_scale + ReactionObservation.command_correction_base, homie_cmd_limits); reference_base shifts cart-COM twist to the robot base point using observed handle geometry and then rotates by observed delayed/noisy relative yaw",
            "cmd_residual_scale": task_cfg.actions.cmd_residual_scale, "homie_cmd_limits": task_cfg.actions.homie_cmd_limits,
            "low_pass": f"homie_cmd += min(1, dt / {task_cfg.actions.cmd_filter_tau_s}) * (target - homie_cmd)",
            "action_clip": task_cfg.actions.clip,
        },
        "action_to_upper_targets": {
            "joints": ["waist_yaw"] + [f"{s}_{n}" for s in ("left", "right") for n in
                                       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
            "delta_target_rad": "gate * a[3:18] * scale, scale = [waist_scale] + [arm_scale] * 14",
            "scale_at_export_level": {"level": level, "waist_scale": lp.waist_scale, "arm_scale": lp.arm_scale},
            "slew_limit_rad_per_s": task_cfg.actions.upper_rate_limit_rad_s,
            "target": "clip(q_grasp + upper_hold_offset + delta, hard joint limits); q_grasp / upper_hold_offset from initial_pose.yaml",
            "correction_scale_obs": lp.arm_scale / task_cfg.actions.arm_scale_range[1],
        },
        "companion": "HOMIE low-level ONNX (456 -> 12), fed the same 76-dim frames; upper body follows the PD targets above",
        "files": {"torchscript": str(jit_path), "onnx": str(onnx_path)},
        "self_check_max_abs_diff": float((reference - check).abs().max()),
    }
    (output_dir / "hl_interface.json").write_text(json.dumps(interface, indent=2), encoding="utf-8")
    print(f"[export] {jit_path}\n[export] {onnx_path}\n[export] {output_dir / 'hl_interface.json'} (jit self-check diff {interface['self_check_max_abs_diff']:.2e})")
    return interface
