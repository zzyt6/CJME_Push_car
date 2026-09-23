"""Export a high-level checkpoint to TorchScript / ONNX with a plain-tensor interface, plus an interface JSON.

Deployment runs two networks side by side at 50 Hz:
    HOMIE  : 456 -> 12 leg actions (existing ONNX)                  actor input = HOMIE's 6 x 76 history
    upper  : 620 -> 18 (this export; normaliser folded in)          actor input = 5 x 124 frames (see layout)
The interface file documents every slice of the 124-dim frame, the units/scales, and how the 18 outputs
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))


def frame_layout(task_cfg: PushCartTaskCfg) -> list[dict]:
    """Slices of one 124-dim actor frame, in order, with scaling conventions."""
    s = task_cfg.observations
    items = [
        ("homie_cmd", 3, "current HOMIE command (vx, vy, wz) x [2, 2, 0.5]"),
        ("homie_height_cmd", 1, "HOMIE height command (constant 0.74)"),
        ("imu_ang_vel", 3, "pelvis IMU angular velocity x 0.5"),
        ("imu_gravity", 3, "projected gravity in the pelvis frame"),
        ("dof_pos", 27, "joint position - HOMIE default, HOMIE order (12 legs, waist_yaw, 7 + 7 arm)"),
        ("dof_vel", 27, "joint velocity x 0.05"),
        ("leg_actions", 12, "previous HOMIE leg actions"),
        ("task_cmd", 3, f"task command reference (vx*, vy*, wz*) in the cart frame x [2, 2, 0.5]; first-order response of the "
                        f"commanded step, tau {task_cfg.commands.ref_tau_s} s (0 = the raw command)"),
        ("cart_mass", 1, f"(cart total mass kg - {s.cart_mass_norm[0]}) / {s.cart_mass_norm[1]}"),
        ("upper_torque", 15, f"measured upper-body joint torques x {s.torque_scale} (waist_yaw, 7 + 7 arm)"),
        ("est_base_lin_vel", 3, "HIM estimator base linear velocity x 2 (m/s x 2)"),
        ("cmd_minus_est", 3, "(homie vx - est vx, homie vy - est vy, homie wz - imu wz) x [2, 2, 0.5]"),
        ("handle_pose_in_base", 4, "handle bar centre in the pelvis frame (x, y, z) [m] and cart yaw - robot yaw [rad]"),
        ("last_hl_action", 18, "previous clipped high-level action"),
        ("correction_scale", 1, f"arm correction scale / {task_cfg.actions.arm_scale_range[1]} (1.0 at the final level)"),
    ]
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
    hl = HighLevelPolicy(checkpoint, num_actor_obs(task_cfg), num_critic_obs(task_cfg), task_cfg.env.num_hl_actions, "cpu")
    plain = _PlainActor(hl.actor).eval()
    output_dir = output_dir or (Path(checkpoint).parent / "exported")
    output_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, hl.num_actor_obs)
    with torch.inference_mode():
        reference = plain(example)
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
        "checkpoint": str(checkpoint), "iteration": hl.iteration,
        "rate_hz": 50, "actor_obs_dim": hl.num_actor_obs, "frames": task_cfg.env.actor_history, "frame_dim": HL_FRAME,
        "frame_order": "oldest first; the newest frame occupies the last 124 entries",
        "frame_layout": frame_layout(task_cfg),
        "obs_clip": 100.0,
        "action_dim": hl.num_actions, "action_clip": task_cfg.actions.clip,
        "task_config_source": getattr(task_cfg, "source_yaml", None),
        "action_to_command": {
            "reference": f"cmd_ref += min(1, dt / {task_cfg.commands.ref_tau_s}) * (task_cmd - cmd_ref)  (tau 0: cmd_ref = task_cmd)",
            "idle_gate": f"gate = clip(|cmd_ref| / {task_cfg.actions.idle_gate_speed}, 0, 1)  (0: no gate); scales a[0:3] and a[3:18]",
            "idle_correction": {
                "config": class_to_dict(task_cfg.idle_correction),
                "runtime": "homie_pushcart.idle_correction.IdleCorrectionTracker",
                "inputs": "actor-visible handle pose, HIM estimated base velocity, IMU angular velocity and projected gravity; unscaled; known operator task command",
                "timing": "reset history on episode reset; update once per observed control step; next action uses max(command gate, tracker.gate(cmd_ref, task_cmd))",
                "network_contains_gate": False,
            },
            "homie_cmd_target": "clip(cmd_ref + gate * a[0:3] * cmd_residual_scale, homie_cmd_limits)",
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
