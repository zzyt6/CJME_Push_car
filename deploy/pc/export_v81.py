"""Export the released V8.1 push-cart stack for the real G1 (run on the training PC, RLisaac601 env).

    python deploy/pc/export_v81.py                 # -> deploy/robot/models/

Writes (everything the robot needs; the robot needs no torch, no Isaac, no MuJoCo):

* v81_student.onnx   policy (1, 630) + history (1, 4032) -> actions (1, 30), velocity_estimate (1, 19)
                      = homie_pushcart_v8.rma_policy.RMAStudentExport of the released student (z = phi(history)).
* homie_v81.onnx      obs_history (1, 456) + use_finetuned (1, 1) -> leg_actions (1, 12), vel_used (1, 3),
                      vel_homie (1, 3). HOMIE model_60000 actor with HOMIE's own latent; the velocity input is the
                      push-cart fine-tuned estimator (use_finetuned = 1, V8.1) or HOMIE's own (= 0, plain HOMIE).
* v81_config.json     every constant of the inference pipeline, read from the same MujocoV8Runner object that the
                      sim2sim bench uses (so the robot code cannot drift from the validated pipeline), plus provenance
                      (paths + sha256 of the three checkpoints).

Each ONNX graph is checked against the PyTorch module on random inputs before the files are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HomieRL"))
sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "deploy" / "robot" / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StudentGraph(nn.Module):
    """RMAStudentExport + its explicit (velocity head) output, one graph."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, policy, history):
        return self.student(policy, history), self.student.explicit(history)


class HomieGraph(nn.Module):
    """HOMIE's leg actor with HOMIE's own (L2-normalised) latent and a selectable velocity estimate.

    Same computation as MujocoV8Runner.step: vel_homie, latent = estimator(history); vel = hv(history)[:3] with the high
    level on, else vel_homie; legs = actor(cat(last frame, vel, latent)).
    """

    def __init__(self, model, hv):
        super().__init__()
        self.encoder = model.estimator.encoder
        self.actor = model.actor
        self.hv = hv
        self.frame = int(model.num_one_step_obs)

    def forward(self, obs_history, use_finetuned):
        parts = self.encoder(obs_history)
        vel_homie = parts[:, :3]
        latent = nn.functional.normalize(parts[:, 3:], dim=-1, p=2.0)
        vel_ft = self.hv(obs_history)[:, :3]
        vel = use_finetuned * vel_ft + (1.0 - use_finetuned) * vel_homie
        legs = self.actor(torch.cat((obs_history[:, -self.frame:], vel, latent), dim=-1))
        return legs, vel, vel_homie


def export_onnx(module, inputs, names_in, names_out, path: Path):
    torch.onnx.export(module, inputs, str(path), input_names=names_in, output_names=names_out, opset_version=17,
                      dynamo=False)


def check_onnx(module, path: Path, names_in, make_inputs, trials=64):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(trials):
        inputs = make_inputs(rng)
        with torch.inference_mode():
            ref = module(*[torch.from_numpy(x) for x in inputs])
        out = sess.run(None, {n: x for n, x in zip(names_in, inputs)})
        for r, o in zip(ref, out):
            worst = max(worst, float(np.max(np.abs(r.numpy() - o))))
    return worst


def build_config(runner, checkpoint: Path) -> dict:
    """Every constant MujocoV8Runner.step / _update_hl_obs uses, by name."""
    from homie_pushcart.paths import FINGER_JOINTS, HOMIE_JOINTS
    from homie_pushcart_v8.task_config import (ACTION_DIM, FRAME_DIM, HL_ACTIONS, LONG_HISTORY, POLICY_HISTORY,
                                               leg_residual_filter_alpha)
    from homie_pushcart_v7.assets.build import load_v7_pose
    import xml.etree.ElementTree as ET
    from homie_pushcart.paths import G1_URDF

    e, tc, v8 = runner.env, runner.task_cfg, runner.v8
    n_low = e.num_lower
    lr = v8.leg_residual
    cam = v8.camera
    urdf = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
            for j in ET.parse(G1_URDF).getroot().findall("joint") if j.get("type") != "fixed"}
    finger = load_v7_pose()["robot"]["finger_pos"]
    lp = runner.lp
    runner.reset(seed=0)
    f = lambda a: [float(x) for x in np.asarray(a).reshape(-1)]           # noqa: E731
    return {
        "schema": "pushcart_v81_deploy_v1",
        "release": "V8.1",
        "dt": float(runner.dt),
        "joints": list(HOMIE_JOINTS),
        "num_lower": int(n_low),
        # --- HOMIE (low level) ---
        "default_dof_pos": f(e.default_dof_pos),
        "kp": f(e.p_gains),                 # legs: HOMIE training gains; waist + arms: V8.1 explicit upper drive
        "kd": f(e.d_gains),
        "torque_limits": f(e.torque_limits),
        "joint_lower": [urdf[n][0] for n in HOMIE_JOINTS],
        "joint_upper": [urdf[n][1] for n in HOMIE_JOINTS],
        "action_scale": float(e.action_scale),
        "clip_actions": float(e.clip_actions),
        "clip_obs": float(e.clip_obs),
        "obs_scales": {k: float(v) for k, v in e.obs_scales.items()},
        "commands_scale": f(e.commands_scale),
        "height_command": float(e.height_target),
        "homie_frame": int(e.one_step_obs),
        "homie_history": int(e.history_len),
        # --- V8.1 high level ---
        "frame_dim": FRAME_DIM, "policy_history": POLICY_HISTORY, "long_history": LONG_HISTORY,
        "hl_actions": HL_ACTIONS, "action_dim": ACTION_DIM,
        "camera_index_in_frame": 94,
        "hl_clip": float(tc.actions.clip),
        "ref_alpha": float(runner.ref_alpha),
        "cmd_alpha": float(runner.cmd_alpha),
        "gate_speed": float(runner.gate_speed),
        "cmd_residual_scale": f(runner.residual_scale),
        "homie_cmd_lo": f(runner.cmd_lo), "homie_cmd_hi": f(runner.cmd_hi),
        "upper_scale": f(runner.upper_scale),
        "upper_step_limit": float(runner.step_lim),
        "hold_target_q": f(runner.hold_target_q),
        "upper_lo": f(runner.upper_lo), "upper_hi": f(runner.upper_hi),
        "torque_obs_scale": float(tc.observations.torque_scale),
        "scale_obs": float(lp.arm_scale / tc.actions.arm_scale_range[1]),
        "leg_residual": {"enabled": bool(lr.enabled), "mode": str(lr.mode), "range": float(lr.range),
                         "clip": float(lr.clip), "alpha": float(leg_residual_filter_alpha(tc, runner.dt))},
        "task_command_range": {"vx": f(lp.vx_range), "vy_max": float(lp.vy_max), "wz_max": float(lp.wz_max),
                               "curriculum_level": int(lp.level), "payload_max_kg": float(lp.payload_max_kg)},
        # --- camera (training model; the robot must stay inside it) ---
        "camera_training_model": {"latency_s": [s * runner.dt for s in cam.latency_steps],
                                  "update_hz": [1.0 / (p * runner.dt) for p in cam.update_period_steps[::-1]],
                                  "noise_yaw_deg": math.degrees(cam.noise_yaw_rad), "dropout_prob": cam.dropout_prob},
        "rel_yaw_at_nominal_grasp_rad": float(runner.ts["rel_yaw"]),
        # --- hands (Dex-3, 7 per hand: thumb 0-2, middle 0-1, index 0-1) ---
        "finger_joints": list(FINGER_JOINTS),
        "finger_grasp_fit": [float(finger[n]) for n in FINGER_JOINTS],
        # --- provenance ---
        "provenance": {
            "student_checkpoint": str(checkpoint.relative_to(ROOT)), "student_sha256": sha256(checkpoint),
            "homie_checkpoint": str(runner.homie_checkpoint.relative_to(ROOT)),
            "homie_sha256": sha256(runner.homie_checkpoint),
            "homie_velocity_estimator": str(Path(runner.hv_path).relative_to(ROOT)),
            "homie_velocity_estimator_sha256": sha256(Path(runner.hv_path)),
            "student_iteration": getattr(runner.policy, "iteration", None),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="final", help="V8 student checkpoint; 'final' = released V8.1")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    from pushcart_bench.policies_v8 import resolve_v8_checkpoint
    from homie_pushcart_v8.mujoco_runner import MujocoV8Runner
    from homie_pushcart_v8.task_config import FRAME_DIM, LONG_HISTORY, POLICY_HISTORY

    checkpoint = resolve_v8_checkpoint(args.checkpoint)
    runner = MujocoV8Runner(checkpoint, level=17)
    if not runner.student or runner.hv is None:
        raise SystemExit(f"{checkpoint} is not a deployable V8 student with a fine-tuned HOMIE estimator")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    student = StudentGraph(runner.policy).eval()
    p_dim, h_dim = FRAME_DIM * POLICY_HISTORY, FRAME_DIM * LONG_HISTORY
    s_path = out / "v81_student.onnx"
    export_onnx(student, (torch.zeros(1, p_dim), torch.zeros(1, h_dim)), ["policy", "history"],
                ["actions", "velocity_estimate"], s_path)
    s_err = check_onnx(student, s_path, ["policy", "history"],
                       lambda r: [r.normal(size=(1, p_dim)).astype(np.float32),
                                  r.normal(size=(1, h_dim)).astype(np.float32)])

    homie = HomieGraph(runner.lowlevel.model, runner.hv).eval()
    h_path = out / "homie_v81.onnx"
    export_onnx(homie, (torch.zeros(1, 456), torch.ones(1, 1)), ["obs_history", "use_finetuned"],
                ["leg_actions", "vel_used", "vel_homie"], h_path)
    h_err = check_onnx(homie, h_path, ["obs_history", "use_finetuned"],
                       lambda r: [r.normal(size=(1, 456)).astype(np.float32),
                                  np.array([[float(r.integers(0, 2))]], dtype=np.float32)])

    # the HOMIE graph must reproduce the runner's own HOMIE call
    with torch.inference_mode():
        x = torch.randn(4, 456)
        m = runner.lowlevel.model
        vel_h, latent = m.estimator(x)
        ref = m.actor(torch.cat((x[:, -76:], runner.hv(x)[:, :3], latent), dim=-1))
        g_err = float((homie(x, torch.ones(4, 1))[0] - ref).abs().max())

    cfg = build_config(runner, checkpoint)
    cfg["export_check"] = {"student_onnx_max_abs_err": s_err, "homie_onnx_max_abs_err": h_err,
                           "homie_graph_vs_runner_max_abs_err": g_err}
    (out / "v81_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[export] {s_path.name}: max |onnx - torch| = {s_err:.2e}")
    print(f"[export] {h_path.name}: max |onnx - torch| = {h_err:.2e}; graph vs runner {g_err:.2e}")
    print(f"[export] rel_yaw at the nominal grasp = {cfg['rel_yaw_at_nominal_grasp_rad']:.4f} rad")
    print(f"[export] wrote {out}")
    if max(s_err, h_err, g_err) > 1e-4:
        raise SystemExit("ONNX export mismatch above 1e-4")


if __name__ == "__main__":
    main()
