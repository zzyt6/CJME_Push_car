"""Export the HOMIE v2 balance model for deployment: policy.onnx, config.json, verification_vectors.npz.

policy.onnx: the HIM estimator encoder + actor (act_inference, 456 -> 12), input 'input', output 'output',
dynamic batch, opset 11 (OpenHomie deploy.onnx interface). config.json: observation layout and scales, history
order, joint names/order and default angles, action-to-target mapping, Kp/Kd, policy rate -- all read from the
training configuration. verification_vectors.npz: observations with the original PyTorch model's outputs.

    E:\\envs\\RLisaac601\\python.exe scripts/export_homie_v2_balance.py [--checkpoint model_N.pt] [--out dir]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "HomieRL"), str(ROOT / "HomieRL" / "rsl_rl")]
CHECKPOINT = ROOT / "logs/rsl_rl/homie_g1_v2_balance/2026-10-14_01-17-28_balance_pelvis_4096_10k/model_68600.pt"
BANK = ROOT / ".cache/homie_v2_balance/switch_bank.pt"
URDF = ROOT / "HomieRL/legged_gym/resources/robots/g1_description/g1.urdf"


class DeployPolicy(torch.nn.Module):
    """HIMActorCritic.act_inference with plain tensors: estimator encoder -> (velocity, unit latent) -> actor."""

    def __init__(self, model):
        super().__init__()
        self.encoder = copy.deepcopy(model.estimator.encoder)
        self.actor = copy.deepcopy(model.actor)
        self.frame = model.num_one_step_obs

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.encoder(obs_history)
        vel, z = parts[..., :3], F.normalize(parts[..., 3:], dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, -self.frame:], vel, z), dim=-1))


def config(checkpoint: Path, onnx_path: Path) -> dict:
    from homie_isaaclab_v2.config import G1RoughCfg
    cfg = G1RoughCfg()
    joints = [j.get("name") for j in ET.parse(URDF).getroot().findall("joint") if j.get("type") == "revolute"]
    gain = lambda table, n: next(v for k, v in table.items() if k in n)                   # noqa: E731
    s = cfg.normalization.obs_scales
    frame = [
        ("command_vx_vy_wz", 3, "0, 0, 0 for the balance model (trained with an exact zero command); scale 2.0, 2.0, 0.5"),
        ("height_command", 1, "0.74, unscaled"),
        ("base_ang_vel", 3, f"pelvis IMU angular velocity in the IMU frame (IMU axes = pelvis axes), x {s.ang_vel}"),
        ("projected_gravity", 3, "gravity unit vector (0, 0, -1) in the IMU frame, unscaled"),
        ("dof_pos", 27, f"(q - default_joint_pos) x {s.dof_pos}, joint_names order"),
        ("dof_vel", 27, f"dq x {s.dof_vel}, joint_names order"),
        ("last_action", 12, "the policy's previous raw output, unscaled"),
    ]
    layout, start = [], 0
    for name, dim, note in frame:
        layout.append({"name": name, "start": start, "dim": dim, "note": note})
        start += dim
    assert start == cfg.env.num_one_step_observations
    return {
        "model": {"file": onnx_path.name, "input": "input", "output": "output",
                  "input_dim": cfg.env.num_observations, "output_dim": cfg.env.num_actions,
                  "source_checkpoint": checkpoint.name, "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest()},
        "policy_rate_hz": round(1 / (cfg.sim.dt * cfg.control.decimation)),
        "joint_names": joints,
        "default_joint_pos": [cfg.init_state.default_joint_angles[n] for n in joints],
        "observation": {
            "frame_dim": cfg.env.num_one_step_observations, "history_frames": cfg.env.num_actor_history,
            "history_order": "oldest frame first, newest frame in the last 76 entries; all zeros at start",
            "clip": cfg.normalization.clip_observations,
            "frame": layout,
        },
        "action": {
            "dim": cfg.env.num_actions, "joints": joints[:cfg.env.num_actions],
            "action_scale": cfg.control.action_scale, "clip": cfg.normalization.clip_actions,
            "target": "q_target = default_joint_pos + action_scale x clip(action, -clip, clip), first 12 joints",
        },
        "pd": {
            "kp": [gain(cfg.control.stiffness, n) for n in joints],
            "kd": [gain(cfg.control.damping, n) for n in joints],
            "effort_limit": [float(j.find("limit").get("effort")) for j in ET.parse(URDF).getroot().findall("joint")
                             if j.get("type") == "revolute"],
            "note": "joint_names order. Legs: tau = kp (q_target - q) - kd dq; waist and arms (not policy outputs) hold "
                    "position targets with the same PD.",
        },
    }


def main():
    import onnxruntime as ort
    from homie_pushcart.lowlevel import HimLowLevelPolicy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--out", type=Path, default=ROOT / "deploy/balance")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    loaded = HimLowLevelPolicy(a.checkpoint, "cpu")
    net = DeployPolicy(loaded.model).eval()
    onnx_path = a.out / "policy.onnx"
    torch.onnx.export(net, torch.zeros(1, loaded.num_obs), str(onnx_path), export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, dynamo=False)
    torch.manual_seed(0)
    bank = torch.load(BANK, map_location="cpu")["states"]["obs"]
    check = torch.cat((bank[torch.randperm(len(bank))[:4096]], torch.randn(256, bank.shape[1])))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        diff = float((loaded.model.act_inference(check) - torch.from_numpy(
            session.run(None, {"input": check.numpy()})[0])).abs().max())
    assert diff < 1e-4, diff
    (a.out / "config.json").write_text(json.dumps(config(a.checkpoint, onnx_path), indent=2), encoding="utf-8")
    # Verification vectors: recorded walk-to-stop observations (real histories) plus the all-zero start history.
    vectors = torch.cat((torch.zeros(1, bank.shape[1]), bank[torch.randperm(len(bank))[:63]]))
    with torch.inference_mode():
        expected = loaded.model.act_inference(vectors)
    np.savez(a.out / "verification_vectors.npz", inputs=vectors.numpy().astype(np.float32),
             outputs=expected.numpy().astype(np.float32))
    print(f"[export] {a.checkpoint.name} -> {a.out}: policy.onnx (max |diff| vs PyTorch {diff:.1e} on {len(check)} "
          f"observations), config.json, verification_vectors.npz ({len(vectors)} pairs)", flush=True)


if __name__ == "__main__":
    main()
