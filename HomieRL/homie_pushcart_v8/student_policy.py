"""V8 student side: the adaptation module (RMA phase 2 estimator) and the loader of the deployable student.

The deployable V8 student is the ROA teacher's base policy with its latent z taken from the adaptation module over the
deployable 32-frame history (rma_policy.RMAStudentExport); no privileged input at inference.
"""
from __future__ import annotations

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization

from .task_config import FRAME_DIM, LONG_HISTORY, POLICY_HISTORY

POLICY_DIM = FRAME_DIM * POLICY_HISTORY
HISTORY_DIM = FRAME_DIM * LONG_HISTORY


class AdaptationModule(nn.Module):
    """[N, 32 x 126] -> (explicit, latent). Temporal 1D convolution over the history (RMA's adaptation module design);
    the V8 ROA student uses explicit_dim = 0 (latent only)."""

    def __init__(self, latent_dim: int = 20, channels: int = 64, explicit_dim: int = 0):
        super().__init__()
        self.explicit_dim = int(explicit_dim)
        self.normalizer = EmpiricalNormalization(FRAME_DIM)          # per-frame statistics, shared over time
        self.conv = nn.Sequential(
            nn.Conv1d(FRAME_DIM, channels, kernel_size=4, stride=2), nn.ELU(),      # 32 -> 15
            nn.Conv1d(channels, channels, kernel_size=3, stride=2), nn.ELU(),       # 15 -> 7
            nn.Conv1d(channels, channels, kernel_size=3, stride=1), nn.ELU(),       # 7 -> 5
        )
        self.head = nn.Sequential(nn.Linear(channels * 5, 256), nn.ELU(), nn.Linear(256, self.explicit_dim + latent_dim))

    def frames(self, history):
        return history.reshape(history.shape[0], LONG_HISTORY, FRAME_DIM)

    def forward(self, history):
        f = self.normalizer(self.frames(history).reshape(-1, FRAME_DIM)).reshape(-1, LONG_HISTORY, FRAME_DIM)
        out = self.head(self.conv(f.transpose(1, 2)).flatten(1))
        return out[:, :self.explicit_dim], out[:, self.explicit_dim:]

    @torch.no_grad()
    def update_normalization(self, history):
        self.normalizer.update(self.frames(history).reshape(-1, FRAME_DIM))


def rma_deploy_requested() -> bool:
    """Evaluate a ROA teacher run as its deployable student (z from the adaptation module): V8_RMA_DEPLOY=1."""
    import os
    return os.environ.get("V8_RMA_DEPLOY", "") == "1"


def is_student_run(checkpoint) -> bool:
    """True for a deployable-student run (RMA phase 2 = adapt_only, or student RL), or any ROA run with
    V8_RMA_DEPLOY=1."""
    from pathlib import Path
    import yaml
    path = Path(checkpoint)
    run = path.parent if path.suffix == ".pt" else path
    agent = run / "params" / "agent.yaml"
    if not agent.is_file():
        return False
    cfg = yaml.load(agent.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    if "RMATeacherActor" not in str((cfg.get("actor") or {}).get("class_name", "")):
        return False
    alg = cfg.get("algorithm") or {}
    return rma_deploy_requested() or bool(alg.get("adapt_only", False)) or bool(alg.get("student_rl", False))


def load_student(checkpoint, device="cpu"):
    """Deployable V8 student (policy 630, history 4032 -> 30 actions) from a ROA / phase-2 / student-RL checkpoint."""
    from .rma_policy import load_rma
    return load_rma(checkpoint, deploy=True, device=device)
