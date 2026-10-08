"""V8 student Actor (RMA-style phase 2): the teacher's policy body with (explicit, z) estimated from the deployable
32-frame history by an adaptation module. No privileged input at inference.

Body modules (``obs_normalizer``, ``mlp``) have the teacher's names and shapes; ``load_body_from(teacher)`` copies them
and, by default, freezes them (only the adaptation module learns: RMA / SteadyTray phase 2).
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization

from .task_config import ACTION_DIM, EXPLICIT_DIM, FRAME_DIM, HL_ACTIONS, LONG_HISTORY, POLICY_HISTORY
from .teacher_policy import LegGate

POLICY_DIM = FRAME_DIM * POLICY_HISTORY
HISTORY_DIM = FRAME_DIM * LONG_HISTORY
STUDENT_SCHEMA = "v8_student_policy630_history32x126_explicit19_latent24_act30_v2"


class AdaptationModule(nn.Module):
    """[N, 32 x 126] -> (explicit 19, latent 24). Temporal 1D convolution (RMA uses a 1D CNN over 50 steps)."""

    def __init__(self, latent_dim: int = 24, channels: int = 64):
        super().__init__()
        self.normalizer = EmpiricalNormalization(FRAME_DIM)          # per-frame statistics, shared over time
        self.conv = nn.Sequential(
            nn.Conv1d(FRAME_DIM, channels, kernel_size=4, stride=2), nn.ELU(),      # 32 -> 15
            nn.Conv1d(channels, channels, kernel_size=3, stride=2), nn.ELU(),       # 15 -> 7
            nn.Conv1d(channels, channels, kernel_size=3, stride=1), nn.ELU(),       # 7 -> 5
        )
        self.head = nn.Sequential(nn.Linear(channels * 5, 256), nn.ELU(), nn.Linear(256, EXPLICIT_DIM + latent_dim))

    def frames(self, history):
        return history.reshape(history.shape[0], LONG_HISTORY, FRAME_DIM)

    def forward(self, history):
        f = self.normalizer(self.frames(history).reshape(-1, FRAME_DIM)).reshape(-1, LONG_HISTORY, FRAME_DIM)
        out = self.head(self.conv(f.transpose(1, 2)).flatten(1))
        return out[:, :EXPLICIT_DIM], out[:, EXPLICIT_DIM:]

    @torch.no_grad()
    def update_normalization(self, history):
        self.normalizer.update(self.frames(history).reshape(-1, FRAME_DIM))


class StudentActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=True, distribution_cfg=None, latent_dim=24, freeze_body=True,
                 adaptation_channels=64):
        if output_dim != ACTION_DIM:
            raise ValueError(f"V8 Actor outputs {ACTION_DIM} actions")
        self.latent_dim = int(latent_dim)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         copy.deepcopy(distribution_cfg))
        self.mlp = nn.Sequential(self.mlp, LegGate())              # same structure as the teacher body
        self.adaptation = AdaptationModule(self.latent_dim, adaptation_channels)
        self.freeze_body = bool(freeze_body)
        self.body_loaded = False

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_groups[obs_set] != ["policy", "history"]:
            raise ValueError("V8 student Actor groups must be ['policy', 'history'].")
        if obs["policy"].shape[-1] != POLICY_DIM or obs["history"].shape[-1] != HISTORY_DIM:
            raise ValueError(f"V8 student needs policy {POLICY_DIM} and history {HISTORY_DIM}.")
        return list(obs_groups[obs_set]), POLICY_DIM + EXPLICIT_DIM

    def _get_latent_dim(self):
        return POLICY_DIM + EXPLICIT_DIM + self.latent_dim

    def context(self, obs):
        return self.adaptation(obs["history"])

    def get_latent(self, obs, masks=None, hidden_state=None):
        explicit, z = self.context(obs)
        x = self.obs_normalizer(torch.cat((obs["policy"], explicit), -1))
        return torch.cat((x, z), -1)

    @torch.no_grad()
    def update_normalization(self, obs):
        """The body normalizer is the teacher's (frozen with the body); only the history normalizer adapts."""
        self.adaptation.update_normalization(obs["history"])
        if self.obs_normalization and not (self.freeze_body and self.body_loaded):
            explicit, _ = self.context(obs)
            self.obs_normalizer.update(torch.cat((obs["policy"], explicit), -1))

    @torch.no_grad()
    def load_body_from(self, teacher):
        """Copy the teacher's body (normalizer + MLP incl. leg gate); freeze it unless freeze_body is False."""
        self.obs_normalizer.load_state_dict(teacher.obs_normalizer.state_dict())
        self.mlp.load_state_dict(teacher.mlp.state_dict())
        if self.distribution is not None and teacher.distribution is not None:
            self.distribution.load_state_dict(teacher.distribution.state_dict())
        for module in (self.obs_normalizer, self.mlp):
            for p in module.parameters():
                p.requires_grad_(not self.freeze_body)
        self.body_loaded = True

    def metadata(self):
        return {"schema": STUDENT_SCHEMA, "release": "8.0", "policy": POLICY_DIM, "history": HISTORY_DIM,
                "explicit": EXPLICIT_DIM, "latent": self.latent_dim, "actions": ACTION_DIM, "hl_actions": HL_ACTIONS,
                "freeze_body": self.freeze_body}

    def as_jit(self):
        return StudentExport(self)

    def as_onnx(self, verbose=False):
        return StudentExport(self)


class StudentExport(nn.Module):
    """Deterministic student: (policy 630, history 4032) -> 30 actions; also exposes the explicit estimates."""

    def __init__(self, actor: StudentActor):
        super().__init__()
        for name in ("obs_normalizer", "adaptation", "mlp"):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic = (actor.distribution.as_deterministic_output_module()
                              if actor.distribution is not None else nn.Identity())

    def forward(self, policy: torch.Tensor, history: torch.Tensor):
        explicit, z = self.adaptation(history)
        x = self.obs_normalizer(torch.cat((policy, explicit), -1))
        return self.deterministic(self.mlp(torch.cat((x, z), -1)))

    @torch.jit.export
    def explicit(self, history: torch.Tensor):
        return self.adaptation(history)[0]
