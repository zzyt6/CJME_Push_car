"""V8 teacher Actor base: privileged encoder -> latent z, policy MLP with the gated leg residual. The first V8 teacher
(retired) also fed the 19 explicit truths to the body; the final V8 teacher (rma_policy.RMATeacherActor) subclasses this
class and sees the privileged information only through z.

Leg residual (last 12 outputs u): the MLP rows are zero-initialised, so training starts from HOMIE exactly. The
environment maps u to the applied residual (tanh mode: 0.3 * tanh(u), low-passed). The output gate is fixed at 1 in
tanh mode; the first V8 teacher (clip mode) learned it from 0.1 (SteadyTray's residual-adapter start) and it grew to
1.38 while u drifted beyond the clip.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP

from .task_config import ACTION_DIM, EXPLICIT_DIM, FRAME_DIM, HL_ACTIONS, LEG_ACTIONS, POLICY_HISTORY, PRIVILEGED_DIM

POLICY_DIM = FRAME_DIM * POLICY_HISTORY
TEACHER_SCHEMA = "v8_teacher_policy630_priv62_explicit19_latent24_act30_v2"


class LegGate(nn.Module):
    """Multiplies the leg-residual outputs by a learnable gate."""

    def __init__(self, init: float = 0.1, learnable: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)), requires_grad=bool(learnable))

    def forward(self, x):
        return torch.cat((x[..., :HL_ACTIONS], self.alpha * x[..., HL_ACTIONS:]), -1)


class TeacherActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=True, distribution_cfg=None, latent_dim=24, encoder_hidden_dims=(128, 64),
                 leg_gate_init=0.1, leg_init_std=0.05, leg_gate_learnable=True):
        if output_dim != ACTION_DIM:
            raise ValueError(f"V8 Actor outputs {ACTION_DIM} actions (18 high-level + 12 leg residual)")
        self.latent_dim = int(latent_dim)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         copy.deepcopy(distribution_cfg))
        self.privileged_normalizer = EmpiricalNormalization(PRIVILEGED_DIM)
        self.encoder = MLP(PRIVILEGED_DIM, self.latent_dim, list(encoder_hidden_dims), activation)
        last = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():                    # leg residual starts at exactly zero
            last.weight[HL_ACTIONS:ACTION_DIM].zero_()
            last.bias[HL_ACTIONS:ACTION_DIM].zero_()
        self.mlp = nn.Sequential(self.mlp, LegGate(leg_gate_init, leg_gate_learnable))
        dist = self.distribution
        if dist is not None:
            with torch.no_grad():
                if hasattr(dist, "std_param"):
                    dist.std_param[HL_ACTIONS:] = leg_init_std
                elif hasattr(dist, "log_std_param"):
                    dist.log_std_param[HL_ACTIONS:] = torch.log(torch.tensor(leg_init_std))
        self.leg_init_std, self.leg_gate_init = float(leg_init_std), float(leg_gate_init)
        self.leg_gate_learnable = bool(leg_gate_learnable)

    # observation layout ---------------------------------------------------------------------------------------
    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_groups[obs_set] != ["policy", "privileged"]:
            raise ValueError("V8 teacher Actor groups must be ['policy', 'privileged'].")
        if obs["policy"].shape[-1] != POLICY_DIM or obs["privileged"].shape[-1] != PRIVILEGED_DIM:
            raise ValueError(f"V8 teacher needs policy {POLICY_DIM} and privileged {PRIVILEGED_DIM}.")
        return list(obs_groups[obs_set]), POLICY_DIM + EXPLICIT_DIM      # normalized part (latent z appended raw)

    def _get_latent_dim(self):
        return POLICY_DIM + EXPLICIT_DIM + self.latent_dim

    def context(self, obs):
        """Teacher context: (explicit physical quantities, encoded latent)."""
        p = obs["privileged"]
        return p[:, :EXPLICIT_DIM], self.encoder(self.privileged_normalizer(p))

    def get_latent(self, obs, masks=None, hidden_state=None):
        explicit, z = self.context(obs)
        x = self.obs_normalizer(torch.cat((obs["policy"], explicit), -1))
        return torch.cat((x, z), -1)

    @torch.no_grad()
    def update_normalization(self, obs):
        if self.obs_normalization:
            self.obs_normalizer.update(torch.cat((obs["policy"], obs["privileged"][:, :EXPLICIT_DIM]), -1))
        self.privileged_normalizer.update(obs["privileged"])

    def metadata(self):
        return {"schema": TEACHER_SCHEMA, "release": "8.0", "policy": POLICY_DIM, "privileged": PRIVILEGED_DIM,
                "explicit": EXPLICIT_DIM, "latent": self.latent_dim, "actions": ACTION_DIM,
                "leg_actions": LEG_ACTIONS, "leg_gate_init": self.leg_gate_init, "leg_init_std": self.leg_init_std,
                "leg_gate_learnable": self.leg_gate_learnable}

    def as_jit(self):
        return _TeacherExport(self)

    def as_onnx(self, verbose=False):
        return _TeacherExport(self)


class _TeacherExport(nn.Module):
    """Deterministic teacher (simulation evaluation only: it needs privileged inputs)."""

    def __init__(self, actor: TeacherActor):
        super().__init__()
        for name in ("obs_normalizer", "privileged_normalizer", "encoder", "mlp"):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic = (actor.distribution.as_deterministic_output_module()
                              if actor.distribution is not None else nn.Identity())

    def forward(self, policy: torch.Tensor, privileged: torch.Tensor):
        z = self.encoder(self.privileged_normalizer(privileged))
        x = self.obs_normalizer(torch.cat((policy, privileged[:, :EXPLICIT_DIM]), -1))
        return self.deterministic(self.mlp(torch.cat((x, z), -1)))


def load_teacher(checkpoint, device="cpu"):
    """Deterministic V8 ROA teacher (policy 630, privileged 62 -> 30 actions; simulation only)."""
    from .rma_policy import load_rma
    return load_rma(checkpoint, deploy=False, device=device)
