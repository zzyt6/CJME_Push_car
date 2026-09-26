"""Standalone force estimators for controlled, simulator-independent training.

The proprioceptive variant borrows FALCON's short history / deployment-observable
input principle, not its complete dual-agent method. Our output remains twelve
per-hand force/moment components, not FALCON appendix A.3's 3D force baseline.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from .observation import FRAME_DIM, HAND_WRENCH_SCALE, WRENCH_SCALE, frame_layout

FEATURE_SETS = {
    'full': None,
    'falcon': (),  # Separate 5 x 72 history, including actual lower actions.
    'falcon_fullbody': (),  # Add the 15 actually commanded upper targets.
    'proprio': ('joint_position', 'joint_velocity', 'imu_gyro', 'projected_gravity',
                'homie_command_applied', 'upper_target_applied', 'previous_high_level_action'),
}


def feature_indices(features, history=5):
    if features not in FEATURE_SETS:
        raise ValueError(f'Unknown feature set: {features}')
    if features in ('falcon','falcon_fullbody'):
        return torch.arange(history * (72 if features == 'falcon' else 87), dtype=torch.long)
    names = FEATURE_SETS[features]
    one = [i for f in frame_layout() if names is None or f['name'] in names
           for i in range(f['start'], f['start'] + f['dim'])]
    return torch.tensor([t * FRAME_DIM + i for t in range(history) for i in one], dtype=torch.long)


class TransferEstimator(nn.Module):
    """Raw last-five V5.2 frames -> normalized twelve-dimensional hand wrench."""
    def __init__(self, features='full', hidden=(512, 256, 128), history=5):
        super().__init__()
        self.features, self.hidden, self.history = features, list(hidden), int(history)
        self.input_width = {'falcon':72,'falcon_fullbody':87}.get(features,FRAME_DIM) * self.history
        indices = feature_indices(features, history)
        self.register_buffer('indices', indices)
        self.register_buffer('mean', torch.zeros(len(indices)))
        self.register_buffer('std', torch.ones(len(indices)))
        layers, n = [], len(indices)
        for width in hidden:
            layers.extend((nn.Linear(n, width), nn.ELU()))
            n = width
        layers.append(nn.Linear(n, 12))
        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalization(self, train_x):
        x = train_x[:, self.indices]
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0, unbiased=False).clamp_min(1.e-3))

    def forward(self, x):
        if self.features in ('falcon','falcon_fullbody') and x.shape[-1] != self.input_width:
            raise ValueError('FALCON head requires its measured 360D/435D history, not a V5.2 policy history')
        selected = x[..., -self.input_width:].index_select(-1, self.indices)
        return self.network((selected - self.mean) / self.std)


class PhysicalWrenchExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model).cpu().eval()
        self.register_buffer('scale', torch.tensor(HAND_WRENCH_SCALE).repeat(2))

    def forward(self, recent_frames):
        return self.model(recent_frames) * self.scale


class SelectedWrenchExport(nn.Module):
    """Only selected, already fixed-scaled chronological features -> SI wrench."""
    def __init__(self, model):
        super().__init__()
        self.network = copy.deepcopy(model.network).cpu().eval()
        self.register_buffer('mean', model.mean.detach().cpu().clone())
        self.register_buffer('std', model.std.detach().cpu().clone())
        self.register_buffer('scale', torch.tensor(HAND_WRENCH_SCALE).repeat(2))

    def forward(self, selected_history):
        return self.network((selected_history-self.mean)/self.std)*self.scale


def load_estimator(path, device='cpu'):
    bundle = torch.load(Path(path), map_location=device, weights_only=False)
    if bundle.get('schema') != 'v52_transfer_estimator_v1':
        raise ValueError('Unsupported standalone estimator schema')
    model = TransferEstimator(**bundle['architecture']).to(device)
    model.load_state_dict(bundle['state_dict'], strict=True)
    return model.eval(), bundle


def install_estimator(actor, path):
    """Inference attachment: the new head participates in actual policy actions."""
    model, bundle = load_estimator(path, next(actor.parameters()).device)
    if model.history != actor.estimator_history or actor.frame_dim != FRAME_DIM:
        raise ValueError('Estimator and actor history contracts differ')
    actor.estimator_observation_key = model.features if model.features.startswith('falcon') else 'policy'
    actor.estimator_normalizer = nn.Identity()
    actor.force_estimator = model
    return bundle


def resultants(hands, offsets):
    """Normalized hand wrench -> physical handle [Fx,Fy,Mz], without RSL imports."""
    y = hands * torch.tensor(HAND_WRENCH_SCALE, device=hands.device).repeat(2)
    l, r = offsets[0], offsets[1]
    return torch.stack((y[..., 0]+y[..., 6], y[..., 1]+y[..., 7],
                        y[..., 5]+y[..., 11]+l[0]*y[..., 1]-l[1]*y[..., 0]
                        +r[0]*y[..., 7]-r[1]*y[..., 6]), -1)
