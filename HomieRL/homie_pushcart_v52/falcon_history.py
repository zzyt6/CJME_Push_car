"""FALCON appendix A.3 input contract, adapted to HOMIE's 27 joints.

At observation t, each chronological row contains [q_t, dq_t, omega_t,
gravity_t, a_lower_(t-1)]. The five rows therefore contain measurements
t-4:t and actual clipped HOMIE actions t-5:t-1. No upper torque, inverse
Jacobian, command, FK, HIM estimate, or privileged force enters this buffer.
Measurements use the existing fixed encoder/IMU scaling. Lower actions are
dimensionless policy outputs before action_scale, after clipping.
"""
import torch

from .observation import frame_layout

HISTORY = 5
LOWER_DIM = 12
FRAME_DIM = 72
INPUT_DIM = HISTORY * FRAME_DIM
FIELDS = ('joint_position', 'joint_velocity', 'imu_gyro', 'projected_gravity')


class FalconHistory:
    def __init__(self, num_envs, device='cpu'):
        self.values = torch.zeros(num_envs, HISTORY, FRAME_DIM, device=device)
        self.indices = torch.tensor([i for f in frame_layout() if f['name'] in FIELDS
                                     for i in range(f['start'], f['start']+f['dim'])],
                                    device=device, dtype=torch.long)

    def push(self, frame, previous_lower_action, fresh):
        if frame.shape != (len(self.values), 136) or previous_lower_action.shape != (len(self.values), LOWER_DIM):
            raise ValueError('Expected a V5.2 measurement frame and twelve actual lower actions')
        self.values[:, :-1] = self.values[:, 1:].clone()
        self.values[fresh] = 0.
        action = torch.where(fresh[:, None], 0., previous_lower_action)
        self.values[:, -1] = torch.cat((frame.index_select(-1, self.indices), action), -1)
        # Independent snapshot: later steps must not mutate collected samples.
        return self.values.flatten(1).clone()


def fullbody_history(policy_history, falcon_history):
    """Add the actually commanded upper target history to all-body measurements.

    Each 87D row is [q27,dq27,gyro3,gravity3,lower_action12,upper_target15].
    Upper targets are the logged (q_des-q_default)*dof_pos_scale, not torques
    or an inferred action. At reset the current upper target is the known
    initialized hold command; earlier frames are zero in both engines.
    """
    if falcon_history.shape[-1] != INPUT_DIM or policy_history.shape[-1] < 680:
        raise ValueError('Full-body history needs five measured frames and actual lower actions')
    frame=next(f for f in frame_layout() if f['name']=='upper_target_applied')
    recent=policy_history[..., -680:].reshape(*policy_history.shape[:-1],5,136)
    upper=recent[..., frame['start']:frame['start']+frame['dim']]
    original=falcon_history.reshape(*falcon_history.shape[:-1],5,72)
    return torch.cat((original,upper),-1).flatten(-2)
