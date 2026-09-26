"""Simulator-free measurement packing for the proprioceptive transfer head."""
import numpy as np


class ProprioHistory:
    """Five chronological 96D frames at 50 Hz; uses no torque, FK or force truth."""
    def __init__(self, control, history=5):
        self.control = control
        self.frames = np.zeros((history, 96), dtype=np.float32)

    def reset(self):
        self.frames[:] = 0.

    def push(self, q, dq, gyro_pelvis, projected_gravity, homie_command, upper_target, previous_high_level_action):
        arrays = [np.asarray(x, dtype=float) for x in (q,dq,gyro_pelvis,projected_gravity,homie_command,
                                                      upper_target,previous_high_level_action)]
        for x,n in zip(arrays,(27,27,3,3,3,15,18)):
            if x.shape!=(n,) or not np.isfinite(x).all():
                raise ValueError('Invalid deployment proprioception vector')
        q,dq,gyro,gravity,command,upper,action=arrays
        c=self.control
        sc=c['observation_scales']
        default=np.asarray(c['default_joint_angles'])
        frame=np.concatenate(((q-default)*sc['dof_pos'],dq*sc['dof_vel'],gyro*sc['ang_vel'],gravity,
                              command*np.array([sc['lin_vel'],sc['lin_vel'],sc['ang_vel']]),
                              (upper-default[12:])*sc['dof_pos'],action)).astype(np.float32)
        self.frames[:-1]=self.frames[1:].copy()
        self.frames[-1]=frame
        return self.frames.reshape(-1).copy()
