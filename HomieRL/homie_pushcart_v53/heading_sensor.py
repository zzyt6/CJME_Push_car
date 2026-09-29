"""Simulation model of the vision yaw interface; truth is confined to this adapter."""
import torch


class SimulatedVisualHeading:
    def __init__(self, num_envs, dt, cfg, device='cpu'):
        self.cfg, self.dt = cfg, dt
        self.history = torch.zeros(num_envs, cfg.heading_delay_steps_max+1, device=device)
        self.delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.count = torch.zeros_like(self.delay)
        self.strength = torch.ones(num_envs, device=device)
        self.bias = torch.zeros(num_envs, device=device)

    def reset(self, ids, randomize=True, fraction=1.):
        self.history[ids] = 0.
        self.count[ids] = 0
        if not 0 <= float(fraction) <= 1:
            raise ValueError('vision randomization fraction must be in [0,1]')
        self.strength[ids] = float(fraction)
        maximum = int(round((self.history.shape[1]-1)*float(fraction)))
        self.delay[ids] = (torch.randint(maximum+1, self.delay[ids].shape, device=self.delay.device)
                           if randomize else 0)
        self.bias[ids] = ((2*torch.rand_like(self.bias[ids])-1)*getattr(self.cfg, 'heading_bias_rad', 0.)
                          * self.strength[ids] if randomize else 0.)

    def sample(self, truth_relative_yaw, randomize=True):
        measurement = truth_relative_yaw
        if randomize:
            measurement = measurement + self.bias + (2*torch.rand_like(measurement)-1)*self.cfg.heading_noise_rad*self.strength
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = measurement
        available = self.count >= self.delay
        angle = self.history.gather(1, self.delay[:, None])[:, 0]
        valid = available & torch.isfinite(angle)
        if randomize:
            valid &= torch.rand_like(angle) >= self.cfg.heading_dropout_probability*self.strength
        self.count += 1
        return angle, valid, self.delay.float()*self.dt
