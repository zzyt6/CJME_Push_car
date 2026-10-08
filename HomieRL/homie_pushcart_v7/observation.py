"""Exact V3.2 field projection and independent HIM sensor history; torch only."""
import torch

LEGACY_FRAME = 124
FRAME_DIM = 121
ACTOR_HISTORY = 5
HIM_FRAME = 77
HIM_HISTORY = 6
CART_VELOCITY_SCALE = (2., 2., .5)
HAND_WRENCH_SCALE = (50., 50., 50., 5., 5., 5.) * 2
KEEP_INDICES = tuple(range(101)) + tuple(range(104, 124))


def without_handle_position(legacy_history):
    """Remove only each frame's original indices101:104, retaining frame order."""
    if legacy_history.shape[-1] != ACTOR_HISTORY * LEGACY_FRAME:
        raise ValueError("V7 requires the unchanged V3.2 five-frame source history")
    frames = legacy_history.reshape(*legacy_history.shape[:-1], ACTOR_HISTORY, LEGACY_FRAME)
    return frames[..., list(KEEP_INDICES)].flatten(-2)


# V3.2 legacy frame (124): HOMIE 76 | task cmd 76:79 | mass 79 | upper torque 80:95 | HOMIE velocity estimate 95:98 |
# command - estimate (x, y from the estimate, z from the gyro) 98:101 | cart pose 101:105 | actions 105:123 | scale 123
HOMIE_ESTIMATE_SLOTS = (95, 96, 97, 98, 99)


def mask_homie_velocity(legacy_history):
    """V7.1: zero HOMIE's velocity estimate and the command-minus-estimate (x, y) in every frame of a [..., 5 x 124]
    (actor) or [..., 5 x 124 + k] (critic, legacy prefix) history; the push-cart-tuned estimator replaces them."""
    out = legacy_history.clone()
    frames = out[..., :ACTOR_HISTORY * LEGACY_FRAME].reshape(*out.shape[:-1], ACTOR_HISTORY, LEGACY_FRAME)
    frames[..., list(HOMIE_ESTIMATE_SLOTS)] = 0.
    out[..., :ACTOR_HISTORY * LEGACY_FRAME] = frames.flatten(-2)
    return out


def him_frame(legacy_frame):
    if legacy_frame.shape[-1] != LEGACY_FRAME:
        raise ValueError("Expected one V3.2 124-dimensional frame")
    return torch.cat((legacy_frame[..., :76], legacy_frame[..., 104:105]), -1)


class HIMHistory:
    def __init__(self, num_envs, device="cpu", dtype=torch.float32):
        self.buffer = torch.zeros(num_envs, HIM_HISTORY, HIM_FRAME, device=device, dtype=dtype)

    def reset(self, ids):
        self.buffer[ids] = 0.

    def push(self, frame, fresh):
        if frame.shape != (len(self.buffer), HIM_FRAME):
            raise ValueError("Expected 77-dimensional HOMIE plus relative-yaw measurements")
        history = torch.cat((self.buffer[:, 1:], frame[:, None]), 1)
        history[fresh, :-1] = 0.
        self.buffer = history
        return history.flatten(1)


def world_to_cart_wrenches(wrench, yaw):
    """Retain palm moment origins; rotate both force and moment into cart-heading axes."""
    v = wrench.reshape(-1, 2, 2, 3)
    c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
    return torch.stack((c*v[..., 0]+s*v[..., 1], -s*v[..., 0]+c*v[..., 1], v[..., 2]), -1).reshape(-1, 12)
