"""Tensor helpers with the legacy HOMIE xyzw quaternion convention."""
import torch


def quat_rotate_inverse(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    return v * (2 * w.square() - 1) - 2 * w * torch.cross(xyz, v, dim=-1) + 2 * xyz * (xyz * v).sum(-1, keepdim=True)


def quat_apply(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_mul(a, b):
    xyz = a[..., 3:4] * b[..., :3] + b[..., 3:4] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def euler_from_quaternion(q):
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = torch.asin(torch.clamp(2 * (w*y - z*x), -1., 1.))
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return roll[:, None], pitch[:, None], yaw[:, None]


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device) + lower


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, '__dict__'):
        return obj
    return {key: class_to_dict(getattr(obj, key)) for key in dir(obj)
            if not key.startswith('_') and not callable(getattr(obj, key))}
