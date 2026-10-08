"""Read-only dynamic hand supervision on the unchanged V3.2 simulation.

Does not create ground contact views, change physical properties, override the
legacy reward wrench, or integrate extra steps. These remain model-based labels.
"""
import torch
from .observation import world_to_cart_wrenches


# Pure measurement algebra from the audited subtree load calculation. Keeping
# it local avoids importing any V5 simulation, drive, or ground implementation.
def rotation_matrix_xyzw(quaternion):
    q = quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def rotate_inertia(inertia, quaternion):
    rotation = rotation_matrix_xyzw(quaternion)
    return rotation @ inertia @ rotation.transpose(-1,-2)


def compensate_subtree_wrench(incoming, origin, mass, position, angular_velocity,
                              acceleration, angular_acceleration, inertia, gravity):
    effective = mass[...,None]*(acceleration-gravity)
    spin = (inertia @ angular_velocity.unsqueeze(-1)).squeeze(-1)
    spin_rate = ((inertia @ angular_acceleration.unsqueeze(-1)).squeeze(-1)
                 + torch.cross(angular_velocity,spin,dim=-1))
    moment = (spin_rate+torch.cross(position-origin.unsqueeze(-2),effective,dim=-1)).sum(-2)
    return incoming-torch.cat((effective.sum(-2),moment),-1)


class ReadOnlyHandLabels:
    def __init__(self, env):
        self.env = env
        names = list(env.robot.body_names)
        self.subtrees = [torch.tensor([i for i, name in enumerate(names) if name.startswith(side+"_hand")],
                                      device=env.device, dtype=torch.long) for side in ("left", "right")]
        if any(len(ids) == 0 for ids in self.subtrees):
            raise ValueError("V3.2 hand subtrees were not found")
        self.gravity = torch.tensor(env.cfg.sim.gravity, device=env.device)
        if env.cfg.asset.disable_gravity:
            self.gravity.zero_()
        self.refresh_mass_properties()

    def refresh_mass_properties(self):
        """Cache the hand-subtree masses and local inertias on the GPU. Isaac Lab fetches body_mass / body_inertia
        from PhysX's CPU model on every access (10 + 18 ms at 4096 envs); the robot's are set once, by
        _randomize_body_properties before this object exists, so the cache equals a fresh read. Call again after
        any later robot set_masses / set_inertias."""
        data, N = self.env.robot.data, self.env.num_envs
        self.mass = [data.body_mass.torch[:, ids].clone() for ids in self.subtrees]
        # COM offsets in the link frames: body_com_pose_w re-fetches them from the CPU model on its first access
        # after every physics step (10 ms at 4096 envs); the world COM is link pose (+) this constant offset.
        self.com_b = [data.body_com_pose_b.torch[:, ids, :3].clone() for ids in self.subtrees]
        self.inertia_link = [data.body_inertia.torch[:, ids].reshape(N, -1, 3, 3).transpose(-1, -2).contiguous()
                             for ids in self.subtrees]

    @torch.no_grad()
    def sample(self):
        import warp as wp
        env, robot = self.env, self.env.robot
        data = robot.data
        palms = env.palm_body_ids
        pose = data.body_link_pose_w.torch[:, palms]
        rotation = rotation_matrix_xyzw(pose[..., 3:7])
        incoming = wp.to_torch(robot.root_view.get_link_incoming_joint_force()).reshape(
            env.num_envs, robot.num_bodies, 6)[:, palms]
        incoming = torch.cat(((rotation @ incoming[..., :3, None]).squeeze(-1),
                              (rotation @ incoming[..., 3:, None]).squeeze(-1)), -1)
        loads = []
        for side, ids in enumerate(self.subtrees):
            mass = self.mass[side]
            link = data.body_link_pose_w.torch[:, ids]
            position = link[..., :3] + (rotation_matrix_xyzw(link[..., 3:7]) @ self.com_b[side].unsqueeze(-1)).squeeze(-1)
            velocity = data.body_com_vel_w.torch[:, ids]
            acceleration = data.body_com_acc_w.torch[:, ids]
            inertia = rotate_inertia(self.inertia_link[side], data.body_link_pose_w.torch[:, ids, 3:7])
            loads.append(compensate_subtree_wrench(incoming[:, side], pose[:, side, :3], mass,
                position, velocity[..., 3:], acceleration[..., :3], acceleration[..., 3:], inertia, self.gravity))
        world = torch.stack(loads, 1)
        valid = (env.episode_length_buf > 0) & torch.isfinite(world).all(dim=(1, 2))
        world = torch.where(valid[:, None, None], world, 0.)
        return world_to_cart_wrenches(world, env.ts.cart_yaw), valid

    @staticmethod
    def metadata():
        return {"source": "incoming joint wrench minus hand-subtree Newton-Euler load",
                "reference": "individual palm origins", "axes": "cart heading, world vertical",
                "direction": "hand on cart", "sampling": "last physical substep at 50 Hz control sample",
                "reward_source_changed": False, "ground_contact_report_required": False,
                "dynamic_contact_wrench_calibrated": False}
